//! TEMP (lab): a per-thread cache of small blocks in front of the system heap.
//!
//! The 5K drag profile (in-process drain sampler naming DLL frames, 2026-10-01)
//! put `RtlAllocateHeap` / `RtlFreeHeap` at ~9 % of the drain thread's samples:
//! `Vec` growth, `collect`, `BTreeMap` inserts, `Arc::make_mut`, `format!` — a
//! few dozen heap trips per draw, each a call into ntdll's heap with its lock.
//! This keeps freed blocks of up to 4 KiB in per-thread lists by size class and
//! hands them out again without leaving the thread.
//!
//! Every block of a class is allocated from the system heap with the class's
//! own size and 16-byte alignment, so a block freed on another thread than the
//! one that allocated it simply joins that thread's list. A list holds at most
//! [`MAX_PER_CLASS`] blocks; beyond that, frees go back to the system heap.
//! Larger or more aligned requests go straight to the system heap. Thread
//! storage is a Win32 TLS slot holding a cache allocated from the process heap,
//! so the allocator never allocates through itself; a thread's cache is left
//! behind when the thread exits (the device's threads live as long as QEMU).
//!
//! `REIMS_VGPU_LAB_TCACHE=off` passes everything to the system heap. Read once,
//! at the first allocation, before any block exists, so no block can be freed
//! by a path other than the one that allocated it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};

const CLASS_SIZES: [usize; 16] = [
    16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];
const MAX_SMALL: usize = 4096;
const MAX_PER_CLASS: u32 = 256;
const ALIGN: usize = 16;

/// Class index for a size of at most [`MAX_SMALL`] bytes (a size of 0 is class 0).
#[inline]
fn class_of(size: usize) -> usize {
    // A lookup by 16-byte granule: 256 entries cover 4 KiB.
    static TABLE: [u8; 257] = {
        let mut t = [0u8; 257];
        let mut g = 0;
        while g <= 256 {
            let size = g * 16;
            let mut c = 0;
            while CLASS_SIZES[c] < size {
                c += 1;
            }
            t[g] = c as u8;
            g += 1;
        }
        t
    };
    TABLE[size.div_ceil(16)] as usize
}

#[repr(C)]
struct Cache {
    heads: [*mut u8; CLASS_SIZES.len()],
    counts: [u32; CLASS_SIZES.len()],
}

type Handle = *mut core::ffi::c_void;

extern "system" {
    fn TlsAlloc() -> u32;
    fn TlsGetValue(index: u32) -> *mut core::ffi::c_void;
    fn TlsSetValue(index: u32, value: *mut core::ffi::c_void) -> i32;
    fn GetProcessHeap() -> Handle;
    fn HeapAlloc(heap: Handle, flags: u32, bytes: usize) -> *mut core::ffi::c_void;
    fn GetEnvironmentVariableW(name: *const u16, buf: *mut u16, size: u32) -> u32;
}

const TLS_OUT_OF_INDEXES: u32 = 0xFFFF_FFFF;
const HEAP_ZERO_MEMORY: u32 = 0x8;

/// 0 = not decided, 1 = on, 2 = off.
static MODE: AtomicU8 = AtomicU8::new(0);
/// TLS slot + 1 (0 = not allocated yet).
static SLOT: AtomicU32 = AtomicU32::new(0);

pub struct ThreadCache;

#[cold]
fn decide() -> bool {
    // "REIMS_VGPU_LAB_TCACHE" as UTF-16, NUL-terminated; no allocation here.
    const NAME: &[u8] = b"REIMS_VGPU_LAB_TCACHE\0";
    let mut name = [0u16; 32];
    for (i, b) in NAME.iter().enumerate() {
        name[i] = u16::from(*b);
    }
    let mut buf = [0u16; 8];
    // SAFETY: both buffers are ours and sized as passed.
    let n = unsafe { GetEnvironmentVariableW(name.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) };
    let off = n == 3 && buf[0] == u16::from(b'o') && buf[1] == u16::from(b'f') && buf[2] == u16::from(b'f');
    let mut on = !off;
    if on {
        // SAFETY: plain Win32 call.
        let idx = unsafe { TlsAlloc() };
        if idx == TLS_OUT_OF_INDEXES {
            on = false;
        } else {
            SLOT.store(idx + 1, Ordering::Release);
        }
    }
    // A racing first allocation may have decided already; the first answer stands.
    match MODE.compare_exchange(0, if on { 1 } else { 2 }, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => on,
        Err(m) => m == 1,
    }
}

#[inline]
fn enabled() -> bool {
    match MODE.load(Ordering::Acquire) {
        1 => true,
        2 => false,
        _ => decide(),
    }
}

/// This thread's cache, created on first use; null if none could be made.
#[inline]
fn cache() -> *mut Cache {
    let idx = SLOT.load(Ordering::Acquire).wrapping_sub(1);
    // SAFETY: `idx` is the slot `decide` allocated (enabled() was true).
    let c = unsafe { TlsGetValue(idx) } as *mut Cache;
    if !c.is_null() {
        return c;
    }
    new_cache(idx)
}

#[cold]
fn new_cache(idx: u32) -> *mut Cache {
    // SAFETY: a zeroed `Cache` is valid (null heads, zero counts).
    unsafe {
        let c = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, core::mem::size_of::<Cache>()) as *mut Cache;
        if !c.is_null() {
            TlsSetValue(idx, c.cast());
        }
        c
    }
}

#[inline]
fn class_layout(c: usize) -> Layout {
    // SAFETY: every class size is a non-zero multiple of 16 and ALIGN is a power of two.
    unsafe { Layout::from_size_align_unchecked(CLASS_SIZES[c], ALIGN) }
}

#[inline]
fn small(layout: &Layout) -> bool {
    layout.size() <= MAX_SMALL && layout.align() <= ALIGN
}

unsafe impl GlobalAlloc for ThreadCache {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY (whole body): `layout` is the caller's, a cached block is a
        // live block of at least the class size and 16-byte alignment, and the
        // cache pointer is this thread's own.
        unsafe {
            if !small(&layout) || !enabled() {
                return System.alloc(layout);
            }
            let c = class_of(layout.size());
            let cache = cache();
            if !cache.is_null() {
                let head = (*cache).heads[c];
                if !head.is_null() {
                    (*cache).heads[c] = *(head as *mut *mut u8);
                    (*cache).counts[c] -= 1;
                    return head;
                }
            }
            System.alloc(class_layout(c))
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`; the block is at least `layout.size()` bytes.
        unsafe {
            if !small(&layout) || !enabled() {
                return System.alloc_zeroed(layout);
            }
            let p = self.alloc(layout);
            if !p.is_null() {
                core::ptr::write_bytes(p, 0, layout.size());
            }
            p
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by this allocator with `layout`, so a small
        // one is a class block (at least 16 bytes: room for the list link).
        unsafe {
            if !small(&layout) || !enabled() {
                return System.dealloc(ptr, layout);
            }
            let c = class_of(layout.size());
            let cache = cache();
            if !cache.is_null() && (*cache).counts[c] < MAX_PER_CLASS {
                *(ptr as *mut *mut u8) = (*cache).heads[c];
                (*cache).heads[c] = ptr;
                (*cache).counts[c] += 1;
                return;
            }
            System.dealloc(ptr, class_layout(c))
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract for `realloc`; blocks move only between
        // this allocator's own paths.
        unsafe {
            if !enabled() {
                return System.realloc(ptr, layout, new_size);
            }
            let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
            let (old_small, new_small) = (small(&layout), small(&new_layout));
            if !old_small && !new_small {
                return System.realloc(ptr, layout, new_size);
            }
            if old_small && new_small && class_of(layout.size()) == class_of(new_size) {
                return ptr;
            }
            let p = self.alloc(new_layout);
            if !p.is_null() {
                core::ptr::copy_nonoverlapping(ptr, p, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
            p
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_size_fits_its_class() {
        for size in 0..=MAX_SMALL {
            let c = class_of(size);
            assert!(CLASS_SIZES[c] >= size, "size {size} class {c}");
            assert!(c == 0 || CLASS_SIZES[c - 1] < size, "size {size} class {c} not the smallest");
        }
    }

    #[test]
    fn blocks_round_trip_through_the_cache() {
        let a = ThreadCache;
        unsafe {
            let l = Layout::from_size_align(100, 8).unwrap();
            let p = a.alloc(l);
            assert!(!p.is_null());
            core::ptr::write_bytes(p, 0xAB, 100);
            a.dealloc(p, l);
            let q = a.alloc(l);
            assert!(!q.is_null());
            let r = a.realloc(q, l, 120); // same class (128)
            assert_eq!(r, q);
            let s = a.realloc(r, Layout::from_size_align(120, 8).unwrap(), 5000);
            assert!(!s.is_null());
            a.dealloc(s, Layout::from_size_align(5000, 8).unwrap());
            let z = a.alloc_zeroed(Layout::from_size_align(64, 16).unwrap());
            assert!((0..64).all(|i| *z.add(i) == 0));
            a.dealloc(z, Layout::from_size_align(64, 16).unwrap());
        }
    }
}
