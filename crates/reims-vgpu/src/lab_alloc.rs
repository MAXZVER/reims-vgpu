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
//! A thread whose list of a class is full hands half of it, as one batch, to a
//! shared per-class depot, and a thread whose list is empty takes a batch back
//! before it asks the system heap. Without that, a block allocated on one thread
//! and freed on another — every deferred draw's resource vectors, built on the
//! drain and dropped on the recorder — went from the producer's empty list to
//! the system heap and from the consumer's full list back to it, a locked
//! `RtlAllocateHeap`/`RtlFreeHeap` pair per block (b101). `REIMS_VGPU_LAB_DEPOT=off`
//! frees past the cap to the system heap as before.
//!
//! `REIMS_VGPU_LAB_TCACHE=big` caches blocks up to 64 KiB (the system heap
//! serves blocks past its low-fragmentation sizes under its lock), each class
//! holding at most 1 MiB. `REIMS_VGPU_LAB_TCACHE=off` passes everything to the system heap. Read once,
//! at the first allocation, before any block exists, so no block can be freed
//! by a path other than the one that allocated it.
//!
//! Blocks of 1 MiB and more (whole frames: a 5K BGRA frame is 42 MiB) are kept
//! in one shared, bounded list of freed blocks, by 1 MiB bucket. The system heap
//! hands every such request fresh pages from the OS and returns them on free, so
//! each 5K readback copy paid ~11 000 first-touch faults (8-15 ms `map` in
//! `readback_split` during full-screen transitions). `REIMS_VGPU_LAB_LCACHE=off`
//! sends them to the system heap again.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, AtomicUsize, Ordering};

const CLASS_SIZES: [usize; 24] = [
    16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
    12288, 16384, 24576, 32768, 49152, 65536,
];
const MAX_SMALL: usize = 4096;
const MAX_BIG: usize = 65536;
const MAX_PER_CLASS: u32 = 256;
/// Bytes one class list may hold (caps the counts of the big classes).
const MAX_CLASS_BYTES: usize = 1 << 20;
/// The largest cached size, decided with the mode: 4 KiB (`on`) or 64 KiB (`big`).
static LIMIT: AtomicUsize = AtomicUsize::new(0);
const ALIGN: usize = 16;
/// Blocks of at least this size go through the large-block list.
const LARGE_MIN: usize = 1 << 20;
/// Large blocks are allocated in whole buckets of this size.
const LARGE_STEP: usize = 1 << 20;
/// Entries and bytes the large-block list may hold.
const LARGE_SLOTS: usize = 32;
const LARGE_CAP_BYTES: usize = 384 << 20;
/// Whether the large-block list is in use (1), decided with the mode.
static LARGE_ON: AtomicU8 = AtomicU8::new(0);
/// Whether the cross-thread depot is in use (1), decided with the mode.
static DEPOT_ON: AtomicU8 = AtomicU8::new(0);
/// Batches one class's depot holds; a batch past that goes back to the system heap.
const DEPOT_BATCHES: usize = 16;

/// One class's shared batches: intrusive lists (the block's first word links
/// to the next) with their lengths, under a spin lock.
struct Depot {
    lock: AtomicBool,
    heads: core::cell::UnsafeCell<[*mut u8; DEPOT_BATCHES]>,
    counts: core::cell::UnsafeCell<[u32; DEPOT_BATCHES]>,
    n: core::cell::UnsafeCell<usize>,
}

// SAFETY: the cells are only touched under `lock`.
unsafe impl Sync for Depot {}

#[allow(clippy::declare_interior_mutable_const)]
const DEPOT_INIT: Depot = Depot {
    lock: AtomicBool::new(false),
    heads: core::cell::UnsafeCell::new([core::ptr::null_mut(); DEPOT_BATCHES]),
    counts: core::cell::UnsafeCell::new([0; DEPOT_BATCHES]),
    n: core::cell::UnsafeCell::new(0),
};

static DEPOTS: [Depot; CLASS_SIZES.len()] = [DEPOT_INIT; CLASS_SIZES.len()];

impl Depot {
    fn lock(&self) {
        let mut spins = 0u32;
        while self
            .lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spins += 1;
            if spins < 64 {
                core::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    fn unlock(&self) {
        self.lock.store(false, Ordering::Release);
    }

    /// Keep a batch, or hand it back when the depot is full.
    fn put(&self, head: *mut u8, count: u32) -> bool {
        self.lock();
        // SAFETY: under `lock`.
        let kept = unsafe {
            let n = &mut *self.n.get();
            if *n < DEPOT_BATCHES {
                (*self.heads.get())[*n] = head;
                (*self.counts.get())[*n] = count;
                *n += 1;
                true
            } else {
                false
            }
        };
        self.unlock();
        kept
    }

    /// A batch, or `(null, 0)`.
    fn take(&self) -> (*mut u8, u32) {
        self.lock();
        // SAFETY: under `lock`.
        let got = unsafe {
            let n = &mut *self.n.get();
            if *n == 0 {
                (core::ptr::null_mut(), 0)
            } else {
                *n -= 1;
                ((*self.heads.get())[*n], (*self.counts.get())[*n])
            }
        };
        self.unlock();
        got
    }
}
static LARGE_LOCK: AtomicBool = AtomicBool::new(false);

struct LargeList {
    sizes: [usize; LARGE_SLOTS],
    ptrs: [*mut u8; LARGE_SLOTS],
    bytes: usize,
    victim: usize,
}

static mut LARGE: LargeList = LargeList {
    sizes: [0; LARGE_SLOTS],
    ptrs: [core::ptr::null_mut(); LARGE_SLOTS],
    bytes: 0,
    victim: 0,
};

/// Class index for a size of at most [`MAX_BIG`] bytes (a size of 0 is class 0).
#[inline]
fn class_of(size: usize) -> usize {
    if size > MAX_SMALL {
        let mut c = 16;
        while CLASS_SIZES[c] < size {
            c += 1;
        }
        return c;
    }
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
    let big = n == 3 && buf[0] == u16::from(b'b') && buf[1] == u16::from(b'i') && buf[2] == u16::from(b'g');
    LIMIT.store(if big { MAX_BIG } else { MAX_SMALL }, Ordering::Relaxed);
    const LNAME: &[u8] = b"REIMS_VGPU_LAB_LCACHE\0";
    let mut lname = [0u16; 32];
    for (i, b) in LNAME.iter().enumerate() {
        lname[i] = u16::from(*b);
    }
    let mut lbuf = [0u16; 8];
    // SAFETY: both buffers are ours and sized as passed.
    let ln = unsafe { GetEnvironmentVariableW(lname.as_ptr(), lbuf.as_mut_ptr(), lbuf.len() as u32) };
    let loff = ln == 3 && lbuf[0] == u16::from(b'o') && lbuf[1] == u16::from(b'f') && lbuf[2] == u16::from(b'f');
    LARGE_ON.store(u8::from(!off && !loff), Ordering::Relaxed);
    const DNAME: &[u8] = b"REIMS_VGPU_LAB_DEPOT\0";
    let mut dname = [0u16; 32];
    for (i, b) in DNAME.iter().enumerate() {
        dname[i] = u16::from(*b);
    }
    let mut dbuf = [0u16; 8];
    // SAFETY: both buffers are ours and sized as passed.
    let dn = unsafe { GetEnvironmentVariableW(dname.as_ptr(), dbuf.as_mut_ptr(), dbuf.len() as u32) };
    let doff = dn == 3 && dbuf[0] == u16::from(b'o') && dbuf[1] == u16::from(b'f') && dbuf[2] == u16::from(b'f');
    DEPOT_ON.store(u8::from(!off && !doff), Ordering::Relaxed);
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

/// Whether the block is cached; asked only once `enabled()` has decided the mode.
#[inline]
fn small(layout: &Layout) -> bool {
    layout.size() <= LIMIT.load(Ordering::Relaxed) && layout.align() <= ALIGN
}

#[inline]
fn class_cap(c: usize) -> u32 {
    MAX_PER_CLASS.min((MAX_CLASS_BYTES / CLASS_SIZES[c]) as u32)
}

/// Whether the block goes through the large-block list; asked only once
/// `enabled()` has decided the mode.
#[inline]
fn large(layout: &Layout) -> bool {
    layout.size() >= LARGE_MIN && layout.align() <= ALIGN && LARGE_ON.load(Ordering::Relaxed) == 1
}

#[inline]
fn bucket_layout(size: usize) -> Layout {
    // SAFETY: a non-zero multiple of 1 MiB with ALIGN a power of two.
    unsafe { Layout::from_size_align_unchecked(size.div_ceil(LARGE_STEP) * LARGE_STEP, ALIGN) }
}

fn large_lock() {
    let mut spins = 0u32;
    while LARGE_LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        spins += 1;
        if spins < 64 {
            core::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

fn large_unlock() {
    LARGE_LOCK.store(false, Ordering::Release);
}

/// A cached block of exactly `size` bytes (a bucket size), or null.
fn large_take(size: usize) -> *mut u8 {
    large_lock();
    // SAFETY: the list is only touched under LARGE_LOCK.
    let p = unsafe {
        let l = &mut *core::ptr::addr_of_mut!(LARGE);
        let mut p = core::ptr::null_mut();
        for i in 0..LARGE_SLOTS {
            if l.sizes[i] == size {
                p = l.ptrs[i];
                l.sizes[i] = 0;
                l.ptrs[i] = core::ptr::null_mut();
                l.bytes -= size;
                break;
            }
        }
        p
    };
    large_unlock();
    p
}

/// Keep a freed block of `size` bytes (a bucket size), evicting the oldest
/// entries when the list is full; evicted blocks go back to the system heap
/// outside the lock.
fn large_put(ptr: *mut u8, size: usize) {
    if size > LARGE_CAP_BYTES {
        // SAFETY: allocated by the system heap with this bucket layout.
        unsafe { System.dealloc(ptr, bucket_layout(size)) };
        return;
    }
    let mut evicted: [(*mut u8, usize); LARGE_SLOTS] = [(core::ptr::null_mut(), 0); LARGE_SLOTS];
    let mut n_evicted = 0;
    large_lock();
    // SAFETY: the list is only touched under LARGE_LOCK.
    unsafe {
        let l = &mut *core::ptr::addr_of_mut!(LARGE);
        loop {
            let free = (0..LARGE_SLOTS).find(|&i| l.sizes[i] == 0);
            if let (Some(i), true) = (free, l.bytes + size <= LARGE_CAP_BYTES) {
                l.sizes[i] = size;
                l.ptrs[i] = ptr;
                l.bytes += size;
                break;
            }
            let v = l.victim;
            l.victim = (v + 1) % LARGE_SLOTS;
            if l.sizes[v] != 0 {
                evicted[n_evicted] = (l.ptrs[v], l.sizes[v]);
                n_evicted += 1;
                l.bytes -= l.sizes[v];
                l.sizes[v] = 0;
                l.ptrs[v] = core::ptr::null_mut();
            }
        }
    }
    large_unlock();
    for &(p, sz) in &evicted[..n_evicted] {
        // SAFETY: each was allocated by the system heap with its bucket layout.
        unsafe { System.dealloc(p, bucket_layout(sz)) };
    }
}

unsafe impl GlobalAlloc for ThreadCache {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY (whole body): `layout` is the caller's, a cached block is a
        // live block of at least the class size and 16-byte alignment, and the
        // cache pointer is this thread's own.
        unsafe {
            if !enabled() {
                return System.alloc(layout);
            }
            if large(&layout) {
                let b = bucket_layout(layout.size());
                let p = large_take(b.size());
                return if p.is_null() { System.alloc(b) } else { p };
            }
            if !small(&layout) {
                return System.alloc(layout);
            }
            let c = class_of(layout.size());
            let cache = cache();
            if !cache.is_null() {
                let mut head = (*cache).heads[c];
                if head.is_null() && DEPOT_ON.load(Ordering::Relaxed) == 1 {
                    // Empty: a batch some other thread freed, before the heap.
                    let (h, n) = DEPOTS[c].take();
                    if !h.is_null() {
                        (*cache).heads[c] = h;
                        (*cache).counts[c] = n;
                        head = h;
                    }
                }
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
            if !enabled() {
                return System.alloc_zeroed(layout);
            }
            if large(&layout) {
                let b = bucket_layout(layout.size());
                let p = large_take(b.size());
                if p.is_null() {
                    return System.alloc_zeroed(b);
                }
                core::ptr::write_bytes(p, 0, layout.size());
                return p;
            }
            if !small(&layout) {
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
            if !enabled() {
                return System.dealloc(ptr, layout);
            }
            if large(&layout) {
                return large_put(ptr, bucket_layout(layout.size()).size());
            }
            if !small(&layout) {
                return System.dealloc(ptr, layout);
            }
            let c = class_of(layout.size());
            let cache = cache();
            if !cache.is_null() && (*cache).counts[c] >= class_cap(c) && class_cap(c) >= 2
                && DEPOT_ON.load(Ordering::Relaxed) == 1
            {
                // Full: the older half of the list (the blocks past its first
                // half) goes to the depot as one batch, for a thread that
                // allocates what this one frees.
                let keep = class_cap(c) / 2;
                let mut tail = (*cache).heads[c];
                for _ in 1..keep {
                    tail = *(tail as *mut *mut u8);
                }
                let batch = *(tail as *mut *mut u8);
                let moved = (*cache).counts[c] - keep;
                if !batch.is_null() && DEPOTS[c].put(batch, moved) {
                    *(tail as *mut *mut u8) = core::ptr::null_mut();
                    (*cache).counts[c] = keep;
                }
            }
            if !cache.is_null() && (*cache).counts[c] < class_cap(c) {
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
            let (old_large, new_large) = (large(&layout), large(&new_layout));
            if old_large || new_large {
                // A large block is a whole bucket and is returned as one, so it
                // never goes through the system heap's realloc (which would hand
                // back a block of the requested size, not the bucket's).
                if old_large
                    && new_large
                    && bucket_layout(layout.size()).size() == bucket_layout(new_size).size()
                {
                    return ptr;
                }
                let p = self.alloc(new_layout);
                if !p.is_null() {
                    core::ptr::copy_nonoverlapping(ptr, p, layout.size().min(new_size));
                    self.dealloc(ptr, layout);
                }
                return p;
            }
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

    /// Blocks made on one thread and freed on another come back through the
    /// depot intact: every block handed out is a whole, writable class block,
    /// and the producer's later allocations are served (the depot holds what
    /// the consumer's full list passed on).
    #[test]
    fn blocks_freed_on_another_thread_return_through_the_depot() {
        if !enabled() {
            return; // switched off in this environment
        }
        let l = Layout::from_size_align(200, 8).unwrap();
        for round in 0..20u8 {
            let blocks: Vec<usize> = (0..600)
                .map(|i| unsafe {
                    let p = ThreadCache.alloc(l);
                    assert!(!p.is_null());
                    core::ptr::write_bytes(p, round ^ (i as u8), l.size());
                    p as usize
                })
                .collect();
            std::thread::spawn(move || {
                for (i, &p) in blocks.iter().enumerate() {
                    unsafe {
                        let p = p as *mut u8;
                        // Past the link word, the bytes are still the producer's.
                        assert!((8..l.size()).all(|k| *p.add(k) == round ^ (i as u8)));
                        ThreadCache.dealloc(p, l);
                    }
                }
            })
            .join()
            .unwrap();
        }
    }

    #[test]
    fn every_size_fits_its_class() {
        for size in 0..=MAX_BIG {
            let c = class_of(size);
            assert!(CLASS_SIZES[c] >= size, "size {size} class {c}");
            assert!(c == 0 || CLASS_SIZES[c - 1] < size, "size {size} class {c} not the smallest");
        }
    }

    #[test]
    fn large_blocks_are_whole_buckets_reused_and_zeroed_on_request() {
        let a = ThreadCache;
        unsafe {
            if !enabled() || LARGE_ON.load(Ordering::Relaxed) != 1 {
                return; // switched off in this environment
            }
            let l = Layout::from_size_align(5120 * 2160 * 4, 16).unwrap();
            let p = a.alloc(l);
            assert!(!p.is_null());
            core::ptr::write_bytes(p, 0xCD, l.size());
            a.dealloc(p, l);
            // Same bucket: the freed block comes back, and zeroed when asked.
            let z = a.alloc_zeroed(Layout::from_size_align(5120 * 2160 * 4 - 100, 16).unwrap());
            assert!(!z.is_null());
            assert!((0..5120 * 2160 * 4 - 100).step_by(4093).all(|i| *z.add(i) == 0));
            // Growing within the bucket keeps the block; past it moves the bytes.
            let zl = Layout::from_size_align(5120 * 2160 * 4 - 100, 16).unwrap();
            *z = 7;
            let g = a.realloc(z, zl, 5120 * 2160 * 4);
            assert_eq!(g, z);
            let h = a.realloc(g, l, 64 << 20);
            assert!(!h.is_null());
            assert_eq!(*h, 7);
            // Shrinking to a small block leaves the large path.
            let s = a.realloc(h, Layout::from_size_align(64 << 20, 16).unwrap(), 100);
            assert_eq!(*s, 7);
            a.dealloc(s, Layout::from_size_align(100, 16).unwrap());
            // Filling past the cap evicts instead of growing without bound.
            let big = Layout::from_size_align(100 << 20, 16).unwrap();
            let ps: Vec<*mut u8> = (0..6).map(|_| a.alloc(big)).collect();
            for &q in &ps {
                assert!(!q.is_null());
                a.dealloc(q, big);
            }
            let held = (*core::ptr::addr_of!(LARGE)).bytes;
            assert!(held <= LARGE_CAP_BYTES, "held {held}");
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
