//! TEMP (lab, `REIMS_VGPU_LAB_CONTENT_RESIDENT=on`, Mish's own-allocator idea in
//! the form that fits this guest): staging copies of guest buffer windows kept
//! by *content*, not by address or write witness.
//!
//! # Why not BUF_RESIDENT
//!
//! b653: 89-91 % of the bytes Maps copies out of guest RAM repeat the bytes the
//! same window was copied with before. BUF_RESIDENT (b657/b658) tried to keep
//! those copies on the write witness and lost: the guest *writes the same bytes
//! again* every frame, so the pages are dirty while the content is not, and the
//! witness refused nearly everything (80 MB/s kept of 1.6 GB/s copied) while
//! costing ~20 % canvas to run.
//!
//! # The rule here
//!
//! A kept slot is **immutable**: it is keyed by `(len, 128-bit digest of its
//! bytes)`, and the digest is computed by the very loop that wrote those bytes
//! into it ([`copy_and_digest`]), so key and content cannot disagree even when
//! the guest races the copy. A bind digests the window's guest bytes (a read,
//! no write); a kept slot under that key holds exactly those bytes, so binding
//! it is the copy the CPU would have made at that instant. A miss copies into
//! a fresh staging slot with the fused loop and keeps it under the digest the
//! copy produced.
//!
//! Because a kept slot is never written again there is no write-after-read
//! hazard against earlier command buffers, and no barrier: the only lifetime
//! rule is that a slot let go (replaced or LRU-evicted) returns to the pool
//! through the *current* entry's live list, freed after this command buffer's
//! fence, which on the one queue is after every earlier command buffer that
//! could have bound it.

use std::collections::HashMap;

use super::BufferSlot;
use super::ResourcePools;

/// `(len, digest lo, digest hi)`.
pub(crate) type ContentKey = (u64, u64, u64);

struct Kept {
    slot: BufferSlot,
    last: u64,
}

/// The digest is already uniform: hash the key by folding its words, not
/// through SipHash.
#[derive(Default, Clone, Copy)]
pub(crate) struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0 ^ u64::from(*b)).wrapping_mul(P1);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = self.0.rotate_left(23) ^ v;
    }
}

type KeyBuild = std::hash::BuildHasherDefault<KeyHasher>;

/// Slots of the first-sight table (a power of two).
const SEEN_SLOTS: usize = 1 << 16;

#[derive(Default)]
pub(crate) struct ContentResident {
    kept: HashMap<ContentKey, Kept, KeyBuild>,
    /// Direct-mapped digests seen once and not kept: a content is kept on its
    /// second sight, so one-off bytes (most of a scene's first pass) leave
    /// their slot to the pool instead of pinning it.
    seen: Vec<ContentKey>,
    bytes: u64,
    tick: u64,
}

/// Whether the content cache is on (`REIMS_VGPU_LAB_CONTENT_RESIDENT=on`, live
/// `CONTENT_RESIDENT`).
pub(crate) fn on() -> bool {
    if let Some(on) = crate::runtime::lab_live::on("CONTENT_RESIDENT") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_CONTENT_RESIDENT").as_deref() == Ok("on"))
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Bytes of kept slots before the least recently bound is let go
/// (`REIMS_VGPU_LAB_CONTENT_RESIDENT_MB`, default 512).
fn byte_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| env_u64("REIMS_VGPU_LAB_CONTENT_RESIDENT_MB", 512) << 20)
}

/// Windows outside `[min, max]` bytes take the plain copy: below, the lookup
/// costs about what the copy does; above, one slot would pin a large share of
/// the cap (`REIMS_VGPU_LAB_CONTENT_RESIDENT_MIN` / `_MAX`, default 256 B and
/// 8 MiB).
pub(crate) fn window_fits(len: u64) -> bool {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    static MAX: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let min = *MIN.get_or_init(|| env_u64("REIMS_VGPU_LAB_CONTENT_RESIDENT_MIN", 256));
    let max = *MAX.get_or_init(|| env_u64("REIMS_VGPU_LAB_CONTENT_RESIDENT_MAX", 8 << 20));
    (min..=max).contains(&len)
}

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;

#[inline(always)]
fn round(acc: u64, w: u64) -> u64 {
    acc.wrapping_add(w.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

#[inline(always)]
fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

/// Running state of the 128-bit digest: xxh64's four-lane round over 32-byte
/// stripes, with the lanes kept apart until the end so two independent 64-bit
/// halves come out. Streaming, so a window made of several guest runs folds
/// run by run with the same result as one contiguous slice.
#[derive(Clone)]
pub(crate) struct Digest {
    acc: [u64; 4],
    tail: [u8; 32],
    tail_len: usize,
    total: u64,
}

impl Digest {
    pub(crate) fn new() -> Self {
        Self {
            acc: [
                P1.wrapping_add(P2),
                P2,
                0,
                0u64.wrapping_sub(P1),
            ],
            tail: [0; 32],
            tail_len: 0,
            total: 0,
        }
    }

    #[inline(always)]
    fn stripe(&mut self, s: &[u8]) {
        for (lane, acc) in self.acc.iter_mut().enumerate() {
            let w = u64::from_le_bytes(s[lane * 8..lane * 8 + 8].try_into().unwrap());
            *acc = round(*acc, w);
        }
    }

    /// Fold `src`, and when `dst` is given write the same bytes there, each
    /// destination byte exactly once and never read (a mapped staging span may
    /// be write-combined).
    ///
    /// # Safety
    /// `src` is readable and `dst` (when non-null) writable for `n` bytes, and
    /// they do not overlap.
    #[inline]
    pub(crate) unsafe fn update(&mut self, src: *const u8, dst: *mut u8, n: usize) {
        self.total += n as u64;
        let mut i = 0usize;
        // Finish a stripe left over from the previous run.
        if self.tail_len > 0 {
            let take = (32 - self.tail_len).min(n);
            // SAFETY: i + take <= n.
            unsafe {
                std::ptr::copy_nonoverlapping(src, self.tail.as_mut_ptr().add(self.tail_len), take);
                if !dst.is_null() {
                    std::ptr::copy_nonoverlapping(src, dst, take);
                }
            }
            self.tail_len += take;
            i = take;
            if self.tail_len == 32 {
                let t = self.tail;
                self.stripe(&t);
                self.tail_len = 0;
            }
        }
        while i + 32 <= n {
            // SAFETY: i + 32 <= n.
            let mut s = [0u8; 32];
            unsafe {
                std::ptr::copy_nonoverlapping(src.add(i), s.as_mut_ptr(), 32);
                if !dst.is_null() {
                    std::ptr::copy_nonoverlapping(s.as_ptr(), dst.add(i), 32);
                }
            }
            self.stripe(&s);
            i += 32;
        }
        if i < n {
            let rest = n - i;
            // SAFETY: i + rest == n; rest < 32 and the tail is empty here.
            unsafe {
                std::ptr::copy_nonoverlapping(src.add(i), self.tail.as_mut_ptr(), rest);
                if !dst.is_null() {
                    std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), rest);
                }
            }
            self.tail_len = rest;
        }
    }

    pub(crate) fn finish(&self) -> (u64, u64) {
        let mut acc = self.acc;
        // The partial stripe, zero-padded; the length below tells it apart from
        // real zeros.
        if self.tail_len > 0 {
            let mut s = [0u8; 32];
            s[..self.tail_len].copy_from_slice(&self.tail[..self.tail_len]);
            for (lane, a) in acc.iter_mut().enumerate() {
                let w = u64::from_le_bytes(s[lane * 8..lane * 8 + 8].try_into().unwrap());
                *a = round(*a, w ^ P4);
            }
        }
        let lo = avalanche(
            acc[0].rotate_left(1)
                ^ acc[1].rotate_left(7)
                ^ acc[2].rotate_left(12).wrapping_mul(P3)
                ^ acc[3].rotate_left(18).wrapping_mul(P4)
                ^ self.total.wrapping_mul(P1),
        );
        let hi = avalanche(
            acc[0].wrapping_mul(P4)
                ^ acc[1].rotate_left(23).wrapping_mul(P2)
                ^ acc[2].rotate_left(41)
                ^ acc[3].wrapping_mul(P3)
                ^ (self.total ^ self.tail_len as u64).wrapping_mul(P2).rotate_left(17),
        );
        (lo, hi)
    }
}

/// Digest of the window `[skip, skip + total)` of `runs`, read only. `None` when
/// the runs do not cover it (the caller's copy zero-fills a short read; a short
/// window is never kept).
pub(crate) fn digest_runs(
    runs: &[crate::backend::vulkan::engine::types::GuestRun],
    mut skip: u64,
    total: u64,
) -> Option<ContentKey> {
    let mut d = Digest::new();
    let mut left = total;
    for run in runs {
        if left == 0 {
            break;
        }
        if skip >= run.len() {
            skip -= run.len();
            continue;
        }
        let n = (run.len() - skip).min(left);
        // SAFETY: `host_ptr` is a stable RAMBlock alias valid for the VM
        // lifetime, readable for `run.len()` bytes; skip + n <= run.len().
        unsafe {
            d.update((run.host_ptr() as *const u8).add(skip as usize), std::ptr::null_mut(), n as usize)
        };
        skip = 0;
        left -= n;
    }
    if left != 0 {
        return None;
    }
    let (lo, hi) = d.finish();
    Some((total, lo, hi))
}

impl ResourcePools {
    /// Whether a miss on `key` should be kept: true from its second sight on
    /// (and the first sight is remembered).
    pub(crate) fn content_resident_second_sight(&mut self, key: ContentKey) -> bool {
        let res = &mut self.content_resident;
        if res.seen.is_empty() {
            res.seen = vec![(0, 0, 0); SEEN_SLOTS];
        }
        let at = (key.1 as usize) & (SEEN_SLOTS - 1);
        if res.seen[at] == key {
            return true;
        }
        res.seen[at] = key;
        crate::runtime::drain::note_store_route("crs_first_sight");
        false
    }

    /// The kept slot holding exactly `key`'s bytes.
    pub(crate) fn content_resident_hit(&mut self, key: ContentKey) -> Option<BufferSlot> {
        let res = &mut self.content_resident;
        res.tick += 1;
        let kept = res.kept.get_mut(&key)?;
        kept.last = res.tick;
        crate::runtime::drain::note_store_route("crs_hit");
        crate::runtime::drain::note_store_route_n("crs_hit_kb", key.0 / 1024);
        Some(kept.slot)
    }

    /// Keep `slot` — just taken from the staging pool and filled with bytes
    /// whose digest is `key` — past this command buffer. Whatever the byte cap
    /// pushes out returns to the pool after this entry's fence.
    pub(crate) fn content_resident_adopt(&mut self, key: ContentKey, slot: BufferSlot) {
        let live = &mut self.entry.staging_live;
        let Some(at) = live.iter().rposition(|s| s.buffer == slot.buffer) else {
            return;
        };
        let res = &mut self.content_resident;
        if res.kept.contains_key(&key) {
            // Two binds of one content in one command buffer that both missed
            // (the second came before the first was kept): the first copy stays.
            return;
        }
        live.swap_remove(at);
        res.tick += 1;
        res.bytes += slot.size;
        crate::runtime::drain::note_store_route("crs_adopt");
        crate::runtime::drain::note_store_route_n("crs_adopt_kb", key.0 / 1024);
        res.kept.insert(
            key,
            Kept {
                slot,
                last: res.tick,
            },
        );
        if res.bytes > byte_cap() {
            // Let the oldest eighth go in one pass rather than scanning the
            // whole map once per adopt at the cap.
            let mut by_age: Vec<(u64, ContentKey)> =
                res.kept.iter().filter(|(k, _)| **k != key).map(|(k, v)| (v.last, *k)).collect();
            by_age.sort_unstable();
            let target = byte_cap() - byte_cap() / 8;
            for (_, victim) in by_age {
                if res.bytes <= target {
                    break;
                }
                if let Some(gone) = res.kept.remove(&victim) {
                    res.bytes -= gone.slot.size;
                    live.push(gone.slot);
                    crate::runtime::drain::note_store_route("crs_evict");
                }
            }
        }
    }

    /// `write_staging_from_runs` and the digest of what it wrote, in one pass
    /// over the guest bytes. `None` for a short read (the zero-filled tail is
    /// not guest content, so that slot is never kept).
    ///
    /// Always on this thread, never deferred to the recording lane: the digest
    /// has to describe the bytes the slot holds when it is kept.
    pub(crate) unsafe fn content_resident_fill(
        &self,
        ctx: &super::DeviceContext,
        slot: &BufferSlot,
        runs: &[crate::backend::vulkan::engine::types::GuestRun],
        mut skip: u64,
        total: u64,
    ) -> Result<Option<ContentKey>, crate::backend::vulkan::engine::DrawError> {
        let size = total.max(4);
        // SAFETY: forwarded; the slot is this command buffer's, unaliased.
        let ptr = unsafe { super::staging_write_ptr(ctx, slot, size)? };
        let mut d = Digest::new();
        let mut off = 0u64;
        for run in runs {
            if off >= total {
                break;
            }
            if skip >= run.len() {
                skip -= run.len();
                continue;
            }
            let n = (run.len() - skip).min(total - off);
            // SAFETY: guest-RAM aliases are valid for the VM lifetime; the
            // mapped span is `size >= total` bytes and off + n <= total.
            unsafe {
                d.update(
                    (run.host_ptr() as *const u8).add(skip as usize),
                    ptr.add(off as usize),
                    n as usize,
                )
            };
            skip = 0;
            off += n;
        }
        if off < size {
            // SAFETY: inside the mapped span.
            unsafe { std::ptr::write_bytes(ptr.add(off as usize), 0, (size - off) as usize) };
        }
        if off != total {
            return Ok(None);
        }
        let (lo, hi) = d.finish();
        Ok(Some((total, lo, hi)))
    }

    /// Every kept slot, for teardown and device loss.
    pub(crate) fn content_resident_take_all(&mut self) -> Vec<BufferSlot> {
        let res = &mut self.content_resident;
        res.bytes = 0;
        res.kept.drain().map(|(_, k)| k.slot).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Digest;

    fn digest(parts: &[&[u8]]) -> (u64, u64) {
        let mut d = Digest::new();
        for p in parts {
            unsafe { d.update(p.as_ptr(), std::ptr::null_mut(), p.len()) };
        }
        d.finish()
    }

    #[test]
    fn split_points_do_not_change_the_digest() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + i / 13) as u8).collect();
        let whole = digest(&[&data]);
        for cut in [1usize, 5, 31, 32, 33, 64, 500, 999] {
            assert_eq!(digest(&[&data[..cut], &data[cut..]]), whole, "cut {cut}");
        }
        assert_eq!(digest(&[&data[..10], &data[10..40], &data[40..]]), whole);
    }

    #[test]
    fn length_and_content_change_the_digest() {
        let a = vec![0u8; 64];
        let b = vec![0u8; 65];
        assert_ne!(digest(&[&a]), digest(&[&b]));
        let mut c = a.clone();
        c[63] = 1;
        assert_ne!(digest(&[&a]), digest(&[&c]));
        assert_ne!(digest(&[&a[..3]]), digest(&[&a[..4]]));
    }

    #[test]
    fn the_copy_writes_exactly_the_source() {
        let data: Vec<u8> = (0..777u32).map(|i| (i ^ (i >> 3)) as u8).collect();
        let mut out = vec![0xAAu8; 777];
        let mut d = Digest::new();
        unsafe {
            d.update(data.as_ptr(), out.as_mut_ptr(), 100);
            d.update(data.as_ptr().add(100), out.as_mut_ptr().add(100), 677);
        }
        assert_eq!(out, data);
        assert_eq!(d.finish(), digest(&[&data]));
    }
}
