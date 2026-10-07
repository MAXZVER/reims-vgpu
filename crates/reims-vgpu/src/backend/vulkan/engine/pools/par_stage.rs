//! TEMP (lab, live `PAR_STAGE=on|off`, off until its A/B): a large staging
//! copy from guest runs into a mapped staging span, split over a few helper
//! threads.
//!
//! The recorder stages a frame-sized guest texture (a 4935x1895 window
//! snapshot is 37 MB) into a mapped staging slot with one `memmove` per guest
//! run, on its own thread. p078: in the tranches that open a transition's
//! animation and in its hitches the drain waits on the recorder (17 % / 16 %),
//! and that copy is 31 % of the recorder there - one thread moving cold guest
//! pages at a few GB/s while the host has cores idle. The copy has no ordering
//! inside it (disjoint destinations, read-only sources that stay mapped for the
//! VM's life), so it splits by bytes.
//!
//! Helpers touch only the two byte ranges they are handed - never QEMU state or
//! the guest's FlatView - so they need no RCU registration.

use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Bytes below which a copy stays on the caller: a hand-off costs tens of
/// microseconds, which one thread spends copying about 256 KiB.
pub(crate) const PAR_STAGE_MIN_BYTES: usize = 8 << 20;
/// Helper threads (the caller copies a share too).
const HELPERS: usize = 3;

/// One contiguous copy: `n` bytes from `src` to `dst`.
#[derive(Clone, Copy)]
pub(crate) struct Segment {
    pub(crate) src: *const u8,
    pub(crate) dst: *mut u8,
    pub(crate) n: usize,
}

// SAFETY: a segment names a source inside guest RAM that stays mapped for the
// VM's lifetime and a destination inside the caller's mapped staging span;
// `copy_segments` returns only after every helper has finished with it, and the
// segments it hands out are pairwise disjoint in their destinations.
unsafe impl Send for Segment {}

struct Batch {
    segs: Vec<Segment>,
    done: Arc<(Mutex<usize>, Condvar)>,
}

fn helpers() -> &'static [Mutex<Sender<Batch>>] {
    static POOL: OnceLock<Vec<Mutex<Sender<Batch>>>> = OnceLock::new();
    POOL.get_or_init(|| {
        (0..HELPERS)
            .filter_map(|i| {
                let (tx, rx) = channel::<Batch>();
                std::thread::Builder::new()
                    .name(format!("reims-vgpu-stage-{i}"))
                    .spawn(move || {
                        while let Ok(batch) = rx.recv() {
                            for s in &batch.segs {
                                // SAFETY: see `Segment`.
                                unsafe { std::ptr::copy_nonoverlapping(s.src, s.dst, s.n) };
                            }
                            let (lock, cv) = &*batch.done;
                            let mut left = lock.lock().unwrap_or_else(|e| e.into_inner());
                            *left -= 1;
                            if *left == 0 {
                                cv.notify_all();
                            }
                        }
                    })
                    .ok()
                    .map(|_| Mutex::new(tx))
            })
            .collect()
    })
}

/// Whether the live switch asks for the split.
pub(crate) fn on() -> bool {
    if cfg!(test) {
        return false;
    }
    crate::runtime::lab_live::on("PAR_STAGE").unwrap_or(false)
}

/// Copy `segs` (destinations pairwise disjoint) using the helpers and this
/// thread; returns when every byte is in. Falls back to copying here when no
/// helper could be started or one refuses the batch.
///
/// # Safety
/// Every segment's source must be readable and destination writable for `n`
/// bytes until this returns, and no two destinations may overlap.
pub(crate) unsafe fn copy_segments(segs: &[Segment]) {
    let total: usize = segs.iter().map(|s| s.n).sum();
    let pool = helpers();
    if pool.is_empty() || total < PAR_STAGE_MIN_BYTES {
        for s in segs {
            unsafe { std::ptr::copy_nonoverlapping(s.src, s.dst, s.n) };
        }
        return;
    }
    // Cut the work into `pool.len() + 1` shares of about equal bytes, splitting
    // a segment where a share boundary falls inside it.
    let shares = pool.len() + 1;
    let per = total.div_ceil(shares);
    let mut groups: Vec<Vec<Segment>> = vec![Vec::new(); shares];
    let (mut g, mut room) = (0usize, per);
    for s in segs {
        let (mut src, mut dst, mut n) = (s.src, s.dst, s.n);
        while n > 0 {
            if room == 0 && g + 1 < shares {
                g += 1;
                room = per;
            }
            let take = if g + 1 == shares { n } else { n.min(room) };
            groups[g].push(Segment { src, dst, n: take });
            // SAFETY: `take <= n`, inside the segment.
            unsafe {
                src = src.add(take);
                dst = dst.add(take);
            }
            n -= take;
            room = room.saturating_sub(take);
        }
    }
    let mine = groups.pop().unwrap_or_default();
    let done = Arc::new((Mutex::new(0usize), Condvar::new()));
    let mut handed = 0usize;
    for (tx, segs) in pool.iter().zip(groups) {
        if segs.is_empty() {
            continue;
        }
        *done.0.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        let batch = Batch {
            segs,
            done: Arc::clone(&done),
        };
        let sent = tx.lock().unwrap_or_else(|e| e.into_inner()).send(batch);
        match sent {
            Ok(()) => handed += 1,
            Err(back) => {
                // The helper is gone: its share is copied here.
                *done.0.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
                for s in &back.0.segs {
                    unsafe { std::ptr::copy_nonoverlapping(s.src, s.dst, s.n) };
                }
            }
        }
    }
    for s in &mine {
        unsafe { std::ptr::copy_nonoverlapping(s.src, s.dst, s.n) };
    }
    if handed > 0 {
        let (lock, cv) = &*done;
        let mut left = lock.lock().unwrap_or_else(|e| e.into_inner());
        while *left > 0 {
            left = cv.wait(left).unwrap_or_else(|e| e.into_inner());
        }
    }
    crate::runtime::drain::note_store_route("lab_par_stage");
    crate::runtime::drain::note_store_route_n("lab_par_stage_kb", (total / 1024) as u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shares cut segments where a boundary falls and lose no byte.
    #[test]
    fn a_split_copy_lands_every_byte_where_the_serial_copy_does() {
        let src: Vec<u8> = (0..(PAR_STAGE_MIN_BYTES + 12_345)).map(|i| (i * 7 + i / 4096) as u8).collect();
        let mut dst = vec![0u8; src.len() + 64];
        // Three uneven segments, reordered in the destination.
        let a = 1_000_003usize;
        let b = 5_000_011usize;
        let segs = [
            Segment { src: src.as_ptr(), dst: unsafe { dst.as_mut_ptr().add(64) }, n: a },
            Segment { src: unsafe { src.as_ptr().add(a) }, dst: unsafe { dst.as_mut_ptr().add(64 + a) }, n: b },
            Segment {
                src: unsafe { src.as_ptr().add(a + b) },
                dst: unsafe { dst.as_mut_ptr().add(64 + a + b) },
                n: src.len() - a - b,
            },
        ];
        unsafe { copy_segments(&segs) };
        assert!(dst[..64].iter().all(|&x| x == 0));
        assert!(dst[64..] == src[..]);
    }
}
