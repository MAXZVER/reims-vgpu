//! TEMP (lab, `REIMS_VGPU_LAB_PASS_STAGE`): the planning half of worker
//! staging (design §8.4 / C1).
//!
//! The engine half (`crate::backend::vulkan::engine::pass_stage`) owns the
//! arenas and the facts; this module is what a pool job holds while it plans:
//! a [`Stager`] over its own thread's arena, with the three memcpys the
//! design moves off the recorder —
//!
//! * **bind bytes** — the CPU read + staging the serial buffer ladder pays
//!   when every zero-copy rail refuses ([`buffer_plan`]'s staged arm);
//! * **sampled gathers** — `write_staging_from_runs`'s window, staged beside
//!   the guest-runs plan that would pay it ([`sample_plan`]'s attach);
//! * **seeds** — the R/B exchange and staging of a CPU load seed, staged
//!   while the request sits in the recorder FIFO ([`arm_seed_stage`] and the
//!   seed jobs `PassPrep::ahead` feeds).
//!
//! # Freshness
//!
//! A staged read happens at plan time; the serial CPU paths read at bind
//! time. Both precede command-buffer execution, and the seed/gather CPU
//! paths' own contract already accepts a read that races guest CPU writes
//! ("exactly as the encode-time staging read does"). What a plan may not
//! miss is a **device** write: the serial paths settle writeback debt before
//! reading, so the staged arms bail on any debt (`OwedFrame`-style) and the
//! bind plan re-checks `host_writes` between its read and the commit — a
//! writeback that landed in the window invalidates the plan and the serial
//! ladder runs, exactly as it would have.

use crate::backend::vulkan::engine::pass_stage::{StagedSlice, WorkerArena};

/// What one pool job stages through: its own thread's arena.
///
/// Built per job from the [`crate::runtime::par::WorkerEnv`], so a job stages
/// in the arena of whoever runs it -- no cross-thread carving, no lock on the
/// write path. The arena set travels by `Arc` (a job's stager must outlive
/// the call that produced the set). `None` (from [`stager_for`]) whenever the
/// rail cannot run: the flag off, `shadow` mode (the compare must see the
/// untouched serial answers), or the arenas not yet created.
pub(crate) struct Stager {
    arenas: std::sync::Arc<[WorkerArena]>,
    index: usize,
    align: u64,
}

impl Stager {
    /// A stager over a stated member of a stated set -- the constructor
    /// [`stager_for`] resolves from the job's environment, spelled for the
    /// tests that have no pool to build one.
    pub(crate) fn of(arenas: std::sync::Arc<[WorkerArena]>, index: usize) -> Option<Self> {
        let align = arenas.get(index)?.bind_align;
        Some(Self {
            arenas,
            index,
            align,
        })
    }

    fn arena(&self) -> &WorkerArena {
        &self.arenas[self.index]
    }

    /// Stage plain bytes (a bind's CPU read, a seed in its own order).
    pub(crate) fn stage_bytes(&self, bytes: &[u8]) -> Option<StagedSlice> {
        if bytes.is_empty() {
            return None;
        }
        let slice = self.arena().alloc(bytes.len() as u64, self.align).ok()?;
        // SAFETY: the window was just carved for `bytes.len()` bytes and
        // holds one reference (the slice's own); the mapping is host-coherent
        // (`MemoryClass::Upload`), so the copy is what the GPU will read.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (slice.mapped as *mut u8).add(slice.window.offset as usize),
                bytes.len(),
            );
        }
        Some(slice)
    }

    /// Stage a seed with the R/B exchange folded into the copy -- the exact
    /// bytes `write_staging_swap_rb` would have written (whole-pixel stores,
    /// never a read of the mapped span).
    pub(crate) fn stage_seed(&self, bytes: &[u8], swap_rb: bool) -> Option<StagedSlice> {
        if !swap_rb {
            return self.stage_bytes(bytes);
        }
        if bytes.len() % 4 != 0 {
            // A seed is whole RGBA8 pixels; a ragged tail is not a seed this
            // rail knows how to state, and the serial arm keeps it.
            return None;
        }
        let slice = self.stage_bytes(bytes)?;
        // SAFETY: as `stage_bytes`, over a whole number of pixels.
        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                (slice.mapped as *mut u8).add(slice.window.offset as usize),
                bytes.len(),
            );
            for (src, px) in bytes.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                px[0] = src[2];
                px[1] = src[1];
                px[2] = src[0];
                px[3] = src[3];
            }
        }
        Some(slice)
    }

    /// Stage a guest-runs window -- the exact copy `write_staging_from_runs`
    /// performs: each run's bytes at their running offset past
    /// `source_offset`, the tail zeroed on a short read.
    pub(crate) fn stage_runs(
        &self,
        runs: &[crate::backend::vulkan::engine::GuestRun],
        source_offset: u64,
        total_len: u64,
    ) -> Option<StagedSlice> {
        if total_len == 0 {
            return None;
        }
        let slice = self.arena().alloc(total_len, self.align).ok()?;
        // SAFETY: the window covers `total_len`; each `run.host_ptr()` is a
        // stable RAMBlock alias valid for the VM's lifetime, and the offsets
        // walk the same arithmetic `write_staging_from_runs` walks.
        unsafe {
            let dst = (slice.mapped as *mut u8).add(slice.window.offset as usize);
            let total = total_len as usize;
            let mut off = 0usize;
            let mut skip = source_offset;
            for run in runs {
                if off >= total {
                    break;
                }
                if skip >= run.len() {
                    skip -= run.len();
                    continue;
                }
                let within = skip as usize;
                skip = 0;
                let available = (run.len() as usize).saturating_sub(within);
                let n = available.min(total - off);
                std::ptr::copy_nonoverlapping(
                    (run.host_ptr() as *const u8).add(within),
                    dst.add(off),
                    n,
                );
                off += n;
            }
            if off < total {
                std::ptr::write_bytes(dst.add(off), 0, total - off);
            }
        }
        Some(slice)
    }
}

/// The stager for the thread `env` describes, when the rail may run here.
pub(crate) fn stager_for(env: &crate::runtime::par::WorkerEnv<'_>) -> Option<Stager> {
    if !crate::runtime::par::staging::enabled() {
        return None;
    }
    // `shadow` plans and compares the serial ladder; a staged fact answers
    // before the ladder runs, so the mode sees none of them.
    if super::pass_prep::pass_prep_mode() != super::pass_prep::PassPrepMode::On {
        return None;
    }
    let arenas = crate::backend::vulkan::engine::pass_stage::arenas()?;
    Stager::of(arenas, usize::from(env.worker.0))
}

// The seed cells armed on this drain thread and not yet offered to a scope.
//
// The drain arms a cell when it defers a seeded draw; the next chunk's
// `ahead` takes the list (filtering cells already spent) and offers each as
// a staging job. Thread-local because only the drain arms and only the
// drain's `ahead` collects.
thread_local! {
    static PENDING_SEEDS: std::cell::RefCell<Vec<std::sync::Arc<crate::backend::vulkan::engine::pass_stage::SeedStageCell>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Arm the seed staging of a deferred draw about to be submitted: attach a
/// cell to the request when every gate admits, and remember it for the next
/// chunk's scope.
///
/// Called at the defer site, where the request is whole: the seed bytes, the
/// attachment's channel order and four-byte-ness, and the recorder FIFO the
/// worker races are all decided. A gate that refuses leaves the request
/// byte-for-byte the old one (`target_seed_stage: None`).
pub(crate) fn arm_seed_stage(resources: &mut crate::backend::vulkan::engine::DrawRequest) {
    use crate::backend::vulkan::engine::pass_stage::SeedStageCell;
    if !crate::runtime::par::staging::enabled() {
        return;
    }
    if super::pass_prep::pass_prep_mode() != super::pass_prep::PassPrepMode::On {
        return;
    }
    let Some(bytes) = resources.target_rgba8.as_ref() else {
        return;
    };
    // The four-byte arm only: the wide arm restates the seed in the
    // attachment's texel first, a transformation this rail does not carry.
    let four_byte = resources
        .color_attachment
        .map(|a| a.format())
        .or_else(|| {
            resources
                .target_identity
                .as_ref()
                .map(|id| id.resident_format())
        })
        .and_then(crate::backend::vulkan::translate::pixel::texel_layout_of)
        .is_some_and(|layout| layout.is_four_byte_color());
    if !four_byte {
        return;
    }
    if bytes.len() as u64 > crate::runtime::par::staging::chunk_bytes() {
        // Larger than a chunk: the serial staging slot is the only home.
        crate::runtime::drain::note_store_route_n("pass_stage_seed_oversize", bytes.len() as u64);
        return;
    }
    let Some(arenas) = crate::backend::vulkan::engine::pass_stage::arenas() else {
        // Not created yet (the boot's first draws); the serial arm serves.
        return;
    };
    let _ = arenas;
    // The engine's own exchange predicate, spelled from the same two facts it
    // reads: the seed's order against the attachment's.
    let swap_rb = matches!(
        resources.target_seed_order,
        crate::backend::vulkan::engine::SeedOrder::Bgra8
    ) != resources
        .target_identity
        .as_ref()
        .is_some_and(|id| id.is_bgra());
    let cell = std::sync::Arc::new(SeedStageCell::new(std::sync::Arc::clone(bytes), swap_rb));
    resources.target_seed_stage = Some(std::sync::Arc::clone(&cell));
    PENDING_SEEDS.with(|pending| pending.borrow_mut().push(cell));
}

/// Take the armed seed cells (the caller filters the spent ones); the next
/// chunk's scope turns each into a staging job.
pub(crate) fn take_pending_seeds(
) -> Vec<std::sync::Arc<crate::backend::vulkan::engine::pass_stage::SeedStageCell>> {
    PENDING_SEEDS.with(|pending| std::mem::take(&mut *pending.borrow_mut()))
}

/// One seed staging job: copy the cell's bytes into `arena`, exchange folded,
/// and publish the window — or give the cell back when the arena refused.
pub(crate) fn stage_seed_job(
    stager: &Stager,
    cell: &crate::backend::vulkan::engine::pass_stage::SeedStageCell,
) {
    let Some((bytes, swap_rb)) = cell.take_pending() else {
        return;
    };
    match stager.stage_seed(&bytes, swap_rb) {
        Some(slice) => {
            crate::runtime::drain::note_store_route("pass_stage_seed_planned");
            crate::runtime::drain::note_store_route_n("pass_stage_seed_bytes", slice.window.len);
            cell.stage_done(slice);
        }
        None => {
            crate::runtime::drain::note_store_route("pass_stage_seed_arena_full");
            cell.give_back();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::pass_stage::SeedStageCell;
    use std::sync::Arc;

    /// Test arenas over leaked host allocations, and a stager over member 0.
    fn stager(chunk: u64, chunks: usize) -> Option<Stager> {
        crate::backend::vulkan::engine::pass_stage::install_test_arenas(1, chunk, chunks);
        Stager::of(
            crate::backend::vulkan::engine::pass_stage::arenas().expect("installed"),
            0,
        )
    }

    /// The staged bytes are the bytes, and every clone of the fact owns a
    /// reference: dropping the clones is not enough to free the window, and
    /// dropping the last one is.
    #[test]
    fn staged_bytes_round_trip_and_ownership() {
        let stager = stager(4096, 1).expect("arenas");
        let pattern: Vec<u8> = (0..255u8).cycle().take(1000).collect();
        let slice = stager.stage_bytes(&pattern).expect("room");
        // SAFETY: the slice is alive for this borrow.
        assert_eq!(unsafe { slice.bytes() }, &pattern[..], "the copy is exact");
        assert_eq!(
            slice.ring.in_flight(),
            1,
            "the fact's own reference, and only it"
        );
        let clone = slice.clone();
        assert_eq!(slice.ring.in_flight(), 2, "a clone owns a reference");
        drop(clone);
        assert_eq!(slice.ring.in_flight(), 1);
        drop(slice);
        let arenas = crate::backend::vulkan::engine::pass_stage::take_test_arenas().expect("set");
        assert_eq!(
            arenas[0].ring.in_flight(),
            0,
            "the last fact out frees the window"
        );
    }

    /// The seed exchange folds into the staging copy: the staged bytes are
    /// what `write_staging_swap_rb` would have written, whole pixels only.
    #[test]
    fn staged_seed_exchanges_rb() {
        let stager = stager(4096, 1).expect("arenas");
        let seed = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let plain = stager.stage_seed(&seed, false).expect("no swap");
        // SAFETY: alive for the borrow.
        assert_eq!(unsafe { plain.bytes() }, &seed);
        let swapped = stager.stage_seed(&seed, true).expect("swap");
        // SAFETY: alive for the borrow.
        assert_eq!(
            unsafe { swapped.bytes() },
            &[30, 20, 10, 40, 70, 60, 50, 80],
            "R and B exchanged per pixel, G and A kept"
        );
        // A ragged tail is not a seed this rail states.
        let ragged = [1u8, 2, 3];
        assert!(stager.stage_seed(&ragged, true).is_none());
    }

    /// `stage_runs` is `write_staging_from_runs` on a worker: the window at
    /// its running offset past `source_offset`, the tail zeroed on a short
    /// read.
    #[test]
    fn stage_runs_windows_and_zeroes_the_tail() {
        let stager = stager(8192, 1).expect("arenas");
        // One 32-byte host buffer as two runs; the window starts 4 bytes in
        // and wants 12, then 20 (past the mapping: the tail zeroes).
        let host: Box<[u8]> = (0u8..32).collect();
        let base = Box::leak(host).as_ptr() as usize;
        let runs = vec![
            crate::backend::vulkan::engine::GuestRun::in_mapping(base, 12, 0, 12).expect("run"),
            crate::backend::vulkan::engine::GuestRun::in_mapping(base, 32, 12, 20).expect("run"),
        ];
        let slice = stager.stage_runs(&runs, 4, 12).expect("room");
        // SAFETY: alive for the borrow.
        assert_eq!(
            unsafe { slice.bytes() },
            &(4u8..16u8).collect::<Vec<_>>()[..]
        );
        // A window the runs underfill: bytes 4..32 cover 28; asking for 34
        // zeroes the last six (the 4-byte arena alignment may pad the ask, so
        // the compare is over the requested prefix).
        let short = stager.stage_runs(&runs, 4, 36).expect("room");
        let mut expect: Vec<u8> = (4u8..32).collect();
        expect.resize(36, 0);
        // SAFETY: alive for the borrow.
        assert_eq!(&unsafe { short.bytes() }[..36], &expect[..]);
    }

    /// The seed cell's three races: the worker wins (the recorder takes the
    /// window), the recorder wins (the serial path runs, a late worker's
    /// window dies unused), and a late worker after a spent cell gives up.
    #[test]
    fn seed_cell_worker_wins() {
        let stager = stager(4096, 2).expect("arenas");
        let bytes = Arc::new(vec![9u8, 8, 7, 6]);
        let cell = SeedStageCell::new(Arc::clone(&bytes), true);
        assert!(cell.is_pending());
        let (taken, swap) = cell.take_pending().expect("the worker takes it");
        assert!(taken.iter().eq(bytes.iter()) && swap);
        assert!(!cell.is_pending(), "no second worker takes it");
        let slice = stager.stage_seed(&taken, swap).expect("room");
        assert_eq!(slice.ring.in_flight(), 1);
        cell.stage_done(slice.clone());
        let served = cell.take_staged().expect("the recorder takes the window");
        // SAFETY: alive for the borrow.
        assert_eq!(unsafe { served.bytes() }, &[7, 8, 9, 6]);
        assert_eq!(served.ring.in_flight(), 2, "cell + recorder holds");
        drop(served);
        drop(cell);
        drop(slice);
        assert_eq!(
            crate::backend::vulkan::engine::pass_stage::arenas().expect("installed")[0]
                .ring
                .in_flight(),
            0,
            "everything released"
        );
    }

    #[test]
    fn seed_cell_recorder_wins_and_the_late_worker_gives_back() {
        let stager = stager(4096, 2).expect("arenas");
        let bytes = Arc::new(vec![1u8, 2, 3, 4]);
        let cell = SeedStageCell::new(Arc::clone(&bytes), false);
        let (taken, _) = cell.take_pending().expect("the worker takes it");
        // The recorder arrives mid-copy and goes serial.
        assert!(cell.take_staged().is_none(), "nothing staged yet");
        let slice = stager.stage_bytes(&taken).expect("room");
        cell.stage_done(slice.clone());
        assert_eq!(
            slice.ring.in_flight(),
            1,
            "the unused window is released by its own drop"
        );
        // And a cell nobody took is spent by a no-seed draw without leaking.
        let stranded = SeedStageCell::new(Arc::clone(&bytes), false);
        assert!(stranded.take_staged().is_none());
    }

    /// The rail refuses without its flag: no arenas are consulted, the seed
    /// job is a no-op, and the staging arms are the serial ones.
    #[test]
    fn the_rail_stays_off_without_the_flag() {
        let _off = crate::runtime::par::staging::test_enable(false);
        assert!(!crate::runtime::par::staging::enabled());
        // An armed-but-disabled seed cell spends nothing.
        let cell = SeedStageCell::new(Arc::new(vec![0u8; 4]), false);
        let job = |_cell: &SeedStageCell| {};
        job(&cell);
        assert!(cell.is_pending(), "nothing touched it");
    }
}
