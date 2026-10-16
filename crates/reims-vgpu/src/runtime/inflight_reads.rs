//! Guest resources the GPU will read from guest RAM when already-recorded work
//! executes, so that a host-side write into them can be ordered after it.
//!
//! Metal orders a blit after every command encoded before it on the queue. This
//! device executes blits on the CPU, straight into guest pages, while it walks
//! the stream - but the draws of earlier passes read their guest-backed sources
//! (`guest_runs` textures, imported buffers) only when the GPU executes them,
//! which can be later. A blit that rewrites such a source therefore reached the
//! earlier passes' draws before they ran: Skia Graphite re-uploading atlas rows
//! between passes during a window resize made glyphs and masks vanish from the
//! tiles of the earlier passes (Claude.app, 2026-10-16, b594-b596).
//!
//! The guest-read debt the completion stamp keeps cannot answer this: the
//! GPU-ordered stamp path takes it the moment the stamp is queued behind the
//! batch, without waiting, so it says "a stamp still has to be ordered behind a
//! read", not "a read is still in flight".
//!
//! So draw preparation names every `(task, reference)` whose bytes a recorded
//! draw will read at execute time, and a host-side writer asks
//! [`take_if_pending`] about its destination before it writes.
//!
//! # When a named read is over
//!
//! A name only says "some draw read this". Whether that draw has executed is
//! answered by two watermarks the engine publishes as submissions retire
//! ([`note_sealed`], [`note_retired`]): the number of the last retired
//! submission, and how many recorder jobs had finished recording when it was
//! sealed. A name is first resolved at a writer's question, on the drain, when
//! every draw prepared before it has been posted to the recorder or recorded:
//! it then needs the submission that will hold the open batch (the drain's own
//! synchronous draws) and the recorder jobs posted so far (the deferred ones).
//! A retired submission covers every earlier one on the queue, so once both
//! watermarks reach a name's need its reads are over and the write goes ahead
//! without waiting. Only a read still in flight costs a wait - for all recorded
//! work, which also forgets every name.
//!
//! Without the watermarks a name lived until the next wait: on Maps nearly every
//! tile upload into a once-sampled texture waited for the whole GPU (b610: Maps
//! 50 -> 20 fps).
//!
//! Keyed by reference, not by pages: a view and the texture it views are two
//! names for one storage, and a write through one is not seen as a hit on the
//! other. The draws that hit the hazard measured so far read and are written
//! under one name.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// What a name's reads need retired before a write may land: the submission
/// number and the recorder jobs finished at its seal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Need {
    seal: u64,
    jobs: u64,
}

/// Named at preparation; resolved at the first writer's question.
const UNRESOLVED: Need = Need {
    seal: u64::MAX,
    jobs: u64::MAX,
};

struct Names {
    map: HashMap<(u32, u32), Need>,
    /// Names still [`UNRESOLVED`], so a question resolves only the new ones.
    unresolved: Vec<(u32, u32)>,
}

static PENDING: Mutex<Option<Names>> = Mutex::new(None);

/// Submissions sealed so far (each one's number is this after its seal).
static SEALS: AtomicU64 = AtomicU64::new(0);
/// The highest retired submission's number, and the recorder jobs finished
/// when it was sealed.
static RETIRED_SEAL: AtomicU64 = AtomicU64::new(0);
static RETIRED_JOBS: AtomicU64 = AtomicU64::new(0);

/// Whether the ordering is on. `REIMS_VGPU_BLIT_READ_ORDER=off` restores the
/// unordered writes (A/B only).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::config::switch(crate::config::BLIT_READ_ORDER) != crate::config::Switch::Off
    })
}

/// A recorded draw will read `(task_id, reference)` from guest RAM when it runs.
pub fn note(task_id: u32, reference: u32) {
    if reference == 0 || !enabled() {
        return;
    }
    let mut g = PENDING.lock().unwrap_or_else(|p| p.into_inner());
    let names = g.get_or_insert_with(|| Names {
        map: HashMap::new(),
        unresolved: Vec::new(),
    });
    if names.map.insert((task_id, reference), UNRESOLVED) != Some(UNRESOLVED) {
        names.unresolved.push((task_id, reference));
    }
}

/// The engine sealed a submission: its number, for [`note_retired`].
pub fn note_sealed() -> u64 {
    SEALS.fetch_add(1, Ordering::AcqRel) + 1
}

/// Submissions sealed so far.
pub fn seals() -> u64 {
    SEALS.load(Ordering::Acquire)
}

/// Submission `seal` retired; `jobs` recorder jobs had finished recording when
/// it was sealed (0 when that cannot be vouched for).
pub fn note_retired(seal: u64, jobs: u64) {
    RETIRED_SEAL.fetch_max(seal, Ordering::AcqRel);
    RETIRED_JOBS.fetch_max(jobs, Ordering::AcqRel);
}

fn retired(need: Need) -> bool {
    RETIRED_SEAL.load(Ordering::Acquire) >= need.seal
        && RETIRED_JOBS.load(Ordering::Acquire) >= need.jobs
}

/// Whether a host-side write into `(task_id, reference)` must first wait for
/// recorded GPU work; on `true` every name is forgotten, because the caller
/// is about to wait for all of it.
///
/// Called on the drain, between records. `horizon` is asked at most once and
/// answers `(submission holding everything recorded so far, recorder jobs
/// posted so far)`, after retiring what has completed - `None` when the engine
/// cannot say, which keeps the wait.
pub fn take_if_pending(
    task_id: u32,
    reference: u32,
    horizon: impl FnOnce() -> Option<(u64, u64)>,
) -> bool {
    if reference == 0 || !enabled() {
        return false;
    }
    let key = (task_id, reference);
    // The names to resolve are the ones noted before the engine is asked: a
    // name noted while it answers belongs to a later draw.
    let to_resolve = {
        let mut g = PENDING.lock().unwrap_or_else(|p| p.into_inner());
        let Some(names) = g.as_mut() else {
            return false;
        };
        match names.map.get(&key) {
            None => return false,
            Some(need) if *need != UNRESOLVED && retired(*need) => {
                names.map.remove(&key);
                return false;
            }
            Some(_) => std::mem::take(&mut names.unresolved),
        }
    };
    // Unresolved, or resolved and not yet seen retired: ask the engine, which
    // also retires the submissions that completed meanwhile.
    let resolved = horizon().map(|(seal, jobs)| Need { seal, jobs });
    let mut g = PENDING.lock().unwrap_or_else(|p| p.into_inner());
    let Some(names) = g.as_mut() else {
        return false;
    };
    match resolved {
        Some(resolved) => {
            for k in to_resolve {
                if let Some(n) = names.map.get_mut(&k) {
                    if *n == UNRESOLVED {
                        *n = resolved;
                    }
                }
            }
        }
        None => names.unresolved.extend(to_resolve),
    }
    let Some(&need) = names.map.get(&key) else {
        return false;
    };
    if need != UNRESOLVED && retired(need) {
        names.map.remove(&key);
        if names.map.len() > 4096 {
            names.map.retain(|_, n| *n == UNRESOLVED || !retired(*n));
        }
        return false;
    }
    names.map.clear();
    names.unresolved.clear();
    true
}

/// Everything recorded has executed (a full wait elsewhere): no name is pending.
pub fn forget_all() {
    let mut g = PENDING.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(names) = g.as_mut() {
        names.map.clear();
        names.unresolved.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test: the watermarks and names are process-wide.
    #[test]
    fn a_write_waits_only_for_a_read_still_in_flight() {
        forget_all();
        let base = seals();
        let j0 = RETIRED_JOBS.load(Ordering::Acquire);
        let never = || -> Option<(u64, u64)> { panic!("no engine question expected") };

        note(7, 11);
        note(7, 12);
        assert!(
            !take_if_pending(7, 13, never),
            "an unread reference never waits"
        );

        // Resolved against the open batch (next seal) and 5 posted jobs;
        // nothing has retired, so the write waits and forgets every name.
        assert!(take_if_pending(7, 11, || Some((base + 1, j0 + 5))));
        assert!(
            !take_if_pending(7, 12, never),
            "the wait covered every recorded read"
        );

        // A read whose submission has retired lets the write through.
        note(8, 11);
        let s = note_sealed();
        assert_eq!(s, base + 1);
        note_retired(s, j0 + 6);
        assert!(!take_if_pending(8, 11, || Some((s, j0 + 6))));
        assert!(!take_if_pending(8, 11, never), "the name was dropped");

        // Retired submissions, but a deferred job not yet recorded: waits.
        note(9, 11);
        assert!(take_if_pending(9, 11, || Some((s, j0 + 7))));

        // No answer from the engine keeps the wait.
        note(9, 12);
        assert!(take_if_pending(9, 12, || None));

        assert!(
            !take_if_pending(8, 11, never),
            "another task's reference is another name"
        );
    }
}
