//! Parallel preparation: a fixed pool of worker threads for work whose result
//! the drain thread validates before it uses it.
//!
//! # What this is for
//!
//! One thread, the drain, executes every guest transaction in the order the
//! ordering core chose, and most of a heavy frame's wall clock is the drain
//! doing work whose *answer* does not depend on being done in that order —
//! resolving which bytes a sampled texture holds, for instance. The design
//! (`lab/journal/2026-09-30-lever1-parallel-design.md`) keeps the commit order
//! and every guest-visible effect exactly where they are and moves only that
//! speculative preparation onto threads that return a result together with what
//! they read to get it. The drain checks that the state read is still the state
//! that holds, applies what the job deferred, and uses the result; when the check
//! fails it does the work itself, the way it always did.
//!
//! This module is the mechanism under that and nothing more: threads, a
//! fork-join, and the rules that make it safe to hand them a `&DeviceState`.
//! What a job computes, what it reads and what it defers is the caller's.
//!
//! # The rules a job lives by
//!
//! 1. **Read-only.** A job holds `&DeviceState` (shared, for the length of the
//!    scope, which is why `DeviceState: Sync` is asserted at its definition) and
//!    a [`crate::qemu::host_ops::WorkerHost`]. It writes nothing the guest can
//!    see, nothing in the device state and nothing in the engine. Anything it
//!    would have mutated it *describes* instead, for the drain to apply.
//! 2. **No lock the caller holds.** The drain waits for the scope with whatever
//!    it holds; a job that waits on one of those locks is a deadlock. In
//!    particular the engine's locks belong to the drain across a scope.
//! 3. **Guest memory through the worker host only.** A worker thread is
//!    registered with QEMU's RCU when it starts (`thread_attach`), which is what
//!    makes `read_gpa` and the dirty-tracker queries safe from it. A shim that
//!    offers no such registration gets no pool at all ([`Pool::new`] is `None`),
//!    and the sysbus shim is one.
//! 4. **A panic is data.** It comes back as [`JobPanic`] in that item's slot,
//!    and the caller treats it like any other reason to do the work serially.
//! 5. **No recorder settle.** While a job runs the thread reports
//!    [`planning()`] as `true` (the coordinator for the scope's length, a
//!    helper for its life), and the engine's drain-side doors skip their wait
//!    for the draw recorder on that flag — waiting for the recorder from
//!    inside a scope is what the scope exists to overlap. A job that *did*
//!    reach engine state would still be caught by the `debug_assert!`s on the
//!    engine's locks; the flag only removes the wait, never the rule.
//!
//! # Lifetime
//!
//! The pool is built cold with the device and owned by its bound device; its
//! threads are spawned by the first scope that wants one, so a device that never
//! forks owns none. `Pool::shutdown` stops and joins them and is called by
//! `device_destroy` — before QEMU unrealizes the device, so no worker is
//! registered with RCU (or holds a host pointer) while the machine is taken
//! apart.
//!
//! # Switches (TEMP, lab)
//!
//! * `REIMS_VGPU_LAB_PAR_WORKERS=N` — helper threads. `0` is the serial pool
//!   (every scope runs on its caller). Unset is [`default_workers`]. The value
//!   chooses how many of the threads the shim allows; it can never create the
//!   allowance.
//! * `REIMS_VGPU_LAB_PAR_SPIN_US=W[,J]` — how long an idle helper (`W`) and a
//!   waiting coordinator (`J`) spin before parking. Defaults 0 and 50.
//! * `REIMS_VGPU_LAB_PAR_SELFTEST=noop|read|gen` — run a fork-join per drain
//!   tranche that does no device work, to measure the pool itself; see
//!   [`selftest`].

pub(crate) mod ahead;
pub(crate) mod gate;
mod pool;
pub(crate) mod reap;
pub(crate) mod selftest;
pub(crate) mod staging;
#[cfg(test)]
mod tests;

pub use gate::{FrameLock, Gate};
pub use pool::{FrameHandle, Hot, JobPanic, Overlap, Pool, PoolStats, Scoped, Ticket, WorkerEnv};
pub(crate) use pool::push_followup;

use std::cell::Cell;
use std::time::Duration;

thread_local! {
    /// Whether this thread is inside a pool scope right now: the coordinator
    /// for the length of [`Pool::scope`] (see [`enter_planning`]), a helper for
    /// its whole life. A job running under this flag must not wait for the
    /// recorder — the recorder is exactly what the scope is meant to overlap —
    /// so the engine's drain-side doors consult [`planning`] before they settle.
    static PLANNING: Cell<bool> = const { Cell::new(false) };
}

/// Whether the calling thread is running pool jobs — as a scope's coordinator
/// or as a helper. See [`PLANNING`].
#[inline]
pub fn planning() -> bool {
    PLANNING.with(Cell::get)
}

/// Mark the calling thread as planning for the lifetime of the returned token,
/// unwinding included. The token restores the previous value rather than
/// clearing, so a scope nested inside another scope's job leaves the outer
/// flag standing.
pub(crate) fn enter_planning() -> PlanningToken {
    let was = PLANNING.with(|flag| flag.replace(true));
    PlanningToken { was }
}

/// Restores the planning flag [`enter_planning`] found. Invariant: the token
/// never outlives its `enter_planning` call's frame.
pub(crate) struct PlanningToken {
    was: bool,
}

impl Drop for PlanningToken {
    fn drop(&mut self) {
        PLANNING.with(|flag| flag.set(self.was));
    }
}

/// How the pool is sized and how it waits. Read once from the environment by
/// [`Settings::from_env`]; built directly by tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    /// Helper threads; `0` runs every scope on its caller.
    pub workers: usize,
    /// How long an idle helper spins on the batch epoch before parking.
    pub worker_spin: Duration,
    /// How long the coordinator spins for the last items before parking.
    pub join_spin: Duration,
}

impl Settings {
    /// Upper bound on helpers, whatever the environment asks for. Past it the
    /// executor's `u16` population and any sensible host stop agreeing.
    pub const MAX_WORKERS: usize = 64;

    /// Coordinator spin when nothing says otherwise: long enough to catch the
    /// tail of a 5-50 µs item without a park and wake round trip, short enough
    /// that a slow item is not busy-waited.
    pub const DEFAULT_JOIN_SPIN_US: u64 = 50;

    /// The settings the environment asks for, parsed once per process.
    pub fn from_env() -> Self {
        static SETTINGS: std::sync::OnceLock<Settings> = std::sync::OnceLock::new();
        *SETTINGS.get_or_init(|| {
            let workers = match std::env::var("REIMS_VGPU_LAB_PAR_WORKERS") {
                Ok(v) => match v.trim().parse::<usize>() {
                    Ok(n) => n.min(Self::MAX_WORKERS),
                    Err(_) => {
                        crate::observe::fail(format!(
                            "par_settings_unparsed var=REIMS_VGPU_LAB_PAR_WORKERS value={v} \
                             (using the default)"
                        ));
                        default_workers()
                    }
                },
                Err(_) => default_workers(),
            };
            let (worker_spin_us, join_spin_us) = std::env::var("REIMS_VGPU_LAB_PAR_SPIN_US")
                .ok()
                .map(|v| {
                    let mut parts = v.split(',').map(|p| p.trim().parse::<u64>().ok());
                    (
                        parts.next().flatten().unwrap_or(0),
                        parts.next().flatten().unwrap_or(Self::DEFAULT_JOIN_SPIN_US),
                    )
                })
                .unwrap_or((0, Self::DEFAULT_JOIN_SPIN_US));
            Settings {
                workers,
                worker_spin: Duration::from_micros(worker_spin_us),
                join_spin: Duration::from_micros(join_spin_us),
            }
        })
    }
}

/// Logical CPUs the pool assumes are already spoken for when it sizes itself:
/// the eight vCPUs the lab boots a guest with, plus the device's own threads
/// that are busy under load (drain, harvest, submit, completion).
///
/// A heuristic and named as one. The pool does not know the guest's `-smp`, and
/// what matters — not stealing time from a vCPU — is a measurement
/// (`lab_present_gaps`), not something this constant knows.
const RESERVED_LOGICAL_CPUS: usize = 12;

/// Helpers the pool starts when nothing says otherwise: the logical CPUs left
/// after [`RESERVED_LOGICAL_CPUS`], at most [`DEFAULT_WORKER_CAP`].
pub fn default_workers() -> usize {
    let logical = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    logical
        .saturating_sub(RESERVED_LOGICAL_CPUS)
        .min(DEFAULT_WORKER_CAP)
}

/// Beyond this the fork-join's wake cost and the vCPUs' claim on the host
/// outweigh what a stream's handful of independent sources can use.
pub const DEFAULT_WORKER_CAP: usize = 6;
