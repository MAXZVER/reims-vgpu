//! TEMP (lab, S5a): a recorder thread that records the engine half of a chained
//! draw while the drain prepares the next one.
//!
//! On by default since the 5K in-boot A/B (busy -17.7 +- 3.6 ms/s, +0.9 fps at 240 Hz);
//! `REIMS_VGPU_LAB_RECORDER=off` is the old path, byte for byte: nothing here is reached then.
//!
//! # What it moves, and what it must not
//!
//! A render stream is one pass, and every record but the last is a resident
//! chain intermediate: after `execute_draw_request` returns, the runtime does
//! nothing with the engine's answer except pick `ResidentChain`. So the engine
//! half of such a draw (~7 us at 1080p, ~17 us at 5K) can run on this thread
//! while the drain spends its ~35 us preparing the next record's binds,
//! sampled sources and request.
//!
//! The engine has to see exactly the sequence it saw before, so the rule is
//! one-sided and lives in the engine's own doors, not at each call site:
//!
//! * **At most one job is in flight** (lab `REIMS_VGPU_LAB_RECORDER_DEPTH=N`:
//!   up to N, run in order), and only the drain submits one.
//! * **Every drain-side entry into engine state first waits for the job.**
//!   [`lock_engine`](super::lock_engine) on the drain thread, the drain's
//!   [`DeviceObjectCaches::with`](super::DeviceObjectCaches), and the few
//!   lock-free readers of engine-published state (`resident_readiness`,
//!   `resident_content_backing`, `guest_*_outstanding`) all call
//!   [`settle`]. A drain operation that touches neither therefore overlaps the
//!   recording, and any that does observes the draw as already recorded — the
//!   sequence the engine and the drain's own reads saw before.
//! * The recorder thread is **not** the drain thread (it never marks itself),
//!   so its own `lock_engine` / `with` never wait on itself.
//!
//! # Lock order
//!
//! The job takes the caches mutex and then the engine mutex — the order
//! `execute_draw_request` has always taken them in. The drain waits for the job
//! **before** it takes either, and by induction holds neither while a job is in
//! flight: it settles in the door that acquires them, and it submits only from
//! a point where it holds neither. So the one wait that could close a cycle
//! (drain holds ENGINE, waits for a job that needs ENGINE) cannot be entered.
//!
//! # What a job may touch
//!
//! `DrawRequest` carries raw host pointers (guest RAM aliases from
//! `HostOps::map_pages`, valid for the VM's lifetime, and the stable host
//! allocations of packed task buffers, owned by `Arc`s inside the request) and
//! an `Arc<GuestRamImport>` (pure data; no QEMU call on drop). Nothing in a
//! request borrows from the drain's stack or from `DeviceState`: it is moved
//! whole. The one pointer that does point into `DeviceState` is the caches'
//! address, and it is valid because the drain holds the device lock for the
//! whole tranche and settles every job before the tranche ends
//! ([`settle_deferred_draws`] at `flush_batched_draws`, at the end of every
//! stream, and in `encode_draw_chain` for any record that does not leave a job
//! behind), so the job cannot outlive the `&DeviceState` it was built from.
//!
//! # Census
//!
//! Everything the engine counts on the way (`draw_phase`, `stage_phase`,
//! store routes) is global atomics and a registry with per-thread caches, so
//! it lands in the per-second lines from this thread unchanged. This module
//! adds the `recorder` line: jobs deferred, the recorder's busy time, and what
//! the drain waited, split by the door it waited in.

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::types::DrawError;

/// Where the drain waited for the job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JoinSite {
    /// `lock_engine()` on the drain thread.
    Lock,
    /// The drain's `DeviceObjectCaches::with`.
    Caches,
    /// A lock-free reader of engine-published state.
    Reader,
    /// The explicit join before the next draw's engine call, and the stream /
    /// tranche / chain-exit joins.
    Explicit,
}

impl JoinSite {
    const ALL: [Self; 4] = [Self::Lock, Self::Caches, Self::Reader, Self::Explicit];

    fn index(self) -> usize {
        match self {
            Self::Lock => 0,
            Self::Caches => 1,
            Self::Reader => 2,
            Self::Explicit => 3,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Lock => "lock",
            Self::Caches => "caches",
            Self::Reader => "reader",
            Self::Explicit => "explicit",
        }
    }
}

/// What a job reports. The draw's own output is dropped on the recorder: a
/// deferred draw is one whose caller never reads it.
pub(crate) enum JobResult {
    Done,
    /// The engine refused or failed the draw. Carried to the next explicit
    /// join, which reports it.
    Failed(Box<DeferredFailure>),
    /// The job panicked. Re-raised on the drain thread at the join that takes
    /// it, which is where the sequential path would have panicked.
    Panicked(Box<dyn Any + Send>),
}

/// What an explicit join needs to name a deferred draw's failure after the
/// request that made it is gone.
#[derive(Debug)]
pub struct DeferredFailure {
    pub error: DrawError,
    pub meta: DeferredMeta,
}

/// The few request fields a failure report names.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeferredMeta {
    pub pipeline_ref: u32,
    pub task_id: u32,
    pub width: u32,
    pub height: u32,
    pub vertex_count: u32,
}

type Job = Box<dyn FnOnce() -> JobResult + Send + 'static>;

/// Keep a finished job's result for the taker. An earlier failure nobody has
/// taken yet outranks a later result: the first loss is the one that must be
/// reported.
fn stash_result(inner: &mut Inner, flag: &AtomicBool, result: JobResult) {
    let keep_older = matches!(
        inner.stash,
        Some(JobResult::Failed(_)) | Some(JobResult::Panicked(_))
    );
    if !keep_older {
        inner.stash = Some(result);
    }
    flag.store(inner.stash.is_some(), Ordering::Release);
}

/// Asserts that a value may cross to the recorder thread. See the module doc for
/// why each raw pointer in a `DrawRequest` stays valid until the job completes.
pub(crate) struct AssertSend<T>(pub T);

// SAFETY: the module doc's argument — host pointers are RAMBlock aliases valid
// for the VM's lifetime or are kept alive by `Arc`s moved with the value.
unsafe impl<T> Send for AssertSend<T> {}

impl<T> AssertSend<T> {
    /// A method rather than a field access so a `move` closure captures the
    /// wrapper whole (edition 2021 would otherwise capture the field).
    pub(crate) fn into_inner(self) -> T {
        self.0
    }
}

struct Inner {
    /// Posted and not yet taken by the worker, oldest first.
    jobs: std::collections::VecDeque<Job>,
    /// Posted and not yet finished (queued or running).
    outstanding: u32,
    /// A result the drain already waited for (an implicit join) and nobody has
    /// taken yet.
    stash: Option<JobResult>,
    worker_parked: bool,
    /// Set by [`Handoff::prewake`] to bring a parked worker out of its park
    /// without a job: it goes back to spinning for the one that is coming.
    kick: bool,
    /// Threads inside `settle_slow` waiting for a result. At most one at a time
    /// (`settle_lock`), but counted and woken with `notify_all` anyway so the
    /// protocol does not rest on that.
    waiters: u32,
    shutdown: bool,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// One-slot handoff to one worker thread.
///
/// Three locks, taken in this order and never the other way: `lifecycle`
/// (submit and stop exclude each other, so a submit never posts a job to a
/// worker that stop is retiring, and a restart cannot race stop's bookkeeping),
/// then `settle_lock` (one settler at a time: the drain and a teardown thread
/// can both ask, and the second finds nothing in flight once the first is
/// done), then `inner` (the slot, taken for a few instructions at a time and
/// by the worker alone otherwise).
pub(crate) struct Handoff {
    name: &'static str,
    lifecycle: Mutex<()>,
    settle_lock: Mutex<()>,
    inner: Mutex<Inner>,
    work_cv: Condvar,
    done_cv: Condvar,
    /// A finished job's result is in the stash and nobody has taken it.
    stash_pending: AtomicBool,
    /// A job is posted and the worker has not taken it yet (spin target).
    pending: AtomicBool,
    /// Some posted job has not finished (spin target of a settle). Set by the
    /// submitter and cleared by the worker when the last one finishes, both
    /// under `inner`; the worker's job writes happen before its clear.
    in_flight: AtomicBool,
    /// `Inner::outstanding`, mirrored for a submitter that spins for a free
    /// slot instead of parking on `done_cv` (written under `inner` alongside it).
    out_n: AtomicU32,
    alive: AtomicBool,
    /// TEMP (lab, STAMP_SEQ): jobs posted / finished since the start, so a
    /// waiter can ask for "everything posted before *this* point" instead of
    /// for an idle worker.
    posted: AtomicU64,
    finished: AtomicU64,
    stats: Stats,
}

struct Stats {
    submitted: AtomicU64,
    busy_ns: AtomicU64,
    parked_wakes: AtomicU64,
    prewakes: AtomicU64,
    join_n: [AtomicU64; 4],
    join_blocked: [AtomicU64; 4],
    join_ns: [AtomicU64; 4],
    join_max_ns: [AtomicU64; 4],
    deferred_err: AtomicU64,
    submit_declined: AtomicU64,
    /// Where a blocked join came from: the call site of the engine door, and
    /// what it waited. Touched only when a job is in flight.
    by_caller: Mutex<Vec<((&'static str, u32), (u64, u64))>>,
}

#[allow(clippy::declare_interior_mutable_const)]
const Z: AtomicU64 = AtomicU64::new(0);

impl Stats {
    const fn new() -> Self {
        Self {
            submitted: Z,
            busy_ns: Z,
            parked_wakes: Z,
            prewakes: Z,
            join_n: [Z; 4],
            join_blocked: [Z; 4],
            join_ns: [Z; 4],
            join_max_ns: [Z; 4],
            deferred_err: Z,
            submit_declined: Z,
            by_caller: Mutex::new(Vec::new()),
        }
    }
}

/// Spin before parking, on either side. Draws arrive 30-80 us apart inside a
/// stream and a park/wake round trip costs about as much as one, so the
/// recorder stays hot across a stream and sleeps between streams.
const JOIN_SPIN: Duration = Duration::from_micros(30);

/// How long the worker spins for the next job before it parks.
/// `REIMS_VGPU_LAB_RECORDER_SPIN_US` overrides (default 200); read once.
fn worker_spin() -> Duration {
    static SPIN: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *SPIN.get_or_init(|| {
        Duration::from_micros(
            std::env::var("REIMS_VGPU_LAB_RECORDER_SPIN_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(200),
        )
    })
}

/// TEMP (lab, BELOW_SPIN): how long a submit that finds the queue full spins
/// for a free slot before it parks. Live `BELOW_SPIN=N` (microseconds) or
/// `REIMS_VGPU_LAB_BELOW_SPIN_US`; default 50, `0` restores the bare park and
/// a settle's fixed 30 us spin (it otherwise spins 8 us a job out).
fn below_spin() -> Duration {
    #[cfg(not(test))]
    if let Some(n) =
        crate::runtime::lab_live::value("BELOW_SPIN").and_then(|v| v.trim().parse::<u64>().ok())
    {
        return Duration::from_micros(n.min(1_000));
    }
    static SPIN: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *SPIN.get_or_init(|| {
        Duration::from_micros(
            std::env::var("REIMS_VGPU_LAB_BELOW_SPIN_US")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(50)
                .min(1_000),
        )
    })
}

/// Lab `REIMS_VGPU_LAB_RECORDER_DEPTH=N` (1..=16, default 4): how many deferred
/// draws may be out at once. Above 1 the drain's explicit join before the next
/// draw takes only a failure already found and does not wait; a failure is then
/// reported up to N-1 draws later, which those draws' preparation has run past
/// (observations, caches and debt payment, correct on their own). Read once.
pub(crate) fn depth() -> u32 {
    // Live (`RECORDER_DEPTH=N` in the lab live file): `submit` reads it per job;
    // a smaller value than the jobs already out only makes the next submit wait
    // below it, as a full queue always did.
    #[cfg(not(test))]
    if let Some(n) = crate::runtime::lab_live::value("RECORDER_DEPTH")
        .and_then(|v| v.trim().parse::<u32>().ok())
    {
        return n.clamp(1, 512);
    }
    // TEMP (lab, DRAW_PAR): a chunk's commits arrive at ~1 us a record against
    // the recorder's ~5.5; a queue of 8 made the drain wait on it (b120: Weather
    // 11.4 -> 10.8 us a draw at 32).
    #[cfg(not(test))]
    if crate::runtime::draw::vulkan::draw_par::draw_par_on() {
        // 64 since b126 (light Weather scene, 65 -> 69 fps against 32); 256
        // since b130: with the overlapped commit the drain fills 64 in a
        // stream and waits on it (wait_below 8.9 % of the drain) while the
        // recorder idles 55 % of the time between streams.
        // TEMP (lab, 10-21): `REIMS_VGPU_LAB_RECORDER_DEPTH` overrides it here too (up to 4096) - b941 (21c, 5K
        // Maps, ROUTES off): the drain still waits on a full 256-job queue (wait_below 12.6 % of its samples)
        // while the recorder idles between the bursts.
        static PAR_DEPTH: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        return *PAR_DEPTH.get_or_init(|| {
            std::env::var("REIMS_VGPU_LAB_RECORDER_DEPTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256u32)
                .clamp(1, 4096)
        });
    }
    static DEPTH: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_RECORDER_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
            // 4 since the 5K drag A/B (fq*, interleaved x2): busy 849/840 ->
            // 822/825 ms/s, 14.2 -> 13.4 us a draw, explicit join 80 -> 50 ms/s.
            // 8 since b086 (2026-10-09, the working launcher's ROUTES=off mode,
            // after STAMP_DEFER and the timers going off made the drain wait
            // on a full queue 8.7 % of its busy samples): drain us a draw
            // Weather 10.67 -> 10.53, reveal-desktop 17.1 -> 16.6, Finder
            // 20.95 -> 20.25, Safari even; fps in noise. n219/p042 (8 and 16
            // without signal, 16 worse) predate both changes.
            .unwrap_or(8u32)
            .clamp(1, 64)
    })
}

impl Handoff {
    pub(crate) const fn new(name: &'static str) -> Self {
        Self {
            name,
            lifecycle: Mutex::new(()),
            settle_lock: Mutex::new(()),
            inner: Mutex::new(Inner {
                jobs: std::collections::VecDeque::new(),
                outstanding: 0,
                stash: None,
                worker_parked: false,
                kick: false,
                waiters: 0,
                shutdown: false,
                thread: None,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
            stash_pending: AtomicBool::new(false),
            pending: AtomicBool::new(false),
            in_flight: AtomicBool::new(false),
            out_n: AtomicU32::new(0),
            alive: AtomicBool::new(false),
            posted: AtomicU64::new(0),
            finished: AtomicU64::new(0),
            stats: Stats::new(),
        }
    }

    /// Whether a job is submitted and not yet waited for.
    #[inline]
    pub(crate) fn in_flight(&self) -> bool {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Whether a job is in flight or its result has not been taken: the state
    /// in which the drain still owes the recorder a join, whatever the gate says
    /// now (it may have been turned off while a draw was out).
    #[inline]
    pub(crate) fn pending(&self) -> bool {
        self.in_flight.load(Ordering::Acquire) || self.stash_pending.load(Ordering::Acquire)
    }

    /// TEMP (lab, DRAW_PAR overlap): whether the stash holds a failure (or a
    /// panic) nobody has taken, without taking it. A shared-state commit
    /// cannot report one (the report abandons the chain, which needs the
    /// drain's exclusive state), so it asks first and hands the record to the
    /// exclusive commit, whose take finds it exactly as before.
    pub(crate) fn failure_stashed(&self) -> bool {
        self.stash_pending.load(Ordering::Acquire)
            && matches!(
                self.inner.lock().stash,
                Some(JobResult::Failed(_)) | Some(JobResult::Panicked(_))
            )
    }

    /// TEMP (lab, PASS_PREP): jobs posted and not yet finished (queued or
    /// running). The census reads this at a planning scope's entry: a positive
    /// reading is the overlap the scope exists for — the recorder had draws to
    /// encode while the pool planned the next chunk.
    pub(crate) fn outstanding(&self) -> u32 {
        self.inner.lock().outstanding
    }

    /// TEMP (lab, STAMP_SEQ): jobs posted so far (the sequence number the next
    /// one finishes at is this plus one).
    pub(crate) fn posted_seq(&self) -> u64 {
        self.posted.load(Ordering::Acquire)
    }

    /// TEMP (lab, STAMP_SEQ): jobs finished so far, in posting order (one
    /// worker, FIFO), so `finished_seq() >= n` means the first `n` are done.
    pub(crate) fn finished_seq(&self) -> u64 {
        self.finished.load(Ordering::Acquire)
    }

    /// Called with `lifecycle` held, so no stop is retiring the worker.
    fn ensure_worker(&'static self) -> bool {
        if self.alive.load(Ordering::Acquire) {
            return true;
        }
        let mut g = self.inner.lock();
        g.shutdown = false;
        let priority = current_thread_priority();
        // A generous stack: `execute_draw_inner` is one very large frame and
        // this thread inherits no size from the drain's.
        let spawned = std::thread::Builder::new()
            .name(self.name.to_string())
            .stack_size(8 << 20)
            .spawn(move || self.worker_loop(priority));
        match spawned {
            Ok(handle) => {
                g.thread = Some(handle);
                self.alive.store(true, Ordering::Release);
                true
            }
            Err(_) => false,
        }
    }

    fn worker_loop(&'static self, priority: Option<i32>) {
        apply_thread_priority(priority);
        #[cfg(windows)]
        crate::lab_cores::pin_to_performance_cores(self.name);
        #[cfg(windows)]
        if std::ptr::eq(self, &DRAW) {
            crate::lab_cores::pin_recorder_excl_core(self.name);
        }
        crate::lab_cores::device_thread_priority(self.name);
        if std::ptr::eq(self, &DRAW) {
            mark_recorder_thread();
            #[cfg(windows)]
            crate::qemu::lab_sampler::start_recorder_once(); // TEMP (lab)
        }
        loop {
            let spin_until = Instant::now() + worker_spin();
            let mut i = 0u32;
            while !self.pending.load(Ordering::Acquire) {
                std::hint::spin_loop();
                i = i.wrapping_add(1);
                if i & 63 == 0 && Instant::now() >= spin_until {
                    break;
                }
            }
            let mut g = self.inner.lock();
            while g.jobs.is_empty() && !g.shutdown && !g.kick {
                g.worker_parked = true;
                self.work_cv.wait(&mut g);
            }
            g.worker_parked = false;
            if g.kick {
                // Woken ahead of a job (a stream is starting): spin for it
                // instead of sleeping on it.
                g.kick = false;
                if g.jobs.is_empty() && !g.shutdown {
                    continue;
                }
            }
            // Retiring: `stop` clears `alive` after it has joined this thread,
            // under `lifecycle`, so there is nothing for the exit to publish.
            // It settled first, so nothing is queued.
            let Some(job) = g.jobs.pop_front() else {
                return;
            };
            self.pending.store(!g.jobs.is_empty(), Ordering::Release);
            drop(g);
            let started = Instant::now();
            let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
                Ok(result) => result,
                Err(payload) => JobResult::Panicked(payload),
            };
            self.stats
                .busy_ns
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let mut g = self.inner.lock();
            stash_result(&mut g, &self.stash_pending, result);
            self.finished.fetch_add(1, Ordering::AcqRel);
            g.outstanding -= 1;
            self.out_n.store(g.outstanding, Ordering::Release);
            if g.outstanding == 0 {
                self.in_flight.store(false, Ordering::Release);
            }
            if g.waiters != 0 {
                self.done_cv.notify_all();
            }
        }
    }

    /// Post a job. The submitter must have settled any earlier one. When the
    /// worker cannot be started the job runs here, on the submitting thread,
    /// before this returns: the draw is recorded either way and the result waits
    /// in the stash like any other.
    pub(crate) fn submit(&'static self, job: Job) {
        let _lifecycle = self.lifecycle.lock();
        let depth = depth();
        if depth <= 1 && self.in_flight() {
            // Defensive: wait for it rather than overwrite it.
            self.settle(JoinSite::Explicit);
        } else if depth > 1 && self.out_n.load(Ordering::Acquire) >= depth {
            self.wait_below(depth);
        }
        if !self.ensure_worker() {
            self.stats.submit_declined.fetch_add(1, Ordering::Relaxed);
            let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
                Ok(result) => result,
                Err(payload) => JobResult::Panicked(payload),
            };
            stash_result(&mut self.inner.lock(), &self.stash_pending, result);
            self.posted.fetch_add(1, Ordering::AcqRel);
            self.finished.fetch_add(1, Ordering::AcqRel);
            return;
        }
        let mut g = self.inner.lock();
        self.posted.fetch_add(1, Ordering::AcqRel);
        g.jobs.push_back(job);
        g.outstanding += 1;
        self.out_n.store(g.outstanding, Ordering::Release);
        self.in_flight.store(true, Ordering::Release);
        self.pending.store(true, Ordering::Release);
        if g.worker_parked {
            self.stats.parked_wakes.fetch_add(1, Ordering::Relaxed);
            self.work_cv.notify_one();
        }
        drop(g);
        self.stats.submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Bring a parked worker awake ahead of a job. The first draw of a stream finds
    /// the worker asleep (streams arrive ~2 ms apart and the worker spins for
    /// 200 us), and a wake costs 20-50 us, which the *next* record's first engine
    /// door then waits out. A stream knows it will draw before its first record
    /// is prepared, so the wake is paid in parallel with that preparation.
    pub(crate) fn prewake(&'static self) {
        let _lifecycle = self.lifecycle.lock();
        if !self.ensure_worker() {
            return;
        }
        let mut g = self.inner.lock();
        if g.worker_parked {
            g.kick = true;
            self.stats.prewakes.fetch_add(1, Ordering::Relaxed);
            self.work_cv.notify_one();
        }
    }

    /// Wait for the in-flight job, if any, and keep its result for
    /// [`Self::take`]. A no-op (one atomic load) when nothing is in flight.
    #[inline]
    #[track_caller]
    pub(crate) fn settle(&self, site: JoinSite) {
        if !self.in_flight.load(Ordering::Acquire) {
            return;
        }
        self.settle_slow(site);
    }

    /// TEMP (lab, RETAIN_TARGETED): wait until `done` holds - a condition the
    /// worker's job completions make true (a resident's last writer finished) -
    /// rather than for every job out. Spins first (a job is ~5 us), then parks
    /// on the per-job completion signal; counted at `site` like a settle.
    #[cold]
    #[track_caller]
    pub(crate) fn settle_until(&self, site: JoinSite, mut done: impl FnMut() -> bool) {
        let started = Instant::now();
        let spin_until = started + JOIN_SPIN.max(Duration::from_micros(50)).min(Duration::from_micros(200));
        let mut i = 0u32;
        while !done() {
            if !self.in_flight.load(Ordering::Acquire) {
                break;
            }
            std::hint::spin_loop();
            i = i.wrapping_add(1);
            if i & 7 == 0 && Instant::now() >= spin_until {
                let mut g = self.inner.lock();
                while !done() && g.outstanding != 0 {
                    g.waiters += 1;
                    self.done_cv.wait_for(&mut g, Duration::from_micros(500));
                    g.waiters -= 1;
                }
                break;
            }
        }
        let ns = started.elapsed().as_nanos() as u64;
        let s = site.index();
        self.stats.join_n[s].fetch_add(1, Ordering::Relaxed);
        if ns > 1_000 {
            self.stats.join_blocked[s].fetch_add(1, Ordering::Relaxed);
        }
        self.stats.join_ns[s].fetch_add(ns, Ordering::Relaxed);
        self.stats.join_max_ns[s].fetch_max(ns, Ordering::Relaxed);
    }

    #[cold]
    #[track_caller]
    fn settle_slow(&self, site: JoinSite) {
        let caller = std::panic::Location::caller();
        let started = Instant::now();
        // One settler at a time. A second (a teardown thread asking while the
        // drain waits) queues here and, once the first has the result stashed,
        // finds nothing in flight and returns.
        let _settler = self.settle_lock.lock();
        if !self.in_flight.load(Ordering::Acquire) {
            return;
        }
        // TEMP (lab, BELOW_SPIN): a settle waits for every job out, up to the
        // queue's depth of them, so its spin scales with how many there are.
        let spin = if below_spin().is_zero() {
            JOIN_SPIN
        } else {
            JOIN_SPIN.max(Duration::from_micros(8 * u64::from(self.out_n.load(Ordering::Acquire))))
        };
        let spin_until = started + spin.min(Duration::from_micros(200));
        let mut i = 0u32;
        let mut blocked = false;
        while self.in_flight.load(Ordering::Acquire) {
            std::hint::spin_loop();
            i = i.wrapping_add(1);
            if i & 31 == 0 && Instant::now() >= spin_until {
                break;
            }
        }
        let mut g = self.inner.lock();
        while g.outstanding != 0 {
            blocked = true;
            g.waiters += 1;
            self.done_cv.wait(&mut g);
            g.waiters -= 1;
        }
        drop(g);
        let ns = started.elapsed().as_nanos() as u64;
        let s = site.index();
        self.stats.join_n[s].fetch_add(1, Ordering::Relaxed);
        if blocked || ns > 1_000 {
            self.stats.join_blocked[s].fetch_add(1, Ordering::Relaxed);
        }
        self.stats.join_ns[s].fetch_add(ns, Ordering::Relaxed);
        self.stats.join_max_ns[s].fetch_max(ns, Ordering::Relaxed);
        let key = (caller.file(), caller.line());
        let mut by = self.stats.by_caller.lock();
        match by.iter_mut().find(|(k, _)| *k == key) {
            Some((_, (n, total))) => {
                *n += 1;
                *total += ns;
            }
            None => by.push((key, (1, ns))),
        }
    }

    /// The finished job's result, waiting for it if it is still running, or
    /// `None` when no job has finished since the last take.
    #[track_caller]
    pub(crate) fn take(&self, site: JoinSite) -> Option<JobResult> {
        self.settle(site);
        let mut g = self.inner.lock();
        let taken = g.stash.take();
        self.stash_pending.store(false, Ordering::Release);
        taken
    }

    /// Lab `RECORDER_DEPTH` > 1: what finished jobs have stashed, without
    /// waiting for the ones still out. A failure among them is the first one.
    pub(crate) fn take_nowait(&self) -> Option<JobResult> {
        if !self.stash_pending.load(Ordering::Acquire) {
            return None;
        }
        let mut g = self.inner.lock();
        let taken = g.stash.take();
        self.stash_pending.store(false, Ordering::Release);
        taken
    }

    /// Wait until fewer than `depth` jobs are out (a full queue at submit),
    /// counted as an explicit join.
    #[cold]
    #[track_caller]
    fn wait_below(&self, depth: u32) {
        let started = Instant::now();
        // TEMP (lab): the depth this wait was asked for and the queue it found -
        // the recorder idles ~80 % of full-screen Maps (b270) while the drain
        // waits here 10 % of its busy time, which only a small depth explains.
        LAB_BELOW_DEPTH.store(depth, Ordering::Relaxed);
        LAB_BELOW_OUT_MAX.fetch_max(self.out_n.load(Ordering::Acquire), Ordering::Relaxed);
        LAB_BELOW_N.fetch_add(1, Ordering::Relaxed);
        // A slot frees within one job (~5 us a draw), while a park on
        // `done_cv` costs a Windows wake (20-50 us) and makes the worker pay a
        // notify per job: spin for it first, as a settle does.
        let spin_until = started + below_spin();
        let mut i = 0u32;
        while self.out_n.load(Ordering::Acquire) >= depth {
            std::hint::spin_loop();
            i = i.wrapping_add(1);
            if i & 31 == 0 && Instant::now() >= spin_until {
                break;
            }
        }
        let mut g = self.inner.lock();
        while g.outstanding >= depth {
            g.waiters += 1;
            self.done_cv.wait(&mut g);
            g.waiters -= 1;
        }
        drop(g);
        let ns = started.elapsed().as_nanos() as u64;
        let s = JoinSite::Explicit.index();
        self.stats.join_n[s].fetch_add(1, Ordering::Relaxed);
        self.stats.join_blocked[s].fetch_add(1, Ordering::Relaxed);
        self.stats.join_ns[s].fetch_add(ns, Ordering::Relaxed);
        self.stats.join_max_ns[s].fetch_max(ns, Ordering::Relaxed);
    }

    pub(crate) fn note_deferred_error(&self) {
        self.stats.deferred_err.fetch_add(1, Ordering::Relaxed);
    }

    /// Settle, then stop the worker and wait for it to exit, and hand back the
    /// result nobody had taken (the caller reports a failure: it belongs to the
    /// device that is ending, and must not be found by the next one). A later
    /// `submit` starts a new worker.
    pub(crate) fn stop(&'static self) -> Option<JobResult> {
        // Excludes `submit`: no job can be posted to the worker being retired,
        // and the restart cannot interleave with the bookkeeping below.
        let _lifecycle = self.lifecycle.lock();
        self.settle(JoinSite::Explicit);
        let handle = {
            let mut g = self.inner.lock();
            g.shutdown = true;
            self.work_cv.notify_all();
            g.thread.take()
        };
        if let Some(handle) = handle {
            let _ = handle.join();
        }
        self.alive.store(false, Ordering::Release);
        let mut g = self.inner.lock();
        g.shutdown = false;
        let taken = g.stash.take();
        self.stash_pending.store(false, Ordering::Release);
        taken
    }

    /// Drain the window into one line, or `None` when nothing happened in it.
    pub(crate) fn take_window(&self, win_ms: u64) -> Option<String> {
        let s = &self.stats;
        let submitted = s.submitted.swap(0, Ordering::Relaxed);
        let busy = s.busy_ns.swap(0, Ordering::Relaxed);
        let wakes = s.parked_wakes.swap(0, Ordering::Relaxed);
        let prewakes = s.prewakes.swap(0, Ordering::Relaxed);
        let err = s.deferred_err.swap(0, Ordering::Relaxed);
        let declined = s.submit_declined.swap(0, Ordering::Relaxed);
        let mut body = String::new();
        let mut any = submitted != 0 || err != 0 || declined != 0;
        for site in JoinSite::ALL {
            let i = site.index();
            let n = s.join_n[i].swap(0, Ordering::Relaxed);
            let blocked = s.join_blocked[i].swap(0, Ordering::Relaxed);
            let ns = s.join_ns[i].swap(0, Ordering::Relaxed);
            let max = s.join_max_ns[i].swap(0, Ordering::Relaxed);
            any |= n != 0;
            let label = site.label();
            body.push_str(&format!(
                " join_{label}_n={n} join_{label}_blocked={blocked} join_{label}_us={} \
                 join_{label}_max_us={}",
                ns / 1000,
                max / 1000
            ));
        }
        let mut by = std::mem::take(&mut *s.by_caller.lock());
        by.sort_by_key(|(_, (_, ns))| std::cmp::Reverse(*ns));
        let top: Vec<String> = by
            .iter()
            .take(6)
            .map(|((file, line), (n, ns))| {
                let name = file.rsplit(['/', '\\']).next().unwrap_or(file);
                format!("{name}:{line}={n}/{}us", ns / 1000)
            })
            .collect();
        any.then(|| {
            format!(
                "recorder win_ms={win_ms} deferred={submitted} busy_us={} wakes={wakes} \
                 prewakes={prewakes} deferred_err={err} submit_declined={declined}{body} \
                 lab_below_n={} lab_below_depth={} lab_below_out_max={}{} top={}",
                busy / 1000,
                LAB_BELOW_N.swap(0, Ordering::Relaxed),
                LAB_BELOW_DEPTH.load(Ordering::Relaxed),
                LAB_BELOW_OUT_MAX.swap(0, Ordering::Relaxed),
                super::cmd_sink::take_window_text(),
                if top.is_empty() {
                    "-".to_string()
                } else {
                    top.join(",")
                }
            )
        })
    }
}

#[cfg(windows)]
pub(super) fn current_thread_priority() -> Option<i32> {
    extern "system" {
        fn GetCurrentThread() -> *mut core::ffi::c_void;
        fn GetThreadPriority(h: *mut core::ffi::c_void) -> i32;
    }
    // SAFETY: the pseudo handle of the calling thread.
    let p = unsafe { GetThreadPriority(GetCurrentThread()) };
    // THREAD_PRIORITY_ERROR_RETURN
    (p != i32::MAX).then_some(p)
}

#[cfg(windows)]
pub(super) fn apply_thread_priority(priority: Option<i32>) {
    extern "system" {
        fn GetCurrentThread() -> *mut core::ffi::c_void;
        fn SetThreadPriority(h: *mut core::ffi::c_void, p: i32) -> i32;
    }
    if let Some(p) = priority {
        // SAFETY: the pseudo handle of the calling thread.
        unsafe {
            SetThreadPriority(GetCurrentThread(), p);
        }
    }
}

#[cfg(not(windows))]
pub(super) fn current_thread_priority() -> Option<i32> {
    None
}

#[cfg(not(windows))]
pub(super) fn apply_thread_priority(_priority: Option<i32>) {}

/// The process's recorder: draws only.
pub(crate) static DRAW: Handoff = Handoff::new("reims-vgpu-recorder");

// TEMP (lab): see `wait_below`.
static LAB_BELOW_DEPTH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static LAB_BELOW_OUT_MAX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static LAB_BELOW_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// TEMP (lab, live `RETAIN_FAST`): the resident identities the deferred draws
/// still out will register or write - each job's colour targets and depth -
/// counted, so a drain-side question about some *other* resident need not wait
/// for them (see [`settle_unless_pending`]).
static PENDING_TARGETS: Mutex<Vec<(super::types::TargetIdentity, u32)>> = Mutex::new(Vec::new());

/// TEMP (lab, EPOCH_PREDICT): per identity with a job out, the content epoch
/// the *last* job queued for it stamps - `None` when the last one is a draw
/// (which clears the stamp when it readies the resident). Kept beside
/// `PENDING_TARGETS` and under the same holds.
static LAST_STAMP: Mutex<Vec<(super::types::TargetIdentity, Option<u32>)>> = Mutex::new(Vec::new());

fn note_last(targets: &[super::types::TargetIdentity], stamp: Option<u32>) {
    let mut last = LAST_STAMP.lock();
    for t in targets {
        match last.iter_mut().find(|(id, _)| id == t) {
            Some((_, e)) => *e = stamp,
            None => last.push((t.clone(), stamp)),
        }
    }
}

/// TEMP (lab, EPOCH_PREDICT): `Some(epoch)` when jobs are out for `identity`
/// and the last of them is a content-epoch stamp: once the queue reaches it the
/// registry answers `epoch` (the draws before it ready the resident and clear
/// the stamp; the stamp then writes it), so a drain question about the epoch
/// need not wait for the queue. A failed draw before it is reported as a
/// deferred failure and abandons its chain, as it would after the wait.
/// TEMP (lab, RETAIN_DEFER): whether a job still out writes `identity`.
pub(crate) fn pending_writer(identity: &super::types::TargetIdentity) -> bool {
    PENDING_TARGETS.lock().iter().any(|(id, _)| id == identity)
}

/// TEMP (lab, REAP, on unless `REIMS_VGPU_LAB_REAP=off`): drop `value` on a
/// background thread instead of the caller's. A deferred draw's request owns
/// its stage-in bytes (`Arc<Vec<u8>>`, ~125 KB a Maps tile); freeing them at the
/// end of the job cost the recorder 10.5 % of its samples on 5K Maps (b937,
/// `drop_glue` -> `Arc::drop_slow` -> `RtlFreeHeap` / `ZwFreeVirtualMemory`), and
/// the recorder is what the drain waits on. Nothing a request owns needs its
/// drop ordered against later work: its buffers are plain memory, a lease or a
/// staged window it holds only delays its slot's reclaim, and the targets a job
/// holds pending are released by the job itself, not by the request.
pub(crate) fn reap<T: Send + 'static>(value: T) {
    use std::sync::mpsc::{channel, Sender};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static TX: parking_lot::Mutex<Option<Sender<Box<dyn Send>>>> = parking_lot::Mutex::new(None);
    if !*ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_REAP").as_deref() != Ok("off")) {
        return;
    }
    let mut tx = TX.lock();
    if tx.is_none() {
        let (sender, receiver) = channel::<Box<dyn Send>>();
        let spawned = std::thread::Builder::new()
            .name("vgpu-reaper".into())
            .spawn(move || {
                for garbage in receiver {
                    drop(garbage);
                }
            });
        if spawned.is_err() {
            return;
        }
        *tx = Some(sender);
    }
    if let Some(sender) = tx.as_ref() {
        // A send fails only once the reaper is gone; the value then drops here.
        let _ = sender.send(Box::new(value));
    }
}

/// TEMP (lab, RETAIN_DEFER): run `f` on the recorder after every job already
/// queued (a resident's writers included), without holding any target.
pub(crate) fn submit_after_queued(f: impl FnOnce() + Send + 'static) {
    DRAW.submit(Box::new(move || {
        f();
        JobResult::Done
    }));
}

pub(crate) fn pending_last_stamp(identity: &super::types::TargetIdentity) -> Option<u32> {
    if !PENDING_TARGETS.lock().iter().any(|(id, _)| id == identity) {
        return None;
    }
    LAST_STAMP
        .lock()
        .iter()
        .find(|(id, _)| id == identity)
        .and_then(|(_, e)| *e)
}

/// The targets one deferred draw writes, held from its submit to the end of its
/// job (run, failed or panicked - the drop is the release).
pub(crate) struct PendingTargets(Vec<super::types::TargetIdentity>);

impl PendingTargets {
    pub(crate) fn hold(targets: Vec<super::types::TargetIdentity>) -> Self {
        Self::hold_as(targets, None)
    }

    /// TEMP (lab, EPOCH_PREDICT): the hold of a job that stamps `epoch`.
    pub(crate) fn hold_stamp(target: super::types::TargetIdentity, epoch: u32) -> Self {
        Self::hold_as(vec![target], Some(epoch))
    }

    fn hold_as(targets: Vec<super::types::TargetIdentity>, stamp: Option<u32>) -> Self {
        if !targets.is_empty() {
            note_last(&targets, stamp);
            let mut held = PENDING_TARGETS.lock();
            for t in &targets {
                match held.iter_mut().find(|(id, _)| id == t) {
                    Some((_, n)) => *n += 1,
                    None => held.push((t.clone(), 1)),
                }
            }
        }
        Self(targets)
    }
}

impl Drop for PendingTargets {
    fn drop(&mut self) {
        if self.0.is_empty() {
            return;
        }
        let mut held = PENDING_TARGETS.lock();
        for t in &self.0 {
            if let Some(i) = held.iter().position(|(id, _)| id == t) {
                held[i].1 -= 1;
                if held[i].1 == 0 {
                    held.swap_remove(i);
                    let mut last = LAST_STAMP.lock();
                    if let Some(j) = last.iter().position(|(id, _)| id == t) {
                        last.swap_remove(j);
                    }
                }
            }
        }
    }
}

/// TEMP (lab, live `RETAIN_FAST=on|off`, else `REIMS_VGPU_LAB_RETAIN_FAST=off` to
/// disable; on by default since b091): whether a drain question about one
/// resident ([`settle_unless_pending`]) skips the wait for deferred draws that
/// do not write it. p070/p077 left it opt-in because the waits it removed moved
/// to the stream-end completion word; STAMP_DEFER now holds that word behind
/// the recorder instead of the drain. b091 (ROUTES=off): drain joins 15 -> 10
/// ms/s on the reveal-desktop, window fps reveal 126 against 122, Safari 126
/// against 122, Finder 105 against 103, Weather even; ART 0.
pub(crate) fn lab_retain_fast() -> bool {
    if cfg!(test) {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("RETAIN_FAST") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_RETAIN_FAST").as_deref() != Ok("off"))
}

/// The drain's settle before a question about `identity` alone: skipped when no
/// deferred draw still out writes it, since only those can change what the
/// registry answers for it (create, replace or ready its image). Every other
/// door into engine state still settles. p062: the drain waited for the
/// recorder 13-15 % of the tranches that open an animation and of its hitches,
/// 45 % of that at `retain_resident_resource` and 16 % at
/// `resident_content_backing` - for sampled surfaces the queued draws did not
/// touch, while the recorder itself sat idle ~60 % of those windows.
#[track_caller]
pub(crate) fn settle_unless_pending(is_drain: bool, site: JoinSite, identity: &super::types::TargetIdentity) -> bool {
    if is_drain && lab_retain_fast() && DRAW.in_flight() {
        let pending = PENDING_TARGETS.lock().iter().any(|(id, _)| id == identity);
        if !pending {
            crate::runtime::drain::note_store_route("lab_retain_fast_skipped");
            return false;
        }
        // TEMP (lab, live `RETAIN_TARGETED=off` restores the full settle): wait
        // for the jobs that write this resident, not for the whole queue. The
        // queue is FIFO and only a writer changes what the registry answers for
        // it, so once the last one is done the answer is the one a full settle
        // gives - the state the no-writer case above already acts on. b129
        // (overlap, queue 64): the full settle here was 7.5 % of the drain.
        if !crate::runtime::par::planning()
            && crate::runtime::lab_live::on("RETAIN_TARGETED") != Some(false)
        {
            DRAW.settle_until(site, || {
                !PENDING_TARGETS.lock().iter().any(|(id, _)| id == identity)
            });
            crate::runtime::drain::note_store_route("lab_retain_targeted");
            return true;
        }
    }
    settle_if_drain(is_drain, site);
    true
}

thread_local! {
    static IS_RECORDER_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn mark_recorder_thread() {
    IS_RECORDER_THREAD.with(|c| c.set(true));
}

/// Whether the calling thread is the draw recorder, for the engine-lock census
/// (its acquires are neither the drain's nor a device entry point's).
#[inline]
pub(crate) fn is_recorder_thread() -> bool {
    IS_RECORDER_THREAD.with(std::cell::Cell::get)
}

/// How `REIMS_VGPU_LAB_RECORDER` was set.
enum Mode {
    Off,
    On,
    /// `toggle:<flag file>`: on while the file exists, so an A/B can alternate
    /// the two arms inside one boot, where the host's load is shared between
    /// them, instead of across boots, where it is not.
    Toggle(String),
}

fn mode() -> &'static Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    MODE.get_or_init(|| match std::env::var("REIMS_VGPU_LAB_RECORDER") {
        Ok(v) if v == "off" => Mode::Off,
        Ok(v) => match v.strip_prefix("toggle:") {
            Some(path) if !path.is_empty() => Mode::Toggle(path.to_string()),
            _ => Mode::On,
        },
        Err(_) => Mode::On,
    })
}

/// `REIMS_VGPU_LAB_RECORDER=on` (or `toggle:<file>` with the file present).
pub(crate) fn enabled() -> bool {
    match mode() {
        Mode::Off => false,
        Mode::On => true,
        Mode::Toggle(path) => {
            static LAST_MS: AtomicU64 = AtomicU64::new(0);
            static OPEN: AtomicBool = AtomicBool::new(false);
            let now = crate::observe::elapsed_us() / 1000 + 1;
            let last = LAST_MS.load(Ordering::Relaxed);
            if last != 0 && now.saturating_sub(last) < 100 {
                return OPEN.load(Ordering::Relaxed);
            }
            LAST_MS.store(now, Ordering::Relaxed);
            let open = std::path::Path::new(path).exists();
            OPEN.store(open, Ordering::Relaxed);
            open
        }
    }
}

/// The wait every drain-side engine door performs. The caller says whether it
/// is the drain; any other thread (the recorder itself, the window, a vCPU)
/// must not wait for a job it may be part of or has no ordering claim on.
///
/// TEMP (lab, PASS_PREP): a thread inside a pool scope (`crate::runtime::par`)
/// does not wait either — waiting for the recorder from inside a scope is what
/// the scope exists to overlap (the scope plans the next chunk of draws while
/// the recorder encodes the previous one). Safe because a planning job only
/// reads: whatever it read as "not settled yet" is a version the commit after
/// the scope re-validates, and the engine's own locks carry a
/// `debug_assert!` that no planning thread ever takes them.
#[inline]
#[track_caller]
thread_local! {
    /// TEMP (lab, STAMP_SEQ): while set, the drain's settles wait only for the
    /// jobs posted before this sequence number ([`with_settle_bound`]).
    static SETTLE_BOUND: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// TEMP (lab, STAMP_SEQ): run `f` with the drain's settles bounded to the
/// first `target` recorder jobs - what a completion word held behind exactly
/// those jobs owes, rather than an idle recorder.
pub(crate) fn with_settle_bound<R>(target: u64, f: impl FnOnce() -> R) -> R {
    let prev = SETTLE_BOUND.with(|b| b.replace(Some(target)));
    let out = f();
    SETTLE_BOUND.with(|b| b.set(prev));
    out
}

pub(crate) fn settle_if_drain(is_drain: bool, site: JoinSite) {
    if !is_drain {
        return;
    }
    if let Some(target) = SETTLE_BOUND.with(std::cell::Cell::get) {
        if DRAW.finished_seq() >= target {
            return;
        }
        DRAW.settle_until(site, || DRAW.finished_seq() >= target);
        return;
    }
    // TEMP (lab, PASS_PREP): a planning thread skips the wait and the census
    // says so — but only when there is a job to skip, so the route's count is
    // the number of overlapped opportunities, not of door calls.
    if crate::runtime::par::planning() && DRAW.in_flight() {
        crate::runtime::drain::note_store_route("pass_prep_settle_skipped");
        return;
    }
    DRAW.settle(site);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    fn handoff(name: &'static str) -> &'static Handoff {
        Box::leak(Box::new(Handoff::new(name)))
    }

    fn ok_job(f: impl FnOnce() + Send + 'static) -> Job {
        Box::new(move || {
            f();
            JobResult::Done
        })
    }

    /// Jobs run in submission order, one at a time, and a settle returns only
    /// after the job's effects are visible to the settler.
    #[test]
    fn jobs_run_in_order_and_settle_publishes_their_effects() {
        let h = handoff("rec-test-order");
        let log = Arc::new(Mutex::new(Vec::new()));
        for i in 0..200u32 {
            let job_log = Arc::clone(&log);
            let in_job = Arc::new(AtomicUsize::new(0));
            let probe = Arc::clone(&in_job);
            let job = ok_job(move || {
                assert_eq!(probe.fetch_add(1, Ordering::SeqCst), 0, "two jobs at once");
                if i % 7 == 0 {
                    std::thread::sleep(Duration::from_micros(300));
                }
                job_log.lock().push(i);
                probe.fetch_sub(1, Ordering::SeqCst);
            });
            h.submit(job);
            if i % 3 == 0 {
                h.settle(JoinSite::Explicit);
                assert_eq!(
                    log.lock().last().copied(),
                    Some(i),
                    "settle must see the job"
                );
                assert!(!h.in_flight());
                assert!(matches!(h.take(JoinSite::Explicit), Some(JobResult::Done)));
            } else {
                // Taking is a settle too, and consumes the result.
                assert!(matches!(h.take(JoinSite::Explicit), Some(JobResult::Done)));
            }
        }
        assert_eq!(*log.lock(), (0..200).collect::<Vec<_>>());
        let _ = h.stop();
    }

    /// A failure travels to the join that takes it, intact, and a panic is
    /// re-raised there rather than lost with the worker.
    #[test]
    fn a_failure_and_a_panic_reach_the_taker() {
        let h = handoff("rec-test-err");
        let error = DrawError::FenceTimeout;
        let sent = error.clone();
        let job: Job = Box::new(move || {
            JobResult::Failed(Box::new(DeferredFailure {
                error: sent,
                meta: DeferredMeta {
                    pipeline_ref: 9,
                    ..Default::default()
                },
            }))
        });
        h.submit(job);
        match h.take(JoinSite::Explicit) {
            Some(JobResult::Failed(f)) => {
                assert_eq!(f.error, error);
                assert_eq!(f.meta.pipeline_ref, 9);
            }
            _ => panic!("the failure was lost"),
        }
        // Nothing left to take.
        assert!(h.take(JoinSite::Explicit).is_none());

        let job: Job = Box::new(|| panic!("boom in the recorder"));
        h.submit(job);
        match h.take(JoinSite::Explicit) {
            Some(JobResult::Panicked(payload)) => {
                let text = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .unwrap_or("<not a str>");
                assert_eq!(text, "boom in the recorder");
            }
            _ => panic!("the panic was lost"),
        }
        // The worker survived the panic and runs the next job.
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        h.submit(ok_job(move || flag.store(true, Ordering::SeqCst)));
        h.settle(JoinSite::Explicit);
        assert!(ran.load(Ordering::SeqCst));
        let _ = h.stop();
    }

    /// `stop` waits for the job, ends the thread, and a later submit starts a
    /// fresh one.
    #[test]
    fn stop_joins_the_worker_and_a_later_submit_restarts_it() {
        let h = handoff("rec-test-stop");
        let n = Arc::new(AtomicUsize::new(0));
        for _ in 0..3 {
            let c = Arc::clone(&n);
            h.submit(ok_job(move || {
                std::thread::sleep(Duration::from_millis(2));
                c.fetch_add(1, Ordering::SeqCst);
            }));
            let _ = h.stop();
            assert!(!h.alive.load(Ordering::SeqCst));
            assert!(!h.in_flight());
        }
        assert_eq!(n.load(Ordering::SeqCst), 3);
    }

    fn failing_job(pipeline_ref: u32) -> Job {
        Box::new(move || {
            JobResult::Failed(Box::new(DeferredFailure {
                error: DrawError::FenceTimeout,
                meta: DeferredMeta {
                    pipeline_ref,
                    ..Default::default()
                },
            }))
        })
    }

    /// Run `f` on its own thread and fail rather than hang if it does not finish.
    fn within(secs: u64, name: &str, f: impl FnOnce() + Send + 'static) {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            f();
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(Duration::from_secs(secs)).is_ok(),
            "{name} hung"
        );
    }

    /// The drain and a teardown thread can both settle: neither waits forever,
    /// and the one result is kept for whoever takes it.
    #[test]
    fn two_settlers_both_return_and_the_result_is_kept_once() {
        within(20, "two settlers", || {
            let h = handoff("rec-test-two-settlers");
            for round in 0..40u32 {
                let ran = Arc::new(AtomicUsize::new(0));
                let job_ran = Arc::clone(&ran);
                h.submit(ok_job(move || {
                    std::thread::sleep(Duration::from_millis(2));
                    job_ran.fetch_add(1, Ordering::SeqCst);
                }));
                let other = std::thread::spawn(move || h.settle(JoinSite::Explicit));
                h.settle(JoinSite::Explicit);
                other.join().unwrap();
                assert!(!h.in_flight(), "round {round}");
                assert_eq!(ran.load(Ordering::SeqCst), 1);
                assert!(matches!(h.take(JoinSite::Explicit), Some(JobResult::Done)));
                assert!(h.take(JoinSite::Explicit).is_none());
            }
            let _ = h.stop();
        });
    }

    /// A stop that races a submit neither strands a job (posted to a worker that
    /// is retiring) nor lets a restart's bookkeeping be overwritten: every job
    /// submitted runs exactly once and every settle returns.
    #[test]
    fn stop_racing_submit_never_strands_a_job() {
        within(30, "stop racing submit", || {
            let h = handoff("rec-test-stop-race");
            let ran = Arc::new(AtomicUsize::new(0));
            let stopper = std::thread::spawn(move || {
                for _ in 0..200 {
                    let _ = h.stop();
                    std::thread::yield_now();
                }
            });
            for _ in 0..400 {
                let job_ran = Arc::clone(&ran);
                h.submit(ok_job(move || {
                    job_ran.fetch_add(1, Ordering::SeqCst);
                }));
                h.settle(JoinSite::Explicit);
            }
            stopper.join().unwrap();
            let _ = h.stop();
            assert_eq!(ran.load(Ordering::SeqCst), 400);
        });
    }

    /// What `stop` hands back is the failure nobody had taken, and a restart does
    /// not find it again: the next device's first draw must not see the last one's.
    #[test]
    fn stop_hands_back_the_untaken_failure_once() {
        let h = handoff("rec-test-stop-failure");
        h.submit(failing_job(77));
        match h.stop() {
            Some(JobResult::Failed(f)) => assert_eq!(f.meta.pipeline_ref, 77),
            _ => panic!("the failure was not handed back"),
        }
        assert!(!h.pending(), "nothing is owed after a stop");
        h.submit(ok_job(|| {}));
        assert!(matches!(h.take(JoinSite::Explicit), Some(JobResult::Done)));
        let _ = h.stop();
    }

    /// An earlier failure outranks a later result, so the first loss is the one
    /// reported, and `pending` stays true until it is taken.
    #[test]
    fn an_older_failure_outranks_a_newer_result_and_stays_pending() {
        let h = handoff("rec-test-older");
        h.submit(failing_job(5));
        h.settle(JoinSite::Explicit);
        assert!(!h.in_flight());
        assert!(h.pending(), "the failure is not yet taken");
        h.submit(ok_job(|| {}));
        h.settle(JoinSite::Explicit);
        match h.take(JoinSite::Explicit) {
            Some(JobResult::Failed(f)) => assert_eq!(f.meta.pipeline_ref, 5),
            _ => panic!("the older failure was overwritten"),
        }
        assert!(!h.pending());
        let _ = h.stop();
    }

    /// The recorder's own lock acquires are counted under their own site, not as
    /// a device entry point or the drain, and the drain's CPU write into guest
    /// pages waits for the job where the engine lock does.
    #[test]
    fn the_recorder_has_its_own_lock_site_and_a_guest_write_waits_for_the_job() {
        static SERIAL: Mutex<()> = Mutex::new(());
        let _serial = SERIAL.lock();
        let site = Arc::new(AtomicUsize::new(usize::MAX));
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let effect = Arc::new(AtomicBool::new(false));
        let (site_in, started_in, release_in, effect_in) = (
            Arc::clone(&site),
            Arc::clone(&started),
            Arc::clone(&release),
            Arc::clone(&effect),
        );
        super::super::mark_drain_thread();
        assert_eq!(
            super::super::calling_site(),
            super::super::EngineLockSite::Worker
        );
        DRAW.submit(ok_job(move || {
            site_in.store(super::super::calling_site().index(), Ordering::SeqCst);
            started_in.store(true, Ordering::SeqCst);
            while !release_in.load(Ordering::SeqCst) {
                std::hint::spin_loop();
            }
            effect_in.store(true, Ordering::SeqCst);
        }));
        while !started.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
        assert_eq!(
            site.load(Ordering::SeqCst),
            super::super::EngineLockSite::Recorder.index(),
            "the worker is neither the drain nor a device entry point"
        );
        let releaser = std::thread::spawn({
            let release = Arc::clone(&release);
            move || {
                std::thread::sleep(Duration::from_millis(5));
                release.store(true, Ordering::SeqCst);
            }
        });
        // The drain is about to write guest pages with the CPU.
        super::super::settle_for_guest_write();
        assert!(
            effect.load(Ordering::SeqCst),
            "the drain wrote guest pages while the recorder still held a draw"
        );
        releaser.join().unwrap();
        let _ = DRAW.take(JoinSite::Explicit);
        let _ = DRAW.stop();
    }

    /// A prewake brings a parked worker out of its park without a job, counts
    /// itself once, runs nothing, and leaves the worker serving the next job; on a
    /// worker that is not parked it does nothing.
    #[test]
    fn a_prewake_wakes_a_parked_worker_without_running_anything() {
        let h = handoff("rec-test-prewake");
        let ran = Arc::new(AtomicUsize::new(0));
        let r = Arc::clone(&ran);
        h.submit(ok_job(move || {
            r.fetch_add(1, Ordering::SeqCst);
        }));
        h.settle(JoinSite::Explicit);
        // Long enough for the worker's spin to expire and for it to park.
        std::thread::sleep(Duration::from_millis(30));
        assert!(h.inner.lock().worker_parked, "the worker should be asleep");
        h.prewake();
        assert_eq!(h.stats.prewakes.load(Ordering::SeqCst), 1);
        assert_eq!(ran.load(Ordering::SeqCst), 1, "a prewake runs no job");
        // The kick is consumed by the wake: the worker spins and parks again, and
        // a later prewake wakes it once more.
        std::thread::sleep(Duration::from_millis(30));
        assert!(!h.inner.lock().kick, "the kick is cleared by the worker");
        h.prewake();
        assert_eq!(h.stats.prewakes.load(Ordering::SeqCst), 2);
        std::thread::sleep(Duration::from_millis(30));
        let r = Arc::clone(&ran);
        h.submit(ok_job(move || {
            r.fetch_add(1, Ordering::SeqCst);
        }));
        h.settle(JoinSite::Explicit);
        assert_eq!(
            ran.load(Ordering::SeqCst),
            2,
            "the worker still serves jobs"
        );
        // Not parked (a job just ran, the worker is spinning): nothing to do.
        let before = h.stats.prewakes.load(Ordering::SeqCst);
        h.prewake();
        assert_eq!(h.stats.prewakes.load(Ordering::SeqCst), before);
        let _ = h.stop();
    }

    /// What the handoff costs the drain, measured (run with `--ignored
    /// --nocapture`): a job of `job_us` is submitted, the drain spends `prep_us`
    /// of its own, then settles. Prints the mean submit cost and the mean wait at
    /// the settle; with the job shorter than the prep the wait should be ~0.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn handoff_cost_against_a_prep_of_the_drain() {
        fn spin_us(us: u64) {
            let end = Instant::now() + Duration::from_micros(us);
            while Instant::now() < end {
                std::hint::spin_loop();
            }
        }
        let h = handoff("rec-test-cost");
        for (job_us, prep_us) in [(7, 30), (17, 35), (7, 5), (40, 35)] {
            let n = 5_000u32;
            let (mut submit_ns, mut wait_ns) = (0u128, 0u128);
            for _ in 0..n {
                let t = Instant::now();
                h.submit(ok_job(move || spin_us(job_us)));
                submit_ns += t.elapsed().as_nanos();
                spin_us(prep_us);
                let t = Instant::now();
                h.settle(JoinSite::Explicit);
                wait_ns += t.elapsed().as_nanos();
                let _ = h.take(JoinSite::Explicit);
            }
            eprintln!(
                "job {job_us} us, prep {prep_us} us: submit {} ns, settle wait {} ns (mean of {n})",
                submit_ns / u128::from(n),
                wait_ns / u128::from(n)
            );
        }
        let _ = h.stop();
    }

    /// A thread that is not the drain never waits: the recorder's own
    /// `lock_engine` and `with` must not settle the job they belong to.
    #[test]
    fn only_the_drain_settles() {
        let before = DRAW.stats.join_n[JoinSite::Lock.index()].load(Ordering::Relaxed);
        settle_if_drain(false, JoinSite::Lock);
        settle_if_drain(true, JoinSite::Lock);
        // Nothing in flight: a settle is one load and counts nothing either way.
        assert_eq!(
            DRAW.stats.join_n[JoinSite::Lock.index()].load(Ordering::Relaxed),
            before
        );
    }

    /// The drain's door waits for the job: an engine lock taken by the drain
    /// after a submit observes the job's effect, and one taken by another
    /// thread does not wait.
    #[test]
    fn the_engine_lock_waits_for_the_job_on_the_drain_only() {
        static SERIAL: Mutex<()> = Mutex::new(());
        let _serial = SERIAL.lock();
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let effect = Arc::new(AtomicBool::new(false));
        let (s, r, e) = (
            Arc::clone(&started),
            Arc::clone(&release),
            Arc::clone(&effect),
        );
        super::super::mark_drain_thread();
        // A job that holds up until released, then makes its effect visible
        // while holding the engine lock, as a recorded draw does.
        let job = ok_job(move || {
            s.store(true, Ordering::SeqCst);
            while !r.load(Ordering::SeqCst) {
                std::hint::spin_loop();
            }
            let _engine = super::super::lock_engine();
            e.store(true, Ordering::SeqCst);
        });
        DRAW.submit(job);
        while !started.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
        // The window thread does not wait for the job: it asks for the engine
        // lock, finds it free (the job holds it only after release), and gets it.
        let other = std::thread::spawn(|| {
            let _g = super::super::lock_engine_at(super::super::EngineLockSite::Window);
        });
        other.join().unwrap();
        assert!(!effect.load(Ordering::SeqCst));
        // The drain's lock waits: release the job from another thread while the
        // drain blocks in `lock_engine`.
        let releaser = std::thread::spawn({
            let release = Arc::clone(&release);
            move || {
                std::thread::sleep(Duration::from_millis(5));
                release.store(true, Ordering::SeqCst);
            }
        });
        let guard = super::super::lock_engine();
        assert!(
            effect.load(Ordering::SeqCst),
            "the drain took the engine lock before the deferred draw was recorded"
        );
        drop(guard);
        releaser.join().unwrap();
        assert!(!DRAW.in_flight());
        assert!(matches!(
            DRAW.take(JoinSite::Explicit),
            Some(JobResult::Done)
        ));
        let _ = DRAW.stop();
    }

    /// TEMP (lab, PASS_PREP): a drain thread inside a pool scope — a planning
    /// coordinator — does not wait for the recorder job; the same thread
    /// outside the scope does. The scope is the serial pool (no helpers), which
    /// is the honest form of the question: it is the *scope*, not the fork,
    /// that carries the bypass.
    #[test]
    fn planning_coordinator_does_not_settle() {
        static SERIAL: Mutex<()> = Mutex::new(());
        let _serial = SERIAL.lock();
        unsafe extern "C" fn nothing(_ctx: *mut std::os::raw::c_void) {}
        let mut table = crate::qemu::host_ops::ReimsVgpuHostOps::null();
        table.thread_attach = Some(nothing);
        table.thread_detach = Some(nothing);
        let pool = crate::runtime::par::Pool::new(
            &table,
            &crate::runtime::par::Settings {
                workers: 0,
                worker_spin: Duration::ZERO,
                join_spin: Duration::ZERO,
            },
        )
        .expect("the table offers the registration pair");
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let effect = Arc::new(AtomicBool::new(false));
        let (s, r, e) = (
            Arc::clone(&started),
            Arc::clone(&release),
            Arc::clone(&effect),
        );
        super::super::mark_drain_thread();
        DRAW.submit(ok_job(move || {
            s.store(true, Ordering::SeqCst);
            while !r.load(Ordering::SeqCst) {
                std::hint::spin_loop();
            }
            e.store(true, Ordering::SeqCst);
        }));
        while !started.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
        // Released from outside rather than after the assertions, so a scope
        // that wrongly waits for the job still unwedges and fails a check
        // instead of hanging the suite.
        let releaser = std::thread::spawn({
            let release = Arc::clone(&release);
            move || {
                std::thread::sleep(Duration::from_millis(100));
                release.store(true, Ordering::SeqCst);
            }
        });
        let skipped_before =
            crate::runtime::drain::census::store_route_count("pass_prep_settle_skipped");
        let items = [0u8, 1];
        pool.scope(&(), &items, |_, _, _| {
            settle_if_drain(true, JoinSite::Reader);
        });
        assert_eq!(
            crate::runtime::drain::census::store_route_count("pass_prep_settle_skipped"),
            skipped_before + items.len() as u64,
            "each in-scope settle is counted as skipped"
        );
        assert!(
            !effect.load(Ordering::SeqCst),
            "the planning coordinator did not join the job"
        );
        // Outside the scope the same door waits, as it always did.
        settle_if_drain(true, JoinSite::Reader);
        assert!(
            effect.load(Ordering::SeqCst),
            "the ordinary drain still joins the job"
        );
        releaser.join().unwrap();
        let _ = DRAW.take(JoinSite::Explicit);
        let _ = DRAW.stop();
        pool.shutdown();
    }
}
