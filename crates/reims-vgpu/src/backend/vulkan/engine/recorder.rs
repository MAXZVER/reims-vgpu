//! TEMP (lab, S5a): a recorder thread that records the engine half of a chained
//! draw while the drain prepares the next one.
//!
//! `REIMS_VGPU_LAB_RECORDER=on` turns it on (default off; `off` is the old path,
//! byte for byte: nothing here is reached when it is off).
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
//! * **At most one job is in flight**, and only the drain submits one.
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
pub(crate) struct DeferredFailure {
    pub error: DrawError,
    pub meta: DeferredMeta,
}

/// The few request fields a failure report names.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DeferredMeta {
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
fn stash_result(inner: &mut Inner, result: JobResult) {
    let keep_older = matches!(
        inner.stash,
        Some(JobResult::Failed(_)) | Some(JobResult::Panicked(_))
    );
    if !keep_older {
        inner.stash = Some(result);
    }
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
    job: Option<Job>,
    result: Option<JobResult>,
    /// A result the drain already waited for (an implicit join) and nobody has
    /// taken yet.
    stash: Option<JobResult>,
    worker_parked: bool,
    submitter_waiting: bool,
    shutdown: bool,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// One-slot handoff to one worker thread.
pub(crate) struct Handoff {
    name: &'static str,
    inner: Mutex<Inner>,
    work_cv: Condvar,
    done_cv: Condvar,
    /// A job is posted and the worker has not taken it yet (spin target).
    pending: AtomicBool,
    /// The worker posted a result (spin target).
    done: AtomicBool,
    /// A job was submitted and its result is not yet in the stash. Written only
    /// by the submitting side.
    in_flight: AtomicBool,
    alive: AtomicBool,
    stats: Stats,
}

struct Stats {
    submitted: AtomicU64,
    busy_ns: AtomicU64,
    parked_wakes: AtomicU64,
    join_n: [AtomicU64; 4],
    join_blocked: [AtomicU64; 4],
    join_ns: [AtomicU64; 4],
    join_max_ns: [AtomicU64; 4],
    deferred_err: AtomicU64,
    submit_declined: AtomicU64,
}

#[allow(clippy::declare_interior_mutable_const)]
const Z: AtomicU64 = AtomicU64::new(0);

impl Stats {
    const fn new() -> Self {
        Self {
            submitted: Z,
            busy_ns: Z,
            parked_wakes: Z,
            join_n: [Z; 4],
            join_blocked: [Z; 4],
            join_ns: [Z; 4],
            join_max_ns: [Z; 4],
            deferred_err: Z,
            submit_declined: Z,
        }
    }
}

/// Spin before parking, on either side. Draws arrive 30-80 us apart inside a
/// stream and a park/wake round trip costs about as much as one, so the
/// recorder stays hot across a stream and sleeps between streams.
const WORKER_SPIN: Duration = Duration::from_micros(200);
const JOIN_SPIN: Duration = Duration::from_micros(30);

impl Handoff {
    pub(crate) const fn new(name: &'static str) -> Self {
        Self {
            name,
            inner: Mutex::new(Inner {
                job: None,
                result: None,
                stash: None,
                worker_parked: false,
                submitter_waiting: false,
                shutdown: false,
                thread: None,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
            pending: AtomicBool::new(false),
            done: AtomicBool::new(false),
            in_flight: AtomicBool::new(false),
            alive: AtomicBool::new(false),
            stats: Stats::new(),
        }
    }

    /// Whether a job is submitted and not yet waited for.
    #[inline]
    pub(crate) fn in_flight(&self) -> bool {
        self.in_flight.load(Ordering::Acquire)
    }

    fn ensure_worker(&'static self) -> bool {
        if self.alive.load(Ordering::Acquire) {
            return true;
        }
        let mut g = self.inner.lock();
        if self.alive.load(Ordering::Acquire) {
            return true;
        }
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
        loop {
            let spin_until = Instant::now() + WORKER_SPIN;
            let mut i = 0u32;
            while !self.pending.load(Ordering::Acquire) {
                std::hint::spin_loop();
                i = i.wrapping_add(1);
                if i & 63 == 0 && Instant::now() >= spin_until {
                    break;
                }
            }
            let mut g = self.inner.lock();
            while g.job.is_none() && !g.shutdown {
                g.worker_parked = true;
                self.work_cv.wait(&mut g);
            }
            g.worker_parked = false;
            let Some(job) = g.job.take() else {
                self.alive.store(false, Ordering::Release);
                return;
            };
            self.pending.store(false, Ordering::Release);
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
            g.result = Some(result);
            self.done.store(true, Ordering::Release);
            if g.submitter_waiting {
                self.done_cv.notify_one();
            }
        }
    }

    /// Post a job. The submitter must have settled any earlier one. When the
    /// worker cannot be started the job runs here, on the submitting thread,
    /// before this returns: the draw is recorded either way and the result waits
    /// in the stash like any other.
    pub(crate) fn submit(&'static self, job: Job) {
        debug_assert!(!self.in_flight(), "one job in flight");
        if self.in_flight() {
            // Defensive: wait for it rather than overwrite it.
            self.settle(JoinSite::Explicit);
        }
        if !self.ensure_worker() {
            self.stats.submit_declined.fetch_add(1, Ordering::Relaxed);
            let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
                Ok(result) => result,
                Err(payload) => JobResult::Panicked(payload),
            };
            stash_result(&mut self.inner.lock(), result);
            return;
        }
        let mut g = self.inner.lock();
        g.job = Some(job);
        self.in_flight.store(true, Ordering::Release);
        self.pending.store(true, Ordering::Release);
        if g.worker_parked {
            self.stats.parked_wakes.fetch_add(1, Ordering::Relaxed);
            self.work_cv.notify_one();
        }
        drop(g);
        self.stats.submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Wait for the in-flight job, if any, and keep its result for
    /// [`Self::take`]. A no-op (one atomic load) when nothing is in flight.
    #[inline]
    pub(crate) fn settle(&self, site: JoinSite) {
        if !self.in_flight.load(Ordering::Acquire) {
            return;
        }
        self.settle_slow(site);
    }

    #[cold]
    fn settle_slow(&self, site: JoinSite) {
        let started = Instant::now();
        let spin_until = started + JOIN_SPIN;
        let mut i = 0u32;
        let mut blocked = false;
        while !self.done.load(Ordering::Acquire) {
            std::hint::spin_loop();
            i = i.wrapping_add(1);
            if i & 31 == 0 && Instant::now() >= spin_until {
                break;
            }
        }
        let mut g = self.inner.lock();
        while g.result.is_none() {
            blocked = true;
            g.submitter_waiting = true;
            self.done_cv.wait(&mut g);
        }
        g.submitter_waiting = false;
        if let Some(result) = g.result.take() {
            stash_result(&mut g, result);
        }
        self.done.store(false, Ordering::Release);
        self.in_flight.store(false, Ordering::Release);
        drop(g);
        let ns = started.elapsed().as_nanos() as u64;
        let s = site.index();
        self.stats.join_n[s].fetch_add(1, Ordering::Relaxed);
        if blocked || ns > 1_000 {
            self.stats.join_blocked[s].fetch_add(1, Ordering::Relaxed);
        }
        self.stats.join_ns[s].fetch_add(ns, Ordering::Relaxed);
        self.stats.join_max_ns[s].fetch_max(ns, Ordering::Relaxed);
    }

    /// The finished job's result, waiting for it if it is still running, or
    /// `None` when no job has finished since the last take.
    pub(crate) fn take(&self, site: JoinSite) -> Option<JobResult> {
        self.settle(site);
        self.inner.lock().stash.take()
    }

    pub(crate) fn note_deferred_error(&self) {
        self.stats.deferred_err.fetch_add(1, Ordering::Relaxed);
    }

    /// Settle, then stop the worker and wait for it to exit. A later `submit`
    /// starts a new one.
    pub(crate) fn stop(&'static self) {
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
    }

    /// Drain the window into one line, or `None` when nothing happened in it.
    pub(crate) fn take_window(&self, win_ms: u64) -> Option<String> {
        let s = &self.stats;
        let submitted = s.submitted.swap(0, Ordering::Relaxed);
        let busy = s.busy_ns.swap(0, Ordering::Relaxed);
        let wakes = s.parked_wakes.swap(0, Ordering::Relaxed);
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
        any.then(|| {
            format!(
                "recorder win_ms={win_ms} deferred={submitted} busy_us={} wakes={wakes} \
                 deferred_err={err} submit_declined={declined}{body}",
                busy / 1000
            )
        })
    }
}

#[cfg(windows)]
fn current_thread_priority() -> Option<i32> {
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
fn apply_thread_priority(priority: Option<i32>) {
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
fn current_thread_priority() -> Option<i32> {
    None
}

#[cfg(not(windows))]
fn apply_thread_priority(_priority: Option<i32>) {}

/// The process's recorder: draws only.
pub(crate) static DRAW: Handoff = Handoff::new("reims-vgpu-recorder");

/// `REIMS_VGPU_LAB_RECORDER=on`. Read once.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_RECORDER").is_ok_and(|v| v == "on"))
}

/// The wait every drain-side engine door performs. The caller says whether it
/// is the drain; any other thread (the recorder itself, the window, a vCPU)
/// must not wait for a job it may be part of or has no ordering claim on.
#[inline]
pub(crate) fn settle_if_drain(is_drain: bool, site: JoinSite) {
    if is_drain {
        DRAW.settle(site);
    }
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
                assert_eq!(log.lock().last().copied(), Some(i), "settle must see the job");
                assert!(!h.in_flight());
                assert!(matches!(h.take(JoinSite::Explicit), Some(JobResult::Done)));
            } else {
                // Taking is a settle too, and consumes the result.
                assert!(matches!(h.take(JoinSite::Explicit), Some(JobResult::Done)));
            }
        }
        assert_eq!(*log.lock(), (0..200).collect::<Vec<_>>());
        h.stop();
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
        h.stop();
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
            h.stop();
            assert!(!h.alive.load(Ordering::SeqCst));
            assert!(!h.in_flight());
        }
        assert_eq!(n.load(Ordering::SeqCst), 3);
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
        assert!(matches!(DRAW.take(JoinSite::Explicit), Some(JobResult::Done)));
        DRAW.stop();
    }
}
