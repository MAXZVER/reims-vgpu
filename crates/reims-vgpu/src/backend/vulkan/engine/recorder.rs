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
    job: Option<Job>,
    result: Option<JobResult>,
    /// A result the drain already waited for (an implicit join) and nobody has
    /// taken yet.
    stash: Option<JobResult>,
    worker_parked: bool,
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

impl Handoff {
    pub(crate) const fn new(name: &'static str) -> Self {
        Self {
            name,
            lifecycle: Mutex::new(()),
            settle_lock: Mutex::new(()),
            inner: Mutex::new(Inner {
                job: None,
                result: None,
                stash: None,
                worker_parked: false,
                waiters: 0,
                shutdown: false,
                thread: None,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
            stash_pending: AtomicBool::new(false),
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

    /// Whether a job is in flight or its result has not been taken: the state
    /// in which the drain still owes the recorder a join, whatever the gate says
    /// now (it may have been turned off while a draw was out).
    #[inline]
    pub(crate) fn pending(&self) -> bool {
        self.in_flight.load(Ordering::Acquire) || self.stash_pending.load(Ordering::Acquire)
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
        if std::ptr::eq(self, &DRAW) {
            mark_recorder_thread();
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
            while g.job.is_none() && !g.shutdown {
                g.worker_parked = true;
                self.work_cv.wait(&mut g);
            }
            g.worker_parked = false;
            // Retiring: `stop` clears `alive` after it has joined this thread,
            // under `lifecycle`, so there is nothing for the exit to publish.
            let Some(job) = g.job.take() else {
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
            stash_result(&mut self.inner.lock(), &self.stash_pending, result);
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
    #[track_caller]
    pub(crate) fn settle(&self, site: JoinSite) {
        if !self.in_flight.load(Ordering::Acquire) {
            return;
        }
        self.settle_slow(site);
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
            g.waiters += 1;
            self.done_cv.wait(&mut g);
            g.waiters -= 1;
        }
        if let Some(result) = g.result.take() {
            stash_result(&mut g, &self.stash_pending, result);
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
                 deferred_err={err} submit_declined={declined}{body} top={}",
                busy / 1000,
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

/// `REIMS_VGPU_LAB_RECORDER=on`. Read once.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_RECORDER").is_ok_and(|v| v == "on"))
}

/// The wait every drain-side engine door performs. The caller says whether it
/// is the drain; any other thread (the recorder itself, the window, a vCPU)
/// must not wait for a job it may be part of or has no ordering claim on.
#[inline]
#[track_caller]
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
            site_in.store(
                super::super::calling_site().index(),
                Ordering::SeqCst,
            );
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
        assert!(matches!(DRAW.take(JoinSite::Explicit), Some(JobResult::Done)));
        let _ = DRAW.stop();
    }
}
