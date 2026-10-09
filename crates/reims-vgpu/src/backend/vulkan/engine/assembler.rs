//! TEMP (lab, DRAW_ASM, 2026-10-09): assembler threads that build a
//! [`DrawRequest`](super::DrawRequest) off the drain thread, feeding the one
//! recorder FIFO in the order the drain submitted.
//!
//! `REIMS_VGPU_LAB_ASM_WORKERS=0` (or the live `ASM_WORKERS=0`) is the serial
//! path, byte for byte: the assembling closure runs on the submitting thread
//! before [`submit_assembled_draw`] returns, and the recorder job finds its slot
//! already full.
//!
//! # Shape
//!
//! The drain used to prepare a record's binds, sampled sources and request, and
//! then post the engine half to the recorder. Here the drain hands the
//! preparation itself (an [`AssembleFn`]) to a small pool and posts the recorder
//! job at once; that job's first act is to wait for *its own* slot. The
//! recorder FIFO is untouched, so the engine sees exactly the sequence it saw
//! before, however the assemblers interleave:
//!
//! * a slot is private to one job, so assembler N finishing before assembler N-1
//!   changes nothing -- the recorder takes N-1's slot first;
//! * the pool's queue is FIFO and an assembler never waits for anything but its
//!   own closure, so the oldest slot always has a thread working on it or about
//!   to (no cycle: the recorder waits for assemblers, never the reverse).
//!
//! # Threads and soundness
//!
//! A [`DrawRequest`](super::DrawRequest) is not `Send` (raw host pointers), and
//! it now crosses twice: assembler to slot to recorder. It travels in
//! [`AssertSend`] under the same argument as `recorder`'s module doc: the
//! pointers are RAMBlock aliases valid for the VM's lifetime, or the stable
//! allocations of buffers owned by `Arc`s inside the request, which move with
//! it.
//!
//! The assemblers read guest memory only through what the closure captured, and
//! the closure's owner is responsible for the same thing the planning pool's
//! jobs are (read-only, no engine lock). They do **not** attach to QEMU's RCU
//! (`par/mod.rs` rule 3): an assembler never calls a `HostOps` function that
//! touches a memory-region tree, it only dereferences host pointers that were
//! mapped before the closure was built.
//!
//! # Teardown
//!
//! [`shutdown`] runs the jobs still queued and joins the workers; it is called
//! from `end_device` right before the recorder is stopped. A job already posted
//! to the recorder keeps its slot alive through its `Arc`, so a recorder job that
//! has not run yet still finds its slot filled afterwards.
//!
//! # Census
//!
//! One `assembler` line per window: draws submitted, how many assembled inline
//! (pool off or unreachable), how often and how long the recorder waited on a
//! slot that was not ready (the cost the pool failed to hide), each worker's
//! busy time, and the deepest the queue got.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::recorder::{
    apply_thread_priority, current_thread_priority, AssertSend, DeferredFailure, DeferredMeta,
    JobResult, PendingTargets,
};
use super::types::{DrawError, TargetIdentity};
use super::{
    device_caches, execute_draw_request_with, facade_decline, recorder, DeviceObjectCaches,
    DrawRequest,
};

/// What an assembler runs: the preparation of one draw request. It owns
/// everything it reads; it never borrows from the drain's stack.
#[allow(dead_code)] // named by the runtime half of DRAW_ASM, which is not wired yet
pub(crate) type AssembleFn = Box<dyn FnOnce() -> Result<DrawRequest, DrawError> + Send + 'static>;

/// Upper bound on assembler threads (and on the per-worker busy counters).
const MAX_WORKERS: usize = 16;

/// Default worker count: the drain prepares ~35 us a record against the
/// recorder's ~7-17 us, so a few assemblers cover the drain's lead.
const DEFAULT_WORKERS: usize = 3;

/// How long the recorder spins on a slot before it parks on it. A slot that is
/// not ready after this has an assembler mid-closure; a park/wake round trip
/// costs about as much as the rest of one.
const SLOT_SPIN: Duration = Duration::from_micros(20);

/// How often the live `ASM_WORKERS` switch is looked at.
const LIVE_RECHECK_US: u64 = 250_000;

/// `live_workers` before the switch was seen or when it names no value.
const LIVE_UNSET: u32 = u32::MAX;

/// `REIMS_VGPU_LAB_ASM_WORKERS` (default 3, `0` = no pool); read once.
fn env_workers() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_ASM_WORKERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_WORKERS)
            .min(MAX_WORKERS)
    })
}

/// How long a worker spins for the next job before it parks.
/// `REIMS_VGPU_LAB_ASM_SPIN_US` overrides (default 50); read once.
fn worker_spin() -> Duration {
    static SPIN: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *SPIN.get_or_init(|| {
        Duration::from_micros(
            std::env::var("REIMS_VGPU_LAB_ASM_SPIN_US")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(50),
        )
    })
}

/// What an assembler left in a slot.
pub(crate) enum SlotResult {
    Ready(AssertSend<DrawRequest>),
    /// The closure refused: reported like an engine refusal of the same draw.
    Failed(DrawError),
    /// The closure panicked. Re-raised on the recorder thread as the job's
    /// `Panicked` result, which the drain re-raises at its join.
    Panicked(Box<dyn Any + Send>),
}

/// One draw's hand-back from an assembler to its recorder job.
pub(crate) struct AsmSlot {
    result: Mutex<Option<SlotResult>>,
    cv: Condvar,
    /// Set (release) after the result is stored; the recorder's spin target.
    ready: std::sync::atomic::AtomicBool,
}

impl AsmSlot {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            result: Mutex::new(None),
            cv: Condvar::new(),
            ready: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn fill(&self, result: SlotResult) {
        *self.result.lock() = Some(result);
        self.ready.store(true, Ordering::Release);
        self.cv.notify_all();
    }

    /// Take the result, waiting for it. The second element is how long the
    /// caller waited in nanoseconds, `0` when the slot was already full.
    /// Never times out: an assembly closure always finishes or panics, and
    /// `shutdown` runs the queue out before it joins. A slot is taken once.
    pub(crate) fn wait(&self) -> (SlotResult, u64) {
        let started = Instant::now();
        let mut waited = false;
        if !self.ready.load(Ordering::Acquire) {
            waited = true;
            let until = started + SLOT_SPIN;
            let mut i = 0u32;
            while !self.ready.load(Ordering::Acquire) {
                std::hint::spin_loop();
                i = i.wrapping_add(1);
                if i & 31 == 0 && Instant::now() >= until {
                    break;
                }
            }
        }
        let mut g = self.result.lock();
        loop {
            if let Some(result) = g.take() {
                let ns = if waited {
                    started.elapsed().as_nanos() as u64
                } else {
                    0
                };
                return (result, ns);
            }
            waited = true;
            self.cv.wait(&mut g);
        }
    }
}

type PoolJob = (Arc<AsmSlot>, AssembleFn);

struct Inner {
    /// Posted and not yet taken by a worker, oldest first.
    jobs: VecDeque<PoolJob>,
    threads: Vec<std::thread::JoinHandle<()>>,
    /// Workers exist (or are being retired: see `shutdown`).
    started: bool,
    /// Workers are running the queue out and exiting; a submit meanwhile
    /// assembles inline.
    shutdown: bool,
    /// A start found no thread to be had: latched until the next shutdown so a
    /// host that cannot spawn does not retry on every draw.
    spawn_failed: bool,
    parked: u32,
}

struct Stats {
    submitted: AtomicU64,
    inline: AtomicU64,
    wait_n: AtomicU64,
    wait_ns: AtomicU64,
    queue_max: AtomicU64,
    busy_ns: [AtomicU64; MAX_WORKERS],
}

#[allow(clippy::declare_interior_mutable_const)]
const Z: AtomicU64 = AtomicU64::new(0);

impl Stats {
    const fn new() -> Self {
        Self {
            submitted: Z,
            inline: Z,
            wait_n: Z,
            wait_ns: Z,
            queue_max: Z,
            busy_ns: [Z; MAX_WORKERS],
        }
    }
}

pub(crate) struct AssemblerPool {
    name: &'static str,
    /// Tests pin the worker count; the process pool reads the environment and
    /// the live switch.
    fixed: Option<usize>,
    inner: Mutex<Inner>,
    work_cv: Condvar,
    /// `Inner::jobs.len()`, mirrored for the workers' spin (written under
    /// `inner`).
    queued: AtomicUsize,
    /// Serializes shutdowns, so a second caller returns after the first has
    /// joined, not before.
    shutdown_lock: Mutex<()>,
    live_checked_us: AtomicU64,
    live_workers: AtomicU32,
    stats: Stats,
}

/// The process's assemblers.
pub(crate) static ASSEMBLER: AssemblerPool = AssemblerPool::new("reims-vgpu-asm", None);

impl AssemblerPool {
    const fn new(name: &'static str, fixed: Option<usize>) -> Self {
        Self {
            name,
            fixed,
            inner: Mutex::new(Inner {
                jobs: VecDeque::new(),
                threads: Vec::new(),
                started: false,
                shutdown: false,
                spawn_failed: false,
                parked: 0,
            }),
            work_cv: Condvar::new(),
            queued: AtomicUsize::new(0),
            shutdown_lock: Mutex::new(()),
            live_checked_us: AtomicU64::new(0),
            live_workers: AtomicU32::new(LIVE_UNSET),
            stats: Stats::new(),
        }
    }

    /// How many workers a submit should use: `0` is "assemble inline". The live
    /// file's `ASM_WORKERS` (looked at every 250 ms) wins over the environment;
    /// it can switch the pool off and on, and it sets the count the threads are
    /// started with, but threads already running stay (parked when unused).
    fn desired_workers(&self) -> usize {
        if let Some(n) = self.fixed {
            return n;
        }
        let now = crate::observe::elapsed_us();
        let last = self.live_checked_us.load(Ordering::Relaxed);
        if last == 0 || now.saturating_sub(last) >= LIVE_RECHECK_US {
            self.live_checked_us.store(now.max(1), Ordering::Relaxed);
            let live = crate::runtime::lab_live::value("ASM_WORKERS")
                .and_then(|v| v.trim().parse::<usize>().ok())
                .map_or(LIVE_UNSET, |n| n.min(MAX_WORKERS) as u32);
            self.live_workers.store(live, Ordering::Relaxed);
        }
        match self.live_workers.load(Ordering::Relaxed) {
            LIVE_UNSET => env_workers(),
            n => n as usize,
        }
    }

    /// Post `assemble` and return the slot it will fill. When the pool is off,
    /// retiring, or cannot start a thread, the closure runs here, before this
    /// returns, and the slot is already full.
    pub(crate) fn post(&'static self, assemble: AssembleFn) -> Arc<AsmSlot> {
        let slot = AsmSlot::new();
        self.stats.submitted.fetch_add(1, Ordering::Relaxed);
        let wanted = self.desired_workers();
        let mut job = Some((slot.clone(), assemble));
        if wanted != 0 {
            let mut g = self.inner.lock();
            if !g.shutdown && !g.spawn_failed {
                if !g.started {
                    self.start_workers(&mut g, wanted);
                }
                if g.started {
                    // `job` is always `Some` here; the take only moves it.
                    if let Some(j) = job.take() {
                        g.jobs.push_back(j);
                        let depth = g.jobs.len();
                        self.queued.store(depth, Ordering::Release);
                        self.stats
                            .queue_max
                            .fetch_max(depth as u64, Ordering::Relaxed);
                        if g.parked != 0 {
                            self.work_cv.notify_one();
                        }
                    }
                }
            }
        }
        if let Some((slot, assemble)) = job {
            self.stats.inline.fetch_add(1, Ordering::Relaxed);
            slot.fill(run_assemble(assemble));
            return slot;
        }
        slot
    }

    /// Called with `inner` held, `started` false. Leaves `started` false (and
    /// latches `spawn_failed`) when not even one thread could be had.
    fn start_workers(&'static self, g: &mut Inner, wanted: usize) {
        let priority = current_thread_priority();
        for index in 0..wanted.min(MAX_WORKERS) {
            let spawned = std::thread::Builder::new()
                .name(format!("{}-{index}", self.name))
                // `execute_draw_request`'s frame is the recorder's concern; a
                // request is assembled on shallower stacks, but the closure is
                // the drain's own preparation, which has run on 8 MiB so far.
                .stack_size(8 << 20)
                .spawn(move || self.worker_loop(index, priority));
            match spawned {
                Ok(handle) => g.threads.push(handle),
                Err(_) => break,
            }
        }
        if g.threads.is_empty() {
            g.spawn_failed = true;
        } else {
            g.started = true;
        }
    }

    fn worker_loop(&'static self, index: usize, priority: Option<i32>) {
        apply_thread_priority(priority);
        #[cfg(windows)]
        crate::lab_cores::pin_to_performance_cores(self.name);
        crate::lab_cores::device_thread_priority(self.name);
        mark_assembler_thread();
        loop {
            let spin_until = Instant::now() + worker_spin();
            let mut i = 0u32;
            while self.queued.load(Ordering::Acquire) == 0 {
                std::hint::spin_loop();
                i = i.wrapping_add(1);
                if i & 63 == 0 && Instant::now() >= spin_until {
                    break;
                }
            }
            let mut g = self.inner.lock();
            let job = loop {
                // The queue runs out before a retiring worker exits: a
                // recorder job already posted waits on one of these slots.
                if let Some(job) = g.jobs.pop_front() {
                    self.queued.store(g.jobs.len(), Ordering::Release);
                    break Some(job);
                }
                if g.shutdown {
                    break None;
                }
                g.parked += 1;
                self.work_cv.wait(&mut g);
                g.parked -= 1;
            };
            drop(g);
            let Some((slot, assemble)) = job else {
                return;
            };
            let started = Instant::now();
            let result = run_assemble(assemble);
            self.stats.busy_ns[index.min(MAX_WORKERS - 1)]
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            slot.fill(result);
        }
    }

    fn note_wait(&self, ns: u64) {
        self.stats.wait_n.fetch_add(1, Ordering::Relaxed);
        self.stats.wait_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Run the queue out, stop the workers and join them. A later submit starts
    /// a fresh set.
    pub(crate) fn shutdown(&self) {
        let _one_at_a_time = self.shutdown_lock.lock();
        let threads = {
            let mut g = self.inner.lock();
            if !g.started && !g.spawn_failed {
                return;
            }
            g.shutdown = true;
            self.work_cv.notify_all();
            std::mem::take(&mut g.threads)
        };
        for handle in threads {
            let _ = handle.join();
        }
        let mut g = self.inner.lock();
        g.shutdown = false;
        g.started = false;
        g.spawn_failed = false;
    }

    /// Drain the window into one line, or `None` when nothing happened in it.
    pub(crate) fn take_window(&self, win_ms: u64) -> Option<String> {
        let s = &self.stats;
        let submitted = s.submitted.swap(0, Ordering::Relaxed);
        let inline = s.inline.swap(0, Ordering::Relaxed);
        let wait_n = s.wait_n.swap(0, Ordering::Relaxed);
        let wait_ns = s.wait_ns.swap(0, Ordering::Relaxed);
        let qmax = s.queue_max.swap(0, Ordering::Relaxed);
        let workers = self.inner.lock().threads.len();
        let busy: Vec<String> = s.busy_ns[..workers.min(MAX_WORKERS)]
            .iter()
            .map(|b| (b.swap(0, Ordering::Relaxed) / 1000).to_string())
            .collect();
        for b in &s.busy_ns[workers.min(MAX_WORKERS)..] {
            b.store(0, Ordering::Relaxed);
        }
        (submitted != 0).then(|| {
            format!(
                "assembler win_ms={win_ms} workers={workers} submitted={submitted} \
                 inline={inline} wait_n={wait_n} wait_us={} busy_us=[{}] qmax={qmax}",
                wait_ns / 1000,
                busy.join(",")
            )
        })
    }
}

/// Run one closure, keeping a panic for the recorder job to re-raise.
fn run_assemble(assemble: AssembleFn) -> SlotResult {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(assemble)) {
        Ok(Ok(req)) => SlotResult::Ready(AssertSend(req)),
        Ok(Err(error)) => SlotResult::Failed(error),
        Err(payload) => SlotResult::Panicked(payload),
    }
}

thread_local! {
    static IS_ASSEMBLER_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn mark_assembler_thread() {
    IS_ASSEMBLER_THREAD.with(|c| c.set(true));
}

/// Whether the calling thread is an assembler: it is neither the drain nor the
/// recorder, so it takes no engine-state door that waits for a job.
#[allow(dead_code)] // consumed by the runtime half of DRAW_ASM
#[inline]
pub(crate) fn is_assembler_thread() -> bool {
    IS_ASSEMBLER_THREAD.with(std::cell::Cell::get)
}

/// Stop and join the assemblers. Called by `end_device` before the recorder is
/// stopped: the queue runs out first, so a recorder job still posted finds its
/// slot filled.
pub(crate) fn shutdown() {
    ASSEMBLER.shutdown();
}

/// TEMP (lab, DRAW_ASM): the assemblers' window, beside the `recorder` line.
pub(crate) fn take_window(win_ms: u64) -> Option<String> {
    ASSEMBLER.take_window(win_ms)
}

/// Like [`super::submit_deferred_draw`], but the request is not built yet:
/// `assemble` builds it on an assembler thread (or here, when the pool is off)
/// while the drain moves on, and the one recorder job posted here waits for it.
/// `writes` is what the draw will write (known without the request), held until
/// the job ends for `RETAIN_FAST`.
///
/// An assembly refusal is reported like an engine refusal of the same draw, at
/// the drain's next explicit join. When the device's caches are unreachable the
/// assembly is not run at all (the synchronous path refuses before preparing
/// anything either).
///
/// Called on the drain thread, with neither the engine nor the caches held.
#[allow(dead_code)] // called by the runtime half of DRAW_ASM
pub fn submit_assembled_draw(
    state: &crate::model::DeviceState,
    assemble: AssembleFn,
    meta: DeferredMeta,
    writes: Vec<TargetIdentity>,
) {
    let Some(caches) = device_caches(state) else {
        let error = DrawError::Facade(facade_decline::EngineFacadeDecline::DeviceCachesUnreachable);
        recorder::DRAW.submit(Box::new(move || {
            JobResult::Failed(Box::new(DeferredFailure { error, meta }))
        }));
        return;
    };
    let caches = AssertSend(caches as *const DeviceObjectCaches);
    // TEMP (lab, RETAIN_FAST): held from this submit to the end of the job.
    let writes = PendingTargets::hold(writes);
    let slot = ASSEMBLER.post(assemble);
    // The only borrow the job carries is the caches' address, valid until the
    // drain settles it (see `recorder`'s module doc).
    let job: Box<dyn FnOnce() -> JobResult + Send> = Box::new(move || {
        let _writes = writes;
        let (result, waited_ns) = slot.wait();
        if waited_ns != 0 {
            ASSEMBLER.note_wait(waited_ns);
        }
        let req = match result {
            SlotResult::Ready(req) => req.into_inner(),
            SlotResult::Failed(error) => {
                return JobResult::Failed(Box::new(DeferredFailure { error, meta }));
            }
            SlotResult::Panicked(payload) => return JobResult::Panicked(payload),
        };
        let caches = caches.into_inner();
        // SAFETY: `recorder`'s module doc -- the drain holds the device the
        // caches belong to for the whole tranche and settles this job before it
        // ends.
        let caches = unsafe { &*caches };
        match caches.with(|caches| execute_draw_request_with(caches, &req)) {
            Ok(_) => JobResult::Done,
            Err(error) => JobResult::Failed(Box::new(DeferredFailure { error, meta })),
        }
    });
    recorder::DRAW.submit(job);
}

/// Test door: post `assemble` to the process pool and hand back its slot.
#[cfg(test)]
pub(crate) fn submit_to_pool_for_test(assemble: AssembleFn) -> Arc<AsmSlot> {
    ASSEMBLER.post(assemble)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::vk_call::{VkCall, VkOp};

    fn pool(workers: usize) -> &'static AssemblerPool {
        Box::leak(Box::new(AssemblerPool::new("asm-test", Some(workers))))
    }

    /// An error that names the draw it came from.
    fn tagged(i: i32) -> DrawError {
        DrawError::VkCall(VkCall::new(
            VkOp::ReadbackResetCb,
            ash::vk::Result::from_raw(-1000 - i),
        ))
    }

    fn failed_with(result: SlotResult) -> DrawError {
        match result {
            SlotResult::Failed(error) => error,
            _ => panic!("expected the closure's refusal"),
        }
    }

    /// With no workers the closure has run, on this thread, before `post`
    /// returns, and the wait finds the slot full.
    #[test]
    fn a_disabled_pool_assembles_inline() {
        let p = pool(0);
        let here = std::thread::current().id();
        let ran_on = Arc::new(Mutex::new(None));
        let record = Arc::clone(&ran_on);
        let slot = p.post(Box::new(move || {
            *record.lock() = Some(std::thread::current().id());
            Err(tagged(7))
        }));
        assert_eq!(*ran_on.lock(), Some(here), "the closure ran on the caller");
        assert!(
            slot.ready.load(Ordering::SeqCst),
            "the slot is already full"
        );
        let (result, waited) = slot.wait();
        assert_eq!(waited, 0);
        assert_eq!(failed_with(result), tagged(7));
        assert!(p.inner.lock().threads.is_empty(), "no thread was started");
        assert_eq!(p.stats.inline.load(Ordering::SeqCst), 1);
        assert!(p.take_window(1000).unwrap().contains("inline=1"));
    }

    /// Slots are private to their draw: closures that finish in the opposite
    /// order still hand each slot its own result, waited in submission order.
    #[test]
    fn slots_fill_with_their_own_result_in_submission_order() {
        let p = pool(2);
        let slots: Vec<_> = (0..50)
            .map(|i| {
                p.post(Box::new(move || {
                    // Earlier draws take longer, so the later ones finish first.
                    std::thread::sleep(Duration::from_micros(((50 - i) * 20) as u64));
                    Err(tagged(i))
                }))
            })
            .collect();
        for (i, slot) in slots.iter().enumerate() {
            let (result, _) = slot.wait();
            assert_eq!(failed_with(result), tagged(i as i32), "slot {i}");
        }
        assert_eq!(p.inner.lock().threads.len(), 2);
        assert_eq!(p.stats.inline.load(Ordering::SeqCst), 0);
        let line = p.take_window(1000).unwrap();
        assert!(line.contains("workers=2 submitted=50"), "{line}");
        p.shutdown();
    }

    /// The closures run on assembler threads, which know they are.
    #[test]
    fn workers_are_named_and_marked() {
        let p = pool(1);
        let seen = Arc::new(Mutex::new(None));
        let record = Arc::clone(&seen);
        let slot = p.post(Box::new(move || {
            *record.lock() = Some((
                std::thread::current().name().map(str::to_string),
                is_assembler_thread(),
            ));
            Err(tagged(1))
        }));
        let _ = slot.wait();
        let (name, marked) = seen.lock().clone().unwrap();
        assert_eq!(name.as_deref(), Some("asm-test-0"));
        assert!(marked);
        assert!(!is_assembler_thread(), "the test thread is not one");
        p.shutdown();
    }

    /// A panic in a closure is kept for the recorder job and the worker lives
    /// on to run the next one.
    #[test]
    fn a_panic_is_carried_in_the_slot_and_the_worker_survives() {
        let p = pool(1);
        let boom = p.post(Box::new(|| panic!("boom in the assembler")));
        let next = p.post(Box::new(|| Err(tagged(3))));
        match boom.wait().0 {
            SlotResult::Panicked(payload) => {
                assert_eq!(
                    payload.downcast_ref::<&str>().copied(),
                    Some("boom in the assembler")
                );
            }
            _ => panic!("the panic was lost"),
        }
        assert_eq!(failed_with(next.wait().0), tagged(3));
        p.shutdown();
    }

    /// `shutdown` runs queued closures out, joins the workers, and a later post
    /// starts a new set; a second shutdown with nothing running is a no-op.
    #[test]
    fn shutdown_runs_the_queue_out_and_a_later_post_restarts() {
        let p = pool(2);
        let ran = Arc::new(AtomicUsize::new(0));
        let slots: Vec<_> = (0..20)
            .map(|i| {
                let ran = Arc::clone(&ran);
                p.post(Box::new(move || {
                    std::thread::sleep(Duration::from_micros(200));
                    ran.fetch_add(1, Ordering::SeqCst);
                    Err(tagged(i))
                }))
            })
            .collect();
        p.shutdown();
        assert_eq!(ran.load(Ordering::SeqCst), 20, "the queue ran out");
        assert!(slots.iter().all(|s| s.ready.load(Ordering::SeqCst)));
        assert!(!p.inner.lock().started);
        p.shutdown();
        let again = p.post(Box::new(|| Err(tagged(99))));
        assert_eq!(failed_with(again.wait().0), tagged(99));
        assert_eq!(p.inner.lock().threads.len(), 2, "a fresh set started");
        p.shutdown();
    }

    /// The process pool's test door fills a slot too (default workers, or none
    /// when the environment turned them off).
    #[test]
    fn the_process_pool_fills_a_slot() {
        let slot = submit_to_pool_for_test(Box::new(|| Err(tagged(5))));
        assert_eq!(failed_with(slot.wait().0), tagged(5));
        shutdown();
    }
}
