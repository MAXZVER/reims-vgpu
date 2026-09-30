//! The worker pool: persistent threads, a fork-join `scope`, read-only jobs.
//!
//! See the module doc of [`super`] for what the pool is for and what it must
//! never do. This file is the mechanism.
//!
//! # Shape
//!
//! One *coordinator* (the thread that calls [`Pool::scope`], which is the drain
//! thread) and `workers` helper threads parked on a condition variable. A scope
//! is a parallel map: the coordinator publishes one batch holding a cursor over
//! the items, wakes at most `items - 1` helpers, and then **claims items from the
//! same cursor itself**. Helpers that wake late find the cursor exhausted and go
//! back to sleep, so the scope's latency never depends on a helper having woken
//! — only on the items a helper *did* claim finishing.
//!
//! Claiming from one atomic cursor rather than handing each worker its share is
//! deliberate. The items are 5–200 µs and of very uneven cost, so a static split
//! leaves the coordinator waiting on whichever helper drew the expensive one,
//! where a shared cursor gives it to whoever is free.
//!
//! # Accounting
//!
//! [`reims_vgpu_core::executor::FixedExecutor`] holds the population (the
//! helpers plus the coordinator, which is the last [`WorkerId`]) and answers the
//! questions the fork-join itself cannot: how many batches each member took part
//! in ([`PoolStats::accepted`]), whether one is in a batch right now, and whether
//! it is still taking work. A member joins a batch with `dispatch` and leaves
//! with `finished`, once per member per batch rather than once per item, so the
//! shared lock is taken a handful of times per scope instead of per item. A
//! stopped helper is *refused* by `dispatch` and simply does not join; the items
//! it would have claimed stay on the cursor for whoever is left, which is always
//! the coordinator.
//!
//! # Liveness does not depend on the helpers
//!
//! The coordinator can run every item itself, and does whenever a fork is not
//! worth it (fewer than two items, no helpers, a pool that was stopped, or a
//! second scope arriving while one is in flight). A helper that never wakes,
//! never starts, or has been joined therefore costs time, never a hang.
//!
//! # Soundness of the borrowed items
//!
//! The closure a scope runs borrows the caller's stack (the shared state, the
//! items, the result slots), and the helpers are `'static` threads. The one
//! `unsafe` here is the erasure of that borrow's lifetime, and it rests on a
//! single invariant: **a batch's runner is dereferenced only between a
//! successful claim (`index < total`) and the matching decrement of `pending`,
//! and [`Pool::scope`] does not return — nor unwind past [`Join`] — until
//! `pending` is zero.** A thread that claims an index at or past `total` never
//! touches the runner, and the batch itself is an `Arc`, so a helper still
//! holding it after the scope returned reads only atomics.

use super::Settings;
use crate::qemu::host_ops::{ReimsVgpuHostOps, ThreadRegistrar, WorkerHost};
use parking_lot::{Condvar, Mutex};
use reims_vgpu_core::executor::{Assignment, FixedExecutor, WorkerId};
use reims_vgpu_core::identity::IngressOrdinal;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Stack for a helper thread. Virtual address space, committed on touch: the
/// jobs are the same resolve code the drain thread runs, which is not shallow.
const WORKER_STACK_BYTES: usize = 4 * 1024 * 1024;

/// A job that panicked, carried back to the coordinator as data.
///
/// The pool never lets a panic cross a thread boundary or unwind into the
/// coordinator's frames: the item's slot holds this instead of a result, and
/// the coordinator decides what that means. The plan/commit split reads it as a
/// bail to the serial path — the work is not lost, it is done the ordinary way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobPanic {
    /// The panic payload as text, whitespace-flattened (see `observe::panic`).
    pub message: String,
}

impl std::fmt::Display for JobPanic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "job panicked: {}", self.message)
    }
}

/// What a job is told about where it is running.
pub struct WorkerEnv<'a> {
    /// The pool member running the job; the coordinator is the last id.
    pub worker: WorkerId,
    /// Whether this is a pool thread rather than the coordinator.
    pub helper: bool,
    ops: &'a ReimsVgpuHostOps,
}

impl<'a> WorkerEnv<'a> {
    /// The read-only view of the host a job reads guest memory through.
    ///
    /// Owned and cheap, so a job that needs a `&mut H` for a generic signature
    /// takes one; it is the same view on every thread, the coordinator included,
    /// which is what makes a job's answer independent of who ran it.
    pub fn host(&self) -> WorkerHost<'a> {
        WorkerHost::new(self.ops)
    }
}

/// The outcome of one [`Pool::scope`].
#[derive(Debug)]
pub struct Scoped<T> {
    /// One entry per item, in item order.
    pub results: Vec<Result<T, JobPanic>>,
    /// Entry to the point every wanted helper had been signalled, in µs. The
    /// price of the fork: slot allocation, publication and the wake calls.
    pub fork_us: u64,
    /// The coordinator's idle time after it had no item left to claim, in µs.
    /// What the slowest helper item cost beyond the coordinator's own share.
    pub wait_us: u64,
    /// Items a pool thread ran, as opposed to the coordinator.
    pub helped: usize,
    /// Time pool threads spent inside items, in µs, summed over helpers.
    pub helper_busy_us: u64,
    /// The scope ran every item on the coordinator without forking.
    pub serial: bool,
}

/// The pool's lifetime counters, for tests and the census.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Configured helpers (not necessarily started).
    pub workers: usize,
    /// Helper threads that have attached to RCU and not yet detached.
    pub attached: usize,
    /// Helper threads that have detached, i.e. exited cleanly.
    pub detached: usize,
    pub scopes: u64,
    pub serial_scopes: u64,
    pub jobs: u64,
    pub helped_jobs: u64,
    pub job_panics: u64,
    /// Batches each member took part in, in [`WorkerId`] order (helpers, then
    /// the coordinator).
    pub accepted: Vec<usize>,
}

/// Where the helpers are in their life. The pool is built cold and starts its
/// threads on the first scope that wants one, so a device that never forks
/// never owns a thread.
enum Lifecycle {
    Cold,
    Running(Vec<JoinHandle<()>>),
    Stopped,
}

const COLD: u8 = 0;
const RUNNING: u8 = 1;
const STOPPED: u8 = 2;

/// A type-erased job runner: `(item index, who is running it)`.
type Runner<'a> = dyn Fn(usize, &WorkerEnv<'_>) + Sync + 'a;

/// The raw form of a [`Runner`] once its borrow is forgotten. See the module
/// doc's "Soundness" section for the invariant that makes dereferencing it
/// sound.
struct RunPtr(*const Runner<'static>);

// SAFETY: the pointee is `Sync` (the `Runner` type says so) and is only read
// through shared references; who may dereference it, and when, is the
// invariant in the module doc.
unsafe impl Send for RunPtr {}
unsafe impl Sync for RunPtr {}

/// One fork-join's work, shared between the coordinator and whichever helpers
/// woke in time.
struct Batch {
    seq: u64,
    total: usize,
    /// The next item to claim. Past `total` means nothing is left.
    next: AtomicUsize,
    /// Items claimed and not yet finished, plus items not yet claimed. Zero
    /// means every item is done and the runner will never be dereferenced again.
    pending: AtomicUsize,
    run: RunPtr,
    /// Items a helper ran, and µs a helper spent in them. Updated *before* the
    /// `pending` decrement that publishes the item, so the coordinator reads
    /// them complete after it sees zero.
    helped: AtomicUsize,
    helper_busy_ns: AtomicU64,
    /// The last finisher signals here, for a coordinator that has parked.
    done: Mutex<()>,
    done_cv: Condvar,
}

struct HandOff {
    /// Bumped once per published batch.
    epoch: u64,
    batch: Option<Arc<Batch>>,
    stopping: bool,
}

struct Shared {
    ops: ReimsVgpuHostOps,
    registrar: ThreadRegistrar,
    workers: usize,
    worker_spin: Duration,
    join_spin: Duration,
    hand_off: Mutex<HandOff>,
    wake: Condvar,
    /// `hand_off.epoch`, mirrored so an idle helper can spin on it without the
    /// lock.
    epoch: AtomicU64,
    executor: Mutex<FixedExecutor>,
    batch_seq: AtomicU64,
    attached: AtomicUsize,
    detached: AtomicUsize,
    scopes: AtomicU64,
    serial_scopes: AtomicU64,
    jobs: AtomicU64,
    helped_jobs: AtomicU64,
    job_panics: AtomicU64,
}

/// The fixed pool of worker threads of one device.
///
/// Built cold by [`Pool::new`]: nothing is spawned until a scope wants a
/// helper, and [`Pool::shutdown`] (also run on drop) stops and joins them.
pub struct Pool {
    shared: Arc<Shared>,
    lifecycle: Mutex<Lifecycle>,
    state: AtomicU8,
    /// Held for the length of a forked scope. A second scope (from another
    /// thread, or nested inside a job) does not wait for it: it runs serially.
    scope_lock: Mutex<()>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("workers", &self.shared.workers)
            .field("state", &self.state.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl Pool {
    /// A pool for the device whose shim table is `ops`, or `None` when the shim
    /// offers no way to register a worker thread with RCU.
    ///
    /// That `None` is the structural gate: a thread that reads guest memory
    /// without being registered can see a freed FlatView, so a table without
    /// both `thread_attach` and `thread_detach` gets no helper at all. The
    /// settings can narrow the pool to zero helpers, never widen past this.
    pub fn new(ops: &ReimsVgpuHostOps, settings: &Settings) -> Option<Pool> {
        let registrar = ops.thread_registrar()?;
        let workers = settings.workers.min(Settings::MAX_WORKERS);
        // The coordinator is one more member of the population.
        let population = u16::try_from(workers + 1).unwrap_or(u16::MAX);
        Some(Pool {
            shared: Arc::new(Shared {
                ops: *ops,
                registrar,
                workers,
                worker_spin: settings.worker_spin,
                join_spin: settings.join_spin,
                hand_off: Mutex::new(HandOff {
                    epoch: 0,
                    batch: None,
                    stopping: false,
                }),
                wake: Condvar::new(),
                epoch: AtomicU64::new(0),
                executor: Mutex::new(FixedExecutor::with_population(population)),
                batch_seq: AtomicU64::new(0),
                attached: AtomicUsize::new(0),
                detached: AtomicUsize::new(0),
                scopes: AtomicU64::new(0),
                serial_scopes: AtomicU64::new(0),
                jobs: AtomicU64::new(0),
                helped_jobs: AtomicU64::new(0),
                job_panics: AtomicU64::new(0),
            }),
            lifecycle: Mutex::new(Lifecycle::Cold),
            state: AtomicU8::new(COLD),
            scope_lock: Mutex::new(()),
        })
    }

    /// Configured helper count. Zero is the serial pool: every scope runs on its
    /// caller.
    pub fn workers(&self) -> usize {
        self.shared.workers
    }

    /// The coordinator's identity in the executor's population.
    fn coordinator(&self) -> WorkerId {
        WorkerId(self.shared.workers as u16)
    }

    /// The lifetime counters.
    pub fn stats(&self) -> PoolStats {
        let s = &self.shared;
        PoolStats {
            workers: s.workers,
            attached: s
                .attached
                .load(Ordering::Acquire)
                .saturating_sub(s.detached.load(Ordering::Acquire)),
            detached: s.detached.load(Ordering::Acquire),
            scopes: s.scopes.load(Ordering::Relaxed),
            serial_scopes: s.serial_scopes.load(Ordering::Relaxed),
            jobs: s.jobs.load(Ordering::Relaxed),
            helped_jobs: s.helped_jobs.load(Ordering::Relaxed),
            job_panics: s.job_panics.load(Ordering::Relaxed),
            accepted: s.executor.lock().accepted(),
        }
    }

    /// Run `f` over every item, in parallel where it pays, and return the
    /// results in item order.
    ///
    /// `shared` is what every job reads — the device state, held shared for the
    /// length of the call, which is why `S: Sync` is the bound
    /// [`crate::model::DeviceState`] is asserted against. Jobs may borrow the
    /// caller's stack freely; the call returns only after every job has
    /// finished. **A job must not take a lock the caller holds across this call**
    /// (the coordinator would wait on it while the job waits on the coordinator),
    /// and must not write guest memory, the device state or the engine: the
    /// worker's host refuses the first and the type system the second, and the
    /// third is the module's rule.
    ///
    /// A panic in a job becomes [`JobPanic`] in that item's slot and nothing
    /// else: the helper survives, the other items are unaffected, and the panic
    /// is reported on the always-on channel.
    ///
    /// Forks only when it can pay — at least two items, at least one helper, a
    /// started-or-startable pool that has not been stopped, and no other scope in
    /// flight. Otherwise every item runs on the caller, through the same runner.
    pub fn scope<S, I, T, F>(&self, shared: &S, items: &[I], f: F) -> Scoped<T>
    where
        S: Sync + ?Sized,
        I: Sync,
        T: Send,
        F: Fn(&S, &WorkerEnv<'_>, &I) -> T + Sync,
    {
        let entered = Instant::now();
        let total = items.len();
        let sh = &self.shared;
        let slots: Vec<Mutex<Option<Result<T, JobPanic>>>> =
            (0..total).map(|_| Mutex::new(None)).collect();
        let run = |i: usize, env: &WorkerEnv<'_>| {
            let outcome = catch_unwind(AssertUnwindSafe(|| f(shared, env, &items[i])));
            let outcome = outcome.map_err(|payload| {
                sh.job_panics.fetch_add(1, Ordering::Relaxed);
                crate::runtime::drain::note_store_route("par_job_panicked");
                crate::observe::panic::report("par_job", payload.as_ref());
                JobPanic {
                    message: crate::observe::panic::payload_text(payload.as_ref()),
                }
            });
            *slots[i].lock() = Some(outcome);
        };

        let forked =
            if total >= 2 && sh.workers > 0 && self.state.load(Ordering::Acquire) != STOPPED {
                self.scope_lock
                    .try_lock()
                    .and_then(|guard| self.fork(&run, total, entered, guard))
            } else {
                None
            };
        let (serial, fork_us, wait_us, helped, helper_busy_ns) = match forked {
            Some((fork_us, wait_us, helped, busy_ns)) => (false, fork_us, wait_us, helped, busy_ns),
            // Not forked, or the fork could not start its helpers: serial.
            None => {
                let env = WorkerEnv {
                    worker: self.coordinator(),
                    helper: false,
                    ops: &sh.ops,
                };
                for i in 0..total {
                    run(i, &env);
                }
                (true, 0, 0, 0, 0)
            }
        };

        let results = slots
            .into_iter()
            .map(|slot| {
                slot.into_inner().unwrap_or_else(|| {
                    // The runner returned without filling the slot, which only
                    // an unwind out of the runner itself can do.
                    Err(JobPanic {
                        message: "lost".to_string(),
                    })
                })
            })
            .collect();
        let helper_busy_us = helper_busy_ns / 1000;

        sh.scopes.fetch_add(1, Ordering::Relaxed);
        if serial {
            sh.serial_scopes.fetch_add(1, Ordering::Relaxed);
        }
        sh.jobs.fetch_add(total as u64, Ordering::Relaxed);
        sh.helped_jobs.fetch_add(helped as u64, Ordering::Relaxed);
        use crate::runtime::drain::{note_store_route, note_store_route_n, note_store_route_us};
        note_store_route("par_scopes");
        if serial {
            note_store_route("par_scopes_serial");
        }
        note_store_route_n("par_jobs", total as u64);
        note_store_route_n("par_jobs_helped", helped as u64);
        note_store_route_us("par_fork_us", fork_us);
        note_store_route_us("par_wait_us", wait_us);
        note_store_route_us("par_worker_busy_us", helper_busy_us);
        // The mean hides the case that matters: the coordinator waiting on an
        // item a *descheduled* helper has claimed, which costs a scheduler
        // quantum rather than the item. Counting the scopes past two thresholds
        // says how often that happens without a histogram.
        if wait_us >= 200 {
            note_store_route("par_wait_ge_200us");
        }
        if wait_us >= 1000 {
            note_store_route("par_wait_ge_1ms");
        }
        if fork_us >= 50 {
            note_store_route("par_fork_ge_50us");
        }

        Scoped {
            results,
            fork_us,
            wait_us,
            helped,
            helper_busy_us,
            serial,
        }
    }

    /// The forking half of [`Self::scope`]: publish, wake, take a share, wait.
    ///
    /// `None` when the helpers could not be started, in which case nothing was
    /// published and the caller runs serially. The guard is held for the whole
    /// call, so scopes do not interleave on one batch slot.
    fn fork(
        &self,
        run: &Runner<'_>,
        total: usize,
        entered: Instant,
        _scope_guard: parking_lot::MutexGuard<'_, ()>,
    ) -> Option<(u64, u64, usize, u64)> {
        if !self.ensure_started() {
            return None;
        }
        let sh = &self.shared;
        // SAFETY: the batch built below is unpublished and waited to completion
        // by `Join` before this frame returns or unwinds, so `run` outlives
        // every dereference of the erased pointer; see the module doc.
        let erased: &'static Runner<'static> =
            unsafe { std::mem::transmute::<&Runner<'_>, &'static Runner<'static>>(run) };
        let batch = Arc::new(Batch {
            seq: sh.batch_seq.fetch_add(1, Ordering::Relaxed) + 1,
            total,
            next: AtomicUsize::new(0),
            pending: AtomicUsize::new(total),
            run: RunPtr(erased),
            helped: AtomicUsize::new(0),
            helper_busy_ns: AtomicU64::new(0),
            done: Mutex::new(()),
            done_cv: Condvar::new(),
        });
        let join = Join {
            shared: sh,
            batch: &batch,
        };

        {
            let mut hand_off = sh.hand_off.lock();
            hand_off.epoch += 1;
            hand_off.batch = Some(Arc::clone(&batch));
            sh.epoch.store(hand_off.epoch, Ordering::Release);
        }
        // The coordinator takes a share itself, so `total - 1` helpers at most.
        for _ in 0..sh.workers.min(total - 1) {
            sh.wake.notify_one();
        }
        let fork_us = entered.elapsed().as_micros() as u64;

        let env = WorkerEnv {
            worker: self.coordinator(),
            helper: false,
            ops: &sh.ops,
        };
        participate(sh, self.coordinator(), &batch, &env);

        let waited = Instant::now();
        drop(join);
        let wait_us = waited.elapsed().as_micros() as u64;

        Some((
            fork_us,
            wait_us,
            batch.helped.load(Ordering::Acquire),
            batch.helper_busy_ns.load(Ordering::Acquire),
        ))
    }

    /// Start the helper threads on first use. `false` when they cannot run:
    /// the pool was stopped, or a thread failed to spawn.
    fn ensure_started(&self) -> bool {
        match self.state.load(Ordering::Acquire) {
            RUNNING => return true,
            STOPPED => return false,
            _ => {}
        }
        let mut lifecycle = self.lifecycle.lock();
        match &*lifecycle {
            Lifecycle::Running(_) => return true,
            Lifecycle::Stopped => return false,
            Lifecycle::Cold => {}
        }
        let mut handles = Vec::with_capacity(self.shared.workers);
        for index in 0..self.shared.workers {
            let shared = Arc::clone(&self.shared);
            let spawned = std::thread::Builder::new()
                .name(format!("reims-vgpu-par-{index}"))
                .stack_size(WORKER_STACK_BYTES)
                .spawn(move || worker_main(&shared, index));
            match spawned {
                Ok(handle) => handles.push(handle),
                Err(e) => {
                    crate::observe::fail(format!(
                        "par_pool_spawn_failed index={index} workers={} err={e}",
                        self.shared.workers
                    ));
                    // A pool of fewer helpers than its population says would
                    // break the accounting, so it is all or none: stop what
                    // started and stay serial for good.
                    drop(lifecycle);
                    self.stop(handles);
                    return false;
                }
            }
        }
        *lifecycle = Lifecycle::Running(handles);
        self.state.store(RUNNING, Ordering::Release);
        crate::observe::off(format!(
            "par_pool_started workers={} worker_spin_us={} join_spin_us={}",
            self.shared.workers,
            self.shared.worker_spin.as_micros(),
            self.shared.join_spin.as_micros()
        ));
        true
    }

    /// Stop taking work, wake every helper, and join them. Idempotent.
    ///
    /// After it returns every helper has detached from RCU and exited, and every
    /// later scope runs serially on its caller. Waits for a scope in flight.
    pub fn shutdown(&self) {
        let _scopes = self.scope_lock.lock();
        let handles = match std::mem::replace(&mut *self.lifecycle.lock(), Lifecycle::Stopped) {
            Lifecycle::Running(handles) => handles,
            Lifecycle::Cold | Lifecycle::Stopped => Vec::new(),
        };
        self.state.store(STOPPED, Ordering::Release);
        self.stop(handles);
    }

    fn stop(&self, handles: Vec<JoinHandle<()>>) {
        self.state.store(STOPPED, Ordering::Release);
        *self.lifecycle.lock() = Lifecycle::Stopped;
        {
            let mut executor = self.shared.executor.lock();
            for index in 0..self.shared.workers {
                executor.stop(WorkerId(index as u16));
            }
        }
        self.shared.hand_off.lock().stopping = true;
        self.shared.wake.notify_all();
        let joined = handles.len();
        for handle in handles {
            // A helper that panicked outside a job has nothing more to say
            // than the panic report already made.
            let _ = handle.join();
        }
        if joined > 0 {
            // The evidence that teardown reached every helper before the shim
            // unrealizes the device: each one detached from RCU and exited.
            let stats = self.stats();
            crate::observe::off(format!(
                "par_pool_stopped workers={} joined={joined} detached={} scopes={} jobs={} helped={} job_panics={} accepted={}",
                stats.workers,
                stats.detached,
                stats.scopes,
                stats.jobs,
                stats.helped_jobs,
                stats.job_panics,
                stats
                    .accepted
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("/"),
            ));
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Waits for a batch to finish and withdraws it, on drop.
///
/// A guard rather than straight-line code because the lifetime erasure in
/// [`Pool::fork`] is only sound if **no path** leaves that frame while an item
/// can still be running, unwinding included.
struct Join<'a> {
    shared: &'a Shared,
    batch: &'a Arc<Batch>,
}

impl Drop for Join<'_> {
    fn drop(&mut self) {
        let batch = self.batch;
        if batch.pending.load(Ordering::Acquire) != 0 {
            let spin_until = Instant::now() + self.shared.join_spin;
            while batch.pending.load(Ordering::Acquire) != 0 && Instant::now() < spin_until {
                std::hint::spin_loop();
            }
            let mut guard = batch.done.lock();
            while batch.pending.load(Ordering::Acquire) != 0 {
                batch.done_cv.wait(&mut guard);
            }
        }
        let mut hand_off = self.shared.hand_off.lock();
        if hand_off
            .batch
            .as_ref()
            .is_some_and(|published| published.seq == batch.seq)
        {
            hand_off.batch = None;
        }
    }
}

/// Claim items from `batch` and run them until none is left.
///
/// The one loop the coordinator and every helper share, so an item runs the
/// same way wherever it lands.
fn participate(shared: &Shared, member: WorkerId, batch: &Arc<Batch>, env: &WorkerEnv<'_>) {
    // Cheap exit for a helper that woke after the work was gone, which is the
    // common case for a small batch.
    if batch.next.load(Ordering::Relaxed) >= batch.total {
        return;
    }
    let receipt = match shared.executor.lock().dispatch(Assignment {
        worker: member,
        ingress: IngressOrdinal(batch.seq),
        payload: (),
    }) {
        Ok(receipt) => receipt,
        // Stopped, or not a member: does not join. What it would have taken
        // stays on the cursor for the coordinator.
        Err(_refused) => return,
    };
    loop {
        let index = batch.next.fetch_add(1, Ordering::Relaxed);
        if index >= batch.total {
            break;
        }
        let started = env.helper.then(Instant::now);
        // Decrements `pending` however the item ends, so a coordinator never
        // waits on an item that unwound.
        let _finished = ItemDone {
            batch,
            started,
            helper: env.helper,
        };
        // SAFETY: `index < total` was claimed and `pending` has not yet been
        // decremented for it, so `Pool::fork`'s `Join` is still waiting and the
        // runner's borrow is alive.
        let runner = unsafe { &*batch.run.0 };
        let _ = catch_unwind(AssertUnwindSafe(|| runner(index, env)));
    }
    shared.executor.lock().finished(receipt);
}

/// Marks one claimed item finished, publishing what the coordinator reads.
struct ItemDone<'a> {
    batch: &'a Batch,
    started: Option<Instant>,
    helper: bool,
}

impl Drop for ItemDone<'_> {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            self.batch
                .helper_busy_ns
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        if self.helper {
            self.batch.helped.fetch_add(1, Ordering::Relaxed);
        }
        if self.batch.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Last one out. Taken under the lock so a coordinator between its
            // check and its wait cannot miss the wake.
            let _guard = self.batch.done.lock();
            self.batch.done_cv.notify_one();
        }
    }
}

/// A helper's whole life: register with RCU, serve batches, deregister.
fn worker_main(shared: &Arc<Shared>, index: usize) {
    let attachment = shared.registrar.attach();
    shared.attached.fetch_add(1, Ordering::AcqRel);
    let member = WorkerId(index as u16);
    let env = WorkerEnv {
        worker: member,
        helper: true,
        ops: &shared.ops,
    };
    let mut seen = 0u64;
    while let Some((epoch, batch)) = next_batch(shared, seen) {
        seen = epoch;
        if let Some(batch) = batch {
            participate(shared, member, &batch, &env);
        }
    }
    // Detach as the thread's last act, and count it after, so a test (or the
    // census) that reads `detached == workers` knows the registration is gone.
    drop(attachment);
    shared.detached.fetch_add(1, Ordering::AcqRel);
}

/// Wait for a batch newer than `seen`, or for the pool to stop (`None`).
///
/// Returns the epoch and the batch current at that epoch, which may already be
/// gone if the coordinator finished alone first.
fn next_batch(shared: &Shared, seen: u64) -> Option<(u64, Option<Arc<Batch>>)> {
    if !shared.worker_spin.is_zero() {
        let spin_until = Instant::now() + shared.worker_spin;
        while shared.epoch.load(Ordering::Acquire) == seen && Instant::now() < spin_until {
            std::hint::spin_loop();
        }
    }
    let mut hand_off = shared.hand_off.lock();
    loop {
        if hand_off.stopping {
            return None;
        }
        if hand_off.epoch != seen {
            return Some((hand_off.epoch, hand_off.batch.clone()));
        }
        shared.wake.wait(&mut hand_off);
    }
}
