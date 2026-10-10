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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
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
    /// TEMP (lab, FRAME_GATE): the frame the helpers serve, when one is open.
    frame: Option<Arc<FrameShared>>,
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
    /// Helpers parked on `wake` right now (counted under `hand_off`). A
    /// publish wakes at most this many: a helper still spinning on `epoch`, or
    /// one between batches, sees the new epoch without a `notify_one` (~1.2 us
    /// each on Windows, the bulk of a fork's cost).
    parked: AtomicUsize,
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
                    frame: None,
                    stopping: false,
                }),
                wake: Condvar::new(),
                epoch: AtomicU64::new(0),
                executor: Mutex::new(FixedExecutor::with_population(population)),
                parked: AtomicUsize::new(0),
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
        // The coordinator plans for the whole call, the serial path included:
        // a scope that could not fork still runs its items through the same
        // read-only job code, and the engine's doors must not settle for it
        // either. Restores the previous value, so a scope nested inside a job
        // leaves the outer one standing.
        let _planning = super::enter_planning();
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
        // Only parked helpers need a wake: the others see the epoch.
        let wanted = sh.workers.min(total - 1).min(lab_wake_cap(total));
        for _ in 0..wanted.min(sh.parked.load(Ordering::Acquire)) {
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

    /// TEMP (lab, DRAW_PAR overlap): run `f` over `items` on the helpers while
    /// `body` consumes the results in order on the caller.
    ///
    /// Unlike [`Self::scope`] the caller does not take a share up front: it
    /// publishes the batch and goes straight into `body`, whose
    /// [`Overlap::wait`] hands it item `i` as soon as `i` is done - running `i`
    /// itself only when no helper has claimed it yet, so a pool whose helpers
    /// are slow to wake (or absent) costs time, never progress. `body` runs on
    /// the caller with `shared` still borrowed shared by the jobs, which is the
    /// point: the caller commits item `i` against the same state the helpers are
    /// preparing `i + 1..` from.
    ///
    /// The caller is **not** a planning thread inside `body` (its engine doors
    /// settle as on the serial path); it is one only while it runs an item.
    /// Leaving `body` - normally or by unwinding - cancels the items nobody has
    /// claimed and waits for the claimed ones, so the lifetime erasure rests on
    /// the same invariant as `scope`'s (see the module doc).
    pub fn overlap<S, I, T, F, R>(
        &self,
        shared: &S,
        items: &[I],
        f: F,
        body: impl FnOnce(&mut Overlap<'_, T>) -> R,
    ) -> R
    where
        S: Sync + ?Sized,
        I: Sync,
        T: Send,
        F: Fn(&S, &WorkerEnv<'_>, &I) -> T + Sync,
    {
        let entered = Instant::now();
        let total = items.len();
        let sh = &self.shared;
        let slots: Vec<OverlapSlot<T>> = (0..total)
            .map(|_| OverlapSlot {
                ready: AtomicBool::new(false),
                value: Mutex::new(None),
            })
            .collect();
        let waiting = AtomicUsize::new(usize::MAX);
        let notify: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());
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
            *slots[i].value.lock() = Some(outcome);
            slots[i].ready.store(true, Ordering::SeqCst);
            if waiting.load(Ordering::SeqCst) == i {
                let _g = notify.0.lock();
                notify.1.notify_all();
            }
        };
        let runner: &Runner<'_> = &run;
        let guard = if total >= 2 && sh.workers > 0 && self.state.load(Ordering::Acquire) != STOPPED {
            self.scope_lock.try_lock().filter(|_| self.ensure_started())
        } else {
            None
        };
        let batch = guard.as_ref().map(|_| {
            // SAFETY: the batch is waited to completion (claimed items) or
            // cancelled (unclaimed ones) by `OverlapJoin` before this frame
            // returns or unwinds, so `run` outlives every dereference of the
            // erased pointer; see the module doc.
            let erased: &'static Runner<'static> =
                unsafe { std::mem::transmute::<&Runner<'_>, &'static Runner<'static>>(runner) };
            Arc::new(Batch {
                seq: sh.batch_seq.fetch_add(1, Ordering::Relaxed) + 1,
                total,
                next: AtomicUsize::new(0),
                pending: AtomicUsize::new(total),
                run: RunPtr(erased),
                helped: AtomicUsize::new(0),
                helper_busy_ns: AtomicU64::new(0),
                done: Mutex::new(()),
                done_cv: Condvar::new(),
            })
        });
        let join = batch.as_ref().map(|batch| OverlapJoin { shared: sh, batch });
        if let Some(batch) = batch.as_ref() {
            {
                let mut hand_off = sh.hand_off.lock();
                hand_off.epoch += 1;
                hand_off.batch = Some(Arc::clone(batch));
                sh.epoch.store(hand_off.epoch, Ordering::Release);
            }
            let wanted = sh.workers.min(total).min(lab_wake_cap(total));
            for _ in 0..wanted.min(sh.parked.load(Ordering::Acquire)) {
                sh.wake.notify_one();
            }
        }
        let fork_us = entered.elapsed().as_micros() as u64;
        let mut ov = Overlap {
            sh,
            batch: batch.as_ref(),
            run: runner,
            slots: &slots,
            waiting: &waiting,
            notify: &notify,
            coordinator: self.coordinator(),
            cancelled_from: usize::MAX,
            stopped: false,
            wait_ns: 0,
            ran_here: 0,
        };
        let out = body(&mut ov);
        let (wait_ns, ran_here, stopped) = (ov.wait_ns, ov.ran_here, ov.stopped);
        drop(ov);
        drop(join);
        drop(guard);
        let helped = batch.as_ref().map_or(0, |b| b.helped.load(Ordering::Acquire));
        let busy_us = batch
            .as_ref()
            .map_or(0, |b| b.helper_busy_ns.load(Ordering::Acquire))
            / 1000;
        sh.scopes.fetch_add(1, Ordering::Relaxed);
        sh.jobs.fetch_add(total as u64, Ordering::Relaxed);
        sh.helped_jobs.fetch_add(helped as u64, Ordering::Relaxed);
        use crate::runtime::drain::{note_store_route, note_store_route_n, note_store_route_us};
        note_store_route("par_overlaps");
        if batch.is_none() {
            note_store_route("par_overlaps_serial");
        }
        if stopped {
            note_store_route("par_overlaps_stopped");
        }
        note_store_route_n("par_overlap_items", total as u64);
        note_store_route_n("par_overlap_helped", helped as u64);
        note_store_route_n("par_overlap_ran_here", ran_here as u64);
        note_store_route_us("par_overlap_fork_us", fork_us);
        note_store_route_us("par_overlap_wait_us", wait_ns / 1000);
        note_store_route_us("par_overlap_busy_us", busy_us);
        out
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

/// TEMP (lab, live `PAR_WAKE=N`, else `REIMS_VGPU_LAB_PAR_WAKE=N`; unset: no
/// cap): wake at most one helper per `N` items. b058 (Weather): PassPrep forks
/// ~5.2k scopes a second of ~8.8 items at ~1.2 us each, and the fork itself -
/// six `notify_one` wakes, ~1.2 us each on Windows - cost 7 us a scope, more
/// than the items; `Condvar::notify_one_slow` was 5.4 % of the drain's busy
/// samples. Waking fewer helpers for a small batch trades a little item
/// parallelism for most of the wake cost.
fn lab_wake_cap(total: usize) -> usize {
    if cfg!(test) {
        return usize::MAX;
    }
    static ENV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let per = crate::runtime::lab_live::value("PAR_WAKE")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or_else(|| {
            *ENV.get_or_init(|| {
                std::env::var("REIMS_VGPU_LAB_PAR_WAKE")
                    .ok()
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0)
            })
        });
    if per == 0 {
        usize::MAX
    } else {
        total.div_ceil(per).max(1)
    }
}

/// One item's result in an [`Overlap`], flagged ready after the fill.
struct OverlapSlot<T> {
    ready: AtomicBool,
    value: Mutex<Option<Result<T, JobPanic>>>,
}

/// The caller's side of a [`Pool::overlap`]: take item `i` when it is done.
pub struct Overlap<'a, T> {
    sh: &'a Shared,
    /// `None`: the batch was not forked; `wait` runs each item on the caller.
    batch: Option<&'a Arc<Batch>>,
    run: &'a Runner<'a>,
    slots: &'a [OverlapSlot<T>],
    waiting: &'a AtomicUsize,
    notify: &'a (Mutex<()>, Condvar),
    coordinator: WorkerId,
    /// Items at or past this were cancelled unclaimed (`usize::MAX`: none).
    cancelled_from: usize,
    stopped: bool,
    wait_ns: u64,
    ran_here: usize,
}

impl<T> Overlap<'_, T> {
    /// Whether the batch runs on helpers (`false`: every item on the caller).
    pub fn forked(&self) -> bool {
        self.batch.is_some()
    }

    /// Item `i`'s result: at once if it is done, by running it here if no
    /// helper has claimed it, else after waiting for the helper that has.
    /// `None` for an item cancelled by [`Self::stop`] or already taken.
    pub fn wait(&mut self, i: usize) -> Option<Result<T, JobPanic>> {
        let slots = self.slots;
        let slot = slots.get(i)?;
        if i >= self.cancelled_from {
            return None;
        }
        let Some(batch) = self.batch else {
            if !slot.ready.load(Ordering::Acquire) {
                self.run_here(i, None);
            }
            return slot.value.lock().take();
        };
        let mut started: Option<Instant> = None;
        loop {
            if slot.ready.load(Ordering::Acquire) {
                break;
            }
            if batch.next.load(Ordering::Relaxed) <= i {
                let idx = batch.next.fetch_add(1, Ordering::AcqRel);
                if idx < batch.total {
                    self.run_here(idx, Some(batch));
                    continue;
                }
            }
            // Claimed by a helper and still running: spin, then park.
            let t0 = *started.get_or_insert_with(Instant::now);
            let spin_until = Instant::now().max(t0) + self.sh.join_spin;
            while !slot.ready.load(Ordering::Acquire) && Instant::now() < spin_until {
                std::hint::spin_loop();
            }
            if slot.ready.load(Ordering::Acquire) {
                break;
            }
            self.waiting.store(i, Ordering::SeqCst);
            {
                let mut g = self.notify.0.lock();
                while !slot.ready.load(Ordering::SeqCst) {
                    self.notify.1.wait_for(&mut g, Duration::from_micros(500));
                }
            }
            self.waiting.store(usize::MAX, Ordering::SeqCst);
        }
        if let Some(t0) = started {
            self.wait_ns += t0.elapsed().as_nanos() as u64;
        }
        slot.value.lock().take()
    }

    /// Cancel the items no helper has claimed, wait for the claimed ones, and
    /// return every result not yet taken, from item `from` on (`None` for a
    /// cancelled item). The overlap is over: later `wait`s answer `None`.
    pub fn stop(&mut self, from: usize) -> Vec<Option<Result<T, JobPanic>>> {
        self.stopped = true;
        if let Some(batch) = self.batch {
            cancel_unclaimed(batch);
            wait_pending(self.sh, batch);
        }
        let out = self
            .slots
            .iter()
            .skip(from)
            .map(|slot| {
                if slot.ready.load(Ordering::Acquire) {
                    slot.value.lock().take()
                } else {
                    None
                }
            })
            .collect();
        self.cancelled_from = 0;
        out
    }

    fn run_here(&mut self, idx: usize, batch: Option<&Arc<Batch>>) {
        // The caller is a planning thread for exactly the item it runs.
        let _planning = super::enter_planning();
        let _finished = batch.map(|batch| ItemDone {
            batch,
            started: None,
            helper: false,
        });
        let env = WorkerEnv {
            worker: self.coordinator,
            helper: false,
            ops: &self.sh.ops,
        };
        let _ = catch_unwind(AssertUnwindSafe(|| (self.run)(idx, &env)));
        self.ran_here += 1;
    }
}

/// Take every unclaimed item off the cursor; the first index cancelled
/// (`total` when every item was claimed).
fn cancel_unclaimed(batch: &Batch) -> usize {
    let prev = batch.next.swap(batch.total, Ordering::AcqRel).min(batch.total);
    let unclaimed = batch.total - prev;
    if unclaimed > 0 && batch.pending.fetch_sub(unclaimed, Ordering::AcqRel) == unclaimed {
        let _guard = batch.done.lock();
        batch.done_cv.notify_one();
    }
    prev
}

/// Wait for a batch's claimed items to finish.
fn wait_pending(shared: &Shared, batch: &Batch) {
    if batch.pending.load(Ordering::Acquire) != 0 {
        let spin_until = Instant::now() + shared.join_spin;
        while batch.pending.load(Ordering::Acquire) != 0 && Instant::now() < spin_until {
            std::hint::spin_loop();
        }
        let mut guard = batch.done.lock();
        while batch.pending.load(Ordering::Acquire) != 0 {
            batch.done_cv.wait(&mut guard);
        }
    }
}

/// An [`Overlap`]'s exit, on drop: cancel what nobody claimed, wait for what
/// somebody did, withdraw the batch. A guard for the same reason as [`Join`].
struct OverlapJoin<'a> {
    shared: &'a Shared,
    batch: &'a Arc<Batch>,
}

impl Drop for OverlapJoin<'_> {
    fn drop(&mut self) {
        cancel_unclaimed(self.batch);
        wait_pending(self.shared, self.batch);
        let mut hand_off = self.shared.hand_off.lock();
        if hand_off
            .batch
            .as_ref()
            .is_some_and(|published| published.seq == self.batch.seq)
        {
            hand_off.batch = None;
        }
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
    crate::lab_cores::device_thread_priority("par-pool");
    crate::lab_cores::pin_pool_to_performance_cores("par-pool");
    let attachment = shared.registrar.attach();
    shared.attached.fetch_add(1, Ordering::AcqRel);
    let member = WorkerId(index as u16);
    // A helper is a planning thread for its whole life: every engine door it
    // could reach (it must reach none) would skip the recorder settle. The
    // token is dropped when the thread's stack unwinds, which is never a
    // concern anyone else can observe.
    let _planning = super::enter_planning();
    let env = WorkerEnv {
        worker: member,
        helper: true,
        ops: &shared.ops,
    };
    let mut seen = 0u64;
    while let Some((epoch, batch, frame)) = next_batch(shared, seen) {
        seen = epoch;
        if let Some(batch) = batch {
            participate(shared, member, &batch, &env);
        }
        if let Some(frame) = frame {
            serve_frame(&frame, &env);
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
#[allow(clippy::type_complexity)]
fn next_batch(
    shared: &Shared,
    seen: u64,
) -> Option<(u64, Option<Arc<Batch>>, Option<Arc<FrameShared>>)> {
    // TEMP (lab, live `PAR_SPIN=us`): the idle helper's spin, overriding the
    // latched `REIMS_VGPU_LAB_PAR_SPIN_US` - a helper still spinning when the
    // next scope publishes needs no wake (see `lab_wake_cap`).
    let worker_spin = if cfg!(test) {
        shared.worker_spin
    } else {
        crate::runtime::lab_live::value("PAR_SPIN")
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(shared.worker_spin, Duration::from_micros)
    };
    if !worker_spin.is_zero() {
        let spin_until = Instant::now() + worker_spin;
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
            return Some((hand_off.epoch, hand_off.batch.clone(), hand_off.frame.clone()));
        }
        shared.parked.fetch_add(1, Ordering::AcqRel);
        shared.wake.wait(&mut hand_off);
        shared.parked.fetch_sub(1, Ordering::AcqRel);
    }
}

// ---------------------------------------------------------------------------
// TEMP (lab, FRAME_GATE): a frame - the helpers serving one gate's job queue
// for a whole tranche (see `gate.rs`).
// ---------------------------------------------------------------------------

const J_QUEUED: u8 = 0;
const J_RUNNING: u8 = 1;
const J_DONE: u8 = 2;
const J_CANCELLED: u8 = 3;

type FrameFn<T> = Box<dyn FnOnce(&crate::model::DeviceState, &WorkerEnv<'_>) -> T + Send>;

/// A frame job, type-erased for the queue.
pub(crate) trait FrameJob: Send + Sync {
    /// Queued -> running; `false` when somebody else has it or it was cancelled.
    fn claim(&self) -> bool;
    /// Run a claimed job.
    fn run(&self, st: &crate::model::DeviceState, env: &WorkerEnv<'_>);
    /// Queued -> cancelled; a running job finishes.
    fn cancel(&self);
}

struct TypedJob<T> {
    status: AtomicU8,
    f: Mutex<Option<FrameFn<T>>>,
    out: Mutex<Option<Result<T, JobPanic>>>,
}

impl<T: Send + 'static> FrameJob for TypedJob<T> {
    fn claim(&self) -> bool {
        self.status
            .compare_exchange(J_QUEUED, J_RUNNING, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn run(&self, st: &crate::model::DeviceState, env: &WorkerEnv<'_>) {
        let f = self.f.lock().take();
        let outcome = match f {
            Some(f) => catch_unwind(AssertUnwindSafe(|| f(st, env))).map_err(|payload| {
                crate::runtime::drain::note_store_route("par_job_panicked");
                crate::observe::panic::report("frame_job", payload.as_ref());
                JobPanic {
                    message: crate::observe::panic::payload_text(payload.as_ref()),
                }
            }),
            None => Err(JobPanic {
                message: "lost".to_string(),
            }),
        };
        *self.out.lock() = Some(outcome);
        self.status.store(J_DONE, Ordering::Release);
    }

    fn cancel(&self) {
        let _ = self
            .status
            .compare_exchange(J_QUEUED, J_CANCELLED, Ordering::AcqRel, Ordering::Acquire);
    }
}

/// The drain's handle on one pushed job.
pub struct Ticket<T> {
    job: Arc<TypedJob<T>>,
}

impl<T: Send + 'static> Ticket<T> {
    pub(super) fn new(f: FrameFn<T>) -> Self {
        Ticket {
            job: Arc::new(TypedJob {
                status: AtomicU8::new(J_QUEUED),
                f: Mutex::new(Some(f)),
                out: Mutex::new(None),
            }),
        }
    }

    pub(super) fn erased(&self) -> Arc<dyn FrameJob> {
        Arc::clone(&self.job) as Arc<dyn FrameJob>
    }

    /// Whether a helper (or the drain) has finished it.
    pub fn done(&self) -> bool {
        self.job.status.load(Ordering::Acquire) == J_DONE
    }

    /// Whether nobody has started it yet.
    pub fn queued(&self) -> bool {
        self.job.status.load(Ordering::Acquire) == J_QUEUED
    }

    /// Cancel it if nobody has started it.
    pub fn cancel(&self) {
        self.job.cancel();
    }

    pub(super) fn try_take(&self) -> Option<Result<T, JobPanic>> {
        if self.done() {
            self.job.out.lock().take()
        } else {
            None
        }
    }

    pub(super) fn wait_or_run(
        &self,
        st: &crate::model::DeviceState,
        frame: Option<(&ReimsVgpuHostOps, WorkerId)>,
    ) -> Option<Result<T, JobPanic>> {
        let job = &*self.job;
        if let Some((ops, coordinator)) = frame {
            if job.claim() {
                let _planning = super::enter_planning();
                let env = WorkerEnv {
                    worker: coordinator,
                    helper: false,
                    ops,
                };
                job.run(st, &env);
                crate::runtime::drain::note_store_route("frame_job_ran_here");
                return job.out.lock().take();
            }
        } else {
            job.cancel();
        }
        let t0 = Instant::now();
        let mut waited = false;
        loop {
            match job.status.load(Ordering::Acquire) {
                J_DONE => break,
                J_CANCELLED => return None,
                _ => {}
            }
            waited = true;
            if t0.elapsed() < Duration::from_micros(300) {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        if waited {
            crate::runtime::drain::note_store_route("frame_job_waited");
            crate::runtime::drain::note_store_route_us(
                "frame_job_wait_us",
                t0.elapsed().as_micros() as u64,
            );
        }
        job.out.lock().take()
    }
}

/// The erased pointer to a frame's lock. See `FrameHandle`'s drop.
struct LockPtr(*const super::gate::FrameLock<'static>);

// SAFETY: the pointee is `Sync` (an `RwLock` over `&mut DeviceState`, which is
// `Send + Sync`), and it is dereferenced only by a helper counted in `inside`
// of a frame not yet closed, which `FrameHandle::drop` waits out.
unsafe impl Send for LockPtr {}
unsafe impl Sync for LockPtr {}

pub(crate) struct FrameShared {
    lock: LockPtr,
    queue: Mutex<std::collections::VecDeque<Arc<dyn FrameJob>>>,
    queued: AtomicUsize,
    closed: AtomicBool,
    /// Helpers inside `serve_frame` (they may dereference `lock`).
    inside: AtomicUsize,
    parked: AtomicUsize,
    park: Mutex<()>,
    park_cv: Condvar,
    spin: Duration,
    helped: AtomicU64,
    busy_ns: AtomicU64,
}

/// The gate's handle on its frame: the pool's helpers serve its queue until
/// this drops.
pub struct FrameHandle {
    pool: Arc<Pool>,
    frame: Arc<FrameShared>,
    pushed: std::cell::Cell<u64>,
}

impl Pool {
    /// Open a frame over `lock`: the helpers serve its queue until the handle
    /// drops. `None` when the pool has no helpers, was stopped, or a scope is
    /// in flight. **The handle must drop before `lock`** (the gate owns both
    /// in that order).
    pub(crate) fn open_frame(
        self: &Arc<Self>,
        lock: &super::gate::FrameLock<'_>,
    ) -> Option<FrameHandle> {
        let sh = &self.shared;
        if sh.workers == 0 || self.state.load(Ordering::Acquire) == STOPPED {
            return None;
        }
        let guard = self.scope_lock.try_lock()?;
        if !self.ensure_started() {
            return None;
        }
        // Released by the handle's drop (`force_unlock`): a scope or overlap
        // inside the frame runs serially, as for any second scope.
        std::mem::forget(guard);
        let spin = crate::runtime::lab_live::value("FRAME_SPIN")
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(Duration::from_micros(200), Duration::from_micros);
        // Lifetime erasure only; see `LockPtr`.
        let ptr = lock as *const super::gate::FrameLock<'_> as *const super::gate::FrameLock<'static>;
        let frame = Arc::new(FrameShared {
            lock: LockPtr(ptr),
            queue: Mutex::new(std::collections::VecDeque::new()),
            queued: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            inside: AtomicUsize::new(0),
            parked: AtomicUsize::new(0),
            park: Mutex::new(()),
            park_cv: Condvar::new(),
            spin,
            helped: AtomicU64::new(0),
            busy_ns: AtomicU64::new(0),
        });
        {
            let mut hand_off = sh.hand_off.lock();
            hand_off.epoch += 1;
            hand_off.batch = None;
            hand_off.frame = Some(Arc::clone(&frame));
            sh.epoch.store(hand_off.epoch, Ordering::Release);
        }
        // No wake here: a tranche that pushes nothing costs no helper, and
        // the first jobs wake them one at a time (`FrameHandle::push`).
        Some(FrameHandle {
            pool: Arc::clone(self),
            frame,
            pushed: std::cell::Cell::new(0),
        })
    }
}

impl FrameHandle {
    pub(super) fn ops(&self) -> (&ReimsVgpuHostOps, WorkerId) {
        (&self.pool.shared.ops, self.pool.coordinator())
    }

    pub(super) fn push(&self, job: Arc<dyn FrameJob>, open: bool) {
        let fr = &*self.frame;
        fr.queue.lock().push_back(job);
        let queued = fr.queued.fetch_add(1, Ordering::AcqRel) + 1;
        self.pushed.set(self.pushed.get() + 1);
        if open {
            self.wake_for(queued);
        }
    }

    /// Wake one parked helper when the queue holds more jobs than helpers
    /// are awake in the frame: a frame parker first, else one parked at the
    /// pool (not yet in this frame).
    fn wake_for(&self, queued: usize) {
        let fr = &*self.frame;
        let sh = &self.pool.shared;
        let awake = fr.inside.load(Ordering::Acquire) - fr.parked.load(Ordering::Acquire).min(fr.inside.load(Ordering::Acquire));
        if queued <= awake {
            return;
        }
        if fr.parked.load(Ordering::Acquire) > 0 {
            let _g = fr.park.lock();
            fr.park_cv.notify_one();
        } else if sh.parked.load(Ordering::Acquire) > 0 {
            sh.wake.notify_one();
        }
    }

    /// The gate opened: wake parked helpers for the queued jobs.
    pub(super) fn wake(&self) {
        let queued = self.frame.queued.load(Ordering::Acquire);
        if queued > 0 {
            self.wake_for(queued);
        }
    }
}

impl Drop for FrameHandle {
    fn drop(&mut self) {
        let fr = &*self.frame;
        fr.closed.store(true, Ordering::SeqCst);
        // Unstarted jobs are cancelled: their tickets answer `None`.
        let left: Vec<_> = fr.queue.lock().drain(..).collect();
        fr.queued.store(0, Ordering::Release);
        for job in &left {
            job.cancel();
        }
        {
            let _g = fr.park.lock();
            fr.park_cv.notify_all();
        }
        while fr.inside.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
        {
            let mut hand_off = self.pool.shared.hand_off.lock();
            if hand_off.frame.as_ref().is_some_and(|f| Arc::ptr_eq(f, &self.frame)) {
                hand_off.frame = None;
            }
        }
        // SAFETY: `open_frame` forgot the guard of this lock; this handle is
        // its only owner.
        unsafe { self.pool.scope_lock.force_unlock() };
        use crate::runtime::drain::{note_store_route_n, note_store_route_us};
        note_store_route_n("frame_pushed", self.pushed.get());
        note_store_route_n("frame_cancelled", left.len() as u64);
        note_store_route_n("frame_helped", fr.helped.load(Ordering::Relaxed));
        note_store_route_us("frame_busy_us", fr.busy_ns.load(Ordering::Relaxed) / 1000);
    }
}

thread_local! {
    /// The frame this helper serves, for jobs that queue a follow-up.
    static CURRENT_FRAME: std::cell::RefCell<Option<Arc<FrameShared>>> =
        const { std::cell::RefCell::new(None) };
}

/// TEMP (lab, FRAME_GATE): queue `f` on the frame the calling helper serves,
/// behind what is queued (nobody waits for it). `false` outside a frame job
/// or when the frame is closing: the caller does the work itself.
pub(crate) fn push_followup(f: Box<dyn FnOnce(&crate::model::DeviceState, &WorkerEnv<'_>) + Send>) -> bool {
    CURRENT_FRAME.with(|c| {
        let c = c.borrow();
        let Some(fr) = c.as_ref() else {
            return false;
        };
        if fr.closed.load(Ordering::Acquire) {
            return false;
        }
        let ticket: Ticket<()> = Ticket::new(Box::new(move |st, env| f(st, env)));
        fr.queue.lock().push_back(ticket.erased());
        fr.queued.fetch_add(1, Ordering::AcqRel);
        crate::runtime::drain::note_store_route("frame_followups");
        true
    })
}

/// A helper serving a frame until it closes.
fn serve_frame(frame: &Arc<FrameShared>, env: &WorkerEnv<'_>) {
    struct Current;
    impl Drop for Current {
        fn drop(&mut self) {
            CURRENT_FRAME.with(|c| *c.borrow_mut() = None);
        }
    }
    CURRENT_FRAME.with(|c| *c.borrow_mut() = Some(Arc::clone(frame)));
    let _current = Current;
    let fr: &FrameShared = frame;
    struct Out<'a>(&'a AtomicUsize);
    impl Drop for Out<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    fr.inside.fetch_add(1, Ordering::SeqCst);
    let _out = Out(&fr.inside);
    let mut idle_since: Option<Instant> = None;
    loop {
        if fr.closed.load(Ordering::SeqCst) {
            return;
        }
        // SAFETY: counted in `inside` and the frame not closed: the handle's
        // drop has not finished waiting, so the lock is alive.
        let lock = unsafe { &*fr.lock.0 };
        if fr.queued.load(Ordering::Acquire) > 0 {
            if let Some(rg) = lock.try_read() {
                let job = fr.queue.lock().pop_front();
                if let Some(job) = job {
                    fr.queued.fetch_sub(1, Ordering::AcqRel);
                    if job.claim() {
                        let t = Instant::now();
                        let _scope = super::gate::JobScope::enter(lock);
                        job.run(&rg, env);
                        fr.busy_ns
                            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        fr.helped.fetch_add(1, Ordering::Relaxed);
                    }
                    idle_since = None;
                    continue;
                }
            }
        }
        let t0 = *idle_since.get_or_insert_with(Instant::now);
        if t0.elapsed() < fr.spin {
            std::hint::spin_loop();
            continue;
        }
        let mut g = fr.park.lock();
        fr.parked.fetch_add(1, Ordering::AcqRel);
        if !fr.closed.load(Ordering::SeqCst) {
            fr.park_cv.wait_for(&mut g, Duration::from_millis(2));
        }
        fr.parked.fetch_sub(1, Ordering::AcqRel);
        idle_since = None;
    }
}
