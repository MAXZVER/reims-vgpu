//! TEMP (lab, FRAME_GATE): the drain's device state behind a reader-writer
//! lock for one tranche, so pool helpers can prepare work across stream and
//! packet boundaries while the drain holds the state shared.
//!
//! Design: `lab/journal/2026-10-10-frame-region-design.md` section 5. The
//! drain owns the upgradable read side for the whole tranche. In its *shared*
//! phase (S) it reads the state through [`Gate::sh`] and helpers run jobs
//! under their own read guards; for a write it calls [`Gate::st`], which closes
//! the gate (no new job starts), waits for the running jobs and upgrades - the
//! *exclusive* phase (X). The exclusive phase is sticky: the drain stays in it
//! until it calls [`Gate::open`], so code that has not been taught where its
//! shared stretches are behaves exactly as before, only slower to reopen.
//!
//! Soundness rests on the lock alone (no raw pointer of ours derives a
//! reference to the state): `RwLock<&mut DeviceState>` is `Sync` because
//! `DeviceState` is, a helper's `&DeviceState` lives inside its read guard,
//! and the drain's `&mut` inside the write guard. The one erasure is the
//! frame's pointer to the [`FrameLock`] (see `pool.rs`), whose handle the gate
//! drops - waiting every helper out - before the lock.

use super::pool::{FrameHandle, Ticket};
use super::{JobPanic, WorkerEnv};
use crate::model::DeviceState;
use parking_lot::{RwLock, RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The device state of one tranche, lockable by the drain and the frame's
/// helpers.
pub struct FrameLock<'s> {
    lock: RwLock<&'s mut DeviceState>,
    /// The drain wants the exclusive phase: no new job starts, running ones
    /// may stop early at a poll point.
    closing: AtomicBool,
    /// Exclusive phases ended so far (a prepared job's stamp).
    xseq: AtomicU64,
}

impl<'s> FrameLock<'s> {
    pub fn new(state: &'s mut DeviceState) -> Self {
        FrameLock {
            lock: RwLock::new(state),
            closing: AtomicBool::new(false),
            xseq: AtomicU64::new(0),
        }
    }

    /// A helper's read guard, unless the drain is closing the gate or holds it
    /// exclusively.
    pub(super) fn try_read(&self) -> Option<RwLockReadGuard<'_, &'s mut DeviceState>> {
        if self.closing.load(Ordering::Acquire) {
            return None;
        }
        self.lock.try_read()
    }

    /// Whether the drain is waiting for the exclusive phase: a job polls this
    /// at its bail points and gives up early (its result is then not used).
    pub(crate) fn closing(&self) -> bool {
        self.closing.load(Ordering::Acquire)
    }

    pub(crate) fn xseq(&self) -> u64 {
        self.xseq.load(Ordering::Acquire)
    }
}

enum Held<'g, 's> {
    Shared(RwLockUpgradableReadGuard<'g, &'s mut DeviceState>),
    Excl(RwLockWriteGuard<'g, &'s mut DeviceState>),
    Moving,
}

/// The drain's side of a [`FrameLock`] for one tranche.
pub struct Gate<'g, 's> {
    // Dropped first: the frame waits every helper out before the guard goes.
    frame: Option<FrameHandle>,
    held: Held<'g, 's>,
    lock: &'g FrameLock<'s>,
    opened_at: Option<Instant>,
    stats: GateStats,
}

#[derive(Default)]
struct GateStats {
    x_entries: u64,
    x_wait_ns: u64,
    s_ns: u64,
    opens: u64,
}

impl<'g, 's> Gate<'g, 's> {
    /// The drain's gate over `lock`, in the exclusive phase. With a frame the
    /// pool's helpers serve this gate's jobs until it drops.
    pub fn new(lock: &'g FrameLock<'s>, frame: Option<FrameHandle>) -> Self {
        let guard = lock.lock.upgradable_read();
        let excl = RwLockUpgradableReadGuard::upgrade(guard);
        Gate {
            frame,
            held: Held::Excl(excl),
            lock,
            opened_at: None,
            stats: GateStats::default(),
        }
    }

    /// The state, shared. Valid in either phase.
    #[inline]
    pub fn sh(&self) -> &DeviceState {
        match &self.held {
            Held::Shared(g) => g,
            Held::Excl(g) => g,
            Held::Moving => unreachable!("gate held across a phase change"),
        }
    }

    /// The state, exclusive: closes the gate if it is open (sticky until
    /// [`Self::open`]).
    #[inline]
    pub fn st(&mut self) -> &mut DeviceState {
        if matches!(self.held, Held::Shared(_)) {
            self.close();
        }
        match &mut self.held {
            Held::Excl(g) => g,
            _ => unreachable!("gate not exclusive after close"),
        }
    }

    /// Whether helpers may run now.
    pub fn is_open(&self) -> bool {
        matches!(self.held, Held::Shared(_))
    }

    /// Enter the shared phase: helpers run queued jobs until the next
    /// [`Self::st`].
    pub fn open(&mut self) {
        let Held::Excl(_) = self.held else {
            return;
        };
        let Held::Excl(w) = std::mem::replace(&mut self.held, Held::Moving) else {
            unreachable!()
        };
        self.lock.xseq.fetch_add(1, Ordering::AcqRel);
        self.held = Held::Shared(RwLockWriteGuard::downgrade_to_upgradable(w));
        self.stats.opens += 1;
        self.opened_at = Some(Instant::now());
        if let Some(frame) = self.frame.as_ref() {
            frame.wake();
        }
    }

    fn close(&mut self) {
        let Held::Shared(up) = std::mem::replace(&mut self.held, Held::Moving) else {
            unreachable!()
        };
        let t0 = Instant::now();
        if let Some(at) = self.opened_at.take() {
            self.stats.s_ns += t0.duration_since(at).as_nanos() as u64;
        }
        self.lock.closing.store(true, Ordering::SeqCst);
        let mut up = up;
        let spin_until = t0 + Duration::from_micros(200);
        let w = loop {
            match RwLockUpgradableReadGuard::try_upgrade(up) {
                Ok(w) => break w,
                Err(back) => up = back,
            }
            if Instant::now() >= spin_until {
                break RwLockUpgradableReadGuard::upgrade(up);
            }
            std::hint::spin_loop();
        };
        self.lock.closing.store(false, Ordering::Release);
        self.held = Held::Excl(w);
        self.stats.x_entries += 1;
        self.stats.x_wait_ns += t0.elapsed().as_nanos() as u64;
    }

    /// Whether this gate has helpers serving it.
    pub fn has_frame(&self) -> bool {
        self.frame.is_some()
    }

    /// Exclusive phases ended so far.
    pub fn xseq(&self) -> u64 {
        self.lock.xseq()
    }

    /// Queue `job` for the frame's helpers. Without a frame the job waits for
    /// the drain's own [`Self::wait`].
    pub fn push<T, F>(&self, job: F) -> Ticket<T>
    where
        T: Send + 'static,
        F: FnOnce(&DeviceState, &WorkerEnv<'_>) -> T + Send + 'static,
    {
        let ticket = Ticket::new(Box::new(job));
        if let Some(frame) = self.frame.as_ref() {
            frame.push(ticket.erased(), self.is_open());
        }
        ticket
    }

    /// `ticket`'s result: at once when it is done, by running it here when no
    /// helper has started it, else after the helper that has. `None`: the job
    /// was cancelled.
    pub fn wait<T: Send + 'static>(&self, ticket: &Ticket<T>) -> Option<Result<T, JobPanic>> {
        let ops = self.frame.as_ref().map(FrameHandle::ops);
        ticket.wait_or_run(self.sh(), ops)
    }

    /// The result if a helper has finished the job, without waiting.
    pub fn try_take<T: Send + 'static>(&self, ticket: &Ticket<T>) -> Option<Result<T, JobPanic>> {
        ticket.try_take()
    }
}

impl Drop for Gate<'_, '_> {
    fn drop(&mut self) {
        if let Some(at) = self.opened_at.take() {
            self.stats.s_ns += at.elapsed().as_nanos() as u64;
        }
        // The frame first (field order), explicitly so the census below sees
        // its counters.
        if let Some(frame) = self.frame.take() {
            drop(frame);
        }
        if self.stats.opens > 0 || self.stats.x_entries > 0 {
            use crate::runtime::drain::{note_store_route_n, note_store_route_us};
            note_store_route_n("gate_opens", self.stats.opens);
            note_store_route_n("gate_x_entries", self.stats.x_entries);
            note_store_route_us("gate_x_wait_us", self.stats.x_wait_ns / 1000);
            note_store_route_us("gate_s_us", self.stats.s_ns / 1000);
        }
    }
}

/// TEMP (lab, live `FRAME_GATE=on`, else `REIMS_VGPU_LAB_FRAME_GATE=on`;
/// default off): a tranche opens a frame on the pool.
pub(crate) fn frame_gate_on() -> bool {
    if cfg!(test) {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("FRAME_GATE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_FRAME_GATE").as_deref() == Ok("on"))
}

thread_local! {
    /// The frame lock a helper's current job runs under, for its poll points.
    static CLOSING: std::cell::Cell<*const AtomicBool> = const { std::cell::Cell::new(std::ptr::null()) };
}

/// Whether the drain is closing the gate the calling job runs under. `false`
/// outside a frame job.
#[inline]
pub(crate) fn job_should_stop() -> bool {
    CLOSING.with(|c| {
        let p = c.get();
        // SAFETY: set only for the length of a frame job by `JobScope`, while
        // the job's read guard keeps the frame lock alive.
        !p.is_null() && unsafe { (*p).load(Ordering::Relaxed) }
    })
}

/// Marks the calling thread as running a frame job of `lock` until dropped.
pub(super) struct JobScope {
    was: *const AtomicBool,
}

impl JobScope {
    pub(super) fn enter(lock: &FrameLock<'_>) -> JobScope {
        let was = CLOSING.with(|c| c.replace(&lock.closing as *const AtomicBool));
        JobScope { was }
    }
}

impl Drop for JobScope {
    fn drop(&mut self) {
        CLOSING.with(|c| c.set(self.was));
    }
}
