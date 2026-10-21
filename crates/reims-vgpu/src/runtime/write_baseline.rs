//! TEMP (lab, `REIMS_VGPU_LAB_ASYNC_BASELINE=on`): a writer witness's synced
//! generation read taken on a thread of its own instead of the drain.
//!
//! # Why
//!
//! Arming a GVA Store's write witness ([`crate::runtime::gva_store_witness`])
//! reads the target set's generation synced, and for a target the witness has
//! not seen — every new allocation, which on full-screen transitions is a new
//! 5K surface each frame — that read is a whole dirty-bitmap query of the set
//! (~10 800 pages, 0.3-0.5 ms) on the drain thread. n137: `gva_store_witness::stamp`
//! 12.5 % of the drain on transitions; n141 (no foreground syncs at all, not
//! correct): cssscroll 78 -> 94/110 Hz.
//!
//! # Why a later read is the same read
//!
//! The Store is encoded inside the guest packet that rendered the target, and
//! the guest does not write a render target's pages with the CPU until it has
//! seen that packet's completion stamp (Metal: a resource is not CPU-written
//! while a command buffer using it is in flight). So any sync taken between the
//! Store and the publication of that stamp sees exactly the guest writes a sync
//! at the Store would have seen. This module keeps the second half of that
//! sentence true: **no completion stamp queued after a read was issued is
//! published before the read resolves** — the completion thread waits for it
//! ([`wait_resolved`] with the ticket a queued stamp recorded), and a stamp the
//! drain publishes itself joins first.
//!
//! The same argument answers a reader that asks while the read is still out:
//! the stamp is held, so the guest has not written the target since the Store
//! (`gva_store_witness::reach`).
//!
//! A stamp with nothing else to wait for — a deferred Store writes no guest
//! memory, so its packet's stamp is CPU-ready — is not joined on the drain
//! (n142 fab0: 2559 joins, the whole read time back on the drain): it becomes a
//! job of its own ([`publish_after`]) that this thread stores and announces
//! after the reads ahead of it, with a ticket of its own so every later stamp
//! of any FIFO is ordered after it.
//!
//! One thread, jobs in issue order, so "resolved" is a single watermark.
//!
//! TEMP (lab LAZY_SYNC, Opus 10-21): the same gate holds the stamps after a synchronize whose frame is still being
//! copied into the guest's pages on the landing thread ([`resolve_after`]) - the guest does not read a synchronized
//! resource before the command's completion, so no later stamp may say "done" before the copy is in. The thread
//! runs for that alone when `REIMS_VGPU_LAB_LAZY_SYNC_PX` is set; the deferred reads stay behind their own switch.

use crate::qemu::host_ops::ReimsVgpuHostOps;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::{Duration, Instant};

const UNRESOLVED: u64 = u64::MAX;

/// Reads allowed in flight before a Store reads synced on the drain again, so
/// a slow host bounds how long stamps can be held instead of growing a queue.
const MAX_OUTSTANDING: u64 = 8;

/// How long a waiter trusts the thread before reporting it and giving up: a
/// guest stamp held forever is a hung guest, a read answered "unarmed" is a
/// re-read.
const WAIT_LIMIT: Duration = Duration::from_secs(1);

/// The answer of one deferred read: the generation (0 = the host could not
/// answer), once the baseline thread has it.
#[derive(Clone, Debug)]
pub struct PendingGen(Arc<AtomicU64>);

impl PendingGen {
    /// The generation, or `None` while the read is still out.
    pub fn get(&self) -> Option<u64> {
        let v = self.0.load(Ordering::Acquire);
        (v != UNRESOLVED).then_some(v)
    }

    /// The generation, waiting for the read if it is still out. 0 if the thread
    /// does not answer within [`WAIT_LIMIT`].
    pub fn wait(&self) -> u64 {
        if let Some(v) = self.get() {
            return v;
        }
        let t0 = Instant::now();
        let mut g = GATE.lock();
        loop {
            if let Some(v) = self.get() {
                drop(g);
                note_wait("lab_baseline_join", t0.elapsed());
                return v;
            }
            if t0.elapsed() > WAIT_LIMIT {
                drop(g);
                stalled("pending_gen");
                return 0;
            }
            GATE_CV.wait_for(&mut g, Duration::from_millis(20));
        }
    }

    #[cfg(test)]
    pub(crate) fn resolved(gen_: u64) -> Self {
        Self(Arc::new(AtomicU64::new(gen_)))
    }

    #[cfg(test)]
    pub(crate) fn unresolved() -> (Self, Arc<AtomicU64>) {
        let slot = Arc::new(AtomicU64::new(UNRESOLVED));
        (Self(Arc::clone(&slot)), slot)
    }
}

enum Job {
    Read {
        token: u64,
        slot: Arc<AtomicU64>,
        ticket: u64,
    },
    /// TEMP (lab LAZY_SYNC): resolve `ticket` once `wait` returns (a landing copy is in).
    Wait {
        wait: Box<dyn FnOnce() + Send>,
        ticket: u64,
        queued_at: Instant,
    },
    Publish {
        gpa: u64,
        index: u32,
        stamp: u32,
        announce: fn(u32),
        ticket: u64,
        queued_at: Instant,
    },
}

struct Service {
    tx: Mutex<mpsc::Sender<Job>>,
}

static SERVICE: OnceLock<Service> = OnceLock::new();
/// Tickets handed out. Only the drain issues.
static ISSUED: AtomicU64 = AtomicU64::new(0);
/// Highest ticket answered. One thread answers, in ticket order.
static RESOLVED: AtomicU64 = AtomicU64::new(0);
static GATE: Mutex<()> = Mutex::new(());
static GATE_CV: Condvar = Condvar::new();

fn enabled() -> bool {
    reads_enabled() || std::env::var_os("REIMS_VGPU_LAB_LAZY_SYNC_PX").is_some()
}

/// The deferred generation reads ([`issue`]) - this module's own switch.
fn reads_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("REIMS_VGPU_LAB_ASYNC_BASELINE").as_deref(),
            Ok("on")
        )
    })
}

/// Start the baseline thread for the device whose shim table is `ops`, when
/// the switch is on and the shim can register a thread with RCU (the read goes
/// through the dirty tracker's RCU-protected tables). Idempotent.
pub fn install(ops: &ReimsVgpuHostOps) {
    if !enabled() || SERVICE.get().is_some() {
        return;
    }
    let Some(registrar) = ops.thread_registrar() else {
        crate::observe::off("lab_baseline off reason=no_thread_registrar".to_string());
        return;
    };
    let table = *ops;
    let (tx, rx) = mpsc::channel::<Job>();
    let spawned = std::thread::Builder::new()
        .name("reims-vgpu-baseline".into())
        .spawn(move || {
            let _attachment = registrar.attach();
            while let Ok(job) = rx.recv() {
                match job {
                    Job::Read {
                        token,
                        slot,
                        ticket,
                    } => {
                        let t0 = Instant::now();
                        let gen_ = table.write_gen(token).unwrap_or(0);
                        slot.store(gen_, Ordering::Release);
                        resolve(ticket);
                        note_wait("lab_baseline_read", t0.elapsed());
                        if gen_ == 0 {
                            crate::runtime::drain::census::note_store_route(
                                "lab_baseline_unarmed",
                            );
                        }
                    }
                    Job::Wait {
                        wait,
                        ticket,
                        queued_at,
                    } => {
                        wait();
                        resolve(ticket);
                        note_wait("lab_baseline_land", queued_at.elapsed());
                    }
                    Job::Publish {
                        gpa,
                        index,
                        stamp,
                        announce,
                        ticket,
                        queued_at,
                    } => {
                        // The word lands before the ticket resolves, so a stamp
                        // gated on this ticket can never be stored ahead of it.
                        // Through QEMU's address space, as the drain's inline
                        // publication does (this thread is RCU-registered).
                        if !write_word(&table, gpa, stamp) {
                            crate::observe::fail(format!(
                                "lab_baseline_publish_failed index={index} gpa={gpa:#x} \
                                 value={stamp:#x}"
                            ));
                        }
                        resolve(ticket);
                        announce(index);
                        note_wait("lab_baseline_publish", queued_at.elapsed());
                    }
                }
            }
        });
    match spawned {
        Ok(_) => {
            let _ = SERVICE.set(Service { tx: Mutex::new(tx) });
            crate::observe::off("lab_baseline on".to_string());
        }
        Err(e) => crate::observe::fail(format!("lab_baseline_spawn_failed err={e}")),
    }
}

fn write_word(table: &ReimsVgpuHostOps, gpa: u64, value: u32) -> bool {
    let Some(f) = table.write_gpa else {
        return false;
    };
    let bytes = value.to_le_bytes();
    // SAFETY: QEMU owns ctx for the device lifetime; the buffer outlives the call.
    unsafe { f(table.ctx, gpa, bytes.as_ptr(), bytes.len()) == 0 }
}

fn resolve(ticket: u64) {
    RESOLVED.fetch_max(ticket, Ordering::AcqRel);
    let _g = GATE.lock();
    GATE_CV.notify_all();
}

/// Hand a completion stamp with nothing but reads to wait for to the baseline
/// thread, which stores `stamp` at `gpa` and calls `announce(index)` once
/// every read issued before it has resolved. `false` when there is no thread —
/// the caller joins and publishes itself.
pub fn publish_after(
    gpa: u64,
    index: u32,
    stamp: u32,
    announce: fn(u32),
) -> bool {
    let Some(svc) = SERVICE.get() else {
        return false;
    };
    let ticket = ISSUED.load(Ordering::Acquire) + 1;
    ISSUED.store(ticket, Ordering::Release);
    let job = Job::Publish {
        gpa,
        index,
        stamp,
        announce,
        ticket,
        queued_at: Instant::now(),
    };
    if svc.tx.lock().send(job).is_err() {
        RESOLVED.fetch_max(ticket, Ordering::AcqRel);
        return false;
    }
    true
}

/// TEMP (lab LAZY_SYNC): gate every completion stamp queued from now on behind `wait` - run on the baseline
/// thread in ticket order. `false` when there is no thread: the caller must wait itself before the packet's stamp.
pub fn resolve_after(wait: Box<dyn FnOnce() + Send>) -> bool {
    let Some(svc) = SERVICE.get() else {
        return false;
    };
    // Issued before it is sent, so a stamp queued after this call is gated on it.
    let ticket = ISSUED.load(Ordering::Acquire) + 1;
    ISSUED.store(ticket, Ordering::Release);
    let job = Job::Wait {
        wait,
        ticket,
        queued_at: Instant::now(),
    };
    if let Err(mpsc::SendError(job)) = svc.tx.lock().send(job) {
        // The thread is gone: run the wait here, then let the ticket gate nothing.
        if let Job::Wait { wait, .. } = job {
            wait();
        }
        RESOLVED.fetch_max(ticket, Ordering::AcqRel);
    }
    true
}

/// Start a synced generation read of `token` on the baseline thread. `None`
/// when there is no thread or too many reads are already out — read synced.
pub fn issue(token: u64) -> Option<PendingGen> {
    if !reads_enabled() {
        return None;
    }
    let svc = SERVICE.get()?;
    let issued = ISSUED.load(Ordering::Acquire);
    if issued.saturating_sub(RESOLVED.load(Ordering::Acquire)) >= MAX_OUTSTANDING {
        crate::runtime::drain::census::note_store_route("lab_baseline_backpressure");
        return None;
    }
    let ticket = issued + 1;
    let slot = Arc::new(AtomicU64::new(UNRESOLVED));
    // Issued before it is sent, so a stamp queued after this call is gated on it.
    ISSUED.store(ticket, Ordering::Release);
    let job = Job::Read {
        token,
        slot: Arc::clone(&slot),
        ticket,
    };
    if svc.tx.lock().send(job).is_err() {
        // The thread is gone: nothing will answer this ticket, so it must not
        // gate anything. The caller reads synced.
        RESOLVED.fetch_max(ticket, Ordering::AcqRel);
        return None;
    }
    Some(PendingGen(slot))
}

/// The newest ticket handed out. A completion stamp records it when it is
/// queued and is not published before [`wait_resolved`] says it is answered.
pub fn issued() -> u64 {
    ISSUED.load(Ordering::Acquire)
}

/// Whether a read is still out.
pub fn outstanding() -> bool {
    RESOLVED.load(Ordering::Acquire) < ISSUED.load(Ordering::Acquire)
}

/// Wait until every read up to `ticket` is answered. Gives up after
/// [`WAIT_LIMIT`], reported, rather than holding a guest stamp forever.
pub fn wait_resolved(ticket: u64) {
    if ticket == 0 || RESOLVED.load(Ordering::Acquire) >= ticket {
        return;
    }
    let t0 = Instant::now();
    let mut g = GATE.lock();
    while RESOLVED.load(Ordering::Acquire) < ticket {
        if t0.elapsed() > WAIT_LIMIT {
            drop(g);
            stalled("stamp_gate");
            return;
        }
        GATE_CV.wait_for(&mut g, Duration::from_millis(20));
    }
    drop(g);
    note_wait("lab_baseline_gate", t0.elapsed());
}

fn stalled(site: &str) {
    crate::runtime::drain::census::note_store_route("lab_baseline_stalled");
    crate::observe::fail(format!(
        "lab_baseline_stalled site={site} issued={} resolved={} (the baseline thread has not \
         answered within {} ms; proceeding as unanswered)",
        ISSUED.load(Ordering::Acquire),
        RESOLVED.load(Ordering::Acquire),
        WAIT_LIMIT.as_millis()
    ));
}

/// Count and band one wait (or read) under `prefix`: `_n`, `_us` and a band.
fn note_wait(prefix: &'static str, elapsed: Duration) {
    use crate::runtime::drain::census::{note_store_route, note_store_route_n};
    let us = elapsed.as_micros() as u64;
    let band = match (prefix, us) {
        ("lab_baseline_read", 0..=99) => "lab_baseline_read_lt100us",
        ("lab_baseline_read", 100..=499) => "lab_baseline_read_lt500us",
        ("lab_baseline_read", 500..=1999) => "lab_baseline_read_lt2ms",
        ("lab_baseline_read", _) => "lab_baseline_read_ge2ms",
        ("lab_baseline_gate", 0..=99) => "lab_baseline_gate_lt100us",
        ("lab_baseline_gate", 100..=499) => "lab_baseline_gate_lt500us",
        ("lab_baseline_gate", 500..=1999) => "lab_baseline_gate_lt2ms",
        ("lab_baseline_gate", _) => "lab_baseline_gate_ge2ms",
        ("lab_baseline_publish", 0..=99) => "lab_baseline_publish_lt100us",
        ("lab_baseline_publish", 100..=499) => "lab_baseline_publish_lt500us",
        ("lab_baseline_publish", 500..=1999) => "lab_baseline_publish_lt2ms",
        ("lab_baseline_publish", _) => "lab_baseline_publish_ge2ms",
        ("lab_baseline_land", 0..=1999) => "lab_baseline_land_lt2ms",
        ("lab_baseline_land", 2000..=7999) => "lab_baseline_land_lt8ms",
        ("lab_baseline_land", _) => "lab_baseline_land_ge8ms",
        (_, 0..=99) => "lab_baseline_join_lt100us",
        (_, 100..=499) => "lab_baseline_join_lt500us",
        (_, 500..=1999) => "lab_baseline_join_lt2ms",
        _ => "lab_baseline_join_ge2ms",
    };
    note_store_route(band);
    note_store_route_n(
        match prefix {
            "lab_baseline_read" => "lab_baseline_read_us",
            "lab_baseline_gate" => "lab_baseline_gate_us",
            "lab_baseline_publish" => "lab_baseline_publish_us",
            "lab_baseline_land" => "lab_baseline_land_us",
            _ => "lab_baseline_join_us",
        },
        us,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pending_generation_reads_none_until_its_slot_is_answered() {
        let (pending, slot) = PendingGen::unresolved();
        assert_eq!(pending.get(), None);
        slot.store(7, Ordering::Release);
        assert_eq!(pending.get(), Some(7));
        assert_eq!(pending.wait(), 7);
        assert_eq!(PendingGen::resolved(0).get(), Some(0), "0 is an answer: unarmed");
    }

    #[test]
    fn nothing_issued_gates_nothing() {
        // No service in a test build: issue declines and the gate is open.
        assert!(issue(1).is_none());
        wait_resolved(0);
    }
}
