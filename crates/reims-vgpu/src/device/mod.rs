//! The process-wide registry of bound devices, and the entry surface
//! `crate::qemu::abi` wraps.
//!
//! One `BoundDevice` per QEMU device, keyed by the opaque handle the C shim
//! holds, plus the twelve functions that shim calls: create/reset/destroy, the
//! two MMIO windows, drain, poll, the action queue, and the backend name. The
//! locking policy lives here too, because it is a property of this map rather
//! than of any one device — `lock_for_drain` and `lock_device_for_vcpu` are two
//! answers to "who may block whom" and are chosen per caller. (Named in prose:
//! a bare name in a `//!` doc resolves to nothing whatever it names.)
//!
//! Split out of the crate root, which was 800 lines of which 690 were this. The
//! root is the module table and the three-arm guards now, and the two sibling
//! entry-surface modules — `display_surface` and `window_publish` — already had
//! this shape: a private `mod`, with the names `crate::qemu::abi` reaches re-exported
//! at the root.

/// The device side of that window: its link, its four QEMU entry points, and
/// the two publish paths the drain and the poll call. Unconditional, because
/// the entry points keep a stub arm without the feature and the QEMU ABI
/// surface must be the same shape either way.
mod window_publish;
pub(crate) use window_publish::{
    device_window_run_main, device_window_set_early_fb, device_window_start, device_window_stop,
};

/// The display half of the QEMU ABI surface: console-feed ownership, scanout
/// and EFI-console copies, and the cursor glyph.
mod display_surface;
#[cfg(any(test, feature = "host-window"))]
pub(crate) use display_surface::host_console_uses_bar1;
pub(crate) use display_surface::{
    device_console_feed, device_cursor_glyph_copy, device_cursor_glyph_info,
    device_efi_console_copy, device_scanout_copy, device_scanout_may_paint, ConsoleFeed,
    CursorGlyphInfo,
};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::qemu::host_ops::{NullHost, QemuHost, ReimsVgpuHostOps};
// The four names the two chapter modules below reach through `use super::*`,
// and this module uses itself. They were the crate root's "convenience
// re-exports used by qemu ABI and tests" and came with the registry.
use crate::backend::Backend as _;
use crate::model::{Device, DeviceId};
use crate::runtime::{HostAction, HostOps};

/// Mutable protocol/backend state. The drain worker may hold this lock across
/// shader translation and a GPU wait, so MMIO producers must never wait for it.
struct DeviceInner {
    device: Device,
    /// Actions for the QEMU BH to apply after drain.
    actions: VecDeque<HostAction>,
}

#[derive(Clone, Copy, Debug)]
struct QueuedGfxWrite {
    offset: u64,
    data: u64,
    size: u32,
    /// When the vCPU published this write, or `None` when it was applied
    /// straight through without ever entering the queue.
    ///
    /// The guest's store retires the moment this is pushed, so the guest cannot
    /// see the delay; the age measured against this stamp is the only place the
    /// deferral becomes visible. See [`crate::runtime::drain::DoorbellCensus`].
    queued_at: Option<std::time::Instant>,
}

/// One live device. Registry lookup and MMIO ingress remain short even while
/// `inner` is owned by the ordered render worker.
struct BoundDevice {
    inner: Mutex<DeviceInner>,
    gfx_ingress: Mutex<VecDeque<QueuedGfxWrite>>,
    gfx_read_cache: Mutex<HashMap<(u64, u32), u64>>,
    gfx_read_busy_logged: AtomicBool,
    /// Prompt HostActions (IRQ pulses, cursor moves): poppable without `inner`
    /// so the BH delivers them while the drain worker still owns the device
    /// lock. Scanout/glyph actions stay in `DeviceInner::actions`.
    prompt_actions: Mutex<VecDeque<HostAction>>,
    /// Lock-free clones of the read-to-clear interrupt-status registers
    /// (`state.gfx.interrupt_status_disp` / `_gpu`): the guest ISR read at
    /// 0x1014/0x1018 must observe live bits mid-drain, never a stale cache.
    intr_disp: Arc<AtomicU32>,
    intr_gpu: Arc<AtomicU32>,
    /// Child channels the guest has rung, OR'd from the vCPU thread with no
    /// device lock; see [`crate::model::GfxRegs::child_doorbell_rung`].
    child_doorbell_rung: Arc<AtomicU32>,
    /// Lock-free clone of the fault status (0x102c) — the ISR's third read.
    intr_fault: Arc<AtomicU32>,
    /// Lock-free clone of the main-FIFO consumer counter (0x100c): the guest
    /// `writeFifo` producer spin must observe drain progress live, not a
    /// cached snapshot from before the tranche.
    fifo_read_live: Arc<AtomicU32>,
    /// An accepted present is waiting for QEMU to consume its scanout action.
    /// Kept outside `inner` so new worker wakeups can yield without racing the
    /// main-loop copy for the same device lock.
    present_action_pending: AtomicBool,
    /// Monotonic per-boot publication of `frame_flush_seen`. QEMU refresh must
    /// never reinterpret a contended device lock as a return to BAR1/EFI.
    present_boundary_seen: AtomicBool,
    /// Monotonic reset sequence for cross-boot lifecycle diagnosis.
    reset_count: AtomicU64,
    /// Lock-free snapshot of the display VBL state, republished on every
    /// lock-acquired `device_poll`. Lets a *contended* poll (the drain worker
    /// owns `inner`) still pulse VBL so the guest keeps its display time base
    /// under load — without it, `device_poll` early-returns on the `try_lock`
    /// miss and drops the VBL entirely (kb present-thrash-proxies: VBL collapses
    /// to ~7 Hz under interaction). `vbl_shared_gpa == 0` ⇒ not online yet.
    vbl_shared_gpa: AtomicU64,
    vbl_display_index: AtomicU32,
    vbl_online: AtomicBool,
    /// Guest page size, published with the rest of the lock-free VBL snapshot.
    /// The refresh tick's pending write is page-bounded like every other write
    /// into the shared page, and this arm has no `DeviceState` to ask.
    vbl_page_size: AtomicU64,
    /// Wall-clock ms of the last VBL claimed by either the locked or contended
    /// poll path. One shared limiter keeps guest pacing independent of which
    /// path happens to win the device lock.
    vbl_last_us: AtomicU64,
    /// TEMP (lab, POLL_YIELD): the drain worker is about to take, or holds,
    /// `inner`. The heartbeat poll then leaves `inner` alone (see
    /// [`device_poll`]).
    drain_wants: std::sync::atomic::AtomicBool,
    /// QEMU HostOps (GPA / clock / schedule worker). None in pure unit tests.
    ops: Option<ReimsVgpuHostOps>,
    /// The device's parallel worker pool, when its shim can register worker
    /// threads with RCU. The same pool `DeviceState::par` hands the draw rails;
    /// held here as well so `device_destroy` can stop and join its threads
    /// without taking the device lock.
    par: Option<Arc<crate::runtime::par::Pool>>,
    /// Host-owned presentation window ([[host-window]]), once
    /// `device_window_start` has spawned it. `None` on a normal QEMU-display
    /// boot (the window is opt-in behind `REIMS_VGPU_WINDOW`).
    #[cfg(feature = "host-window")]
    window: Mutex<Option<window_publish::WindowLink>>,
    /// Early-boot framebuffer (BAR1 GOP) registered by the C shim, shown in the
    /// window until the product present path latches.
    #[cfg(feature = "host-window")]
    early_fb: Mutex<Option<window_publish::EarlyFb>>,
    /// Monotonic ns of the last early frame pushed to the window (poll-path
    /// throttle so the pre-boundary pump does not memcpy the FB every 4 ms).
    #[cfg(feature = "host-window")]
    early_last_ns: AtomicU64,
}

type DeviceMap = HashMap<u64, Arc<BoundDevice>>;

static DEVICES: Lazy<Mutex<DeviceMap>> = Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_ID: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(1));

fn device_slot(id: u64) -> Option<Arc<BoundDevice>> {
    DEVICES.lock().get(&id).cloned()
}

fn schedule_device(slot: &BoundDevice) {
    let Some(ops) = slot.ops else {
        return;
    };
    if let Some(schedule) = ops.schedule_bh {
        // SAFETY: QEMU owns ctx for the device lifetime; schedule_bh is the
        // thread-safe wake callback supplied by the shim.
        unsafe { schedule(ops.ctx) }
    }
}

#[inline]
fn publish_present_boundary(slot: &BoundDevice, frame_flush_seen: bool) {
    if frame_flush_seen {
        slot.present_boundary_seen.store(true, Ordering::Release);
    }
}

fn apply_gfx_write(inner: &mut DeviceInner, slot: &BoundDevice, write: QueuedGfxWrite) {
    match write.queued_at {
        Some(at) => crate::runtime::drain::note_doorbell_queued(
            write.offset,
            at.elapsed().as_micros() as u64,
        ),
        None => crate::runtime::drain::note_doorbell_direct(),
    }
    if let Some(ops) = slot.ops {
        let mut host = QemuHost::new(&ops, &mut inner.actions, &slot.prompt_actions);
        inner
            .device
            .gfx_write(&mut host, write.offset, write.data, write.size);
    } else {
        let mut host = NullHost;
        inner
            .device
            .gfx_write(&mut host, write.offset, write.data, write.size);
    }
}

/// TEMP (lab): which site last took `inner` (0 = the drain), and the drain's waits
/// for it by that holder: 1 gfx read, 2 MMIO write, 3 vCPU iosfc, 4 heartbeat poll,
/// 5 action pop, 6 console feed, 7 display surface, 8 scanout copy, 9 cursor info,
/// 10 cursor copy.
pub(crate) static LAB_INNER_HOLDER: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
const LAB_HOLDERS: usize = 11;
static LAB_LOCK_WAIT_US: [std::sync::atomic::AtomicU64; LAB_HOLDERS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; LAB_HOLDERS];
static LAB_LOCK_WAIT_N: [std::sync::atomic::AtomicU64; LAB_HOLDERS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; LAB_HOLDERS];
static LAB_LOCK_INGRESS_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAB_LOCK_INGRESS_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAB_LOCK_LAST_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAB_LOCK_APPLY_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAB_LOCK_TOTAL_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAB_LOCK_TOTAL_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// TEMP (lab): mark the site that now holds `inner`.
#[inline]
pub(crate) fn lab_held(site: u8) {
    LAB_INNER_HOLDER.store(site, Ordering::Relaxed);
}

/// TEMP (lab): one contended drain lock; a line a second while any wait happened.
fn lab_drain_lock_note(holder: usize, us: u64) {
    let h = holder.min(LAB_HOLDERS - 1);
    LAB_LOCK_WAIT_US[h].fetch_add(us, Ordering::Relaxed);
    LAB_LOCK_WAIT_N[h].fetch_add(1, Ordering::Relaxed);
    let now = crate::observe::elapsed_ms() as u64;
    let last = LAB_LOCK_LAST_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 1000
        || LAB_LOCK_LAST_MS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    let mut line = String::from("lab_drain_lock_wait");
    for i in 0..LAB_HOLDERS {
        let n = LAB_LOCK_WAIT_N[i].swap(0, Ordering::Relaxed);
        let w = LAB_LOCK_WAIT_US[i].swap(0, Ordering::Relaxed);
        if n > 0 {
            line.push_str(&format!(" h{i}={n}/{w}us"));
        }
    }
    let n = LAB_LOCK_INGRESS_N.swap(0, Ordering::Relaxed);
    let w = LAB_LOCK_INGRESS_US.swap(0, Ordering::Relaxed);
    line.push_str(&format!(" ingress={n}/{w}us"));
    line.push_str(&format!(
        " total={}/{}us apply={}us",
        LAB_LOCK_TOTAL_N.swap(0, Ordering::Relaxed),
        LAB_LOCK_TOTAL_US.swap(0, Ordering::Relaxed),
        LAB_LOCK_APPLY_US.swap(0, Ordering::Relaxed)
    ));
    crate::observe::off(line);
}

/// Apply queued MMIO writes in publication order. Lock order is ingress then
/// inner everywhere; producers use `try_lock` for inner and therefore never
/// wait behind shader translation/GPU work.
fn lock_for_drain(slot: &BoundDevice) -> DrainGuard<'_> {
    // TEMP (lab): the uncontended path's clock reads feed only the LAB_LOCK_*
    // census, so they follow the phase timers (off in the working launcher's
    // ROUTES=off mode; b101: ~4.7 % of the drain's busy samples on the Finder
    // toggles were these reads).
    let lab_timed = crate::lab_clock::timers_on();
    let lab_t0 = lab_timed.then(std::time::Instant::now); // TEMP (lab)
    let lab_us = |t: Option<std::time::Instant>| t.map_or(0, |t| t.elapsed().as_micros() as u64);
    // TEMP (lab, POLL_YIELD): announce the take before trying it, so a poll
    // that has not yet taken `inner` stands aside.
    slot.drain_wants.store(true, Ordering::SeqCst);
    let mut ingress = slot.gfx_ingress.lock();
    let lab_ingress_us = lab_us(lab_t0);
    let mut inner = match slot.inner.try_lock() {
        Some(g) => g,
        None => {
            // TEMP (lab): who held it, and for how long the drain waited.
            let holder = LAB_INNER_HOLDER.load(Ordering::Relaxed);
            let t1 = std::time::Instant::now();
            // TEMP (lab, POLL_SPIN, b042): the heartbeat poll holds `inner`
            // for microseconds (b041: ~4 us a tick, at most ~0.6 ms), yet a
            // collision cost the drain ~200 us - the park and the wake, not the
            // hold. Spin on `try_lock` for up to POLL_SPIN_US before parking
            // when the poll is the holder.
            let spun = (holder == 4)
                .then(|| {
                    let budget = std::time::Duration::from_micros(lab_poll_spin_us());
                    loop {
                        if let Some(g) = slot.inner.try_lock() {
                            break Some(g);
                        }
                        if t1.elapsed() >= budget {
                            break None;
                        }
                        std::hint::spin_loop();
                    }
                })
                .flatten();
            let g = match spun {
                Some(g) => {
                    crate::runtime::drain::note_store_route("lab_drain_lock_spun");
                    g
                }
                None => slot.inner.lock(),
            };
            let waited = t1.elapsed().as_micros() as u64;
            lab_drain_lock_note(holder as usize, waited);
            // TEMP (lab, b044): the holder the drain finally took the lock from,
            // when it is not the one it found - a waiter that jumped in between.
            let last = LAB_INNER_HOLDER.load(Ordering::Relaxed);
            if last != holder {
                lab_drain_lock_handoff(holder, last, waited);
            }
            g
        }
    };
    LAB_INNER_HOLDER.store(0, Ordering::Relaxed);
    if lab_ingress_us >= 20 {
        LAB_LOCK_INGRESS_US.fetch_add(lab_ingress_us, Ordering::Relaxed);
        LAB_LOCK_INGRESS_N.fetch_add(1, Ordering::Relaxed);
    }
    let lab_t2 = lab_timed.then(std::time::Instant::now);
    while let Some(write) = ingress.pop_front() {
        apply_gfx_write(&mut inner, slot, write);
    }
    drop(ingress);
    if lab_timed {
        LAB_LOCK_APPLY_US.fetch_add(lab_us(lab_t2), Ordering::Relaxed);
        LAB_LOCK_TOTAL_US.fetch_add(lab_us(lab_t0), Ordering::Relaxed);
        LAB_LOCK_TOTAL_N.fetch_add(1, Ordering::Relaxed);
    }
    // Here rather than only inside `drain_pending`, because this is the one
    // point every entry to the drain passes through. `device_drain` returns
    // before `drain_pending` when the device has no host ops, and
    // `publish_stranded_fifos` re-publishes from the open-domain set — a ring
    // left unfolded would be invisible to both.
    crate::runtime::drain::fold_rung_child_doorbells(&mut inner.device.state);
    DrainGuard {
        inner: Some(inner),
        slot,
    }
}

/// TEMP (lab, POLL_YIELD): the drain's hold on `inner`, which clears
/// [`BoundDevice::drain_wants`] as it lets go so the heartbeat poll takes the
/// device again only while the drain is out.
struct DrainGuard<'a> {
    inner: Option<parking_lot::MutexGuard<'a, DeviceInner>>,
    slot: &'a BoundDevice,
}

impl std::ops::Deref for DrainGuard<'_> {
    type Target = DeviceInner;
    fn deref(&self) -> &DeviceInner {
        self.inner.as_ref().expect("DrainGuard holds inner until drop")
    }
}

impl std::ops::DerefMut for DrainGuard<'_> {
    fn deref_mut(&mut self) -> &mut DeviceInner {
        self.inner.as_mut().expect("DrainGuard holds inner until drop")
    }
}

impl Drop for DrainGuard<'_> {
    fn drop(&mut self) {
        // Release first, then clear the flag, then wake the BH: a BH woken
        // while `inner` is still held would find it taken and stop with the
        // lock-owning actions still queued.
        let pending = self.inner.as_ref().is_some_and(|g| !g.actions.is_empty());
        drop(self.inner.take());
        self.slot.drain_wants.store(false, Ordering::SeqCst);
        // The action BH, not the drain worker: `schedule_bh` wakes the drain
        // (b045's first cut called it here and re-ran a needless tranche per
        // release with actions queued, leaving the actions undelivered).
        if pending && lab_poll_yield() {
            if let Some(ops) = self.slot.ops {
                if let Some(notify) = ops.notify_actions {
                    // SAFETY: QEMU owns ctx for the device lifetime;
                    // notify_actions is the thread-safe action-BH wake the
                    // prompt rail uses.
                    unsafe { notify(ops.ctx) }
                }
            }
        }
    }
}

/// Create a device. `ops` is the QEMU host-service table (nullable for tests).
///
/// `page_shift` must be [`crate::model::PAGE_SHIFT_X86`] (12) or [`crate::model::PAGE_SHIFT_ARM64E`] (14).
/// There is no default (including no `0` → arm); unsupported values return `None`.
pub fn device_create(ops: Option<ReimsVgpuHostOps>, page_shift: u32) -> Option<u64> {
    use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
    if page_shift != PAGE_SHIFT_ARM64E && page_shift != PAGE_SHIFT_X86 {
        return None;
    }
    let mut id_guard = NEXT_ID.lock();
    let id = *id_guard;
    *id_guard = id.saturating_add(1);
    drop(id_guard);
    // Resolve the process's backend here rather than at first draw: on the
    // Metal rail this is what brings up the `MTLDevice`, and a host with none
    // says so on the fail channel at create, in front of the guest work that
    // would otherwise be the first to notice.
    crate::observe::off(format!("backend_selected name={}", backend_name()));
    let mut dev = Device::new(DeviceId(id), page_shift);
    // Built cold: no thread exists until a scope wants one, so a device whose
    // switches are all off runs exactly as before. `None` for a shim that offers
    // no RCU registration for worker threads (the sysbus one), which is how the
    // pool stays off where nobody verified a second guest-memory reader.
    let par = ops
        .as_ref()
        .and_then(|ops| {
            crate::runtime::par::Pool::new(ops, &crate::runtime::par::Settings::from_env())
        })
        .map(Arc::new);
    if let Some(pool) = &par {
        dev.state.install_par_pool(Arc::clone(pool));
    }
    if let Some(ops) = ops.as_ref() {
        crate::runtime::write_baseline::install(ops);
        // TEMP (lab, SKIP_GUARD): the consumption-side re-check of a gather
        // vouch needs a generation read the engine cannot reach through its
        // recorded request; the shim table installed here is that read.
        crate::runtime::gather_witness::install_skip_guard(ops);
    }
    let intr_disp = Arc::clone(&dev.state.gfx.interrupt_status_disp);
    let intr_gpu = Arc::clone(&dev.state.gfx.interrupt_status_gpu);
    let child_doorbell_rung = Arc::clone(&dev.state.gfx.child_doorbell_rung);
    let intr_fault = Arc::clone(&dev.state.gfx.interrupt_fault);
    let fifo_read_live = Arc::clone(&dev.state.gfx.fifo_read);
    DEVICES.lock().insert(
        id,
        Arc::new(BoundDevice {
            inner: Mutex::new(DeviceInner {
                device: dev,
                actions: VecDeque::new(),
            }),
            gfx_ingress: Mutex::new(VecDeque::new()),
            gfx_read_cache: Mutex::new(HashMap::new()),
            gfx_read_busy_logged: AtomicBool::new(false),
            prompt_actions: Mutex::new(VecDeque::new()),
            intr_disp,
            intr_gpu,
            child_doorbell_rung,
            intr_fault,
            fifo_read_live,
            present_action_pending: AtomicBool::new(false),
            present_boundary_seen: AtomicBool::new(false),
            reset_count: AtomicU64::new(0),
            vbl_shared_gpa: AtomicU64::new(0),
            vbl_display_index: AtomicU32::new(0),
            vbl_online: AtomicBool::new(false),
            vbl_page_size: AtomicU64::new(0),
            vbl_last_us: AtomicU64::new(0),
            drain_wants: std::sync::atomic::AtomicBool::new(false),
            ops,
            par,
            #[cfg(feature = "host-window")]
            window: Mutex::new(None),
            #[cfg(feature = "host-window")]
            early_fb: Mutex::new(None),
            #[cfg(feature = "host-window")]
            early_last_ns: AtomicU64::new(0),
        }),
    );
    // The completion thread's way back to the guest. Installed here rather than
    // built into the rail because a rail must not know what a `BoundDevice` is,
    // and looked up by id rather than captured by `Arc` so a stale hook cannot
    // keep a torn-down device alive. Offered to whichever rail is running: one
    // with no completion thread has nobody to announce from and drops it.
    crate::backend::selected().install_stamp_announce(std::sync::Arc::new(move |index: u32| {
        announce_stamp_interrupt(id, index)
    }));
    Some(id)
}

/// Raise the gfx interrupt for a stamp whose word the GPU has written.
///
/// Runs on the engine's completion thread, so it may touch nothing that needs
/// the device `inner` lock — the drain worker holds it for most of a busy
/// second and blocking here would put the guest's wakeup behind exactly the work
/// it is waiting to be told about.
///
/// It does not have to. This is the same three-step the display VBL already
/// takes from a contended poll ([`vbl_contended_pulse`]): OR the bit into the
/// lock-free `Arc<AtomicU32>` clone of the interrupt-status register that the
/// guest ISR reads at 0x1018, push the pulse onto `prompt_actions` — which
/// `device_pop_action` drains without the device lock — and let `enqueue`'s
/// prompt rail call the thread-safe `notify_actions`. The stamp *word* is
/// already in guest memory by construction: the submission that signalled this
/// thread's timeline value wrote it.
fn announce_stamp_interrupt(id: u64, index: u32) {
    let Some(slot) = device_slot(id) else {
        // The device is gone, so there is no interrupt-status register to set
        // and nobody to interrupt. Quiet: this is teardown, not a loss.
        return;
    };
    slot.intr_gpu
        .fetch_or(1u32 << (index & 0x1f), Ordering::AcqRel);
    let Some(ops) = slot.ops else {
        return;
    };
    let mut scratch = VecDeque::new();
    QemuHost::new(&ops, &mut scratch, &slot.prompt_actions).enqueue(HostAction::irq_gfx());
    crate::runtime::drain::lab_rewake::note(index);
}

/// TEMP (lab, `REIMS_VGPU_LAB_STAMP_REWAKE_MS`): the heartbeat's re-raise of
/// recently published stamp slots; see [`crate::runtime::drain::lab_rewake`].
/// Lock-free like [`announce_stamp_interrupt`], so a drain holding `inner` does
/// not delay it.
fn lab_stamp_rewake_tick(slot: &BoundDevice) {
    let mask = crate::runtime::drain::lab_rewake::due(crate::observe::elapsed_us());
    if mask == 0 {
        return;
    }
    let Some(ops) = slot.ops else {
        return;
    };
    slot.intr_gpu.fetch_or(mask, Ordering::AcqRel);
    let mut scratch = VecDeque::new();
    QemuHost::new(&ops, &mut scratch, &slot.prompt_actions).enqueue(HostAction::irq_gfx());
    crate::runtime::drain::census::note_store_route("lab_stamp_rewake");
}

pub fn device_reset(id: u64) -> bool {
    if let Some(slot) = device_slot(id) {
        let mut d = lock_for_drain(&slot);
        let seq = slot.reset_count.fetch_add(1, Ordering::Relaxed) + 1;
        let state = &d.device.state;
        let mappings = state.mappings.len();
        let tasks = state.tasks.live_count();
        let host_surfaces = state.host_surfaces.len();
        let host_textures = state.host_texture_surfaces.len();
        let host_gvas = state.host_gva_surfaces.len();
        let host_linear = state.host_linear_textures.len();
        let frame_valid = state.present.frame_valid;
        let frame_mapping = state.present.frame_mapping;
        let boundary = state.present.frame_flush_seen;
        let views = if let Some(ops) = slot.ops {
            let DeviceInner { device, actions } = &mut *d;
            let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
            device.reset_with_host(&mut host)
        } else {
            d.device.reset();
            0
        };
        crate::observe::off(format!(
            "device_reset id={id} seq={seq} mappings={mappings} tasks={tasks} host_surface={host_surfaces} host_texture={host_textures} host_gva={host_gvas} host_linear={host_linear} frame_valid={} frame_mapping={frame_mapping} boundary={} unmapped_views={views}",
            u8::from(frame_valid),
            u8::from(boundary)
        ));
        d.actions.clear();
        slot.prompt_actions.lock().clear();
        slot.gfx_read_cache.lock().clear();
        slot.present_action_pending.store(false, Ordering::Release);
        slot.present_boundary_seen.store(false, Ordering::Release);
        crate::runtime::census::present_proxy::reset_for_device();
        true
    } else {
        false
    }
}

pub fn device_destroy(id: u64) -> bool {
    let removed = DEVICES.lock().remove(&id);
    // Stop and join the worker pool here, on the destroy path, rather than
    // leaving it to the last `Arc<BoundDevice>` to drop: the shim calls this
    // before it unrealizes the device, and a worker registered with RCU must be
    // gone by then. Idempotent, and the drain thread has already been joined.
    if let Some(pool) = removed.as_ref().and_then(|slot| slot.par.as_ref()) {
        pool.shutdown();
    }
    removed.is_some()
}

pub fn device_gfx_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    use crate::model::{
        GFX_REG_FIFO_READ, GFX_REG_INTR_FAULT, GFX_REG_INTR_STATUS_DISP, GFX_REG_INTR_STATUS_GPU,
    };
    let slot = device_slot(id)?;
    // Guest spin/ISR registers: served lock-free from the shared atomics so a
    // drain-tranche-held device lock never turns a fresh stamp signal into a
    // stale cached mask (0x1014/0x1018 r2c) nor hides drain progress from the
    // writeFifo producer spin (0x100c) or the ISR fault read (0x102c).
    if size == 4 {
        if offset == GFX_REG_INTR_STATUS_DISP {
            return Some(slot.intr_disp.swap(0, Ordering::AcqRel) as u64);
        }
        if offset == GFX_REG_INTR_STATUS_GPU {
            return Some(slot.intr_gpu.swap(0, Ordering::AcqRel) as u64);
        }
        if offset == GFX_REG_FIFO_READ {
            return Some(slot.fifo_read_live.load(Ordering::Acquire) as u64);
        }
        if offset == GFX_REG_INTR_FAULT {
            return Some(slot.intr_fault.load(Ordering::Acquire) as u64);
        }
    }
    if let Some(mut d) = slot.inner.try_lock() {
        lab_held(1);
        let value = d.device.gfx_read(offset, size);
        slot.gfx_read_cache.lock().insert((offset, size), value);
        slot.gfx_read_busy_logged.store(false, Ordering::Relaxed);
        return Some(value);
    }
    let value = slot
        .gfx_read_cache
        .lock()
        .get(&(offset, size))
        .copied()
        .unwrap_or(0);
    if !slot.gfx_read_busy_logged.swap(true, Ordering::Relaxed) {
        crate::observe::fail(format!(
            "device_lock_busy reason=gfx_read_deferred offset={offset:#x} size={size} cached={value:#x}"
        ));
    }
    Some(value)
}

pub fn device_gfx_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    use crate::model::{GFX_REG_INTR_STATUS_DISP, GFX_REG_INTR_STATUS_GPU};
    let Some(slot) = device_slot(id) else {
        return false;
    };
    // Interrupt-status mask clears are order-independent of FIFO doorbells;
    // apply them lock-free instead of queueing behind a busy drain tranche.
    if size == 4 {
        if offset == GFX_REG_INTR_STATUS_DISP {
            slot.intr_disp.fetch_and(!(data as u32), Ordering::AcqRel);
            return true;
        }
        if offset == GFX_REG_INTR_STATUS_GPU {
            slot.intr_gpu.fetch_and(!(data as u32), Ordering::AcqRel);
            return true;
        }
        // The child doorbell, which measurement says is the *entire* queueing
        // stall on this pathway: `gfx_doorbell_delay` reads `offsets=1` on
        // every window that queued anything, ~100 rings a second applied up to
        // 45 ms late, and that delay is the drain tranche the write could not
        // take the lock through.
        //
        // It is the one register that can be served this way, because it
        // carries no state the decode depends on — its effect is to say a
        // channel has work. `fold_rung_child_doorbells` turns the bit into
        // the open-domain set / `pending.child_mask`, which is exactly what the
        // locked handler in `crate::runtime::mmio` does for the same register.
        //
        // The channel-number check mirrors that handler rather than trusting
        // the guest: a value outside the channel range names no channel, and
        // shifting by it would be undefined. An out-of-range ring still
        // schedules nothing here, as it does there — but it is reported rather
        // than dropped in silence, through the one spelling of the rule that
        // both handlers share.
        if offset == crate::model::GFX_REG_CHILD_DOORBELL
            || offset == crate::model::GFX_REG_CHILD_REPLAY_DOORBELL
        {
            let channel = data as u32;
            if crate::model::accept_child_channel(channel, "lock_free_child_doorbell") {
                slot.child_doorbell_rung
                    .fetch_or(1u32 << channel, Ordering::AcqRel);
                crate::runtime::drain::note_doorbell_lock_free();
                crate::runtime::drain::lab_note_doorbell(channel); // TEMP (lab)
                schedule_device(&slot);
            }
            return true;
        }
    }
    let mut write = QueuedGfxWrite {
        offset,
        data,
        size,
        queued_at: None,
    };
    let mut ingress = slot.gfx_ingress.lock();
    if ingress.is_empty() {
        if let Some(mut inner) = slot.inner.try_lock() {
            lab_held(2);
            apply_gfx_write(&mut inner, &slot, write);
            return true;
        }
    }
    // Stamped only on the path that actually defers, so the direct path pays no
    // clock read at all.
    write.queued_at = Some(std::time::Instant::now());
    ingress.push_back(write);
    drop(ingress);
    schedule_device(&slot);
    true
}

/// Take the device lock from the vCPU thread, measuring the wait.
///
/// The guest's MMIO access is stopped for exactly as long as this blocks, and
/// the drain worker holds this same lock across a full-surface readback. Every
/// other figure about that stall is taken from the holder's side, which makes
/// the step to "the guest missed a frame" an inference; this measures it where
/// it is actually paid.
///
/// The uncontended path takes `try_lock` and never reads the clock, so a fast
/// access pays nothing for the instrument.
fn lock_device_for_vcpu(slot: &BoundDevice) -> impl std::ops::DerefMut<Target = DeviceInner> + '_ {
    if let Some(guard) = slot.inner.try_lock() {
        lab_held(3);
        crate::runtime::drain::note_vcpu_lock_free();
        return guard;
    }
    let waited = std::time::Instant::now();
    let guard = slot.inner.lock();
    lab_held(3);
    crate::runtime::drain::note_vcpu_lock_wait(waited.elapsed().as_micros() as u64);
    guard
}

pub fn device_iosfc_read(id: u64, offset: u64, size: u32) -> Option<u64> {
    let slot = device_slot(id)?;
    let d = lock_device_for_vcpu(&slot);
    Some(d.device.iosfc_read(offset, size))
}

pub fn device_iosfc_write(id: u64, offset: u64, data: u64, size: u32) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let mut d = lock_device_for_vcpu(&slot);
    if let Some(ops) = slot.ops {
        let DeviceInner { device, actions } = &mut *d;
        let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
        device.iosfc_write(&mut host, offset, data, size);
    } else {
        let mut host = NullHost;
        d.device.iosfc_write(&mut host, offset, data, size);
    }
    true
}

/// Worker body: drain pending FIFOs using QEMU GPA callbacks; enqueue HostActions.
/// TEMP (lab): see the early publish in [`device_drain`]. Latched.
fn lab_early_publish() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("REIMS_VGPU_LAB_EARLY_PUBLISH").map_or(true, |v| v != "off");
        crate::observe::off(format!("lab_early_publish on={on}"));
        on
    })
}

pub fn device_drain(id: u64) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    // Opens this worker's wall-clock accounting: everything since the previous
    // exit was the condvar wait the C shim parks in, and everything from here to
    // the lock is contention with the vCPU thread. `duty` says how much of a
    // second the worker was busy; these say what the rest of it was.
    let entry_us = crate::runtime::drain::note_drain_entry();
    // The action BH needs the same device state to copy +0x188. A doorbell may
    // wake this worker before that BH runs; do not reacquire the lock and hide
    // the queued scanout behind another synchronous render/compute tranche.
    if slot.present_action_pending.load(Ordering::Acquire) {
        crate::runtime::drain::note_drain_skipped();
        crate::runtime::drain::note_drain_exit(entry_us, true);
        return true;
    }
    let mut d = lock_for_drain(&slot);
    crate::runtime::drain::note_drain_lock_wait(
        crate::observe::elapsed_us().saturating_sub(entry_us),
    );
    let Some(ops) = slot.ops else {
        // No host services — nothing to resolve from guest RAM. The lock wait is
        // already banked, so this closes with no post-tranche span rather than
        // leaving the entry open for the next one to absorb.
        crate::runtime::drain::note_drain_exit(crate::observe::elapsed_us(), false);
        return true;
    };
    let DeviceInner { device, actions } = &mut *d;
    let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
    // Presentation-path selector for this tranche: with a live host window the
    // drain publishes frames + self-acks; without one every present must
    // enqueue the CPU `ScanoutUpdate` and the ack belongs to the console paint
    // (see `enqueue_present_scanout` / the drain tail below).
    #[cfg(feature = "host-window")]
    {
        device.state.present.window_active = slot.window.lock().is_some();
    }
    #[cfg(not(feature = "host-window"))]
    {
        device.state.present.window_active = false;
    }
    // Split the tranche's two phases: guest work, then our host-window export.
    // Both hold the device lock, and which one owns the worker's wall clock is
    // the question `drain_duty` exists to answer.
    let tranche_started = crate::lab_clock::LabInstant::now();
    // The same instant on the crate's own clock, so a lookup inside the drain
    // can say how late in this tranche it happened without threading a start
    // time through every call. See `census::tranche_elapsed_us`.
    crate::runtime::drain::note_tranche_started(crate::observe::elapsed_us());
    // TEMP (lab, MONSTER_PREWARM): a new tranche retires every held currency
    // answer, so the doorbell's first binds ask for themselves. Free when the
    // switch is off (one relaxed fetch_add).
    crate::runtime::surface_currency::prewarm_new_tranche();
    // TEMP (lab, TIMERS): the census phase timers follow the live switch.
    crate::lab_clock::refresh_timers();
    crate::runtime::drain::lab_act(1); // TEMP (lab)
    device.drain(&mut host);
    // TEMP (lab, REIMS_VGPU_LAB_IRQ_BATCH): the gfx pulse held for this tranche.
    host.flush_deferred_irqs();
    // TEMP (lab, REIMS_VGPU_LAB_EARLY_PUBLISH, on by default): the finished frame
    // goes to the window before the tranche's tail work (the pay-after-stamp
    // readback and the debt trim), which held every present back by a median
    // 0.7 ms (p90 ~2 ms) in a quiet 5K scene. The deferred draw batch is
    // submitted first, so the window's blit is queued after the frame's draws.
    let early_publish = lab_early_publish();
    if early_publish {
        crate::runtime::drain::lab_act(2);
        crate::backend::selected().flush_deferred_submissions(&device.state);
        publish_present_boundary(&slot, device.state.present.frame_flush_seen);
        #[cfg(feature = "host-window")]
        window_publish::publish_window_frame(&slot, &mut device.state);
    }
    // TEMP (lab, REIMS_VGPU_LAB_PAY_AFTER): the replayed presented frame, paid
    // now that the guest has its completion.
    crate::runtime::drain::lab_act(3);
    // TEMP (lab T2T_DEFER): the tranche's deferred blit frames land before it ends.
    crate::runtime::render_writeback::land_t2t(&mut device.state, &mut host, None);
    crate::runtime::writeback_debt::lab_pay_after_stamp(
        &mut device.state,
        &mut host,
        slot.child_doorbell_rung.load(Ordering::Acquire) != 0,
    );
    // TEMP (lab, REIMS_VGPU_LAB_DEBT_TRIM): the ledger paid back to its bound.
    crate::runtime::drain::lab_act(4);
    crate::runtime::writeback_debt::lab_trim_debts(
        &mut device.state,
        &mut host,
        slot.child_doorbell_rung.load(Ordering::Acquire) != 0,
    );
    host.flush_deferred_irqs();
    // TEMP (lab): REIMS_VGPU_LAB_PAR_SELFTEST. A no-op fork-join on this thread,
    // under this lock, to measure the pool; free when the switch is unset.
    crate::runtime::par::selftest::tick(&device.state);
    // The tail is timed apart from `Device::drain` because it is inside
    // `drain_us` and inside no `DrainPhase`, and that residue is a third of the
    // drain worker's wall clock on every workload measured — 933 ms a second of
    // `drain_us` against 604 of `draw_us` on a driven `blur=40` boot. A gap that
    // size on the one thread every guest packet serializes through cannot be
    // left to inference.
    let tail_started = crate::lab_clock::LabInstant::now();
    crate::runtime::drain::lab_act(5);
    // Submit any deferred draw batch before the worker sleeps: consumers
    // inside the tranche flush on their own (engine begin_entry), this bounds
    // only the idle-tail latency of the last same-target run.
    crate::backend::selected().flush_deferred_submissions(&device.state);
    let tail_us = tail_started.elapsed().as_micros() as u64;
    let boundary_started = crate::lab_clock::LabInstant::now();
    if !early_publish {
        publish_present_boundary(&slot, device.state.present.frame_flush_seen);
    }
    crate::runtime::drain::note_drain_tail(tail_us, boundary_started.elapsed().as_micros() as u64);
    let drain_us = tranche_started.elapsed().as_micros() as u64;
    let publish_started = crate::lab_clock::LabInstant::now();
    // Push the finished present frame to the host-owned window (if running).
    // Off the QEMU main loop; a small dedicated mutex, never the render lock.
    crate::runtime::drain::lab_act(6);
    #[cfg(feature = "host-window")]
    if !early_publish {
        window_publish::publish_window_frame(&slot, &mut device.state);
    }
    crate::runtime::drain::note_drain_tranche(
        &device.state,
        &host,
        drain_us,
        publish_started.elapsed().as_micros() as u64,
    );
    // Everything from here to the return is `gap_post_us`: the per-tranche
    // sweeps below run on the worker's own wall clock and are outside both
    // `drain_us` and `publish_us`, so `duty` cannot see them.
    let busy_end_us = crate::observe::elapsed_us();
    crate::runtime::drain::lab_act(7);
    use crate::runtime::drain::{post_sweep, PostSweep};
    // Same one-second cadence, so the cache trend lines up row-for-row with
    // `store_routes` and `drain_duty`. Measure-only; see `note_cache_levels`.
    post_sweep(PostSweep::CacheLevels, || {
        crate::runtime::surface_cache::note_cache_levels(&device.state, &host)
    });
    // Per tranche rather than per census window, unlike the levels above: this
    // measures how long a slot the guest named takes to appear, so the sampling
    // interval is the resolution of the answer. Returns immediately when nothing
    // is watched, which is every tranche on every rail but macos-26.
    // TEMP (lab, 2026-10-10): at most every 10 ms, and not at all with the
    // route census off - measure-only, and per tranche it walked 300-430
    // watches with their guest reads (~5-6 ms/s of the drain on Weather, b115).
    if crate::runtime::drain::census::route_census_on() {
        static LAST_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let last = LAST_US.load(std::sync::atomic::Ordering::Relaxed);
        if busy_end_us.saturating_sub(last) >= 10_000 {
            LAST_US.store(busy_end_us, std::sync::atomic::Ordering::Relaxed);
            post_sweep(PostSweep::SlotRecheck, || {
                crate::runtime::objects::slot_recheck::sweep(&device.state, &host)
            });
        }
    }
    // The ordering plane's own residue, on the same one-second cadence as the
    // levels above and for the reason `backing_outstanding_census` is emitted
    // beside `store_routes`: the routes count what *happened* to a pipeline and
    // a pipeline that was declared and never advanced is counted once and never
    // again, so a table accumulating builds nobody is running reads exactly like
    // a healthy one. `pending` is the pipelines a transaction can be waiting on.
    post_sweep(PostSweep::PipelineTable, || {
        if let Some(line) = device.state.pipeline_occupancy_census() {
            crate::observe::off(line);
        }
        // And the graph those pipelines' transactions are admitted into: how much
        // of its ordering is bought with knowledge and how much with ignorance.
        if let Some(line) = device.state.dependency_graph_census() {
            crate::observe::off(line);
        }
    });
    // Beside it and on the same cadence: a page the guest released is judged
    // against the write census, which only moves when this device writes. Also
    // returns immediately when nothing is watched.
    post_sweep(PostSweep::ReleasedPages, || {
        crate::runtime::released_pages::sweep(&mut device.state);
        crate::runtime::released_pages::note_levels(&device.state);
    });
    // The bind registry's own levels, on that same cadence and read against the
    // `bb_retire_*` routes: what the retirements dropped, and what the survivors
    // look like.
    post_sweep(PostSweep::BindLevels, || {
        crate::runtime::bound_buffers::note_registry_levels(&device.state)
    });
    // The present-completion ack, re-homed off the QEMU paint — ONLY while the
    // host window is the display. With the window live no per-present
    // `ScanoutUpdate` is enqueued, so `display_surface::device_scanout_copy` —
    // the only other caller of `note_present_paint_consumed` — will not run for
    // this present.
    // Acking here clears `unpainted_presents` (releasing the DisplaySwap
    // backpressure gate at `MAX_UNPAINTED_PRESENTS`) and `host_action_yield`, so
    // the check below leaves `present_action_pending` clear and the worker keeps
    // draining. Without this the display channel wedges on the second present.
    //
    // On the window path the ack is deliberately NOT keyed on "a frame was
    // published": the publish legitimately early-returns on a duplicate
    // (mapping, generation), on a frame not yet valid, and on a short buffer. An
    // ack that fired only when the window took a fresh frame would wedge on the
    // first repeated present.
    //
    // Without a window the QEMU console owns the paint: `enqueue_present_scanout`
    // enqueued the `ScanoutUpdate`, `host_action_yield` stays set, the flag below
    // arms `present_action_pending`, and `device_scanout_copy` both paints and
    // acks. Pre-acking here would let `device_scanout_copy`'s nonblocking
    // `try_lock` path swallow the paint as `Unchanged` under worker contention —
    // the frozen-console class this split fixes.
    if device.state.present.window_active {
        crate::runtime::drain::note_present_paint_consumed(&mut device.state);
    }
    if device.state.pending.host_action_yield {
        slot.present_action_pending.store(true, Ordering::Release);
    }
    crate::runtime::drain::note_drain_exit(busy_end_us, false);
    crate::runtime::drain::lab_act(0); // TEMP (lab): idle
    true
}

/// Periodic tick (gfx_update / poll): archive `poll_tick` subset.
///
/// - Dekker rescue: publish main/child/iosfc work to the asynchronous drain
///   owner when producer state may have advanced without a doorbell.
/// - Re-drive display ONLINE after guest enable() publishes the mask.
///
/// Enqueues HostActions (gfx IRQ / scanout); QEMU must deliver actions after
/// this call.
/// TEMP (lab): microseconds until the display's next VBL grid point
/// ([`crate::runtime::drain::claim_display_vbl`]), for a heartbeat that sleeps
/// to it instead of polling on a coarse period; 0 when it is due, or before the
/// first VBL set a grid.
pub fn device_vbl_due_us(id: u64) -> u64 {
    let Some(slot) = device_slot(id) else {
        return 0;
    };
    let last = slot.vbl_last_us.load(Ordering::Acquire);
    if last == 0 {
        return 0;
    }
    (last + crate::runtime::drain::display_vbl_interval_us())
        .saturating_sub(crate::observe::elapsed_us())
}

pub fn device_poll(id: u64) -> bool {
    // Before the lock, and before the `device_slot` miss can return: this is the
    // only periodic callback that runs on a thread other than the drain worker,
    // so it is the only place that can report a driver call the drain worker is
    // still inside. See `observe::driver_watch` for what that failure looks like
    // from the log (it looks like nothing at all).
    crate::observe::driver_watch::note_tick();
    let Some(slot) = device_slot(id) else {
        return false;
    };
    lab_stamp_rewake_tick(&slot);
    ring_ingest_tick(&slot, id);
    // TEMP (lab, POLL_YIELD, b041): the drain worker is taking or holding
    // `inner`. b041: the poll held `inner` only ~160 ms a phase (a few
    // microseconds a tick, 1000 ticks a second), yet the drain waited 0.9-1.3 s
    // a phase behind it (7-10 thousand collisions, ~120 us each, up to 104 ms in
    // one second) - the cost of the collision itself: the drain parks, and a
    // heartbeat descheduled while holding the lock keeps it parked. Everything
    // the poll does under `inner` the drain does itself on its way in (doorbell
    // fold, stranded FIFOs) or is the VBL the lock-free pulse below already
    // delivers, so while the drain is in, the poll stands aside.
    if lab_poll_yield() && slot.drain_wants.load(Ordering::SeqCst) {
        vbl_contended_pulse(&slot);
        // The rescue half of the poll without `inner`: `publish_stranded_fifos`
        // wakes the drain for ring work no doorbell announced, and the drain
        // reads its rings to a snapshot taken at tranche start - work landing
        // during the tranche waits for the next wake. b042/b045/b046: with the
        // poll standing aside and not waking, Safari's mid-animation frames ran
        // 5-14 % slower (smooth fps 135 -> 118-127). A spurious wake costs the
        // drain one empty check.
        if lab_poll_yield_wake() {
            schedule_device(&slot);
        }
        crate::runtime::drain::note_store_route("lab_poll_yielded");
        crate::backend::selected().maintain(crate::observe::elapsed_ms() as u64);
        return true;
    }
    let Some(mut d) = slot.inner.try_lock() else {
        // Contended: the drain worker owns `inner` doing present/GPU-encode.
        // The full poll below would early-return and drop the VBL — under load
        // that starves the guest's only display time base (present-complete is
        // inert; kb present-thrash-proxies). Pulse VBL lock-free from the state
        // the last successful poll published, so pacing survives the contention.
        vbl_contended_pulse(&slot);
        return true;
    };
    lab_held(4);
    let lab_t = crate::lab_clock::LabInstant::now(); // TEMP (lab, b041): parts under `inner`
    let Some(ops) = slot.ops else {
        return true;
    };
    let DeviceInner { device, actions } = &mut *d;
    let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
    // Before the rescue reads the open-domain set, which is the mask a
    // lock-free ring lands in only once folded. Without this the Dekker rescue
    // could not see the very channels the doorbell rail is responsible for.
    crate::runtime::drain::fold_rung_child_doorbells(&mut device.state);
    crate::runtime::drain::publish_stranded_fifos(&mut device.state, &mut host);
    let lab_t1 = lab_t.elapsed();
    crate::runtime::drain::try_display_online(&mut device.state, &mut host);
    // After ONLINE, pulse VBL so the guest compositor has a display time base
    //. Missing VBL → clear-only dual-mid present thrash.
    crate::runtime::drain::signal_display_vbl(&mut device.state, &mut host, &slot.vbl_last_us);
    let lab_t2 = lab_t.elapsed();
    // Republish the lock-free VBL snapshot for the contended fast path above.
    // These change only at online-ack/reinit, but publishing every poll keeps
    // the snapshot fresh with no extra synchronization on the rare-change path.
    slot.vbl_shared_gpa
        .store(device.state.display.shared_gpa, Ordering::Release);
    slot.vbl_display_index
        .store(device.state.display.display_index, Ordering::Release);
    slot.vbl_online
        .store(device.state.display.online_acked, Ordering::Release);
    slot.vbl_page_size
        .store(device.state.page_size(), Ordering::Release);
    // Aliases the previous tick's maintenance released. Needs the host view of
    // the device's action queue, so it stays under `inner`.
    crate::runtime::mapper::drain_deferred_unmaps(&mut host);
    let lab_t3 = lab_t.elapsed();
    // Pre-boundary early-console → host window (headless-safe: the heartbeat
    // drives poll even under -display none). No-op post-boundary or with no
    // window attached.
    #[cfg(feature = "host-window")]
    {
        let now_ns = host.mono_ns();
        window_publish::publish_window_early_frame(&slot, &device.state, &host, now_ns);
    }
    drop(host);
    drop(d);
    lab_poll_parts([lab_t1, lab_t2 - lab_t1, lab_t3 - lab_t2, lab_t.elapsed() - lab_t3]);
    // Census both source polls and the independently time-gated VBL rate.
    // Drive bounded maintenance from the poll heartbeat, which ticks even when
    // the guest stops publishing. The wall clock returns already-dead resources
    // and free-pool memory; it has no authority over live residency, which is
    // governed by resource lifetime and allocation pressure.
    //
    // After `inner` is released: maintenance touches only backend state, which
    // the engine lock serializes, and its fence polls and graveyard releases are
    // driver calls. Run under `inner` they held the drain worker off the device
    // for their whole duration — at 5K a drain blocked on this lock was ~6% of
    // its samples, with the heartbeat inside `maintain_resources` in most of
    // them. The aliases it releases are unmapped by the next tick above.
    crate::backend::selected().maintain(crate::observe::elapsed_ms() as u64);
    true
}

/// TEMP (lab, b041): what the heartbeat poll spends holding `inner` - b039: the
/// drain waited 250-510 ms a phase for it (holder 4), up to 107 ms in one
/// second, concentrated in the gaps at a transition's end. Parts: 0 doorbell
/// fold + stranded FIFOs, 1 display online + VBL, 2 deferred unmaps, 3 the
/// early window frame. A line a second: per part, sum / max in microseconds.
fn lab_poll_parts(parts: [std::time::Duration; 4]) {
    use std::sync::atomic::AtomicU64;
    static SUM: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    static MAX: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    static LAST_MS: AtomicU64 = AtomicU64::new(0);
    for (i, d) in parts.iter().enumerate() {
        let us = d.as_micros() as u64;
        SUM[i].fetch_add(us, Ordering::Relaxed);
        MAX[i].fetch_max(us, Ordering::Relaxed);
    }
    let now = crate::observe::elapsed_ms() as u64;
    let last = LAST_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 1000
        || LAST_MS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    let mut line = String::from("lab_poll_parts");
    for i in 0..4 {
        let s = SUM[i].swap(0, Ordering::Relaxed);
        let m = MAX[i].swap(0, Ordering::Relaxed);
        line.push_str(&format!(" p{i}={s}/{m}us"));
    }
    crate::observe::off(line);
}

/// TEMP (lab, RING_INGEST): the heartbeat takes whole published packets out of
/// every armed child ring into the device's queue without the device lock, so a
/// guest ring keeps moving while the drain runs a long EXEC (Maps fills ch1 and
/// the guest's driver resets the channel - b196, b207). Channels it moved are
/// rung the way the lock-free doorbell rings them.
fn ring_ingest_tick(slot: &BoundDevice, id: u64) {
    if !crate::runtime::ring_ingest::enabled() {
        return;
    }
    let Some(ops) = slot.ops else {
        return;
    };
    let page_size = slot.vbl_page_size.load(Ordering::Acquire);
    if page_size == 0 {
        return;
    }
    let mut scratch = VecDeque::new();
    let mut host = QemuHost::new(&ops, &mut scratch, &slot.prompt_actions);
    let moved = crate::runtime::ring_ingest::heartbeat(id, &mut host, page_size as usize);
    if moved != 0 {
        slot.child_doorbell_rung.fetch_or(moved, Ordering::AcqRel);
        schedule_device(slot);
    }
}

/// Lock-free VBL pulse for a `device_poll` that could not take `inner`.
///
/// Raises the display VBL — OR the VBL bit into the shared-page pending word,
/// set the read-to-clear display interrupt bit, enqueue the gfx IRQ pulse — all
/// through paths that never touch the device `inner` lock (guest-memory RMW via
/// HostOps, the `Arc<AtomicU32>` interrupt clone, the lock-free `prompt_actions`
/// queue). Uses the VBL state the last lock-acquired poll published. No-op until
/// ONLINE is acked. It shares the same time limiter as the locked path, so a
/// change in lock ownership cannot change the guest's pacing rate.
///
/// The pending-word RMW can race the worker's own present-complete write; the
/// loser drops one bit for one heartbeat (re-raised ~16 ms later). Both writers
/// clear the acked ONLINE bit, so a torn write cannot resurrect it — far better
/// than dropping ~90% of VBLs, which is the pre-fix behaviour under load.
fn vbl_contended_pulse(slot: &BoundDevice) {
    let gpa = slot.vbl_shared_gpa.load(Ordering::Acquire);
    let now = crate::observe::elapsed_ms() as u64;
    if gpa == 0 || !slot.vbl_online.load(Ordering::Acquire) {
        crate::runtime::drain::note_vbl(crate::runtime::drain::VBL_NOT_ONLINE, now);
        return;
    }
    let Some(ops) = slot.ops else {
        return;
    };
    let page_size = slot.vbl_page_size.load(Ordering::Acquire);
    if page_size == 0 {
        // The locked poll publishes this with the rest of the snapshot, so a
        // zero means no locked poll has run since bind. Nothing is owed yet.
        crate::runtime::drain::note_vbl(crate::runtime::drain::VBL_NOT_ONLINE, now);
        return;
    }
    let mut scratch = VecDeque::new();
    let mut host = QemuHost::new(&ops, &mut scratch, &slot.prompt_actions);
    // One body decides what a refresh tick writes, and both poll arms call it.
    // This arm used to carry its own copy, and the copy had already lost a term:
    // it never read the enable word at all, so it set a pending bit the guest's
    // ISR would never clear and counted the write as `delivered`.
    //
    // The shared limiter lives in there too, so both arms report into one census
    // and neither can spend a grid slot on a tick that found the guest disarmed.
    crate::runtime::drain::signal_display_refresh_classes(
        &mut host,
        gpa,
        slot.vbl_display_index.load(Ordering::Acquire),
        &slot.intr_disp,
        page_size as usize,
        &slot.vbl_last_us,
        crate::observe::elapsed_us(),
    );
}

/// Pop one HostAction for the QEMU BH. Returns false if the queue is empty.
///
/// Prompt actions (IRQ pulses, cursor moves) pop without the device lock so
/// they deliver mid-drain; lock-owning actions (scanout, cursor glyph) keep
/// their after-drain semantics behind `try_lock`.
pub fn device_pop_action(id: u64) -> Option<HostAction> {
    let slot = device_slot(id)?;
    {
        let mut q = slot.prompt_actions.lock();
        if let Some(a) = q.pop_front() {
            // The hop this closes is enqueue-to-BH, so it is banked when the
            // queue empties rather than per action: an IRQ pulse behind a cursor
            // move waited for the same BH and would double-count. See
            // `irq_wait_us`.
            if q.is_empty() {
                crate::runtime::drain::note_irq_delivered();
            }
            return Some(a);
        }
    }
    // TEMP (lab, POLL_YIELD, b044): the drain is taking `inner`. b044: almost
    // all the drain's remaining lock wait (450 of 454 ms a Safari phase, 661 of
    // 673 Finder) was this pop barging in on an unfair lock while the woken
    // drain thread was still being scheduled - each pop holds microseconds, but
    // the BH loops on it. The lock-owning actions keep their after-drain
    // semantics: `DrainGuard` reschedules the BH when it lets go with any left.
    // b045: yielding unconditionally cost frames (Safari 84 -> 67/80, Finder
    // 100 -> 84/93 while the drain's lock wait fell 1442 -> 554-611 ms): a
    // queued scanout action that used to slip in while the drain waited now
    // waited out the whole next tranche. A pending present is delivered at once;
    // only the empty-queue checks the BH loops on stand aside.
    if lab_poll_yield()
        && slot.drain_wants.load(Ordering::SeqCst)
        && !slot.present_action_pending.load(Ordering::Acquire)
    {
        crate::runtime::drain::note_store_route("lab_pop_yielded");
        return None;
    }
    let mut d = slot.inner.try_lock()?;
    lab_held(5);
    d.actions.pop_front()
}

/// What the process's backend calls itself, for QEMU's realize trace.
///
/// Asks the backend rather than restating the build's feature set. The two used
/// to be the same claim written twice, and once a build can carry both arms they
/// stop being: the feature set says what was compiled, and this says what is
/// executing.
pub fn backend_name() -> &'static str {
    use crate::backend::Backend as _;
    crate::backend::selected().name()
}

/// Run one C ABI entry body, turning a panic into `on_panic` rather than
/// letting it unwind into QEMU's C frames.
///
/// `entry` is the C symbol this body belongs to, and it is what the record
/// names — a panic here is the largest thing this device can drop (the whole
/// call, not one refused record), so it goes through the always-on failure path
/// like any other loss of guest work. See [`crate::observe::panic`] for why the
/// location needs a hook and why arming it lives on this path.
///
/// Every caller must pass its own symbol name. A copied call site that still
/// names its neighbour reports the wrong entry point for a panic, and nothing
/// checks for it — a source scan used to.
pub fn unwind_safe<T, F>(entry: &'static str, f: F, on_panic: T) -> T
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    crate::observe::panic::arm();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(payload) => {
            crate::observe::panic::report(entry, payload.as_ref());
            on_panic
        }
    }
}

#[cfg(test)]
mod tests;

/// TEMP (lab): POLL_YIELD - live `POLL_YIELD=on|off`, else
/// `REIMS_VGPU_LAB_POLL_YIELD=off`; on by default.
fn lab_poll_yield() -> bool {
    if let Some(on) = crate::runtime::lab_live::on("POLL_YIELD") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_POLL_YIELD").as_deref() != Ok("off"))
}

/// TEMP (lab): POLL_SPIN - how long the drain spins on `inner` held by the
/// heartbeat poll before parking: live `POLL_SPIN_US=N` (0..5000), else
/// `REIMS_VGPU_LAB_POLL_SPIN_US`, default 0 (parks at once, as before):
/// b043 - spinning up to 1 ms did not lower the drain's total wait.
fn lab_poll_spin_us() -> u64 {
    if let Some(n) = crate::runtime::lab_live::value("POLL_SPIN_US").and_then(|v| v.trim().parse::<u64>().ok()) {
        return n.min(5000);
    }
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_POLL_SPIN_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(0, |n| n.min(5000))
    })
}

/// TEMP (lab, b044): drain lock waits whose holder changed while the drain
/// waited - `first->last` pairs, count and wait. A line a second while any.
fn lab_drain_lock_handoff(first: u8, last: u8, us: u64) {
    use std::sync::atomic::AtomicU64;
    static N: [AtomicU64; 121] = [const { AtomicU64::new(0) }; 121];
    static W: [AtomicU64; 121] = [const { AtomicU64::new(0) }; 121];
    static LAST_MS: AtomicU64 = AtomicU64::new(0);
    let i = (usize::from(first.min(10))) * 11 + usize::from(last.min(10));
    N[i].fetch_add(1, Ordering::Relaxed);
    W[i].fetch_add(us, Ordering::Relaxed);
    let now = crate::observe::elapsed_ms() as u64;
    let prev = LAST_MS.load(Ordering::Relaxed);
    if now.saturating_sub(prev) < 1000
        || LAST_MS
            .compare_exchange(prev, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    let mut line = String::from("lab_drain_lock_handoff");
    for j in 0..121 {
        let n = N[j].swap(0, Ordering::Relaxed);
        let w = W[j].swap(0, Ordering::Relaxed);
        if n > 0 {
            line.push_str(&format!(" h{}->h{}={n}/{w}us", j / 11, j % 11));
        }
    }
    crate::observe::off(line);
}

/// TEMP (lab): POLL_YIELD_WAKE - a yielding poll still wakes the drain; live
/// `POLL_YIELD_WAKE=on|off`, else `REIMS_VGPU_LAB_POLL_YIELD_WAKE=off`; on by default.
fn lab_poll_yield_wake() -> bool {
    if let Some(on) = crate::runtime::lab_live::on("POLL_YIELD_WAKE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_POLL_YIELD_WAKE").as_deref() != Ok("off"))
}
