//! HostOps / HostActions: services Rust cannot provide (guest memory, IRQ, display).
//!
//! Pattern mirrors apple-gfx ↔ ParavirtualizedGraphics.framework:
//! QEMU C owns only the host-service callbacks; Rust owns protocol + drain and
//! enqueues [`crate::runtime::host::HostAction`]s for a QEMU BH to apply on
//! the main loop.

use crate::runtime::host::{
    GuestRamRegionsError, HostAction, HostActionKind, HostMemory, HostOps, MemError,
    PageAliasCensus,
};
use std::collections::VecDeque;
use std::os::raw::{c_int, c_void};

const MAP_PAGES_FAILURE_NONE: u32 = 0;
const MAP_PAGES_FAILURE_RESERVATION: u32 = 1;
const MAP_PAGES_FAILURE_ALIAS: u32 = 2;
const MAP_PAGES_FAILURE_INVALID_GUEST_PAGE: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ReimsVgpuMapPagesFailure {
    pub(crate) stage: u32,
    pub(crate) host_errno: i32,
    pub(crate) page_index: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapPagesFailure {
    ReservationFailed { host_errno: i32 },
    AliasFailed { page_index: u64, host_errno: i32 },
    InvalidGuestPage { page_index: u64 },
    Unspecified { stage: u32, host_errno: i32 },
}

/// Versioned host callback table offered by QEMU C to Rust.
///
/// Layout must match `ReimsVgpuHostOps` in `include/reims_vgpu_qemu_abi.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReimsVgpuHostOps {
    pub abi_version: u32,
    pub struct_size: u32,
    pub ctx: *mut c_void,
    /// Read guest physical memory into `buf` (`len` bytes) from `gpa`.
    /// Returns 0 on success.
    pub read_gpa:
        Option<unsafe extern "C" fn(ctx: *mut c_void, gpa: u64, buf: *mut u8, len: usize) -> i32>,
    /// Write guest physical memory from `buf`.
    pub write_gpa:
        Option<unsafe extern "C" fn(ctx: *mut c_void, gpa: u64, buf: *const u8, len: usize) -> i32>,
    /// Monotonic nanoseconds.
    pub mono_ns: Option<unsafe extern "C" fn(ctx: *mut c_void) -> u64>,
    /// Wake QEMU main-loop BH to drain pending work / HostActions.
    /// Safe from any thread (schedules oneshot BH).
    pub schedule_bh: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
    /// Read guest kernel VA (cpu_memory_rw_debug). Returns 0 on success.
    pub read_kva:
        Option<unsafe extern "C" fn(ctx: *mut c_void, kva: u64, buf: *mut u8, len: usize) -> i32>,
    /// Read guest CPU X-register `index` into `*out`. Returns 0 on success.
    pub read_xreg: Option<unsafe extern "C" fn(ctx: *mut c_void, index: u32, out: *mut u64) -> i32>,
    /// Contiguous host-VA view of guest pages (mach_vm_remap of guest RAM).
    pub map_pages: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            gpas: *const u64,
            count: usize,
            out_ptr: *mut *mut c_void,
            failure: *mut ReimsVgpuMapPagesFailure,
        ) -> i32,
    >,
    pub unmap_pages: Option<unsafe extern "C" fn(ctx: *mut c_void, ptr: *mut c_void, len: usize)>,
    /// 1 = guest RAM, 0 = not RAM. Optional (None → treat as RAM for unit fixtures).
    pub is_ram_gpa: Option<unsafe extern "C" fn(ctx: *mut c_void, gpa: u64) -> i32>,
    /// Write at most `max` guest-RAM spans into `out` and return the total this
    /// host has — a return greater than `max` says the array was short — or a
    /// negative `REIMS_VGPU_GUEST_RAM_ERR_*`.
    ///
    /// The spans are the mappings QEMU already holds over its RAMBlocks, stable
    /// for the VM's lifetime, so nothing is allocated and nothing is released.
    /// That is what separates this from `map_pages`, which answers about
    /// specific pages and on one shim builds a transient view the caller owns.
    pub guest_ram_regions: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            out: *mut crate::runtime::guest_ram::GuestRamRegion,
            max: usize,
        ) -> i32,
    >,
    /// Schedule the HostAction-delivery BH (pop_action consumer). Safe from any
    /// thread. Distinct from `schedule_bh` (drain-worker wake): prompt actions
    /// (IRQ pulses, cursor moves) must be deliverable mid-drain.
    pub notify_actions: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
    /// 1 when a `map_pages` result remains valid until its matching
    /// `unmap_pages` call, so a caller may retain it in a GPU import.
    ///
    /// x86 answers **1**: direct RAMBlock pointers remain borrowed, while a
    /// fragmented list's packed alias remains valid until explicit retirement.
    /// `unmap_pages` recognizes direct pointers by their absence from the
    /// shim's live-view registry and leaves them alone. arm64 answers **0**.
    ///
    /// It used to also license retaining the pointer inside a cached host-pointer
    /// import, which is where the stronger promise came from — MMIO could claim
    /// 1 only because it never released a view at all, so every fragmented map
    /// leaked a VA reservation until teardown. The GPU rail does not read this
    /// flag for the base RAMBlock import: those spans come from
    /// `guest_ram_regions` and neither shim built them. Resource-shaped packed
    /// imports do read it, because retaining such an import requires the
    /// `map_pages` alias itself to outlive submitted GPU work.
    pub map_pages_stable: c_int,
    /// Register `count` page-aligned GPAs as one guest-write-tracked set and
    /// return a non-zero opaque token, or 0 when the host has no dirty bitmap.
    /// Mutates QEMU MemoryRegion logging state, so the shim may only do the
    /// enabling part with the BQL held — see the C side for how it defers.
    pub track_guest_writes: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            gpas: *const u64,
            count: usize,
            page_size: usize,
        ) -> u64,
    >,
    /// Release a token from `track_guest_writes`.
    pub untrack_guest_writes: Option<unsafe extern "C" fn(ctx: *mut c_void, token: u64)>,
    /// Monotonic count of host observations that some page of the token's set
    /// was written; 0 for an unknown or not-yet-armed token. Safe from any
    /// thread.
    pub guest_write_gen: Option<unsafe extern "C" fn(ctx: *mut c_void, token: u64) -> u64>,
    /// Page-aligned GPAs of the token's set written since `since_gen`, into
    /// `out`, returning how many — or -1 for every case where the answer is not
    /// knowable and the caller must assume the whole set was written. Safe from
    /// any thread.
    pub guest_written_pages: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            token: u64,
            since_gen: u64,
            out: *mut u64,
            max: usize,
        ) -> i64,
    >,
    pub page_alias_census:
        Option<unsafe extern "C" fn(ctx: *mut c_void, out: *mut PageAliasCensus) -> i32>,
    /// Non-zero when every dirty-log harvest a doorbell asked for has finished.
    /// Never blocks. `None` is a shim that harvests inline, which is always
    /// settled.
    pub harvests_settled: Option<unsafe extern "C" fn(ctx: *mut c_void) -> c_int>,
    /// Raise an IRQ pulse now, from any thread, without the BQL; 1 when raised,
    /// 0 when the caller must queue it for the action BH instead.
    pub irq_pulse: Option<unsafe extern "C" fn(ctx: *mut c_void, kind: u32) -> c_int>,
    /// Which doorbells bind the generation reads that follow; see the ABI header.
    pub note_work_scope: Option<unsafe extern "C" fn(ctx: *mut c_void, scope: u32)>,
    /// Register the *calling* thread with the shim's RCU domain. A thread this
    /// crate spawns must do so before it reads guest memory through `read_gpa`
    /// or asks the dirty tracker anything. Reached only through
    /// [`ReimsVgpuHostOps::thread_registrar`], which refuses a table that
    /// offers one half of the pair.
    pub thread_attach: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
    /// The other half: unregister the calling thread, as its last act.
    pub thread_detach: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
    /// The drain read child channel `channel`'s ring tail; see the ABI header.
    pub note_ring_tail: Option<unsafe extern "C" fn(ctx: *mut c_void, channel: u32, tail: u32)>,
}

// SAFETY: QEMU keeps the table valid for the device lifetime; callbacks only
// touch QEMU state under the BQL / from the AIO BH. We store the table as raw
// pointers and never move the C context.
unsafe impl Send for ReimsVgpuHostOps {}
unsafe impl Sync for ReimsVgpuHostOps {}

impl ReimsVgpuHostOps {
    /// The means to put a worker thread this crate spawns under the shim's RCU,
    /// or `None` when the shim offers no such thread.
    ///
    /// Both callbacks or neither: attaching without a way to detach would leave
    /// a registered reader behind a thread that has exited, and `synchronize_rcu`
    /// would then wait for it forever. A half-offered pair is a refusal, not a
    /// best effort, and the caller reads `None` as "start no worker thread" —
    /// the structural gate that keeps the parallel pool off on a shim whose
    /// guest-memory reads were never made safe for it.
    pub fn thread_registrar(&self) -> Option<ThreadRegistrar> {
        Some(ThreadRegistrar {
            ctx: self.ctx,
            attach: self.thread_attach?,
            detach: self.thread_detach?,
        })
    }
}

/// How a worker thread joins and leaves the shim's RCU domain.
///
/// `Copy` and shareable so one registrar can be handed to every worker of a
/// pool; what is not shareable is the [`ThreadAttachment`] it makes, which is
/// the calling thread's own.
#[derive(Clone, Copy)]
pub struct ThreadRegistrar {
    ctx: *mut c_void,
    attach: unsafe extern "C" fn(ctx: *mut c_void),
    detach: unsafe extern "C" fn(ctx: *mut c_void),
}

// SAFETY: the table the registrar was cut from is `Send + Sync` for the same
// reason — QEMU keeps `ctx` valid for the device lifetime, and both callbacks
// are documented as safe from any thread that attaches and detaches for itself.
unsafe impl Send for ThreadRegistrar {}
unsafe impl Sync for ThreadRegistrar {}

impl ThreadRegistrar {
    /// Register the calling thread, for as long as the returned guard lives.
    ///
    /// Call it as the thread's first act and drop the guard as its last, on the
    /// same thread: the registration is thread-local, and QEMU asserts on a
    /// second one.
    pub fn attach(&self) -> ThreadAttachment {
        // SAFETY: QEMU owns `ctx` and keeps it valid for the device lifetime;
        // the shim's attach touches only the calling thread's RCU record.
        unsafe { (self.attach)(self.ctx) };
        ThreadAttachment {
            ctx: self.ctx,
            detach: self.detach,
        }
    }
}

/// A live RCU registration of the thread that made it. Dropping it unregisters.
///
/// Not `Send`: it names the thread that attached, and unregistering from any
/// other is the asserting case the ABI header rules out. The raw `ctx` pointer
/// is what makes the type `!Send`; nothing here adds an impl.
#[must_use = "dropping the guard unregisters the thread at once"]
pub struct ThreadAttachment {
    ctx: *mut c_void,
    detach: unsafe extern "C" fn(ctx: *mut c_void),
}

impl Drop for ThreadAttachment {
    fn drop(&mut self) {
        // SAFETY: paired with the attach that made this guard, on the same
        // thread, and run before the thread exits.
        unsafe { (self.detach)(self.ctx) };
    }
}

#[cfg(test)]
impl ReimsVgpuHostOps {
    /// A correctly-versioned table with every callback absent.
    ///
    /// Tests that exercise the missing-callback declines start here and fill in
    /// only the callback under test. It lives beside the struct because it
    /// names every field: two test modules each kept an identical copy of this
    /// list, so a new field in the C ABI had to be written out in three places
    /// that agreed.
    pub(crate) fn null() -> Self {
        Self {
            abi_version: crate::qemu::abi::REIMS_VGPU_QEMU_ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            ctx: std::ptr::null_mut(),
            read_gpa: None,
            write_gpa: None,
            mono_ns: None,
            schedule_bh: None,
            read_kva: None,
            read_xreg: None,
            map_pages: None,
            unmap_pages: None,
            map_pages_stable: 0,
            track_guest_writes: None,
            untrack_guest_writes: None,
            guest_write_gen: None,
            guest_written_pages: None,
            page_alias_census: None,
            is_ram_gpa: None,
            guest_ram_regions: None,
            notify_actions: None,
            harvests_settled: None,
            irq_pulse: None,
            note_work_scope: None,
            thread_attach: None,
            thread_detach: None,
            note_ring_tail: None,
        }
    }
}

/// Failures in the QEMU service adapter that cannot ride a fallible HostOps
/// return value.
///
/// Guest-memory reads and writes return [`MemError`] directly. The clock,
/// wake, and page-map methods predate I2 and still expose `u64`, `()`, or
/// `Option<usize>`, so an omitted callback or failed map would otherwise be
/// indistinguishable from an ordinary higher-level miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QemuHostDecline {
    MonoNsCallbackMissing,
    ScheduleBhCallbackMissing,
    MapPagesCallbackMissing {
        first_gpa: u64,
        page_count: usize,
        page_size: usize,
    },
    MapPagesCallbackFailed {
        rc: i32,
        failure: MapPagesFailure,
        first_gpa: u64,
        page_count: usize,
        page_size: usize,
    },
    MapPagesNullPointer {
        first_gpa: u64,
        page_count: usize,
        page_size: usize,
    },
    UnmapPagesCallbackMissing {
        ptr: usize,
        len: usize,
    },
}

impl crate::observe::Decline for QemuHostDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::MonoNsCallbackMissing => "qemu_mono_ns_callback_missing",
            Self::ScheduleBhCallbackMissing => "qemu_schedule_bh_callback_missing",
            Self::MapPagesCallbackMissing { .. } => "qemu_map_pages_callback_missing",
            Self::MapPagesCallbackFailed { failure, .. } => match failure {
                MapPagesFailure::ReservationFailed { .. } => "qemu_map_pages_reservation_failed",
                MapPagesFailure::AliasFailed { .. } => "qemu_map_pages_alias_failed",
                MapPagesFailure::InvalidGuestPage { .. } => "qemu_map_pages_invalid_guest_page",
                MapPagesFailure::Unspecified { .. } => "qemu_map_pages_failure_unspecified",
            },
            Self::MapPagesNullPointer { .. } => "qemu_map_pages_null_pointer",
            Self::UnmapPagesCallbackMissing { .. } => "qemu_unmap_pages_callback_missing",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::MonoNsCallbackMissing | Self::ScheduleBhCallbackMissing => Vec::new(),
            Self::MapPagesCallbackMissing {
                first_gpa,
                page_count,
                page_size,
            }
            | Self::MapPagesNullPointer {
                first_gpa,
                page_count,
                page_size,
            } => vec![
                ("first_gpa", format!("{first_gpa:#x}")),
                ("page_count", page_count.to_string()),
                ("page_size", page_size.to_string()),
            ],
            Self::MapPagesCallbackFailed {
                rc,
                failure,
                first_gpa,
                page_count,
                page_size,
            } => {
                let (stage, page_index, host_errno) = match failure {
                    MapPagesFailure::ReservationFailed { host_errno } => {
                        ("reservation", None, *host_errno)
                    }
                    MapPagesFailure::AliasFailed {
                        page_index,
                        host_errno,
                    } => ("alias", Some(*page_index), *host_errno),
                    MapPagesFailure::InvalidGuestPage { page_index } => {
                        ("invalid_guest_page", Some(*page_index), 0)
                    }
                    MapPagesFailure::Unspecified { stage, host_errno } => {
                        ("unspecified", Some(u64::from(*stage)), *host_errno)
                    }
                };
                let mut fields = vec![
                    ("rc", rc.to_string()),
                    ("stage", stage.to_string()),
                    ("errno", host_errno.to_string()),
                    ("first_gpa", format!("{first_gpa:#x}")),
                    ("page_count", page_count.to_string()),
                    ("page_size", page_size.to_string()),
                ];
                if let Some(page_index) = page_index {
                    fields.push(("page_index", page_index.to_string()));
                }
                fields
            }
            Self::UnmapPagesCallbackMissing { ptr, len } => {
                vec![("ptr", format!("{ptr:#x}")), ("len", len.to_string())]
            }
        }
    }
}

impl QemuHostDecline {
    fn emit(self, discriminant: u64) {
        crate::observe::Emit::decline("qemu_host_adapter", &self).fail_once(discriminant);
    }
}

/// Production host bridge: GPA/KVA via C callbacks, actions queued for the BH.
///
/// Two action rails:
/// - `actions` (inside the device lock): scanout / cursor-glyph / trace —
///   delivered by the BH after the drain tranche releases the lock (the
///   scanout apply re-enters the device for the +0x188 copy).
/// - `prompt` (outside the device lock): IRQ pulses + cursor moves — pushed to
///   the slot-level queue and `notify_actions`-scheduled immediately, so a
///   guest ISR sees its stamp-completion MSI while the drain worker is still
///   rendering later packets (ack fast / render async).
///
/// Both rails are always present. There was a second constructor that left
/// `prompt` unset, so that `enqueue` fell through to the single lock-owning
/// queue; no product site ever called it, which made the `None` arm — and the
/// "IRQ pulses wait for the drain tranche" behaviour behind it — reachable
/// only from the test written to describe it.
pub struct QemuHost<'a> {
    ops: &'a ReimsVgpuHostOps,
    actions: &'a mut VecDeque<HostAction>,
    prompt: &'a parking_lot::Mutex<VecDeque<HostAction>>,
    /// TEMP (lab, `REIMS_VGPU_LAB_IRQ_BATCH`): when (device clock, us) the held
    /// gfx pulse of the current batch was first asked for, if one is held.
    deferred_gfx: Option<u64>,
}

/// TEMP (lab): `REIMS_VGPU_LAB_IRQ_BATCH=<us>` (default 500; 0 = off). The drain
/// raises a gfx pulse per completed packet (~7 000/s on the 5K drag, ~60 per
/// frame), and the guest's interrupt handler answers each with three register
/// reads, one of them an emulated exit. With a budget, a pulse is held and the
/// ones after it fold into it until the budget has run since the first, and
/// whatever is held is raised when the tranche ends (and on drop): an interrupt
/// raised later is still raised, and the status bits it reports accumulate.
fn lab_irq_batch_us() -> u64 {
    static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *US.get_or_init(|| {
        // 500 since the 5K continuous drag A/B (fb*, x2): gfx interrupts
        // 8 170 -> 1 240/s, guest emulated MMIO exits 15 100 -> 9 400/s, drain
        // duty 0.76/0.77 -> 0.75/0.75, late frames 1.8/1.3 -> 1.3/1.2 a second.
        std::env::var("REIMS_VGPU_LAB_IRQ_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(500)
    })
}

impl Drop for QemuHost<'_> {
    fn drop(&mut self) {
        self.flush_deferred_irqs();
    }
}

impl<'a> QemuHost<'a> {
    pub fn new(
        ops: &'a ReimsVgpuHostOps,
        actions: &'a mut VecDeque<HostAction>,
        prompt: &'a parking_lot::Mutex<VecDeque<HostAction>>,
    ) -> Self {
        Self {
            ops,
            actions,
            prompt,
            deferred_gfx: None,
        }
    }

    /// TEMP (lab, `REIMS_VGPU_LAB_IRQ_BATCH`): raise the held gfx pulse, if any.
    /// The drain calls this at the end of every tranche; drop does too.
    pub fn flush_deferred_irqs(&mut self) {
        if self.deferred_gfx.take().is_some() {
            crate::runtime::drain::note_store_route("lab_irq_batch_flushed");
            self.raise_pulse(HostAction::irq_gfx());
        }
    }

    /// An IRQ pulse: raised in place when the shim can, else queued for the BH.
    fn raise_pulse(&mut self, action: HostAction) {
        let prompt = self.prompt;
        // Raised in place when the shim can: the queue's only purpose
        // for a pulse is to reach a thread holding the BQL, and on a
        // host whose interrupt injection needs none that hop is pure
        // latency — measured at ~0.2-0.5 ms mean and 5-11 ms tails,
        // long enough for a VBL pulse to coalesce into the next one.
        if let Some(pulse) = self.ops.irq_pulse {
            // SAFETY: QEMU owns the live ctx; the shim documents this
            // as callable from any thread holding the device lock.
            if unsafe { pulse(self.ops.ctx, action.kind as u32) } != 0 {
                crate::runtime::drain::note_store_route("irq_pulse_direct");
                return;
            }
        }
        let mut q = prompt.lock();
        // Coalesce: an undelivered pulse of the same kind already
        // covers this one (status bits accumulate in the r2c regs).
        //
        // That reasoning holds for a *status* the guest reads back. It
        // does not hold for an event the guest **timestamps**: a VBL
        // pulse that coalesces into an undelivered one is a vblank the
        // guest never sees, so the interval it measures between vblanks
        // is two grid periods rather than one. This counter says how
        // often it happens; `note_irq_coalesced`'s doc says why that
        // number decides a boot's frame rate.
        let coalesced = q.iter().any(|a| a.kind == action.kind);
        if !coalesced {
            // Arm the delivery clock while the queue lock is held, so
            // the stamp cannot be taken after the BH has already popped
            // this action on another thread. See `irq_wait_us`: the
            // guest cannot doorbell the drain worker until this pulse
            // reaches it, so this hop is the one candidate for
            // `gap_idle_us` that is ours.
            if q.is_empty() {
                crate::runtime::drain::note_irq_armed();
            }
            q.push_back(action);
        }
        drop(q);
        if coalesced {
            crate::runtime::drain::note_irq_coalesced(action.kind);
        }
        self.notify_actions();
    }

    fn notify_actions(&self) {
        if let Some(f) = self.ops.notify_actions {
            // SAFETY: QEMU owns ctx; thread-safe oneshot BH schedule.
            unsafe { f(self.ops.ctx) }
        }
    }

    fn callback_decline(error: MemError, address: u64, len: usize, discriminant: u64) -> MemError {
        callback_decline(error, address, len, discriminant)
    }
}

/// Record that a HostOps callback refused or was missing, and pass the error on.
///
/// Free rather than a method of [`QemuHost`] because [`WorkerHost`] reads
/// through the same table and must decline in the same words.
fn callback_decline(error: MemError, address: u64, len: usize, discriminant: u64) -> MemError {
    crate::observe::Emit::decline("qemu_host_callback", &error)
        .field("address", format!("{address:#x}"))
        .field("len", len)
        .fail_once(discriminant);
    error
}

/// The read side of the table, callable from any thread the shim knows about.
///
/// One body for [`QemuHost`] (the drain thread) and [`WorkerHost`] (a pool
/// worker): what a read or a dirty-tracker query means, and how it declines,
/// must not depend on which thread asked. Every one takes `&self` and touches
/// nothing but the table and the caller's buffer.
impl ReimsVgpuHostOps {
    /// [`HostMemory::read_gpa`] through the shim's `read_gpa`.
    pub(crate) fn read_guest(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        let Some(f) = self.read_gpa else {
            return Err(callback_decline(
                MemError::QemuReadGpaCallbackMissing,
                gpa,
                buf.len(),
                0,
            ));
        };
        // SAFETY: QEMU owns ctx; buf is valid for len.
        let rc = unsafe { f(self.ctx, gpa, buf.as_mut_ptr(), buf.len()) };
        match rc {
            0 => Ok(()),
            -2 => Err(MemError::NoCpu),
            _ => Err(callback_decline(
                MemError::QemuReadGpaCallbackFailed(rc),
                gpa,
                buf.len(),
                gpa,
            )),
        }
    }

    /// [`HostOps::is_ram_gpa`]: an absent callback answers "yes".
    pub(crate) fn is_ram(&self, gpa: u64) -> bool {
        match self.is_ram_gpa {
            // SAFETY: QEMU owns ctx; pure address-space query.
            Some(f) => unsafe { f(self.ctx, gpa) != 0 },
            // Older/missing table: do not invent a reject (map_pages still RAM-checks).
            None => true,
        }
    }

    /// [`HostOps::guest_write_gen`].
    pub(crate) fn write_gen(&self, token: u64) -> Option<u64> {
        if token == 0 {
            return None;
        }
        let f = self.guest_write_gen?;
        // SAFETY: QEMU owns ctx; the shim answers from its own lock.
        let gen_ = unsafe { f(self.ctx, token) };
        // 0 is the shim's "unknown token, or not yet armed" answer. Reading it
        // as a generation would let two pre-arm checks agree and vouch for a
        // resident nothing has ever validated.
        (gen_ != 0).then_some(gen_)
    }

    /// [`HostOps::guest_write_gen_lagged`].
    pub(crate) fn write_gen_lagged(&self, token: u64) -> Option<u64> {
        // TEMP (lab): the shim reads bit 63 of the token as "answer from the
        // last sync, do not sync now" (REIMS_VGPU_DIRTY_TOKEN_LAGGED). Tokens
        // are a sequential counter and never reach it.
        if token == 0 || token >> 63 != 0 {
            return None;
        }
        let f = self.guest_write_gen?;
        // SAFETY: QEMU owns ctx; the shim answers from its own lock.
        let gen_ = unsafe { f(self.ctx, token | (1 << 63)) };
        (gen_ != 0).then_some(gen_)
    }

    /// [`HostOps::guest_written_pages`].
    pub(crate) fn written_pages(&self, token: u64, since_gen: u64) -> Option<Vec<u64>> {
        if token == 0 || since_gen == 0 {
            return None;
        }
        let f = self.guest_written_pages?;
        // The shim refuses rather than truncating, so the buffer has to be able
        // to hold a whole surface's page list. A display-sized BGRA8 surface is
        // ~2 000 x86 pages; the cap is a few times that so a larger one still
        // gets an exact answer, and a set beyond it declines into "assume the
        // whole surface was written" rather than into a short list.
        const MAX_PAGES: usize = 16 * 1024;
        let mut out = vec![0u64; MAX_PAGES];
        // SAFETY: QEMU owns ctx; `out` is writable for MAX_PAGES u64s for the
        // duration of the call, and the shim writes at most `max` of them.
        let n = unsafe { f(self.ctx, token, since_gen, out.as_mut_ptr(), MAX_PAGES) };
        if n < 0 {
            return None;
        }
        let n = usize::try_from(n).ok()?;
        if n > MAX_PAGES {
            return None;
        }
        out.truncate(n);
        Some(out)
    }
}

impl HostMemory for QemuHost<'_> {
    fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError> {
        self.ops.read_guest(gpa, buf)
    }

    fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        crate::backend::settle_deferred_draws_for_guest_write();
        let Some(f) = self.ops.write_gpa else {
            return Err(Self::callback_decline(
                MemError::QemuWriteGpaCallbackMissing,
                gpa,
                buf.len(),
                0,
            ));
        };
        // SAFETY: QEMU owns ctx; buf is valid for len.
        let rc = unsafe { f(self.ops.ctx, gpa, buf.as_ptr(), buf.len()) };
        if rc == 0 {
            // The only `HostMemory::write_gpa` that reaches real guest RAM, so
            // it is where that whole funnel is recorded. `FakeHost` deliberately
            // does not mark: a fixture's writes are not this device's, and
            // counting them would put test addresses in a set whose entire
            // purpose is to be compared against a live guest's panic.
            crate::observe::footprint::note_written_range(gpa, buf.len() as u64);
            Ok(())
        } else {
            Err(Self::callback_decline(
                MemError::QemuWriteGpaCallbackFailed(rc),
                gpa,
                buf.len(),
                gpa,
            ))
        }
    }
}

impl HostOps for QemuHost<'_> {
    fn mono_ns(&self) -> u64 {
        match self.ops.mono_ns {
            // SAFETY: QEMU owns ctx.
            Some(f) => unsafe { f(self.ops.ctx) },
            None => {
                QemuHostDecline::MonoNsCallbackMissing.emit(0);
                0
            }
        }
    }

    fn enqueue(&mut self, action: HostAction) {
        // Prompt rail: IRQ pulses and cursor moves carry no device state and
        // must not wait for the drain tranche to finish. Push to the slot
        // queue (poppable without the device lock) and wake the delivery BH.
        let prompt = self.prompt;
        match action.kind {
            // TEMP (lab): a control word a guest thread is blocked on skips the
            // hold (`lab_ctrl_stamp`); the pulse below also carries any held one.
            HostActionKind::IrqGfxPulse if crate::runtime::drain::lab_ctrl_stamp::urgent() => {
                self.deferred_gfx = None;
                crate::runtime::drain::note_store_route("lab_ctrl_stamp_irq_now");
                self.raise_pulse(action);
                return;
            }
            HostActionKind::IrqGfxPulse if lab_irq_batch_us() > 0 => {
                let now = crate::observe::elapsed_us();
                match self.deferred_gfx {
                    None => {
                        self.deferred_gfx = Some(now);
                        crate::runtime::drain::note_store_route("lab_irq_batch_held");
                    }
                    Some(first) if now.saturating_sub(first) >= lab_irq_batch_us() => {
                        self.deferred_gfx = None;
                        crate::runtime::drain::note_store_route("lab_irq_batch_budget");
                        self.raise_pulse(action);
                    }
                    Some(_) => crate::runtime::drain::note_store_route("lab_irq_batch_folded"),
                }
                return;
            }
            HostActionKind::IrqGfxPulse | HostActionKind::IrqIosfcPulse => {
                self.raise_pulse(action);
                return;
            }
            HostActionKind::CursorUpdate => {
                let mut q = prompt.lock();
                q.retain(|a| a.kind != HostActionKind::CursorUpdate);
                if q.is_empty() {
                    crate::runtime::drain::note_irq_armed();
                }
                q.push_back(action);
                drop(q);
                self.notify_actions();
                return;
            }
            HostActionKind::InputKey
            | HostActionKind::InputPointerMove
            | HostActionKind::InputPointerButton
            | HostActionKind::WindowClosed => {
                // Host-window input + the window-closed signal: ordered and
                // lossless. Unlike cursor moves and IRQ pulses these must NOT
                // coalesce — a dropped key-up sticks a modifier, a reordered
                // move+click lands the click at the wrong spot, and a dropped
                // WindowClosed would leave the VM running headless. Push in
                // arrival order and wake the delivery BH so the guest (or the
                // shutdown path) sees it without waiting for a drain tranche.
                prompt.lock().push_back(action);
                self.notify_actions();
                return;
            }
            _ => {}
        }
        // apple-gfx new_frame_handler_bh: drop frames when guest gets too far
        // ahead of encode (pending_frames >= 2). Product: coalesce pending
        // ScanoutUpdates so BH paint encodes current +0x188 once (latest
        // presentFrame), not a backlog of dual-mid halves.
        if action.kind == HostActionKind::ScanoutUpdate {
            self.actions
                .retain(|a| a.kind != HostActionKind::ScanoutUpdate);
        }
        self.actions.push_back(action);
    }

    fn schedule_bh(&mut self) {
        if let Some(f) = self.ops.schedule_bh {
            // SAFETY: QEMU owns ctx; schedules oneshot BH (apple-gfx pattern).
            unsafe { f(self.ops.ctx) }
        } else {
            QemuHostDecline::ScheduleBhCallbackMissing.emit(0);
        }
    }

    fn read_kva(&self, kva: u64, buf: &mut [u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        let Some(f) = self.ops.read_kva else {
            return Err(Self::callback_decline(
                MemError::QemuReadKvaCallbackMissing,
                kva,
                buf.len(),
                0,
            ));
        };
        // SAFETY: QEMU owns ctx; buf valid for len.
        let rc = unsafe { f(self.ops.ctx, kva, buf.as_mut_ptr(), buf.len()) };
        match rc {
            0 => Ok(()),
            -2 => Err(MemError::NoCpu),
            _ => Err(Self::callback_decline(
                MemError::QemuReadKvaCallbackFailed(rc),
                kva,
                buf.len(),
                kva,
            )),
        }
    }

    fn read_xreg(&self, index: u32) -> Result<u64, MemError> {
        let Some(f) = self.ops.read_xreg else {
            return Err(Self::callback_decline(
                MemError::QemuReadXregCallbackMissing,
                u64::from(index),
                std::mem::size_of::<u64>(),
                0,
            ));
        };
        let mut out = 0u64;
        // SAFETY: QEMU owns ctx; out is stack local.
        let rc = unsafe { f(self.ops.ctx, index, &mut out) };
        if rc == 0 {
            Ok(out)
        } else {
            Err(Self::callback_decline(
                MemError::QemuReadXregCallbackFailed(rc),
                u64::from(index),
                std::mem::size_of::<u64>(),
                u64::from(index),
            ))
        }
    }

    fn map_pages(&mut self, gpas: &[u64], page_size: usize) -> Option<usize> {
        if gpas.is_empty() {
            return None;
        }
        // QEMU C side uses the device guest page shift (x86 4 KiB / arm 16 KiB).
        let first_gpa = gpas[0];
        let Some(f) = self.ops.map_pages else {
            QemuHostDecline::MapPagesCallbackMissing {
                first_gpa,
                page_count: gpas.len(),
                page_size,
            }
            .emit(first_gpa);
            return None;
        };
        let mut out: *mut c_void = std::ptr::null_mut();
        let mut raw_failure = ReimsVgpuMapPagesFailure::default();
        // SAFETY: QEMU owns ctx; gpas valid for count; out is stack local.
        let rc = unsafe {
            f(
                self.ops.ctx,
                gpas.as_ptr(),
                gpas.len(),
                &mut out,
                &mut raw_failure,
            )
        };
        if rc != 0 {
            let failure = match raw_failure.stage {
                MAP_PAGES_FAILURE_RESERVATION => MapPagesFailure::ReservationFailed {
                    host_errno: raw_failure.host_errno,
                },
                MAP_PAGES_FAILURE_ALIAS => MapPagesFailure::AliasFailed {
                    page_index: raw_failure.page_index,
                    host_errno: raw_failure.host_errno,
                },
                MAP_PAGES_FAILURE_INVALID_GUEST_PAGE => MapPagesFailure::InvalidGuestPage {
                    page_index: raw_failure.page_index,
                },
                MAP_PAGES_FAILURE_NONE => MapPagesFailure::Unspecified {
                    stage: MAP_PAGES_FAILURE_NONE,
                    host_errno: raw_failure.host_errno,
                },
                stage => MapPagesFailure::Unspecified {
                    stage,
                    host_errno: raw_failure.host_errno,
                },
            };
            QemuHostDecline::MapPagesCallbackFailed {
                rc,
                failure,
                first_gpa,
                page_count: gpas.len(),
                page_size,
            }
            .emit(first_gpa);
            return None;
        }
        if out.is_null() {
            QemuHostDecline::MapPagesNullPointer {
                first_gpa,
                page_count: gpas.len(),
                page_size,
            }
            .emit(first_gpa);
            return None;
        }
        Some(out as usize)
    }

    fn map_pages_stable(&self) -> bool {
        self.ops.map_pages_stable != 0
    }

    fn note_work_scope(&self, scope: crate::runtime::host::WorkScope) {
        if let Some(callback) = self.ops.note_work_scope {
            let raw = match scope {
                crate::runtime::host::WorkScope::All => crate::qemu::abi::REIMS_VGPU_WORK_SCOPE_ALL,
                crate::runtime::host::WorkScope::Channel(ch) => ch,
                // TEMP lab: census notifications, told apart by their high bits.
                crate::runtime::host::WorkScope::PacketRead(ch) => 0x8000_0000 | ch,
                crate::runtime::host::WorkScope::RingDrained(ch) => 0x4000_0000 | ch,
                crate::runtime::host::WorkScope::RingEmpty(ch) => 0x2000_0000 | ch,
            };
            // SAFETY: QEMU owns the live ctx; the callback takes a leaf lock.
            unsafe { callback(self.ops.ctx, raw) }
        }
    }

    fn note_ring_tail(&self, channel: u32, tail: u32) {
        if let Some(callback) = self.ops.note_ring_tail {
            // SAFETY: QEMU owns the live ctx; the callback takes a leaf lock.
            unsafe { callback(self.ops.ctx, channel, tail) }
        }
    }

    fn harvests_settled(&self) -> bool {
        match self.ops.harvests_settled {
            // SAFETY: QEMU owns the live ctx; the callback never blocks.
            Some(callback) => (unsafe { callback(self.ops.ctx) }) != 0,
            None => true,
        }
    }

    fn page_alias_census(&self) -> Option<PageAliasCensus> {
        let callback = self.ops.page_alias_census?;
        let mut out = PageAliasCensus::default();
        // SAFETY: `out` is writable for this call and QEMU owns the live ctx.
        (unsafe { callback(self.ops.ctx, &mut out) } == 0).then_some(out)
    }

    fn guest_ram_regions(
        &mut self,
    ) -> Result<Vec<crate::runtime::guest_ram::GuestRamRegion>, GuestRamRegionsError> {
        /// Spans the first call asks for.
        ///
        /// An x86 machine with a PCI hole has two RAMBlocks and a vmapple one
        /// has one, so this is already several times the answer. It is a first
        /// guess and not a bound: a host with more spans reports its total and
        /// gets asked again at that size, which is why nothing here is a cap
        /// that could silently import part of the guest's memory.
        const FIRST_TRY: usize = 8;

        let f = self
            .ops
            .guest_ram_regions
            .ok_or(GuestRamRegionsError::CallbackMissing)?;
        let mut buf = vec![crate::runtime::guest_ram::GuestRamRegion::default(); FIRST_TRY];
        // SAFETY: QEMU owns ctx and keeps it valid for the device lifetime;
        // `buf` is valid for `len` writes of the shared `#[repr(C)]` struct for
        // the duration of the call.
        let mut rc = unsafe { f(self.ops.ctx, buf.as_mut_ptr(), buf.len()) };
        if rc < 0 {
            return Err(GuestRamRegionsError::from_code(rc));
        }
        let mut total = rc as usize;
        if total > buf.len() {
            buf.resize(total, crate::runtime::guest_ram::GuestRamRegion::default());
            // SAFETY: as above, with the array the shim's own total sized.
            rc = unsafe { f(self.ops.ctx, buf.as_mut_ptr(), buf.len()) };
            if rc < 0 {
                return Err(GuestRamRegionsError::from_code(rc));
            }
            let retried = rc as usize;
            if retried > buf.len() {
                return Err(GuestRamRegionsError::StillTruncated {
                    total: retried,
                    capacity: buf.len(),
                });
            }
            total = retried;
        }
        buf.truncate(total);
        Ok(buf)
    }

    fn unmap_pages(&mut self, ptr: usize, len: usize) {
        if ptr == 0 || len == 0 {
            return;
        }
        if let Some(f) = self.ops.unmap_pages {
            // SAFETY: ptr/len came from a successful map_pages.
            unsafe { f(self.ops.ctx, ptr as *mut c_void, len) }
        } else if !self.map_pages_stable() {
            // Stable aliases explicitly require no release. A missing callback
            // is a leak only for a transient view.
            QemuHostDecline::UnmapPagesCallbackMissing { ptr, len }.emit(ptr as u64);
        }
    }

    fn is_ram_gpa(&self, gpa: u64) -> bool {
        self.ops.is_ram(gpa)
    }

    fn track_guest_writes(&mut self, gpas: &[u64], page_size: usize) -> Option<u64> {
        if gpas.is_empty() || page_size == 0 {
            return None;
        }
        let f = self.ops.track_guest_writes?;
        // SAFETY: QEMU owns ctx; the slice outlives the call and the shim
        // copies the page list before returning.
        let token = unsafe { f(self.ops.ctx, gpas.as_ptr(), gpas.len(), page_size) };
        // A shim that cannot track answers 0 rather than failing. Not a
        // decline: "this host has no dirty bitmap" is a capability, and every
        // consumer already has to handle it on the fixture hosts.
        (token != 0).then_some(token)
    }

    fn untrack_guest_writes(&mut self, token: u64) {
        if token == 0 {
            return;
        }
        if let Some(f) = self.ops.untrack_guest_writes {
            // SAFETY: token came from a successful track_guest_writes.
            unsafe { f(self.ops.ctx, token) }
        }
    }

    #[track_caller]
    fn guest_write_gen(&self, token: u64) -> Option<u64> {
        if !lab_gen_prof::on() {
            return self.ops.write_gen(token);
        }
        let site = std::panic::Location::caller();
        let t0 = std::time::Instant::now();
        let answer = self.ops.write_gen(token);
        lab_gen_prof::note(site, t0.elapsed());
        answer
    }

    fn guest_write_gen_lagged(&self, token: u64) -> Option<u64> {
        self.ops.write_gen_lagged(token)
    }

    fn guest_write_gen_deferred(
        &self,
        token: u64,
    ) -> Option<crate::runtime::write_baseline::PendingGen> {
        if token == 0 || self.ops.guest_write_gen.is_none() {
            return None;
        }
        crate::runtime::write_baseline::issue(token)
    }

    fn guest_written_pages(&self, token: u64, since_gen: u64) -> Option<Vec<u64>> {
        self.ops.written_pages(token, since_gen)
    }
}

/// TEMP (lab, `REIMS_VGPU_LAB_GEN_PROF=on`): the drain's synced generation
/// reads timed per call site (`#[track_caller]`), into the per-second
/// `store_routes` window as `lab_gw_<file>_<line>_n` / `_us` / `_slow` (>= 50 us,
/// i.e. a read that synced rather than hit the epoch's answer).
mod lab_gen_prof {
    use std::collections::HashMap;
    use std::panic::Location;
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    type Names = (&'static str, &'static str, &'static str);

    pub(super) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            matches!(std::env::var("REIMS_VGPU_LAB_GEN_PROF").as_deref(), Ok("on"))
        })
    }

    fn names(site: &'static Location<'static>) -> Names {
        static SITES: OnceLock<Mutex<HashMap<(&'static str, u32), Names>>> = OnceLock::new();
        let map = SITES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = map.lock().unwrap_or_else(|e| e.into_inner());
        *map.entry((site.file(), site.line())).or_insert_with(|| {
            let parts: Vec<&str> = site
                .file()
                .split(['/', '\\'])
                .rev()
                .take(2)
                .collect();
            let file: String = parts
                .iter()
                .rev()
                .map(|p| p.trim_end_matches(".rs"))
                .collect::<Vec<_>>()
                .join("_")
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
                .collect();
            let base = format!("lab_gw_{file}_{}", site.line());
            (
                Box::leak(format!("{base}_n").into_boxed_str()),
                Box::leak(format!("{base}_us").into_boxed_str()),
                Box::leak(format!("{base}_slow").into_boxed_str()),
            )
        })
    }

    pub(super) fn note(site: &'static Location<'static>, elapsed: Duration) {
        use crate::runtime::drain::census::{note_store_route, note_store_route_n};
        let (n, us_name, slow) = names(site);
        let us = elapsed.as_micros() as u64;
        note_store_route(n);
        note_store_route_n(us_name, us);
        if us >= 50 {
            note_store_route(slow);
        }
    }
}

/// Host used when no QEMU ops table is bound (unit tests / headless create).
pub struct NullHost;

impl HostMemory for NullHost {
    fn read_gpa(&self, _gpa: u64, _buf: &mut [u8]) -> Result<(), MemError> {
        Err(MemError::Unmapped)
    }
    fn write_gpa(&mut self, _gpa: u64, _buf: &[u8]) -> Result<(), MemError> {
        Err(MemError::Unmapped)
    }
}

impl HostOps for NullHost {
    fn mono_ns(&self) -> u64 {
        0
    }
    fn enqueue(&mut self, _action: HostAction) {}
    fn schedule_bh(&mut self) {}
}

/// What a parallel-pool worker may ask the shim: the read side of the table,
/// and nothing else.
///
/// The worker threads of [`crate::runtime::par`] run on behalf of a transaction
/// they do not own. Everything a transaction does that the guest can observe —
/// a write into guest pages, an interrupt, a scanout, the order stamps appear
/// in — is the commit's, on the drain thread, in the order the ordering core
/// chose. So a worker's view of the host is the subset that cannot be seen:
///
/// | Call | Worker | Why |
/// |---|---|---|
/// | `read_gpa` | yes | `address_space_read`; the thread is RCU-registered |
/// | `is_ram_gpa`, `guest_write_gen`, `guest_written_pages` | yes | pure queries, safe from any thread |
/// | `write_gpa` | **refused** ([`MemError::ReadOnlyHost`]) | ordering, `host_writes`, stamps |
/// | `enqueue`, `schedule_bh` | **dropped and counted** | `&mut actions` belong to the drain |
/// | `read_kva`, `read_xreg`, `map_pages` | default (refuse / absent) | `current_cpu`, registry mutation |
/// | `note_work_scope` | default (nothing) | the drain set the scope for this tranche and it binds the worker's reads too |
///
/// It is a *view*: `Copy`-sized, built per worker from the table the device was
/// created with, and holds no state of its own.
pub struct WorkerHost<'a> {
    ops: &'a ReimsVgpuHostOps,
}

impl<'a> WorkerHost<'a> {
    pub fn new(ops: &'a ReimsVgpuHostOps) -> Self {
        Self { ops }
    }

    fn misuse(what: &'static str) {
        crate::runtime::drain::note_store_route("par_worker_host_misuse");
        crate::observe::fail(format!(
            "par_worker_host_misuse call={what} (a pool worker reached a HostOps call that \
             belongs to the drain thread; dropped)"
        ));
    }
}

impl HostMemory for WorkerHost<'_> {
    fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError> {
        self.ops.read_guest(gpa, buf)
    }

    fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError> {
        Err(callback_decline(
            MemError::ReadOnlyHost,
            gpa,
            buf.len(),
            gpa,
        ))
    }
}

impl HostOps for WorkerHost<'_> {
    fn mono_ns(&self) -> u64 {
        match self.ops.mono_ns {
            // SAFETY: QEMU owns ctx; a pure clock read.
            Some(f) => unsafe { f(self.ops.ctx) },
            None => 0,
        }
    }

    fn enqueue(&mut self, _action: HostAction) {
        Self::misuse("enqueue");
    }

    fn schedule_bh(&mut self) {
        Self::misuse("schedule_bh");
    }

    fn is_ram_gpa(&self, gpa: u64) -> bool {
        self.ops.is_ram(gpa)
    }

    fn guest_write_gen(&self, token: u64) -> Option<u64> {
        self.ops.write_gen(token)
    }

    fn guest_write_gen_lagged(&self, token: u64) -> Option<u64> {
        self.ops.write_gen_lagged(token)
    }

    fn guest_written_pages(&self, token: u64, since_gen: u64) -> Option<Vec<u64>> {
        self.ops.written_pages(token, since_gen)
    }

    /// The host's own answer, not the trait's conservative default. Aliasing
    /// is a property of the host (`map_pages_stable=1` on x86), not of which
    /// thread asks, and the whole point of a stable alias is that the worker
    /// can read through the pointers the drain mapped without remapping them.
    ///
    /// While this answered `false`, every pool-planned ref-view bind declined
    /// at `guest_run_alias_available` (`par_prefetch_bail_zero_copy_declined`,
    /// the S1 run's zero ref-view plans whose reason was not chased down) —
    /// the plan could never see the rail it exists for.
    fn map_pages_stable(&self) -> bool {
        self.ops.map_pages_stable != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::host::HostAction as HA;

    unsafe extern "C" fn fail_read_gpa(
        _ctx: *mut c_void,
        _gpa: u64,
        _buf: *mut u8,
        _len: usize,
    ) -> i32 {
        -7
    }

    unsafe extern "C" fn fail_write_gpa(
        _ctx: *mut c_void,
        _gpa: u64,
        _buf: *const u8,
        _len: usize,
    ) -> i32 {
        -8
    }

    unsafe extern "C" fn no_cpu_read_kva(
        _ctx: *mut c_void,
        _kva: u64,
        _buf: *mut u8,
        _len: usize,
    ) -> i32 {
        -2
    }

    unsafe extern "C" fn fail_read_xreg(_ctx: *mut c_void, _index: u32, _out: *mut u64) -> i32 {
        -9
    }

    unsafe extern "C" fn fail_map_pages(
        _ctx: *mut c_void,
        _gpas: *const u64,
        _count: usize,
        _out: *mut *mut c_void,
        failure: *mut ReimsVgpuMapPagesFailure,
    ) -> i32 {
        // SAFETY: the HostOps callback contract supplies a writable failure slot.
        unsafe {
            *failure = ReimsVgpuMapPagesFailure {
                stage: MAP_PAGES_FAILURE_ALIAS,
                host_errno: 12,
                page_index: 3,
            };
        }
        -11
    }

    unsafe extern "C" fn null_map_pages(
        _ctx: *mut c_void,
        _gpas: *const u64,
        _count: usize,
        out: *mut *mut c_void,
        _failure: *mut ReimsVgpuMapPagesFailure,
    ) -> i32 {
        // SAFETY: the HostOps callback contract supplies a writable out slot.
        unsafe {
            *out = std::ptr::null_mut();
        }
        0
    }

    /// How many spans [`counting_ram_regions`] claims, and how many times it has
    /// been asked. Process-global, which the suite's serial convention
    /// (`--test-threads=1`) makes safe; each test that uses them sets both.
    static FAKE_RAM_SPANS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static FAKE_RAM_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// A shim with [`FAKE_RAM_SPANS`] spans that honours `max` and reports its
    /// true total — the contract the header states, including the case where
    /// the total is larger than the array it was given.
    unsafe extern "C" fn counting_ram_regions(
        _ctx: *mut c_void,
        out: *mut crate::runtime::guest_ram::GuestRamRegion,
        max: usize,
    ) -> i32 {
        use std::sync::atomic::Ordering::Relaxed;
        FAKE_RAM_CALLS.fetch_add(1, Relaxed);
        let total = FAKE_RAM_SPANS.load(Relaxed);
        for i in 0..total.min(max) {
            // SAFETY: the caller's contract is that `out` is writable for `max`
            // entries, and `i < max`.
            unsafe {
                *out.add(i) = crate::runtime::guest_ram::GuestRamRegion {
                    gpa_base: (i as u64) << 32,
                    host_va: 0x7f00_0000_0000 + ((i as u64) << 32),
                    len: 0x1000,
                };
            }
        }
        total as i32
    }

    /// A shim that always claims one more span than it was asked for, whatever
    /// the array size. The retry cannot converge against it.
    unsafe extern "C" fn always_short_ram_regions(
        _ctx: *mut c_void,
        _out: *mut crate::runtime::guest_ram::GuestRamRegion,
        max: usize,
    ) -> i32 {
        (max + 1) as i32
    }

    unsafe extern "C" fn no_ram_regions(
        _ctx: *mut c_void,
        _out: *mut crate::runtime::guest_ram::GuestRamRegion,
        _max: usize,
    ) -> i32 {
        crate::qemu::abi::REIMS_VGPU_GUEST_RAM_ERR_NO_RAM
    }

    unsafe extern "C" fn future_code_ram_regions(
        _ctx: *mut c_void,
        _out: *mut crate::runtime::guest_ram::GuestRamRegion,
        _max: usize,
    ) -> i32 {
        -99
    }

    fn ram_regions_with(
        callback: Option<
            unsafe extern "C" fn(
                *mut c_void,
                *mut crate::runtime::guest_ram::GuestRamRegion,
                usize,
            ) -> i32,
        >,
    ) -> Result<Vec<crate::runtime::guest_ram::GuestRamRegion>, GuestRamRegionsError> {
        let mut ops = ReimsVgpuHostOps::null();
        ops.guest_ram_regions = callback;
        let mut actions = VecDeque::new();
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        // Bound first so the host (it flushes on drop) is gone before the locals
        // it borrows; as a tail expression its temporary would outlive them.
        let regions = QemuHost::new(&ops, &mut actions, &prompt).guest_ram_regions();
        regions
    }

    /// A shim older than v17 has no such callback, and that is a named refusal
    /// rather than an empty answer. An empty `Vec` would read as "this machine
    /// has no RAM", which is a different thing to go fix.
    #[test]
    fn a_shim_without_the_callback_refuses_by_name() {
        assert_eq!(
            ram_regions_with(None),
            Err(GuestRamRegionsError::CallbackMissing)
        );
    }

    /// The ordinary machine: fewer spans than the first array, answered in one
    /// call. Two calls here would be two address-space walks per boot for
    /// nothing.
    #[test]
    fn a_machine_that_fits_the_first_array_is_asked_once() {
        use std::sync::atomic::Ordering::Relaxed;
        FAKE_RAM_SPANS.store(2, Relaxed);
        FAKE_RAM_CALLS.store(0, Relaxed);
        let regions = ram_regions_with(Some(counting_ram_regions)).expect("two spans");
        assert_eq!(regions.len(), 2);
        assert_eq!(FAKE_RAM_CALLS.load(Relaxed), 1);
        assert_eq!(regions[0].gpa_base, 0);
        assert_eq!(regions[1].gpa_base, 1 << 32);
        assert_eq!(regions[1].host_va, 0x7f00_0000_0000 + (1 << 32));
    }

    /// More spans than the first array: the caller learns its array was short
    /// from the total, grows to it, and comes back with every span.
    ///
    /// This is the case that must not silently truncate. A device that imported
    /// the first eight spans and dropped the rest would run the copying rails
    /// for part of the guest's RAM with nothing in the log saying which part —
    /// the "no silent caps" rule in `AGENTS.md`, at the one call that decides
    /// how much of guest memory the GPU can reach at all.
    #[test]
    fn a_machine_with_more_spans_than_the_first_array_still_gets_all_of_them() {
        use std::sync::atomic::Ordering::Relaxed;
        FAKE_RAM_SPANS.store(21, Relaxed);
        FAKE_RAM_CALLS.store(0, Relaxed);
        let regions = ram_regions_with(Some(counting_ram_regions)).expect("21 spans");
        assert_eq!(regions.len(), 21, "the answer must not be capped");
        assert_eq!(FAKE_RAM_CALLS.load(Relaxed), 2, "one probe, one full read");
        assert_eq!(regions[20].gpa_base, 20 << 32);
        assert!(
            regions.iter().all(|r| r.len == 0x1000),
            "the grown array must be filled, not left at its default"
        );
    }

    /// A shim whose total keeps growing does not loop and does not truncate: it
    /// refuses, naming both numbers. RAMBlock mappings do not change, so this is
    /// a shim defect rather than a race to retry through.
    #[test]
    fn a_total_that_never_converges_is_refused_rather_than_retried() {
        assert_eq!(
            ram_regions_with(Some(always_short_ram_regions)),
            // The probe of 8 was answered 9, the retry of 9 answered 10, and
            // the second answer is where it stops. The numbers reported are the
            // retry's, which is the pair that shows the total moved.
            Err(GuestRamRegionsError::StillTruncated {
                total: 10,
                capacity: 9,
            })
        );
    }

    /// Each negative code keeps its own check, and a code this build has no name
    /// for carries the number rather than being folded into a neighbour.
    #[test]
    fn each_refusal_code_keeps_its_own_check() {
        assert_eq!(
            ram_regions_with(Some(no_ram_regions)),
            Err(GuestRamRegionsError::NoRam)
        );
        assert_eq!(
            ram_regions_with(Some(future_code_ram_regions)),
            Err(GuestRamRegionsError::UnknownCode(-99))
        );
    }

    #[test]
    fn enqueue_routes_prompt_kinds_to_prompt_queue() {
        let ops = ReimsVgpuHostOps::null();
        let mut actions = VecDeque::new();
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        let mut host = QemuHost::new(&ops, &mut actions, &prompt);

        host.enqueue(HA::irq_gfx());
        host.enqueue(HA::irq_gfx()); // coalesced: one undelivered pulse covers both
        host.enqueue(HA::irq_iosfc());
        host.enqueue(HA::cursor(10, 20, true));
        host.enqueue(HA::cursor(30, 40, true)); // latest cursor wins
        host.enqueue(HA::scanout_gen(1, 1920, 1080, 7));
        drop(host);

        let q = prompt.lock();
        assert_eq!(q.len(), 3, "gfx pulse + iosfc pulse + one cursor");
        assert_eq!(q[0].kind, HostActionKind::IrqGfxPulse);
        assert_eq!(q[1].kind, HostActionKind::IrqIosfcPulse);
        assert_eq!(q[2].kind, HostActionKind::CursorUpdate);
        assert_eq!(q[2].a0, 30);
        drop(q);
        assert_eq!(actions.len(), 1, "scanout stays on the lock-owning rail");
        assert_eq!(actions[0].kind, HostActionKind::ScanoutUpdate);
    }

    #[test]
    fn missing_qemu_memory_callbacks_are_exact() {
        let ops = ReimsVgpuHostOps::null();
        let mut actions = VecDeque::new();
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        let mut host = QemuHost::new(&ops, &mut actions, &prompt);
        assert_eq!(
            host.read_gpa(0x1000, &mut [0; 1]),
            Err(MemError::QemuReadGpaCallbackMissing)
        );
        assert_eq!(
            host.write_gpa(0x1000, &[1]),
            Err(MemError::QemuWriteGpaCallbackMissing)
        );
        assert_eq!(
            host.read_kva(0xffff_fe00_1000, &mut [0; 1]),
            Err(MemError::QemuReadKvaCallbackMissing)
        );
        assert_eq!(
            host.read_xreg(19),
            Err(MemError::QemuReadXregCallbackMissing)
        );
    }

    #[test]
    fn qemu_callback_return_codes_keep_their_operation_and_value() {
        let mut ops = ReimsVgpuHostOps::null();
        ops.read_gpa = Some(fail_read_gpa);
        ops.write_gpa = Some(fail_write_gpa);
        ops.read_kva = Some(no_cpu_read_kva);
        ops.read_xreg = Some(fail_read_xreg);
        let mut actions = VecDeque::new();
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        let mut host = QemuHost::new(&ops, &mut actions, &prompt);
        assert_eq!(
            host.read_gpa(0x1000, &mut [0; 1]),
            Err(MemError::QemuReadGpaCallbackFailed(-7))
        );
        assert_eq!(
            host.write_gpa(0x1000, &[1]),
            Err(MemError::QemuWriteGpaCallbackFailed(-8))
        );
        assert_eq!(
            host.read_kva(0xffff_fe00_1000, &mut [0; 1]),
            Err(MemError::NoCpu),
            "-2 is the no-current-vCPU state, not an unmapped KVA"
        );
        assert_eq!(
            host.read_xreg(22),
            Err(MemError::QemuReadXregCallbackFailed(-9))
        );
        assert_eq!(
            crate::observe::Emit::decline(
                "qemu_host_callback",
                &MemError::QemuReadGpaCallbackFailed(-7),
            )
            .field("address", "0x1000")
            .field("len", 4)
            .render(),
            "qemu_host_callback reason=mem_qemu_read_gpa_callback_failed \
             rc=-7 address=0x1000 len=4"
        );
    }

    #[test]
    fn qemu_host_adapter_declines_are_exact_registered_and_log_safe() {
        use crate::observe::Decline;

        let declines = [
            QemuHostDecline::MonoNsCallbackMissing,
            QemuHostDecline::ScheduleBhCallbackMissing,
            QemuHostDecline::MapPagesCallbackMissing {
                first_gpa: 0x4000,
                page_count: 2,
                page_size: 0x4000,
            },
            QemuHostDecline::MapPagesCallbackFailed {
                rc: -11,
                failure: MapPagesFailure::AliasFailed {
                    page_index: 3,
                    host_errno: 12,
                },
                first_gpa: 0x4000,
                page_count: 2,
                page_size: 0x4000,
            },
            QemuHostDecline::MapPagesNullPointer {
                first_gpa: 0x4000,
                page_count: 2,
                page_size: 0x4000,
            },
            QemuHostDecline::UnmapPagesCallbackMissing {
                ptr: 0x10000,
                len: 0x8000,
            },
        ];
        let expected = [
            "qemu_mono_ns_callback_missing",
            "qemu_schedule_bh_callback_missing",
            "qemu_map_pages_callback_missing",
            "qemu_map_pages_alias_failed",
            "qemu_map_pages_null_pointer",
            "qemu_unmap_pages_callback_missing",
        ];
        assert_eq!(declines.len(), expected.len());
        for (decline, expected_slug) in declines.iter().zip(expected) {
            assert_eq!(decline.slug(), expected_slug);
            assert!(expected_slug
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_' || byte.is_ascii_digit()));
        }
        assert_eq!(
            crate::observe::Emit::decline("qemu_host_adapter", &declines[3]).render(),
            "qemu_host_adapter reason=qemu_map_pages_alias_failed \
             rc=-11 stage=alias errno=12 first_gpa=0x4000 page_count=2 \
             page_size=16384 page_index=3"
        );
    }

    #[test]
    fn map_pages_distinguishes_missing_failed_and_null_callbacks() {
        let mut ops = ReimsVgpuHostOps::null();
        let mut actions = VecDeque::new();
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        let mut host = QemuHost::new(&ops, &mut actions, &prompt);
        assert_eq!(host.map_pages(&[0x4000], 0x4000), None);
        drop(host);

        ops.map_pages = Some(fail_map_pages);
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        let mut host = QemuHost::new(&ops, &mut actions, &prompt);
        assert_eq!(host.map_pages(&[0x8000], 0x4000), None);
        drop(host);

        ops.map_pages = Some(null_map_pages);
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        let mut host = QemuHost::new(&ops, &mut actions, &prompt);
        assert_eq!(host.map_pages(&[0xc000], 0x4000), None);
    }

    // Attach and detach calls recorded per thread: `(attaches, detaches)`. A
    // `thread_local` because the contract is per calling thread, and a
    // process-global counter could not tell "the worker attached" from "some
    // other test's thread did".
    thread_local! {
        static REGISTRATIONS: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((0, 0)) };
    }

    unsafe extern "C" fn count_attach(_ctx: *mut c_void) {
        REGISTRATIONS.with(|r| r.set((r.get().0 + 1, r.get().1)));
    }

    unsafe extern "C" fn count_detach(_ctx: *mut c_void) {
        REGISTRATIONS.with(|r| r.set((r.get().0, r.get().1 + 1)));
    }

    /// The pool's structural gate: a table offers worker threads only when it
    /// offers the whole pair. One half would leave a registered RCU reader
    /// behind an exited thread, which `synchronize_rcu` then waits on forever.
    #[test]
    fn a_thread_registrar_needs_both_halves_of_the_pair() {
        let mut ops = ReimsVgpuHostOps::null();
        assert!(
            ops.thread_registrar().is_none(),
            "no callbacks, no registrar"
        );

        ops.thread_attach = Some(count_attach);
        assert!(ops.thread_registrar().is_none(), "attach alone is refused");

        ops.thread_attach = None;
        ops.thread_detach = Some(count_detach);
        assert!(ops.thread_registrar().is_none(), "detach alone is refused");

        ops.thread_attach = Some(count_attach);
        assert!(ops.thread_registrar().is_some(), "the pair is accepted");
    }

    /// Attach happens when the guard is made and detach exactly when it is
    /// dropped, on the thread that made it — the pairing the header promises the
    /// shim.
    #[test]
    fn a_thread_attachment_detaches_on_drop_on_its_own_thread() {
        let mut ops = ReimsVgpuHostOps::null();
        ops.thread_attach = Some(count_attach);
        ops.thread_detach = Some(count_detach);
        let registrar = ops.thread_registrar().expect("the pair is offered");

        // Another thread's registrations are invisible here: the counters are
        // per thread, so this is also the proof that the guard calls through on
        // the thread that holds it and nowhere else.
        let seen = std::thread::spawn(move || {
            assert_eq!(REGISTRATIONS.with(std::cell::Cell::get), (0, 0));
            let guard = registrar.attach();
            assert_eq!(REGISTRATIONS.with(std::cell::Cell::get), (1, 0));
            drop(guard);
            REGISTRATIONS.with(std::cell::Cell::get)
        })
        .join()
        .expect("the worker does not panic");
        assert_eq!(seen, (1, 1), "attach, then detach, once each");
        assert_eq!(REGISTRATIONS.with(std::cell::Cell::get), (0, 0));
    }

    /// Fake shim callbacks for the worker-host test: a read that fills the
    /// buffer with the low byte of the address, a RAM predicate that says yes
    /// below 1 MiB, and a generation that is the token plus one.
    unsafe extern "C" fn pattern_read_gpa(
        _ctx: *mut c_void,
        gpa: u64,
        buf: *mut u8,
        len: usize,
    ) -> i32 {
        for i in 0..len {
            // SAFETY: the caller's contract is `buf` writable for `len`.
            unsafe { *buf.add(i) = (gpa as u8).wrapping_add(i as u8) };
        }
        0
    }

    unsafe extern "C" fn low_ram_only(_ctx: *mut c_void, gpa: u64) -> i32 {
        i32::from(gpa < 0x10_0000)
    }

    unsafe extern "C" fn token_plus_one(_ctx: *mut c_void, token: u64) -> u64 {
        // The lagged form arrives with bit 63 set; the answer is the same.
        (token & !(1 << 63)) + 1
    }

    static LAST_WORK_SCOPE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    unsafe extern "C" fn remember_work_scope(_ctx: *mut c_void, scope: u32) {
        LAST_WORK_SCOPE.store(scope, std::sync::atomic::Ordering::SeqCst);
    }

    /// TEMP lab (`REIMS_VGPU_LAB_DIRTY_HOLE=census`): the drain tells the shim, besides the
    /// scope its reads are owed, when it read a packet and how it left a ring. The shim
    /// tells the three apart from a scope by the high bits, and a scope below 64 is a channel.
    #[test]
    fn the_work_scope_codes_the_shim_receives_are_told_apart() {
        use crate::runtime::host::{HostOps as _, WorkScope};
        let mut table = ReimsVgpuHostOps::null();
        table.note_work_scope = Some(remember_work_scope);
        let mut actions = VecDeque::new();
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        let drain = QemuHost::new(&table, &mut actions, &prompt);
        for (scope, raw) in [
            (WorkScope::All, 0xffff_ffff),
            (WorkScope::Channel(3), 3),
            (WorkScope::PacketRead(3), 0x8000_0003),
            (WorkScope::RingDrained(3), 0x4000_0003),
            (WorkScope::RingEmpty(0), 0x2000_0000),
        ] {
            drain.note_work_scope(scope);
            assert_eq!(
                LAST_WORK_SCOPE.load(std::sync::atomic::Ordering::SeqCst),
                raw,
                "{scope:?}"
            );
        }
    }

    /// A pool worker's host reads through the shim's table exactly as the drain's
    /// does — same callbacks, same declines — and cannot write, raise an
    /// interrupt or wake the drain.
    #[test]
    fn a_worker_host_reads_through_the_table_and_does_not_write() {
        use crate::runtime::host::HostMemory as _;
        let mut table = ReimsVgpuHostOps::null();
        table.read_gpa = Some(pattern_read_gpa);
        table.is_ram_gpa = Some(low_ram_only);
        table.guest_write_gen = Some(token_plus_one);

        let mut worker = WorkerHost::new(&table);
        let mut buf = [0u8; 4];
        worker.read_gpa(0x1000, &mut buf).expect("the table reads");
        assert_eq!(buf, [0x00, 0x01, 0x02, 0x03]);
        assert!(worker.is_ram_gpa(0x8000));
        assert!(!worker.is_ram_gpa(0x20_0000));
        assert_eq!(worker.guest_write_gen(41), Some(42));
        assert_eq!(worker.guest_write_gen(0), None, "0 is no token");
        assert_eq!(worker.guest_write_gen_lagged(41), Some(42));

        // The same answers as the drain's host, from the same table.
        let mut actions = VecDeque::new();
        let prompt = parking_lot::Mutex::new(VecDeque::new());
        let drain = QemuHost::new(&table, &mut actions, &prompt);
        let mut again = [0u8; 4];
        drain.read_gpa(0x1000, &mut again).expect("the table reads");
        assert_eq!(again, buf);
        assert_eq!(drain.guest_write_gen(41), worker.guest_write_gen(41));
        drop(drain);

        // Everything the drain alone may do is refused or dropped, and said so.
        assert_eq!(worker.write_gpa(0x1000, &[1]), Err(MemError::ReadOnlyHost));
        let before = crate::runtime::drain::census::store_route_count("par_worker_host_misuse");
        worker.enqueue(HA::irq_gfx());
        worker.schedule_bh();
        assert_eq!(
            crate::runtime::drain::census::store_route_count("par_worker_host_misuse") - before,
            2,
            "both calls were counted and dropped"
        );
        assert!(
            prompt.lock().is_empty() && actions.is_empty(),
            "the worker queued nothing for the guest"
        );
    }
}
