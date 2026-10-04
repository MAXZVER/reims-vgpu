//! TEMP (lab, `REIMS_VGPU_LAB_PASS_STAGE`): the engine half of worker staging
//! (design §8.4 / C1).
//!
//! The runtime half (`crate::runtime::par::staging`) owns the ring
//! bookkeeping; this module owns what the bookkeeping names — one
//! host-visible, persistently mapped `VkBuffer` per pool thread, created once
//! under the engine lock and handed to the planners by thread — and the facts
//! that carry a carved window to the engine:
//!
//! * [`crate::backend::vulkan::engine::types::BufferContent::Staged`] — a
//!   vertex/index/storage bind whose bytes a worker already memcpy'd; the
//!   engine binds `(buffer, offset)` and records nothing else.
//! * [`crate::backend::vulkan::engine::types::SampledSource::Staged`] — a
//!   guest-runs sampled bind with the window pre-gathered; the engine records
//!   the buffer→image copy from the arena instead of the CPU gather.
//! * [`SeedStageCell`] — a CPU load seed a later chunk's scope stages while
//!   the draw sits in the recorder FIFO (the PAY_DEFER shape: the request
//!   carries a cell, a worker fills it, the recorder takes whichever won).
//!
//! # Lifetime
//!
//! A window's bytes are read by the command buffer of whichever submission
//! consumes the fact. That submission's seal moves a *reference* into the
//! slot's [`super::pools::PendingGpuCleanup`] (`ResourcePools::note_staged_read`),
//! and the retire that drains the cleanup releases it — the staging slots'
//! own lifetime, stated in references rather than in slot objects because one
//! window can outlive the entry that first read it (a batch joiner records
//! from a window an earlier plan carved).
//!
//! # The census
//!
//! `pass_stage_copies` / `pass_stage_bytes` are what the workers moved,
//! `pass_stage_arena_full_n` / `_bytes` what the ring refused (the serial
//! fallback ran instead), `pass_stage_bind_hit` / `_sampled_hit` / `_seed_hit`
//! what the engine consumed, and the ring census itself is noted per tranche
//! by [`note_census`] from the batch tail.

use crate::runtime::par::staging::{StagingRefusal, StagingRing, StagingWindow};
use ash::vk;
use std::sync::Arc;

use super::context::DeviceContext;

/// One pool thread's arena: the ring and the Vulkan objects its windows name.
pub(crate) struct WorkerArena {
    pub ring: Arc<StagingRing>,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// The mapping, as a raw address the same way `BufferSlot` carries it (a
    /// `usize`, so engine state stays `Send`).
    pub mapped: usize,
    /// The alignment every carve takes (see [`ensure_installed`]): wide
    /// enough for the storage-buffer descriptors a staged bind may feed.
    pub bind_align: u64,
}

impl WorkerArena {
    /// Carve `len` bytes at the arena's bind alignment, taking the fact's
    /// reference, and say where they landed. `None` is any ring refusal — the
    /// caller falls back to the serial staging path and the census names why.
    pub fn alloc(&self, len: u64, align: u64) -> Result<StagedSlice, StagingRefusal> {
        let window = self.ring.alloc(len, align)?;
        Ok(StagedSlice {
            ring: Arc::clone(&self.ring),
            buffer: self.buffer,
            window,
            mapped: self.mapped,
        })
    }
}

/// A carved window a fact carries: the release handle, the buffer the engine
/// binds or copies from, and where the bytes sit in it.
///
/// **Every slice owns one ring reference**, the one [`WorkerArena::alloc`]
/// took: [`Clone`] takes another ([`StagingRing::retain`]) and [`Drop`]
/// releases one, so a fact that dies anywhere — an invalid plan popped with
/// its frame, a request the engine declined, a planner that unwound — gives
/// its window back by construction and no ledger has to follow it around. A
/// submission that records a read takes its own reference through
/// [`Self::retain_for_submission`], released by the retiring slot.
pub(crate) struct StagedSlice {
    pub ring: Arc<StagingRing>,
    pub buffer: vk::Buffer,
    pub window: StagingWindow,
    /// The mapping; a `usize` so the fact stays `Send` (see `BufferSlot`).
    pub mapped: usize,
}

impl Drop for StagedSlice {
    fn drop(&mut self) {
        self.ring.release(self.window);
    }
}

impl Clone for StagedSlice {
    fn clone(&self) -> Self {
        // A ring rebuilt behind a device loss cannot retain; the clone then
        // names a dead window whose drop is ignored — the same honesty the
        // engine's read registration has there.
        let _ = self.ring.retain(self.window);
        Self {
            ring: Arc::clone(&self.ring),
            buffer: self.buffer,
            window: self.window,
            mapped: self.mapped,
        }
    }
}

impl std::fmt::Debug for StagedSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedSlice")
            .field("buffer", &self.buffer)
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

impl StagedSlice {
    /// The staged bytes, for the shadow compare and the tests. The mapping is
    /// host-coherent write-through (`MemoryClass::Upload`), so this reads what
    /// the worker wrote and the GPU will read.
    ///
    /// # Safety
    ///
    /// The slice must still be alive (not released past zero references).
    pub unsafe fn bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self.mapped as *const u8).add(self.window.offset as usize),
                self.window.len as usize,
            )
        }
    }

    /// One submission is about to record a read of this window: take its
    /// reference. The submission's retire releases it.
    pub fn retain_for_submission(&self) -> bool {
        self.ring.retain(self.window)
    }
}

/// The arenas, once installed. `None` until the first draw on a PASS_STAGE
/// boot creates them; reset by teardown and device loss.
static ARENAS: std::sync::RwLock<Option<Arc<[WorkerArena]>>> = std::sync::RwLock::new(None);

/// The arenas a planning scope borrows, by pool member. `None` until the
/// engine created them (the first chunk after the first draw of a boot) — the
/// planner's `ColdRings` bail, served by the serial path.
pub(crate) fn arenas() -> Option<Arc<[WorkerArena]>> {
    ARENAS.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The arena of pool member `worker` (helpers `0..workers`, the coordinator
/// last) out of a borrowed set. See
/// [`crate::runtime::par::staging::population`].
pub(crate) fn arena_of<'a>(arenas: &'a Arc<[WorkerArena]>, worker: u16) -> Option<&'a WorkerArena> {
    arenas.get(usize::from(worker))
}

/// Create the arenas, once, under the engine lock (Vulkan objects are
/// externally synchronized). Idempotent; a failure leaves `None` installed,
/// names itself once, and the rail stays on the serial staging path.
///
/// # Safety
///
/// `ctx` must be the live device, and the caller must hold the engine (every
/// other Vulkan call in the process goes through it).
pub(crate) unsafe fn ensure_installed(ctx: &DeviceContext) {
    if arenas().is_some() || !crate::runtime::par::staging::enabled() {
        return;
    }
    let population = crate::runtime::par::staging::population();
    let chunk = crate::runtime::par::staging::chunk_bytes();
    let chunks = crate::runtime::par::staging::chunks_per_arena();
    // The bind alignment: storage descriptors are the widest consumer
    // (`minStorageBufferOffsetAlignment`), and index offsets (the index type's
    // size) and vertex offsets (no limit) are narrower.
    let align = ctx.storage_buffer_offset_align.max(16).next_power_of_two();
    let mut built: Vec<WorkerArena> = Vec::with_capacity(population);
    for member in 0..population {
        let total = chunk.saturating_mul(chunks as u64);
        match unsafe { create_arena(ctx, total, chunk, chunks) } {
            Ok((buffer, memory, mapped)) => built.push(WorkerArena {
                ring: Arc::new(StagingRing::new(chunk, chunks)),
                buffer,
                memory,
                mapped: mapped as usize,
                bind_align: align,
            }),
            Err(e) => {
                crate::observe::fail(format!(
                    "pass_stage_arena_create_failed member={member} total={total} err={e:?}"
                ));
                // Half a population is worse than none: a member without an
                // arena plans serial while its peers stage, and the boot
                // cannot tell which it got. Destroy what was built and leave
                // the rail off for the process.
                unsafe { destroy_arenas(ctx, built) };
                return;
            }
        }
    }
    crate::runtime::drain::note_store_route("pass_stage_arenas_installed");
    *ARENAS.write().unwrap_or_else(|e| e.into_inner()) = Some(built.into());
}

/// One arena's objects: a buffer of `chunks * chunk` bytes over one
/// host-coherent mapping, bound and mapped for the device's life.
unsafe fn create_arena(
    ctx: &DeviceContext,
    total: u64,
    _chunk: u64,
    _chunks: usize,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), vk::Result> {
    let buffer = unsafe {
        ctx.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(total)
                .usage(super::pools::POOL_SLOT_USAGE)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?
    };
    let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
    let mt = ctx
        .memory_type_for(
            req.memory_type_bits,
            req.size,
            crate::backend::vulkan::caps::MemoryClass::Upload,
        )
        .ok_or(vk::Result::ERROR_UNKNOWN)?;
    let memory = unsafe {
        ctx.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size.max(total))
                .memory_type_index(mt),
            None,
        )?
    };
    if let Err(e) = unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe { ctx.device.free_memory(memory, None) };
        unsafe { ctx.device.destroy_buffer(buffer, None) };
        return Err(e);
    }
    let mapped = match unsafe {
        ctx.device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
    } {
        Ok(ptr) => ptr as *mut u8,
        Err(e) => {
            unsafe { ctx.device.destroy_buffer(buffer, None) };
            unsafe { ctx.device.free_memory(memory, None) };
            return Err(e);
        }
    };
    Ok((buffer, memory, mapped))
}

/// Destroy a set of arenas. The teardown's fence waits have already run; a
/// lost device destroys nothing (the handles died with it).
unsafe fn destroy_arenas(ctx: &DeviceContext, arenas: Vec<WorkerArena>) {
    for arena in arenas {
        unsafe { ctx.device.destroy_buffer(arena.buffer, None) };
        unsafe { ctx.device.free_memory(arena.memory, None) };
    }
}

/// The engine teardown: take the arenas out and destroy their objects. Called
/// from the pools' own `destroy_all`, after its fence waits.
///
/// # Safety
///
/// The device is still alive, and every submission that read an arena window
/// has completed.
pub(crate) unsafe fn destroy_all(device: &ash::Device) {
    if let Some(set) = ARENAS.write().unwrap_or_else(|e| e.into_inner()).take() {
        for arena in set.iter() {
            unsafe { device.destroy_buffer(arena.buffer, None) };
            unsafe { device.free_memory(arena.memory, None) };
        }
    }
}

/// The device was lost: the handles are dead, the timeline will never
/// advance, and an arena that waited for it would strand its memory for the
/// process. Drop them without touching the device.
pub(crate) fn reset_lost() {
    *ARENAS.write().unwrap_or_else(|e| e.into_inner()) = None;
}

/// A CPU load seed a later chunk's scope stages while the draw it belongs to
/// sits in the recorder FIFO. The PAY_DEFER shape, applied to bytes: the
/// request carries a token, a pool job fills it, the consumer takes whichever
/// side won.
///
/// The R/B exchange folds into the worker's memcpy (`swap_rb`), so a staged
/// seed is already in the attachment's physical order and the engine records
/// only the buffer→image copy.
pub struct SeedStageCell {
    state: std::sync::Mutex<SeedStageState>,
}

enum SeedStageState {
    /// Nobody has taken it. A worker may.
    Pending { bytes: Arc<Vec<u8>>, swap_rb: bool },
    /// A worker is copying. A consumer that arrives now takes the serial path
    /// rather than waiting on a pool thread.
    Staging,
    /// A worker finished. `StagedSlice` is already in the attachment's order.
    Staged(StagedSlice),
    /// The engine consumed the seed (either arm) or the draw declined; a late
    /// worker's window is released by its own completion.
    Spent,
}

impl Drop for SeedStageCell {
    fn drop(&mut self) {
        // Nothing to do: a `Staged` state's slice owns its reference and
        // releases it as the state drops. A consumed seed left `Spent`, and
        // the submission that read the window holds its own reference.
    }
}

impl std::fmt::Debug for SeedStageCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeedStageCell").finish_non_exhaustive()
    }
}

impl SeedStageCell {
    /// A cell over `bytes` (the same `Arc` the request carries, so the serial
    /// fallback reads them without a second copy).
    pub fn new(bytes: Arc<Vec<u8>>, swap_rb: bool) -> Self {
        Self {
            state: std::sync::Mutex::new(SeedStageState::Pending { bytes, swap_rb }),
        }
    }

    /// A worker's claim: take the bytes out (`Pending` → `Staging`), for a
    /// memcpy that ends in [`Self::stage_done`] (or [`Self::give_back`] when
    /// the arena refused).
    pub fn take_pending(&self) -> Option<(Arc<Vec<u8>>, bool)> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match &mut *state {
            SeedStageState::Pending { bytes, swap_rb } => {
                let taken = (Arc::clone(bytes), *swap_rb);
                *state = SeedStageState::Staging;
                Some(taken)
            }
            _ => None,
        }
    }

    /// The worker's memcpy finished into `slice`.
    pub fn stage_done(&self, slice: StagedSlice) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match *state {
            SeedStageState::Staging => *state = SeedStageState::Staged(slice),
            // A consumer arrived mid-copy and went serial; the slice drops
            // here and its own reference goes back with it.
            SeedStageState::Spent => {}
            _ => unreachable!("a staging worker owns the cell"),
        }
    }

    /// The arena refused; the serial path stays the answer.
    pub fn give_back(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*state, SeedStageState::Staging) {
            *state = SeedStageState::Spent;
        }
    }

    /// The engine's consumption, at the seed staging phase: a staged window
    /// if a worker won, else the serial path (and the cell is spent either
    /// way, so a late worker releases its window).
    pub fn take_staged(&self) -> Option<StagedSlice> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match std::mem::replace(&mut *state, SeedStageState::Spent) {
            SeedStageState::Staged(slice) => Some(slice),
            _ => None,
        }
    }

    /// Whether a cell is worth offering to a scope: only a `Pending` one.
    pub fn is_pending(&self) -> bool {
        matches!(
            &*self.state.lock().unwrap_or_else(|e| e.into_inner()),
            SeedStageState::Pending { .. }
        )
    }
}

/// TEMP (lab, PASS_STAGE, tests): arenas over leaked host allocations, so
/// the planning arms are driven on a machine with no GPU. `REIMS_VGPU_LAB_`
/// sizing knobs are ignored here: the caller states the chunk, because the
/// tests assert on exact exhaustion points.
#[cfg(test)]
pub(crate) fn install_test_arenas(count: usize, chunk: u64, chunks: usize) {
    let mut built: Vec<WorkerArena> = Vec::with_capacity(count);
    for _ in 0..count {
        let total = chunk.saturating_mul(chunks as u64) as usize;
        let mapped: Box<[u8]> = vec![0u8; total].into_boxed_slice();
        let leaked = Box::leak(mapped) as *mut [u8] as *mut u8;
        built.push(WorkerArena {
            ring: Arc::new(StagingRing::new(chunk, chunks)),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: leaked as usize,
            bind_align: 4,
        });
    }
    *ARENAS.write().unwrap_or_else(|e| e.into_inner()) = Some(built.into());
}

/// TEMP (lab, PASS_STAGE, tests): drop the test arenas and check nothing was
/// stranded: a non-zero flight is a fact that outlived its frame.
#[cfg(test)]
pub(crate) fn take_test_arenas() -> Option<Arc<[WorkerArena]>> {
    ARENAS.write().unwrap_or_else(|e| e.into_inner()).take()
}

/// Note the arenas' census deltas, as the `pass_stage_*` routes. Called from
/// the drain's batch tail, once per tranche, on whichever thread flushed.
pub(crate) fn note_census() {
    let Some(set) = arenas() else {
        return;
    };
    static LAST: std::sync::Mutex<Vec<crate::runtime::par::staging::StagingCensus>> =
        std::sync::Mutex::new(Vec::new());
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last.len() != set.len() {
        last.clear();
        last.resize(
            set.len(),
            crate::runtime::par::staging::StagingCensus::default(),
        );
    }
    for (index, arena) in set.iter().enumerate() {
        let now = arena.ring.census();
        let before = last[index];
        use crate::runtime::drain::note_store_route_n;
        if now.allocated > before.allocated {
            note_store_route_n("pass_stage_copies", now.allocated - before.allocated);
        }
        if now.allocated_bytes > before.allocated_bytes {
            note_store_route_n(
                "pass_stage_bytes",
                now.allocated_bytes - before.allocated_bytes,
            );
        }
        if now.refusals > before.refusals {
            note_store_route_n("pass_stage_arena_full_n", now.refusals - before.refusals);
        }
        if now.refused_bytes > before.refused_bytes {
            note_store_route_n(
                "pass_stage_arena_full_bytes",
                now.refused_bytes - before.refused_bytes,
            );
        }
        if now.recycled > before.recycled {
            note_store_route_n("pass_stage_arena_recycled", now.recycled - before.recycled);
        }
        last[index] = now;
    }
}
