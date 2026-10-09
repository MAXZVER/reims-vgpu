//! TEMP (lab, DRAW_PAR, 2026-10-09): whole-draw preparation on the worker pool.
//!
//! The drain encodes a render pass's draws one after another, and on Weather
//! (b115) that encode — pipeline memo, buffer and index loads, sampled source
//! resolution, the request's assembly — is ~44 % of the drain's busy time at
//! ~3.4 us a draw, with no single line above 1 %. PassPrep (`pass_prep.rs`)
//! planned *binds* ahead on the pool and lost (b067/b068): a fork per ~9 items
//! of ~1 us cost more than the plans saved, and the serial ladder still ran per
//! draw to consume them. This module forks once per chunk of whole draws
//! instead: every worker runs the same [`super::try_metal2vulkan_draw_with`]
//! body over a shared `&DeviceState` and hands back a finished
//! [`crate::backend::vulkan::engine::DrawRequest`]; the drain then commits each
//! prepared draw in order — validates the plans it rests on, lands their facts,
//! submits to the recorder — and runs the serial encode only for a draw the
//! workers could not prepare or whose plans no longer hold.
//!
//! What a worker may do is what [`crate::runtime::par`]'s rules allow: read the
//! device state and guest memory, write nothing. Every state mutation the
//! serial encode makes has a worker-side twin here that either *describes* the
//! mutation for the commit (a buffer plan, a sampled plan) or bails — the seed
//! doors, the zero-copy constructions, MRT, cube faces, the synchronous engine
//! rails. A bail costs the draw nothing but the worker's time: the drain's
//! serial path is unchanged and runs exactly as before.
//!
//! Only records 2+ of a resident render-pass chain are offered (draw 0 records
//! the pass invariants the later records reuse, and its seed doors need the
//! drain); the last record stores and keeps the serial path too.

use super::*;
use crate::backend::vulkan::engine::recorder::AssertSend;
use crate::backend::vulkan::engine::{DeferredMeta, DrawRequest, TargetIdentity};
use std::sync::Arc;

/// How a draw's encode reaches the device state: exclusively on the drain, or
/// shared on a pool worker. One body, two instantiations — a worker's arm at a
/// mutating site either plans the mutation or bails, and the type system is
/// what keeps it from doing anything else.
pub(crate) enum StateAccess<'a> {
    Drain(&'a mut DeviceState),
    Worker(&'a DeviceState),
}

impl StateAccess<'_> {
    #[inline]
    pub(crate) fn shared(&self) -> &DeviceState {
        match self {
            StateAccess::Drain(s) => s,
            StateAccess::Worker(s) => s,
        }
    }

    #[inline]
    pub(crate) fn exclusive(&mut self) -> Option<&mut DeviceState> {
        match self {
            StateAccess::Drain(s) => Some(s),
            StateAccess::Worker(_) => None,
        }
    }

    #[inline]
    pub(crate) fn is_worker(&self) -> bool {
        matches!(self, StateAccess::Worker(_))
    }
}

/// Why a worker gave a draw back to the serial path. Each is a census route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bail {
    /// No pass invariants to reuse (draw 0, or a record whose template moved).
    PassInvariants,
    /// A bind without a retained resource: the serial ladder reads the guest.
    BufferNoResource,
    /// The buffer plan refused with a reason the serial ladder constructs for.
    BufferPlan(buffer_plan::BufferBail),
    /// The plan answered but not for this bind's key.
    BufferPredict,
    /// Above the zero-copy floor the serial ladder constructs a rail.
    BufferAboveFloor,
    /// The backing did not resolve read-only.
    BufferBacking,
    /// A debt or an in-flight writeback stands over the bytes: the settle is
    /// the drain's.
    BufferOwed,
    /// The guest read failed; the serial path reports it.
    BufferRead,
    /// The index bind's window did not resolve.
    IndexWindow,
    /// The index buffer's resource did not resolve.
    IndexResource,
    /// The index plan refused.
    IndexPlan(buffer_plan::BufferBail),
    /// The index plan answered but not for this key.
    IndexPredict,
    /// Above the floor the serial ladder constructs the index window.
    IndexAboveFloor,
    /// A sampled bind without a retained resource.
    SampledNoResource,
    /// The sampled planner bailed (a lower rung, a cold construction).
    SampledPlan,
    /// A linear texture without a decodable texture descriptor or geometry.
    LinearShape,
    /// A linear texture whose allocation has a packed alias: the serial
    /// ladder serves it zero-copy from there.
    LinearPacked,
    /// A linear texture whose window the gather floor admits: the serial
    /// ladder gathers it on the witness.
    LinearFloor,
    /// The linear rows could not be converted for upload.
    LinearConvert,
    /// A cube map: six faces the drain reloads.
    Cube,
    /// An attachment alias of a secondary slot.
    AliasSecondary,
    /// A miss whose detail the drain reports.
    MissDetail,
    /// More than one colour attachment.
    Mrt,
    /// The record does not load from the chain's resident: its seed doors
    /// are the drain's.
    NoChainLoad,
    /// The recorder is off: nothing is deferred.
    RecorderOff,
    /// Not a resident-chain intermediate (no chain identity resolved).
    NotChain,
    /// An occlusion query is armed.
    Query,
    /// The record stores or reads back (the last record of the pass).
    Store,
    /// Deferrable by every test but without a target identity (a wiring gap).
    NotDeferrable,
    /// The chain the draw was prepared against is not the one at commit.
    ChainMoved,
    /// A plan the draw rests on no longer held at commit.
    PlanInvalid,
}

impl Bail {
    pub(crate) fn route(self) -> &'static str {
        match self {
            Bail::PassInvariants => "draw_par_bail_pass_inv",
            Bail::BufferNoResource => "draw_par_bail_buf_no_resource",
            Bail::BufferPlan(b) => buffer_bail_route(b, false),
            Bail::BufferPredict => "draw_par_bail_buf_predict",
            Bail::BufferAboveFloor => "draw_par_bail_buf_above_floor",
            Bail::BufferBacking => "draw_par_bail_buf_backing",
            Bail::BufferOwed => "draw_par_bail_buf_owed",
            Bail::BufferRead => "draw_par_bail_buf_read",
            Bail::IndexWindow => "draw_par_bail_idx_window",
            Bail::IndexResource => "draw_par_bail_idx_resource",
            Bail::IndexPlan(b) => buffer_bail_route(b, true),
            Bail::IndexPredict => "draw_par_bail_idx_predict",
            Bail::IndexAboveFloor => "draw_par_bail_idx_above_floor",
            Bail::SampledNoResource => "draw_par_bail_smp_no_resource",
            Bail::SampledPlan => "draw_par_bail_smp_plan",
            Bail::LinearShape => "draw_par_bail_lin_shape",
            Bail::LinearPacked => "draw_par_bail_lin_packed",
            Bail::LinearFloor => "draw_par_bail_lin_floor",
            Bail::LinearConvert => "draw_par_bail_lin_convert",
            Bail::Cube => "draw_par_bail_cube",
            Bail::AliasSecondary => "draw_par_bail_alias_secondary",
            Bail::MissDetail => "draw_par_bail_miss_detail",
            Bail::Mrt => "draw_par_bail_mrt",
            Bail::NoChainLoad => "draw_par_bail_no_chain_load",
            Bail::RecorderOff => "draw_par_bail_recorder_off",
            Bail::NotChain => "draw_par_bail_not_chain",
            Bail::Query => "draw_par_bail_query",
            Bail::Store => "draw_par_bail_store",
            Bail::NotDeferrable => "draw_par_bail_not_deferrable",
            Bail::ChainMoved => "draw_par_commit_chain_moved",
            Bail::PlanInvalid => "draw_par_commit_plan_invalid",
        }
    }
}

fn buffer_bail_route(bail: buffer_plan::BufferBail, index: bool) -> &'static str {
    use buffer_plan::BufferBail as B;
    match (index, bail) {
        (false, B::NullRef) => "draw_par_bail_buf_null_ref",
        (false, B::ColdBacking) => "draw_par_bail_buf_cold_backing",
        (false, B::ColdPacked) => "draw_par_bail_buf_cold_packed",
        (false, B::ColdWhole) => "draw_par_bail_buf_cold_whole",
        (false, B::ColdWindow) => "draw_par_bail_buf_cold_window",
        (false, B::BelowFloor) => "draw_par_bail_buf_below_floor",
        (false, B::OffsetPastEnd) => "draw_par_bail_buf_offset_past_end",
        (false, B::OwedStage) => "draw_par_bail_buf_owed_stage",
        (true, B::NullRef) => "draw_par_bail_idx_null_ref",
        (true, B::ColdBacking) => "draw_par_bail_idx_cold_backing",
        (true, B::ColdPacked) => "draw_par_bail_idx_cold_packed",
        (true, B::ColdWhole) => "draw_par_bail_idx_cold_whole",
        (true, B::ColdWindow) => "draw_par_bail_idx_cold_window",
        (true, B::BelowFloor) => "draw_par_bail_idx_below_floor",
        (true, B::OffsetPastEnd) => "draw_par_bail_idx_offset_past_end",
        (true, B::OwedStage) => "draw_par_bail_idx_owed_stage",
    }
}

/// A buffer bind a worker served from a plan; the commit validates the plan at
/// this bind's key and lands its facts.
pub(crate) struct BufferCommit {
    plan: buffer_plan::BufferContentPlan,
    buffer_ref: u32,
    resource: Arc<crate::model::TaskResource>,
    offset: u64,
    cap: Option<u64>,
}

/// A sampled bind a worker served from a surface plan; the commit validates
/// the plan's reads and notes its routes.
pub(crate) struct SampleCommit {
    plan: sample_plan::SamplePlan,
    texture_ref: u32,
    resource: Option<Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
}

/// A linear texture's guest rows a worker read and compared against the
/// device's linear memo (`DeviceState::guest_linear_memo`): the memo entry it
/// matched (`native: None`, touched at the commit) or the entry the serial
/// rung would have inserted (`native: Some`, inserted at the commit unless an
/// equal one landed first).
pub(crate) struct LinearMemoFact {
    key: (u32, u64, u32, u32, u32, u16),
    native: Option<Vec<u8>>,
    rgba: Arc<Vec<u8>>,
    layout: TexelLayout,
    generation: u64,
}

impl LinearMemoFact {
    fn land(self, state: &mut DeviceState) {
        match self.native {
            None => {
                let _ = state.guest_linear_memo.get_touch(&self.key);
            }
            Some(native) => {
                if state
                    .guest_linear_memo
                    .peek(&self.key)
                    .is_some_and(|m| m.native == native)
                {
                    let _ = state.guest_linear_memo.get_touch(&self.key);
                    return;
                }
                let entry_bytes = native.len() + self.rgba.len();
                state.guest_linear_memo.insert(
                    self.key,
                    crate::model::GuestLinearMemo {
                        native,
                        rgba: self.rgba,
                        layout: self.layout,
                        generation: self.generation,
                    },
                    entry_bytes,
                );
            }
        }
    }
}

/// What a fragment-stage sampled bind resolved to, for the two censuses the
/// serial path notes where it resolves (deferred to the commit so an
/// uncommitted preparation records nothing).
pub(crate) struct SampledNote {
    texture_ref: u32,
    rw: u32,
    rh: u32,
    mid: u32,
    route: &'static str,
}

/// The worker's running record of one draw's preparation: the plans the
/// request rests on, the notes the commit owes, and — once the body reaches
/// the engine door — the finished request.
#[derive(Default)]
pub(crate) struct PrepareLog {
    pub(super) buffers: Vec<BufferCommit>,
    pub(super) samples: Vec<SampleCommit>,
    pub(super) linear: Vec<LinearMemoFact>,
    pub(super) notes: Vec<SampledNote>,
    pub(super) bail: Option<Bail>,
    pub(super) done: Option<PreparedRequest>,
}

/// The finished half of a prepared draw.
pub(crate) struct PreparedRequest {
    pub(super) resources: AssertSend<DrawRequest>,
    pub(super) meta: DeferredMeta,
    pub(super) chain_identity: TargetIdentity,
}

/// One draw a worker prepared, ready for the drain's commit.
pub(crate) struct PreparedDraw {
    log: PrepareLog,
    task_id: u32,
}

impl PrepareLog {
    pub(super) fn note_sampled(&mut self, texture_ref: u32, rw: u32, rh: u32, mid: u32, route: &'static str) {
        self.notes.push(SampledNote {
            texture_ref,
            rw,
            rh,
            mid,
            route,
        });
    }
}

// ---------------------------------------------------------------------------
// Switches
// ---------------------------------------------------------------------------

/// `REIMS_VGPU_LAB_DRAW_PAR=off` disables; live `DRAW_PAR=on|off`. On by
/// default on the Vulkan rail with a pool.
pub(crate) fn draw_par_on() -> bool {
    if cfg!(test) {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("DRAW_PAR") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("REIMS_VGPU_LAB_DRAW_PAR").as_deref() != Ok("off");
        crate::observe::off(format!("lab_draw_par on={}", u8::from(on)));
        on
    })
}

/// Draws per fork. `REIMS_VGPU_LAB_DRAW_PAR_CHUNK=K` (default 32, 4..=256);
/// live `DRAW_PAR_CHUNK`.
pub(crate) fn chunk() -> usize {
    if let Some(v) = crate::runtime::lab_live::value("DRAW_PAR_CHUNK") {
        if let Ok(k) = v.trim().parse::<usize>() {
            return k.clamp(4, 256);
        }
    }
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_DRAW_PAR_CHUNK")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(32)
            .clamp(4, 256)
    })
}

/// A pass shorter than this (records 2+) is encoded serially: the fork would
/// cost more than the draws.
pub(crate) const MIN_DRAWS: usize = 4;

// ---------------------------------------------------------------------------
// Worker-side loaders
// ---------------------------------------------------------------------------

/// The CPU read `read_buffer_bytes_resolved` makes, without its settle: when a
/// debt or an in-flight writeback could reach the bytes the read is the
/// drain's, and the worker bails.
fn cpu_read_worker<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    backing: &BufferBacking,
    offset: u64,
    extent_cap: Option<u64>,
) -> Result<crate::backend::vulkan::engine::BufferContent, Bail> {
    let (gva, size) = (backing.gva, backing.size);
    if offset >= size {
        return Err(Bail::BufferRead);
    }
    let full = size - offset;
    let avail = extent_cap.map_or(full, |cap| full.min(cap));
    let want = crate::runtime::draw::host_alloc_len(avail)
        .filter(|&n| n > 0)
        .ok_or(Bail::BufferRead)?;
    let (read_gva, read_span) = (gva + offset, want as u64);
    guest_bytes_readable(state, host, task_id, buffer_ref, read_gva, read_span)?;
    let mut buf = vec![0u8; want];
    if gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        read_gva,
        &mut buf,
        state.page_shift,
    )
    .is_err()
    {
        return Err(Bail::BufferRead);
    }
    Ok(crate::backend::vulkan::engine::BufferContent::from(buf))
}

/// The obligations a serial CPU read of guest bytes settles — a debt the
/// reference or the span owes, a device write still in flight over the pages —
/// asked read-only: when any stands, the settle is the drain's and the worker
/// bails.
fn guest_bytes_readable<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    read_gva: u64,
    read_span: u64,
) -> Result<(), Bail> {
    if !state.pending_writebacks.is_empty() {
        if crate::runtime::writeback_debt::owes_texture(state, task_id, texture_ref)
            || state
                .pending_writebacks
                .owes_gva_overlapping(task_id, read_gva, read_span)
        {
            return Err(Bail::BufferOwed);
        }
        let deferred = state.pending_writebacks.deferred_mapping();
        if let Some(deferred) = deferred {
            let named = state.mappings_named_by(task_id, texture_ref);
            if named.iter().any(|m| m == deferred) {
                return Err(Bail::BufferOwed);
            }
        }
    }
    let backend = crate::backend::selected();
    if backend.guest_writes_outstanding() {
        let page_size = state.page_size();
        let want_pages = reims_vgpu_paging::span::pages_spanned(read_gva, read_span, page_size);
        let gpas = crate::runtime::gva_mem::task_gva_page_gpas(
            host,
            &state.tasks,
            task_id,
            read_gva,
            read_span,
            state.page_shift,
        );
        if gpas.len() as u64 != want_pages
            || backend.guest_writes_reaching(&gpas) != crate::backend::GuestWriteReach::Disjoint
        {
            return Err(Bail::BufferOwed);
        }
    }
    Ok(())
}

/// The worker's [`super::load_buffer_content`]: a warm-rail plan where one
/// answers (recorded for the commit), the CPU read where the serial ladder
/// would read, a bail where it would construct.
#[allow(clippy::too_many_arguments)]
pub(super) fn load_buffer_content_worker<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    log: &mut PrepareLog,
    task_id: u32,
    buffer_ref: u32,
    resource: Option<&Arc<crate::model::TaskResource>>,
    offset: u64,
    allow_zero_copy: bool,
    extent_cap: Option<u64>,
) -> Result<crate::backend::vulkan::engine::BufferContent, Bail> {
    let resource = resource.ok_or(Bail::BufferNoResource)?;
    if allow_zero_copy {
        match buffer_plan::plan_buffer_rails(
            state,
            host,
            task_id,
            buffer_ref,
            resource.clone(),
            offset,
            None,
        ) {
            Ok(plan) => {
                match buffer_plan::predict_buffer_plan(
                    state,
                    &plan,
                    task_id,
                    buffer_ref,
                    Some(resource.as_ref()),
                    offset,
                    extent_cap,
                ) {
                    Ok(Some(content)) => {
                        log.buffers.push(BufferCommit {
                            plan,
                            buffer_ref,
                            resource: resource.clone(),
                            offset,
                            cap: extent_cap,
                        });
                        return Ok(content);
                    }
                    // Not this bind's rail: the serial ladder reads on.
                    Ok(None) => {}
                    Err(_) => return Err(Bail::BufferPredict),
                }
            }
            // A sub-floor window the serial ladder reads on the CPU, or a
            // never-primed reference whose window is sub-floor — the population
            // PASS_STAGE's `stage_or_bail` reads on a worker for the same
            // reason: the floor already refused the gather.
            Err(buffer_plan::BufferBail::BelowFloor | buffer_plan::BufferBail::ColdPacked) => {}
            // Everything else is a construction (whole, a walked window) or a
            // refusal the serial ladder names itself.
            Err(bail) => return Err(Bail::BufferPlan(bail)),
        }
    }
    let backing = resolve_buffer_backing(state, host, task_id, buffer_ref, Some(resource.as_ref()))
        .ok_or(Bail::BufferBacking)?;
    if allow_zero_copy && !buffer_plan::below_floor(backing.size, offset) {
        // Above the floor the serial ladder constructs a rail (packed, whole,
        // a walked window) rather than reading.
        return Err(Bail::BufferAboveFloor);
    }
    cpu_read_worker(state, host, task_id, buffer_ref, &backing, offset, extent_cap)
}

/// The worker's [`super::load_index_content_reason`].
pub(super) fn load_index_content_worker<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    log: &mut PrepareLog,
    task_id: u32,
    info: &IndexedDrawInfo,
) -> Result<crate::backend::vulkan::engine::BufferContent, Bail> {
    let (backing, need) =
        resolve_index_window_reason(state, host, task_id, info).map_err(|_| Bail::IndexWindow)?;
    let resource = objects::resolve_resource(state, host, task_id, info.index_buffer_ref)
        .map_err(|_| Bail::IndexResource)?;
    match buffer_plan::plan_index_content(state, host, task_id, info, None) {
        Ok(plan) => {
            match buffer_plan::predict_buffer_plan(
                state,
                &plan,
                task_id,
                info.index_buffer_ref,
                Some(resource.as_ref()),
                info.index_buffer_offset,
                Some(need as u64),
            ) {
                Ok(Some(content)) => {
                    log.buffers.push(BufferCommit {
                        plan,
                        buffer_ref: info.index_buffer_ref,
                        resource,
                        offset: info.index_buffer_offset,
                        cap: Some(need as u64),
                    });
                    return Ok(content);
                }
                Ok(None) => {}
                Err(_) => return Err(Bail::IndexPredict),
            }
        }
        Err(buffer_plan::BufferBail::BelowFloor | buffer_plan::BufferBail::ColdPacked) => {}
        Err(bail) => return Err(Bail::IndexPlan(bail)),
    }
    if !buffer_plan::below_floor(backing.size, info.index_buffer_offset) {
        return Err(Bail::IndexAboveFloor);
    }
    cpu_read_worker(
        state,
        host,
        task_id,
        info.index_buffer_ref,
        &backing,
        info.index_buffer_offset,
        Some(need as u64),
    )
}

/// The worker's [`super::resolve_sampled_source`]: the encoder's resolve memo
/// (a snapshot the drain took at the fork), then a surface plan (a resident or
/// the host cache — the rungs whose commit is a validation and a note). The
/// zero-copy and memo-fill rungs construct or move the gather witness, and
/// those binds bail to the serial ladder.
pub(super) fn resolve_sampled_source_worker<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    log: &mut PrepareLog,
    task_id: u32,
    texture_ref: u32,
    resource: Option<Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
) -> Result<(u32, u32, u32, SampledSourceRequest), Bail> {
    let resource = resource.ok_or(Bail::SampledNoResource)?;
    match sample_plan::plan_sampled_source(
        state,
        host,
        task_id,
        texture_ref,
        resource.clone(),
        may_bind_resident,
    ) {
        Ok(plan) => {
            let served = sample_plan::take_sampled_plan(state, &plan, false);
            log.samples.push(SampleCommit {
                plan,
                texture_ref,
                resource: Some(resource),
                may_bind_resident,
            });
            Ok(served)
        }
        // A linear texture: the serial ladder's `is_linear_tex` branch.
        Err(sample_plan::SampleBail::NotSurface) => {
            resolve_linear_worker(state, host, log, task_id, texture_ref, &resource)
        }
        Err(_) => Err(Bail::SampledPlan),
    }
}

/// The worker's twin of the serial ladder's linear branch — its memo rung
/// (`load_linear_guest_memoized`) read-only: the guest rows are read and
/// compared against the device's linear memo, the match or the new entry
/// becomes a fact the commit lands. The rungs above it on the serial ladder
/// construct (a packed alias, a gather window on the witness) or serve a GVA
/// resident; where one of those would answer, the worker leaves the bind to
/// the drain so the two paths serve the same rail.
fn resolve_linear_worker<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    log: &mut PrepareLog,
    task_id: u32,
    texture_ref: u32,
    resource: &Arc<crate::model::TaskResource>,
) -> Result<(u32, u32, u32, SampledSourceRequest), Bail> {
    let tex = match resource.decoded() {
        Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => tex,
        _ => return Err(Bail::LinearShape),
    };
    let declared_format = tex.declared_pixel_format().ok_or(Bail::LinearShape)?;
    let sample_fmt = crate::runtime::draw::texture_view::effective_view_sample_format(declared_format, None)
        .ok_or(Bail::LinearShape)?;
    let (gva, layout) = tex.level_gva(0, state.page_shift).ok_or(Bail::LinearShape)?;
    let (w, h) = (layout.width, layout.height);
    if w == 0 || h == 0 {
        return Err(Bail::LinearShape);
    }
    let bpr = layout.row_stride;
    let planes = layout.planes();
    let tight = pixel_format::tight_row_bytes(w, declared_format).ok_or(Bail::LinearShape)? as u64;
    if bpr < tight {
        return Err(Bail::LinearShape);
    }
    let storage_rows = pixel_format::tight_row_count(h, declared_format).ok_or(Bail::LinearShape)?;
    let span = bpr
        .checked_mul(u64::from(storage_rows))
        .and_then(|s| s.checked_mul(u64::from(planes)))
        .ok_or(Bail::LinearShape)?;
    let native_len = crate::runtime::draw::host_alloc_len(span).ok_or(Bail::LinearShape)?;
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return Err(Bail::LinearShape);
    }
    // The zero-copy rail above this rung: a packed alias of the allocation
    // serves directly, and a window the gather floor admits is gathered on
    // the witness. Both are the drain's.
    if matches!(
        state.bound_buffers.packed(task_id, texture_ref),
        Some(crate::runtime::bound_buffers::PackedBufferResolution::Available(_))
    ) {
        return Err(Bail::LinearPacked);
    }
    if let Ok((native, _, _)) = translate::pixel::sampled_pixels(declared_format) {
        if sampled_gather_floor_pure(native, span) {
            return Err(Bail::LinearFloor);
        }
    }
    guest_bytes_readable(state, host, task_id, texture_ref, gva, span)?;
    let mut scratch = vec![0u8; native_len];
    if gva_mem::read_task_gva_by_id(host, &state.tasks, task_id, gva, &mut scratch, state.page_shift)
        .is_err()
    {
        return Err(Bail::BufferRead);
    }
    let key = (task_id, gva, w, h, planes, sample_fmt);
    if let Some(m) = state.guest_linear_memo.peek(&key) {
        if m.native == scratch {
            crate::runtime::drain::note_store_route("draw_par_lin_memo_hit");
            log.linear.push(LinearMemoFact {
                key,
                native: None,
                rgba: m.rgba.clone(),
                layout: m.layout,
                generation: m.generation,
            });
            return Ok((
                w,
                h,
                0,
                SampledSourceRequest::Bytes(
                    m.rgba.clone(),
                    Some(LinearSampleIdentity {
                        key: gva,
                        generation: m.generation,
                    }),
                    SampledByteFormat::from_source(m.layout, sample_fmt),
                    crate::backend::vulkan::engine::SampledByteOrigin::LinearTexture,
                ),
            ));
        }
    }
    let (rgba, fmt) = native_scratch_to_upload(&scratch, w, h, planes, bpr, sample_fmt, tight)
        .ok_or(Bail::LinearConvert)?;
    let generation = state.next_sampled_content_generation();
    let rgba = Arc::new(rgba);
    crate::runtime::drain::note_store_route("draw_par_lin_memo_fill");
    log.linear.push(LinearMemoFact {
        key,
        native: Some(scratch),
        rgba: rgba.clone(),
        layout: fmt,
        generation,
    });
    Ok((
        w,
        h,
        0,
        SampledSourceRequest::Bytes(
            rgba,
            Some(LinearSampleIdentity {
                key: gva,
                generation,
            }),
            SampledByteFormat::from_source(fmt, sample_fmt),
            crate::backend::vulkan::engine::SampledByteOrigin::LinearTexture,
        ),
    ))
}

// ---------------------------------------------------------------------------
// The drain's commit
// ---------------------------------------------------------------------------

thread_local! {
    /// The prepared draw the drain offered for the record it is about to
    /// encode, taken by `try_metal2vulkan_draw` at its top.
    static OFFERED: std::cell::RefCell<Option<PreparedDraw>> = const { std::cell::RefCell::new(None) };
}

/// Offer the prepared draw for the next encode on this thread.
pub(crate) fn offer(prepared: PreparedDraw) {
    OFFERED.with(|o| *o.borrow_mut() = Some(prepared));
}

/// Drop whatever was offered and not taken (a chain that broke before its
/// record encoded).
pub(crate) fn clear_offer() {
    OFFERED.with(|o| *o.borrow_mut() = None);
}

fn take_offer() -> Option<PreparedDraw> {
    OFFERED.with(|o| o.borrow_mut().take())
}

/// What the commit returned to the encode.
pub(crate) enum Committed {
    /// The prepared draw was submitted; the span is the serial path's.
    Span(Result<M2vDrawSpan, DrawError>),
    /// Nothing was offered or the offer did not hold: run the serial encode.
    Serial,
}

/// Commit the draw offered for `req`, if one was.
pub(crate) fn commit_offered<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
) -> Committed {
    let Some(prepared) = take_offer() else {
        return Committed::Serial;
    };
    match commit_prepared(state, host, req, prepared) {
        Ok(span) => {
            crate::runtime::drain::note_store_route("draw_par_committed");
            Committed::Span(span)
        }
        Err(bail) => {
            crate::runtime::drain::note_store_route(bail.route());
            Committed::Serial
        }
    }
}

fn commit_prepared<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
    prepared: PreparedDraw,
) -> Result<Result<M2vDrawSpan, DrawError>, Bail> {
    let PreparedDraw { log, task_id } = prepared;
    let done = log.done.ok_or(Bail::NotDeferrable)?;
    if task_id != req.task_id
        || !req.chain_from_resident
        || pass_inv::chain_identity(state, req).as_ref() != Some(&done.chain_identity)
    {
        return Err(Bail::ChainMoved);
    }
    // The plans, in bind order, each validated at the key it served and its
    // facts landed — the same facts the serial ladder would have made.
    for b in &log.buffers {
        match buffer_plan::commit_prepared_buffer(
            state,
            &b.plan,
            task_id,
            b.buffer_ref,
            Some(b.resource.as_ref()),
            b.offset,
            b.cap,
        ) {
            Ok(Some(_)) => {}
            _ => return Err(Bail::PlanInvalid),
        }
    }
    for s in &log.samples {
        if sample_plan::validate_sampled_plan(
            state,
            host,
            &s.plan,
            task_id,
            s.texture_ref,
            s.resource.as_ref(),
            s.may_bind_resident,
        )
        .is_err()
        {
            return Err(Bail::PlanInvalid);
        }
        let _ = sample_plan::take_sampled_plan(state, &s.plan, true);
    }
    for fact in log.linear {
        fact.land(state);
    }
    for n in &log.notes {
        note_sampled_ref_backing(state, task_id, n.texture_ref, n.rw, n.rh, n.mid, n.route);
        crate::runtime::scanout::note_sampled_surface_field(
            state,
            &*host,
            n.mid,
            n.texture_ref,
            n.route,
        );
    }
    // From here the draw is committed: the serial path's engine door, verbatim.
    if let Some(failure) = crate::backend::vulkan::engine::take_deferred_failure() {
        crate::backend::vulkan::engine::report_deferred_failure(&failure);
        req.deferred_failure_of_previous = true;
        return Ok(Err(failure.error));
    }
    let mut resources = done.resources.into_inner();
    pass_stage::arm_seed_stage(&mut resources);
    crate::backend::vulkan::engine::submit_deferred_draw(state, resources, done.meta);
    req.visibility_samples = None;
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Engine);
    Ok(Ok(M2vDrawSpan::ResidentChain))
}

// ---------------------------------------------------------------------------
// The fork
// ---------------------------------------------------------------------------

/// Prepare a chunk of records on the pool. `reqs` are the records' requests
/// exactly as the drain's loop builds them (template retargeted, binds filled,
/// chain fields set), each with the `writeback_guest` the loop will encode it
/// with. Returns one slot per request: the prepared draw, or `None` where the
/// worker bailed (the route says why).
pub(crate) fn prepare_chunk(
    state: &DeviceState,
    reqs: &[(DrawEncodeRequest, bool)],
) -> Vec<Option<PreparedDraw>> {
    let Some(pool) = state.par() else {
        return reqs.iter().map(|_| None).collect();
    };
    let scoped = pool.scope(state, reqs, |st, env, (req, writeback_guest)| {
        let mut host = env.host();
        let mut req = req.clone();
        let mut log = PrepareLog::default();
        let outcome = super::try_metal2vulkan_draw_with(
            &mut StateAccess::Worker(st),
            &mut host,
            &mut req,
            *writeback_guest,
            Some(&mut log),
        );
        match (outcome, log.bail, log.done.is_some()) {
            (Ok(_), None, true) => Some(PreparedDraw {
                task_id: req.task_id,
                log,
            }),
            (_, Some(bail), _) => {
                crate::runtime::drain::note_store_route(bail.route());
                None
            }
            _ => {
                crate::runtime::drain::note_store_route("draw_par_bail_encode_err");
                None
            }
        }
    });
    crate::runtime::drain::note_store_route("draw_par_chunks");
    crate::runtime::drain::note_store_route_n("draw_par_offered", reqs.len() as u64);
    let mut prepared = 0u64;
    let out: Vec<Option<PreparedDraw>> = scoped
        .results
        .into_iter()
        .map(|r| match r {
            Ok(Some(p)) => {
                prepared += 1;
                Some(p)
            }
            _ => None,
        })
        .collect();
    crate::runtime::drain::note_store_route_n("draw_par_prepared", prepared);
    out
}
