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
    BufferPlan,
    /// The backing did not resolve read-only.
    BufferBacking,
    /// A debt or an in-flight writeback stands over the bytes: the settle is
    /// the drain's.
    BufferOwed,
    /// The guest read failed; the serial path reports it.
    BufferRead,
    /// The index bind's window could not be planned or read.
    Index,
    /// A sampled bind without a retained resource.
    SampledNoResource,
    /// The sampled planner bailed (a lower rung, a cold construction).
    SampledPlan,
    /// The source needs a drain-only rail (cube faces, attachment alias of a
    /// secondary slot, a miss that wants its detail).
    SampledRail,
    /// More than one colour attachment.
    Mrt,
    /// The draw is not a deferrable resident-chain intermediate (a Store, a
    /// readback, a query, or the recorder is off).
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
            Bail::BufferPlan => "draw_par_bail_buf_plan",
            Bail::BufferBacking => "draw_par_bail_buf_backing",
            Bail::BufferOwed => "draw_par_bail_buf_owed",
            Bail::BufferRead => "draw_par_bail_buf_read",
            Bail::Index => "draw_par_bail_index",
            Bail::SampledNoResource => "draw_par_bail_smp_no_resource",
            Bail::SampledPlan => "draw_par_bail_smp_plan",
            Bail::SampledRail => "draw_par_bail_smp_rail",
            Bail::Mrt => "draw_par_bail_mrt",
            Bail::NotDeferrable => "draw_par_bail_not_deferrable",
            Bail::ChainMoved => "draw_par_commit_chain_moved",
            Bail::PlanInvalid => "draw_par_commit_plan_invalid",
        }
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
    if !state.pending_writebacks.is_empty() {
        if crate::runtime::writeback_debt::owes_texture(state, task_id, buffer_ref)
            || state
                .pending_writebacks
                .owes_gva_overlapping(task_id, read_gva, read_span)
        {
            return Err(Bail::BufferOwed);
        }
        let deferred = state.pending_writebacks.deferred_mapping();
        if let Some(deferred) = deferred {
            let named = state.mappings_named_by(task_id, buffer_ref);
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
                    Err(_) => return Err(Bail::BufferPlan),
                }
            }
            // A sub-floor window the serial ladder reads on the CPU, or a
            // never-primed reference whose window is sub-floor — the population
            // PASS_STAGE's `stage_or_bail` reads on a worker for the same
            // reason: the floor already refused the gather.
            Err(buffer_plan::BufferBail::BelowFloor | buffer_plan::BufferBail::ColdPacked) => {}
            // Everything else is a construction (whole, a walked window) or a
            // refusal the serial ladder names itself.
            Err(_) => return Err(Bail::BufferPlan),
        }
    }
    let backing = resolve_buffer_backing(state, host, task_id, buffer_ref, Some(resource.as_ref()))
        .ok_or(Bail::BufferBacking)?;
    if allow_zero_copy && !buffer_plan::below_floor(backing.size, offset) {
        // Above the floor the serial ladder constructs a rail (packed, whole,
        // a walked window) rather than reading.
        return Err(Bail::BufferPlan);
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
        resolve_index_window_reason(state, host, task_id, info).map_err(|_| Bail::Index)?;
    let resource = objects::resolve_resource(state, host, task_id, info.index_buffer_ref)
        .map_err(|_| Bail::Index)?;
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
                Err(_) => return Err(Bail::Index),
            }
        }
        Err(buffer_plan::BufferBail::BelowFloor | buffer_plan::BufferBail::ColdPacked) => {}
        Err(_) => return Err(Bail::Index),
    }
    if !buffer_plan::below_floor(backing.size, info.index_buffer_offset) {
        return Err(Bail::Index);
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
        Err(_) => Err(Bail::SampledPlan),
    }
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
    let memo = super::lab_resolve_memo_snapshot();
    let scoped = pool.scope(state, reqs, |st, env, (req, writeback_guest)| {
        let _memo = super::WorkerMemoGuard::set(memo.clone());
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
