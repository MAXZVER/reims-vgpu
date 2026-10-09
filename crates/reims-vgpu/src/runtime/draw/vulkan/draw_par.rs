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
//! doors, the zero-copy constructions, cube faces, the synchronous engine
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
    /// A linear texture whose window the gather floor admits and whose
    /// zero-copy plan refused: the serial ladder gathers it on the witness.
    LinearFloor,
    /// A zero-copy plan that did not validate on the worker.
    ZcPredict,
    /// A zero-copy plan whose committed outcome differed from the predicted
    /// one the request was assembled with.
    ZcOutcome,
    /// The linear rows could not be converted for upload.
    LinearConvert,
    /// A cube map: six faces the drain reloads.
    Cube,
    /// A secondary colour attachment (MRT) whose resource-ledger entry is cold:
    /// the serial identity call would declare it, which only the drain may do.
    /// Also a record that does not load every colour (a DontCare attachment
    /// consults the engine's residency).
    MrtCold,
    /// A miss whose detail the drain reports.
    MissDetail,
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
            Bail::ZcPredict => "draw_par_bail_zc_predict",
            Bail::ZcOutcome => "draw_par_commit_zc_outcome",
            Bail::LinearConvert => "draw_par_bail_lin_convert",
            Bail::Cube => "draw_par_bail_cube",
            Bail::MrtCold => "draw_par_bail_mrt_cold",
            Bail::MissDetail => "draw_par_bail_miss_detail",
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

fn linear_zc_bail_route(bail: sample_plan::SampleBail) -> &'static str {
    use sample_plan::SampleBail as B;
    match bail {
        B::NullRef => "draw_par_lin_zc_null_ref",
        B::BufferTexture => "draw_par_lin_zc_buffer_texture",
        B::RefTextureView => "draw_par_lin_zc_ref_view",
        B::NotSurface => "draw_par_lin_zc_not_surface",
        B::ColdMapperRef => "draw_par_lin_zc_cold_mapper_ref",
        B::ColdBacking => "draw_par_lin_zc_cold_backing",
        B::NoGeometry => "draw_par_lin_zc_no_geometry",
        B::ColdLease => "draw_par_lin_zc_cold_lease",
        B::GuestReplaced => "draw_par_lin_zc_guest_replaced",
        B::NoHostCache => "draw_par_lin_zc_no_host_cache",
        B::LowerRung => "draw_par_lin_zc_lower_rung",
        B::ZeroCopyDeclined => "draw_par_lin_zc_declined",
        B::OwedFrame => "draw_par_lin_zc_owed_frame",
        B::ColdWindow => "draw_par_lin_zc_cold_window",
        B::ColdWitness => "draw_par_lin_zc_cold_witness",
        B::UpperRung => "draw_par_lin_zc_upper_rung",
        B::ColdPacked => "draw_par_lin_zc_cold_packed",
        B::MemoBudget => "draw_par_lin_zc_memo_budget",
    }
}

/// The witness-backed halves of a gathered request — what a commit may change
/// against the prediction a worker assembled with.
type GatherMarks = (
    Option<LinearSampleIdentity>,
    crate::runtime::gather_witness::GatherVouch,
    Option<crate::runtime::gather_witness::SkipGuardEvidence>,
);

fn gather_marks(request: &SampledSourceRequest) -> Option<GatherMarks> {
    match request {
        SampledSourceRequest::GuestRuns(_, _, _, _, identity, vouch, evidence, _) => {
            Some((*identity, *vouch, *evidence))
        }
        SampledSourceRequest::Staged(staged) => Some((staged.identity, staged.vouch, staged.evidence)),
        _ => None,
    }
}

/// A zero-copy gather a worker served from a plan's prediction; the commit
/// validates the plan, lands the witness and checks the outcome.
pub(crate) struct ZcCommit {
    plan: ZcPlan,
    texture_ref: u32,
    resource: Arc<crate::model::TaskResource>,
    may_bind_resident: bool,
    predicted: Option<GatherMarks>,
}

enum ZcPlan {
    Linear(sample_plan::LinearPlan),
    RefView(sample_plan::RefViewPlan),
    MapperRef(sample_plan::MapperRefPlan),
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

/// What one secondary colour attachment's identity rests on, as the worker
/// peeked it. The serial encode resolves that identity through the drain's
/// `&mut` ledger; a peek that agrees with the ledger at commit is byte-for-byte
/// what the serial call would have answered without mutating anything, so
/// nothing needs landing — only re-deriving.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SecondaryWitness {
    /// A ledgered GVA plane: the generation the ledger held for it at `span`.
    Gva {
        texture_ref: u32,
        gva: u64,
        span: u64,
        generation: u64,
    },
    /// A mapper-ref surface and the identity `surface_identity` named for it.
    Surface {
        mapping_id: u32,
        width: u32,
        height: u32,
        identity: TargetIdentity,
    },
}

/// The worker's running record of one draw's preparation: the plans the
/// request rests on, the notes the commit owes, and — once the body reaches
/// the engine door — the finished request.
#[derive(Default)]
pub(crate) struct PrepareLog {
    pub(super) buffers: Vec<BufferCommit>,
    pub(super) samples: Vec<SampleCommit>,
    pub(super) gathers: Vec<ZcCommit>,
    pub(super) linear: Vec<LinearMemoFact>,
    pub(super) notes: Vec<SampledNote>,
    pub(super) secondaries: Vec<SecondaryWitness>,
    pub(super) bail: Option<Bail>,
    pub(super) done: Option<PreparedRequest>,
}

/// The finished half of a prepared draw.
pub(crate) struct PreparedRequest {
    pub(super) resources: AssertSend<DrawRequest>,
    pub(super) meta: DeferredMeta,
    pub(super) chain_identity: TargetIdentity,
    pub(super) kind: PreparedKind,
}

/// TEMP (lab, DRAW_PAR v13): which deferred arm of the serial ladder a
/// prepared draw takes, and so the span its commit returns.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PreparedKind {
    /// A resident render-pass chain intermediate.
    Chain,
    /// The stream's last record, its Store left on the GVA resident
    /// (`REIMS_VGPU_LAB_RECORDER_GVA_STORE`).
    GvaStore(TargetIdentity),
    /// The stream's last record, its Store into a copied (not guest-backed)
    /// surface resident (`DEFER_SURFACE_STORE`).
    SurfaceStore(TargetIdentity),
}

impl PreparedKind {
    pub(super) fn span(&self) -> M2vDrawSpan {
        match self {
            PreparedKind::Chain => M2vDrawSpan::ResidentChain,
            PreparedKind::GvaStore(identity) => M2vDrawSpan::ResidentGvaStore {
                identity: identity.clone(),
            },
            PreparedKind::SurfaceStore(identity) => M2vDrawSpan::ResidentSurfaceStore {
                identity: identity.clone(),
                guest_store: GuestStoreStatus {
                    guest_backed: false,
                    recorded: false,
                    footprint: None,
                },
            },
        }
    }
}

/// `REIMS_VGPU_LAB_DRAW_PAR_LAST=on` enables; live `DRAW_PAR_LAST`. On: a
/// stream's last record (its deferred Store) is prepared on the pool too and
/// committed through the serial encode's door, which then routes the Store.
/// Off by default since b135: Weather -1 % drain, but the reveal-desktop
/// transitions ran 137/136 window fps against 154 (mid-transition gaps ~150 ms,
/// not explained yet).
pub(crate) fn last_on() -> bool {
    if cfg!(test) {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("DRAW_PAR_LAST") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_DRAW_PAR_LAST").as_deref() == Ok("on"))
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
    thread_local! {
        static CACHED: std::cell::Cell<(u64, bool)> = const { std::cell::Cell::new((0, false)) };
    }
    let now = crate::observe::elapsed_us();
    let (checked, on) = CACHED.with(|c| c.get());
    if checked != 0 && now.saturating_sub(checked) < 100_000 {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = crate::runtime::lab_live::on("DRAW_PAR").unwrap_or_else(|| {
        *ON.get_or_init(|| {
            let on = std::env::var("REIMS_VGPU_LAB_DRAW_PAR").as_deref() != Ok("off");
            crate::observe::off(format!("lab_draw_par on={}", u8::from(on)));
            on
        })
    });
    CACHED.with(|c| c.set((now.max(1), on)));
    on
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

/// TEMP (lab, DRAW_PAR v10, live `DRAW_PAR_COLD_BAIL=on` restores the bail):
/// a sub-floor buffer or index window no record has walked yet (`ColdWindow`:
/// in practice the < 16 KiB remainder of a reference, first sight of its
/// `(ref, offset, cap)`) is read on the worker - the bytes the serial walk's
/// runs would gather, through the same `cpu_read_worker` readability gate the
/// below-floor read uses - instead of bailing the whole record to the drain.
/// The serial walk would also have left a registry hold; on Weather these keys
/// hardly repeat (~3k first sightings a second, ring-allocated offsets), so
/// nothing is warmed: a repeat reads again, on a worker.
fn cold_bytes() -> bool {
    crate::runtime::lab_live::on("DRAW_PAR_COLD_BAIL") != Some(true)
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
    // TEMP (lab, DRAW_PAR v15): into this worker's arena, when it has one.
    if let Some(slice) = JOB_STAGER.with(|s| s.borrow().as_ref().and_then(|st| st.stage_bytes(&buf))) {
        crate::runtime::drain::note_store_route("draw_par_staged_bind");
        return Ok(crate::backend::vulkan::engine::BufferContent::Staged(slice));
    }
    Ok(crate::backend::vulkan::engine::BufferContent::from(buf))
}

thread_local! {
    /// TEMP (lab, DRAW_PAR v15): the arena the running prepare job stages
    /// CPU-read bind bytes into (`prepare_job` sets and clears it).
    static JOB_STAGER: std::cell::RefCell<Option<super::pass_stage::Stager>> =
        const { std::cell::RefCell::new(None) };
}

fn stage_on() -> bool {
    crate::runtime::par::staging::draw_par_stage_wanted()
        && crate::runtime::lab_live::on("DRAW_PAR_STAGE") != Some(false)
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
            // A sub-floor window nobody has walked yet (see `cold_bytes`).
            Err(buffer_plan::BufferBail::ColdWindow) if cold_bytes() => {
                crate::runtime::drain::note_store_route("draw_par_cold_bytes_buf");
            }
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
        Err(buffer_plan::BufferBail::ColdWindow) if cold_bytes() => {
            crate::runtime::drain::note_store_route("draw_par_cold_bytes_idx");
        }
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
        // A ref-texture view's plane on the zero-copy rail.
        Err(sample_plan::SampleBail::RefTextureView) => {
            let plan = sample_plan::plan_ref_view_zero_copy(
                state,
                host,
                task_id,
                texture_ref,
                resource.clone(),
            )
            .map_err(|_| Bail::SampledPlan)?;
            let served = sample_plan::predict_ref_view_plan(
                state,
                host,
                &plan,
                task_id,
                texture_ref,
                Some(&resource),
            )
            .map_err(|_| Bail::ZcPredict)?;
            log.gathers.push(ZcCommit {
                plan: ZcPlan::RefView(plan),
                texture_ref,
                resource,
                may_bind_resident,
                predicted: gather_marks(&served.3),
            });
            Ok(served)
        }
        // A linear texture: the serial ladder's `is_linear_tex` branch — the
        // zero-copy rail where its plan stands, else the memo rung.
        Err(sample_plan::SampleBail::NotSurface) => {
            match sample_plan::plan_linear_zero_copy(
                state,
                host,
                task_id,
                texture_ref,
                resource.clone(),
                may_bind_resident,
            ) {
                Ok(plan) => {
                    let served = sample_plan::predict_linear_plan(
                        state,
                        host,
                        &plan,
                        task_id,
                        texture_ref,
                        Some(&resource),
                        may_bind_resident,
                    )
                    .map_err(|_| Bail::ZcPredict)?;
                    log.gathers.push(ZcCommit {
                        plan: ZcPlan::Linear(plan),
                        texture_ref,
                        resource,
                        may_bind_resident,
                        predicted: gather_marks(&served.3),
                    });
                    Ok(served)
                }
                Err(bail) => {
                    // Which refusal sent the bind to the memo rung, by the
                    // planner's own name (b121: a quarter of Weather's records
                    // still bail at the gather floor below).
                    crate::runtime::drain::note_store_route(linear_zc_bail_route(bail));
                    resolve_linear_worker(state, host, log, task_id, texture_ref, &resource)
                }
            }
        }
        // The surface ladder fell past every host-side copy: the zero-copy
        // window of the surface's own pages.
        Err(sample_plan::SampleBail::NoHostCache | sample_plan::SampleBail::LowerRung) => {
            let plan = sample_plan::plan_mapper_ref_zero_copy(
                state,
                host,
                task_id,
                texture_ref,
                resource.clone(),
            )
            .map_err(|_| Bail::SampledPlan)?;
            let served = sample_plan::predict_mapper_ref_plan(
                state,
                host,
                &plan,
                task_id,
                texture_ref,
                Some(&resource),
            )
            .map_err(|_| Bail::ZcPredict)?;
            log.gathers.push(ZcCommit {
                plan: ZcPlan::MapperRef(plan),
                texture_ref,
                resource,
                may_bind_resident,
                predicted: gather_marks(&served.3),
            });
            Ok(served)
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
    // The serial zero-copy rail rests on the allocation's packed alias: where
    // one stands the rail serves (the drain's); where the construction was
    // refused for this geometry (`Unavailable`) the rail declines and the
    // ladder reads the memo rung whatever the span - b121: a quarter of
    // Weather's and 71 % of Safari's records are those textures. Never primed
    // means the drain would construct, so the worker leaves a floor-clearing
    // window to it.
    match state.bound_buffers.packed(task_id, texture_ref) {
        Some(crate::runtime::bound_buffers::PackedBufferResolution::Available(_)) => {
            return Err(Bail::LinearPacked);
        }
        Some(crate::runtime::bound_buffers::PackedBufferResolution::Unavailable { gva: g, size: sz })
            if tex.allocation_size != 0
                && tex.allocation_base_gva(state.page_shift) == Some(*g)
                && *sz == tex.allocation_size => {}
        _ => {
            if let Ok((native, _, _)) = translate::pixel::sampled_pixels(declared_format) {
                if sampled_gather_floor_pure(native, span) {
                    return Err(Bail::LinearFloor);
                }
            }
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
    if task_id != req.task_id || !req.chain_from_resident || !kind_holds(state, req, &done) {
        return Err(Bail::ChainMoved);
    }
    // The secondary attachments' identities were read, not declared, on the
    // worker; they hold if the ledger still answers them the same way.
    if !log
        .secondaries
        .iter()
        .all(|w| super::secondary_witness_holds(state, task_id, w))
    {
        return Err(Bail::PlanInvalid);
    }
    // The plans, in bind order, each validated at the key it served and its
    // facts landed — the same facts the serial ladder would have made.
    let registry_generation = state.bound_buffers.generation();
    for b in &log.buffers {
        // Planned under this registry generation: the worker validated it
        // against the state the commit sees (see `land_prepared_buffer_unchecked`).
        if b.plan.generation() == registry_generation {
            buffer_plan::land_prepared_buffer_unchecked(
                state,
                &b.plan,
                task_id,
                b.buffer_ref,
                b.offset,
                b.cap,
            );
            continue;
        }
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
    for g in &log.gathers {
        let served = match &g.plan {
            ZcPlan::Linear(plan) => sample_plan::commit_linear_plan(
                state,
                host,
                plan,
                task_id,
                g.texture_ref,
                Some(&g.resource),
                g.may_bind_resident,
                true,
            )
            .map(|(served, _)| served),
            ZcPlan::RefView(plan) => sample_plan::commit_ref_view_plan(
                state,
                host,
                plan,
                task_id,
                g.texture_ref,
                Some(&g.resource),
                true,
            )
            .map(|(served, _)| served),
            ZcPlan::MapperRef(plan) => sample_plan::commit_mapper_ref_plan(
                state,
                host,
                plan,
                task_id,
                g.texture_ref,
                Some(&g.resource),
                true,
            )
            .map(|(served, _)| served),
        };
        let Ok(served) = served else {
            return Err(Bail::PlanInvalid);
        };
        // The request carries the prediction; a witness that answered
        // otherwise makes it the wrong request, not a lesser one.
        if gather_marks(&served.3) != g.predicted {
            return Err(Bail::ZcOutcome);
        }
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
    match &done.kind {
        PreparedKind::Chain => {}
        PreparedKind::GvaStore(_) => {
            req.lab_left_store_job = true;
            crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Store);
            crate::runtime::drain::note_store_route("lab_recorder_gva_store_deferred");
            crate::runtime::drain::note_store_route("draw_par_last_gva");
        }
        PreparedKind::SurfaceStore(_) => {
            req.lab_left_store_job = true;
            crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Store);
            crate::runtime::drain::note_store_route("lab_surface_store_deferred");
            crate::runtime::drain::note_store_route("draw_par_last_surface");
        }
    }
    Ok(Ok(done.kind.span()))
}

/// Whether the target a prepared draw rendered into is still the one the
/// serial ladder would name for `req` now, for its kind.
fn kind_holds(state: &DeviceState, req: &DrawEncodeRequest, done: &PreparedRequest) -> bool {
    match &done.kind {
        PreparedKind::Chain => {
            pass_inv::chain_identity(state, req).as_ref() == Some(&done.chain_identity)
        }
        PreparedKind::GvaStore(identity) => {
            done.chain_identity == *identity
                && super::gva_chain_identity(req).as_ref() == Some(identity)
                && super::lab_recorder_gva_store()
        }
        PreparedKind::SurfaceStore(identity) => {
            done.chain_identity == *identity
                && pass_inv::store_identity(state, req, true).as_ref() == Some(identity)
                && crate::backend::vulkan::engine::surface_store_guest_backed_lock_free(
                    identity,
                    done.resources.0.guest_target_memory.is_some(),
                ) == Some(false)
        }
    }
}

// ---------------------------------------------------------------------------
// The overlapped commit (v9)
// ---------------------------------------------------------------------------

/// `REIMS_VGPU_LAB_DRAW_PAR_OVERLAP=off` disables; live `DRAW_PAR_OVERLAP`.
/// On: a stream's records 2+ are prepared on the pool while the drain commits
/// them in order, each as soon as it is ready (one overlap region per run of
/// committable records). Off: the chunked fork-join of v1-v8.
pub(crate) fn overlap_on() -> bool {
    if cfg!(test) {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("DRAW_PAR_OVERLAP") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_DRAW_PAR_OVERLAP").as_deref() != Ok("off"))
}

/// Prime the resolve memo with the sampled sources of a stream's records 2+,
/// once, on the drain, before its first region (see `lab_resolve_memo_prime`).
/// Once per stream, not per region: a region that stops re-primed its whole
/// remainder in v9 (quadratic, b128), and a region priming only its head left
/// the helpers resolving the rest themselves, each job its own big linear
/// textures (b129 v10: the drain parked on them 19 % of its time).
/// Live `DRAW_PAR_PRIME=off` skips it.
pub(crate) fn prime_stream<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    reqs: &[(DrawEncodeRequest, bool)],
) {
    if reqs.is_empty() || crate::runtime::lab_live::on("DRAW_PAR_PRIME") == Some(false) {
        return;
    }
    let primed = super::lab_resolve_memo_prime(state, host, reqs);
    crate::runtime::drain::note_store_route_n("draw_par_memo_primed", u64::from(primed));
}

/// What a shared-state commit owes the device state, landed by the drain
/// with its exclusive access once the region's helpers are quiescent: the
/// registry holds of walked buffer plans and the linear memo's facts. Both are
/// caches with an "equal entry landed first" rule, so landing them late only
/// costs a later record of the region the re-plan it would have skipped.
#[derive(Default)]
pub(crate) struct RegionOwed {
    holds: Vec<buffer_plan::OwedHold>,
    linear: Vec<LinearMemoFact>,
}

impl RegionOwed {
    pub(crate) fn land(&mut self, state: &mut DeviceState) {
        for hold in self.holds.drain(..) {
            hold.land(state);
        }
        for fact in self.linear.drain(..) {
            fact.land(state);
        }
    }
}

/// The outcome of one shared-state commit.
pub(crate) enum SharedCommit {
    /// Submitted to the recorder.
    Committed,
    /// Needs the exclusive commit (a zero-copy gather lands witness state, or
    /// the recorder holds a failure to report): offer it to the serial encode.
    Exclusive(PreparedDraw),
    /// The preparation no longer holds: the serial encode runs.
    Invalid(Bail),
}

/// [`commit_prepared`] under the drain's shared state, while the pool's
/// helpers read the same state: the same checks in the same order; the
/// registry holds and linear-memo facts go to `owed` instead of landing.
pub(crate) fn commit_shared<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    req: &DrawEncodeRequest,
    prepared: PreparedDraw,
    owed: &mut RegionOwed,
) -> SharedCommit {
    if !prepared.log.gathers.is_empty()
        || crate::backend::vulkan::engine::deferred_failure_stashed()
    {
        return SharedCommit::Exclusive(prepared);
    }
    if prepared
        .log
        .done
        .as_ref()
        .is_some_and(|done| done.kind != PreparedKind::Chain)
    {
        return SharedCommit::Exclusive(prepared);
    }
    let PreparedDraw { log, task_id } = prepared;
    let Some(done) = log.done else {
        return SharedCommit::Invalid(Bail::NotDeferrable);
    };
    if task_id != req.task_id
        || !req.chain_from_resident
        || pass_inv::chain_identity(state, req).as_ref() != Some(&done.chain_identity)
    {
        return SharedCommit::Invalid(Bail::ChainMoved);
    }
    if !log
        .secondaries
        .iter()
        .all(|w| super::secondary_witness_holds(state, task_id, w))
    {
        return SharedCommit::Invalid(Bail::PlanInvalid);
    }
    let first_hold = owed.holds.len();
    for b in &log.buffers {
        match buffer_plan::commit_prepared_buffer_shared(
            state,
            &b.plan,
            task_id,
            b.buffer_ref,
            Some(b.resource.as_ref()),
            b.offset,
            b.cap,
        ) {
            Ok(Some(hold)) => owed.holds.push(hold),
            Ok(None) => {}
            Err(()) => {
                owed.holds.truncate(first_hold);
                return SharedCommit::Invalid(Bail::PlanInvalid);
            }
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
            owed.holds.truncate(first_hold);
            return SharedCommit::Invalid(Bail::PlanInvalid);
        }
        let _ = sample_plan::take_sampled_plan(state, &s.plan, true);
    }
    owed.linear.extend(log.linear);
    for n in &log.notes {
        note_sampled_ref_backing(state, task_id, n.texture_ref, n.rw, n.rh, n.mid, n.route);
        crate::runtime::scanout::note_sampled_surface_field(state, host, n.mid, n.texture_ref, n.route);
    }
    let mut resources = done.resources.into_inner();
    pass_stage::arm_seed_stage(&mut resources);
    crate::backend::vulkan::engine::submit_deferred_draw(state, resources, done.meta);
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Engine);
    SharedCommit::Committed
}

/// What the region's per-record step answered.
pub(crate) enum RegionStep {
    /// Committed; the region goes on.
    Committed,
    /// Ends the region: the record goes to the serial encode, offered this
    /// preparation when there is one.
    Stop(Option<PreparedDraw>),
}

/// One record after a region's stop that the helpers had already attempted.
pub(crate) enum Carried {
    Prepared(PreparedDraw),
    /// The worker bailed (its route is noted): the serial encode runs.
    Bailed,
}

/// Where a region stopped.
pub(crate) struct RegionStop {
    /// Index (into the region's requests) of the first record the region did
    /// not commit; the requests' length when it committed them all.
    pub(crate) at: usize,
    /// That record's preparation, when it needs the exclusive commit.
    pub(crate) exclusive: Option<PreparedDraw>,
    /// The records after `at` the helpers had already attempted, in order.
    pub(crate) after: Vec<Carried>,
}

fn prepare_job(
    st: &DeviceState,
    env: &crate::runtime::par::WorkerEnv<'_>,
    item: &(DrawEncodeRequest, bool),
) -> Option<PreparedDraw> {
    let (req, writeback_guest) = item;
    let mut host = env.host();
    let mut req = req.clone();
    let mut log = PrepareLog::default();
    let stager = if stage_on() {
        crate::backend::vulkan::engine::pass_stage::arenas()
            .and_then(|arenas| super::pass_stage::Stager::of(arenas, usize::from(env.worker.0)))
    } else {
        None
    };
    JOB_STAGER.with(|s| *s.borrow_mut() = stager);
    let outcome = super::try_metal2vulkan_draw_with(
        &mut StateAccess::Worker(st),
        &mut host,
        &mut req,
        *writeback_guest,
        Some(&mut log),
    );
    JOB_STAGER.with(|s| *s.borrow_mut() = None);
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
}

/// Run one overlap region over `reqs` (records 2+ of a resident chain, in
/// order): the records in `carried` were attempted already (a previous
/// region's helpers) and the rest are prepared on the pool now, while the
/// drain commits each record in order through `step` as soon as it is ready.
/// The region ends at the first record `step` stops at, or that has no
/// preparation; the owed holds land before this returns.
pub(crate) fn run_region<M, S>(
    state: &mut DeviceState,
    host: &mut M,
    reqs: &[(DrawEncodeRequest, bool)],
    carried: Vec<Carried>,
    mut step: S,
) -> RegionStop
where
    M: HostMemory + HostOps,
    S: FnMut(&DeviceState, &mut M, usize, &DrawEncodeRequest, &bool, PreparedDraw, &mut RegionOwed) -> RegionStep,
{
    let nc = carried.len().min(reqs.len());
    let jobs = &reqs[nc..];
    // The fresh records' sampled sources, once, on the drain (see
    // `lab_resolve_memo_prime`); the carried ones are already prepared.
    // The sampled sources are primed once per stream by the caller
    // (`prime_stream`), not per region.
    if !jobs.is_empty() {
        crate::runtime::drain::note_store_route_n("draw_par_offered", jobs.len() as u64);
    }
    crate::runtime::drain::note_store_route("draw_par_regions");
    let mut owed = RegionOwed::default();
    let mut carried: Vec<Option<Carried>> = carried.into_iter().take(nc).map(Some).collect();
    let stop = {
        let shared: &DeviceState = &*state;
        let Some(pool) = shared.par() else {
            return RegionStop {
                at: 0,
                exclusive: None,
                after: Vec::new(),
            };
        };
        pool.overlap(shared, jobs, prepare_job, |ov| {
            let mut stop_at = |k: usize,
                               exclusive: Option<PreparedDraw>,
                               ov: &mut crate::runtime::par::Overlap<'_, Option<PreparedDraw>>,
                               carried: &mut Vec<Option<Carried>>|
             -> RegionStop {
                let mut after: Vec<Carried> = carried
                    .iter_mut()
                    .skip(k + 1)
                    .filter_map(Option::take)
                    .collect();
                let from_job = (k + 1).saturating_sub(nc);
                for r in ov.stop(from_job) {
                    match r {
                        Some(Ok(Some(p))) => after.push(Carried::Prepared(p)),
                        Some(_) => after.push(Carried::Bailed),
                        // Cancelled unclaimed: this and everything after it
                        // is prepared afresh by the next region.
                        None => break,
                    }
                }
                RegionStop {
                    at: k,
                    exclusive,
                    after,
                }
            };
            let mut prepared_n = 0u64;
            for k in 0..reqs.len() {
                let prepared = if k < nc {
                    match carried[k].take() {
                        Some(Carried::Prepared(p)) => Some(p),
                        _ => None,
                    }
                } else {
                    match ov.wait(k - nc) {
                        Some(Ok(Some(p))) => {
                            prepared_n += 1;
                            Some(p)
                        }
                        _ => None,
                    }
                };
                let Some(prepared) = prepared else {
                    crate::runtime::drain::note_store_route_n("draw_par_prepared", prepared_n);
                    return stop_at(k, None, ov, &mut carried);
                };
                match step(shared, host, k, &reqs[k].0, &reqs[k].1, prepared, &mut owed) {
                    RegionStep::Committed => {}
                    RegionStep::Stop(exclusive) => {
                        crate::runtime::drain::note_store_route_n("draw_par_prepared", prepared_n);
                        return stop_at(k, exclusive, ov, &mut carried);
                    }
                }
            }
            crate::runtime::drain::note_store_route_n("draw_par_prepared", prepared_n);
            RegionStop {
                at: reqs.len(),
                exclusive: None,
                after: Vec::new(),
            }
        })
    };
    owed.land(state);
    stop
}

// ---------------------------------------------------------------------------
// The fork
// ---------------------------------------------------------------------------

/// Prepare a chunk of records on the pool. `reqs` are the records' requests
/// exactly as the drain's loop builds them (template retargeted, binds filled,
/// chain fields set), each with the `writeback_guest` the loop will encode it
/// with. Returns one slot per request: the prepared draw, or `None` where the
/// worker bailed (the route says why).
pub(crate) fn prepare_chunk<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    reqs: &[(DrawEncodeRequest, bool)],
) -> Vec<Option<PreparedDraw>> {
    if state.par().is_none() {
        return reqs.iter().map(|_| None).collect();
    }
    // The chunk's sampled sources, once, on the drain (see `lab_resolve_memo_prime`).
    let primed = super::lab_resolve_memo_prime(state, host, reqs);
    crate::runtime::drain::note_store_route_n("draw_par_memo_primed", u64::from(primed));
    let state: &DeviceState = &*state;
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
