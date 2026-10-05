//! TEMP (lab, `REIMS_VGPU_LAB_PAR_PREFETCH=off|shadow|on`): plan a render
//! stream's sampled sources on the worker pool before its draws encode.
//!
//! `finish_stream` calls [`prefetch_stream`] once per stream, before the draw
//! loop. It plans every distinct `(task, texture ref)` the stream's draws bind
//! through [`plan_sampled_source`] on [`crate::runtime::par::Pool`], and pushes
//! the plans on a drain-thread stack for the length of the stream. `push_tex`
//! asks [`prefetched_source`] before it resolves: a plan whose reads still hold
//! serves the bind (`on`), or is compared against the serial answer (`shadow`).
//! A missing or invalid plan leaves the serial ladder to run, unchanged.
//!
//! A surface plan is not consumed by its first use. Taking it notes census
//! routes and changes no device state, so its reads still describe the state
//! after it, and every later draw of the stream that binds the same ref
//! re-validates the same plan. The witness-backing plans (a ref-texture view's
//! zero-copy plane, the linear and mapper-ref gather arms) move the witness
//! when they apply, so each commit hands back the repeat for the next bind of
//! the same key — [`crate::runtime::gather_witness::PlannedGather::repeat`],
//! committed from the post state (design H6's removal): one plan per key
//! serves the whole pass, and only a plan that failed validation lands the ref
//! in the frame's `consumed` list for [`super::pass_prep`]'s `ahead` to replan.

use super::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// Which way the stream prefetch runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefetchMode {
    Off,
    /// Plan and validate, then run the serial ladder anyway and compare.
    Shadow,
    /// Serve binds from validated plans.
    On,
}

pub(crate) fn prefetch_mode() -> PrefetchMode {
    static MODE: std::sync::OnceLock<PrefetchMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("REIMS_VGPU_LAB_PAR_PREFETCH").as_deref() {
            Ok("on") => PrefetchMode::On,
            Ok("shadow") => PrefetchMode::Shadow,
            _ => PrefetchMode::Off,
        },
    )
}

/// One stream's (or pass's) plans, plus the refs whose ref-view plan a bind
/// has consumed. `consumed` is drained by [`super::pass_prep::PassPrep::ahead`],
/// which replans those refs from the witness's post-consumption state.
///
/// TEMP (lab, PASS_PREP step 3): the same frame carries the pass's buffer
/// plans — rail plans under `(task, reference)`, exact plans under the bind's
/// own key, and the pass's one Store page walk — served by
/// [`super::buffer_plan`] at the points the serial buffer ladder resolves.
pub(crate) struct PassPlans {
    map: HashMap<(u32, u32), StreamPlan>,
    consumed: Vec<u32>,
    /// TEMP (lab, PASS_PREP step 3): warm-rail plans (packed alias,
    /// whole-allocation runs) — one per `(task, reference)`, serving every
    /// offset and cap of that reference.
    pub(super) buffer_rails: HashMap<(u32, u32), super::buffer_plan::BufferContentPlan>,
    /// TEMP (lab, PASS_PREP step 3): exact plans (held entry, walked window)
    /// under `(task, reference, offset, cap)`.
    pub(super) buffer_exacts:
        HashMap<super::buffer_plan::BufferExactKey, super::buffer_plan::BufferContentPlan>,
    /// TEMP (lab, PASS_PREP step 3): the pass's Store page walk, when planned.
    pub(super) store_pages: Option<super::buffer_plan::StorePagesPlan>,
}

thread_local! {
    /// The plans of each stream being encoded on this thread, innermost last.
    static STACK: RefCell<Vec<PassPlans>> = const { RefCell::new(Vec::new()) };
}

/// One planned bind of a stream.
///
/// TEMP (lab, PASS_STAGE, STAGED_PROVEN): `Clone` is the restage job's — a
/// proven repeat is cloned onto the worker that stages its window, and the
/// staged twin back into the frame's map (every field is an `Arc` or a
/// scalar, so the clone moves no bytes).
#[derive(Clone)]
pub(crate) enum StreamPlan {
    /// A top rung of the surface ladder: a resident or the host cache.
    Surface(SamplePlan),
    /// A ref-texture view's plane on the zero-copy rail.
    RefView(sample_plan::RefViewPlan),
    /// A linear texture's plane on the zero-copy rail (step 2).
    Linear(sample_plan::LinearPlan),
    /// A mapper-ref surface's window on the zero-copy rail (step 2).
    MapperRef(sample_plan::MapperRefPlan),
    /// A mapper-ref surface's guest-memo bind (step 2).
    Memo(sample_plan::MemoPlan),
}

impl StreamPlan {
    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): a gather plan a commit proved
    /// (it survived a bind) whose window no worker has staged yet — the
    /// restage job's selection. See [`sample_plan::attach_staged`] for why
    /// these, and only these, are worth an arena window.
    pub(super) fn wants_restage(&self) -> bool {
        match self {
            StreamPlan::RefView(p) => p.staging_pending(),
            StreamPlan::Linear(p) => p.staging_pending(),
            StreamPlan::MapperRef(p) => p.staging_pending(),
            _ => false,
        }
    }

    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): whether a worker's window is
    /// attached — the restage job's success test.
    pub(super) fn has_staged_window(&self) -> bool {
        match self {
            StreamPlan::RefView(p) => p.has_staged_window(),
            StreamPlan::Linear(p) => p.has_staged_window(),
            StreamPlan::MapperRef(p) => p.has_staged_window(),
            _ => false,
        }
    }
}

/// Pops this stream's plans when the stream ends.
pub(crate) struct PrefetchGuard {
    pushed: bool,
}

impl Drop for PrefetchGuard {
    fn drop(&mut self) {
        if self.pushed {
            STACK.with(|stack| {
                stack.borrow_mut().pop();
            });
        }
    }
}

/// Push an empty plans frame for this stream and return the guard that pops it.
pub(super) fn push_empty_frame() -> PrefetchGuard {
    STACK.with(|stack| {
        stack.borrow_mut().push(PassPlans {
            map: HashMap::new(),
            consumed: Vec::new(),
            buffer_rails: HashMap::new(),
            buffer_exacts: HashMap::new(),
            store_pages: None,
        })
    });
    PrefetchGuard { pushed: true }
}

/// TEMP (lab, PASS_PREP step 3): the innermost frame, for the buffer-plan ask
/// and merge paths. `None` when no pass frame is pushed on this thread.
pub(super) fn with_last_frame<R>(f: impl FnOnce(&mut PassPlans) -> R) -> Option<R> {
    STACK.with(|stack| stack.borrow_mut().last_mut().map(f))
}

/// Merge planned binds into the current (innermost) frame. Plans replace a
/// same-key entry, which is how a replanned ref-view ref supersedes the
/// consumed plan it leaves behind.
pub(super) fn merge_into_frame(plans: impl IntoIterator<Item = ((u32, u32), StreamPlan)>) {
    STACK.with(|stack| {
        if let Some(frame) = stack.borrow_mut().last_mut() {
            frame.map.extend(plans);
        }
    });
}

/// Take the refs whose ref-view plans binds consumed since the last call, so
/// the caller can plan them again for the witness's current state.
pub(super) fn take_consumed_refs() -> Vec<u32> {
    STACK.with(|stack| {
        stack
            .borrow_mut()
            .last_mut()
            .map_or_else(Vec::new, |frame| std::mem::take(&mut frame.consumed))
    })
}

/// TEMP (lab, PASS_STAGE, STAGED_PROVEN): the innermost frame's proven,
/// still-unstaged gather plans — the repeats whose commits proved them on
/// this thread and whose windows the eager attach no longer copies. Cloned
/// out for the next chunk's restage jobs; the chunk's merge replaces the map
/// entries the jobs answer for, which is sound because a scope's fork-join
/// freezes the frame — no bind runs between this take and that merge.
pub(super) fn restageable_plans() -> Vec<((u32, u32), StreamPlan)> {
    STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(frame) = stack.last_mut() else {
            return Vec::new();
        };
        frame
            .map
            .iter()
            .filter(|(_, plan)| plan.wants_restage())
            .map(|(key, plan)| (*key, plan.clone()))
            .collect()
    })
}

/// Whether the memo rung may be planned after the zero-copy rail refused a
/// bind: only where the *serial* ladder refuses that rail too, which is the
/// floor (`ZeroCopyDeclined` — the same `SAMPLED_GATHER_MIN_BYTES` crossover
/// both arms apply).
///
/// Every other refusal is one the serial ladder resolves on the drain and the
/// plan cannot: a cold window cache or a cold witness (`ColdWindow`,
/// `ColdWitness`) the serial rail fills and re-arms before serving the gather,
/// an upper rung, a debt. Letting the memo plan answer those binds would serve
/// `Bytes` where the serial ladder serves `GuestRuns` off the very pages the
/// memo read — the shadow gate's `par_shadow_mismatch_variant` class the fs2s1
/// boot caught (lab/resolve 394d5c9a: ~3 080 binds, surfaces 283x512/512x512,
/// serial on the rungs above).
pub(super) fn memo_plan_may_follow(zero_copy_bail: SampleBail) -> bool {
    matches!(zero_copy_bail, SampleBail::ZeroCopyDeclined)
}

/// The one job body both [`prefetch_stream`] and [`super::pass_prep`] run per
/// item, on the pool: the surface ladder's plannable top, and — where it bails
/// — the rungs under it in the serial ladder's own order: a materializing
/// ref-texture view's zero-copy plane (`RefTextureView`), a linear texture's
/// zero-copy rail (`NotSurface`), and the surface ladder's page rungs
/// (`NoHostCache`/`LowerRung`): the mapper-ref zero-copy window, then the
/// guest memo, whose fills draw on `memo_budget`.
pub(super) fn plan_item(
    st: &DeviceState,
    env: &crate::runtime::par::WorkerEnv<'_>,
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    memo_budget: &std::sync::atomic::AtomicU64,
) -> Result<StreamPlan, SampleBail> {
    let host = env.host();
    match plan_sampled_source(st, &host, task_id, texture_ref, resource.clone(), true) {
        Ok(plan) => Ok(StreamPlan::Surface(plan)),
        Err(SampleBail::RefTextureView) => {
            sample_plan::plan_ref_view_zero_copy(st, &host, task_id, texture_ref, resource)
                .map(StreamPlan::RefView)
        }
        Err(SampleBail::NotSurface) => {
            sample_plan::plan_linear_zero_copy(st, &host, task_id, texture_ref, resource, true)
                .map(StreamPlan::Linear)
        }
        // The surface ladder fell past every host-side copy: the serial ladder
        // reads the surface's own pages from here down. The zero-copy window
        // first; the memo only where the serial ladder would take it too.
        Err(bail @ (SampleBail::NoHostCache | SampleBail::LowerRung)) => {
            match sample_plan::plan_mapper_ref_zero_copy(
                st,
                &host,
                task_id,
                texture_ref,
                resource.clone(),
            ) {
                Ok(plan) => Ok(StreamPlan::MapperRef(plan)),
                Err(zero_copy) => {
                    if !memo_plan_may_follow(zero_copy) {
                        // The refusal names the rung the serial ladder reached
                        // (or resolves itself); the memo must not answer past
                        // it.
                        return Err(zero_copy);
                    }
                    match sample_plan::plan_mapper_ref_memo(
                        st,
                        &host,
                        task_id,
                        texture_ref,
                        resource,
                        memo_budget,
                    ) {
                        Ok(plan) => Ok(StreamPlan::Memo(plan)),
                        // The memo's own budget refusal names a fill the
                        // serial memo would have made; every other memo
                        // refusal is reported as the floor's, which is the
                        // refusal both arms share.
                        Err(memo_bail @ SampleBail::MemoBudget) => Err(memo_bail),
                        Err(_) => Err(zero_copy),
                    }
                }
            }
        }
        Err(bail) => Err(bail),
    }
}

pub(super) fn bail_route(bail: SampleBail) -> &'static str {
    match bail {
        SampleBail::NullRef => "par_prefetch_bail_null_ref",
        SampleBail::BufferTexture => "par_prefetch_bail_buffer_texture",
        SampleBail::RefTextureView => "par_prefetch_bail_ref_texture_view",
        SampleBail::NotSurface => "par_prefetch_bail_not_surface",
        SampleBail::ColdMapperRef => "par_prefetch_bail_cold_mapper_ref",
        SampleBail::ColdBacking => "par_prefetch_bail_cold_backing",
        SampleBail::NoGeometry => "par_prefetch_bail_no_geometry",
        SampleBail::ColdLease => "par_prefetch_bail_cold_lease",
        SampleBail::GuestReplaced => "par_prefetch_bail_guest_replaced",
        SampleBail::NoHostCache => "par_prefetch_bail_no_host_cache",
        SampleBail::LowerRung => "par_prefetch_bail_lower_rung",
        SampleBail::ZeroCopyDeclined => "par_prefetch_bail_zero_copy_declined",
        SampleBail::OwedFrame => "par_prefetch_bail_owed_frame",
        SampleBail::ColdWindow => "par_prefetch_bail_cold_window",
        SampleBail::ColdWitness => "par_prefetch_bail_cold_witness",
        SampleBail::UpperRung => "par_prefetch_bail_upper_rung",
        SampleBail::ColdPacked => "par_prefetch_bail_cold_packed",
        SampleBail::MemoBudget => "par_prefetch_bail_memo_budget",
    }
}

pub(super) fn invalid_route(invalid: SampleInvalid) -> &'static str {
    match invalid {
        SampleInvalid::Key => "par_prefetch_invalid_key",
        SampleInvalid::Resource => "par_prefetch_invalid_resource",
        SampleInvalid::Mapping => "par_prefetch_invalid_mapping",
        SampleInvalid::Backing => "par_prefetch_invalid_backing",
        SampleInvalid::GuestWrite => "par_prefetch_invalid_guest_write",
        SampleInvalid::HostCache => "par_prefetch_invalid_host_cache",
        SampleInvalid::Window => "par_prefetch_invalid_window",
        SampleInvalid::Witness => "par_prefetch_invalid_witness",
        SampleInvalid::Debt => "par_prefetch_invalid_debt",
        SampleInvalid::UpperRung => "par_prefetch_invalid_upper_rung",
        SampleInvalid::Packed => "par_prefetch_invalid_packed",
    }
}

/// Plan every distinct surface-sampled ref of one stream's draws on the pool.
///
/// `tables` are the draws' vertex and fragment texture tables; each bind that
/// retains its resource object is a candidate. Returns the guard that keeps the
/// plans available until the stream ends.
pub(crate) fn prefetch_stream<'a>(
    state: &DeviceState,
    task_id: u32,
    tables: impl Iterator<Item = &'a [TextureBind]>,
) -> PrefetchGuard {
    if prefetch_mode() == PrefetchMode::Off {
        return PrefetchGuard { pushed: false };
    }
    let Some(pool) = state.par() else {
        crate::runtime::drain::note_store_route("par_prefetch_no_pool");
        return PrefetchGuard { pushed: false };
    };
    let mut seen = HashSet::new();
    let mut items: Vec<(u32, std::sync::Arc<crate::model::TaskResource>)> = Vec::new();
    for table in tables {
        for bind in table {
            if bind.texture_ref == 0 {
                continue;
            }
            let Some(resource) = bind.resource.clone() else {
                continue;
            };
            if seen.insert(bind.texture_ref) {
                items.push((bind.texture_ref, resource));
            }
        }
    }
    if items.len() < 2 {
        // One ref gains nothing from the pool: its plan would run on this
        // thread just before the serial ladder would.
        crate::runtime::drain::note_store_route("par_prefetch_stream_small");
        return PrefetchGuard { pushed: false };
    }
    // A whole stream's fills in one scope: the stream prefetch's budget is the
    // chunk limit unchanged (PassPrep supersedes this path while it is on).
    let memo_budget = std::sync::atomic::AtomicU64::new(sample_plan::memo_fill_chunk_limit());
    let scoped = pool.scope(state, &items, |st, env, (texture_ref, resource)| {
        plan_item(
            st,
            env,
            task_id,
            *texture_ref,
            resource.clone(),
            &memo_budget,
        )
    });
    crate::runtime::drain::note_store_route("par_prefetch_stream");
    crate::runtime::drain::note_store_route_n("par_fork_us", scoped.fork_us);
    crate::runtime::drain::note_store_route_n("par_wait_us", scoped.wait_us);
    crate::runtime::drain::note_store_route_n("par_worker_busy_us", scoped.helper_busy_us);
    let mut plans = HashMap::with_capacity(items.len());
    for ((texture_ref, _), result) in items.iter().zip(scoped.results) {
        match result {
            Ok(Ok(plan)) => {
                crate::runtime::drain::note_store_route(match plan {
                    StreamPlan::Surface(_) => "par_prefetch_planned",
                    StreamPlan::RefView(_) => "par_prefetch_planned_ref_view",
                    StreamPlan::Linear(_) => "par_prefetch_planned_linear",
                    StreamPlan::MapperRef(_) => "par_prefetch_planned_mapper_ref",
                    StreamPlan::Memo(_) => "par_prefetch_planned_memo",
                });
                plans.insert((task_id, *texture_ref), plan);
            }
            Ok(Err(bail)) => crate::runtime::drain::note_store_route(bail_route(bail)),
            Err(_) => crate::runtime::drain::note_store_route("par_prefetch_panic"),
        }
    }
    STACK.with(|stack| {
        stack.borrow_mut().push(PassPlans {
            map: plans,
            consumed: Vec::new(),
            buffer_rails: HashMap::new(),
            buffer_exacts: HashMap::new(),
            store_pages: None,
        })
    });
    PrefetchGuard { pushed: true }
}

/// What a stream plan offers one bind.
pub(crate) enum Prefetched {
    /// Serve the bind from the validated plan (mode `on`); routes are noted.
    Serve((u32, u32, u32, SampledSourceRequest)),
    /// The plan's serve, routes not noted, to compare against the serial one.
    Shadow((u32, u32, u32, SampledSourceRequest)),
    /// No plan, or one that no longer holds: run the serial ladder.
    Serial,
}

/// TEMP (lab, `REIMS_VGPU_LAB_DRAIN_LANES=2`): stage a transient lane-only
/// frame for a bind whose pass pushed none — a single-draw pass opens no
/// `PassPrep`, so its one ask would otherwise miss the frame machinery the
/// lane's plans serve through. `None` when a frame already stands (the fill
/// inside [`prefetched_source`] handles that case) or the lane holds nothing;
/// the returned guard pops the frame when the ask is done, so the repeat a
/// witness-moving commit hands back dies with it — correct, because a
/// frame-less pass is a one-draw pass and has no later bind to serve.
fn stage_lane_frame(task_id: u32, texture_ref: u32) -> Option<PrefetchGuard> {
    if STACK.with(|stack| !stack.borrow().is_empty()) {
        return None;
    }
    let key = (task_id, texture_ref);
    let plan = super::lanes::take_plan(&key)?;
    let guard = push_empty_frame();
    STACK.with(|stack| {
        if let Some(frame) = stack.borrow_mut().last_mut() {
            frame.map.insert(key, plan);
        }
    });
    crate::runtime::drain::note_store_route("lanes_merged");
    crate::runtime::drain::note_store_route_n("lanes_merged_n", 1);
    Some(guard)
}

/// The stream plan's answer for one bind, validated at the point the serial
/// ladder would resolve it.
pub(crate) fn prefetched_source<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
) -> Prefetched {
    // `REIMS_VGPU_LAB_PASS_PREP` supersedes `PAR_PREFETCH` while it is not
    // `off`: its chunks push frames on this same stack, and the mode its
    // plans are taken under is the PassPrep one.
    let mode = match super::pass_prep::pass_prep_mode() {
        super::pass_prep::PassPrepMode::Off => prefetch_mode(),
        super::pass_prep::PassPrepMode::Shadow => PrefetchMode::Shadow,
        super::pass_prep::PassPrepMode::On => PrefetchMode::On,
    };
    if mode == PrefetchMode::Off {
        return Prefetched::Serial;
    }
    // TEMP (lab, DRAIN_LANES): the lane's plan for a frame-less ask stages a
    // transient frame, so the validate/serve match below runs unchanged —
    // the lane serves through exactly the path a chunk's plan does.
    let _lane_frame = super::lanes::lanes_on()
        .then(|| stage_lane_frame(task_id, texture_ref))
        .flatten();
    STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(frame) = stack.last_mut() else {
            return Prefetched::Serial;
        };
        let key = (task_id, texture_ref);
        // TEMP (lab, DRAIN_LANES): a frame miss consults the lane store
        // before the serial ladder. The taken plan becomes this frame's entry,
        // so the repeat machinery below owns it exactly as it owns a chunk's
        // plan; an invalidated plan lands in the existing invalid routes and
        // the serial ladder serves the bind unchanged.
        if super::lanes::lanes_on() && !frame.map.contains_key(&key) {
            if let Some(plan) = super::lanes::take_plan(&key) {
                crate::runtime::drain::note_store_route("lanes_merged");
                crate::runtime::drain::note_store_route_n("lanes_merged_n", 1);
                frame.map.insert(key, plan);
            }
        }
        match frame.map.get(&key) {
            None => Prefetched::Serial,
            Some(StreamPlan::Surface(plan)) => {
                if let Err(invalid) = sample_plan::validate_sampled_plan(
                    state,
                    host,
                    plan,
                    task_id,
                    texture_ref,
                    resource,
                    may_bind_resident,
                ) {
                    crate::runtime::drain::note_store_route(invalid_route(invalid));
                    return Prefetched::Serial;
                }
                match mode {
                    PrefetchMode::On => {
                        crate::runtime::drain::note_store_route("par_prefetch_hit");
                        Prefetched::Serve(sample_plan::take_sampled_plan(state, plan, true))
                    }
                    _ => Prefetched::Shadow(sample_plan::take_sampled_plan(state, plan, false)),
                }
            }
            Some(StreamPlan::RefView(plan)) => {
                let apply = mode == PrefetchMode::On;
                match sample_plan::commit_ref_view_plan(
                    state,
                    host,
                    plan,
                    task_id,
                    texture_ref,
                    resource,
                    apply,
                ) {
                    Ok((served, repeat)) if apply => {
                        crate::runtime::drain::note_store_route("par_prefetch_hit_ref_view");
                        // The repeat commits the pass's later binds of the ref
                        // from the post state (H6); a missing repeat only when
                        // the entry vanished under the commit, which replans.
                        match repeat {
                            Some(next) => {
                                frame.map.insert(key, StreamPlan::RefView(next));
                            }
                            None => {
                                frame.map.remove(&key);
                                frame.consumed.push(texture_ref);
                            }
                        }
                        Prefetched::Serve(served)
                    }
                    Ok((predicted, _)) => Prefetched::Shadow(predicted),
                    Err(invalid) => {
                        crate::runtime::drain::note_store_route(invalid_route(invalid));
                        // Spent either way: the witness entry it rests on moves
                        // with whatever the serial rail does next.
                        frame.map.remove(&key);
                        frame.consumed.push(texture_ref);
                        Prefetched::Serial
                    }
                }
            }
            Some(StreamPlan::Linear(plan)) => {
                let apply = mode == PrefetchMode::On;
                match sample_plan::commit_linear_plan(
                    state,
                    host,
                    plan,
                    task_id,
                    texture_ref,
                    resource,
                    may_bind_resident,
                    apply,
                ) {
                    Ok((served, repeat)) if apply => {
                        crate::runtime::drain::note_store_route("par_prefetch_hit_linear");
                        // The direct arm hands its own plan back (it moved
                        // nothing) and the gather arm its repeat; both serve
                        // the next bind of the ref from the post state.
                        match repeat {
                            Some(next) => {
                                frame.map.insert(key, StreamPlan::Linear(next));
                            }
                            None => {
                                frame.map.remove(&key);
                                frame.consumed.push(texture_ref);
                            }
                        }
                        Prefetched::Serve(served)
                    }
                    Ok((predicted, _)) => Prefetched::Shadow(predicted),
                    Err(invalid) => {
                        crate::runtime::drain::note_store_route(invalid_route(invalid));
                        frame.map.remove(&key);
                        frame.consumed.push(texture_ref);
                        Prefetched::Serial
                    }
                }
            }
            Some(StreamPlan::MapperRef(plan)) => {
                let apply = mode == PrefetchMode::On;
                match sample_plan::commit_mapper_ref_plan(
                    state,
                    host,
                    plan,
                    task_id,
                    texture_ref,
                    resource,
                    apply,
                ) {
                    Ok((served, repeat)) if apply => {
                        crate::runtime::drain::note_store_route("par_prefetch_hit_mapper_ref");
                        match repeat {
                            Some(next) => {
                                frame.map.insert(key, StreamPlan::MapperRef(next));
                            }
                            None => {
                                frame.map.remove(&key);
                                frame.consumed.push(texture_ref);
                            }
                        }
                        Prefetched::Serve(served)
                    }
                    Ok((predicted, _)) => Prefetched::Shadow(predicted),
                    Err(invalid) => {
                        crate::runtime::drain::note_store_route(invalid_route(invalid));
                        frame.map.remove(&key);
                        frame.consumed.push(texture_ref);
                        Prefetched::Serial
                    }
                }
            }
            Some(StreamPlan::Memo(plan)) => {
                let apply = mode == PrefetchMode::On;
                match sample_plan::commit_memo_plan(
                    state,
                    host,
                    plan,
                    task_id,
                    texture_ref,
                    resource,
                    apply,
                ) {
                    Ok((served, repeat)) if apply => {
                        crate::runtime::drain::note_store_route("par_prefetch_hit_memo");
                        match repeat {
                            Some(next) => {
                                frame.map.insert(key, StreamPlan::Memo(next));
                            }
                            None => {
                                frame.map.remove(&key);
                                frame.consumed.push(texture_ref);
                            }
                        }
                        Prefetched::Serve(served)
                    }
                    Ok((predicted, _)) => Prefetched::Shadow(predicted),
                    Err(invalid) => {
                        crate::runtime::drain::note_store_route(invalid_route(invalid));
                        frame.map.remove(&key);
                        frame.consumed.push(texture_ref);
                        Prefetched::Serial
                    }
                }
            }
        }
    })
}

/// Compare a plan's serve against the serial ladder's for the same bind.
pub(crate) fn shadow_compare(
    texture_ref: u32,
    planned: &(u32, u32, u32, SampledSourceRequest),
    serial: Option<&(u32, u32, u32, SampledSourceRequest)>,
) {
    let (pw, ph, pmid, psrc) = planned;
    let verdict = match serial {
        None => "par_shadow_mismatch_serial_none",
        Some((sw, sh, smid, _)) if (sw, sh, smid) != (pw, ph, pmid) => {
            "par_shadow_mismatch_geometry"
        }
        Some((_, _, _, ssrc)) => match (psrc, ssrc) {
            (SampledSourceRequest::Target(a, fa), SampledSourceRequest::Target(b, fb)) => {
                if a == b && fa == fb {
                    "par_shadow_match_target"
                } else {
                    "par_shadow_mismatch_target"
                }
            }
            (
                SampledSourceRequest::GuestRuns(asrc, al, af, ad, aid, av, _),
                SampledSourceRequest::GuestRuns(bsrc, bl, bf, bd, bid, bv, _),
            ) => {
                // A generation the witness spent this bind is issued from the
                // device-global counter by whichever arm asked, so only a
                // vouched one (the entry's own) can be compared.
                let same_identity = aid.map(|i| i.key) == bid.map(|i| i.key)
                    && (*av == crate::runtime::gather_witness::GatherVouch::Fresh
                        || aid.map(|i| i.generation) == bid.map(|i| i.generation));
                if std::sync::Arc::ptr_eq(&asrc.runs, &bsrc.runs)
                    && asrc.source_offset == bsrc.source_offset
                    && asrc.total_len == bsrc.total_len
                    && asrc.row_length_texels == bsrc.row_length_texels
                    && (al, af, ad, av) == (bl, bf, bd, bv)
                    && same_identity
                {
                    "par_shadow_match_runs"
                } else {
                    "par_shadow_mismatch_runs"
                }
            }
            (
                SampledSourceRequest::Bytes(ab, aid, al, ao),
                SampledSourceRequest::Bytes(bb, bid, bl, bo),
            ) => {
                let same_bytes = std::sync::Arc::ptr_eq(ab, bb) || ab[..] == bb[..];
                // A planned memo *fill* predicts its bytes with no identity:
                // the generation a fill names is minted by whichever arm runs,
                // and either arm's identity forces the same engine upload, so
                // a plan-side `None` matches whatever the serial fill minted.
                // Everywhere else the identity is an entry's own and compares.
                let same_identity = match (aid, bid) {
                    (Some(a), Some(b)) => (a.key, a.generation) == (b.key, b.generation),
                    (None, _) | (_, None) => true,
                };
                if same_bytes
                    && same_identity
                    && al.layout() == bl.layout()
                    && std::mem::discriminant(ao) == std::mem::discriminant(bo)
                {
                    "par_shadow_match_bytes"
                } else {
                    "par_shadow_mismatch_bytes"
                }
            }
            // A cross-variant answer. The total keeps the gate's name (it
            // stays zero on a healthy boot); the split beside it names the
            // pair, so a boot that breaks it says *which* rung answered
            // differently without a rebuild — the fs2s1 diagnosis needed
            // exactly this and had to infer the pair from the census.
            pair => {
                crate::runtime::drain::note_store_route(variant_pair_route(pair));
                "par_shadow_mismatch_variant"
            }
        },
    };
    crate::runtime::drain::note_store_route(verdict);
    if verdict.contains("mismatch")
        && crate::observe::first_sight("par_shadow_mismatch", u64::from(texture_ref))
    {
        crate::observe::fail(format!(
            "par_shadow_mismatch ref={texture_ref} kind={verdict} planned_mid={pmid} {pw}x{ph}{}",
            match serial {
                Some((_, _, _, ssrc)) => format!(
                    " planned={} serial={}",
                    variant_name(psrc),
                    variant_name(ssrc)
                ),
                None => String::new(),
            }
        ));
    }
}

/// The cross-variant pair a `par_shadow_mismatch_variant` bind answered, as a
/// census route: e.g. `par_shadow_variant_runs_vs_target` — the plan served
/// `GuestRuns`, the serial ladder a resident `Target` (the readiness-flip
/// class the fs2s1 boot hit).
fn variant_pair_route(pair: (&SampledSourceRequest, &SampledSourceRequest)) -> &'static str {
    let (planned, serial) = pair;
    match (variant_name(planned), variant_name(serial)) {
        ("target", "runs") => "par_shadow_variant_target_vs_runs",
        ("target", "bytes") => "par_shadow_variant_target_vs_bytes",
        ("runs", "target") => "par_shadow_variant_runs_vs_target",
        ("runs", "bytes") => "par_shadow_variant_runs_vs_bytes",
        ("bytes", "target") => "par_shadow_variant_bytes_vs_target",
        _ => "par_shadow_variant_bytes_vs_runs",
    }
}

fn variant_name(src: &SampledSourceRequest) -> &'static str {
    match src {
        SampledSourceRequest::Target(..) => "target",
        SampledSourceRequest::GuestRuns(..) => "runs",
        SampledSourceRequest::Bytes(..) => "bytes",
        // TEMP (lab, PASS_STAGE): the staged twin of the runs rail.
        SampledSourceRequest::Staged(..) => "staged",
    }
}
