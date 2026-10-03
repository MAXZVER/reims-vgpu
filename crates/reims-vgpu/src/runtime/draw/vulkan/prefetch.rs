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
//! re-validates the same plan. A ref-view plan (the zero-copy plane of a
//! ref-texture view) applies a gather-witness decision, which moves the
//! witness, so it serves one bind and is then dropped: the ref lands in the
//! frame's `consumed` list, which [`super::pass_prep`]'s `ahead` reads to drop
//! the ref from its planned set, so the next chunk that needs it plans it
//! again from the witness's new state (the H6 repeat of the design).

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
pub(crate) struct PassPlans {
    map: HashMap<(u32, u32), StreamPlan>,
    consumed: Vec<u32>,
}

thread_local! {
    /// The plans of each stream being encoded on this thread, innermost last.
    static STACK: RefCell<Vec<PassPlans>> = const { RefCell::new(Vec::new()) };
}

/// One planned bind of a stream.
pub(crate) enum StreamPlan {
    /// A top rung of the surface ladder: a resident or the host cache.
    Surface(SamplePlan),
    /// A ref-texture view's plane on the zero-copy rail.
    RefView(sample_plan::RefViewPlan),
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
    STACK.with(|stack| stack.borrow_mut().push(PassPlans {
        map: HashMap::new(),
        consumed: Vec::new(),
    }));
    PrefetchGuard { pushed: true }
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

/// The one job body both [`prefetch_stream`] and [`super::pass_prep`] run per
/// item, on the pool: the surface ladder's plannable top, and — where it bails
/// with a materializing ref-texture view — that view's zero-copy plane.
pub(super) fn plan_item(
    st: &DeviceState,
    env: &crate::runtime::par::WorkerEnv<'_>,
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
) -> Result<StreamPlan, SampleBail> {
    let host = env.host();
    match plan_sampled_source(st, &host, task_id, texture_ref, resource.clone(), true) {
        Ok(plan) => Ok(StreamPlan::Surface(plan)),
        Err(SampleBail::RefTextureView) => {
            sample_plan::plan_ref_view_zero_copy(st, &host, task_id, texture_ref, resource)
                .map(StreamPlan::RefView)
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
    let scoped = pool.scope(state, &items, |st, env, (texture_ref, resource)| {
        plan_item(st, env, task_id, *texture_ref, resource.clone())
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
                });
                plans.insert((task_id, *texture_ref), plan);
            }
            Ok(Err(bail)) => crate::runtime::drain::note_store_route(bail_route(bail)),
            Err(_) => crate::runtime::drain::note_store_route("par_prefetch_panic"),
        }
    }
    STACK.with(|stack| stack.borrow_mut().push(PassPlans {
        map: plans,
        consumed: Vec::new(),
    }));
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
    STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(frame) = stack.last_mut() else {
            return Prefetched::Serial;
        };
        let key = (task_id, texture_ref);
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
                    Ok(served) if apply => {
                        crate::runtime::drain::note_store_route("par_prefetch_hit_ref_view");
                        frame.map.remove(&key);
                        frame.consumed.push(texture_ref);
                        Prefetched::Serve(served)
                    }
                    Ok(predicted) => Prefetched::Shadow(predicted),
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
                if same_bytes
                    && aid.as_ref().map(|i| (i.key, i.generation))
                        == bid.as_ref().map(|i| (i.key, i.generation))
                    && al.layout() == bl.layout()
                    && std::mem::discriminant(ao) == std::mem::discriminant(bo)
                {
                    "par_shadow_match_bytes"
                } else {
                    "par_shadow_mismatch_bytes"
                }
            }
            _ => "par_shadow_mismatch_variant",
        },
    };
    crate::runtime::drain::note_store_route(verdict);
    if verdict.contains("mismatch")
        && crate::observe::first_sight("par_shadow_mismatch", u64::from(texture_ref))
    {
        crate::observe::fail(format!(
            "par_shadow_mismatch ref={texture_ref} kind={verdict} planned_mid={pmid} {pw}x{ph}"
        ));
    }
}
