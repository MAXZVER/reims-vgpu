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
//! A plan is not consumed by its first use. Taking it notes census routes and
//! changes no device state, so its reads still describe the state after it,
//! and every later draw of the stream that binds the same ref re-validates the
//! same plan.

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

thread_local! {
    /// The plans of each stream being encoded on this thread, innermost last.
    static STACK: RefCell<Vec<HashMap<(u32, u32), SamplePlan>>> = const { RefCell::new(Vec::new()) };
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

fn bail_route(bail: SampleBail) -> &'static str {
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
    }
}

fn invalid_route(invalid: SampleInvalid) -> &'static str {
    match invalid {
        SampleInvalid::Key => "par_prefetch_invalid_key",
        SampleInvalid::Resource => "par_prefetch_invalid_resource",
        SampleInvalid::Mapping => "par_prefetch_invalid_mapping",
        SampleInvalid::Backing => "par_prefetch_invalid_backing",
        SampleInvalid::GuestWrite => "par_prefetch_invalid_guest_write",
        SampleInvalid::HostCache => "par_prefetch_invalid_host_cache",
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
        let host = env.host();
        plan_sampled_source(st, &host, task_id, *texture_ref, resource.clone(), true)
    });
    crate::runtime::drain::note_store_route("par_prefetch_stream");
    crate::runtime::drain::note_store_route_n("par_fork_us", scoped.fork_us);
    crate::runtime::drain::note_store_route_n("par_wait_us", scoped.wait_us);
    crate::runtime::drain::note_store_route_n("par_worker_busy_us", scoped.helper_busy_us);
    let mut plans = HashMap::with_capacity(items.len());
    for ((texture_ref, _), result) in items.iter().zip(scoped.results) {
        match result {
            Ok(Ok(plan)) => {
                crate::runtime::drain::note_store_route("par_prefetch_planned");
                plans.insert((task_id, *texture_ref), plan);
            }
            Ok(Err(bail)) => crate::runtime::drain::note_store_route(bail_route(bail)),
            Err(_) => crate::runtime::drain::note_store_route("par_prefetch_panic"),
        }
    }
    STACK.with(|stack| stack.borrow_mut().push(plans));
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
pub(crate) fn prefetched_source<M: HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
) -> Prefetched {
    let mode = prefetch_mode();
    if mode == PrefetchMode::Off {
        return Prefetched::Serial;
    }
    STACK.with(|stack| {
        let stack = stack.borrow();
        let Some(plan) = stack
            .last()
            .and_then(|plans| plans.get(&(task_id, texture_ref)))
        else {
            return Prefetched::Serial;
        };
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
