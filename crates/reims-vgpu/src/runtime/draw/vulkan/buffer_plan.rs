//! TEMP (lab, PASS_PREP step 3, design §8.3): planning a draw's buffer binds
//! away from the drain thread, and taking the plan on it.
//!
//! # What is planned, and why only that
//!
//! A bind's bytes cost the drain two things: the resolve (backing decode,
//! registry lookups) and, when nothing warm answers, the zero-copy
//! construction — a packed alias (`ensure_packed_resource`), a whole-allocation
//! walk (`ensure_whole_runs`), or an exact-window walk — each of which mutates
//! [`crate::runtime::bound_buffers::BoundBuffers`] or the host's page
//! registries, and each of which the worker host refuses to perform
//! ([`crate::qemu::host_ops::WorkerHost`] offers no `map_pages`). So the plan
//! covers exactly the arms of [`super::held_buffer_content`] — the three warm
//! rails the registry answers a bind from — and treats every construction as a
//! bail the drain serves serially, unchanged:
//!
//! * **Packed** — the retained packed alias answers every `(offset, cap)` in
//!   O(1); the plan carries its identity and the commit re-derives the slice
//!   with the bind's real cap, which the plan never needed to know.
//! * **Whole** — the retained whole-allocation runs answer every offset the
//!   floor admits; same shape, same cap-freedom.
//! * **Held** — a per-`(offset, cap)` exact-window entry; plannable only where
//!   the cap is known at plan time, which is the index bind (its cap is
//!   `index_count × element size`, a function of the record alone).
//! * **Walked** — an exact window the planner walked read-only. On the pool
//!   this succeeds only where the run memo already holds the window (a worker
//!   cannot `map_pages`); everywhere else it is a `ColdWindow` bail and the
//!   drain walks. The commit lands it as the fact the design names:
//!   [`BoundBufferHold`] — the insert the serial walk would have made.
//!
//! # The contract (the sample-plan one, restated for buffers)
//!
//! Commit order and every guest-observable effect stay exactly the serial
//! ones. The plan is built from state it only reads, it refuses
//! ([`BufferBail`]) wherever the serial ladder would mutate, and it carries
//! the reads it rests on for the commit to re-check: the retained resource
//! identity (`Arc::ptr_eq`), and the registry's retirement generation — the
//! buffer registry's `map_generation`, bumped by every retire rule and every
//! clear, which are exactly the guest's page-table and naming announcements.
//! The rail's own identity (the entry's `Arc`s) is re-checked by pointer, so
//! a replaced entry invalidates even without a retirement. A rail that
//! *appeared* above the planned one since the plan is an `UpperRung`
//! invalidation — the serial ladder would answer from it, so the plan must
//! not answer past it (the same discipline step 2 applies to the GVA-resident
//! rung above the linear rail).
//!
//! # Shadow
//!
//! `shadow` predicts without applying (a walked window is not inserted), the
//! serial ladder runs next, and [`shadow_compare_buffer`] checks the two
//! `BufferContent`s — bytes by value or identity, guest runs by value —
//! noting `par_shadow_buf_*` the way `prefetch::shadow_compare` notes
//! `par_shadow_*` for sampled binds. The registry's end state needs no
//! comparison of its own: shadow applies nothing, so the serial ladder leaves
//! the registry exactly where it would have without the plan. `on` mode's
//! end state is what the tests assert against the serial ladder's.

use super::prefetch::{self, PassPlans};
use super::*;

/// The bind an exact plan serves: `(task, buffer_ref, offset, cap)` — the
/// registry's own key, which is why a planned exact window cannot be served to
/// a shader that proved a different extent.
pub(super) type BufferExactKey = (u32, u32, u64, Option<u64>);

/// Why a buffer bind was not planned. Every arm is served by the serial ladder
/// on the drain thread, unchanged; the arm names which step would have
/// mutated (or which read-only walk refused).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BufferBail {
    /// `buffer_ref == 0`, or a resource the bind did not retain (the plan
    /// resolves through the retained identity; the numeric-ref path reads the
    /// guest, which the serial ladder does better).
    NullRef,
    /// The backing did not resolve (wrong object type, undecodable descriptor,
    /// no span) — the serial ladder reports it through its own vocabulary.
    ColdBacking,
    /// No packed entry stands for this geometry, or a mismatched one does:
    /// `ensure_packed_resource` would construct or replace — a drain-side
    /// mutation.
    ColdPacked,
    /// No whole-allocation entry stands (or a mismatched one does): the
    /// `ensure_whole_runs` walk and insert are the drain's.
    ColdWhole,
    /// No warm rail and no held entry answered, and the exact-window walk is
    /// not plannable read-only (the worker host cannot map pages into runs, or
    /// the window refused).
    ColdWindow,
    /// The exact window's span fell below the zero-copy floor for a window
    /// that would gather — the refusal the serial walk names itself.
    BelowFloor,
    /// The bind's offset is past the end of the allocation: the serial rail
    /// notes `zc_buffer_offset_past_end` and reads on the CPU.
    OffsetPastEnd,
    /// TEMP (lab, PASS_STAGE): a writeback debt stands over the bind's
    /// allocation; the serial CPU read's settle is the drain's.
    OwedStage,
}

/// Why a plan was dropped at commit. The serial ladder serves the bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BufferInvalid {
    /// The commit is for a different bind than the plan (an exact plan keyed
    /// by another offset or cap).
    Key,
    /// The ref resolves to a different retained resource now.
    Resource,
    /// A retirement landed between the plan and the commit: the registry's
    /// generation moved.
    Registry,
    /// The packed alias the plan borrowed is gone or was replaced.
    Packed,
    /// The whole-allocation runs the plan borrowed are gone or were replaced.
    Whole,
    /// The held entry the plan borrowed vanished or was replaced.
    Held,
    /// A rail above the planned one would answer the bind now — the serial
    /// answer's shape changed above this plan.
    UpperRung,
}

/// The warm registry rail a planned bind serves.
#[derive(Clone)]
enum BufferRail {
    /// `bound_buffers.packed_available` answered: the retained alias, whose
    /// identity the commit re-checks by pointer.
    Packed {
        runs: std::sync::Arc<Vec<crate::backend::vulkan::engine::GuestRun>>,
        pages: std::sync::Arc<Vec<crate::runtime::guest_ram_map::GuestWindowRun>>,
    },
    /// `bound_buffers.whole` answered with this allocation's runs.
    Whole {
        runs: std::sync::Arc<Vec<crate::backend::vulkan::engine::GuestRun>>,
    },
    /// A held exact-window entry at `(offset, cap)` answered.
    Held {
        bound: crate::runtime::bound_buffers::BoundBuffer,
    },
    /// The planner walked the exact window read-only; the commit lands it as
    /// [`BoundBufferHold`] — the insert the serial walk would have made.
    Walked {
        bound: crate::runtime::bound_buffers::BoundBuffer,
        /// Bytes the extent cap shaved off the full window, for the
        /// `zc_buffer_extent_narrowed` pair the serial walk notes.
        narrowed: Option<u64>,
    },
    /// TEMP (lab, PASS_STAGE, design §8.4): the window the serial ladder
    /// would have CPU-read, staged into a worker arena instead. The commit
    /// serves [`crate::backend::vulkan::engine::BufferContent::Staged`] and
    /// the engine binds the window — the read and the staging memcpy both ran
    /// on the worker.
    Staged {
        staged: crate::backend::vulkan::engine::pass_stage::StagedSlice,
        /// `host_writes`' epoch at the worker's read; a device write into the
        /// window since is a race the serial read's settle would have closed.
        host_epoch: u64,
        /// The window's guest pages, for that check.
        gpas: std::sync::Arc<Vec<u64>>,
    },
}

/// A planned buffer bind: the warm rail that answers it, the reads the answer
/// rests on, and — for the exact arms — the bind's own key.
#[derive(Clone)]
pub(crate) struct BufferContentPlan {
    task_id: u32,
    buffer_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    /// The backing the plan resolved from the retained resource (pure decode).
    gva: u64,
    size: u64,
    /// `bound_buffers.generation()` when planned.
    generation: u64,
    /// `(offset, cap)` for the exact arms ([`BufferRail::Held`] /
    /// [`BufferRail::Walked`]); `None` for the rail plans, which serve every
    /// bind of the reference.
    exact: Option<(u64, Option<u64>)>,
    rail: BufferRail,
}

impl BufferContentPlan {
    /// TEMP (lab, PASS_STAGE): whether this plan answers one exact bind (and
    /// so merges under its key) rather than every bind of the reference.
    pub(super) fn is_exact(&self) -> bool {
        self.exact.is_some()
    }

    /// TEMP (lab, PASS_STAGE): the exact bind's key halves. `0`/`None` on a
    /// rail plan, which the caller does not ask.
    pub(super) fn exact_offset(&self) -> u64 {
        self.exact.map_or(0, |(offset, _)| offset)
    }

    pub(super) fn exact_cap(&self) -> Option<u64> {
        self.exact.and_then(|(_, cap)| cap)
    }

    /// The hit route family's split for this plan's rail.
    fn rail_slug(&self) -> &'static str {
        match self.rail {
            BufferRail::Packed { .. } => "pass_prep_buf_hit_packed",
            BufferRail::Whole { .. } => "pass_prep_buf_hit_whole",
            BufferRail::Held { .. } => "pass_prep_buf_hit_held",
            BufferRail::Walked { .. } => "pass_prep_buf_hit_walked",
            BufferRail::Staged { .. } => "pass_prep_buf_hit_staged",
        }
    }
}

/// The planned route family's split for a plan's rail (noted by the chunk
/// driver when the scope returns).
pub(super) fn planned_route(plan: &BufferContentPlan) -> &'static str {
    match plan.rail {
        BufferRail::Packed { .. } => "pass_prep_buf_planned_packed",
        BufferRail::Whole { .. } => "pass_prep_buf_planned_whole",
        BufferRail::Held { .. } => "pass_prep_buf_planned_held",
        BufferRail::Walked { .. } => "pass_prep_buf_planned_walked",
        BufferRail::Staged { .. } => "pass_prep_buf_planned_staged",
    }
}

/// The exact key an indexed draw's index bind plans under: the bind's
/// `(ref, offset, need)`, `need` derived from the record the way
/// `resolve_index_window_reason` derives it. `None` where the record states
/// no usable window (an unsupported index type, a zero or overflowing count)
/// — the serial loader reports those itself.
pub(super) fn index_key(task_id: u32, info: &IndexedDrawInfo) -> Option<BufferExactKey> {
    let need = index_elem_size(info.index_type)
        .and_then(|elem| (info.index_count as usize).checked_mul(elem))
        .filter(|n| *n > 0)?;
    Some((
        task_id,
        info.index_buffer_ref,
        info.index_buffer_offset,
        Some(need as u64),
    ))
}

/// The colour-0 key the drain latches for the pass's Store walk — `None`
/// where the walk would refuse before walking, so nothing is planned.
pub(super) fn latch_store_target(c0: &ColorRtRequest) -> Option<StoreTargetKey> {
    StoreTargetKey::of(c0)
}

/// The fact a walked plan lands at commit: the hold the serial walk inserts.
///
/// Named for the design's own shape (`BoundBufferHold{task, ref, offset, cap,
/// bound}`) so the census and the tests can talk about the one mutation this
/// module ever makes.
struct BoundBufferHold<'a> {
    task: u32,
    buffer_ref: u32,
    offset: u64,
    cap: Option<u64>,
    bound: &'a crate::runtime::bound_buffers::BoundBuffer,
}

impl BoundBufferHold<'_> {
    fn land(self, state: &mut DeviceState) {
        state.bound_buffers.insert(
            self.task,
            self.buffer_ref,
            self.offset,
            self.cap,
            self.bound.clone(),
        );
    }
}

/// Whether the packed alias would answer this bind — the first arm of
/// [`super::held_buffer_content`], from state this call only reads.
fn packed_serves(
    state: &DeviceState,
    task_id: u32,
    buffer_ref: u32,
    gva: u64,
    size: u64,
    offset: u64,
) -> bool {
    state
        .bound_buffers
        .packed_available(task_id, buffer_ref, gva, size)
        .is_some_and(|p| offset < p.size)
}

/// Whether the whole-allocation runs would answer this bind — the second arm,
/// exactly `whole_runs_bind`'s own admission (knob, floor and cap included).
fn whole_serves(
    state: &DeviceState,
    task_id: u32,
    buffer_ref: u32,
    gva: u64,
    size: u64,
    offset: u64,
    cap: Option<u64>,
) -> bool {
    state
        .bound_buffers
        .whole(task_id, buffer_ref)
        .is_some_and(|w| w.gva == gva && w.size == size)
        && whole_runs_bind(state, task_id, buffer_ref, offset, cap).is_some()
}

/// Plan a buffer bind's warm rails from state this call only reads.
///
/// This is the planner for the vertex/fragment binds, whose extent cap comes
/// from the pipeline's shader reflection and is therefore not known where the
/// plan runs; the two rails it covers are exactly the ones that answer every
/// cap. `host` is the worker's view; the rail arms never touch it, but the
/// one planner keeps the serial ladder's own order.
pub(super) fn plan_buffer_rails<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    first_offset: u64,
    stager: Option<&super::pass_stage::Stager>,
) -> Result<BufferContentPlan, BufferBail> {
    plan_from_backing(
        state,
        host,
        task_id,
        buffer_ref,
        resource,
        Some((first_offset, None)),
        stager,
    )
}

/// Plan an indexed draw's index-buffer bind: the one buffer bind whose extent
/// (`index_count × element size`) is a function of the record, so the exact
/// arms — the held entry and the read-only window walk — are plannable too.
pub(super) fn plan_index_content<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    info: &IndexedDrawInfo,
    stager: Option<&super::pass_stage::Stager>,
) -> Result<BufferContentPlan, BufferBail> {
    if info.index_buffer_ref == 0 {
        return Err(BufferBail::NullRef);
    }
    let (_backing, need) = resolve_index_window_reason(state, host, task_id, info)
        .map_err(|_| BufferBail::ColdBacking)?;
    // The resource the numeric ref names, resolved the way the serial ladder
    // resolves it, so the commit's identity check compares like with like.
    let resource = objects::resolve_resource(state, host, task_id, info.index_buffer_ref)
        .map_err(|_| BufferBail::ColdBacking)?;
    plan_from_backing(
        state,
        host,
        task_id,
        info.index_buffer_ref,
        resource,
        Some((info.index_buffer_offset, Some(need as u64))),
        stager,
    )
}

/// The planner itself: the serial ladder's `try_buffer_zero_copy_resolved`
/// decision order in read-only form, taking only what each rail needs.
///
/// `bind` is `Some((offset, cap))` where both are known at plan time (the
/// index bind), `None` for the cap-free rail plans. Every construction step —
/// `ensure_packed_resource`, the `ensure_whole_runs` walk, the fresh window
/// walk on a host whose worker cannot map pages — is a [`BufferBail`], never
/// a mutation. The one walk the planner may run is the exact window's, and
/// only where the run memo answers it without `map_pages`; its helpers note
/// their own walk-internal routes wherever they run, while the decision routes
/// the serial arm notes ride in the plan and are noted at commit.
fn plan_from_backing<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    bind: Option<(u64, Option<u64>)>,
    stager: Option<&super::pass_stage::Stager>,
) -> Result<BufferContentPlan, BufferBail> {
    if buffer_ref == 0 {
        return Err(BufferBail::NullRef);
    }
    let (gva, size) = objects::resolve_buffer_span_from_resource(state, &resource)
        .map_err(|_| BufferBail::ColdBacking)?;
    let generation = state.bound_buffers.generation();
    // The packed gate, in the serial ladder's own position: a matching
    // Available entry is the plan's first rail; a matching Unavailable entry
    // means the construction already refused for this geometry (a held
    // negative under the same retirement rules) and the serial ladder falls
    // past it without mutating — so the plan does too. Anything else (no
    // entry, mismatched geometry) is a construction the drain would run.
    match state.bound_buffers.packed(task_id, buffer_ref) {
        // TEMP (lab, PASS_STAGE): a never-primed reference stages where the
        // floor already refused the zero-copy rail (see `stage_or_bail`);
        // floor-clearing cold refs stay the drain's constructions.
        None => {
            return stage_or_bail(
                state,
                host,
                stager,
                task_id,
                buffer_ref,
                &resource,
                gva,
                size,
                bind,
                BufferBail::ColdPacked,
            )
        }
        Some(crate::runtime::bound_buffers::PackedBufferResolution::Unavailable {
            gva: bg,
            size: bs,
        }) => {
            if *bg != gva || *bs != size {
                return Err(BufferBail::ColdPacked);
            }
        }
        Some(crate::runtime::bound_buffers::PackedBufferResolution::Available(_)) => {
            let serves_bind = state
                .bound_buffers
                .packed_available(task_id, buffer_ref, gva, size)
                .is_some_and(|p| bind.is_none_or(|(offset, _)| offset < p.size));
            if serves_bind {
                let p = state
                    .bound_buffers
                    .packed_available(task_id, buffer_ref, gva, size)
                    .expect("the Available arm matched above");
                return Ok(BufferContentPlan {
                    task_id,
                    buffer_ref,
                    resource,
                    gva,
                    size,
                    generation,
                    exact: None,
                    rail: BufferRail::Packed {
                        runs: std::sync::Arc::clone(&p.runs),
                        pages: std::sync::Arc::clone(&p.pages),
                    },
                });
            }
            // An offset past the alias falls past this rail on the serial
            // ladder too; the exact arms below decide it.
        }
    }
    // The staged arm below answers from the bind's own window, so it needs
    // the offset exactly as the serial CPU read would take it.
    let Some((offset, cap)) = bind else {
        // A cap-free plan covers the whole-allocation rail, warm only: a cold
        // or mismatched entry is the drain's walk and insert.
        if lab_fix_whole_runs() {
            match state.bound_buffers.whole(task_id, buffer_ref) {
                Some(w) if w.gva == gva && w.size == size => {
                    return Ok(BufferContentPlan {
                        task_id,
                        buffer_ref,
                        resource,
                        gva,
                        size,
                        generation,
                        exact: None,
                        rail: BufferRail::Whole {
                            runs: std::sync::Arc::clone(&w.runs),
                        },
                    });
                }
                _ => return Err(BufferBail::ColdWhole),
            }
        }
        return Err(BufferBail::ColdWindow);
    };
    if offset >= size {
        return Err(BufferBail::OffsetPastEnd);
    }
    let full = size - offset;
    let gather_eligible = full >= ZERO_COPY_BUFFER_MIN_BYTES;
    // The whole gate, in the serial ladder's own order: the arm runs only when
    // the knob and the floor admit it, and a warm matching entry serves every
    // eligible offset. A cold entry is the drain's walk (a mutation); a warm
    // entry that does not serve this bind (sub-floor remainder) leaves the
    // ladder to the exact arms below, exactly as the serial walk falls through.
    if lab_fix_whole_runs() {
        match state.bound_buffers.whole(task_id, buffer_ref) {
            Some(w) if w.gva == gva && w.size == size => {
                if gather_eligible && host_alloc_len(full).is_some_and(|n| n > 0) {
                    return Ok(BufferContentPlan {
                        task_id,
                        buffer_ref,
                        resource,
                        gva,
                        size,
                        generation,
                        exact: None,
                        rail: BufferRail::Whole {
                            runs: std::sync::Arc::clone(&w.runs),
                        },
                    });
                }
            }
            Some(_) => return Err(BufferBail::ColdWhole),
            None if gather_eligible => {
                // `ensure_whole_runs` would walk and insert — the drain's.
                return Err(BufferBail::ColdWhole);
            }
            None => {
                // Sub-floor remainder: the serial ladder skips the whole arm
                // (`gather_eligible` gates it) and so does the plan.
            }
        }
    }
    // The held exact-window entry.
    if let Some(bound) = state
        .bound_buffers
        .get(task_id, buffer_ref, offset, cap)
        .cloned()
    {
        return Ok(BufferContentPlan {
            task_id,
            buffer_ref,
            resource,
            gva,
            size,
            generation,
            exact: Some((offset, cap)),
            rail: BufferRail::Held { bound },
        });
    }
    // The exact-window walk, read-only. `task_gva_guest_run_window` and
    // `guest_page_window` note their own helper routes wherever they run; the
    // decision routes the serial arm notes ride in the plan and are noted at
    // commit (H8's rule for anything the plan decides).
    let span = cap.map_or(full, |c| full.min(c));
    let narrowed = (span < full).then(|| full - span);
    let (gpas, runs) = match task_gva_guest_run_window(state, host, task_id, gva + offset, span) {
        Ok(w) => w,
        Err(_) => {
            // The walk refused read-only; on the sub-floor population the
            // serial ladder was CPU-reading anyway.
            return stage_or_bail(
                state,
                host,
                stager,
                task_id,
                buffer_ref,
                &resource,
                gva,
                size,
                bind,
                BufferBail::ColdWindow,
            );
        }
    };
    let page = state.page_size();
    let pages = guest_page_window(host, &gpas, page, (gva + offset) % page, span);
    if !window_binds_zero_copy(pages.is_none(), gather_eligible) {
        // The sub-floor refusal the serial ladder takes to its CPU read —
        // and the one place that read is *provable*: the floor refuses
        // zero-copy outright (`gather_span_if_eligible` gates on it before
        // any construction), so the staged arm answers a bind the serial
        // ladder certainly reads on the CPU. Every cold exit above this one
        // leaves the plan bailing: the serial ladder may still CONSTRUCT a
        // rail there (the packed alias, the whole-runs walk), and a staged
        // window answering past a construction the commit cannot see warm is
        // the upper-rung hazard, not a saving.
        return stage_or_bail(
            state,
            host,
            stager,
            task_id,
            buffer_ref,
            &resource,
            gva,
            size,
            bind,
            BufferBail::BelowFloor,
        );
    }
    Ok(BufferContentPlan {
        task_id,
        buffer_ref,
        resource,
        gva,
        size,
        generation,
        exact: Some((offset, cap)),
        rail: BufferRail::Walked {
            bound: crate::runtime::bound_buffers::BoundBuffer {
                gva: gva + offset,
                span,
                source_offset: 0,
                runs: std::sync::Arc::new(runs),
                pages,
            },
            narrowed,
        },
    })
}

/// TEMP (lab, PASS_STAGE): the last rung of the planner — where every
/// zero-copy rail has refused and the serial ladder would fall to its CPU
/// read, stage that read's bytes into the worker's arena instead.
///
/// The plan predicts the serial answer exactly where it is built: the read
/// window (`offset..offset+span`, the cap applied as the serial read applies
/// it), the debt gates the serial read's settle would clear (any debt is a
/// bail — the payment is the drain's, H7), and the pages + host-write epoch
/// for the commit's race check. A refusal of any kind — no stager (the rail
/// off, `shadow`, or the arenas not yet created), a debt, an unreadable
/// window, an arena exhaustion — returns the bail the planner would have
/// returned, and the serial ladder serves the bind unchanged.
#[allow(clippy::too_many_arguments)]
fn stage_or_bail<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    stager: Option<&super::pass_stage::Stager>,
    task_id: u32,
    buffer_ref: u32,
    resource: &std::sync::Arc<crate::model::TaskResource>,
    gva: u64,
    size: u64,
    bind: Option<(u64, Option<u64>)>,
    bail: BufferBail,
) -> Result<BufferContentPlan, BufferBail> {
    let Some(stager) = stager else {
        return Err(bail);
    };
    let Some((offset, cap)) = bind else {
        // A bind with no offset the planner could name; the serial ladder
        // resolves it with the bind in hand.
        return Err(bail);
    };
    if offset >= size {
        return Err(BufferBail::OffsetPastEnd);
    }
    let full = size - offset;
    let span = cap.map_or(full, |c| full.min(c));
    // The honest population: a window the zero-copy floor already refused
    // (`gather_span_if_eligible` gates on the floor before any construction),
    // so the serial ladder's answer is provably its CPU read. A floor-clearing
    // window may still be constructed into a rail the commit cannot see warm —
    // that is the upper-rung hazard, and the bail keeps it the drain's.
    if full >= ZERO_COPY_BUFFER_MIN_BYTES {
        return Err(bail);
    }
    let want = crate::runtime::draw::host_alloc_len(span).filter(|n| *n > 0);
    let Some(want) = want else {
        return Err(bail);
    };
    // The plan never pays (H7): the serial CPU read settles the bind's debt
    // before it reads, so any debt standing is the drain's to clear.
    if crate::runtime::writeback_debt::owes_texture(state, task_id, buffer_ref)
        || state
            .pending_writebacks
            .owes_gva_overlapping(task_id, gva, size)
    {
        crate::runtime::drain::note_store_route("pass_stage_buf_bail_owed");
        return Err(BufferBail::OwedStage);
    }
    let read_gva = gva + offset;
    let gpas = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        read_gva,
        span,
        state.page_shift,
    );
    if gpas.is_empty() {
        return Err(bail);
    }
    let host_epoch = state.host_writes.epoch();
    let mut bytes = vec![0u8; want];
    if crate::runtime::gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        read_gva,
        &mut bytes,
        state.page_shift,
    )
    .is_err()
    {
        return Err(bail);
    }
    let Some(staged) = stager.stage_bytes(&bytes) else {
        crate::runtime::drain::note_store_route("pass_stage_buf_arena_full");
        return Err(bail);
    };
    crate::runtime::drain::note_store_route_n("pass_stage_buf_bytes", staged.window.len);
    Ok(BufferContentPlan {
        task_id,
        buffer_ref,
        resource: std::sync::Arc::clone(resource),
        gva,
        size,
        generation: state.bound_buffers.generation(),
        // The staged window answers exactly the bind it was read for; a cap
        // the plan did not know narrows nothing (the bytes are a superset the
        // commit accepts).
        exact: Some((offset, cap)),
        rail: BufferRail::Staged {
            staged,
            host_epoch,
            gpas: std::sync::Arc::new(gpas),
        },
    })
}

/// Re-check a plan at the point the serial ladder would resolve the bind, and
/// with `apply` take it: the routes the serial arm notes, and — for a walked
/// plan — the [`BoundBufferHold`] insert.
///
/// `Ok(None)` is not a failure: the plan's rail simply does not serve this
/// particular bind (an offset past the alias, a sub-floor remainder), the plan
/// stays in the frame for the binds it does serve, and the serial ladder runs.
fn commit_buffer_plan(
    state: &mut DeviceState,
    plan: &BufferContentPlan,
    task_id: u32,
    buffer_ref: u32,
    resource: Option<&crate::model::TaskResource>,
    offset: u64,
    extent_cap: Option<u64>,
    apply: bool,
) -> Result<Option<crate::backend::vulkan::engine::BufferContent>, BufferInvalid> {
    if plan.task_id != task_id || plan.buffer_ref != buffer_ref {
        return Err(BufferInvalid::Key);
    }
    if let Some((planned_offset, planned_cap)) = plan.exact {
        // TEMP (lab, PASS_STAGE): a staged window read cap-free is a superset
        // of every narrowed cap the reflection may state — the bytes the
        // serial read would have returned, plus the tail it would have cut.
        // The bind it answers is the same; only the bound length widens, so
        // the key admits it rather than spending the plan.
        let staged_superset =
            matches!(plan.rail, BufferRail::Staged { .. }) && planned_cap.is_none();
        let key_matches = (planned_offset, planned_cap) == (offset, extent_cap)
            || (staged_superset && planned_offset == offset);
        if !key_matches {
            return Err(BufferInvalid::Key);
        }
    }
    // The bind hands the deref'd resource (the buffer ladder's own shape), so
    // the identity check compares the allocations' addresses — `Arc::ptr_eq`
    // in the form a `&TaskResource` can take.
    if !resource.is_some_and(|r| std::ptr::eq(r, plan.resource.as_ref())) {
        return Err(BufferInvalid::Resource);
    }
    if state.bound_buffers.generation() != plan.generation {
        return Err(BufferInvalid::Registry);
    }
    let note = |route: &'static str| {
        if apply {
            crate::runtime::drain::note_store_route(route);
        }
    };
    match &plan.rail {
        BufferRail::Packed { runs, pages } => {
            let packed = state
                .bound_buffers
                .packed_available(task_id, buffer_ref, plan.gva, plan.size)
                .ok_or(BufferInvalid::Packed)?;
            if !(std::sync::Arc::ptr_eq(&packed.runs, runs)
                && std::sync::Arc::ptr_eq(&packed.pages, pages))
            {
                return Err(BufferInvalid::Packed);
            }
            if offset >= packed.size {
                // Not this bind's rail; the plan is unspent.
                return Ok(None);
            }
            let full = packed.size - offset;
            let span = extent_cap.map_or(full, |cap| full.min(cap));
            let bound = slice_packed_buffer(packed, offset, span).ok_or(BufferInvalid::Packed)?;
            note("zc_buffer_held");
            Ok(Some(bound_buffer_content(&bound)))
        }
        BufferRail::Whole { runs } => {
            // The rail above, repeated: packed would answer if it appeared.
            if packed_serves(state, task_id, buffer_ref, plan.gva, plan.size, offset) {
                return Err(BufferInvalid::UpperRung);
            }
            let whole = state
                .bound_buffers
                .whole(task_id, buffer_ref)
                .filter(|w| w.gva == plan.gva && w.size == plan.size)
                .ok_or(BufferInvalid::Whole)?;
            if !std::sync::Arc::ptr_eq(&whole.runs, runs) {
                return Err(BufferInvalid::Whole);
            }
            let Some(bound) = whole_runs_bind(state, task_id, buffer_ref, offset, extent_cap)
            else {
                return Ok(None);
            };
            note("zc_buffer_whole_held");
            Ok(Some(bound_buffer_content(&bound)))
        }
        BufferRail::Held { bound } => {
            if packed_serves(state, task_id, buffer_ref, plan.gva, plan.size, offset)
                || whole_serves(
                    state, task_id, buffer_ref, plan.gva, plan.size, offset, extent_cap,
                )
            {
                return Err(BufferInvalid::UpperRung);
            }
            let held = state
                .bound_buffers
                .get(task_id, buffer_ref, offset, extent_cap)
                .ok_or(BufferInvalid::Held)?;
            if !(std::sync::Arc::ptr_eq(&held.runs, &bound.runs)
                && held.span == bound.span
                && held.gva == bound.gva
                && held.source_offset == bound.source_offset)
            {
                return Err(BufferInvalid::Held);
            }
            note("zc_buffer_held");
            Ok(Some(bound_buffer_content(bound)))
        }
        BufferRail::Walked { bound, narrowed } => {
            if packed_serves(state, task_id, buffer_ref, plan.gva, plan.size, offset)
                || whole_serves(
                    state, task_id, buffer_ref, plan.gva, plan.size, offset, extent_cap,
                )
            {
                return Err(BufferInvalid::UpperRung);
            }
            // A serial walk may have landed an entry at this key while the
            // plan waited (an earlier bind of the same draw fell back); the
            // hold is idempotent only against its own bytes.
            if let Some(held) = state
                .bound_buffers
                .get(task_id, buffer_ref, offset, extent_cap)
            {
                if !std::sync::Arc::ptr_eq(&held.runs, &bound.runs) {
                    return Err(BufferInvalid::Held);
                }
                note("zc_buffer_held");
                return Ok(Some(bound_buffer_content(bound)));
            }
            if let Some(saved) = narrowed {
                note("zc_buffer_extent_narrowed");
                if apply {
                    crate::runtime::drain::note_store_route_n(
                        "zc_buffer_extent_saved_bytes",
                        *saved,
                    );
                }
            }
            note(if bound.pages.is_some() {
                "zc_buffer_imported"
            } else {
                "zc_buffer_gathered"
            });
            if apply {
                BoundBufferHold {
                    task: task_id,
                    buffer_ref,
                    offset,
                    cap: extent_cap,
                    bound,
                }
                .land(state);
            }
            Ok(Some(bound_buffer_content(bound)))
        }
        // TEMP (lab, PASS_STAGE): the staged window serves the bind it was
        // read for, provided nothing above it would answer now and no device
        // write reached its pages between the worker's read and this commit.
        BufferRail::Staged {
            staged,
            host_epoch,
            gpas,
        } => {
            if packed_serves(state, task_id, buffer_ref, plan.gva, plan.size, offset)
                || whole_serves(
                    state, task_id, buffer_ref, plan.gva, plan.size, offset, extent_cap,
                )
                || state
                    .bound_buffers
                    .get(task_id, buffer_ref, offset, extent_cap)
                    .is_some()
            {
                return Err(BufferInvalid::UpperRung);
            }
            if state.host_writes.wrote_any_since(*host_epoch, gpas).wrote() {
                crate::runtime::drain::note_store_route("pass_stage_buf_invalid_host_write");
                return Err(BufferInvalid::UpperRung);
            }
            note("cpu_buffer_staged");
            Ok(Some(crate::backend::vulkan::engine::BufferContent::Staged(
                staged.clone(),
            )))
        }
    }
}

/// The colour-0 identity a Store page walk rests on: the fields of
/// [`ColorRtRequest`] the walk reads, which a pass fixes once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StoreTargetKey {
    gva: u64,
    row_stride: u32,
    height: u32,
    store_action: u16,
}

impl StoreTargetKey {
    /// The key of `c0` where the walk would run at all — `None` where the
    /// serial walk refuses before walking (`sync_store_target_pages`'s own
    /// gates), so nothing is planned for it.
    fn of(c0: &ColorRtRequest) -> Option<Self> {
        (c0.target_gva != 0
            && reims_vgpu_protocol::pass_action::store_action_publishes_single_sample(
                c0.store_action,
            )
            && c0.width != 0
            && c0.height != 0)
            .then_some(Self {
                gva: c0.target_gva,
                row_stride: c0.row_stride,
                height: c0.height,
                store_action: c0.store_action,
            })
    }
}

/// The last draw's Store page walk, planned off the drain: the guest pages the
/// eager GVA fallback may write, resolved before the draw submits.
///
/// The walk is the read-only kind the pool can do — a page-table walk through
/// the worker host, no `map_pages`, no registry insertion — which is why it
/// needs no bail of its own: the plan exists whenever the walk resolves pages.
/// The commit re-checks the key against the live colour-0 and the registry
/// generation, and notes `sync_store_bound` where the serial walk notes it.
pub(crate) struct StorePagesPlan {
    task_id: u32,
    key: StoreTargetKey,
    generation: u64,
    pages: StoreTargetPages,
}

/// Plan the Store page walk for `key` from state this call only reads.
pub(super) fn plan_store_pages<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    key: StoreTargetKey,
) -> Option<StorePagesPlan> {
    let span = u64::from(key.row_stride).checked_mul(u64::from(key.height))?;
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        key.gva,
        span,
        state.page_shift,
    );
    if ordered.is_empty() {
        // The serial walk will name it `sync_store_unbounded`; a plan that
        // resolved nothing is no plan.
        return None;
    }
    Some(StorePagesPlan {
        task_id,
        key,
        generation: state.bound_buffers.generation(),
        pages: StoreTargetPages::from_ordered(&ordered, span),
    })
}

/// Re-check the Store plan at the point the serial walk stands.
fn commit_store_pages(
    state: &DeviceState,
    plan: &StorePagesPlan,
    task_id: u32,
    c0: &ColorRtRequest,
    apply: bool,
) -> Result<StoreTargetPages, BufferInvalid> {
    if plan.task_id != task_id || StoreTargetKey::of(c0) != Some(plan.key) {
        return Err(BufferInvalid::Key);
    }
    if state.bound_buffers.generation() != plan.generation {
        return Err(BufferInvalid::Registry);
    }
    if apply {
        crate::runtime::drain::note_store_route("sync_store_bound");
        crate::runtime::drain::note_store_route("pass_prep_buf_store_hit");
    }
    Ok(plan.pages.clone())
}

/// What a plan ask offers one bind.
pub(crate) enum Ask<T> {
    /// Serve the bind from the validated plan (mode `on`); routes are noted.
    Serve(T),
    /// The plan validated; its answer is held for the shadow compare the
    /// serial ladder's answer completes. Nothing is noted or applied.
    Shadow,
    /// No plan, or one that no longer holds: run the serial ladder.
    Serial,
}

// The shadow prediction one bind holds while its serial ladder runs. (A plain
// comment: rustdoc does not document macro invocations.)
thread_local! {
    static SHADOW: std::cell::RefCell<Option<ShadowHold>> =
        const { std::cell::RefCell::new(None) };
}

enum ShadowHold {
    Buffer {
        buffer_ref: u32,
        offset: u64,
        predicted: crate::backend::vulkan::engine::BufferContent,
    },
    Store {
        predicted: StoreTargetPages,
    },
}

/// The mode the buffer plans run under — the PassPrep one, as
/// [`prefetch::prefetched_source`] resolves it.
fn mode() -> super::pass_prep::PassPrepMode {
    super::pass_prep::pass_prep_mode()
}

/// Ask the pass frame for a plan for one buffer bind, at the point the serial
/// ladder would resolve it. See [`Ask`].
pub(super) fn ask_buffer(
    state: &mut DeviceState,
    task_id: u32,
    buffer_ref: u32,
    resource: Option<&crate::model::TaskResource>,
    offset: u64,
    extent_cap: Option<u64>,
) -> Ask<crate::backend::vulkan::engine::BufferContent> {
    let m = mode();
    if m == super::pass_prep::PassPrepMode::Off {
        return Ask::Serial;
    }
    prefetch::with_last_frame(|frame| {
        ask_frame_exact_or_rail(
            frame, state, task_id, buffer_ref, resource, offset, extent_cap, m,
        )
    })
    .unwrap_or(Ask::Serial)
}

/// [`ask_buffer`] against a borrowed frame: the exact plan for this very bind
/// first, then the reference's rail plan.
fn ask_frame_exact_or_rail(
    frame: &mut PassPlans,
    state: &mut DeviceState,
    task_id: u32,
    buffer_ref: u32,
    resource: Option<&crate::model::TaskResource>,
    offset: u64,
    extent_cap: Option<u64>,
    m: super::pass_prep::PassPrepMode,
) -> Ask<crate::backend::vulkan::engine::BufferContent> {
    let apply = m == super::pass_prep::PassPrepMode::On;
    // The exact plan for this very bind; a miss is not spent, just absent.
    // TEMP (lab, PASS_STAGE): a staged plan read cap-free is keyed under
    // `None` — the worker could not know the reflection's cap — and answers
    // the narrowed bind from its superset window, so the miss with a cap
    // retries the cap-free key before falling to the rails.
    let exact_key = (task_id, buffer_ref, offset, extent_cap);
    let exact_key = if frame.buffer_exacts.contains_key(&exact_key) {
        exact_key
    } else if extent_cap.is_some()
        && frame
            .buffer_exacts
            .get(&(task_id, buffer_ref, offset, None))
            .is_some_and(|plan| matches!(plan.rail, BufferRail::Staged { .. }))
    {
        (task_id, buffer_ref, offset, None)
    } else {
        exact_key
    };
    if frame.buffer_exacts.contains_key(&exact_key) {
        let (slug, outcome) = {
            let plan = frame.buffer_exacts.get(&exact_key).expect("just checked");
            (
                plan.rail_slug(),
                commit_buffer_plan(
                    state, plan, task_id, buffer_ref, resource, offset, extent_cap, apply,
                ),
            )
        };
        return match outcome {
            Ok(Some(content)) => finish_ask(content, apply, slug, buffer_ref, offset),
            Ok(None) => Ask::Serial,
            Err(invalid) => {
                note_invalid(invalid);
                frame.buffer_exacts.remove(&exact_key);
                Ask::Serial
            }
        };
    }
    // The reference's rail plan, answering this bind if its offset fits.
    if let Some(slug) = frame
        .buffer_rails
        .get(&(task_id, buffer_ref))
        .map(|p| p.rail_slug())
    {
        let outcome = {
            let plan = frame
                .buffer_rails
                .get(&(task_id, buffer_ref))
                .expect("the slug came from it");
            commit_buffer_plan(
                state, plan, task_id, buffer_ref, resource, offset, extent_cap, apply,
            )
        };
        return match outcome {
            Ok(Some(content)) => finish_ask(content, apply, slug, buffer_ref, offset),
            Ok(None) => Ask::Serial,
            Err(invalid) => {
                note_invalid(invalid);
                frame.buffer_rails.remove(&(task_id, buffer_ref));
                Ask::Serial
            }
        };
    }
    Ask::Serial
}

/// Serve or hold one validated answer, noting the hit where it serves.
fn finish_ask(
    content: crate::backend::vulkan::engine::BufferContent,
    apply: bool,
    slug: &'static str,
    buffer_ref: u32,
    offset: u64,
) -> Ask<crate::backend::vulkan::engine::BufferContent> {
    if apply {
        crate::runtime::drain::note_store_route("pass_prep_buf_hit");
        crate::runtime::drain::note_store_route(slug);
        Ask::Serve(content)
    } else {
        SHADOW.with(|slot| {
            *slot.borrow_mut() = Some(ShadowHold::Buffer {
                buffer_ref,
                offset,
                predicted: content,
            });
        });
        Ask::Shadow
    }
}

fn note_invalid(invalid: BufferInvalid) {
    crate::runtime::drain::note_store_route(match invalid {
        BufferInvalid::Key => "pass_prep_buf_invalid_key",
        BufferInvalid::Resource => "pass_prep_buf_invalid_resource",
        BufferInvalid::Registry => "pass_prep_buf_invalid_registry",
        BufferInvalid::Packed => "pass_prep_buf_invalid_packed",
        BufferInvalid::Whole => "pass_prep_buf_invalid_whole",
        BufferInvalid::Held => "pass_prep_buf_invalid_held",
        BufferInvalid::UpperRung => "pass_prep_buf_invalid_upper_rung",
    });
}

/// Ask the frame for the index bind's plan — the exact key the record names,
/// with the resource re-resolved the way the serial ladder resolves it.
pub(super) fn ask_index<M: HostMemory>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    info: &IndexedDrawInfo,
) -> Ask<crate::backend::vulkan::engine::BufferContent> {
    let m = mode();
    if m == super::pass_prep::PassPrepMode::Off {
        return Ask::Serial;
    }
    // The exact plan is keyed by the need the record states, so the ask
    // re-derives the key (the same arithmetic `resolve_index_window_reason`
    // applies) rather than trusting the plan's own claim.
    let Some(key) = index_key(task_id, info) else {
        return Ask::Serial;
    };
    let need = key.3;
    let resource = objects::resolve_resource(state, host, task_id, info.index_buffer_ref).ok();
    prefetch::with_last_frame(|frame| {
        if !frame.buffer_exacts.contains_key(&key) {
            return Ask::Serial;
        }
        let (slug, outcome) = {
            let plan = frame.buffer_exacts.get(&key).expect("just checked");
            (
                plan.rail_slug(),
                commit_buffer_plan(
                    state,
                    plan,
                    task_id,
                    info.index_buffer_ref,
                    resource.as_deref(),
                    info.index_buffer_offset,
                    need,
                    m == super::pass_prep::PassPrepMode::On,
                ),
            )
        };
        match outcome {
            Ok(Some(content)) => finish_ask(
                content,
                m == super::pass_prep::PassPrepMode::On,
                slug,
                info.index_buffer_ref,
                info.index_buffer_offset,
            ),
            Ok(None) => Ask::Serial,
            Err(invalid) => {
                note_invalid(invalid);
                frame.buffer_exacts.remove(&key);
                Ask::Serial
            }
        }
    })
    .unwrap_or(Ask::Serial)
}

/// Ask the frame for the pass's Store page walk, at the point the serial walk
/// stands (the storing draw only — the caller gates `writeback_guest`).
pub(super) fn ask_store_pages(
    state: &DeviceState,
    task_id: u32,
    color0: Option<&ColorRtRequest>,
) -> Ask<StoreTargetPages> {
    let m = mode();
    if m == super::pass_prep::PassPrepMode::Off {
        return Ask::Serial;
    }
    let Some(c0) = color0 else {
        return Ask::Serial;
    };
    prefetch::with_last_frame(|frame| {
        let Some(plan) = frame.store_pages.take() else {
            return Ask::Serial;
        };
        match commit_store_pages(
            state,
            &plan,
            task_id,
            c0,
            m == super::pass_prep::PassPrepMode::On,
        ) {
            Ok(pages) => {
                if m == super::pass_prep::PassPrepMode::On {
                    Ask::Serve(pages)
                } else {
                    SHADOW.with(|slot| {
                        *slot.borrow_mut() = Some(ShadowHold::Store {
                            predicted: pages.clone(),
                        });
                    });
                    Ask::Shadow
                }
            }
            Err(invalid) => {
                note_invalid(invalid);
                Ask::Serial
            }
        }
    })
    .unwrap_or(Ask::Serial)
}

/// Compare a buffer plan's prediction against the serial ladder's answer for
/// the same bind — the `shadow` gate's honesty check for step 3.
pub(super) fn shadow_compare_buffer(
    buffer_ref: u32,
    offset: u64,
    asked_was_shadow: bool,
    serial: Option<&crate::backend::vulkan::engine::BufferContent>,
) {
    if !asked_was_shadow {
        return;
    }
    // Pair the slot: a stale prediction (a nested ask without its compare)
    // is dropped rather than compared against the wrong bind.
    let held = SHADOW.with(|slot| slot.borrow_mut().take());
    let Some(ShadowHold::Buffer {
        buffer_ref: held_ref,
        offset: held_offset,
        predicted,
    }) = held
    else {
        return;
    };
    if (held_ref, held_offset) != (buffer_ref, offset) {
        return;
    }
    let verdict = match serial {
        None => "par_shadow_buf_mismatch_serial_none",
        Some(s) => match (&predicted, s) {
            (
                crate::backend::vulkan::engine::BufferContent::GuestRuns(a),
                crate::backend::vulkan::engine::BufferContent::GuestRuns(b),
            ) => {
                let same_runs =
                    std::sync::Arc::ptr_eq(&a.runs, &b.runs) || a.runs[..] == b.runs[..];
                if same_runs
                    && a.source_offset == b.source_offset
                    && a.total_len == b.total_len
                    && a.row_length_texels == b.row_length_texels
                    && a.pages.is_some() == b.pages.is_some()
                {
                    "par_shadow_buf_match_runs"
                } else {
                    "par_shadow_buf_mismatch_runs"
                }
            }
            (
                crate::backend::vulkan::engine::BufferContent::Bytes(a),
                crate::backend::vulkan::engine::BufferContent::Bytes(b),
            ) => {
                if std::sync::Arc::ptr_eq(a, b) || a[..] == b[..] {
                    "par_shadow_buf_match_bytes"
                } else {
                    "par_shadow_buf_mismatch_bytes"
                }
            }
            _ => "par_shadow_buf_mismatch_variant",
        },
    };
    crate::runtime::drain::note_store_route(verdict);
    if verdict.contains("mismatch")
        && crate::observe::first_sight("par_shadow_buf_mismatch", u64::from(buffer_ref))
    {
        crate::observe::fail(format!(
            "par_shadow_buf_mismatch ref={buffer_ref} off={offset:#x} kind={verdict} \
             planned={} serial={}",
            content_variant(&predicted),
            serial.map(content_variant).unwrap_or("none"),
        ));
    }
}

fn content_variant(c: &crate::backend::vulkan::engine::BufferContent) -> &'static str {
    match c {
        crate::backend::vulkan::engine::BufferContent::GuestRuns(_) => "runs",
        crate::backend::vulkan::engine::BufferContent::Bytes(_) => "bytes",
        // TEMP (lab, PASS_STAGE): the staged rail the plan serves.
        crate::backend::vulkan::engine::BufferContent::Staged(_) => "staged",
    }
}

/// Compare the Store page walk's prediction against the serial walk.
pub(super) fn shadow_compare_store(asked_was_shadow: bool, serial: Option<&StoreTargetPages>) {
    if !asked_was_shadow {
        return;
    }
    let held = SHADOW.with(|slot| slot.borrow_mut().take());
    let Some(ShadowHold::Store { predicted }) = held else {
        return;
    };
    let verdict = match serial {
        None => "par_shadow_buf_mismatch_store_none",
        Some(s) => {
            if s.ordered == predicted.ordered && s.span == predicted.span {
                "par_shadow_buf_store_match"
            } else {
                "par_shadow_buf_mismatch_store"
            }
        }
    };
    crate::runtime::drain::note_store_route(verdict);
    if verdict.contains("mismatch") {
        crate::observe::fail(String::from(
            "par_shadow_buf_mismatch_store (planned vs serial Store pages)",
        ));
    }
}

/// Merge a chunk's buffer plans into the pass frame: rail plans under the
/// reference, exact plans under the bind. A same-key entry replaces, which is
/// how a replanned bind supersedes a spent one.
pub(super) fn merge_buffer_plans(
    rails: impl IntoIterator<Item = ((u32, u32), BufferContentPlan)>,
    exacts: impl IntoIterator<Item = (BufferExactKey, BufferContentPlan)>,
    store: Option<StorePagesPlan>,
) {
    prefetch::with_last_frame(|frame| {
        frame.buffer_rails.extend(rails);
        frame.buffer_exacts.extend(exacts);
        if store.is_some() {
            frame.store_pages = store;
        }
    });
}

/// The bail census route for a planner refusal.
pub(super) fn bail_route(bail: BufferBail) -> &'static str {
    match bail {
        BufferBail::NullRef => "pass_prep_buf_bail_null_ref",
        BufferBail::ColdBacking => "pass_prep_buf_bail_cold_backing",
        BufferBail::ColdPacked => "pass_prep_buf_bail_cold_packed",
        BufferBail::ColdWhole => "pass_prep_buf_bail_cold_whole",
        BufferBail::ColdWindow => "pass_prep_buf_bail_cold_window",
        BufferBail::BelowFloor => "pass_prep_buf_bail_below_floor",
        BufferBail::OffsetPastEnd => "pass_prep_buf_bail_offset_past_end",
        BufferBail::OwedStage => "pass_prep_buf_bail_owed_stage",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::drain::census::store_route_count;
    use crate::runtime::host::FakeHost;

    /// One `(buffer_ref, allocation_size)` pair the fixture builds a buffer
    /// object for; the allocation sits at GVA `0x4000` (task page 4).
    fn buffer_fixture(specs: &[(u32, u64)]) -> (DeviceState, FakeHost) {
        use crate::protocol::endian::st32;
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER,
        };
        use crate::runtime::gva_mem;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.stable_map_pages = true;
        let (dir_pfn, root_pfn) = (2u32, 3u32);
        let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
        let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        let mut dir = [0u8; 8];
        st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
        st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
        assert!(host.write_gpa(dir_gpa, &dir).is_ok());
        // Task GVA pages 0..36 map one-to-one onto GPA pages 16..52. The
        // guest RAM behind them is ONE contiguous range, so a window's
        // `map_pages` answers the same stable alias whichever arm walks it —
        // separate per-page ranges would hand each walk its own bounce view.
        host.map_range(0x10000, 0x24000, 0x40);
        for i in 0..36u32 {
            let pfn = 16 + i;
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
        }
        state.define_task(1, 0x1000, dir_pfn);
        assert!(state.set_object_list(1, 0, 64));
        for (n, &(buffer_ref, size)) in specs.iter().enumerate() {
            let handle = 4u32 + n as u32;
            let mut bdesc = [0u8; 16];
            bdesc[0..8].copy_from_slice(&size.to_le_bytes());
            st32(&mut bdesc[8..], handle);
            let desc_gva = 0x200u64 + 0x40 * n as u64;
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
                desc_gva,
                &bdesc,
                PAGE_SHIFT_X86
            )
            .is_ok());
            let list_off = list_object_entry_offset(buffer_ref, 64).unwrap();
            let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
            st32(&mut list_entry, packed);
            list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
                list_off,
                &list_entry,
                PAGE_SHIFT_X86
            )
            .is_ok());
        }
        (state, host)
    }

    /// The retained resource of one fixture buffer, resolved the way a draw's
    /// bind table retains it.
    fn buffer_resource(
        state: &DeviceState,
        host: &FakeHost,
        buffer_ref: u32,
    ) -> std::sync::Arc<crate::model::TaskResource> {
        objects::resolve_resource(state, host, 1, buffer_ref).expect("fixture buffer resource")
    }

    /// Two `BufferContent`s agree: guest runs structurally (the arms walk
    /// separately, so host-pointer identity is not expected — lengths, source
    /// offsets and the pages' presence are the answer's shape), bytes by
    /// identity or value.
    fn contents_agree(
        a: &crate::backend::vulkan::engine::BufferContent,
        b: &crate::backend::vulkan::engine::BufferContent,
    ) -> bool {
        use crate::backend::vulkan::engine::BufferContent;
        match (a, b) {
            (BufferContent::GuestRuns(a), BufferContent::GuestRuns(b)) => {
                let lens = |src: &crate::backend::vulkan::engine::GuestRunSource| {
                    src.runs.iter().map(|r| r.len()).collect::<Vec<_>>()
                };
                lens(a) == lens(b)
                    && a.source_offset == b.source_offset
                    && a.total_len == b.total_len
                    && a.pages.is_some() == b.pages.is_some()
            }
            (BufferContent::Bytes(a), BufferContent::Bytes(b)) => {
                std::sync::Arc::ptr_eq(a, b) || a[..] == b[..]
            }
            _ => false,
        }
    }

    /// The registry's end state, compared arm against arm: the shape counts
    /// the held entries, their `(task, reference)` pairs, and how those pairs
    /// spread over offsets. A planned arm that applied a hold the serial arm
    /// also walked leaves the same shape.
    fn registry_shape(state: &DeviceState) -> crate::runtime::bound_buffers::RegistryShape {
        state.bound_buffers.shape()
    }

    fn delta(route: &str, before: u64) -> u64 {
        store_route_count(route).saturating_sub(before)
    }

    /// A retained whole-allocation rail plans, commits to the serial ladder's
    /// answer, and leaves the registry where the serial ladder leaves it.
    ///
    /// The fixture host has no Vulkan import caps, so the packed construction
    /// refuses (a held negative) and the serial prime takes the whole walk —
    /// exactly the shape the planner's rails cover. The packed-alias rail is
    /// the same contract one boot later, where the caps exist.
    #[test]
    fn a_whole_rail_plan_commits_to_the_serial_answer() {
        let (mut planned_state, mut planned_host) = buffer_fixture(&[(7, 0x8000)]);
        let (mut serial_state, mut serial_host) = buffer_fixture(&[(7, 0x8000)]);
        let resource = buffer_resource(&planned_state, &planned_host, 7);
        // One serial bind on each arm walks and retains the whole runs.
        for (state, host) in [
            (&mut planned_state, &mut planned_host),
            (&mut serial_state, &mut serial_host),
        ] {
            let res = buffer_resource(state, host, 7);
            let _prime = load_buffer_content_serial(state, host, 1, 7, Some(&res), 0, true, None)
                .expect("the serial ladder retains the whole runs");
        }
        let plan = plan_buffer_rails(
            &planned_state,
            &mut planned_host,
            1,
            7,
            resource.clone(),
            0,
            None,
        )
        .expect("a retained whole rail plans");
        assert!(
            matches!(plan.rail, BufferRail::Whole { .. }),
            "the primed whole runs answer the rail"
        );
        let _mode = super::super::pass_prep::test_mode(super::super::pass_prep::PassPrepMode::On);
        let _frame = prefetch::push_empty_frame();
        merge_buffer_plans([((1, 7), plan)], [], None);
        let before_hit = store_route_count("pass_prep_buf_hit_whole");
        let served = ask_buffer(
            &mut planned_state,
            1,
            7,
            Some(resource.as_ref()),
            0x100,
            Some(0x400),
        );
        let Ask::Serve(content) = served else {
            panic!("a validated whole plan serves its bind");
        };
        assert_eq!(delta("pass_prep_buf_hit_whole", before_hit), 1);
        let serial_res = buffer_resource(&serial_state, &serial_host, 7);
        let serial = load_buffer_content_serial(
            &mut serial_state,
            &mut serial_host,
            1,
            7,
            Some(serial_res.as_ref()),
            0x100,
            true,
            Some(0x400),
        )
        .expect("the serial ladder serves the same bind");
        assert!(
            contents_agree(&content, &serial),
            "the planned arm serves what the serial ladder serves"
        );
        assert_eq!(
            registry_shape(&planned_state),
            registry_shape(&serial_state),
            "the whole rail moves no registry state"
        );
    }

    /// A retirement between the plan and the commit drops the plan (the
    /// registry's generation moved) and the serial ladder serves the bind.
    #[test]
    fn a_retirement_between_plan_and_commit_falls_back() {
        let (mut state, mut host) = buffer_fixture(&[(7, 0x8000)]);
        let resource = buffer_resource(&state, &host, 7);
        let _prime =
            load_buffer_content_serial(&mut state, &mut host, 1, 7, Some(&resource), 0, true, None)
                .expect("prime");
        let plan =
            plan_buffer_rails(&state, &mut host, 1, 7, resource.clone(), 0, None).expect("plans");
        let _mode = super::super::pass_prep::test_mode(super::super::pass_prep::PassPrepMode::On);
        let _frame = prefetch::push_empty_frame();
        merge_buffer_plans([((1, 7), plan)], [], None);
        // The guest unmaps a page overlapping the allocation: the retirement
        // rule the map notify runs, and the generation with it.
        assert!(state.retire_bound_buffers_in_range(1, 0x4000, 0x1000) >= 0);
        let before = store_route_count("pass_prep_buf_invalid_registry");
        let served = ask_buffer(&mut state, 1, 7, Some(resource.as_ref()), 0, None);
        assert!(matches!(served, Ask::Serial));
        assert_eq!(delta("pass_prep_buf_invalid_registry", before), 1);
        // And the fallback serves: the serial ladder answers for the bind.
        let serial =
            load_buffer_content(&mut state, &mut host, 1, 7, Some(&resource), 0, true, None);
        assert!(serial.is_some(), "the serial ladder still serves the bind");
    }

    /// Cold constructions are bails, never plans: no packed entry (the alias
    /// would be constructed), and a matching Unavailable one with no whole
    /// entry (the whole walk would run) both hand the bind to the drain.
    #[test]
    fn cold_packed_and_whole_bail_to_the_serial_ladder() {
        let (mut state, mut host) = buffer_fixture(&[(7, 0x8000), (8, 0x8000)]);
        let res7 = buffer_resource(&state, &host, 7);
        let res8 = buffer_resource(&state, &host, 8);
        // Nothing retained: every construction is the drain's.
        assert_eq!(
            plan_buffer_rails(&state, &mut host, 1, 7, res7.clone(), 0, None).err(),
            Some(BufferBail::ColdPacked)
        );
        // A matching Unavailable answer stands (the construction refused once);
        // with no whole entry and a gather-eligible remainder, the whole walk
        // is the drain's too.
        state.bound_buffers.insert_packed(
            1,
            8,
            crate::runtime::bound_buffers::PackedBufferResolution::Unavailable {
                gva: 0x5000,
                size: 0x8000,
            },
        );
        assert_eq!(
            plan_buffer_rails(&state, &mut host, 1, 8, res8, 0, None).err(),
            Some(BufferBail::ColdWhole)
        );
        // And the serial ladder serves the cold bind unchanged, constructing
        // on the drain thread exactly as it did before this module existed.
        let serial = load_buffer_content(&mut state, &mut host, 1, 7, Some(&res7), 0, true, None);
        assert!(serial.is_some());
    }

    /// One plan serves every bind of its reference in the pass: the second
    /// bind validates and serves again, moving no registry state.
    #[test]
    fn a_repeated_bind_serves_the_plan_twice() {
        let (mut state, mut host) = buffer_fixture(&[(7, 0x8000)]);
        let resource = buffer_resource(&state, &host, 7);
        let _prime =
            load_buffer_content_serial(&mut state, &mut host, 1, 7, Some(&resource), 0, true, None)
                .expect("prime");
        let plan =
            plan_buffer_rails(&state, &mut host, 1, 7, resource.clone(), 0, None).expect("plans");
        let _mode = super::super::pass_prep::test_mode(super::super::pass_prep::PassPrepMode::On);
        let _frame = prefetch::push_empty_frame();
        merge_buffer_plans([((1, 7), plan)], [], None);
        let shape_before = registry_shape(&state);
        let before_hit = store_route_count("pass_prep_buf_hit");
        let first = ask_buffer(&mut state, 1, 7, Some(resource.as_ref()), 0x40, None);
        let second = ask_buffer(
            &mut state,
            1,
            7,
            Some(resource.as_ref()),
            0x80,
            Some(0x1000),
        );
        let (Ask::Serve(a), Ask::Serve(b)) = (first, second) else {
            panic!("both binds serve from the one rail plan");
        };
        // Different offsets are different binds; what repeats is the rail's
        // shape: guest runs out of the one retained alias, moving nothing.
        assert!(matches!(
            (&a, &b),
            (
                crate::backend::vulkan::engine::BufferContent::GuestRuns(_),
                crate::backend::vulkan::engine::BufferContent::GuestRuns(_),
            )
        ));
        assert_eq!(delta("pass_prep_buf_hit", before_hit), 2);
        assert_eq!(registry_shape(&state), shape_before);
    }

    /// The index bind's plan: the record's exact key holds a rail plan (the
    /// whole runs here — no import caps in a unit test, so the packed alias
    /// never constructs), committed at the point the serial loader resolves,
    /// equal to the serial ladder's answer and registry state.
    ///
    /// The walked window — the arm whose commit lands the `BoundBufferHold` —
    /// needs the exact window to be the arm the serial ladder takes, which on
    /// this host it never is (no imports, so a bindable window is
    /// gather-eligible and the whole rail already owns it); it stays a
    /// pool-only shape, reached where the run memo answers a worker's walk.
    #[test]
    fn an_index_plan_commits_to_the_serial_answer() {
        let (mut planned_state, mut planned_host) = buffer_fixture(&[(7, 0x8000)]);
        let (mut serial_state, mut serial_host) = buffer_fixture(&[(7, 0x8000)]);
        let info = |offset: u64| IndexedDrawInfo {
            index_type: 0,
            index_count: 0x80,
            index_buffer_ref: 7,
            index_buffer_offset: offset,
            base_vertex: 0,
        };
        // One serial index load per arm retains the whole runs.
        for (state, host) in [
            (&mut planned_state, &mut planned_host),
            (&mut serial_state, &mut serial_host),
        ] {
            let prime = info(0);
            let _ = load_index_content_reason_serial(state, host, 1, &prime)
                .expect("the prime retains the whole runs");
        }
        let planned_info = info(0x100);
        let plan = plan_index_content(&planned_state, &mut planned_host, 1, &planned_info, None)
            .expect("the index bind plans");
        assert!(
            matches!(plan.rail, BufferRail::Whole { .. }),
            "the retained whole runs answer the index bind"
        );
        let _mode = super::super::pass_prep::test_mode(super::super::pass_prep::PassPrepMode::On);
        let _frame = prefetch::push_empty_frame();
        merge_buffer_plans(
            [],
            [(index_key(1, &planned_info).expect("the key derives"), plan)],
            None,
        );
        let shape_before = registry_shape(&planned_state);
        let served = ask_index(&mut planned_state, &mut planned_host, 1, &planned_info);
        let Ask::Serve(content) = served else {
            panic!("the index plan serves");
        };
        let serial =
            load_index_content_reason_serial(&mut serial_state, &mut serial_host, 1, &planned_info)
                .expect("the serial ladder serves the same bind");
        assert!(contents_agree(&content, &serial));
        assert_eq!(
            registry_shape(&planned_state),
            shape_before,
            "the rail commit lands no hold"
        );
        assert_eq!(
            registry_shape(&planned_state),
            registry_shape(&serial_state),
            "both arms leave the registry where the other does"
        );
    }

    /// The Store page walk plans off the drain, commits against the live
    /// colour-0, and answers what the serial walk answers.
    #[test]
    fn a_store_pages_plan_answers_the_serial_walk() {
        let (state, host) = buffer_fixture(&[(7, 0x1000)]);
        let _resource = buffer_resource(&state, &host, 7);
        // A GVA colour target over the fixture's own task pages: 64x4 BGRA8
        // with a 256-byte row stride spans one page.
        let c0 = ColorRtRequest {
            target_gva: 0x8000,
            row_stride: 256,
            width: 64,
            height: 4,
            format: reims_vgpu_protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            store_action: reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_STORE,
            ..Default::default()
        };
        let key = latch_store_target(&c0).expect("the walk would run");
        let plan = plan_store_pages(&state, &host, 1, key).expect("the walk resolves pages");
        let _mode = super::super::pass_prep::test_mode(super::super::pass_prep::PassPrepMode::On);
        let _frame = prefetch::push_empty_frame();
        merge_buffer_plans([], [], Some(plan));
        let before = store_route_count("pass_prep_buf_store_hit");
        let served = ask_store_pages(&state, 1, Some(&c0));
        let Ask::Serve(pages) = served else {
            panic!("the store plan serves");
        };
        assert_eq!(delta("pass_prep_buf_store_hit", before), 1);
        let serial = sync_store_target_pages(&state, &host, 1, &c0)
            .expect("the serial walk resolves the same pages");
        assert!(pages.ordered == serial.ordered && pages.span == serial.span);
    }

    /// A rail plan serves the binds its rail covers and falls past the ones it
    /// does not: an offset at the end of the allocation is `Ok(None)` — the
    /// plan stays in the frame, unspent — and the serial ladder reads the bind
    /// on the CPU, exactly as it would without the plan.
    ///
    /// (The `UpperRung` invalidation of the exact arms needs a packed alias or
    /// a whole rail to *appear* above the plan between plan and commit; on a
    /// host with no import caps neither state is reachable from the serial
    /// ladder, so it stays a defensive validation the shadow gate watches for
    /// instead of a unit-tested one.)
    #[test]
    fn a_rail_plan_falls_past_a_past_end_offset() {
        let (mut state, mut host) = buffer_fixture(&[(7, 0x8000)]);
        let resource = buffer_resource(&state, &host, 7);
        let _prime =
            load_buffer_content_serial(&mut state, &mut host, 1, 7, Some(&resource), 0, true, None)
                .expect("prime");
        let plan =
            plan_buffer_rails(&state, &mut host, 1, 7, resource.clone(), 0, None).expect("plans");
        assert!(matches!(plan.rail, BufferRail::Whole { .. }));
        let _mode = super::super::pass_prep::test_mode(super::super::pass_prep::PassPrepMode::On);
        let _frame = prefetch::push_empty_frame();
        merge_buffer_plans([((1, 7), plan)], [], None);
        // A bind the rail covers still serves.
        let covered = ask_buffer(&mut state, 1, 7, Some(resource.as_ref()), 0x40, None);
        assert!(matches!(covered, Ask::Serve(_)));
        // A bind past the allocation's end is not the rail's to answer: no
        // invalidation, no hit, the serial ladder's own past-end refusal.
        let before_invalid = store_route_count("pass_prep_buf_invalid_key");
        let past_end = ask_buffer(&mut state, 1, 7, Some(resource.as_ref()), 0x8000, None);
        assert!(
            matches!(past_end, Ask::Serial),
            "a past-end offset is not the rail's bind"
        );
        assert_eq!(delta("pass_prep_buf_invalid_key", before_invalid), 0);
        // The plan survives: another covered bind still serves from it.
        let again = ask_buffer(&mut state, 1, 7, Some(resource.as_ref()), 0x80, None);
        assert!(matches!(again, Ask::Serve(_)));
    }
}
