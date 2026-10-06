//! TEMP (lab, `REIMS_VGPU_LAB_DRAIN_LANES=2`): one plan-ahead lane for the
//! next task's EXEC — §9.5 stage 1 of the parallel-drain design
//! (`lab/journal/2026-10-05-parallel-drain-design.md`), lifted from PassPrep's
//! intra-pass chunking (§8) to task granularity without moving a byte of
//! encode.
//!
//! # The shape
//!
//! While the drain executes task A's pass, A's [`super::pass_prep`] chunk
//! scopes already hand `&DeviceState` to pool workers for A's own binds. The
//! lane adds task B's binds to those same scopes: B is an *admitted but not
//! yet run* EXEC (the fc0a/n68 picture — the WindowServer compositor task
//! parked behind Safari's 11.7 ms EXECs in the same ring), its inputs were
//! snapshotted at arrival (`ParkedWork` holds the submission bytes and the
//! model's resolved records, which the guest can no longer touch), and its
//! refs plan through the same [`super::prefetch::plan_item`] ladder A's do.
//! The fork-join scope is the one window a worker may share `&DeviceState`,
//! so riding it changes no aliasing contract at all.
//!
//! # The serve
//!
//! A lane plan is served only through the validation a fresh plan passes.
//! When B's pass finally runs, [`super::pass_prep::PassPrep::open`] hands the
//! lane's plans for B into the new frame (and seeds `planned`, so B's own
//! chunks do not replan them); a bind whose pass pushed no frame (a
//! single-draw pass opens no `PassPrep`) stages a transient frame inside
//! [`super::prefetch::prefetched_source`]. From there the existing
//! validate-then-serve match owns the plan exactly as it owns a chunk's:
//! an invalidated plan lands in the existing `par_prefetch_invalid_*` routes
//! and the serial ladder serves the bind unchanged, which is the whole safety
//! argument — a lane miss costs the discarded plan and nothing else.
//!
//! # The lifetime
//!
//! One lane *per task*, bounded to [`LANE_TASK_CAP`] tasks, chosen at take
//! time rather than offer time. The n235 A/B is why: the drain admits and
//! runs one packet at a time (`admit_and_park` then `settle_model_work`, per
//! packet), so a single "newest offer" lane always held the task that was
//! about to run — the very task `take_backlog` must skip — and
//! `lanes_planned_n` never moved off zero while `lanes_retired_run` tracked
//! `lanes_offer` one for one. With a lane per task, the first admitted
//! cross-task EXEC survives beside the runner's own lane, and a scope of
//! task A's pass plans task B's binds out of B's lane.
//!
//! The offer itself carries no resources and touches no guest memory: it
//! scans the retained bind arenas (in-memory) and parks the raw refs. The
//! `Arc<TaskResource>` each item needs is resolved at *take* time, on the
//! drain, inside [`take_backlog`] — the same construction the owning task's
//! walk would perform moments later (memoized in `task_resources`), so the
//! work moves earlier by one pass rather than being added, and it is paid
//! only for items that actually plan. n235's second lesson: resolving at
//! offer time cost ~750k constructions a boot (every admitted EXEC, most
//! lanes never planned) and took the windowed-glass scene from 193 to
//! 95-135 Hz.
//!
//! A newer EXEC of a task supersedes only that lane's **backlog** — the
//! plans, bails and cold marks survive the supersede (n241: 946/977 of the
//! ~970 plans died exactly there, one packet old and still valid by
//! validation; a task binds largely the same refs packet over packet). The
//! oldest lane is evicted at the cap; a lane is **not** dropped when a
//! packet of its task runs — retirement-by-task would conflate packets (a
//! task with two queued EXECs holds its *newest* refs, and the older
//! packet's run would destroy the newer one's plans). Offers for the task
//! currently executing are ignored — its own chunks plan its refs.
//!
//! # Bails and the cold warm-up (n241's coverage lesson)
//!
//! n241's lane bails were dominated by `cold_packed` (1412-1435) and
//! `upper_rung` (401-479). `upper_rung` the planner must keep refusing (an
//! upper rung is state the serial rail resolves itself; predicting it is the
//! readiness-flip class), so those refs are only marked `bailed` and the
//! owning pass's open seeds its `planned` from them — its chunks skip
//! re-attempting what refused one pass ago. `cold_packed` is different: the
//! missing packed alias is *construction work* the drain may legally do
//! (the PACKED_WARM cork's own move), so a cold-bailed ref is marked `cold`
//! and the next offer of its task warms it on the drain —
//! [`warm_cold_refs`] runs `ensure_packed_resource` before the lane's
//! backlog is rebuilt, one attempt per ref per lane epoch, and a construction
//! that lands also clears the bail so a later take replans onto the warm
//! rail.
//!
//! Unset (or `1`) is byte-for-byte today's path: no lane exists, no scope
//! grows an item, no ask consults a store that is always empty.
//!
//! # Threading
//!
//! Drain-thread state, like the pass-frame `STACK` it feeds: every device
//! entry point runs under the device lock on the one QEMU worker that owns
//! it. If a future host ever served the device from more than one thread, a
//! lane planned on one would be invisible on the other — a silent miss and a
//! serial serve, never a wrong answer; the census (`lanes_planned_n` high
//! while `lanes_merged_n` stays low) would name it.

use super::prefetch::StreamPlan;
use super::*;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};

/// The most distinct references one lane will plan. The guest's EXEC binds
/// are bounded by the command buffer's bytes (a record has a minimum encoded
/// length), so this is a fence against a pathological packet, not a working
/// set the lane expects to fill.
const LANE_REF_CAP: usize = 64;

/// The most tasks with lanes at once. Two is the measured transition shape
/// (the app's render task and the compositor's); four leaves room for a
/// compute or blit task beside them without the store growing with the
/// guest's task count.
const LANE_TASK_CAP: usize = 4;

/// TEMP (lab): `REIMS_VGPU_LAB_DRAIN_LANES` — `2` opens the lane, unset and
/// every other value are today's path. Latched, with a test override that
/// stands in front of the latch (the PassPrep `test_mode` discipline).
pub(crate) fn lanes_on() -> bool {
    #[cfg(test)]
    {
        if TEST_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
            return true;
        }
    }
    // TEMP (lab): live through `REIMS_VGPU_LAB_LIVE_FILE` (`DRAIN_LANES=2`
    // opens it, any other value closes it), so an A/B runs as phases of one
    // boot. A lane's plan is checked at commit like every other plan, so one
    // left in the store across a switch is refused, not served.
    #[cfg(not(test))]
    if let Some(v) = crate::runtime::lab_live::value("DRAIN_LANES") {
        return v == "2";
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = matches!(
            std::env::var("REIMS_VGPU_LAB_DRAIN_LANES").as_deref(),
            Ok("2")
        );
        crate::observe::off(format!("lab_drain_lanes on={on}"));
        on
    })
}

#[cfg(test)]
static TEST_FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Force the gate for the lifetime of the returned guard (tests only).
#[cfg(test)]
pub(super) fn test_force() -> TestForceGuard {
    TEST_FORCE.store(true, std::sync::atomic::Ordering::Relaxed);
    TestForceGuard
}

/// Clears the [`test_force`] override. Never outlives its guard's frame.
#[cfg(test)]
pub(super) struct TestForceGuard;

#[cfg(test)]
impl Drop for TestForceGuard {
    fn drop(&mut self) {
        TEST_FORCE.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// One task's lane: the unplanned refs of its most recently admitted EXEC,
/// the plans other tasks' scopes have filled in for them, the refs those
/// scopes attempted and bailed on, and the cold subset of those bails (see
/// the module doc). The task is the slot's key, not a field — every rule
/// below is "which task's lane", asked once.
struct Lane {
    backlog: VecDeque<u32>,
    planned: HashMap<u32, StreamPlan>,
    /// Refs a scope attempted and the ladder refused. Survives supersedes;
    /// taken (one-shot) by the owning pass's open to seed its `planned`.
    bailed: HashSet<u32>,
    /// The `ColdPacked` subset of [`Self::bailed`]: refs whose missing packed
    /// alias is construction work the next offer's warm-up will attempt.
    cold: HashSet<u32>,
}

thread_local! {
    /// The lanes, in first-offer order (the eviction order, and the take
    /// order — an earlier-admitted task runs first). Drain-thread state, like
    /// the pass-frame `STACK` it feeds: every device entry point runs under
    /// the device lock on one thread.
    static LANES: RefCell<Vec<(u32, Lane)>> = const { RefCell::new(Vec::new()) };
    /// The task whose EXEC the drain is inside right now, for the offer rule
    /// (an offer for the running task never installs: its own chunks plan
    /// its refs).
    static RUNNING_TASK: Cell<Option<u32>> = const { Cell::new(None) };
}

/// Count the plans a lane drop throws away, so `lanes_discarded_n` says both
/// halves: a plan invalidated at serve and a plan retired with its lane.
fn note_discarded(n: u64) {
    if n != 0 {
        crate::runtime::drain::note_store_route_n("lanes_discarded_n", n);
    }
}

/// The lane feed: an admitted EXEC's bind inventory, from the retained
/// `ExecWork` and never from the ring.
///
/// Called on the drain at admission, where `arrived.submission` and the
/// resolved work are both in hand and the move into `ParkedWork` has not
/// happened yet. The scan reads the model's own bind arenas — the same data
/// the runtime walk will bind one packet later — and parks the raw refs.
/// **Nothing else happens here**: no guest reads, no resource construction,
/// no model writes. n235 measured the earlier shape — resolve-at-offer, one
/// construction per distinct ref per admitted EXEC — at ~750k constructions
/// a boot, most for lanes that never planned, and it halved the windowed
/// glass's frame rate. The `Arc<TaskResource>` identity an item needs is
/// resolved at take time instead, inside [`take_backlog`], only for items
/// that are about to plan.
///
/// The slot rules: the running task's packets never install (its own chunks
/// plan its refs); a newer EXEC of a task supersedes that lane's *backlog*
/// only — plans, bails and cold marks survive (n241: nearly every plan died
/// on the supersede, one packet old and still valid by validation); the
/// oldest lane is evicted at [`LANE_TASK_CAP`].
///
/// `refs` is the packet's scanned bind inventory (the caller owns the scan
/// so the arrival-declare knob can share it); refs already planned or bailed
/// stay out of the fresh backlog — they are done, not waiting.
pub(crate) fn offer_exec(state: &DeviceState, task_id: u32, refs: &[u32]) {
    if !lanes_on() || pass_prep_mode() == PassPrepMode::Off {
        // No scopes will ever drain a backlog while PassPrep is off, and no
        // ask will consult a store; the scan would be pure cost.
        return;
    }
    if crate::backend::selected().rail() != crate::backend::Rail::Vulkan {
        return;
    }
    if state.par().is_none() {
        // No pool, no scopes, no lane work ever.
        return;
    }
    let running = RUNNING_TASK.with(Cell::get);
    if running == Some(task_id) {
        // The running task plans itself, in its own chunks.
        return;
    }
    // The scanner caps at its own constant; this is the lane's own fence,
    // so the bound below (`LANE_TASK_CAP` lanes of `LANE_REF_CAP` plans)
    // holds whatever the caller hands in.
    let refs = &refs[..refs.len().min(LANE_REF_CAP)];
    crate::runtime::drain::note_store_route("lanes_offer");
    crate::runtime::drain::note_store_route_n("lanes_offer_n", refs.len() as u64);
    LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        if let Some(slot) = lanes.iter_mut().find(|(t, _)| *t == task_id) {
            slot.1.backlog = refs
                .iter()
                .copied()
                .filter(|r| !slot.1.planned.contains_key(r) && !slot.1.bailed.contains(r))
                .collect();
            return;
        }
        if lanes.len() >= LANE_TASK_CAP {
            let (_, evicted) = lanes.remove(0);
            note_discarded(evicted.planned.len() as u64);
            crate::runtime::drain::note_store_route("lanes_evicted");
        }
        lanes.push((
            task_id,
            Lane {
                backlog: refs.iter().copied().collect(),
                planned: HashMap::new(),
                bailed: HashSet::new(),
                cold: HashSet::new(),
            },
        ));
    });
}

/// Record one lane bail: the ref joins the lane's `bailed` set (the owning
/// pass's open will seed its `planned` from it, skipping a re-attempt that
/// would refuse the same way), and a `ColdPacked` bail additionally joins
/// `cold` for the next offer's warm-up.
pub(super) fn note_bail(task_id: u32, texture_ref: u32, bail: SampleBail) {
    LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        let Some((_, lane)) = lanes.iter_mut().find(|(t, _)| *t == task_id) else {
            return;
        };
        lane.bailed.insert(texture_ref);
        if matches!(bail, SampleBail::ColdPacked) {
            lane.cold.insert(texture_ref);
        }
    });
}

/// Warm the task's cold-bailed refs on the drain, at its next offer.
///
/// The PACKED_WARM cork's move at lane granularity: a `ColdPacked` bail
/// names construction work (the packed alias the serial bind would make),
/// and construction is legal here — this is the drain, before the lane's
/// backlog is rebuilt, so a construction that lands also clears the bail and
/// the ref re-enters a future backlog to be replanned onto the warm rail.
/// One attempt per ref per lane epoch either way: a construction that
/// refuses is a held negative the next attempt would meet again.
pub(crate) fn warm_cold_refs<
    M: crate::runtime::host::HostMemory + crate::runtime::host::HostOps,
>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
) {
    if !lanes_on() {
        return;
    }
    // Drained first: cold is one-shot by construction, whatever the attempt
    // answers.
    let cold: Vec<u32> = LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        match lanes.iter_mut().find(|(t, _)| *t == task_id) {
            Some((_, lane)) => std::mem::take(&mut lane.cold).into_iter().collect(),
            None => Vec::new(),
        }
    });
    for texture_ref in cold {
        crate::runtime::drain::note_store_route("lanes_warm");
        let Ok(resource) =
            crate::runtime::objects::resolve_resource(&*state, &*host, task_id, texture_ref)
        else {
            continue;
        };
        // The linear-sample backing, exactly the in-pass cork's shape: the
        // allocation the descriptor names, as one packed window.
        let allocation = match resource.decoded() {
            Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => tex
                .allocation_base_gva(state.page_shift)
                .filter(|_| tex.allocation_size != 0)
                .map(|gva| BufferBacking {
                    gva,
                    size: tex.allocation_size,
                }),
            _ => None,
        };
        let Some(backing) = allocation else {
            continue;
        };
        if ensure_packed_resource(
            state,
            host,
            task_id,
            texture_ref,
            &backing,
            PackedResourceRail::LinearSample,
        ) {
            crate::runtime::drain::note_store_route("lanes_warm_made");
            // The rail is warm: the bail is gone with the cold, and a future
            // backlog replans the ref.
            LANES.with(|lanes| {
                if let Some((_, lane)) =
                    lanes.borrow_mut().iter_mut().find(|(t, _)| *t == task_id)
                {
                    lane.bailed.remove(&texture_ref);
                }
            });
        }
    }
}

/// Take the lane's bailed refs (the `PassPrep::open` seed), one-shot: the
/// owning pass's chunks skip re-attempting them, and the next epoch starts
/// clean (a bail is a fact about the state it was measured against).
pub(super) fn take_bailed_for(task_id: u32) -> HashSet<u32> {
    if !lanes_on() {
        return HashSet::new();
    }
    LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        match lanes.iter_mut().find(|(t, _)| *t == task_id) {
            Some((_, lane)) => std::mem::take(&mut lane.bailed),
            None => HashSet::new(),
        }
    })
}

/// The drain entered a retained EXEC of `task_id`. Sets the running-task
/// marker the offer rule consults.
pub(crate) fn note_exec_started(task_id: u32) {
    if !lanes_on() {
        return;
    }
    RUNNING_TASK.with(|t| t.set(Some(task_id)));
}

/// The drain left a retained EXEC of `task_id`. Clears the running marker —
/// and nothing else.
///
/// Iteration 1 retired the task's lane here; iteration 2 does not, because
/// retirement-by-task conflates *packets*: a task with two queued EXECs (the
/// transition shape — the compositor parks several) has its lane holding the
/// *newest* packet's refs, and the *older* packet's run would have destroyed
/// the plans the drain made for the newer one. The lane outlives any one
/// packet and is retired the honest ways instead: superseded by the task's
/// next offer (packet granularity), evicted at the cap, or invalidated at
/// serve. Memory is bounded by [`LANE_TASK_CAP`] lanes of [`LANE_REF_CAP`]
/// plans, and every surviving plan still passes the serve-time validation a
/// fresh one would.
pub(crate) fn note_exec_finished(_task_id: u32) {
    if !lanes_on() {
        return;
    }
    RUNNING_TASK.with(|t| t.set(None));
}

/// Take up to `budget` items for a scope of `skip_task`'s pass, resolving
/// each taken ref's `Arc<TaskResource>` here, on the drain.
///
/// The lane is chosen at take time — the earliest-offered lane whose task is
/// neither `skip_task`'s (its own refs are that pass's chunks' job; planning
/// them here would duplicate the in-pass work) nor out of backlog. That
/// choice is the n235 fix's heart: with the lane chosen at offer time the
/// store always held the task about to run, and nothing ever planned.
///
/// Resolution failures drop the ref (a descriptor the guest has since
/// rewritten has no identity to plan against; the owning walk re-reads it
/// when the packet runs). The taken entries leave the backlog — they will
/// either come back as a plan through [`store_plan`] or be counted as a
/// bail, never be retaken by a later chunk.
pub(super) fn take_backlog<M: crate::runtime::host::HostMemory>(
    state: &DeviceState,
    host: &M,
    budget: usize,
    skip_task: u32,
) -> Vec<(u32, u32, std::sync::Arc<crate::model::TaskResource>)> {
    if !lanes_on() {
        return Vec::new();
    }
    let Some((task_id, refs)) = LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        let slot = lanes.iter_mut().find(|(t, lane)| {
            *t != skip_task && !lane.backlog.is_empty()
        })?;
        let take = budget.min(slot.1.backlog.len());
        let refs: Vec<u32> = slot.1.backlog.drain(..take).collect();
        Some((slot.0, refs))
    }) else {
        return Vec::new();
    };
    // The resource identity each item will `ptr_eq` against at serve time.
    // This is the construction the owning task's walk performs moments later
    // anyway (memoized in `task_resources`), moved ahead by one pass; paid
    // only for refs that are about to plan, which is what the offer-time
    // version got wrong.
    let mut items = Vec::with_capacity(refs.len());
    for texture_ref in refs {
        if let Ok(resource) =
            crate::runtime::objects::resolve_resource(state, host, task_id, texture_ref)
        {
            items.push((task_id, texture_ref, resource));
        } else {
            crate::runtime::drain::note_store_route("lanes_unresolved");
        }
    }
    items
}

/// Store a plan a scope filled for the lane. A plan that arrives for a lane
/// that no longer exists (superseded while the scope ran) is discarded and
/// counted, which is the honest answer: nobody will ask for it.
pub(super) fn store_plan(task_id: u32, texture_ref: u32, plan: StreamPlan) {
    LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        if let Some((_, lane)) = lanes.iter_mut().find(|(t, _)| *t == task_id) {
            lane.planned.insert(texture_ref, plan);
            return;
        }
        note_discarded(1);
    });
}

/// Take the lane's plan for one bind (the frame-miss fill inside
/// `prefetched_source`). The plan leaves the store — the frame (or the
/// transient frame) owns it from here, and the witness-moving arms are
/// one-shot exactly as they are in a pass frame.
pub(super) fn take_plan(key: &(u32, u32)) -> Option<StreamPlan> {
    if !lanes_on() {
        return None;
    }
    LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        let (_, lane) = lanes.iter_mut().find(|(t, _)| *t == key.0)?;
        lane.planned.remove(&key.1)
    })
}

/// Take every plan the task's lane holds (the `PassPrep::open` hand-off), as
/// `(texture_ref, plan)` pairs. `None` when the task has no lane or the lane
/// holds nothing; the lane keeps its remaining backlog either way, so late
/// plans still arrive for the frame-miss fill.
pub(super) fn take_plans_for(task_id: u32) -> Option<Vec<(u32, StreamPlan)>> {
    if !lanes_on() {
        return None;
    }
    LANES.with(|lanes| {
        let mut lanes = lanes.borrow_mut();
        let (_, lane) = lanes.iter_mut().find(|(t, _)| *t == task_id)?;
        if lane.planned.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut lane.planned).into_iter().collect())
    })
}

/// The census route of one lane bail, in the lane's own namespace so an A/B
/// of the knob does not move PassPrep's `par_prefetch_bail_*` bars.
pub(super) fn bail_route(bail: SampleBail) -> &'static str {
    match bail {
        SampleBail::NullRef => "lanes_bail_null_ref",
        SampleBail::BufferTexture => "lanes_bail_buffer_texture",
        SampleBail::RefTextureView => "lanes_bail_ref_texture_view",
        SampleBail::NotSurface => "lanes_bail_not_surface",
        SampleBail::ColdMapperRef => "lanes_bail_cold_mapper_ref",
        SampleBail::ColdBacking => "lanes_bail_cold_backing",
        SampleBail::NoGeometry => "lanes_bail_no_geometry",
        SampleBail::ColdLease => "lanes_bail_cold_lease",
        SampleBail::GuestReplaced => "lanes_bail_guest_replaced",
        SampleBail::NoHostCache => "lanes_bail_no_host_cache",
        SampleBail::LowerRung => "lanes_bail_lower_rung",
        SampleBail::ZeroCopyDeclined => "lanes_bail_zero_copy_declined",
        SampleBail::OwedFrame => "lanes_bail_owed_frame",
        SampleBail::ColdWindow => "lanes_bail_cold_window",
        SampleBail::ColdWitness => "lanes_bail_cold_witness",
        SampleBail::UpperRung => "lanes_bail_upper_rung",
        SampleBail::ColdPacked => "lanes_bail_cold_packed",
        SampleBail::MemoBudget => "lanes_bail_memo_budget",
    }
}

#[cfg(test)]
mod tests {
    use super::super::prefetch;
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::drain::census::store_route_count;
    use crate::runtime::host::FakeHost;
    use reims_vgpu_protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    fn delta(route: &str, before: u64) -> u64 {
        store_route_count(route).saturating_sub(before)
    }

    /// A pooled device with two tasks: task 1 (the draining one, A) and task
    /// 2 (the lane's, B), each with a page-table directory/root pair and one
    /// host-cache texture ref. The zero-helper pool makes every scope run on
    /// the caller, so a chunk's planning is deterministic and in order.
    fn two_task_state() -> (DeviceState, FakeHost) {
        use crate::protocol::endian::st32;
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
        use crate::runtime::gva_mem;
        use reims_vgpu_wire::device_desc::RefTextureBuilder;

        let mut state = pooled_state();
        let mut host = FakeHost::new();
        let (dir_pfn, root_pfn) = (2u32, 3u32);
        let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
        let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        let mut dir = [0u8; 8];
        st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
        st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
        assert!(host.write_gpa(dir_gpa, &dir).is_ok());
        for i in 0..3u32 {
            let pfn = 4 + i;
            host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
        }
        state.define_task(1, 0x1000, dir_pfn);
        assert!(state.set_object_list(1, 0, 32));
        // Task 2's own directory and root, one-to-one over the same task GVA
        // window, so its object list resolves the same way.
        let (dir2_pfn, root2_pfn) = (9u32, 10u32);
        let dir2_gpa = (dir2_pfn as u64) << PAGE_SHIFT_X86;
        let root2_gpa = (root2_pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(dir2_gpa, 0x20, 0);
        host.map_range(root2_gpa, 0x1000, 0);
        let mut dir2 = [0u8; 8];
        st32(&mut dir2[DIRECTORY_ROOT_PFN as usize..], root2_pfn);
        st32(&mut dir2[DIRECTORY_DEPTH as usize..], 1);
        assert!(host.write_gpa(dir2_gpa, &dir2).is_ok());
        for i in 0..3u32 {
            let pfn = 12 + i;
            host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            assert!(host.write_gpa(root2_gpa + (i as u64) * 4, &pte).is_ok());
        }
        state.define_task(2, 0x1000, dir2_pfn);
        assert!(state.set_object_list(2, 0, 32));

        // One host-cache ref per task: (task, ref, surface, fill).
        for (task_id, texture_ref, surface_id, fill) in
            [(1u32, 2u32, 71u32, 0u8), (2, 2, 81, 90), (2, 3, 82, 120)]
        {
            let desc_gva = 0x1000u64 + 0x100 * u64::from(texture_ref);
            let built =
                RefTextureBuilder::new(surface_id, 0, 0, 0).with_len(objects::TYPE5_MIN_LEN);
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[task_id],
                desc_gva,
                built.bytes(),
                PAGE_SHIFT_X86
            )
            .is_ok());
            let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
            let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed =
                (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((objects::TYPE5_MIN_LEN as u32) << 8);
            st32(&mut list_entry, packed);
            list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[task_id],
                list_off,
                &list_entry,
                PAGE_SHIFT_X86
            )
            .is_ok());
            assert!(state.map_surface(surface_id));
            assert!(state.set_mapping_geom(surface_id, 4, 3, MTL_FORMAT_BGRA8_UNORM));
            state.mappings.get_mut(&surface_id).unwrap().page_entries = vec![1];
            crate::runtime::surface_cache::store(
                &mut state,
                surface_id,
                4,
                3,
                [fill, 8, 7, 255].repeat(4 * 3),
            );
        }
        (state, host)
    }

    /// The zero-helper pooled device of the PassPrep tests, unchanged.
    fn pooled_state() -> DeviceState {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut table = crate::qemu::host_ops::ReimsVgpuHostOps::null();
        unsafe extern "C" fn nothing(_ctx: *mut std::os::raw::c_void) {}
        table.thread_attach = Some(nothing);
        table.thread_detach = Some(nothing);
        table.map_pages_stable = 1;
        let pool = std::sync::Arc::new(
            crate::runtime::par::Pool::new(
                &table,
                &crate::runtime::par::Settings {
                    workers: 0,
                    worker_spin: std::time::Duration::ZERO,
                    join_spin: std::time::Duration::ZERO,
                },
            )
            .expect("the table offers the registration pair"),
        );
        state.install_par_pool(pool);
        state
    }

    fn bind_for(
        state: &DeviceState,
        host: &FakeHost,
        task_id: u32,
        texture_ref: u32,
    ) -> TextureBind {
        TextureBind {
            index: 0,
            texture_ref,
            resource: Some(
                objects::resolve_resource(state, host, task_id, texture_ref)
                    .expect("fixture resource"),
            ),
        }
    }

    /// The whole lane, end to end, **in the n235 admission order**: A's EXEC
    /// is offered first, B's second (the drain admits and runs one packet at
    /// a time, so B is admitted while A's lane already stands — the shape the
    /// single-lane store got wrong, planning nothing). A's pass then plans
    /// B's refs inside A's own chunk scope, and B's frame-less bind (a
    /// single-draw pass never opens a `PassPrep`) serves the serial ladder's
    /// own answer for the same state.
    #[test]
    fn a_lane_plans_in_another_tasks_scope_and_serves_the_serial_answer() {
        let _force = test_force();
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (mut planned_state, mut planned_host) = two_task_state();
        let (mut serial_state, mut serial_host) = two_task_state();
        // The admission order of the drain: A (task 1) offered, then B (task
        // 2, refs 2 and 3) offered beside it — B's lane must survive A's
        // offer, not be shadowed by it.
        let before_offer = store_route_count("lanes_offer");
        offer_exec(&planned_state, 1, &[2]);
        offer_exec(&planned_state, 2, &[2, 3]);
        assert_eq!(delta("lanes_offer", before_offer), 2);
        // A's pass: four draws binding A's own ref (2) — the scope that will
        // carry the lane's items beside them. The take resolves B's items
        // through this host, which is why the real fixture host is passed.
        let a_binds: Vec<Vec<TextureBind>> = (0..4)
            .map(|_| vec![bind_for(&planned_state, &planned_host, 1, 2)])
            .collect();
        let tables: Vec<super::super::DrawTables<'_>> = a_binds
            .iter()
            .map(|one| super::super::DrawTables {
                fragment: one.as_slice(),
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let before_planned = store_route_count("lanes_planned_n");
        let mut prep =
            super::super::pass_prep::PassPrep::open_with_chunk(&planned_state, 1, tables, 4)
                .expect("A's pass opens");
        prep.ahead(&planned_state, &planned_host, 0);
        assert_eq!(
            delta("lanes_planned_n", before_planned),
            2,
            "both of B's refs planned inside A's chunk-0 scope, with A offered first"
        );
        drop(prep);
        // B's bind, with no frame on the stack (its pass is single-draw):
        // the transient lane frame serves it, and the answer is the serial
        // ladder's for the same state.
        let resource =
            objects::resolve_resource(&planned_state, &planned_host, 2, 2).expect("B's resource");
        let before_merged = store_route_count("lanes_merged_n");
        let served = prefetch::prefetched_source(
            &mut planned_state,
            &mut planned_host,
            2,
            2,
            Some(&resource),
            true,
        );
        assert_eq!(delta("lanes_merged_n", before_merged), 1);
        let serial = resolve_sampled_source(
            &mut serial_state,
            &mut serial_host,
            2,
            2,
            Some(resource.clone()),
            true,
        )
        .expect("the serial ladder serves");
        match served {
            prefetch::Prefetched::Serve((pw, ph, pmid, psrc)) => {
                let (sw, sh, smid, ssrc) = serial;
                assert_eq!((pw, ph, pmid), (sw, sh, smid));
                match (psrc, ssrc) {
                    (
                        SampledSourceRequest::Bytes(a, aid, al, ao),
                        SampledSourceRequest::Bytes(b, bid, bl, bo),
                    ) => {
                        assert!(std::sync::Arc::ptr_eq(&a, &b) || a[..] == b[..]);
                        assert_eq!(
                            (
                                aid.as_ref().map(|i| i.key),
                                al.layout(),
                                std::mem::discriminant(&ao)
                            ),
                            (
                                bid.as_ref().map(|i| i.key),
                                bl.layout(),
                                std::mem::discriminant(&bo)
                            )
                        );
                    }
                    (planned_src, _serial_src) => panic!(
                        "both arms serve host-cache bytes, got a different variant ({})",
                        match planned_src {
                            SampledSourceRequest::GuestRuns(..) => "runs",
                            SampledSourceRequest::Target(..) => "target",
                            SampledSourceRequest::TargetRegion(..) => "target_region",
                            SampledSourceRequest::Bytes(..) => "bytes",
                            SampledSourceRequest::Staged(..) => "staged",
                        }
                    ),
                }
            }
            other => panic!(
                "the lane serves B's bind, got shadow={}",
                matches!(other, prefetch::Prefetched::Shadow(_))
            ),
        }
        // And the observable state the two arms leave agrees: the host cache
        // generations and the mappings' epochs.
        for surface_id in [81u32, 82] {
            let p = crate::runtime::surface_cache::get_shared_with_gen(
                &planned_state,
                surface_id,
                4,
                3,
            );
            let s =
                crate::runtime::surface_cache::get_shared_with_gen(&serial_state, surface_id, 4, 3);
            assert_eq!(
                p.map(|(_, g)| g),
                s.map(|(_, g)| g),
                "cache gen {surface_id}"
            );
            let pm = planned_state.mappings.get(&surface_id).unwrap();
            let sm = serial_state.mappings.get(&surface_id).unwrap();
            assert_eq!(
                (pm.map_generation, pm.surface_content_epoch),
                (sm.map_generation, sm.surface_content_epoch),
                "mapping epochs {surface_id}"
            );
        }
    }

    /// A mapping whose generation moved between the scope that planned it and
    /// the bind that asks is dropped through the existing invalidation route
    /// (`par_prefetch_invalid_mapping` — the same check a frame plan fails)
    /// and the serial ladder serves the bind; the store no longer holds the
    /// plan.
    #[test]
    fn a_moved_mapping_discards_the_lane_plan_and_falls_back() {
        let _force = test_force();
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (mut state, mut host) = two_task_state();
        offer_exec(&state, 1, &[2]);
        offer_exec(&state, 2, &[2, 3]);
        let a_binds: Vec<Vec<TextureBind>> = (0..4)
            .map(|_| vec![bind_for(&state, &host, 1, 2)])
            .collect();
        let tables: Vec<super::super::DrawTables<'_>> = a_binds
            .iter()
            .map(|one| super::super::DrawTables {
                fragment: one.as_slice(),
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let mut prep = super::super::pass_prep::PassPrep::open_with_chunk(&state, 1, tables, 4)
            .expect("A's pass opens");
        prep.ahead(&state, &host, 0);
        drop(prep);
        // The guest rewires B's surface 81 between the plan and the ask.
        state.mappings.get_mut(&81u32).unwrap().map_generation += 1;
        let resource = objects::resolve_resource(&state, &host, 2, 2).expect("B's resource");
        let before_invalid = store_route_count("par_prefetch_invalid_mapping");
        let served =
            prefetch::prefetched_source(&mut state, &mut host, 2, 2, Some(&resource), true);
        assert!(matches!(served, prefetch::Prefetched::Serial));
        assert_eq!(delta("par_prefetch_invalid_mapping", before_invalid), 1);
        // The store no longer holds the plan: it left with the take.
        assert!(take_plan(&(2, 2)).is_none());
        let serial = resolve_sampled_source(&mut state, &mut host, 2, 2, Some(resource), true);
        assert!(serial.is_some(), "the serial ladder still serves the bind");
    }

    /// The gate off (the default): the feed, the scope rider and the ask are
    /// all no-ops — byte-for-byte today's path.
    #[test]
    fn the_gate_off_is_the_old_path() {
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (mut state, mut host) = two_task_state();
        let before_offer = store_route_count("lanes_offer");
        offer_exec(&state, 2, &[2, 3]);
        assert_eq!(delta("lanes_offer", before_offer), 0);
        assert!(take_plan(&(2, 2)).is_none());
        assert!(take_plans_for(2).is_none());
        assert!(take_backlog(&state, &host, 8, 1).is_empty());
        let a_binds: Vec<Vec<TextureBind>> = (0..4)
            .map(|_| vec![bind_for(&state, &host, 1, 2)])
            .collect();
        let tables: Vec<super::super::DrawTables<'_>> = a_binds
            .iter()
            .map(|one| super::super::DrawTables {
                fragment: one.as_slice(),
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let before_planned = store_route_count("lanes_planned_n");
        let mut prep = super::super::pass_prep::PassPrep::open_with_chunk(&state, 1, tables, 4)
            .expect("A's pass opens");
        prep.ahead(&state, &host, 0);
        assert_eq!(delta("lanes_planned_n", before_planned), 0);
        drop(prep);
        let resource = objects::resolve_resource(&state, &host, 2, 2).expect("B's resource");
        let served =
            prefetch::prefetched_source(&mut state, &mut host, 2, 2, Some(&resource), true);
        assert!(matches!(served, prefetch::Prefetched::Serial));
    }

    /// The slot rules, in the multi-lane store: every task keeps its own lane
    /// (the n235 fix — a cross-task offer installs beside the runner's, it is
    /// never shadowed); a newer EXEC of a task supersedes that task's lane; a
    /// packet's run does NOT retire its task's lane (a later queued packet of
    /// the same task still owns it); the running task's own offers never
    /// install; the cap evicts the oldest.
    #[test]
    fn the_slot_rules_keep_one_lane_per_task() {
        let _force = test_force();
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (state, host) = two_task_state();
        // The drain's admission order: A (task 1) first, then B (task 2) —
        // both lanes stand, which is what the single-lane store refused.
        offer_exec(&state, 1, &[2]);
        offer_exec(&state, 2, &[2, 3]);
        // A runs: A's own lane is skipped, B's is taken.
        note_exec_started(1);
        let backlog = take_backlog(&state, &host, 8, 1);
        assert_eq!(backlog.len(), 2, "B's backlog, taken from A's scope");
        assert_eq!(backlog[0].0, 2, "the items are B's (task 2)");
        // Safari's next EXEC (task 1) is admitted mid-drain: the running
        // task's offer never installs, and A's taken entries do not come
        // back — but B's lane stands untouched.
        offer_exec(&state, 1, &[2]);
        let backlog = take_backlog(&state, &host, 8, 1);
        assert!(
            backlog.is_empty(),
            "A's taken entries do not come back and A never re-installed"
        );
        // A's lane is drained once here, so every take below names its lane.
        note_exec_finished(1);
        let a_items = take_backlog(&state, &host, 8, 2);
        assert_eq!(a_items.len(), 1, "A's own lane held A's one ref");
        assert_eq!(a_items[0].0, 1);
        // A newer EXEC of B supersedes B's lane: fresh backlog.
        offer_exec(&state, 2, &[3]);
        // B's packet runs: the lane SURVIVES it (a later queued packet of B
        // still owns what it holds) — the take after the run still finds
        // B's refs, drained only by a take.
        note_exec_started(2);
        note_exec_finished(2);
        let backlog = take_backlog(&state, &host, 8, 1);
        assert_eq!(backlog.len(), 1, "the lane survives its task's run");
        assert_eq!(backlog[0].0, 2);
        // And the next same-task offer supersedes it.
        offer_exec(&state, 2, &[2, 3]);
        let backlog = take_backlog(&state, &host, 8, 1);
        assert_eq!(backlog.len(), 2, "the next same-task offer superseded it");
        // The cap: four tasks fill the store, the fifth evicts the oldest
        // (tasks 3-5 have no fixture page tables, so only task 2's lane can
        // resolve items — enough to prove the cap fired and a survivor
        // stands; the "oldest first" order is the Vec's own).
        offer_exec(&state, 3, &[2]);
        offer_exec(&state, 4, &[2]);
        let before_evicted = store_route_count("lanes_evicted");
        offer_exec(&state, 5, &[2]);
        assert_eq!(delta("lanes_evicted", before_evicted), 1);
        // Task 2's lane survived (it replaces in place, not as a new install
        // that would have evicted again) and still serves items.
        offer_exec(&state, 2, &[2, 3]);
        assert_eq!(
            delta("lanes_evicted", before_evicted),
            1,
            "the replace evicted nothing"
        );
        let backlog = take_backlog(&state, &host, 8, 5);
        assert_eq!(
            backlog[0].0, 2,
            "task 2's lane survived the eviction and still serves items"
        );
    }

    /// The `PassPrep::open` hand-off: a lane's plans for the opening task
    /// become the new frame's first entries and seed `planned`, so the pass's
    /// own chunk does not replan them while its binds serve from the frame.
    #[test]
    fn the_opening_pass_adopts_the_lanes_plans() {
        let _force = test_force();
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (mut state, mut host) = two_task_state();
        offer_exec(&state, 1, &[2]);
        offer_exec(&state, 2, &[2, 3]);
        // A's scope plans B's refs.
        let a_binds: Vec<Vec<TextureBind>> = (0..4)
            .map(|_| vec![bind_for(&state, &host, 1, 2)])
            .collect();
        let tables: Vec<super::super::DrawTables<'_>> = a_binds
            .iter()
            .map(|one| super::super::DrawTables {
                fragment: one.as_slice(),
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let mut prep_a = super::super::pass_prep::PassPrep::open_with_chunk(&state, 1, tables, 4)
            .expect("A's pass opens");
        prep_a.ahead(&state, &host, 0);
        drop(prep_a);
        // B's own pass opens (two draws binding ref 2): the lane hands its
        // plans over, and B's chunk finds nothing left to plan for them.
        let b_binds: Vec<Vec<TextureBind>> = (0..2)
            .map(|_| vec![bind_for(&state, &host, 2, 2)])
            .collect();
        let b_tables: Vec<super::super::DrawTables<'_>> = b_binds
            .iter()
            .map(|one| super::super::DrawTables {
                fragment: one.as_slice(),
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let before_merged = store_route_count("lanes_merged_n");
        let before_prefetch = store_route_count("par_prefetch_planned");
        let mut prep_b = super::super::pass_prep::PassPrep::open_with_chunk(&state, 2, b_tables, 2)
            .expect("B's pass opens");
        assert_eq!(
            delta("lanes_merged_n", before_merged),
            2,
            "both plans handed to the frame"
        );
        prep_b.ahead(&state, &host, 0);
        assert_eq!(
            delta("par_prefetch_planned", before_prefetch),
            0,
            "B's own chunk did not replan the lane's refs"
        );
        // And the first bind serves from the adopted frame plan.
        let resource = objects::resolve_resource(&state, &host, 2, 2).expect("B's resource");
        let served =
            prefetch::prefetched_source(&mut state, &mut host, 2, 2, Some(&resource), true);
        assert!(
            matches!(served, prefetch::Prefetched::Serve(_)),
            "the adopted frame plan serves B's bind"
        );
        drop(prep_b);
    }

    /// The n241 funnel fix: a supersede (the task's next offer) replaces only
    /// the backlog — the plans, bails and cold marks survive it, one packet
    /// old and still valid by validation, and the fresh backlog excludes the
    /// refs that are already done.
    #[test]
    fn a_supersede_keeps_plans_and_bails() {
        let _force = test_force();
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (state, host) = two_task_state();
        offer_exec(&state, 1, &[2]);
        offer_exec(&state, 2, &[2, 3]);
        // A's scope plans both of B's refs.
        let a_binds: Vec<Vec<TextureBind>> = (0..4)
            .map(|_| vec![bind_for(&state, &host, 1, 2)])
            .collect();
        let tables: Vec<super::super::DrawTables<'_>> = a_binds
            .iter()
            .map(|one| super::super::DrawTables {
                fragment: one.as_slice(),
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let mut prep_a = super::super::pass_prep::PassPrep::open_with_chunk(&state, 1, tables, 4)
            .expect("A's pass opens");
        prep_a.ahead(&state, &host, 0);
        drop(prep_a);
        // One of B's refs bails (recorded), then B's next packet is offered —
        // the supersede. Iteration 2 discarded the plan here.
        note_bail(2, 3, SampleBail::UpperRung);
        let before_discarded = store_route_count("lanes_discarded_n");
        offer_exec(&state, 2, &[2, 3]);
        assert_eq!(delta("lanes_discarded_n", before_discarded), 0);
        // The plans survived: B's open adopts them.
        let plans = take_plans_for(2).expect("the plans survived the supersede");
        assert_eq!(plans.len(), 2);
        // The bails survived too — one-shot at the owning pass's open.
        let bailed = take_bailed_for(2);
        assert_eq!(bailed.len(), 1);
        assert!(bailed.contains(&3));
        assert!(
            take_bailed_for(2).is_empty(),
            "the bailed seed is one-shot"
        );
        // And the fresh backlog excludes the done refs: nothing left to take.
        let backlog = take_backlog(&state, &host, 8, 1);
        assert!(
            backlog.is_empty(),
            "planned and bailed refs stay out of the fresh backlog"
        );
    }

    /// The cold warm-up: a `ColdPacked` bail joins the lane's cold set, and
    /// the task's next offer warms it once on the drain — this host's
    /// construction refuses (no import caps), which is the held negative the
    /// PACKED_WARM reasoning keeps final, so the ref is not re-warmed and the
    /// bail stays for the owning pass's seed.
    #[test]
    fn a_cold_bail_warms_once_at_the_next_offer() {
        let _force = test_force();
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (mut state, mut host) = two_task_state();
        offer_exec(&state, 2, &[2, 3]);
        note_bail(2, 2, SampleBail::ColdPacked);
        note_bail(2, 3, SampleBail::UpperRung);
        let before_warm = store_route_count("lanes_warm");
        let before_made = store_route_count("lanes_warm_made");
        warm_cold_refs(&mut state, &mut host, 2);
        // Only the cold ref is warm work; the rung bail never was.
        assert_eq!(delta("lanes_warm", before_warm), 1);
        // This host's packed construction refuses, so nothing was made and
        // the bail survives for the owning pass's seed.
        assert_eq!(delta("lanes_warm_made", before_made), 0);
        let bailed = take_bailed_for(2);
        assert_eq!(bailed.len(), 2, "both bails still seed the owning pass");
        // One attempt per ref per epoch: a second warm does nothing.
        warm_cold_refs(&mut state, &mut host, 2);
        assert_eq!(delta("lanes_warm", before_warm), 1);
        // And a task with no lane is a no-op.
        warm_cold_refs(&mut state, &mut host, 7);
        assert_eq!(delta("lanes_warm", before_warm), 1);
    }

    /// The owning pass's open seeds its `planned` from the lane's bails: the
    /// pass's own chunk does not re-attempt what a lane scope refused one
    /// pass ago (n241's upper_rung family — the planner must keep refusing
    /// it, so the saving is the skipped attempt, not a serve).
    #[test]
    fn the_opening_pass_seeds_from_the_lanes_bails() {
        let _force = test_force();
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (mut state, mut host) = two_task_state();
        offer_exec(&state, 2, &[2, 3]);
        note_bail(2, 2, SampleBail::UpperRung);
        // B's pass opens (two draws binding the bailed ref): the seed puts
        // the ref in `planned`, so the chunk does not attempt it.
        let b_binds: Vec<Vec<TextureBind>> = (0..2)
            .map(|_| vec![bind_for(&state, &host, 2, 2)])
            .collect();
        let b_tables: Vec<super::super::DrawTables<'_>> = b_binds
            .iter()
            .map(|one| super::super::DrawTables {
                fragment: one.as_slice(),
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let before_seed = store_route_count("lanes_seed_n");
        let before_prefetch = store_route_count("par_prefetch_planned");
        let mut prep_b = super::super::pass_prep::PassPrep::open_with_chunk(&state, 2, b_tables, 2)
            .expect("B's pass opens");
        assert_eq!(delta("lanes_seed_n", before_seed), 1);
        prep_b.ahead(&state, &host, 0);
        assert_eq!(
            delta("par_prefetch_planned", before_prefetch),
            0,
            "the bailed ref was not re-attempted by B's own chunk"
        );
        // The bind still serves — through the serial ladder, as a bail always
        // meant.
        let resource = objects::resolve_resource(&state, &host, 2, 2).expect("B's resource");
        let served =
            prefetch::prefetched_source(&mut state, &mut host, 2, 2, Some(&resource), true);
        assert!(matches!(served, prefetch::Prefetched::Serial));
        drop(prep_b);
    }
}
