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
//! One lane at a time, bounded: the most recently *admitted* EXEC of a task
//! other than the one currently executing. An offer for the running task
//! never evicts a cross-task lane (that is precisely the compositor-behind-
//! Safari case); a newer EXEC of the lane's own task supersedes it; the lane
//! is dropped when its task's packet has executed. The backlog is capped, so
//! a task that never runs pins at most [`LANE_REF_CAP`] `Arc<TaskResource>`s.
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

/// One backlog entry: the reference and the resource the drain resolved for
/// it at offer time. The `Arc` is the identity the bind's validation will
/// `ptr_eq` against when the lane's task runs — resolved on the drain, where
/// construction is legal, exactly the way a PassPrep item carries its
/// resource rather than letting a worker construct one.
struct LaneRef {
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
}

/// The lane itself: one task's not-yet-run EXEC, its unplanned backlog and
/// the plans A's scopes have filled in.
struct Lane {
    task_id: u32,
    backlog: VecDeque<LaneRef>,
    planned: HashMap<u32, StreamPlan>,
}

thread_local! {
    /// The one lane. Drain-thread state, like the pass-frame `STACK` it feeds:
    /// every device entry point runs under the device lock on one thread.
    static LANE: RefCell<Option<Lane>> = const { RefCell::new(None) };
    /// The task whose EXEC the drain is inside right now, for the offer rule
    /// (an offer for the running task never evicts a cross-task lane).
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
/// the runtime walk will bind one packet later — and resolves each distinct
/// reference through [`crate::runtime::objects::resolve_resource`], which is
/// the construction the walk's first bind of the reference would perform
/// anyway. A reference that does not resolve is not offered: it has no
/// identity to plan against.
///
/// The replacement rules (see the module doc): the running task's packets
/// never become the lane; a newer EXEC of the lane's own task supersedes it
/// (its binds are the next ones that matter for that task); anything else
/// leaves the lane alone — an older admitted cross-task packet runs first,
/// and evicting it for a newer sibling would plan for the wrong next.
pub(crate) fn offer_exec<M: crate::runtime::host::HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    work: &reims_vgpu_core::exec::ExecWork,
) {
    if !lanes_on() || pass_prep_mode() == PassPrepMode::Off {
        // No scopes will ever drain a backlog while PassPrep is off, and no
        // ask will consult a store; the scan would be pure cost.
        return;
    }
    if crate::backend::selected().rail() != crate::backend::Rail::Vulkan {
        return;
    }
    let running = RUNNING_TASK.with(Cell::get);
    if running == Some(task_id) {
        // The running task plans itself, in its own chunks.
        return;
    }
    // The bind inventory: every distinct reference the render records bind.
    // Compute encoders are skipped — their textures do not walk the render
    // pass tables this lane plans for. A span that names no window of the
    // arena (a resolver this lane does not model) contributes nothing rather
    // than panicking: the lane is an accelerator, and a packet it cannot
    // inventory plans serially exactly as it always did.
    let mut seen = HashSet::new();
    let mut refs: Vec<u32> = Vec::new();
    for record in work.records() {
        if let reims_vgpu_core::exec::ResolvedOperation::Render(
            reims_vgpu_core::render::RenderOp::BindTextures { entries, .. },
        ) = &record.op
        {
            let Some(bindings) = work.arenas.object_bindings.get(entries.range()) else {
                continue;
            };
            for binding in bindings {
                if let Some(object) = binding.object {
                    let texture_ref = object.slot.0;
                    if texture_ref != 0 && seen.insert(texture_ref) {
                        refs.push(texture_ref);
                    }
                }
            }
        }
    }
    refs.truncate(LANE_REF_CAP);
    let mut backlog = VecDeque::with_capacity(refs.len());
    for texture_ref in refs {
        if let Ok(resource) =
            crate::runtime::objects::resolve_resource(state, host, task_id, texture_ref)
        {
            backlog.push_back(LaneRef {
                texture_ref,
                resource,
            });
        }
    }
    crate::runtime::drain::note_store_route("lanes_offer");
    crate::runtime::drain::note_store_route_n("lanes_offer_n", backlog.len() as u64);
    LANE.with(|lane| {
        let mut lane = lane.borrow_mut();
        match lane.as_mut() {
            Some(current) if current.task_id != task_id => {
                // An older cross-task lane runs first; the newer sibling
                // waits for it to clear.
                crate::runtime::drain::note_store_route("lanes_offer_shadowed");
                return;
            }
            Some(current) => {
                note_discarded(current.planned.len() as u64);
                *current = Lane {
                    task_id,
                    backlog,
                    planned: HashMap::new(),
                };
            }
            None => {
                *lane = Some(Lane {
                    task_id,
                    backlog,
                    planned: HashMap::new(),
                });
            }
        }
    });
}

/// The drain entered a retained EXEC of `task_id`. Sets the running-task
/// marker the offer rule consults.
pub(crate) fn note_exec_started(task_id: u32) {
    if !lanes_on() {
        return;
    }
    RUNNING_TASK.with(|t| t.set(Some(task_id)));
}

/// The drain left a retained EXEC of `task_id`. Clears the running marker and
/// drops the lane when it was that task's — its packet has run, so whatever
/// the lane still holds will never be asked for by this packet again.
pub(crate) fn note_exec_finished(task_id: u32) {
    if !lanes_on() {
        return;
    }
    RUNNING_TASK.with(|t| t.set(None));
    LANE.with(|lane| {
        let mut lane = lane.borrow_mut();
        if lane.as_ref().is_some_and(|l| l.task_id == task_id) {
            if let Some(current) = lane.take() {
                note_discarded(current.planned.len() as u64);
            }
            crate::runtime::drain::note_store_route("lanes_retired_run");
        }
    });
}

/// Take up to `budget` backlog entries for a scope of `skip_task`'s pass.
///
/// `skip_task` is the task whose PassPrep is about to fork: its own refs are
/// that pass's chunks' job (planning them here would duplicate the in-pass
/// work), so a lane for the same task declines. The taken entries leave the
/// backlog — they will either come back as a plan through [`store_plan`] or
/// be counted as a bail, never be retaken by a later chunk.
pub(super) fn take_backlog(
    budget: usize,
    skip_task: u32,
) -> Vec<(u32, u32, std::sync::Arc<crate::model::TaskResource>)> {
    if !lanes_on() {
        return Vec::new();
    }
    LANE.with(|lane| {
        let mut lane = lane.borrow_mut();
        let Some(current) = lane.as_mut() else {
            return Vec::new();
        };
        if current.task_id == skip_task || current.backlog.is_empty() {
            return Vec::new();
        }
        let take = budget.min(current.backlog.len());
        current
            .backlog
            .drain(..take)
            .map(|entry| (current.task_id, entry.texture_ref, entry.resource))
            .collect()
    })
}

/// Store a plan a scope filled for the lane. A plan that arrives for a lane
/// that no longer exists (superseded while the scope ran) is discarded and
/// counted, which is the honest answer: nobody will ask for it.
pub(super) fn store_plan(task_id: u32, texture_ref: u32, plan: StreamPlan) {
    LANE.with(|lane| {
        let mut lane = lane.borrow_mut();
        if let Some(current) = lane.as_mut() {
            if current.task_id == task_id {
                current.planned.insert(texture_ref, plan);
                return;
            }
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
    LANE.with(|lane| {
        let mut lane = lane.borrow_mut();
        let current = lane.as_mut()?;
        if current.task_id != key.0 {
            return None;
        }
        current.planned.remove(&key.1)
    })
}

/// Take every plan the lane holds for `task_id` (the `PassPrep::open`
/// hand-off), as `(texture_ref, plan)` pairs. `None` when the lane is another
/// task's or holds nothing; the lane keeps its remaining backlog either way,
/// so late plans still arrive for the frame-miss fill.
pub(super) fn take_plans_for(task_id: u32) -> Option<Vec<(u32, StreamPlan)>> {
    if !lanes_on() {
        return None;
    }
    LANE.with(|lane| {
        let mut lane = lane.borrow_mut();
        let current = lane.as_mut()?;
        if current.task_id != task_id || current.planned.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut current.planned).into_iter().collect())
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

    /// An `ExecWork` whose one render segment binds `refs` as fragment
    /// textures — the minimal shape of the model's resolved records the lane
    /// feed scans (the arenas and spans are the real ones the walk fills).
    fn exec_work_with_binds(refs: &[u32]) -> reims_vgpu_core::exec::ExecWork {
        use reims_vgpu_core::bind::ObjectBinding;
        use reims_vgpu_core::exec::{ExecWork, ResolvedOperation, ResolvedStream, StreamRecord};
        use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
        use reims_vgpu_core::render::{RenderOp, ShaderStage};
        use reims_vgpu_core::stream::{SegmentBegin, SegmentKind, StreamPosition};
        use reims_vgpu_core::sync::ResourceSpan;

        let mut work = ExecWork::default();
        let start = work.arenas.object_bindings.len() as u32;
        for &texture_ref in refs {
            work.arenas.object_bindings.push(ObjectBinding {
                object: (texture_ref != 0).then_some(ResourceId {
                    slot: ObjectListRef(texture_ref),
                    generation: SlotGeneration(1),
                }),
                lod_clamps: None,
            });
        }
        work.streams.push(ResolvedStream {
            begin: SegmentBegin {
                kind: SegmentKind::Render,
                protection: None,
            },
            records: vec![StreamRecord {
                at: StreamPosition {
                    segment: 0,
                    record: 0,
                },
                op: ResolvedOperation::Render(RenderOp::BindTextures {
                    stage: ShaderStage::Fragment,
                    first: 0,
                    entries: ResourceSpan {
                        start,
                        len: refs.len() as u32,
                    },
                }),
            }],
        });
        work
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

    /// The whole lane, end to end: task A's pass plans task B's refs inside
    /// A's own chunk scope, and B's frame-less bind (a single-draw pass never
    /// opens a `PassPrep`) serves the serial ladder's own answer for the same
    /// state.
    #[test]
    fn a_lane_plans_in_another_tasks_scope_and_serves_the_serial_answer() {
        let _force = test_force();
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (mut planned_state, mut planned_host) = two_task_state();
        let (mut serial_state, mut serial_host) = two_task_state();
        // B's admitted EXEC offers the lane (task 2, refs 2 and 3).
        let work = exec_work_with_binds(&[2, 3]);
        let before_offer = store_route_count("lanes_offer");
        offer_exec(&planned_state, &planned_host, 2, &work);
        assert_eq!(delta("lanes_offer", before_offer), 1);
        // A's pass: four draws binding A's own ref (2) — the scope that will
        // carry the lane's items beside them.
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
        prep.ahead(&planned_state, &FakeHost::new(), 0);
        assert_eq!(
            delta("lanes_planned_n", before_planned),
            2,
            "both of B's refs planned inside A's chunk-0 scope"
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
        offer_exec(&state, &host, 2, &exec_work_with_binds(&[2, 3]));
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
        prep.ahead(&state, &FakeHost::new(), 0);
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
        let work = exec_work_with_binds(&[2, 3]);
        let before_offer = store_route_count("lanes_offer");
        offer_exec(&state, &host, 2, &work);
        assert_eq!(delta("lanes_offer", before_offer), 0);
        assert!(take_plan(&(2, 2)).is_none());
        assert!(take_plans_for(2).is_none());
        assert!(take_backlog(8, 1).is_empty());
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
        prep.ahead(&state, &FakeHost::new(), 0);
        assert_eq!(delta("lanes_planned_n", before_planned), 0);
        drop(prep);
        let resource = objects::resolve_resource(&state, &host, 2, 2).expect("B's resource");
        let served =
            prefetch::prefetched_source(&mut state, &mut host, 2, 2, Some(&resource), true);
        assert!(matches!(served, prefetch::Prefetched::Serial));
    }

    /// The offer rules: the running task never evicts a cross-task lane (the
    /// fc0a/n68 case — WindowServer admitted behind draining Safari), a newer
    /// EXEC of the lane's own task supersedes it, and the lane retires when
    /// its task's packet has run.
    #[test]
    fn the_offer_rules_keep_the_cross_task_lane() {
        let _force = test_force();
        let _mode = super::super::pass_prep::test_mode(PassPrepMode::On);
        let (mut state, mut host) = two_task_state();
        // B (task 2) is admitted while nothing runs: it becomes the lane.
        offer_exec(&state, &host, 2, &exec_work_with_binds(&[2, 3]));
        // A (task 1) starts draining.
        note_exec_started(1);
        // Safari's next EXEC (task 1) is admitted mid-drain: it must NOT
        // evict B's lane.
        let before_shadowed = store_route_count("lanes_offer_shadowed");
        offer_exec(&state, &host, 1, &exec_work_with_binds(&[2]));
        assert_eq!(delta("lanes_offer_shadowed", before_shadowed), 0);
        // The backlog is still B's (a scope of A's pass takes it).
        let backlog = take_backlog(8, 1);
        assert_eq!(backlog.len(), 2, "B's backlog survived A's mid-drain offer");
        assert_eq!(backlog[0].0, 2);
        // A newer EXEC of B supersedes the lane's packet: same task, fresh
        // backlog (the taken entries do not come back).
        offer_exec(&state, &host, 2, &exec_work_with_binds(&[3]));
        let backlog = take_backlog(8, 1);
        assert_eq!(
            backlog.len(),
            1,
            "the newer same-task offer replaced the lane"
        );
        // B runs: the lane retires with it.
        note_exec_finished(1);
        note_exec_started(2);
        note_exec_finished(2);
        assert!(
            take_backlog(8, 1).is_empty(),
            "the lane is retired after its task ran"
        );
        // And a lane-less offer for the still-running task never installs.
        note_exec_started(1);
        offer_exec(&state, &host, 1, &exec_work_with_binds(&[2]));
        assert!(
            take_backlog(8, 2).is_empty(),
            "the running task does not become a lane"
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
        offer_exec(&state, &host, 2, &exec_work_with_binds(&[2, 3]));
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
        prep_a.ahead(&state, &FakeHost::new(), 0);
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
        prep_b.ahead(&state, &FakeHost::new(), 0);
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
}
