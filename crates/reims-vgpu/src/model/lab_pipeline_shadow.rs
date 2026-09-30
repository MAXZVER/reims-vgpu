//! TEMP (lab, pipeline ids): what the ordering model would have decided if every
//! task's pipeline were its own.
//!
//! The pipeline table is keyed by an object-list id, the object list is per
//! task, and a driven desktop shares 140 to 170 ids between its tasks. This is a
//! second table, named per task, that every door of the real one also steps —
//! the same events in the same order — so the two can be compared at each place
//! the real one decides something:
//!
//! - **admission**: does a packet wait, run or get refused against the shared
//!   entry, and does it against its own task's (`lab_pp_admit_*`,
//!   `lab_pp_lease_*`);
//! - **release**: when one task's build finishes, whose waits does it end, and is
//!   the waiter's own pipeline built (`lab_pp_release_*`); the same for the
//!   packets a deletion or a refusal strands (`lab_pp_strand_*`);
//! - **reflection lookup**: does the walk answer a draw from a reflection its own
//!   task published, from another task's, or from none, and do the two differ
//!   (`lab_pp_usage_*`).
//!
//! It changes what the device does in exactly one way, and only while it is on:
//! the rail's "is this reflection published" question is answered for the asking
//! task, so a reflection is built for each task that binds the pipeline, and the
//! real table keeps the first one published (what it does without this).
//! On with `REIMS_VGPU_LAB_DAG` set and the real table not already per task;
//! counted while the DAG window is open.

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

use reims_vgpu_core::identity::{IngressOrdinal, ResourceId, SessionGeneration, TaskId};
use reims_vgpu_core::pipeline::{
    BindingUsage, Lease, PipelineId, PipelineState, PipelineTable, PublishTo, PublishedUsage,
    RefusalReason,
};
use reims_vgpu_core::session::Ended;
use reims_vgpu_protocol::render::ShaderStage;

use crate::runtime::drain::{note_store_route, note_store_route_n};

/// Whether the shadow table is kept.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_DAG").is_ok())
        && !crate::runtime::objects::pipeline_per_task()
}

/// Whether the census is counting: the DAG window is open.
fn counting() -> bool {
    crate::runtime::drain::lab_dag_window()
}

#[derive(Default)]
pub(crate) struct LabShadow {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// The pipelines, named per task.
    table: PipelineTable,
    /// Which tasks declared an object-list id.
    tasks_of: HashMap<ResourceId, BTreeSet<u32>>,
    /// The parked packets that lease pipelines: task and leases by ordinal.
    waiting: HashMap<IngressOrdinal, (u32, Vec<ResourceId>)>,
}

/// 0 ready, 1 wait, 2 refused or absent.
fn verdict(lease: Lease) -> u8 {
    match lease {
        Lease::Ready => 0,
        Lease::Pending => 1,
        Lease::Refused(_) | Lease::Absent(_) => 2,
    }
}

fn id(task: u32, name: ResourceId) -> PipelineId {
    PipelineId::of(TaskId(task), name)
}

impl std::fmt::Debug for LabShadow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LabShadow")
    }
}

impl Inner {
    fn state(&self, task: u32, name: ResourceId) -> Option<PipelineState> {
        self.table.get(id(task, name)).map(|p| p.state)
    }

    fn live_others(&self, task: u32, name: ResourceId) -> u64 {
        self.tasks_of.get(&name).map_or(0, |tasks| {
            tasks
                .iter()
                .filter(|&&t| {
                    t != task
                        && self
                            .state(t, name)
                            .is_some_and(|s| !s.is_terminal())
                })
                .count() as u64
        })
    }

    /// Classify the packets a shared entry's end or readiness reached.
    fn reached(&self, task: u32, name: ResourceId, waiters: &[IngressOrdinal], routes: [&'static str; 4]) {
        let [own, cross_ready, cross_early, unregistered] = routes;
        for w in waiters {
            match self.waiting.get(w) {
                None => note_store_route(unregistered),
                Some((wtask, _)) if *wtask == task => note_store_route(own),
                Some((wtask, _)) => {
                    if self.state(*wtask, name) == Some(PipelineState::Ready) {
                        note_store_route(cross_ready);
                    } else {
                        note_store_route(cross_early);
                        if crate::observe::first_sight(cross_early, u64::from(name.slot.0)) {
                            crate::observe::fail(format!(
                                "{cross_early} slot={} gen={} ended_by_task={task} waiter_task={wtask} \
                                 waiter_own_state={:?}",
                                name.slot.0,
                                name.generation.0,
                                self.state(*wtask, name).map(PipelineState::name),
                            ));
                        }
                    }
                }
            }
        }
    }
}

impl LabShadow {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn declare(&self, task: u32, name: ResourceId, generation: SessionGeneration) {
        if !enabled() {
            return;
        }
        let mut g = self.lock();
        g.tasks_of.entry(name).or_default().insert(task);
        g.table.declare(id(task, name), generation);
    }

    pub(crate) fn advance(&self, task: u32, name: ResourceId, next: PipelineState) {
        if !enabled() {
            return;
        }
        self.lock().table.advance(id(task, name), next);
    }

    /// `real_took`: the real table's withdrawal took.
    pub(crate) fn withdraw(&self, task: u32, name: ResourceId, real_took: bool) {
        if !enabled() {
            return;
        }
        let mut g = self.lock();
        if real_took && counting() {
            note_store_route(if g.state(task, name) == Some(PipelineState::Ready) {
                "lab_pp_withdraw_own"
            } else {
                "lab_pp_withdraw_foreign"
            });
            let demoted = g
                .tasks_of
                .get(&name)
                .map_or(0, |tasks| {
                    tasks
                        .iter()
                        .filter(|&&t| t != task && g.state(t, name) == Some(PipelineState::Ready))
                        .count() as u64
                });
            note_store_route_n("lab_pp_withdraw_demoted_others", demoted);
        }
        g.table.withdraw(id(task, name));
    }

    /// The real table's `Ready` step, with the packets it was holding.
    pub(crate) fn ready(&self, task: u32, name: ResourceId, waiters: &[IngressOrdinal], real_took: bool) {
        if !enabled() {
            return;
        }
        let mut g = self.lock();
        if real_took && counting() {
            g.reached(
                task,
                name,
                waiters,
                [
                    "lab_pp_release_own",
                    "lab_pp_release_cross_ready",
                    "lab_pp_release_cross_early",
                    "lab_pp_release_unregistered",
                ],
            );
        }
        g.table.advance(id(task, name), PipelineState::Ready);
    }

    pub(crate) fn retire(&self, task: u32, name: ResourceId, ended: &Ended) {
        if !enabled() {
            return;
        }
        let mut g = self.lock();
        if ended.took && counting() {
            note_store_route_n("lab_pp_retire_tombstoned_others", g.live_others(task, name));
            g.reached(
                task,
                name,
                &ended.stranded,
                [
                    "lab_pp_strand_own",
                    "lab_pp_strand_cross_ready",
                    "lab_pp_strand_cross_early",
                    "lab_pp_strand_unregistered",
                ],
            );
        }
        g.table.retire(id(task, name));
    }

    pub(crate) fn refuse(&self, task: u32, name: ResourceId, reason: RefusalReason, ended: &Ended) {
        if !enabled() {
            return;
        }
        let mut g = self.lock();
        if ended.took && counting() {
            note_store_route_n("lab_pp_refuse_spoils_others", g.live_others(task, name));
            g.reached(
                task,
                name,
                &ended.stranded,
                [
                    "lab_pp_refstrand_own",
                    "lab_pp_refstrand_cross_ready",
                    "lab_pp_refstrand_cross_early",
                    "lab_pp_refstrand_unregistered",
                ],
            );
        }
        g.table.refuse(id(task, name), reason);
    }

    pub(crate) fn publish(&self, task: u32, name: ResourceId, usage: &PublishedUsage) {
        if !enabled() {
            return;
        }
        self.lock()
            .table
            .publish_usage_to(id(task, name), usage.clone(), PublishTo::Live);
    }

    pub(crate) fn usage_published(&self, task: u32, name: ResourceId) -> bool {
        self.lock().table.usage_published(id(task, name))
    }

    pub(crate) fn usage_wanted(&self, task: u32, name: ResourceId) -> bool {
        self.lock()
            .table
            .usage_wanted_by(id(task, name), PublishTo::Live)
    }

    pub(crate) fn note_admitted(&self, ingress: IngressOrdinal, task: u32, leases: &[ResourceId]) {
        if !enabled() || leases.is_empty() {
            return;
        }
        self.lock().waiting.insert(ingress, (task, leases.to_vec()));
    }

    pub(crate) fn forget(&self, ingress: IngressOrdinal) {
        if !enabled() {
            return;
        }
        self.lock().waiting.remove(&ingress);
    }

    /// What the real table (`real`, one lease each) and this one say about a
    /// packet about to be admitted with `leases`.
    pub(crate) fn admission(&self, task: u32, leases: &[ResourceId], generation: SessionGeneration, real: &[Lease]) {
        if !enabled() || !counting() || leases.is_empty() {
            return;
        }
        let mut g = self.lock();
        let (mut worst_real, mut worst_own) = (0u8, 0u8);
        for (&lease, &shared) in leases.iter().zip(real) {
            let own = g.table.lease(id(task, lease), generation);
            let (s, t) = (verdict(shared), verdict(own));
            worst_real = worst_real.max(s);
            worst_own = worst_own.max(t);
            note_store_route("lab_pp_lease");
            if s != t {
                note_store_route("lab_pp_lease_diverge");
                note_store_route(match (s, t) {
                    (1, 0) => "lab_pp_lease_spurious_wait",
                    (0, 1) => "lab_pp_lease_early",
                    (2, 0) => "lab_pp_lease_spurious_refuse",
                    (0, 2) => "lab_pp_lease_missed_refuse",
                    (1, 2) => "lab_pp_lease_wait_vs_own_refuse",
                    (2, 1) => "lab_pp_lease_refuse_vs_own_wait",
                    _ => "lab_pp_lease_other",
                });
                if crate::observe::first_sight("lab_pp_lease_diverge", u64::from(lease.slot.0)) {
                    crate::observe::fail(format!(
                        "lab_pp_lease_diverge slot={} gen={} task={task} shared={shared:?} own={own:?}",
                        lease.slot.0, lease.generation.0
                    ));
                }
            }
        }
        note_store_route("lab_pp_admit");
        note_store_route(match (worst_real, worst_own) {
            (0, 0) => "lab_pp_admit_ready_ready",
            (0, 1) => "lab_pp_admit_ready_wait",
            (0, 2) => "lab_pp_admit_ready_refuse",
            (1, 0) => "lab_pp_admit_wait_ready",
            (1, 1) => "lab_pp_admit_wait_wait",
            (1, 2) => "lab_pp_admit_wait_refuse",
            (2, 0) => "lab_pp_admit_refuse_ready",
            (2, 1) => "lab_pp_admit_refuse_wait",
            _ => "lab_pp_admit_refuse_refuse",
        });
    }

    /// One walk lookup: what the real table answered and what this task's own
    /// reflection says.
    pub(crate) fn lookup(
        &self,
        task: u32,
        name: ResourceId,
        stage: Option<ShaderStage>,
        real: Option<&BindingUsage>,
    ) {
        if !enabled() || !counting() {
            return;
        }
        let g = self.lock();
        let own = g.table.binding_usage_of(id(task, name), stage);
        note_store_route("lab_pp_usage_lookup");
        match (real, own) {
            (None, None) => note_store_route("lab_pp_usage_none"),
            (Some(_), None) => note_store_route("lab_pp_usage_borrowed"),
            (None, Some(_)) => note_store_route("lab_pp_usage_missed"),
            (Some(shared), Some(own)) if shared == own => note_store_route("lab_pp_usage_same"),
            (Some(shared), Some(own)) => {
                note_store_route("lab_pp_usage_diff");
                let (under, over) = writes_differ(shared, own);
                if under {
                    // The shared answer says a slot is not written that this
                    // task's own reflection says is: an edge the walk did not
                    // draw.
                    note_store_route("lab_pp_usage_diff_under");
                    if crate::observe::first_sight("lab_pp_usage_diff_under", u64::from(name.slot.0)) {
                        crate::observe::fail(format!(
                            "lab_pp_usage_diff_under slot={} gen={} task={task} shared={shared:?} own={own:?}",
                            name.slot.0, name.generation.0
                        ));
                    }
                }
                if over {
                    note_store_route("lab_pp_usage_diff_over");
                }
            }
        }
    }
}

/// Whether some slot is written by `own` and not by `shared` (`.0`: the shared
/// answer orders less than the truth), and the reverse (`.1`).
fn writes_differ(shared: &BindingUsage, own: &BindingUsage) -> (bool, bool) {
    let writes = |m: Option<reims_vgpu_core::access::AccessMode>| m.is_some_and(|m| m.writes());
    let (mut under, mut over) = (false, false);
    for slot in 0..32u32 {
        let (s, o) = (writes(shared.buffer(slot)), writes(own.buffer(slot)));
        under |= o && !s;
        over |= s && !o;
    }
    for slot in 0..128u32 {
        let (s, o) = (writes(shared.texture(slot)), writes(own.texture(slot)));
        under |= o && !s;
        over |= s && !o;
    }
    (under, over)
}
