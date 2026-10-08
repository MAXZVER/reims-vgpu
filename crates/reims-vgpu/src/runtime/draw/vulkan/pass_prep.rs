//! TEMP (lab, `REIMS_VGPU_LAB_PASS_PREP=off|shadow|on`): draw-ahead preparation
//! of one render pass, in chunks, on the worker pool.
//!
//! Design: `lab/journal/2026-10-03-intra-packet-parallel-design.md` §8.1 (the
//! driver of the chunked fork-join; the coverage that makes it pay is steps
//! 2–4). `prefetch_stream` forks once per *stream*, before the draw loop, so
//! the recorder idles through the fork and the loop waits for nothing — the
//! S1 post-mortem's verdict. `PassPrep` instead forks every `K` draws *inside*
//! the loop: while a scope plans the binds of draws `[offered, offered + K)`,
//! the recorder is encoding the draws whose requests the loop already handed
//! over. That is the overlap this driver exists to prove
//! (`pass_prep_recorder_outstanding_n` at each scope's entry).
//!
//! The plans are the same [`SamplePlan`] / [`RefViewPlan`] the stream prefetch
//! uses, they live on the same drain-thread stack (one frame per pass), and
//! `push_tex` serves them through the same [`prefetch::prefetched_source`]:
//! `shadow` compares against the serial ladder and applies nothing, `on`
//! serves a validated plan, anything else runs the ladder unchanged. A
//! `off` mode is byte-for-byte the old path — neither this module nor the
//! frame it pushes exists then.
//!
//! # The consumed-ref replan (design H6, removed in step 2)
//!
//! The witness-backing plans (ref-view, the linear and mapper-ref gather
//! arms) move the gather witness when they apply; since step 2 each commit
//! hands back [`crate::runtime::gather_witness::PlannedGather::repeat`] — the
//! plan for the next bind of the same key, from the post-commit state — so
//! one plan per key serves the whole pass. A plan that *failed* validation
//! still leaves the ref in the frame's `consumed` list through
//! [`prefetch::prefetched_source`]; the next `ahead` takes that list and drops
//! those refs from `planned`, so a later chunk that binds the ref plans it
//! again from the witness's current state.
//!
//! # Step 3 — the buffer arms (design §8.3)
//!
//! The same chunks plan the draws' buffer binds and index binds beside the
//! sampled sources, and carry the pass's Store page walk
//! ([`PassPrep::note_store_target`] latches the colour-0 the drain's template
//! resolved; the next chunk's scope walks it). Those plans live in the same
//! frame and serve through [`super::buffer_plan`]'s asks at the points the
//! serial buffer ladder resolves — see that module for what is planned, what
//! bails, and the facts its commits land.

use super::buffer_plan;
use super::prefetch::{self, PrefetchGuard, StreamPlan};
use super::*;
use std::collections::HashSet;

/// TEMP (lab): `REIMS_VGPU_LAB_PREP_SYNC=on` — see [`PassPrep::ahead`]. Latched.
fn lab_prep_sync() -> bool {
    if cfg!(test) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("REIMS_VGPU_LAB_PREP_SYNC").as_deref(),
            Ok("on")
        )
    })
}

/// TEMP (lab): `REIMS_VGPU_LAB_SYNC_AHEAD` — see [`PassPrep::sync_ahead`]. On by
/// default (interleaved x2 on both hosts, scene modes pinned: AMD hq043 windowed
/// glass 161.7 / 148.9 Hz against 135.5 / 142.1 and full screen 138.9 / 126.7
/// against 117.3 / 121.7; Intel n224 188.9 / 194.3 against 188.0 / 185.6 and
/// 150.2 / 146.1 against 131.9 / 150.1; PassPrep helper work -30-48 %,
/// foreground hashing -30-50 %); `=off` restores syncing on the items alone.
/// TEMP (lab, `REIMS_VGPU_LAB_PREP_RESOLVE`, live `PREP_RESOLVE=on|off`, off
/// unless `on`): plan the sampled binds the walk left without a retained
/// resource by resolving it in [`PassPrep::ahead`] (b022), and replan a ref
/// whose bind found no plan (a bail stands for the whole pass otherwise:
/// b023 - ~300k binds a phase, all with a retained resource).
pub(super) fn lab_prep_resolve() -> bool {
    if cfg!(test) {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("PREP_RESOLVE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = matches!(std::env::var("REIMS_VGPU_LAB_PREP_RESOLVE").as_deref(), Ok("on"));
        crate::observe::off(format!("lab_prep_resolve on={}", u8::from(on)));
        on
    })
}

/// Latched; off in unit tests (the test hosts have no workers to hand sets to).
fn lab_sync_ahead() -> bool {
    if cfg!(test) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = !matches!(
            std::env::var("REIMS_VGPU_LAB_SYNC_AHEAD").as_deref(),
            Ok("off")
        );
        crate::observe::off(format!("lab_sync_ahead on={on}"));
        on
    })
}

/// TEMP (lab, `REIMS_VGPU_LAB_PACKED_WARM=on`, off by default): close the
/// `cold_packed` bail class — the planner's largest refusal family over the
/// fp9b transition window (`par_prefetch_bail_cold_packed`, 21.9k a window).
///
/// By code the class is not a construction the worker could have made: a
/// worker *cannot* construct — `ensure_packed_resource` needs
/// `host.map_pages`, which [`crate::qemu::host_ops::WorkerHost`] refuses, and
/// `insert_packed`, a write the plan-only contract forbids. The cost is the
/// bail's *stickiness*: `ahead` held every bailed ref in `planned` for the
/// whole pass ("the ladder below it would refuse the same way on the next
/// chunk") — true for every bail except a cold one whose registry entry does
/// not exist at all, because the drain's serial bind CONSTRUCTS that entry
/// (the packed alias, the whole-allocation walk) inside the same pass. So two
/// pieces, both read-only for the planner and legal for the drain:
///
/// * the **warm-up cork** ([`PassPrep::warm_packed`]): once per pass, on the
///   drain, before the first chunk forks, construct the packed aliases (and
///   the whole runs a refused alias falls to) of the first chunk's cold
///   references — the same constructions the first serial bind would run one
///   record later, moved ahead of the fork so chunk 0 plans onto a warm rail
///   instead of bailing;
/// * the **requeue**: a `ColdPacked`/`ColdWhole` bail over a registry with NO
///   entry for the reference is not held sticky — the next chunk replans it
///   (one retry per pass), after the drain's serial binds have constructed
///   the entry. A held entry (a refused construction, a mismatched geometry)
///   keeps the bail final exactly as before: the serial ladder mutates
///   nothing there, so the next chunk would refuse the same way.
///
/// Off is byte-for-byte the old path: the warm-up does nothing, no bail is
/// requeued, no `packed_warm_*` route fires.
pub(super) fn packed_warm() -> bool {
    // Unit tests drive the knob through their own door; the process default
    // stays off so an unrelated test never warms or requeues.
    #[cfg(test)]
    {
        return TEST_PACKED_WARM.load(std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(test))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            matches!(
                std::env::var("REIMS_VGPU_LAB_PACKED_WARM").as_deref(),
                Ok("on")
            )
        })
    }
}

#[cfg(test)]
static TEST_PACKED_WARM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Force the flag for the lifetime of the returned guard (tests only).
#[cfg(test)]
pub(super) fn test_packed_warm(on: bool) -> PackedWarmGuard {
    TEST_PACKED_WARM.store(on, std::sync::atomic::Ordering::Relaxed);
    PackedWarmGuard
}

/// Restores the test flag. Never outlives its guard's frame.
#[cfg(test)]
pub(super) struct PackedWarmGuard;

#[cfg(test)]
impl Drop for PackedWarmGuard {
    fn drop(&mut self) {
        TEST_PACKED_WARM.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Which way the pass preparation runs. `REIMS_VGPU_LAB_PASS_PREP`, read once;
/// supersedes `REIMS_VGPU_LAB_PAR_PREFETCH` while not `off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassPrepMode {
    Off,
    /// Plan each chunk, then run the serial ladder anyway and compare.
    Shadow,
    /// Serve binds from validated plans.
    On,
}

static MODE: std::sync::OnceLock<PassPrepMode> = std::sync::OnceLock::new();

#[cfg(test)]
static TEST_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub(crate) fn pass_prep_mode() -> PassPrepMode {
    #[cfg(test)]
    {
        // Tests cannot re-latch the environment's OnceLock per arm, so a
        // process-wide override stands in front of it; the guard `test_mode`
        // returns clears it, keeping the tests order-independent.
        match TEST_MODE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return PassPrepMode::Off,
            2 => return PassPrepMode::Shadow,
            3 => return PassPrepMode::On,
            // Unit tests keep the serial ladder unless they ask: the product
            // default (on) is for the running device, not for every test.
            _ => return PassPrepMode::Off,
        }
    }
    *MODE.get_or_init(
        || match std::env::var("REIMS_VGPU_LAB_PASS_PREP").as_deref() {
            // On by default (n102: fullscreen transitions 14-15 % frames >25 ms
            // against 21-22 % off, +15-22 % frames, sampled 16 against 21-22 us a
            // chain; drag and input latency neutral); `=off` restores the serial
            // ladder, `=shadow` plans and compares.
            Ok("off") => PassPrepMode::Off,
            Ok("shadow") => PassPrepMode::Shadow,
            _ => PassPrepMode::On,
        },
    )
}

/// Force the mode for the lifetime of the returned guard (tests only).
#[cfg(test)]
pub(crate) fn test_mode(mode: PassPrepMode) -> TestModeGuard {
    TEST_MODE.store(
        match mode {
            PassPrepMode::Off => 1,
            PassPrepMode::Shadow => 2,
            PassPrepMode::On => 3,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
    TestModeGuard
}

/// Clears the [`test_mode`] override. Never outlives its guard's frame.
#[cfg(test)]
pub(crate) struct TestModeGuard;

#[cfg(test)]
impl Drop for TestModeGuard {
    fn drop(&mut self) {
        TEST_MODE.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// TEMP (lab): `REIMS_VGPU_LAB_PASS_PREP_CHUNK=K` — draws per planning chunk.
/// Default 8 (the model's §7.2 assumed K=8: ~1.3 µs of fork per chunk against
/// the ~36 µs a recorder draw buys), clamped 2..=64.
pub(crate) fn chunk() -> usize {
    static CHUNK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        parse_chunk(
            std::env::var("REIMS_VGPU_LAB_PASS_PREP_CHUNK")
                .ok()
                .as_deref(),
        )
    })
}

/// The chunk size one environment value asks for: the default when the value
/// is absent or unparsable, clamped 2..=64 — a chunk of one draw cannot plan
/// ahead of the draw that needs it, and a chunk past 64 holds plans longer
/// than the recorder queue can balance.
fn parse_chunk(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(2, 64)
}

/// One draw's sampled tables, borrowed from the stream's `PendingDraw` list.
///
/// TEMP (lab, PASS_PREP step 3): the same tables carry the draw's buffer
/// binds and index bind, so one chunk plans the binds of both families.
pub(crate) struct DrawTables<'a> {
    pub fragment: &'a [TextureBind],
    pub vertex: &'a [TextureBind],
    /// TEMP (lab, PASS_PREP step 3): the draw's retained buffer tables.
    pub vertex_buffers: &'a [BufferBind],
    pub fragment_buffers: &'a [BufferBind],
    /// TEMP (lab, PASS_PREP step 3): the draw's index bind, when it drew
    /// indexed — the one buffer bind whose extent is a function of the
    /// record, and so the one the planner can key exactly.
    pub indexed: Option<&'a IndexedDrawInfo>,
}

impl DrawTables<'_> {
    /// The draw's buffer binds, vertex stage first (the order the drain's
    /// bind loop reads them in).
    fn buffers(&self) -> impl Iterator<Item = &BufferBind> + '_ {
        self.vertex_buffers
            .iter()
            .chain(self.fragment_buffers.iter())
    }
}

/// One planned bind candidate: the ref and the resource the walk retained.
enum PrepItem {
    /// A sampled-source bind (`plan_item`'s own ladder).
    Sample {
        texture_ref: u32,
        resource: std::sync::Arc<crate::model::TaskResource>,
    },
    /// TEMP (lab, `REIMS_VGPU_LAB_DRAIN_LANES=2`): another task's bind, from
    /// the plan-ahead lane's backlog — planned in this chunk's scope beside
    /// the pass's own items (the one window a worker may share
    /// `&DeviceState`), stored to the lane store after the join instead of
    /// the frame. Carries its own task because it is not this pass's.
    LaneSample {
        task_id: u32,
        texture_ref: u32,
        resource: std::sync::Arc<crate::model::TaskResource>,
    },
    /// TEMP (lab, PASS_PREP step 3): a buffer bind's warm rails. The offset
    /// is the first bind of the chunk that named the reference (PASS_STAGE:
    /// the staged arm reads one window, keyed by it).
    Buffer {
        buffer_ref: u32,
        resource: std::sync::Arc<crate::model::TaskResource>,
        first_offset: u64,
    },
    /// TEMP (lab, PASS_STAGE, design 8.4): a CPU load seed armed on a
    /// deferred draw now sitting in the recorder FIFO -- stage its bytes (R/B
    /// exchange folded) into the job's arena, for the recorder to find at the
    /// seed staging phase.
    SeedStage {
        cell: std::sync::Arc<crate::backend::vulkan::engine::pass_stage::SeedStageCell>,
    },
    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): a proven repeat plan (one a
    /// commit served a bind from) whose gather window the eager attach
    /// skipped as speculative — stage the window here, on this chunk's
    /// worker, and hand the staged twin to the merge.
    Restage { key: (u32, u32), plan: StreamPlan },
    /// TEMP (lab, PASS_PREP step 3): an indexed draw's index bind, planned
    /// under its exact `(ref, offset, need)` key.
    Index {
        key: buffer_plan::BufferExactKey,
        info: IndexedDrawInfo,
    },
    /// TEMP (lab, PASS_PREP step 3): the pass's Store page walk, for the key
    /// the drain latched from the pass's colour-0.
    StorePages { key: buffer_plan::StoreTargetKey },
}

/// What one planned item produced.
enum PrepOutcome {
    Sample(Result<StreamPlan, SampleBail>),
    Buffer(Result<buffer_plan::BufferContentPlan, buffer_plan::BufferBail>),
    /// The Store walk resolves pages or not; `None` is no plan, not a bail.
    Store(Option<buffer_plan::StorePagesPlan>),
    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): a restage job's staged twin of
    /// its item's plan — `None` when no window was carved (the entry stays as
    /// it was and the serial gather keeps serving its binds).
    Restage(Option<((u32, u32), StreamPlan)>),
}

/// Draw-ahead preparation of one render pass. Lives for the draw loop; the
/// guard drops the pass's plan frame when the loop does.
pub(crate) struct PassPrep<'a> {
    task_id: u32,
    tables: Vec<DrawTables<'a>>,
    chunk: usize,
    /// Draws `[0, offered)` have been offered to a scope (or skipped as small).
    offered: usize,
    /// Refs this pass holds a plan or a bail for; a consumed ref-view plan
    /// leaves it, so the next chunk replans the ref.
    planned: HashSet<u32>,
    /// TEMP (lab, PASS_PREP step 3): buffer references this pass holds a rail
    /// plan or a bail for.
    planned_buffers: HashSet<u32>,
    /// TEMP (lab, PASS_PREP step 3): index binds this pass holds an exact
    /// plan or a bail for, under the bind's own key.
    planned_indexes: HashSet<buffer_plan::BufferExactKey>,
    /// TEMP (lab, PACKED_WARM): sampled refs whose cold bail was requeued for
    /// one retry (see [`packed_warm`]) — the retry budget, one per pass.
    requeued_samples: HashSet<u32>,
    /// TEMP (lab, PACKED_WARM): buffer refs whose cold bail was requeued for
    /// one retry.
    requeued_buffers: HashSet<u32>,
    /// TEMP (lab, PACKED_WARM): index binds whose cold bail was requeued for
    /// one retry, under the bind's own key.
    requeued_indexes: HashSet<buffer_plan::BufferExactKey>,
    /// TEMP (lab, PASS_PREP step 3): the pass's colour-0 as the drain latched
    /// it once the attachment template existed, and whether a scope has taken
    /// the Store walk yet.
    store_target: Option<buffer_plan::StoreTargetKey>,
    store_pages_planned: bool,
    /// TEMP (lab, SYNC_AHEAD): the pass's later chunks have been handed to the
    /// host's sync workers.
    synced_ahead: bool,
    _frame: PrefetchGuard,
}

impl<'a> PassPrep<'a> {
    /// Open the pass's preparation. `None` — and the byte-for-byte old path —
    /// when the mode is off, the device has no pool, or the pass has fewer
    /// than two draws (a single draw cannot be planned ahead of itself).
    pub(crate) fn open(
        state: &DeviceState,
        task_id: u32,
        tables: Vec<DrawTables<'a>>,
    ) -> Option<Self> {
        let chunk = chunk();
        Self::open_with_chunk(state, task_id, tables, chunk)
    }

    /// [`Self::open`] with the chunk size named by the caller. Tests use this
    /// to pin K; production always goes through [`chunk()`].
    pub(super) fn open_with_chunk(
        state: &DeviceState,
        task_id: u32,
        tables: Vec<DrawTables<'a>>,
        chunk: usize,
    ) -> Option<Self> {
        if pass_prep_mode() == PassPrepMode::Off {
            return None;
        }
        if tables.len() < 2 {
            return None;
        }
        // The pool is fetched for its existence, not kept: `ahead` re-asks
        // `state.par()` at each chunk, so this struct borrows nothing from the
        // state it is handed.
        state.par()?;
        crate::runtime::drain::note_store_route("pass_prep_open");
        let mut opened = Self {
            task_id,
            tables,
            chunk,
            offered: 0,
            planned: HashSet::new(),
            planned_buffers: HashSet::new(),
            planned_indexes: HashSet::new(),
            requeued_samples: HashSet::new(),
            requeued_buffers: HashSet::new(),
            requeued_indexes: HashSet::new(),
            store_target: None,
            store_pages_planned: false,
            synced_ahead: false,
            _frame: prefetch::push_empty_frame(),
        };
        // TEMP (lab, `REIMS_VGPU_LAB_DRAIN_LANES=2`): the lane hand-off. A
        // lane planned this task's refs while another task drained; those
        // plans become the new frame's first entries (the frame this open
        // just pushed is the innermost — nested EXEC frames do not happen,
        // the preemption drains serve control channels only), and `planned`
        // seeds with them so this pass's own chunks do not replan what the
        // lane already holds. A plan the lane fills later (its backlog can
        // outlive this open) reaches the frame through the miss-fill in
        // `prefetched_source` instead.
        //
        // TEMP (lab, DRAIN_LANES 3): the refs a lane *attempted and bailed
        // on* seed `planned` too (n241: upper_rung is the lane's second
        // largest bail family, and the planner must keep refusing it — an
        // upper rung is state the serial rail resolves itself). This pass's
        // chunks skip re-attempting what refused one pass ago; the seed is
        // one-shot, so an epoch whose state has changed tries again fresh.
        if super::lanes::lanes_on() {
            if let Some(plans) = super::lanes::take_plans_for(task_id) {
                crate::runtime::drain::note_store_route("lanes_handoff_open");
                crate::runtime::drain::note_store_route_n("lanes_merged_n", plans.len() as u64);
                for (texture_ref, plan) in plans {
                    opened.planned.insert(texture_ref);
                    prefetch::merge_into_frame([((task_id, texture_ref), plan)]);
                }
            }
            let bailed = super::lanes::take_bailed_for(task_id);
            if !bailed.is_empty() {
                crate::runtime::drain::note_store_route_n("lanes_seed_n", bailed.len() as u64);
                opened.planned.extend(bailed);
            }
        }
        Some(opened)
    }

    /// TEMP (lab, PASS_PREP step 3): latch the pass's colour-0 for the Store
    /// page walk. The drain calls this once, after the attachment template
    /// exists (the template is built from the first record's request, after
    /// `ahead(0)` has already run); the next chunk's scope plans the walk.
    /// A pass small enough to live in one chunk never plans it — the walk is
    /// once per pass, and the drain does it in the time the store takes.
    pub(crate) fn note_store_target(&mut self, c0: &ColorRtRequest) {
        self.store_target = buffer_plan::latch_store_target(c0);
    }

    /// TEMP (lab, PACKED_WARM): the warm-up cork — see [`packed_warm`].
    ///
    /// Once per pass, on the drain thread (writes are legal here and only
    /// here), before the first chunk's scope forks: for every distinct
    /// reference the first chunk binds whose registry entry does not exist at
    /// all, run the construction the first serial bind of that reference would
    /// run one record later — `ensure_packed_resource` for the buffer and
    /// linear-texture families, and `ensure_whole_runs` where the packed alias
    /// refuses and the bind's window is gather-eligible (the serial ladder's
    /// own next step). The constructions are the same ones the drain pays
    /// either way; moving them ahead of the fork means chunk 0 plans onto a
    /// warm rail instead of bailing `cold_packed` sticky for the pass.
    ///
    /// The sampled family is pre-gated by its own planner —
    /// [`sample_plan::plan_linear_zero_copy`] runs read-only here and only
    /// where it answers `ColdPacked` does the warm-up construct — so a
    /// reference the resident rung or a debt would answer never pays an alias
    /// it will not use. The buffer family needs no such gate: the serial
    /// buffer ladder constructs unconditionally ahead of its rails.
    pub(crate) fn warm_packed<
        M: crate::runtime::host::HostMemory + crate::runtime::host::HostOps,
    >(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
    ) {
        if !packed_warm() || pass_prep_mode() != PassPrepMode::On {
            // `shadow` applies nothing — the warm-up's constructions are a
            // serving-mode optimization, and a shadow boot must stay
            // byte-for-byte the knob-off one it is compared against.
            return;
        }
        let task_id = self.task_id;
        let end = self.chunk.min(self.tables.len());
        let mut warmed = 0u64;
        // The buffer family: vertex/fragment binds plus the indexed draw's
        // index bind, deduplicated by reference. `first_offset` is the first
        // bind of the chunk that named the reference — the window the serial
        // ladder's whole-arm eligibility would be decided against.
        let mut seen_buffers = HashSet::new();
        for table in &self.tables[..end] {
            for bind in table.buffers() {
                if seen_buffers.insert(bind.buffer_ref) {
                    warmed += u64::from(warm_buffer_ref(
                        state,
                        host,
                        task_id,
                        bind.buffer_ref,
                        bind.resource.as_ref(),
                        bind.offset,
                    ));
                }
            }
            if let Some(info) = table.indexed {
                if seen_buffers.insert(info.index_buffer_ref) {
                    let resource = crate::runtime::objects::resolve_resource(
                        state,
                        host,
                        task_id,
                        info.index_buffer_ref,
                    )
                    .ok();
                    warmed += u64::from(warm_buffer_ref(
                        state,
                        host,
                        task_id,
                        info.index_buffer_ref,
                        resource.as_ref(),
                        info.index_buffer_offset,
                    ));
                }
            }
        }
        // The sampled family: linear textures only — the one sampled shape
        // whose planner ends in a packed probe (`plan_linear_zero_copy`'s
        // `packed_available` miss). The planner call is read-only and answers
        // through the same gates the chunk's item will meet, so the warm-up
        // constructs exactly where that item would have bailed no-entry cold.
        let mut seen_samples = HashSet::new();
        for table in &self.tables[..end] {
            for bind in table.fragment.iter().chain(table.vertex.iter()) {
                if bind.texture_ref == 0 || !seen_samples.insert(bind.texture_ref) {
                    continue;
                }
                let Some(resource) = bind.resource.clone() else {
                    continue;
                };
                if !matches!(
                    sample_plan::plan_linear_zero_copy(
                        state,
                        host,
                        task_id,
                        bind.texture_ref,
                        resource.clone(),
                        true,
                    ),
                    Err(SampleBail::ColdPacked)
                ) {
                    continue;
                }
                if state
                    .bound_buffers
                    .packed(task_id, bind.texture_ref)
                    .is_some()
                {
                    // A held entry (a refused construction, a mismatched
                    // geometry) — final for this geometry, the drain's own.
                    continue;
                }
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
                warmed += 1;
                ensure_packed_resource(
                    state,
                    host,
                    task_id,
                    bind.texture_ref,
                    &backing,
                    PackedResourceRail::LinearSample,
                );
            }
        }
        if warmed > 0 {
            crate::runtime::drain::note_store_route("packed_warm_open");
            crate::runtime::drain::note_store_route_n("packed_warm_refs", warmed);
        }
    }

    /// TEMP (lab, `REIMS_VGPU_LAB_SYNC_AHEAD=on`): hand the write-tracking sets of
    /// every sampled source past the first chunk to the host's sync workers, once,
    /// before the first chunk is planned — and do not wait for them.
    ///
    /// The planner's items are mostly that sync (pa2: 93 % of an item is the
    /// tracker bringing the set up to date — a content hash of its pages or a
    /// wait for another thread's), and an item runs it only when the drain is
    /// already waiting for the chunk. The sync needs nothing but the token, so it
    /// can start while the drain encodes the first chunk: the later chunks' items
    /// then find their sets synced for the epoch this pass's packet was rung in
    /// (one packet, one work epoch) and answer from the fast path. The first chunk
    /// is left to the pool, which is about to plan it. A hint only: a set a worker
    /// has not reached when its item runs is synced by the item, as before.
    fn sync_ahead<H: crate::runtime::host::HostOps>(&self, state: &DeviceState, host: &H) {
        if self.tables.len() <= self.chunk {
            return;
        }
        let mut seen = HashSet::new();
        let mut tokens = Vec::new();
        for table in &self.tables[self.chunk..] {
            for bind in table.fragment.iter().chain(table.vertex.iter()) {
                if bind.texture_ref == 0 || !seen.insert(bind.texture_ref) {
                    continue;
                }
                let Some(resource) = bind.resource.as_ref() else {
                    continue;
                };
                if let Some(token) =
                    sample_plan::plan_guest_write_token(state, bind.texture_ref, resource)
                {
                    tokens.push(token);
                }
            }
        }
        if tokens.is_empty() {
            return;
        }
        crate::runtime::drain::note_store_route("lab_sync_ahead_passes");
        crate::runtime::drain::note_store_route_n("lab_sync_ahead_tokens", tokens.len() as u64);
        host.guest_write_sync_ahead(&tokens);
    }

    /// At the top of record `di`: when `di >= offered`, plan the new refs of
    /// draws `[offered, offered + chunk)` on the pool and merge them into the
    /// pass's frame.
    ///
    /// The scope is fork-join, so this call is the drain's share of the
    /// planning and its wait for the helpers' — during which the recorder runs
    /// the draws the loop already submitted, which is the whole point.
    ///
    /// TEMP (lab, DRAIN_LANES): `HostMemory` beside `HostOps` because the
    /// lane's take resolves its items' resources through this host, on the
    /// drain, before the scope forks.
    pub(crate) fn ahead<H: crate::runtime::host::HostMemory + crate::runtime::host::HostOps>(
        &mut self,
        state: &DeviceState,
        host: &H,
        di: usize,
    ) {
        if di < self.offered {
            return;
        }
        if !self.synced_ahead {
            self.synced_ahead = true;
            if lab_sync_ahead() {
                self.sync_ahead(state, host);
            }
        }
        // Refs whose ref-view plan a bind consumed since the last chunk are
        // planned again from the witness's new state (design H6).
        for texture_ref in prefetch::take_consumed_refs() {
            self.planned.remove(&texture_ref);
        }
        let end = (self.offered + self.chunk).min(self.tables.len());
        let mut seen = HashSet::new();
        let mut items: Vec<PrepItem> = Vec::new();
        // TEMP (lab, PASS_PREP step 3): the same chunk plans the draws'
        // buffer rails and index binds beside the sampled sources.
        let mut seen_buffers = HashSet::new();
        let mut seen_indexes = HashSet::new();
        for table in &self.tables[self.offered..end] {
            for bind in table.fragment.iter().chain(table.vertex.iter()) {
                if bind.texture_ref == 0 || self.planned.contains(&bind.texture_ref) {
                    continue;
                }
                let resource = match bind.resource.clone() {
                    Some(resource) => resource,
                    // TEMP (lab, PREP_RESOLVE): a bind the walk did not retain
                    // a resource for was never planned, and its draws resolved
                    // it on the ladder instead - b022: 137k such binds a
                    // Safari phase (`prefetch_serial_nokey`), 3.2 us each on
                    // the target rung against 0.2 us served from a plan. The
                    // lookup is the same read-only one `push_tex` makes, once
                    // per ref per chunk here instead of once per bind there.
                    None if lab_prep_resolve() && !seen.contains(&bind.texture_ref) => {
                        match crate::runtime::objects::resolve_resource(
                            state,
                            host,
                            self.task_id,
                            bind.texture_ref,
                        ) {
                            Ok(resource) => {
                                crate::runtime::drain::note_store_route("pass_prep_resolved_unretained");
                                resource
                            }
                            Err(_) => {
                                crate::runtime::drain::note_store_route("pass_prep_unretained_unresolved");
                                continue;
                            }
                        }
                    }
                    None => continue,
                };
                if seen.insert(bind.texture_ref) {
                    items.push(PrepItem::Sample {
                        texture_ref: bind.texture_ref,
                        resource,
                    });
                }
            }
            for bind in table.buffers() {
                if bind.buffer_ref == 0 || self.planned_buffers.contains(&bind.buffer_ref) {
                    continue;
                }
                let Some(resource) = bind.resource.clone() else {
                    continue;
                };
                if seen_buffers.insert(bind.buffer_ref) {
                    items.push(PrepItem::Buffer {
                        buffer_ref: bind.buffer_ref,
                        resource,
                        first_offset: bind.offset,
                    });
                }
            }
            if let Some(info) = table.indexed {
                let Some(key) = buffer_plan::index_key(self.task_id, info) else {
                    continue;
                };
                if !self.planned_indexes.contains(&key) && seen_indexes.insert(key) {
                    items.push(PrepItem::Index {
                        key,
                        info: info.clone(),
                    });
                }
            }
        }
        // The Store walk, once per pass, once the drain has latched the
        // colour-0 key and a later chunk than the first exists to carry it.
        // It counts toward the fork gate like any other item: a scope of one
        // would run on the drain and be the serial walk with extra steps.
        let mut store_taken = false;
        if let (Some(key), false) = (self.store_target, self.store_pages_planned) {
            store_taken = true;
            items.push(PrepItem::StorePages { key });
        }
        // TEMP (lab, PASS_STAGE): the seeds the drain armed since the last
        // chunk -- cells of draws now sitting in the recorder FIFO. Only
        // `Pending` cells are worth a job; a spent one is dropped.
        let pending_seeds = pass_stage::take_pending_seeds();
        let mut seed_jobs = 0;
        for cell in &pending_seeds {
            if cell.is_pending() {
                seed_jobs += 1;
                items.push(PrepItem::SeedStage {
                    cell: std::sync::Arc::clone(cell),
                });
            }
        }
        crate::runtime::drain::note_store_route_n("pass_stage_seed_armed", seed_jobs);
        // TEMP (lab, PASS_STAGE, STAGED_PROVEN): the repeats a commit proved
        // since the last chunk — their windows are pool work now, the copy the
        // eager attach no longer pays for an unproven plan. Taken only while
        // the staging rail runs; with the rule off no plan is ever left
        // proven-but-unstaged, so the off arm collects nothing.
        if crate::runtime::par::staging::staged_proven()
            && crate::runtime::par::staging::enabled()
            && pass_prep_mode() == PassPrepMode::On
        {
            for (key, plan) in prefetch::restageable_plans() {
                items.push(PrepItem::Restage { key, plan });
            }
        }
        // TEMP (lab, `REIMS_VGPU_LAB_DRAIN_LANES=2`): the lane's backlog
        // rides this chunk's scope — the other task's refs plan beside this
        // pass's, inside the same fork-join (the one window a worker may
        // share `&DeviceState`; no new aliasing), bounded by the chunk size.
        // The take resolves the items' resources here, on the drain, through
        // this host — the construction the owning task's walk would pay
        // moments later anyway. Lane items count toward the fork gate below:
        // they are pool work exactly as this pass's items are. Not staged: a
        // lane plan outlives the job that would own the arena, so the gather
        // arms CPU-stage at the engine as they always did.
        let lane_items = super::lanes::take_backlog(state, host, self.chunk, self.task_id);
        let lane_n = lane_items.len() as u64;
        for (task_id, texture_ref, resource) in lane_items {
            items.push(PrepItem::LaneSample {
                task_id,
                texture_ref,
                resource,
            });
        }
        self.offered = end;
        if items.len() < 2 {
            // As with the stream prefetch: one ref gains nothing from the
            // pool, and planning it here would only double its resolve.
            crate::runtime::drain::note_store_route("pass_prep_chunk_small");
            return;
        }
        if store_taken {
            self.store_pages_planned = true;
        }
        let Some(pool) = state.par() else {
            return;
        };
        // TEMP (lab, `REIMS_VGPU_LAB_PREP_SYNC=on`): one merged sync of the sets
        // the items are about to ask about, instead of each item syncing its own
        // on its own thread and settling behind the others' queries (n156: 20 % of
        // the drain inside those items; n159: one merged list costs a third of the
        // hypervisor time of per-set queries).
        if lab_prep_sync() {
            let tokens: Vec<u64> = items
                .iter()
                .filter_map(|item| match item {
                    PrepItem::Sample {
                        texture_ref,
                        resource,
                    } => sample_plan::plan_guest_write_token(state, *texture_ref, resource),
                    _ => None,
                })
                .collect();
            if !tokens.is_empty() {
                let _span = crate::runtime::chain_phase::CostSpan::new("lab_prep_sync_us");
                crate::runtime::drain::note_store_route_n(
                    "lab_prep_sync_tokens",
                    tokens.len() as u64,
                );
                host.guest_write_sync_many(&tokens);
            }
        }
        crate::runtime::drain::note_store_route("pass_prep_chunk");
        crate::runtime::drain::note_store_route_n("pass_prep_items", items.len() as u64);
        // The overlap this driver exists to prove: draws still recording while
        // the pool plans the next chunk.
        crate::runtime::drain::note_store_route_n(
            "pass_prep_recorder_outstanding_n",
            u64::from(crate::backend::vulkan::engine::recorder_outstanding()),
        );
        let task_id = self.task_id;
        // The chunk's memo-fill budget (design 8.2: a byte limit per portion),
        // spent by whichever workers plan memo fills inside this scope.
        let memo_budget = std::sync::atomic::AtomicU64::new(sample_plan::memo_fill_chunk_limit());
        // TEMP (lab, DRAIN_LANES): the lane items' planning time, priced
        // where it ran (worker or coordinator — the results are counted
        // alike either way).
        let lane_planned_us = std::sync::atomic::AtomicU64::new(0);
        let scoped = pool.scope(state, &items, |st, env, item| {
            let mut host = env.host();
            // TEMP (lab, PASS_STAGE): the job's own arena, when the rail may
            // run on this thread.
            let stager = pass_stage::stager_for(env);
            match item {
                PrepItem::Sample {
                    texture_ref,
                    resource,
                } => {
                    let mut outcome = prefetch::plan_item(
                        st,
                        env,
                        task_id,
                        *texture_ref,
                        resource.clone(),
                        &memo_budget,
                    );
                    // TEMP (lab, PASS_STAGE): the gather arms this plan serves
                    // would CPU-stage at the engine; stage their windows
                    // here, on this thread.
                    if let Ok(plan) = &mut outcome {
                        if let Some(stager) = stager.as_ref() {
                            sample_plan::attach_staged(plan, stager);
                        }
                    }
                    PrepOutcome::Sample(outcome)
                }
                // TEMP (lab, DRAIN_LANES): the lane's item — the same ladder,
                // the other task's key, no staging (see the backlog take).
                PrepItem::LaneSample {
                    task_id,
                    texture_ref,
                    resource,
                } => {
                    let started = crate::lab_clock::LabInstant::now();
                    let outcome = prefetch::plan_item(
                        st,
                        env,
                        *task_id,
                        *texture_ref,
                        resource.clone(),
                        &memo_budget,
                    );
                    lane_planned_us.fetch_add(
                        started.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    PrepOutcome::Sample(outcome)
                }
                PrepItem::Buffer {
                    buffer_ref,
                    resource,
                    first_offset,
                } => PrepOutcome::Buffer(buffer_plan::plan_buffer_rails(
                    st,
                    &mut host,
                    task_id,
                    *buffer_ref,
                    resource.clone(),
                    *first_offset,
                    stager.as_ref(),
                )),
                PrepItem::Index { info, .. } => PrepOutcome::Buffer(
                    buffer_plan::plan_index_content(st, &mut host, task_id, info, stager.as_ref()),
                ),
                PrepItem::StorePages { key } => {
                    PrepOutcome::Store(buffer_plan::plan_store_pages(st, &host, task_id, *key))
                }
                PrepItem::SeedStage { cell } => {
                    if let Some(stager) = stager.as_ref() {
                        pass_stage::stage_seed_job(stager, cell);
                    }
                    PrepOutcome::Store(None)
                }
                PrepItem::Restage { key, plan } => {
                    // TEMP (lab, PASS_STAGE, STAGED_PROVEN): the proven
                    // repeat's window, copied into this worker's arena. The
                    // twin (not the item) is staged: the scope hands items by
                    // shared reference, and the merge below is what replaces
                    // the frame's entry — sound because a scope's fork-join
                    // freezes the frame, so the entry the twin replaces is
                    // exactly the plan this item was cloned from.
                    let mut twin = plan.clone();
                    if let Some(stager) = stager.as_ref() {
                        sample_plan::attach_staged(&mut twin, stager);
                    }
                    PrepOutcome::Restage(twin.has_staged_window().then_some((*key, twin)))
                }
            }
        });
        crate::runtime::drain::note_store_route_n("pass_prep_fork_us", scoped.fork_us);
        crate::runtime::drain::note_store_route_n("pass_prep_wait_us", scoped.wait_us);
        crate::runtime::drain::note_store_route_n("pass_prep_helper_us", scoped.helper_busy_us);
        // TEMP (lab, DRAIN_LANES): what the lane's items cost inside this
        // chunk's scope, beside the pass's own census above.
        if lane_n != 0 {
            crate::runtime::drain::note_store_route_n(
                "lanes_planned_us",
                lane_planned_us.load(std::sync::atomic::Ordering::Relaxed),
            );
        }
        // What the chunk's memo fills cost in bytes, against the budget above:
        // the memcpy this driver added to the pool, priced where it ran.
        crate::runtime::drain::note_store_route_n(
            "pass_prep_memo_fill_bytes",
            sample_plan::memo_fill_chunk_limit()
                - memo_budget.load(std::sync::atomic::Ordering::Relaxed),
        );
        let mut plans = Vec::with_capacity(items.len());
        // TEMP (lab, PASS_PREP step 3): the chunk's buffer plans, merged into
        // the frame beside the sampled ones.
        let mut buffer_rails = Vec::new();
        let mut buffer_exacts = Vec::new();
        let mut store_plan = None;
        for (item, result) in items.iter().zip(scoped.results) {
            // A bail is as final as a plan for this pass: the ladder below it
            // would refuse the same way on the next chunk, so it is not asked
            // again. A consumed ref-view plan leaves `planned` through the
            // consumed list, not here.
            //
            // TEMP (lab, PACKED_WARM): one bail is not final — a cold one over
            // a registry with NO entry for the reference (see [`packed_warm`]):
            // the drain's serial bind constructs that entry inside this pass,
            // so the ref is requeued for one retry on the next chunk instead
            // of held sticky. A held entry (a refused construction, a mismatched
            // geometry) keeps the bail final: the serial ladder mutates nothing
            // there and the next chunk would refuse the same way.
            match item {
                PrepItem::Sample { texture_ref, .. } => {
                    let requeue = packed_warm()
                        && matches!(
                            &result,
                            Ok(PrepOutcome::Sample(Err(SampleBail::ColdPacked)))
                        )
                        && state.bound_buffers.packed(task_id, *texture_ref).is_none()
                        && !self.requeued_samples.contains(texture_ref);
                    if requeue {
                        self.requeued_samples.insert(*texture_ref);
                        crate::runtime::drain::note_store_route("packed_warm_requeue");
                    } else {
                        self.planned.insert(*texture_ref);
                    }
                    match result {
                        Ok(PrepOutcome::Sample(Ok(plan))) => {
                            if self.requeued_samples.remove(texture_ref) {
                                crate::runtime::drain::note_store_route("packed_warm_hit");
                            }
                            crate::runtime::drain::note_store_route(match plan {
                                StreamPlan::Surface(_) => "par_prefetch_planned",
                                StreamPlan::RefView(_) => "par_prefetch_planned_ref_view",
                                StreamPlan::Linear(_) => "par_prefetch_planned_linear",
                                StreamPlan::MapperRef(_) => "par_prefetch_planned_mapper_ref",
                                StreamPlan::Memo(_) => "par_prefetch_planned_memo",
                            });
                            plans.push(((task_id, *texture_ref), plan));
                        }
                        Ok(PrepOutcome::Sample(Err(bail))) => {
                            crate::runtime::drain::note_store_route(prefetch::bail_route(bail))
                        }
                        _ => crate::runtime::drain::note_store_route("par_prefetch_panic"),
                    }
                }
                PrepItem::Buffer { buffer_ref, .. } => {
                    let bail = match &result {
                        Ok(PrepOutcome::Buffer(Err(bail))) => Some(*bail),
                        _ => None,
                    };
                    if packed_warm()
                        && bail
                            .is_some_and(|b| buffer_bail_requeues(state, task_id, *buffer_ref, b))
                        && !self.requeued_buffers.contains(buffer_ref)
                    {
                        self.requeued_buffers.insert(*buffer_ref);
                        crate::runtime::drain::note_store_route("packed_warm_requeue");
                    } else {
                        self.planned_buffers.insert(*buffer_ref);
                    }
                    match result {
                        Ok(PrepOutcome::Buffer(Ok(plan))) => {
                            if self.requeued_buffers.remove(buffer_ref) {
                                crate::runtime::drain::note_store_route("packed_warm_hit");
                            }
                            crate::runtime::drain::note_store_route(buffer_plan::planned_route(
                                &plan,
                            ));
                            // TEMP (lab, PASS_STAGE): a staged plan answers one
                            // exact bind; the rails answer every bind of the
                            // reference.
                            if plan.is_exact() {
                                buffer_exacts.push((
                                    (task_id, *buffer_ref, plan.exact_offset(), plan.exact_cap()),
                                    plan,
                                ));
                            } else {
                                buffer_rails.push(((task_id, *buffer_ref), plan));
                            }
                        }
                        Ok(PrepOutcome::Buffer(Err(bail))) => {
                            crate::runtime::drain::note_store_route(buffer_plan::bail_route(bail))
                        }
                        _ => crate::runtime::drain::note_store_route("pass_prep_buf_panic"),
                    }
                }
                PrepItem::Index { key, .. } => {
                    let bail = match &result {
                        Ok(PrepOutcome::Buffer(Err(bail))) => Some(*bail),
                        _ => None,
                    };
                    if packed_warm()
                        && bail.is_some_and(|b| buffer_bail_requeues(state, task_id, key.1, b))
                        && !self.requeued_indexes.contains(key)
                    {
                        self.requeued_indexes.insert(*key);
                        crate::runtime::drain::note_store_route("packed_warm_requeue");
                    } else {
                        self.planned_indexes.insert(*key);
                    }
                    match result {
                        Ok(PrepOutcome::Buffer(Ok(plan))) => {
                            if self.requeued_indexes.remove(key) {
                                crate::runtime::drain::note_store_route("packed_warm_hit");
                            }
                            crate::runtime::drain::note_store_route(buffer_plan::planned_route(
                                &plan,
                            ));
                            buffer_exacts.push((*key, plan));
                        }
                        Ok(PrepOutcome::Buffer(Err(bail))) => {
                            crate::runtime::drain::note_store_route(buffer_plan::bail_route(bail))
                        }
                        _ => crate::runtime::drain::note_store_route("pass_prep_buf_panic"),
                    }
                }
                PrepItem::StorePages { .. } => {
                    if let Ok(PrepOutcome::Store(Some(plan))) = result {
                        crate::runtime::drain::note_store_route("pass_prep_buf_planned_store");
                        store_plan = Some(plan);
                    }
                }
                // TEMP (lab, DRAIN_LANES): the lane's results go to the lane
                // store, not this pass's frame — the frame dies with the pass
                // and the lane's task has not opened its own yet. A bail is
                // counted in the lane's own namespace and recorded on the
                // lane (`note_bail`): the owning pass's open will seed its
                // `planned` from it (skip the re-attempt), and a cold bail
                // joins the set the task's next offer warms on the drain
                // (the PACKED_WARM requeue shape, n241's dominant bail
                // family).
                PrepItem::LaneSample {
                    task_id: lane_task,
                    texture_ref,
                    ..
                } => match result {
                    Ok(PrepOutcome::Sample(Ok(plan))) => {
                        crate::runtime::drain::note_store_route_n("lanes_planned_n", 1);
                        super::lanes::store_plan(*lane_task, *texture_ref, plan);
                    }
                    Ok(PrepOutcome::Sample(Err(bail))) => {
                        crate::runtime::drain::note_store_route(super::lanes::bail_route(bail));
                        crate::runtime::drain::note_store_route_n("lanes_bail_n", 1);
                        super::lanes::note_bail(*lane_task, *texture_ref, bail);
                    }
                    _ => crate::runtime::drain::note_store_route("lanes_panic"),
                },
                PrepItem::SeedStage { .. } => {
                    // The job's own census routes said what it did.
                }
                PrepItem::Restage { .. } => {
                    // TEMP (lab, PASS_STAGE, STAGED_PROVEN): the staged twin
                    // replaces the frame's entry (an identical plan plus its
                    // window); a refusal leaves the entry and its binds on the
                    // serial gather, retried by the next chunk that offers it.
                    if let Ok(PrepOutcome::Restage(Some(staged))) = result {
                        plans.push(staged);
                    }
                }
            }
        }
        prefetch::merge_into_frame(plans);
        buffer_plan::merge_buffer_plans(buffer_rails, buffer_exacts, store_plan);
    }
}

/// TEMP (lab, PACKED_WARM): whether a buffer-arm bail is the retryable cold —
/// the registry holds NO entry at all for the rail that bailed, so the drain's
/// serial bind constructs that entry (the packed alias, the whole-allocation
/// walk) inside this pass and the next chunk's replan can hit it. A held entry
/// — a refused construction under the same retirement rules, or a mismatched
/// geometry — makes the bail final exactly as before this knob existed: the
/// serial ladder mutates nothing there.
fn buffer_bail_requeues(
    state: &DeviceState,
    task_id: u32,
    buffer_ref: u32,
    bail: buffer_plan::BufferBail,
) -> bool {
    match bail {
        buffer_plan::BufferBail::ColdPacked => {
            state.bound_buffers.packed(task_id, buffer_ref).is_none()
        }
        buffer_plan::BufferBail::ColdWhole => {
            state.bound_buffers.whole(task_id, buffer_ref).is_none()
        }
        _ => false,
    }
}

/// TEMP (lab, PACKED_WARM): one buffer reference's warm-up — the construction
/// the first serial bind of the reference would run one record later, moved
/// ahead of the first chunk's fork. `first_offset` is that first bind's
/// offset, the window the whole-arm's gather eligibility is decided against.
///
/// Nothing happens where an entry already stands (warm, or a held negative the
/// drain answers without mutating) or where the backing does not resolve —
/// both are exactly the states the serial ladder leaves alone.
fn warm_buffer_ref<M: crate::runtime::host::HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    first_offset: u64,
) -> bool {
    if buffer_ref == 0 {
        return false;
    }
    if state.bound_buffers.packed(task_id, buffer_ref).is_some() {
        return false;
    }
    let Some(resource) = resource else {
        return false;
    };
    let Ok((gva, size)) =
        crate::runtime::objects::resolve_buffer_span_from_resource(state, resource)
    else {
        return false;
    };
    let backing = BufferBacking { gva, size };
    let made = ensure_packed_resource(
        state,
        host,
        task_id,
        buffer_ref,
        &backing,
        PackedResourceRail::Buffer,
    );
    if !made && first_offset < size && gather_span_if_eligible(size - first_offset, None).is_some()
    {
        // The serial ladder's own next step under a refused alias: walk the
        // whole allocation once and hold its runs.
        ensure_whole_runs(state, host, task_id, buffer_ref, gva, size);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::drain::census::store_route_count;
    use crate::runtime::host::FakeHost;
    use reims_vgpu_protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use std::sync::Arc;
    use std::time::Duration;

    fn delta(route: &str, before: u64) -> u64 {
        store_route_count(route).saturating_sub(before)
    }

    /// A device with a zero-helper pool: every scope runs on the caller, so a
    /// chunk's planning is deterministic and in order. The table claims stable
    /// page aliasing, as the x86 QEMU shim's does, so the zero-copy rungs the
    /// jobs plan through the worker host are the ones production sees.
    fn pooled_state() -> DeviceState {
        pooled_state_over_table(|table| {
            table.map_pages_stable = 1;
        })
    }

    /// [`pooled_state`] around a table the caller completes — the buffer
    /// fixtures add a `read_gpa` over a leaked host, so a pool job's
    /// `WorkerHost` reads the same guest memory the drain's host serves.
    fn pooled_state_over_table(
        fill: impl FnOnce(&mut crate::qemu::host_ops::ReimsVgpuHostOps),
    ) -> DeviceState {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut table = crate::qemu::host_ops::ReimsVgpuHostOps::null();
        unsafe extern "C" fn nothing(_ctx: *mut std::os::raw::c_void) {}
        table.thread_attach = Some(nothing);
        table.thread_detach = Some(nothing);
        fill(&mut table);
        let pool = Arc::new(
            crate::runtime::par::Pool::new(
                &table,
                &crate::runtime::par::Settings {
                    workers: 0,
                    worker_spin: Duration::ZERO,
                    join_spin: Duration::ZERO,
                },
            )
            .expect("the table offers the registration pair"),
        );
        state.install_par_pool(pool);
        state
    }

    /// `read_gpa` over a leaked host, for the tables the buffer-fixture pool
    /// reads through. The shim reads exactly what `HostMemory for FakeHost`
    /// reads; the drain owns the host and the pool only borrows it, as in the
    /// product where QEMU owns `ctx`.
    unsafe extern "C" fn shim_read_gpa(
        ctx: *mut std::os::raw::c_void,
        gpa: u64,
        buf: *mut u8,
        len: usize,
    ) -> i32 {
        // SAFETY: `ctx` is the leaked FakeHost the table was built with, alive
        // for the test's life; `buf` is the shim's own contract (valid for
        // `len` for the duration of the call).
        let host = unsafe { &*(ctx as *const FakeHost) };
        match host.read_gpa(gpa, unsafe { std::slice::from_raw_parts_mut(buf, len) }) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    }

    /// TEMP (lab, PASS_PREP step 3): a pooled device whose worker host reads
    /// the drain's own host — passed mutable here, with the table's `ctx`
    /// latched to its address (leaked by the caller so the address holds) —
    /// beside buffer objects at `(ref, size)` over allocations starting at
    /// task GVA 0x4000.
    fn pooled_buffer_state(host: &mut FakeHost, specs: &[(u32, u64)]) -> DeviceState {
        use crate::protocol::endian::st32;
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER,
        };
        use crate::runtime::gva_mem;

        let state = pooled_state_over_table(|table| {
            table.map_pages_stable = 1;
            table.read_gpa = Some(shim_read_gpa);
            table.ctx = host as *mut FakeHost as *mut std::os::raw::c_void;
        });
        let mut state = state;
        state.define_task(1, 0x1000, 2);
        assert!(state.set_object_list(1, 0, 64));
        for (n, &(buffer_ref, size)) in specs.iter().enumerate() {
            let handle = 4u32 + n as u32;
            let mut bdesc = [0u8; 16];
            bdesc[0..8].copy_from_slice(&size.to_le_bytes());
            st32(&mut bdesc[8..], handle);
            let desc_gva = 0x200u64 + 0x40 * n as u64;
            assert!(gva_mem::write_task_gva(
                host,
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
                host,
                &state.tasks[1],
                list_off,
                &list_entry,
                PAGE_SHIFT_X86
            )
            .is_ok());
        }
        state
    }

    /// The task, page table and guest RAM the buffer fixtures stand on: task 1
    /// with its directory at pfn 2, root at pfn 3, and task GVA pages 0..36
    /// mapped one-to-one onto GPA pages 16..52 of one contiguous range.
    fn buffer_host() -> FakeHost {
        use crate::protocol::endian::st32;
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

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
        host.map_range(0x10000, 0x24000, 0x40);
        for i in 0..36u32 {
            let pfn = 16 + i;
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
        }
        host
    }

    /// A complete buffer-fixture pair: the leaked host (the drain's, mutable)
    /// and the pooled state whose workers read it. The leak happens first so
    /// the table's `ctx` address holds for the device's life; the state is
    /// built through a reborrow, leaving the leaked reference for the drain.
    fn pooled_buffer_fixture(specs: &[(u32, u64)]) -> (DeviceState, &'static mut FakeHost) {
        let host: &'static mut FakeHost = Box::leak(Box::new(buffer_host()));
        let state = pooled_buffer_state(&mut *host, specs);
        (state, host)
    }

    /// TEMP (lab, PACKED_WARM): the linear-texture twin of
    /// [`pooled_buffer_fixture`] — a pooled device whose workers read the
    /// leaked host, with one OBJECT_TYPE_TEXTURE (`texture_ref` 2, a 64x64
    /// BGRA8 level at task GVA `0x20_000`, four pages) whose packed entry is
    /// deliberately NOT pre-warmed. [`warm_packed_entry`] stands in for the
    /// construction a capable host's drain would land at the first serial
    /// bind; this host's own `ensure_packed_resource` refuses (no import
    /// caps), which is exactly why the entry is inserted by hand.
    fn pooled_linear_fixture() -> (DeviceState, &'static mut FakeHost) {
        use crate::protocol::endian::{st16, st32, st64};
        use crate::runtime::decode::resource::{
            list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_LIST_ENTRY_LEN,
            OBJECT_TYPE_TEXTURE, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_HEIGHT,
            TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_WIDTH,
        };
        use crate::runtime::gva_mem;

        let mut leaked = buffer_host();
        // The worker's shim table carries no `write_gen`, so a witness token
        // would answer differently on the pool and the drain; unobservable
        // writes keep every witness unarmed and both views identical (the
        // `view_fixture` rule).
        leaked.guest_writes_unobservable = true;
        let host: &'static mut FakeHost = Box::leak(Box::new(leaked));
        let mut state = pooled_state_over_table(|table| {
            table.map_pages_stable = 1;
            table.read_gpa = Some(shim_read_gpa);
            table.ctx = host as *mut FakeHost as *mut std::os::raw::c_void;
        });
        state.define_task(1, 0x1000, 2);
        assert!(state.set_object_list(1, 0, 32));
        let (texture_ref, handle, w, h, bpr) = (2u32, 32u32, 64u32, 64u32, 256u32);
        let span = u64::from(bpr) * u64::from(h);
        let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
        st64(&mut desc[LINEAR_DESC_SIZE..], span);
        st32(&mut desc[LINEAR_DESC_HANDLE..], handle);
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], bpr);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], w);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], h);
        // RG16_UINT: an integer texel layout, so the sampled-layout gate the
        // planner asks answers from the table (`is_integer` short-circuits)
        // and no pool worker ever reaches for the engine lock a planning
        // thread must not take — and no test creates a Vulkan device just to
        // publish capabilities that would change what every later fixture's
        // packed construction refuses.
        st16(
            &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
            reims_vgpu_protocol::pixel_format::MTL_FORMAT_RG16_UINT,
        );
        assert!(gva_mem::write_task_gva(
            &mut *host,
            &state.tasks[1],
            0x1000,
            &desc,
            PAGE_SHIFT_X86
        )
        .is_ok());
        let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut list_entry,
            (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8),
        );
        list_entry[4..12].copy_from_slice(&0x1000u64.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            &mut *host,
            &state.tasks[1],
            list_off,
            &list_entry,
            PAGE_SHIFT_X86
        )
        .is_ok());
        (state, host)
    }

    /// TEMP (lab, PACKED_WARM): the warm packed allocation for
    /// [`pooled_linear_fixture`]'s texture — the same shape sample_plan's
    /// `linear_fixture` retains, over the fixture's own host allocation, so
    /// both the plan's commit and the serial rail's warm lookup answer from
    /// one retained identity.
    fn warm_packed_entry(state: &mut DeviceState) {
        use crate::runtime::bound_buffers::{PackedBuffer, PackedBufferResolution};
        let (texture_ref, handle, span) = (2u32, 32u32, 0x4000u64);
        let base = 0x7f00_0000_0000usize;
        let import = std::sync::Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(base, 0x8000, 0x1000)
                .expect("aligned allocation"),
        );
        let run =
            crate::runtime::guest_ram::GuestRun::in_mapping(base, 0x8000, 0, span).expect("run");
        let gpas: Vec<u64> = (0..4u64)
            .map(|i| (handle as u64 + i) << PAGE_SHIFT_X86)
            .collect();
        state.bound_buffers.insert_packed(
            1,
            texture_ref,
            PackedBufferResolution::Available(PackedBuffer {
                gva: u64::from(handle) << PAGE_SHIFT_X86,
                size: span,
                head: 0,
                import,
                gpas: std::sync::Arc::new(gpas),
                runs: std::sync::Arc::new(vec![run]),
                pages: std::sync::Arc::new(Vec::new()),
            }),
        );
    }

    /// TEMP (lab, PACKED_WARM): declare the texture's GVA plane — the state a
    /// freshly stored, now-sampled surface carries, and the one state in which
    /// the linear planner walks past its resident-rung guard to the packed
    /// probe (an undeclared GVA answers the guard conservatively, the
    /// `upper_rung` class this knob must not touch).
    ///
    /// The witness itself stays unarmed (the fixture's
    /// `guest_writes_unobservable`): the worker's shim table carries no
    /// `write_gen`, so a tracked token would answer `Unreadable` on the pool
    /// and `Quiet` on the drain — the split `view_fixture` avoids the same
    /// way. An unarmed witness answers `NoMapping` on both, which the guard
    /// reads as "the resident rung cannot serve" and walks past.
    fn declare_gva_plane(state: &mut DeviceState) {
        let gpas: Vec<u64> = (48u64..52).map(|p| p << PAGE_SHIFT_X86).collect();
        let _generation = state.pending_writebacks.ensure_gva_resource(
            crate::runtime::writeback_debt::GvaResourceKey {
                task_id: 1,
                texture_ref: 2,
            },
            0x20_000,
            0x4000,
            Some(gpas),
        );
    }

    /// Two `BufferContent`s agree structurally — the arms resolve
    /// independently, so run host pointers are not identity; lengths, source
    /// offsets and the pages' presence are the answer's shape.
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
            (BufferContent::Bytes(a), BufferContent::Bytes(b)) => a[..] == b[..],
            _ => false,
        }
    }

    /// The chunked driver's buffer arms: every bind a chunked `on` pass serves
    /// answers what the serial ladder answers for the same bind, and the
    /// registry is left where the serial ladder leaves it.
    #[test]
    fn chunked_buffer_binds_equal_serial() {
        // Staging is on by default now; this test pins the non-staged rails it
        // was written against (the staged twin is a_cold_bind_stages...).
        let _no_stage = crate::runtime::par::staging::test_enable(false);
        let _mode = test_mode(PassPrepMode::On);
        let specs = [(7u32, 0x8000u64), (8, 0x8000)];
        let (mut planned_state, planned_host) = pooled_buffer_fixture(&specs);
        let (mut serial_state, serial_host) = pooled_buffer_fixture(&specs);
        // One serial bind of each reference per arm primes the packed aliases
        // the rail plans borrow.
        for (state, host) in [
            (&mut planned_state, &mut *planned_host),
            (&mut serial_state, &mut *serial_host),
        ] {
            for &(buffer_ref, _) in &specs {
                let resource =
                    crate::runtime::objects::resolve_resource(&*state, &*host, 1, buffer_ref)
                        .expect("fixture resource");
                let _prime = load_buffer_content_serial(
                    state,
                    host,
                    1,
                    buffer_ref,
                    Some(&resource),
                    0,
                    true,
                    None,
                )
                .expect("the prime constructs the packed alias");
            }
        }
        let pattern = [0u32, 1, 0, 1, 0, 1, 0, 1];
        let make_binds = |state: &DeviceState, host: &FakeHost| -> Vec<Vec<BufferBind>> {
            pattern
                .iter()
                .map(|&which| {
                    let buffer_ref = specs[which as usize].0;
                    vec![BufferBind {
                        index: 0,
                        buffer_ref,
                        resource: Some(
                            crate::runtime::objects::resolve_resource(state, host, 1, buffer_ref)
                                .expect("bind resource"),
                        ),
                        offset: 0x40 * which as u64,
                        attribute_stride: None,
                    }]
                })
                .collect()
        };
        let planned_binds = make_binds(&planned_state, &*planned_host);
        let serial_binds = make_binds(&serial_state, &*serial_host);
        // The serial arm runs to completion first: the pass frame is a
        // thread-local, and both arms share this thread, so the serial binds
        // must not see the planned arm's plans.
        let serial_answers: Vec<_> = (0..pattern.len())
            .map(|di| {
                let serial_bind = serial_binds[di].first().expect("one bind per draw");
                load_buffer_content(
                    &mut serial_state,
                    &mut *serial_host,
                    1,
                    serial_bind.buffer_ref,
                    serial_bind.resource.as_deref(),
                    serial_bind.offset,
                    true,
                    None,
                )
                .expect("the serial ladder serves")
            })
            .collect();
        let tables: Vec<DrawTables<'_>> = planned_binds
            .iter()
            .map(|one| DrawTables {
                fragment: &[],
                vertex: &[],
                vertex_buffers: one.as_slice(),
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let before_planned = store_route_count("pass_prep_buf_planned_whole");
        let before_hit = store_route_count("pass_prep_buf_hit");
        let mut prep = PassPrep::open_with_chunk(&planned_state, 1, tables, 4).expect("opens");
        for di in 0..pattern.len() {
            prep.ahead(&planned_state, &FakeHost::new(), di);
            let bind = planned_binds[di].first().expect("one bind per draw");
            let served = load_buffer_content(
                &mut planned_state,
                &mut *planned_host,
                1,
                bind.buffer_ref,
                bind.resource.as_deref(),
                bind.offset,
                true,
                None,
            )
            .expect("the planned arm serves");
            assert!(
                contents_agree(&served, &serial_answers[di]),
                "draw {di} ref {} agrees with the serial ladder",
                bind.buffer_ref
            );
        }
        drop(prep);
        // No import caps on the fixture host, so the packed construction
        // refuses and the primes retain the whole runs: the chunks plan the
        // whole rail. (The packed rail is the same shape one boot later.)
        assert!(delta("pass_prep_buf_planned_whole", before_planned) >= 1);
        assert!(delta("pass_prep_buf_hit", before_hit) >= 1);
        assert_eq!(
            planned_state.bound_buffers.shape(),
            serial_state.bound_buffers.shape(),
            "the planned arm leaves the registry where the serial ladder does"
        );
    }

    /// The pass's Store page walk: latched from the colour-0 after the
    /// template, planned by a later chunk's scope, served at the storing
    /// draw's commit point, and equal to the serial walk's answer.
    #[test]
    fn the_pass_store_pages_plan_serves_the_storing_draw() {
        let _mode = test_mode(PassPrepMode::On);
        let specs = [(7u32, 0x8000u64), (8, 0x8000), (9, 0x8000)];
        let (mut state, host) = pooled_buffer_fixture(&specs);
        // Prime the aliases the chunk-0 items plan.
        for &(buffer_ref, _) in &specs[..2] {
            let resource = crate::runtime::objects::resolve_resource(&state, &*host, 1, buffer_ref)
                .expect("fixture resource");
            let _prime = load_buffer_content_serial(
                &mut state,
                &mut *host,
                1,
                buffer_ref,
                Some(&resource),
                0,
                true,
                None,
            )
            .expect("prime");
        }
        // Six draws: 0..2 bind refs 7, 8 (chunk 0's items), 2..4 bind ref 9
        // beside the store walk (chunk 1's), 4..6 repeat.
        let binds: Vec<Vec<BufferBind>> = [0usize, 1, 2, 2, 0, 1]
            .iter()
            .map(|&which| {
                let buffer_ref = specs[which].0;
                vec![BufferBind {
                    index: 0,
                    buffer_ref,
                    resource: Some(
                        crate::runtime::objects::resolve_resource(&state, &*host, 1, buffer_ref)
                            .expect("bind resource"),
                    ),
                    offset: 0,
                    attribute_stride: None,
                }]
            })
            .collect();
        let tables: Vec<DrawTables<'_>> = binds
            .iter()
            .map(|one| DrawTables {
                fragment: &[],
                vertex: &[],
                vertex_buffers: one.as_slice(),
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let c0 = ColorRtRequest {
            target_gva: 0x8000,
            row_stride: 256,
            width: 64,
            height: 4,
            format: MTL_FORMAT_BGRA8_UNORM,
            store_action: reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_STORE,
            ..Default::default()
        };
        let mut prep = PassPrep::open_with_chunk(&state, 1, tables, 2).expect("opens");
        prep.ahead(&state, &FakeHost::new(), 0);
        // The drain latches the colour-0 once the attachment template exists
        // — after the first chunk has already been offered.
        prep.note_store_target(&c0);
        let before_planned = store_route_count("pass_prep_buf_planned_store");
        prep.ahead(&state, &FakeHost::new(), 2);
        assert_eq!(delta("pass_prep_buf_planned_store", before_planned), 1);
        // The storing draw's commit point: the planned walk answers, with the
        // serial walk's own pages.
        let before_hit = store_route_count("pass_prep_buf_store_hit");
        let served = sync_store_allowed_pages(&state, &*host, 1, Some(&c0), true)
            .expect("the planned walk serves");
        assert_eq!(delta("pass_prep_buf_store_hit", before_hit), 1);
        let serial = crate::runtime::draw::sync_store_target_pages(&state, &*host, 1, &c0)
            .expect("the serial walk resolves the same pages");
        assert!(served.ordered == serial.ordered && served.span == serial.span);
        drop(prep);
    }

    /// The sample-plan fixture's shape, generalized to one ref-texture per
    /// surface: `refs[i]` names surface `71 + i`, 4x3 BGRA8, whose host cache
    /// holds one frame, so every bind plans onto the host-cache rung.
    fn multi_ref_fixture(refs: &[u32]) -> (DeviceState, FakeHost) {
        use crate::protocol::endian::st32;
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
        use crate::runtime::gva_mem;

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
        for (i, &texture_ref) in refs.iter().enumerate() {
            let surface_id = 71u32 + i as u32;
            let desc_gva = 0x1000u64 + 0x100 * i as u64;
            let built = reims_vgpu_wire::device_desc::RefTextureBuilder::new(surface_id, 0, 0, 0)
                .with_len(objects::TYPE5_MIN_LEN);
            assert!(gva_mem::write_task_gva(
                &mut host,
                &state.tasks[1],
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
                &state.tasks[1],
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
                [255u8, 0, (i as u8) * 40, 255].repeat(4 * 3),
            );
        }
        (state, host)
    }

    /// The per-draw fragment tables of `pattern`'s draws, in the fixture's
    /// retained-resource form; [`tables_of`] turns them into borrowed tables.
    fn binds_for(
        state: &DeviceState,
        host: &FakeHost,
        pattern: &[u32],
        refs: &[u32],
    ) -> Vec<Vec<TextureBind>> {
        pattern
            .iter()
            .map(|&which| {
                vec![TextureBind {
                    index: 0,
                    texture_ref: refs[which as usize],
                    resource: Some(
                        objects::resolve_resource(state, host, 1, refs[which as usize])
                            .unwrap_or_else(|e| {
                                panic!("fixture resource ref {}: {e:?}", refs[which as usize])
                            }),
                    ),
                }]
            })
            .collect()
    }

    /// The borrowed tables of per-draw bind vectors.
    fn tables_of(binds: &[Vec<TextureBind>]) -> Vec<DrawTables<'_>> {
        binds
            .iter()
            .map(|one| DrawTables {
                fragment: one.as_slice(),
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            })
            .collect()
    }

    /// K=4 over 20 draws and 5 refs: the first chunk plans three refs, the
    /// second the last two, and every later chunk finds nothing new — five
    /// plans for five refs, each planned exactly once, in at most five scopes.
    #[test]
    fn plans_each_ref_once_across_chunks() {
        let _mode = test_mode(PassPrepMode::On);
        let refs = [2u32, 3, 4, 5, 6];
        let pattern: Vec<u32> = vec![
            0, 1, 2, 0, // draws 0..4 (chunk 0): refs 2, 3, 4 are new
            3, 4, 3, 4, // draws 4..8 (chunk 1): refs 5, 6 are new
            0, 1, 2, 3, 4, 0, 1, 2, // draws 8..16: nothing new
            3, 4, 0, 1, // draws 16..20: nothing new
        ];
        assert_eq!(pattern.len(), 20);
        let (state, host) = multi_ref_fixture(&refs);
        let binds = binds_for(&state, &host, &pattern, &refs);
        let tables = tables_of(&binds);
        let before_planned = store_route_count("par_prefetch_planned");
        let before_chunk = store_route_count("pass_prep_chunk");
        let before_small = store_route_count("pass_prep_chunk_small");
        let mut prep = PassPrep::open_with_chunk(&state, 1, tables, 4).expect("the pass opens");
        for di in 0..20 {
            prep.ahead(&state, &crate::runtime::host::FakeHost::new(), di);
        }
        drop(prep);
        assert_eq!(
            delta("par_prefetch_planned", before_planned),
            5,
            "each of the five refs plans exactly once"
        );
        assert!(delta("pass_prep_chunk", before_chunk) <= 5);
        // The chunks with nothing new say so rather than forking.
        assert!(delta("pass_prep_chunk_small", before_small) >= 1);
    }

    /// Every bind a chunked `on` pass serves answers what the serial ladder
    /// answers for the same bind, and the fixture's observable state — the
    /// host cache each answer came from, the mappings' epochs, and (for the
    /// memo-bound refs) the memo itself — is left where the serial ladder
    /// leaves it.
    #[test]
    fn chunked_answers_equal_serial() {
        let _mode = test_mode(PassPrepMode::On);
        let refs = [2u32, 3, 4, 5, 6, 7, 8];
        // Draws bind the five host-cache refs (pattern 0..5 → refs 2..6) and
        // the two memo refs (pattern 5..6 → refs 7, 8) in a mix that repeats
        // every rung across chunk boundaries.
        let pattern: Vec<u32> = vec![
            0, 1, 2, 0, // draws 0..4 (chunk 0): refs 2, 3, 4 are new
            5, 3, 6, 4, // draws 4..8 (chunk 1): memo refs 7, 8 are new
            0, 1, 2, 5, 3, 6, 4, 5, // draws 8..16: repeats across chunks
            0, 6, 5, 1, // draws 16..20: more repeats
        ];
        assert_eq!(pattern.len(), 20);
        let (mut planned_state, mut planned_host) = multi_ref_fixture(&refs[..5]);
        let (mut serial_state, mut serial_host) = multi_ref_fixture(&refs[..5]);
        for (state, host) in [
            (&mut planned_state, &mut planned_host),
            (&mut serial_state, &mut serial_host),
        ] {
            host.stable_map_pages = true;
            host.strict_linux_map = true;
            add_memo_ref(state, host, 7, 81, 0x2800);
            add_memo_ref(state, host, 8, 82, 0x2900);
        }
        // One serial bind per memo ref on the planned arm primes what the memo
        // plan borrows (the contig refusal, the cached window, the token, the
        // memo entry itself); the serial arm primes itself on its own first
        // bind of each.
        for &memo_ref in &refs[5..] {
            let resource = objects::resolve_resource(&planned_state, &planned_host, 1, memo_ref)
                .expect("memo resource");
            let _prime = resolve_sampled_source(
                &mut planned_state,
                &mut planned_host,
                1,
                memo_ref,
                Some(resource),
                true,
            )
            .expect("the memo rung primes");
        }
        let planned_binds = binds_for(&planned_state, &planned_host, &pattern, &refs);
        let planned_tables = tables_of(&planned_binds);
        let mut prep =
            PassPrep::open_with_chunk(&planned_state, 1, planned_tables, 4).expect("opens");
        for di in 0..pattern.len() {
            prep.ahead(&planned_state, &crate::runtime::host::FakeHost::new(), di);
            let texture_ref = refs[pattern[di] as usize];
            let resource = objects::resolve_resource(&planned_state, &planned_host, 1, texture_ref)
                .expect("resource");
            let served = prefetch::prefetched_source(
                &mut planned_state,
                &mut planned_host,
                1,
                texture_ref,
                Some(&resource),
                true,
            );
            let serial = resolve_sampled_source(
                &mut serial_state,
                &mut serial_host,
                1,
                texture_ref,
                Some(resource.clone()),
                true,
            )
            .expect("the serial ladder serves");
            match served {
                prefetch::Prefetched::Serve((w, h, mid, src)) => {
                    let (sw, sh, smid, ssrc) = serial;
                    assert_eq!((w, h, mid), (sw, sh, smid), "ref {texture_ref} di {di}");
                    match (src, ssrc) {
                        (
                            SampledSourceRequest::Bytes(a, aid, al, ao),
                            SampledSourceRequest::Bytes(b, bid, bl, bo),
                        ) => {
                            assert!(Arc::ptr_eq(&a, &b) || a[..] == b[..]);
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
                                ),
                                "ref {texture_ref} di {di}"
                            );
                        }
                        pair => panic!(
                            "both arms serve bytes, got target/runs {}",
                            matches!(
                                pair,
                                (
                                    SampledSourceRequest::Target(..),
                                    SampledSourceRequest::Target(..)
                                )
                            )
                        ),
                    }
                }
                other => panic!(
                    "a planned bind serves at di {di}, got shadow={}",
                    matches!(other, prefetch::Prefetched::Shadow(_))
                ),
            }
        }
        drop(prep);
        for (i, _) in refs[..5].iter().enumerate() {
            let surface_id = 71u32 + i as u32;
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
        // The memo the two arms leave: same bytes, same generations.
        for (n, &surface_id) in [81u32, 82].iter().enumerate() {
            let key = (surface_id, 60, 64);
            let p = planned_state.mapper_ref_texture_memo.peek(&key);
            let s = serial_state.mapper_ref_texture_memo.peek(&key);
            let shape = |m: Option<&crate::model::GuestLinearMemo>| {
                m.map(|m| {
                    (
                        m.native.clone(),
                        m.rgba[..].to_vec(),
                        m.generation,
                        m.layout,
                    )
                })
            };
            assert_eq!(shape(p), shape(s), "memo state {surface_id} #{n}");
            assert!(p.is_some(), "the memo rung filled {surface_id}");
        }
    }

    /// One memo-bound ref (`texture_ref` over `surface_id`, its descriptor at
    /// `desc_gva`): 60x64 BGRA8 over four scattered pages with no host cache
    /// and no resident, plus a mapping write token — the shape whose binds the
    /// guest-memo rung serves once a serial bind has primed it.
    fn add_memo_ref(
        state: &mut DeviceState,
        host: &mut FakeHost,
        texture_ref: u32,
        surface_id: u32,
        desc_gva: u64,
    ) {
        use crate::protocol::endian::st32;
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
        use crate::runtime::gva_mem;
        let built = reims_vgpu_wire::device_desc::RefTextureBuilder::new(surface_id, 0, 0, 0)
            .with_len(objects::TYPE5_MIN_LEN);
        assert!(gva_mem::write_task_gva(
            host,
            &state.tasks[1],
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
            host,
            &state.tasks[1],
            list_off,
            &list_entry,
            PAGE_SHIFT_X86
        )
        .is_ok());
        assert!(state.map_surface(surface_id));
        assert!(state.set_mapping_geom(surface_id, 60, 64, MTL_FORMAT_BGRA8_UNORM));
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpas: Vec<u64> = (0..4u64).map(|i| 0x6000_0000 + 2 * i * page).collect();
        for (n, &gpa) in gpas.iter().enumerate() {
            host.map_range(gpa, page as usize, 0x60 + n as u8);
        }
        state.mappings.get_mut(&surface_id).unwrap().page_entries = gpas
            .iter()
            .map(|&gpa| {
                (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
        let _ = crate::runtime::mapper::ensure_guest_write_token(state, host, surface_id);
    }

    /// A mapping whose generation moved between the chunk that planned it and
    /// the bind that takes it is dropped (`par_prefetch_invalid_mapping`) and
    /// the serial ladder serves the bind.
    #[test]
    fn moved_mapping_falls_back() {
        let _mode = test_mode(PassPrepMode::On);
        let refs = [2u32, 3];
        let pattern: Vec<u32> = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let (mut state, mut host) = multi_ref_fixture(&refs);
        let binds = binds_for(&state, &host, &pattern, &refs);
        let tables = tables_of(&binds);
        let mut prep = PassPrep::open_with_chunk(&state, 1, tables, 4).expect("opens");
        prep.ahead(&state, &crate::runtime::host::FakeHost::new(), 0);
        // The guest rewires surface 71 between the chunk and the bind.
        state.mappings.get_mut(&71u32).unwrap().map_generation += 1;
        let texture_ref = refs[0];
        let resource = objects::resolve_resource(&state, &host, 1, texture_ref).expect("resource");
        let before = store_route_count("par_prefetch_invalid_mapping");
        let served = prefetch::prefetched_source(
            &mut state,
            &mut host,
            1,
            texture_ref,
            Some(&resource),
            true,
        );
        assert_eq!(delta("par_prefetch_invalid_mapping", before), 1);
        match served {
            prefetch::Prefetched::Serial => {}
            _ => panic!("a moved mapping falls back to the serial ladder"),
        }
        // And the fallback serves: the serial ladder answers for the bind.
        let serial =
            resolve_sampled_source(&mut state, &mut host, 1, texture_ref, Some(resource), true);
        assert!(serial.is_some(), "the serial ladder still serves the bind");
        drop(prep);
    }

    /// A ref-view plan's commit hands back the gather repeat, so the second
    /// bind of the same ref in the pass serves from the post state without a
    /// replan (design H6's removal) — and no later chunk needs to plan the ref
    /// again while the witness holds.
    #[test]
    fn a_repeated_key_serves_the_pass_from_the_post_state() {
        let _mode = test_mode(PassPrepMode::On);
        // The view fixture's shape plus two host-cache refs, so every chunk
        // that matters carries two new refs and forks: the view ref beside a
        // cache ref.
        let (mut state, mut host, view_ref, _) = view_fixture();
        let view_resource = objects::resolve_resource(&state, &host, 1, view_ref).expect("view");
        // Arm the witness and fill the window cache, as a first serial bind
        // does; from here the ref-view rung plans.
        let _arm = resolve_sampled_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(view_resource.clone()),
            true,
        )
        .expect("arm the witness");
        let cache_ref = 7;
        let cache2_ref = 8;
        add_cache_ref(&mut state, &mut host, cache_ref, 99, 0x2000, 0);
        add_cache_ref(&mut state, &mut host, cache2_ref, 91, 0x2100, 5);
        let cache_resource = objects::resolve_resource(&state, &host, 1, cache_ref).expect("cache");
        let cache2_resource =
            objects::resolve_resource(&state, &host, 1, cache2_ref).expect("cache2");
        let pattern: Vec<u32> = vec![0, 1, 1, 1, 0, 2, 1, 1];
        let per_draw: Vec<Vec<TextureBind>> = pattern
            .iter()
            .map(|&which| {
                let (texture_ref, resource) = match which {
                    0 => (view_ref, view_resource.clone()),
                    1 => (cache_ref, cache_resource.clone()),
                    _ => (cache2_ref, cache2_resource.clone()),
                };
                vec![TextureBind {
                    index: 0,
                    texture_ref,
                    resource: Some(resource),
                }]
            })
            .collect();
        let tables: Vec<DrawTables<'_>> = tables_of(&per_draw);
        let before_planned = store_route_count("par_prefetch_planned_ref_view");
        let before_hit = store_route_count("par_prefetch_hit_ref_view");
        let mut prep = PassPrep::open_with_chunk(&state, 1, tables, 2).expect("opens");
        // Chunk 0 = draws 0..2: both refs are new, so the chunk forks and the
        // view ref plans.
        prep.ahead(&state, &crate::runtime::host::FakeHost::new(), 0);
        assert_eq!(delta("par_prefetch_planned_ref_view", before_planned), 1);
        // Draw 0 commits the view plan: the frame entry becomes the repeat.
        let served = prefetch::prefetched_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(&view_resource),
            true,
        );
        assert!(
            matches!(served, prefetch::Prefetched::Serve(_)),
            "the first chunk's view plan serves"
        );
        assert_eq!(delta("par_prefetch_hit_ref_view", before_hit), 1);
        // Chunk 1 = draws 2..4: binds the cache ref only (nothing new); the
        // view ref's repeat stays in the frame untouched.
        prep.ahead(&state, &crate::runtime::host::FakeHost::new(), 2);
        assert_eq!(
            delta("par_prefetch_planned_ref_view", before_planned),
            1,
            "no bind asked for the view ref, so no replan"
        );
        // Chunk 2 = draws 4..6: binds the view ref again — the repeat the
        // first commit left serves it from the post state, with no replan.
        prep.ahead(&state, &crate::runtime::host::FakeHost::new(), 4);
        assert_eq!(
            delta("par_prefetch_planned_ref_view", before_planned),
            1,
            "the repeat covers the later bind; the ref is not replanned"
        );
        let again = prefetch::prefetched_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(&view_resource),
            true,
        );
        assert!(
            matches!(again, prefetch::Prefetched::Serve(_)),
            "the repeated view plan serves the second bind"
        );
        assert_eq!(delta("par_prefetch_hit_ref_view", before_hit), 2);
        drop(prep);
    }

    /// Fill the view fixture's 32 mapping pages with a per-page pattern.
    /// The pages start at `map_range`'s flat 0x33 fill, which would make a
    /// wrong-window copy hard to tell from a right one; a per-page pattern
    /// names the page a byte came from, so the staged/serial compares below
    /// are deterministic *and* diagnostic.
    fn fill_view_pages(host: &mut FakeHost) {
        let page = 1u64 << PAGE_SHIFT_X86;
        for i in 0..32u64 {
            let gpa = 0x5000_0000 + 2 * i * page;
            let fill = vec![((i * 7 + 1) % 251) as u8; page as usize];
            assert!(host.write_gpa(gpa, &fill).is_ok());
        }
    }

    /// The serial CPU gather over a guest-runs answer — the bytes
    /// `write_staging_from_runs` (and a staged window) must reproduce.
    fn gathered_bytes(src: &crate::backend::vulkan::engine::GuestRunSource) -> Vec<u8> {
        let mut out = vec![0u8; src.total_len as usize];
        let mut off = 0usize;
        let mut skip = src.source_offset;
        for run in src.runs.iter() {
            if off >= out.len() {
                break;
            }
            if skip >= run.len() {
                skip -= run.len();
                continue;
            }
            let within = skip as usize;
            skip = 0;
            let available = (run.len() as usize).saturating_sub(within);
            let n = available.min(out.len() - off);
            // SAFETY: the run's host pointer is the fixture's stable host
            // range, valid for the test's life.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (run.host_ptr() as *const u8).add(within),
                    out.as_mut_ptr().add(off),
                    n,
                );
            }
            off += n;
        }
        out
    }

    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): the rule, end to end on the
    /// ref-view rail. The FIRST bind of a key serves the plain guest-runs
    /// request — no window was copied for a plan that may die unbound, and
    /// the engine gathers serially exactly as it did before the rail existed.
    /// The commit proves the plan; the next chunk's restage job copies the
    /// window on the worker (a pool item beside the chunk's own planning);
    /// and every later bind of the key serves the staged request, whose bytes
    /// are the serial gather and whose guest-runs fields are the serial
    /// answer's own. The frame's pop releases the one window.
    #[test]
    fn a_repeat_plan_stages_but_the_first_bind_stays_serial() {
        let _stage = crate::runtime::par::staging::test_enable(true);
        let _proven = crate::runtime::par::staging::test_staged_proven(true);
        let _mode = test_mode(PassPrepMode::On);
        // The run memo is process-global and keyed by GPAs alone; without a
        // clear it would hand this test an earlier test's runs — host pointers
        // into a fixture that has already been dropped.
        run_memo_clear_for_test();
        // The view's window is 256x128 BGRA8 = 128 KiB: chunk the arena to
        // 256 KiB so the one window fits.
        crate::backend::vulkan::engine::pass_stage::install_test_arenas(1, 256 * 1024, 2);
        let (mut state, mut host, view_ref, _) = view_fixture();
        fill_view_pages(&mut host);
        let view_resource = objects::resolve_resource(&state, &host, 1, view_ref).expect("view");
        // Arm the witness and fill the window cache, as a first serial bind
        // does; from here the ref-view rung plans.
        let _arm = resolve_sampled_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(view_resource.clone()),
            true,
        )
        .expect("arm the witness");
        // The serial twin: same fixture, same arm, and its next answer is the
        // ladder the staged window must reproduce.
        let (mut serial_state, mut serial_host, _, _) = view_fixture();
        fill_view_pages(&mut serial_host);
        let serial_resource =
            objects::resolve_resource(&serial_state, &serial_host, 1, view_ref).expect("view");
        let _serial_arm = resolve_sampled_source(
            &mut serial_state,
            &mut serial_host,
            1,
            view_ref,
            Some(serial_resource.clone()),
            true,
        )
        .expect("arm the serial twin");
        let serial = resolve_sampled_source(
            &mut serial_state,
            &mut serial_host,
            1,
            view_ref,
            Some(serial_resource),
            true,
        )
        .expect("the serial ladder serves");
        let serial_src = match &serial.3 {
            SampledSourceRequest::GuestRuns(src, ..) => src,
            _ => panic!("the armed view serves guest runs on the serial ladder"),
        };
        let serial_bytes = gathered_bytes(serial_src);
        // Two cache refs beside the view so the chunks that matter fork.
        add_cache_ref(&mut state, &mut host, 7, 99, 0x2000, 0);
        add_cache_ref(&mut state, &mut host, 8, 91, 0x2100, 5);
        let bind_for = |texture_ref: u32, state: &DeviceState, host: &FakeHost| -> TextureBind {
            TextureBind {
                index: 0,
                texture_ref,
                resource: Some(
                    objects::resolve_resource(state, host, 1, texture_ref).expect("bind resource"),
                ),
            }
        };
        // V = the view ref; C/C2 = the cache refs. Chunk 2 carries C2 beside
        // the view's restage; chunk 4 carries nothing new.
        let pattern = [0u32, 1, 0, 2, 0, 0];
        let per_draw: Vec<Vec<TextureBind>> = pattern
            .iter()
            .map(|&which| {
                let texture_ref = match which {
                    0 => view_ref,
                    1 => 7,
                    _ => 8,
                };
                vec![bind_for(texture_ref, &state, &host)]
            })
            .collect();
        let tables: Vec<DrawTables<'_>> = tables_of(&per_draw);
        let before_skip = store_route_count("pass_stage_sampled_skipped_speculative");
        let before_skip_b = store_route_count("pass_stage_sampled_skipped_bytes");
        let before_staged = store_route_count("pass_stage_sampled_staged_proven");
        let before_planned = store_route_count("pass_stage_sampled_planned");
        let mut prep = PassPrep::open_with_chunk(&state, 1, tables, 2).expect("opens");
        // Chunk 0 (draws 0..2): the view plans FRESH, so nothing is copied —
        // the skip the rule exists to make, measured in windows and bytes.
        prep.ahead(&state, &FakeHost::new(), 0);
        assert!(delta("pass_stage_sampled_skipped_speculative", before_skip) >= 1);
        assert!(
            delta("pass_stage_sampled_skipped_bytes", before_skip_b) >= 128 * 1024,
            "the view's 128 KiB window was not copied"
        );
        assert_eq!(
            delta("pass_stage_sampled_staged_proven", before_staged),
            0,
            "no window yet: nothing is proven"
        );
        // Draw 0 binds the view: the plan serves, and the serve is the plain
        // guest-runs request — the first bind stays serial at the engine.
        let first = prefetch::prefetched_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(&view_resource),
            true,
        );
        match first {
            prefetch::Prefetched::Serve((_, _, _, SampledSourceRequest::GuestRuns(src, ..))) => {
                assert!(
                    Arc::ptr_eq(&src.runs, &serial_src.runs) || src.runs[..] == serial_src.runs[..],
                    "the first bind is the serial ladder's own guest-runs serve"
                );
                assert_eq!(src.total_len, serial_src.total_len);
            }
            other => panic!("unexpected serve variant"),
        }
        // Chunk 2 (draws 2..4): C2 is new beside the view's proven repeat —
        // the restage job stages the repeat's window on the pool.
        prep.ahead(&state, &FakeHost::new(), 2);
        assert!(
            delta("pass_stage_sampled_staged_proven", before_staged) >= 1,
            "the repeat's window was staged on the chunk's worker"
        );
        assert!(delta("pass_stage_sampled_planned", before_planned) >= 1);
        // Draw 2 binds the view again: the staged request, byte-equal to the
        // serial gather and field-equal to the serial answer's own source.
        let second = prefetch::prefetched_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(&view_resource),
            true,
        );
        match second {
            prefetch::Prefetched::Serve((_, _, _, SampledSourceRequest::Staged(req))) => {
                // SAFETY: the fact is alive for this borrow.
                let staged = unsafe { req.staged.bytes() };
                assert_eq!(
                    staged,
                    &serial_bytes[..],
                    "the staged window is the serial CPU gather"
                );
                assert!(
                    Arc::ptr_eq(&req.src.runs, &serial_src.runs)
                        || req.src.runs[..] == serial_src.runs[..]
                );
                assert_eq!(
                    (
                        req.src.source_offset,
                        req.src.total_len,
                        req.src.row_length_texels
                    ),
                    (
                        serial_src.source_offset,
                        serial_src.total_len,
                        serial_src.row_length_texels
                    )
                );
            }
            other => panic!("unexpected serve variant"),
        }
        // Chunk 4 (draws 4..6): nothing new and nothing to restage (the
        // repeat carries its window now); the later binds keep serving it.
        prep.ahead(&state, &FakeHost::new(), 4);
        match prefetch::prefetched_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(&view_resource),
            true,
        ) {
            // The repeat's clone carries the window: the later binds of the
            // key serve staged too (the value dies with this statement, so
            // the window's one reference is the frame's own below).
            prefetch::Prefetched::Serve((_, _, _, SampledSourceRequest::Staged(_))) => {}
            _ => panic!("the repeat carries its window to the later binds"),
        }
        // The frame holds the one window; its pop is what releases it.
        let arenas = crate::backend::vulkan::engine::pass_stage::arenas().expect("installed");
        assert!(arenas[0].ring.in_flight() >= 1);
        drop(prep);
        assert_eq!(
            arenas[0].ring.in_flight(),
            0,
            "the frame's pop releases the staged window"
        );
        drop(
            crate::backend::vulkan::engine::pass_stage::take_test_arenas()
                .expect("the test arenas are still installed"),
        );
    }

    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): the A/B arm. With the rule off
    /// the eager attach returns: the FIRST bind's plan is already staged at
    /// its chunk, the first bind serves the staged window, and no
    /// speculative-skip census fires — the behaviour the queues compare
    /// against.
    #[test]
    fn staged_proven_off_restores_the_eager_first_bind() {
        let _stage = crate::runtime::par::staging::test_enable(true);
        let _proven = crate::runtime::par::staging::test_staged_proven(false);
        let _mode = test_mode(PassPrepMode::On);
        // The run memo is process-global and keyed by GPAs alone; without a
        // clear it would hand this test an earlier test's runs — host pointers
        // into a fixture that has already been dropped.
        run_memo_clear_for_test();
        crate::backend::vulkan::engine::pass_stage::install_test_arenas(1, 256 * 1024, 2);
        let (mut state, mut host, view_ref, _) = view_fixture();
        fill_view_pages(&mut host);
        let view_resource = objects::resolve_resource(&state, &host, 1, view_ref).expect("view");
        let _arm = resolve_sampled_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(view_resource.clone()),
            true,
        )
        .expect("arm the witness");
        let (mut serial_state, mut serial_host, _, _) = view_fixture();
        fill_view_pages(&mut serial_host);
        let serial_resource =
            objects::resolve_resource(&serial_state, &serial_host, 1, view_ref).expect("view");
        let _serial_arm = resolve_sampled_source(
            &mut serial_state,
            &mut serial_host,
            1,
            view_ref,
            Some(serial_resource.clone()),
            true,
        )
        .expect("arm the serial twin");
        let serial = resolve_sampled_source(
            &mut serial_state,
            &mut serial_host,
            1,
            view_ref,
            Some(serial_resource),
            true,
        )
        .expect("the serial ladder serves");
        let serial_src = match &serial.3 {
            SampledSourceRequest::GuestRuns(src, ..) => src,
            _ => panic!("the armed view serves guest runs on the serial ladder"),
        };
        let serial_bytes = gathered_bytes(serial_src);
        add_cache_ref(&mut state, &mut host, 7, 99, 0x2000, 0);
        let bind_for = |texture_ref: u32, state: &DeviceState, host: &FakeHost| -> TextureBind {
            TextureBind {
                index: 0,
                texture_ref,
                resource: Some(
                    objects::resolve_resource(state, host, 1, texture_ref).expect("bind resource"),
                ),
            }
        };
        let view_bind = bind_for(view_ref, &state, &host);
        let cache_bind = bind_for(7, &state, &host);
        let per_draw = vec![vec![view_bind.clone()], vec![cache_bind]];
        let tables: Vec<DrawTables<'_>> = tables_of(&per_draw);
        let before_planned = store_route_count("pass_stage_sampled_planned");
        let before_skip = store_route_count("pass_stage_sampled_skipped_speculative");
        let before_staged = store_route_count("pass_stage_sampled_staged_proven");
        let mut prep = PassPrep::open_with_chunk(&state, 1, tables, 2).expect("opens");
        prep.ahead(&state, &FakeHost::new(), 0);
        assert!(delta("pass_stage_sampled_planned", before_planned) >= 1);
        assert_eq!(
            delta("pass_stage_sampled_skipped_speculative", before_skip),
            0
        );
        // The first bind is already the staged window — the eager rail.
        let first = prefetch::prefetched_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(&view_resource),
            true,
        );
        match first {
            prefetch::Prefetched::Serve((_, _, _, SampledSourceRequest::Staged(req))) => {
                // SAFETY: the fact is alive for this borrow.
                let staged = unsafe { req.staged.bytes() };
                assert_eq!(
                    staged,
                    &serial_bytes[..],
                    "the eager window is the serial CPU gather"
                );
            }
            other => panic!("unexpected serve variant"),
        }
        assert_eq!(delta("pass_stage_sampled_staged_proven", before_staged), 0);
        drop(prep);
        drop(
            crate::backend::vulkan::engine::pass_stage::take_test_arenas()
                .expect("the test arenas are still installed"),
        );
    }

    /// TEMP (lab, PASS_STAGE): the rail's own gate. `REIMS_VGPU_LAB_PASS_STAGE`
    /// off attaches nothing anywhere: the first bind and the repeat both serve
    /// the plain guest-runs request — the serial ladder's own answer, field
    /// for field — and no `pass_stage_sampled_*` census moves.
    #[test]
    fn pass_stage_off_keeps_the_sampled_binds_serial() {
        let _stage = crate::runtime::par::staging::test_enable(false);
        let _mode = test_mode(PassPrepMode::On);
        // The run memo is process-global and keyed by GPAs alone; without a
        // clear it would hand this test an earlier test's runs — host pointers
        // into a fixture that has already been dropped.
        run_memo_clear_for_test();
        let (mut state, mut host, view_ref, _) = view_fixture();
        fill_view_pages(&mut host);
        let view_resource = objects::resolve_resource(&state, &host, 1, view_ref).expect("view");
        let _arm = resolve_sampled_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(view_resource.clone()),
            true,
        )
        .expect("arm the witness");
        let (mut serial_state, mut serial_host, _, _) = view_fixture();
        fill_view_pages(&mut serial_host);
        let serial_resource =
            objects::resolve_resource(&serial_state, &serial_host, 1, view_ref).expect("view");
        let _serial_arm = resolve_sampled_source(
            &mut serial_state,
            &mut serial_host,
            1,
            view_ref,
            Some(serial_resource.clone()),
            true,
        )
        .expect("arm the serial twin");
        let serial = resolve_sampled_source(
            &mut serial_state,
            &mut serial_host,
            1,
            view_ref,
            Some(serial_resource),
            true,
        )
        .expect("the serial ladder serves");
        let serial_src = match &serial.3 {
            SampledSourceRequest::GuestRuns(src, ..) => src,
            _ => panic!("the armed view serves guest runs on the serial ladder"),
        };
        add_cache_ref(&mut state, &mut host, 7, 99, 0x2000, 0);
        add_cache_ref(&mut state, &mut host, 8, 91, 0x2100, 5);
        let bind_for = |texture_ref: u32, state: &DeviceState, host: &FakeHost| -> TextureBind {
            TextureBind {
                index: 0,
                texture_ref,
                resource: Some(
                    objects::resolve_resource(state, host, 1, texture_ref).expect("bind resource"),
                ),
            }
        };
        let pattern = [0u32, 1, 0, 2];
        let per_draw: Vec<Vec<TextureBind>> = pattern
            .iter()
            .map(|&which| {
                let texture_ref = match which {
                    0 => view_ref,
                    1 => 7,
                    _ => 8,
                };
                vec![bind_for(texture_ref, &state, &host)]
            })
            .collect();
        let tables: Vec<DrawTables<'_>> = tables_of(&per_draw);
        let before_planned = store_route_count("pass_stage_sampled_planned");
        let before_skip = store_route_count("pass_stage_sampled_skipped_speculative");
        let before_staged = store_route_count("pass_stage_sampled_staged_proven");
        let mut prep = PassPrep::open_with_chunk(&state, 1, tables, 2).expect("opens");
        prep.ahead(&state, &FakeHost::new(), 0);
        let first = prefetch::prefetched_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(&view_resource),
            true,
        );
        match first {
            prefetch::Prefetched::Serve((_, _, _, SampledSourceRequest::GuestRuns(src, ..))) => {
                assert!(
                    Arc::ptr_eq(&src.runs, &serial_src.runs) || src.runs[..] == serial_src.runs[..]
                );
                assert_eq!(
                    (src.source_offset, src.total_len, src.row_length_texels),
                    (
                        serial_src.source_offset,
                        serial_src.total_len,
                        serial_src.row_length_texels
                    ),
                    "the rail-off bind is the serial ladder's own serve"
                );
            }
            other => panic!("unexpected serve variant"),
        }
        prep.ahead(&state, &FakeHost::new(), 2);
        let second = prefetch::prefetched_source(
            &mut state,
            &mut host,
            1,
            view_ref,
            Some(&view_resource),
            true,
        );
        assert!(matches!(
            second,
            prefetch::Prefetched::Serve((_, _, _, SampledSourceRequest::GuestRuns(..)))
        ));
        assert_eq!(delta("pass_stage_sampled_planned", before_planned), 0);
        assert_eq!(
            delta("pass_stage_sampled_skipped_speculative", before_skip),
            0
        );
        assert_eq!(delta("pass_stage_sampled_staged_proven", before_staged), 0);
        drop(prep);
    }

    /// One more host-cache ref (`texture_ref` over `surface_id`, its descriptor
    /// at `desc_gva`), for fixtures that need plannable refs beside a view ref.
    fn add_cache_ref(
        state: &mut DeviceState,
        host: &mut FakeHost,
        texture_ref: u32,
        surface_id: u32,
        desc_gva: u64,
        fill: u8,
    ) {
        use crate::protocol::endian::st32;
        use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
        use crate::runtime::gva_mem;
        let built = reims_vgpu_wire::device_desc::RefTextureBuilder::new(surface_id, 0, 0, 0)
            .with_len(objects::TYPE5_MIN_LEN);
        assert!(gva_mem::write_task_gva(
            host,
            &state.tasks[1],
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
            host,
            &state.tasks[1],
            list_off,
            &list_entry,
            PAGE_SHIFT_X86
        )
        .is_ok());
        assert!(state.map_surface(surface_id));
        assert!(state.set_mapping_geom(surface_id, 4, 3, MTL_FORMAT_BGRA8_UNORM));
        state.mappings.get_mut(&surface_id).unwrap().page_entries = vec![1];
        crate::runtime::surface_cache::store(
            state,
            surface_id,
            4,
            3,
            [fill, 8, 7, 255].repeat(4 * 3),
        );
    }

    /// The sample-plan view fixture's shape: a ref-texture (`texture_ref` 2)
    /// whose descriptor views surface 72 as sRGB BGRA8 while the surface is
    /// unorm BGRA8, 256x128, over 32 guest pages scattered two apart on a host
    /// that aliases packed runs only — the zero-copy rail gathers the runs.
    fn view_fixture() -> (DeviceState, FakeHost, u32, u32) {
        use crate::protocol::endian::st32;
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
        use crate::runtime::gva_mem;
        use reims_vgpu_protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB;

        let mut state = pooled_state();
        let mut host = FakeHost::new();
        host.stable_map_pages = true;
        host.strict_linux_map = true;
        // The pool plans through a WorkerHost over a table with no dirty
        // callbacks, while the commit validates through this FakeHost — the
        // two agree only while the witness entries carry no token, so the
        // fixture's writes stay unobservable and every entry is unarmed
        // (token 0), the shape both hosts answer identically for.
        host.guest_writes_unobservable = true;
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
        let (texture_ref, surface_id, desc_gva) = (2u32, 72u32, 0x1000u64);
        let built = reims_vgpu_wire::device_desc::RefTextureBuilder::new(
            surface_id,
            0,
            0,
            reims_vgpu_wire::device_desc::TYPE5_RECORD_TAG_COLOR_VIEW,
        )
        .geometry(MTL_FORMAT_BGRA8_UNORM_SRGB, 256, 128, 1);
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            desc_gva,
            built.bytes(),
            PAGE_SHIFT_X86
        )
        .is_ok());
        let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed =
            (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((built.bytes().len() as u32) << 8);
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
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpas: Vec<u64> = (0..32u64).map(|i| 0x5000_0000 + 2 * i * page).collect();
        for &gpa in &gpas {
            host.map_range(gpa, page as usize, 0x33);
        }
        assert!(state.map_surface(surface_id));
        assert!(state.set_mapping_geom(surface_id, 256, 128, MTL_FORMAT_BGRA8_UNORM));
        state.mappings.get_mut(&surface_id).unwrap().page_entries = gpas
            .iter()
            .map(|&gpa| {
                (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
        (state, host, texture_ref, surface_id)
    }

    /// The chunk knob: absent is the model's K=8, a value is clamped into
    /// 2..=64, and junk falls back rather than refusing the boot.
    #[test]
    fn the_chunk_knob_parses_and_clamps() {
        assert_eq!(parse_chunk(None), 8);
        assert_eq!(parse_chunk(Some("8")), 8);
        assert_eq!(parse_chunk(Some("4")), 4);
        assert_eq!(parse_chunk(Some("64")), 64);
        assert_eq!(parse_chunk(Some("1000")), 64);
        assert_eq!(parse_chunk(Some("1")), 2);
        assert_eq!(parse_chunk(Some("0")), 2);
        assert_eq!(parse_chunk(Some("junk")), 8);
        assert_eq!(parse_chunk(Some(" 12 ")), 12);
    }

    /// The memo rung may only be planned where the serial ladder refuses the
    /// zero-copy rail too — the floor. A cold window or witness is a refusal
    /// the serial rail resolves itself (fills the window, re-arms the
    /// witness), so a memo plan answering those binds would serve `Bytes`
    /// where serial serves `GuestRuns`.
    #[test]
    fn the_memo_rung_follows_only_the_floor_refusal() {
        assert!(prefetch::memo_plan_may_follow(SampleBail::ZeroCopyDeclined));
        for other in [
            SampleBail::ColdWindow,
            SampleBail::ColdWitness,
            SampleBail::UpperRung,
            SampleBail::OwedFrame,
            SampleBail::ColdBacking,
        ] {
            assert!(
                !prefetch::memo_plan_may_follow(other),
                "the memo must not answer past {other:?}"
            );
        }
    }

    /// A cross-variant shadow answer is counted by its pair as well as by the
    /// gate's total, and the fail line names both sides — the split the fs2s1
    /// diagnosis had to infer from the census (planned GuestRuns/Bytes against
    /// a serial `Target`, the resident-readiness flip).
    #[test]
    fn a_cross_variant_shadow_answer_names_its_pair() {
        use crate::backend::vulkan::engine::TargetIdentity;
        let target = (
            4u32,
            3u32,
            9u32,
            SampledSourceRequest::Target(
                TargetIdentity::Surface {
                    id: 9,
                    width: 4,
                    height: 3,
                    generation: 1,
                    format: ash::vk::Format::B8G8R8A8_UNORM,
                },
                ash::vk::Format::B8G8R8A8_UNORM,
            ),
        );
        let runs = (
            4u32,
            3u32,
            9u32,
            SampledSourceRequest::GuestRuns(
                crate::backend::vulkan::engine::GuestRunSource {
                    runs: Arc::new(Vec::new()),
                    source_offset: 0,
                    total_len: 48,
                    row_length_texels: 0,
                    pages: None,
                    direct_image: None,
                },
                crate::protocol::pixel_format::TexelLayout::Bgra8,
                ash::vk::Format::B8G8R8A8_UNORM,
                1,
                None,
                crate::runtime::gather_witness::GatherVouch::Fresh,
                None,
                pixel_format::swizzle_identity(),
            ),
        );
        let bytes = (
            4u32,
            3u32,
            9u32,
            SampledSourceRequest::Bytes(
                Arc::new(vec![0u8; 48]),
                None,
                SampledByteFormat::from_source(
                    crate::protocol::pixel_format::TexelLayout::Rgba8,
                    MTL_FORMAT_BGRA8_UNORM,
                ),
                crate::backend::vulkan::engine::SampledByteOrigin::SurfaceGuestFallback,
            ),
        );
        let before_total = store_route_count("par_shadow_mismatch_variant");
        let before_runs = store_route_count("par_shadow_variant_runs_vs_target");
        let before_bytes = store_route_count("par_shadow_variant_bytes_vs_target");
        prefetch::shadow_compare(31, &runs, Some(&target));
        prefetch::shadow_compare(32, &bytes, Some(&target));
        assert_eq!(delta("par_shadow_mismatch_variant", before_total), 2);
        assert_eq!(delta("par_shadow_variant_runs_vs_target", before_runs), 1);
        assert_eq!(delta("par_shadow_variant_bytes_vs_target", before_bytes), 1);
        // And the comparison itself is not relaxed: same-variant binds still
        // compare their contents.
        let before_match = store_route_count("par_shadow_match_bytes");
        prefetch::shadow_compare(33, &bytes, Some(&bytes));
        assert_eq!(delta("par_shadow_match_bytes", before_match), 1);
    }

    /// A pass under two draws, or a device with no pool, opens nothing — the
    /// byte-for-byte old path — and `off` refuses even a full pass.
    #[test]
    fn the_driver_stays_off_where_it_cannot_pay() {
        let refs = [2u32, 3];
        let pattern = vec![0u32, 1, 0, 1];
        let (state, host) = multi_ref_fixture(&refs);
        let binds = binds_for(&state, &host, &pattern, &refs);
        let tables = tables_of(&binds);
        let _mode = test_mode(PassPrepMode::Off);
        assert!(
            PassPrep::open_with_chunk(&state, 1, tables, 4).is_none(),
            "off is the old path"
        );
        let _mode = test_mode(PassPrepMode::On);
        let one = vec![DrawTables {
            fragment: &[],
            vertex: &[],
            vertex_buffers: &[],
            fragment_buffers: &[],
            indexed: None,
        }];
        assert!(
            PassPrep::open_with_chunk(&state, 1, one, 4).is_none(),
            "a one-draw pass cannot plan ahead of itself"
        );
        let bare = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let two = vec![
            DrawTables {
                fragment: &[],
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            },
            DrawTables {
                fragment: &[],
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            },
        ];
        assert!(
            PassPrep::open_with_chunk(&bare, 1, two, 4).is_none(),
            "no pool, no preparation"
        );
    }

    /// TEMP (lab, PASS_STAGE): the staged buffer arm end to end. A cold,
    /// never-primed reference (every zero-copy rail refuses on this host) plans
    /// into a worker arena window, the commit serves `BufferContent::Staged`
    /// whose bytes are the serial ladder's `Bytes`, the census names both halves,
    /// and the frame's pop releases the window.
    #[test]
    fn a_cold_bind_stages_the_serial_bytes() {
        let _stage = crate::runtime::par::staging::test_enable(true);
        let _mode = test_mode(PassPrepMode::On);
        crate::backend::vulkan::engine::pass_stage::install_test_arenas(1, 64 * 1024, 2);
        // Sub-floor references (4 KiB against the 16 KiB zero-copy floor): the
        // one population whose serial answer is provably the CPU read. Two of
        // them, so the chunk's item list is worth a fork.
        let specs = [(7u32, 0x1000u64), (8, 0x1000)];
        let (mut planned_state, planned_host) = pooled_buffer_fixture(&specs);
        let (mut serial_state, serial_host) = pooled_buffer_fixture(&specs);
        // The binds: each draw names both refs at offset 0, so each ref's one
        // staged window serves every draw.
        let make_binds = |state: &DeviceState, host: &FakeHost| -> Vec<Vec<BufferBind>> {
            [7u32, 8]
                .iter()
                .map(|&buffer_ref| {
                    vec![BufferBind {
                        index: 0,
                        buffer_ref,
                        resource: Some(
                            crate::runtime::objects::resolve_resource(state, host, 1, buffer_ref)
                                .expect("bind resource"),
                        ),
                        offset: 0,
                        attribute_stride: None,
                    }]
                })
                .collect()
        };
        let planned_binds = make_binds(&planned_state, &*planned_host);
        let serial_binds = make_binds(&serial_state, &*serial_host);
        // The serial arm first: no priming, so the ladder falls to its CPU read.
        // The serial reference answers must be the ladder's own Bytes, so the
        // (default-on) staging rail is held off for this arm only.
        let mut serial_host = serial_host;
        let _serial_no_stage = crate::runtime::par::staging::test_enable(false);
        let serial_answers: Vec<_> = serial_binds
            .iter()
            .map(|one| {
                let bind = one.first().expect("one bind per draw");
                load_buffer_content(
                    &mut serial_state,
                    &mut *serial_host,
                    1,
                    bind.buffer_ref,
                    bind.resource.as_deref(),
                    bind.offset,
                    true,
                    None,
                )
                .expect("the serial ladder serves")
            })
            .collect();
        for answer in &serial_answers {
            assert!(
                matches!(
                    answer,
                    crate::backend::vulkan::engine::BufferContent::Bytes(_)
                ),
                "the fixture's host refuses every zero-copy rail, so the serial arm reads CPU bytes"
            );
        }
        // The serial comparison is done; the planned arm stages again (the
        // guard's Drop leaves the flag down, so raise it back).
        drop(_serial_no_stage);
        let _stage_again = crate::runtime::par::staging::test_enable(true);
        let tables: Vec<DrawTables<'_>> = planned_binds
            .iter()
            .map(|one| DrawTables {
                fragment: &[],
                vertex: &[],
                vertex_buffers: one.as_slice(),
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let before_planned = store_route_count("pass_prep_buf_planned_staged");
        let before_hit = store_route_count("pass_prep_buf_hit_staged");
        let mut prep = PassPrep::open_with_chunk(&planned_state, 1, tables, 4).expect("opens");
        for di in 0..2 {
            prep.ahead(&planned_state, &FakeHost::new(), di);
            let bind = planned_binds[di].first().expect("one bind per draw");
            let served = load_buffer_content(
                &mut planned_state,
                &mut *planned_host,
                1,
                bind.buffer_ref,
                bind.resource.as_deref(),
                bind.offset,
                true,
                None,
            )
            .expect("the planned arm serves");
            match &served {
                crate::backend::vulkan::engine::BufferContent::Staged(slice) => {
                    // SAFETY: the fact is alive for this borrow.
                    let staged = unsafe { slice.bytes() };
                    match &serial_answers[di] {
                        crate::backend::vulkan::engine::BufferContent::Bytes(b) => {
                            assert_eq!(
                                staged,
                                &b[..],
                                "draw {di}: the staged bytes are the serial ones"
                            );
                        }
                        other => panic!("draw {di}: serial answered {other:?}"),
                    }
                }
                other => panic!("draw {di}: the staged arm serves, got {other:?}"),
            }
        }
        assert!(delta("pass_prep_buf_planned_staged", before_planned) >= 1);
        assert!(delta("pass_prep_buf_hit_staged", before_hit) >= 1);
        // The frame still holds the staged plan (it serves every draw); its pop
        // is what releases the window.
        let arenas = crate::backend::vulkan::engine::pass_stage::arenas().expect("installed");
        assert!(arenas[0].ring.in_flight() >= 1, "the plan holds its window");
        drop(prep);
        assert_eq!(
            arenas[0].ring.in_flight(),
            0,
            "the frame's pop releases every staged window"
        );
        drop(
            crate::backend::vulkan::engine::pass_stage::take_test_arenas()
                .expect("the test arenas are still installed"),
        );
        let _ = &mut planned_state;
    }

    /// TEMP (lab, PACKED_WARM): the sample side's cold_packed class, closed.
    /// A cold linear texture bails `ColdPacked` in chunk 0 — and with the knob
    /// on that bail is *requeued*, not held sticky: once the entry stands (the
    /// construction the drain's first serial bind lands; inserted by hand
    /// because this host's own construction refuses), the next chunk replans
    /// the ref, the replan hits, and the bind serves the serial ladder's
    /// answer for the same state. A retirement between the replan and the ask
    /// drops the plan (`par_prefetch_invalid_packed`) and the ask falls back
    /// to the serial ladder.
    #[test]
    fn a_cold_packed_ref_requeues_and_replans_to_the_serial_answer() {
        let _warm = test_packed_warm(true);
        let _mode = test_mode(PassPrepMode::On);
        let (mut planned_state, planned_host) = pooled_linear_fixture();
        let (mut serial_state, serial_host) = pooled_linear_fixture();
        // The freshly-stored state both arms plan against: the plane declared,
        // the witness unarmed (the shape the worker's shim and the drain's
        // host answer identically for) — everything but the packed entry,
        // which is the cold this test exists to close.
        declare_gva_plane(&mut planned_state);
        declare_gva_plane(&mut serial_state);
        // The warm entry both arms will answer from: the planned arm gets it
        // after chunk 0 (the drain's serial construction, stood in for — this
        // host's `ensure_packed_resource` refuses where a capable host's
        // constructs), the serial arm before its one bind.
        warm_packed_entry(&mut serial_state);
        // Two cache refs beside the texture, so every chunk that matters
        // carries two items and forks: chunk 0 plans the texture and the
        // first cache ref; chunk 1 replans the texture beside the second.
        add_cache_ref(&mut planned_state, &mut *planned_host, 7, 99, 0x2000, 0);
        add_cache_ref(&mut planned_state, &mut *planned_host, 8, 91, 0x2100, 5);
        // L = the linear texture (ref 2); C/C2 = the cache refs.
        let pattern = [0u32, 1, 0, 2, 0, 1];
        let bind_for = |which: u32, state: &DeviceState, host: &FakeHost| -> TextureBind {
            let texture_ref = match which {
                0 => 2,
                other => 6 + other, // 1 -> 7, 2 -> 8
            };
            TextureBind {
                index: 0,
                texture_ref,
                resource: Some(
                    objects::resolve_resource(state, host, 1, texture_ref).expect("bind resource"),
                ),
            }
        };
        let planned_binds: Vec<Vec<TextureBind>> = pattern
            .iter()
            .map(|&which| vec![bind_for(which, &planned_state, &*planned_host)])
            .collect();
        let tables: Vec<DrawTables<'_>> = planned_binds
            .iter()
            .map(|one| DrawTables {
                fragment: one.as_slice(),
                vertex: &[],
                vertex_buffers: &[],
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let before_requeue = store_route_count("packed_warm_requeue");
        let before_cold = store_route_count("par_prefetch_bail_cold_packed");
        let mut prep = PassPrep::open_with_chunk(&planned_state, 1, tables, 2).expect("opens");
        // Chunk 0 (draws 0..2): the texture bails cold (no packed entry) — and
        // is requeued rather than held sticky.
        prep.ahead(&planned_state, &FakeHost::new(), 0);
        assert_eq!(delta("par_prefetch_bail_cold_packed", before_cold), 1);
        assert_eq!(delta("packed_warm_requeue", before_requeue), 1);
        // The drain's first serial binds of the pass run here; the entry they
        // would construct stands from this point on.
        warm_packed_entry(&mut planned_state);
        // Chunk 1 (draws 2..4): the requeued texture replans — onto the now
        // warm rail, so the replan is a hit.
        let before_hit = store_route_count("packed_warm_hit");
        let before_planned = store_route_count("par_prefetch_planned_linear");
        prep.ahead(&planned_state, &FakeHost::new(), 2);
        assert_eq!(delta("packed_warm_hit", before_hit), 1);
        assert_eq!(delta("par_prefetch_planned_linear", before_planned), 1);
        // Draw 2 binds the texture: the plan serves, and the serve is the
        // serial ladder's own answer for the same warm state.
        let resource = objects::resolve_resource(&planned_state, &*planned_host, 1, 2)
            .expect("texture resource");
        let served = prefetch::prefetched_source(
            &mut planned_state,
            &mut *planned_host,
            1,
            2,
            Some(&resource),
            true,
        );
        let serial = resolve_sampled_source(
            &mut serial_state,
            &mut *serial_host,
            1,
            2,
            Some(resource.clone()),
            true,
        )
        .expect("the serial ladder serves the warm entry");
        match (served, serial) {
            (prefetch::Prefetched::Serve((pw, ph, _, psrc)), (sw, sh, _, ssrc)) => {
                assert_eq!((pw, ph), (sw, sh), "the geometry agrees");
                match (psrc, ssrc) {
                    (
                        SampledSourceRequest::GuestRuns(p, playout, _, pplanes, _, _, _, _),
                        SampledSourceRequest::GuestRuns(s, slayout, _, splanes, _, _, _, _),
                    ) => {
                        assert!(Arc::ptr_eq(&p.runs, &s.runs) || p.runs[..] == s.runs[..]);
                        assert_eq!(p.source_offset, s.source_offset);
                        assert_eq!(p.total_len, s.total_len);
                        assert_eq!(p.row_length_texels, s.row_length_texels);
                        assert_eq!((playout, pplanes), (slayout, splanes));
                    }
                    (planned_src, _serial_src) => panic!(
                        "both arms serve guest runs, got a different variant ({})",
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
            (other, _) => panic!(
                "the replanned texture serves, got shadow={}",
                matches!(other, prefetch::Prefetched::Shadow(_))
            ),
        }
        // A retirement between the replan and the ask drops the plan: the
        // packed entry the plan borrowed is gone, the ask falls back, and the
        // serial ladder owns the bind again.
        assert!(planned_state.retire_bound_buffers_in_range(1, 0x20_000, 0x1000) >= 0);
        let before_invalid = store_route_count("par_prefetch_invalid_packed");
        let fell_back = prefetch::prefetched_source(
            &mut planned_state,
            &mut *planned_host,
            1,
            2,
            Some(&resource),
            true,
        );
        assert!(matches!(fell_back, prefetch::Prefetched::Serial));
        assert_eq!(delta("par_prefetch_invalid_packed", before_invalid), 1);
        drop(prep);
    }

    /// TEMP (lab, PACKED_WARM): the warm-up cork, buffer side. The drain warms
    /// the first chunk's cold references before the chunk forks — the packed
    /// alias this host refuses becomes a held negative, and the whole runs the
    /// serial ladder would walk land right there — so chunk 0 plans the whole
    /// rail instead of bailing `cold_packed` sticky, the binds serve from the
    /// plan, and the registry ends where the serial ladder leaves it.
    #[test]
    fn warm_packed_corks_the_first_chunks_cold_binds() {
        let _no_stage = crate::runtime::par::staging::test_enable(false);
        let _warm = test_packed_warm(true);
        let _mode = test_mode(PassPrepMode::On);
        let specs = [(7u32, 0x8000u64), (8, 0x8000)];
        let (mut planned_state, planned_host) = pooled_buffer_fixture(&specs);
        let (mut serial_state, serial_host) = pooled_buffer_fixture(&specs);
        let make_binds = |state: &DeviceState, host: &FakeHost| -> Vec<Vec<BufferBind>> {
            [7u32, 8, 7, 8, 7, 8]
                .iter()
                .map(|&buffer_ref| {
                    vec![BufferBind {
                        index: 0,
                        buffer_ref,
                        resource: Some(
                            crate::runtime::objects::resolve_resource(state, host, 1, buffer_ref)
                                .expect("bind resource"),
                        ),
                        offset: 0x40,
                        attribute_stride: None,
                    }]
                })
                .collect()
        };
        let planned_binds = make_binds(&planned_state, &*planned_host);
        let serial_binds = make_binds(&serial_state, &*serial_host);
        let tables: Vec<DrawTables<'_>> = planned_binds
            .iter()
            .map(|one| DrawTables {
                fragment: &[],
                vertex: &[],
                vertex_buffers: one.as_slice(),
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        let mut prep = PassPrep::open_with_chunk(&planned_state, 1, tables, 4).expect("opens");
        let before_refs = store_route_count("packed_warm_refs");
        let before_cold = store_route_count("pass_prep_buf_bail_cold_packed");
        let before_whole = store_route_count("pass_prep_buf_bail_cold_whole");
        // The cork: the drain constructs before the fork.
        prep.warm_packed(&mut planned_state, &mut *planned_host);
        assert!(delta("packed_warm_refs", before_refs) >= 1);
        assert!(
            planned_state.bound_buffers.whole(1, 7).is_some()
                && planned_state.bound_buffers.whole(1, 8).is_some(),
            "the refused packed alias fell to the whole runs the serial ladder would walk"
        );
        // Chunk 0 plans onto the warm rail — no cold bail at all.
        let before_planned = store_route_count("pass_prep_buf_planned_whole");
        prep.ahead(&planned_state, &FakeHost::new(), 0);
        assert_eq!(delta("pass_prep_buf_bail_cold_packed", before_cold), 0);
        assert_eq!(delta("pass_prep_buf_bail_cold_whole", before_whole), 0);
        assert!(delta("pass_prep_buf_planned_whole", before_planned) >= 1);
        // The binds serve from the plans and answer what the serial ladder
        // answers for the same binds.
        let serial_answers: Vec<_> = serial_binds
            .iter()
            .map(|one| {
                let bind = one.first().expect("one bind per draw");
                load_buffer_content(
                    &mut serial_state,
                    &mut *serial_host,
                    1,
                    bind.buffer_ref,
                    bind.resource.as_deref(),
                    bind.offset,
                    true,
                    None,
                )
                .expect("the serial ladder serves")
            })
            .collect();
        for (di, one) in planned_binds.iter().enumerate() {
            let bind = one.first().expect("one bind per draw");
            let served = load_buffer_content(
                &mut planned_state,
                &mut *planned_host,
                1,
                bind.buffer_ref,
                bind.resource.as_deref(),
                bind.offset,
                true,
                None,
            )
            .expect("the planned arm serves");
            assert!(
                contents_agree(&served, &serial_answers[di]),
                "draw {di} ref {} agrees with the serial ladder",
                bind.buffer_ref
            );
        }
        drop(prep);
        assert_eq!(
            planned_state.bound_buffers.shape(),
            serial_state.bound_buffers.shape(),
            "the warm-up leaves the registry where the serial ladder does"
        );
    }

    /// TEMP (lab, PACKED_WARM): the requeue, buffer side, and the knob-off
    /// control. Without the warm-up, a cold reference bails in chunk 0 — with
    /// the knob on the bail is requeued, the drain's serial binds construct
    /// the entry (the packed refusal plus the whole walk this host can do),
    /// and the next chunk's replan hits. With the knob off the very same bail
    /// stays sticky for the pass: no requeue, no replan, byte-for-byte the
    /// old path.
    #[test]
    fn a_cold_buffer_bail_requeues_once_and_the_knob_off_keeps_it_sticky() {
        let _no_stage = crate::runtime::par::staging::test_enable(false);
        let _mode = test_mode(PassPrepMode::On);
        let specs = [(7u32, 0x8000u64), (8, 0x8000)];
        let (mut state, host) = pooled_buffer_fixture(&specs);
        let make_binds = |state: &DeviceState, host: &FakeHost| -> Vec<Vec<BufferBind>> {
            [7u32, 8, 7, 8, 7, 8]
                .iter()
                .map(|&buffer_ref| {
                    vec![BufferBind {
                        index: 0,
                        buffer_ref,
                        resource: Some(
                            crate::runtime::objects::resolve_resource(state, host, 1, buffer_ref)
                                .expect("bind resource"),
                        ),
                        offset: 0x40,
                        attribute_stride: None,
                    }]
                })
                .collect()
        };
        let binds = make_binds(&state, &*host);
        let tables: Vec<DrawTables<'_>> = binds
            .iter()
            .map(|one| DrawTables {
                fragment: &[],
                vertex: &[],
                vertex_buffers: one.as_slice(),
                fragment_buffers: &[],
                indexed: None,
            })
            .collect();
        // -- The knob off: the bail is sticky, byte-for-byte the old path.
        {
            let _off = test_packed_warm(false);
            let before_cold = store_route_count("pass_prep_buf_bail_cold_packed");
            let before_warm = store_route_count("packed_warm_refs");
            let mut prep = PassPrep::open_with_chunk(&state, 1, tables, 4).expect("opens");
            prep.warm_packed(&mut state, &mut *host);
            assert_eq!(delta("packed_warm_refs", before_warm), 0);
            prep.ahead(&state, &FakeHost::new(), 0);
            assert_eq!(delta("pass_prep_buf_bail_cold_packed", before_cold), 2);
            // The drain's serial binds warm the registry (this host refuses
            // the packed alias and walks the whole runs).
            for one in binds.iter().take(4) {
                let bind = one.first().expect("one bind per draw");
                let _ = load_buffer_content(
                    &mut state,
                    &mut *host,
                    1,
                    bind.buffer_ref,
                    bind.resource.as_deref(),
                    bind.offset,
                    true,
                    None,
                )
                .expect("the serial ladder serves");
            }
            // The next chunk does not replan the bailed refs: sticky.
            let before_planned = store_route_count("pass_prep_buf_planned_whole");
            let before_requeue = store_route_count("packed_warm_requeue");
            prep.ahead(&state, &FakeHost::new(), 4);
            assert_eq!(delta("pass_prep_buf_planned_whole", before_planned), 0);
            assert_eq!(delta("packed_warm_requeue", before_requeue), 0);
            drop(prep);
        }
        // -- The knob on, no warm-up (the later-chunk cold a cork cannot
        //    reach): the bail requeues, the drain's binds warm the registry,
        //    and the next chunk's replan hits.
        {
            let _on = test_packed_warm(true);
            let (mut state, host) = pooled_buffer_fixture(&specs);
            let binds = make_binds(&state, &*host);
            let tables: Vec<DrawTables<'_>> = binds
                .iter()
                .map(|one| DrawTables {
                    fragment: &[],
                    vertex: &[],
                    vertex_buffers: one.as_slice(),
                    fragment_buffers: &[],
                    indexed: None,
                })
                .collect();
            let before_requeue = store_route_count("packed_warm_requeue");
            let mut prep = PassPrep::open_with_chunk(&state, 1, tables, 4).expect("opens");
            prep.ahead(&state, &FakeHost::new(), 0);
            assert_eq!(delta("packed_warm_requeue", before_requeue), 2);
            for one in binds.iter().take(4) {
                let bind = one.first().expect("one bind per draw");
                let _ = load_buffer_content(
                    &mut state,
                    &mut *host,
                    1,
                    bind.buffer_ref,
                    bind.resource.as_deref(),
                    bind.offset,
                    true,
                    None,
                )
                .expect("the serial ladder serves");
            }
            let before_hit = store_route_count("packed_warm_hit");
            let before_planned = store_route_count("pass_prep_buf_planned_whole");
            prep.ahead(&state, &FakeHost::new(), 4);
            assert_eq!(delta("pass_prep_buf_planned_whole", before_planned), 2);
            assert_eq!(delta("packed_warm_hit", before_hit), 2);
            // And the requeued plan serves the later binds.
            let bind = binds[4].first().expect("one bind per draw");
            let before_served = store_route_count("pass_prep_buf_hit_whole");
            let served = load_buffer_content(
                &mut state,
                &mut *host,
                1,
                bind.buffer_ref,
                bind.resource.as_deref(),
                bind.offset,
                true,
                None,
            )
            .expect("the replanned rail serves");
            assert!(matches!(
                served,
                crate::backend::vulkan::engine::BufferContent::GuestRuns(_)
            ));
            assert_eq!(delta("pass_prep_buf_hit_whole", before_served), 1);
            drop(prep);
        }
    }
}
