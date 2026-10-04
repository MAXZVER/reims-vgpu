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
    *ON.get_or_init(|| matches!(std::env::var("REIMS_VGPU_LAB_PREP_SYNC").as_deref(), Ok("on")))
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
        parse_chunk(std::env::var("REIMS_VGPU_LAB_PASS_PREP_CHUNK").ok().as_deref())
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
        self.vertex_buffers.iter().chain(self.fragment_buffers.iter())
    }
}

/// One planned bind candidate: the ref and the resource the walk retained.
enum PrepItem {
    /// A sampled-source bind (`plan_item`'s own ladder).
    Sample {
        texture_ref: u32,
        resource: std::sync::Arc<crate::model::TaskResource>,
    },
    /// TEMP (lab, PASS_PREP step 3): a buffer bind's warm rails.
    Buffer {
        buffer_ref: u32,
        resource: std::sync::Arc<crate::model::TaskResource>,
    },
    /// TEMP (lab, PASS_PREP step 3): an indexed draw's index bind, planned
    /// under its exact `(ref, offset, need)` key.
    Index {
        key: buffer_plan::BufferExactKey,
        info: IndexedDrawInfo,
    },
    /// TEMP (lab, PASS_PREP step 3): the pass's Store page walk, for the key
    /// the drain latched from the pass's colour-0.
    StorePages {
        key: buffer_plan::StoreTargetKey,
    },
}

/// What one planned item produced.
enum PrepOutcome {
    Sample(Result<StreamPlan, SampleBail>),
    Buffer(Result<buffer_plan::BufferContentPlan, buffer_plan::BufferBail>),
    /// The Store walk resolves pages or not; `None` is no plan, not a bail.
    Store(Option<buffer_plan::StorePagesPlan>),
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
    /// TEMP (lab, PASS_PREP step 3): the pass's colour-0 as the drain latched
    /// it once the attachment template existed, and whether a scope has taken
    /// the Store walk yet.
    store_target: Option<buffer_plan::StoreTargetKey>,
    store_pages_planned: bool,
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
        Some(Self {
            task_id,
            tables,
            chunk,
            offered: 0,
            planned: HashSet::new(),
            planned_buffers: HashSet::new(),
            planned_indexes: HashSet::new(),
            store_target: None,
            store_pages_planned: false,
            _frame: prefetch::push_empty_frame(),
        })
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

    /// At the top of record `di`: when `di >= offered`, plan the new refs of
    /// draws `[offered, offered + chunk)` on the pool and merge them into the
    /// pass's frame.
    ///
    /// The scope is fork-join, so this call is the drain's share of the
    /// planning and its wait for the helpers' — during which the recorder runs
    /// the draws the loop already submitted, which is the whole point.
    pub(crate) fn ahead<H: crate::runtime::host::HostOps>(
        &mut self,
        state: &DeviceState,
        host: &H,
        di: usize,
    ) {
        if di < self.offered {
            return;
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
                let Some(resource) = bind.resource.clone() else {
                    continue;
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
                crate::runtime::drain::note_store_route_n("lab_prep_sync_tokens", tokens.len() as u64);
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
        let memo_budget =
            std::sync::atomic::AtomicU64::new(sample_plan::memo_fill_chunk_limit());
        let scoped = pool.scope(state, &items, |st, env, item| {
            let mut host = env.host();
            match item {
                PrepItem::Sample {
                    texture_ref,
                    resource,
                } => PrepOutcome::Sample(prefetch::plan_item(
                    st,
                    env,
                    task_id,
                    *texture_ref,
                    resource.clone(),
                    &memo_budget,
                )),
                PrepItem::Buffer {
                    buffer_ref,
                    resource,
                } => PrepOutcome::Buffer(buffer_plan::plan_buffer_rails(
                    st,
                    &mut host,
                    task_id,
                    *buffer_ref,
                    resource.clone(),
                )),
                PrepItem::Index { info, .. } => {
                    PrepOutcome::Buffer(buffer_plan::plan_index_content(st, &mut host, task_id, info))
                }
                PrepItem::StorePages { key } => {
                    PrepOutcome::Store(buffer_plan::plan_store_pages(st, &host, task_id, *key))
                }
            }
        });
        crate::runtime::drain::note_store_route_n("pass_prep_fork_us", scoped.fork_us);
        crate::runtime::drain::note_store_route_n("pass_prep_wait_us", scoped.wait_us);
        crate::runtime::drain::note_store_route_n("pass_prep_helper_us", scoped.helper_busy_us);
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
            match item {
                PrepItem::Sample { texture_ref, .. } => {
                    self.planned.insert(*texture_ref);
                    match result {
                        Ok(PrepOutcome::Sample(Ok(plan))) => {
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
                    self.planned_buffers.insert(*buffer_ref);
                    match result {
                        Ok(PrepOutcome::Buffer(Ok(plan))) => {
                            crate::runtime::drain::note_store_route(buffer_plan::planned_route(&plan));
                            buffer_rails.push(((task_id, *buffer_ref), plan));
                        }
                        Ok(PrepOutcome::Buffer(Err(bail))) => {
                            crate::runtime::drain::note_store_route(buffer_plan::bail_route(bail))
                        }
                        _ => crate::runtime::drain::note_store_route("pass_prep_buf_panic"),
                    }
                }
                PrepItem::Index { key, .. } => {
                    self.planned_indexes.insert(*key);
                    match result {
                        Ok(PrepOutcome::Buffer(Ok(plan))) => {
                            crate::runtime::drain::note_store_route(buffer_plan::planned_route(&plan));
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
            }
        }
        prefetch::merge_into_frame(plans);
        buffer_plan::merge_buffer_plans(buffer_rails, buffer_exacts, store_plan);
    }
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
    fn pooled_state_over_table(fill: impl FnOnce(&mut crate::qemu::host_ops::ReimsVgpuHostOps)) -> DeviceState {
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
    /// one leaked `FakeHost` — the drain's own host, borrowed — beside buffer
    /// objects at `(ref, size)` over allocations starting at task GVA 0x4000.
    /// The guest RAM behind the task's pages is one contiguous range, so a
    /// window's `map_pages` answer is the same whichever arm walks it.
    fn pooled_buffer_state(
        host: &'static mut FakeHost,
        specs: &[(u32, u64)],
    ) -> DeviceState {
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
            assert!(
                gva_mem::write_task_gva(host, &state.tasks[1], desc_gva, &bdesc, PAGE_SHIFT_X86)
                    .is_ok()
            );
            let list_off = list_object_entry_offset(buffer_ref, 64).unwrap();
            let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
            st32(&mut list_entry, packed);
            list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
            assert!(
                gva_mem::write_task_gva(host, &state.tasks[1], list_off, &list_entry, PAGE_SHIFT_X86)
                    .is_ok()
            );
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
        host.map_range(0x10000, 0xA000, 0x40);
        for i in 0..36u32 {
            let pfn = 16 + i;
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
        }
        host
    }

    /// A complete buffer-fixture pair: the leaked host (the drain's, mutable)
    /// and the pooled state whose workers read it.
    fn pooled_buffer_fixture(specs: &[(u32, u64)]) -> (DeviceState, &'static mut FakeHost) {
        let host = Box::leak(Box::new(buffer_host()));
        let state = pooled_buffer_state(host, specs);
        (state, host)
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
                let resource = crate::runtime::objects::resolve_resource(
                    &*state, &*host, 1, buffer_ref,
                )
                .expect("fixture resource");
                let _prime = load_buffer_content_serial(
                    state, host, 1, buffer_ref, Some(&resource), 0, true, None,
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
                            crate::runtime::objects::resolve_resource(
                                state, host, 1, buffer_ref,
                            )
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
                let serial_bind = &serial_binds[di];
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
        let before_planned = store_route_count("pass_prep_buf_planned_packed");
        let before_hit = store_route_count("pass_prep_buf_hit");
        let mut prep = PassPrep::open_with_chunk(&planned_state, 1, tables, 4).expect("opens");
        for di in 0..pattern.len() {
            prep.ahead(&planned_state, &FakeHost::new(), di);
            let bind = &planned_binds[di];
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
        assert!(delta("pass_prep_buf_planned_packed", before_planned) >= 1);
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
            let resource =
                crate::runtime::objects::resolve_resource(&state, &*host, 1, buffer_ref)
                    .expect("fixture resource");
            let _prime = load_buffer_content_serial(
                &mut state, &mut *host, 1, buffer_ref, Some(&resource), 0, true, None,
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
            let resource =
                objects::resolve_resource(&planned_state, &planned_host, 1, memo_ref)
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
            let resource =
                objects::resolve_resource(&planned_state, &planned_host, 1, texture_ref)
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
                                (aid.as_ref().map(|i| i.key), al.layout(), std::mem::discriminant(&ao)),
                                (bid.as_ref().map(|i| i.key), bl.layout(), std::mem::discriminant(&bo)),
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
            let s = crate::runtime::surface_cache::get_shared_with_gen(
                &serial_state,
                surface_id,
                4,
                3,
            );
            assert_eq!(p.map(|(_, g)| g), s.map(|(_, g)| g), "cache gen {surface_id}");
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
                m.map(|m| (m.native.clone(), m.rgba[..].to_vec(), m.generation, m.layout))
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
        let serial = resolve_sampled_source(&mut state, &mut host, 1, texture_ref, Some(resource), true);
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
}
