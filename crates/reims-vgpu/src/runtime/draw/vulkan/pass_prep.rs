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
//! # The consumed-ref replan (design H6)
//!
//! A ref-view plan moves the gather witness when it is applied, so it serves
//! one bind; [`prefetch::prefetched_source`] leaves the ref in the frame's
//! `consumed` list. The next `ahead` takes that list and drops those refs from
//! `planned`, so a later chunk that binds the ref again plans it *again* —
//! from the witness's post-commit state — instead of re-serving a plan whose
//! witness readings can no longer hold.

use super::prefetch::{self, PrefetchGuard, StreamPlan};
use super::*;
use std::collections::HashSet;

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
            _ => {}
        }
    }
    *MODE.get_or_init(
        || match std::env::var("REIMS_VGPU_LAB_PASS_PREP").as_deref() {
            Ok("on") => PassPrepMode::On,
            Ok("shadow") => PassPrepMode::Shadow,
            _ => PassPrepMode::Off,
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
pub(crate) struct DrawTables<'a> {
    pub fragment: &'a [TextureBind],
    pub vertex: &'a [TextureBind],
}

/// One planned bind candidate: the ref and the resource the walk retained.
struct PrepItem {
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
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
            _frame: prefetch::push_empty_frame(),
        })
    }

    /// At the top of record `di`: when `di >= offered`, plan the new refs of
    /// draws `[offered, offered + chunk)` on the pool and merge them into the
    /// pass's frame.
    ///
    /// The scope is fork-join, so this call is the drain's share of the
    /// planning and its wait for the helpers' — during which the recorder runs
    /// the draws the loop already submitted, which is the whole point.
    pub(crate) fn ahead(&mut self, state: &DeviceState, di: usize) {
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
        let mut items = Vec::new();
        for table in &self.tables[self.offered..end] {
            for bind in table.fragment.iter().chain(table.vertex.iter()) {
                if bind.texture_ref == 0 || self.planned.contains(&bind.texture_ref) {
                    continue;
                }
                let Some(resource) = bind.resource.clone() else {
                    continue;
                };
                if seen.insert(bind.texture_ref) {
                    items.push(PrepItem {
                        texture_ref: bind.texture_ref,
                        resource,
                    });
                }
            }
        }
        self.offered = end;
        if items.len() < 2 {
            // As with the stream prefetch: one ref gains nothing from the
            // pool, and planning it here would only double its resolve.
            crate::runtime::drain::note_store_route("pass_prep_chunk_small");
            return;
        }
        let Some(pool) = state.par() else {
            return;
        };
        crate::runtime::drain::note_store_route("pass_prep_chunk");
        crate::runtime::drain::note_store_route_n("pass_prep_items", items.len() as u64);
        // The overlap this driver exists to prove: draws still recording while
        // the pool plans the next chunk.
        crate::runtime::drain::note_store_route_n(
            "pass_prep_recorder_outstanding_n",
            u64::from(crate::backend::vulkan::engine::recorder_outstanding()),
        );
        let task_id = self.task_id;
        let scoped = pool.scope(state, &items, |st, env, item| {
            prefetch::plan_item(st, env, task_id, item.texture_ref, item.resource.clone())
        });
        crate::runtime::drain::note_store_route_n("pass_prep_fork_us", scoped.fork_us);
        crate::runtime::drain::note_store_route_n("pass_prep_wait_us", scoped.wait_us);
        crate::runtime::drain::note_store_route_n("pass_prep_helper_us", scoped.helper_busy_us);
        let mut plans = Vec::with_capacity(items.len());
        for (item, result) in items.iter().zip(scoped.results) {
            // A bail is as final as a plan for this pass: the ladder below it
            // would refuse the same way on the next chunk, so it is not asked
            // again. A consumed ref-view plan leaves `planned` through the
            // consumed list, not here.
            self.planned.insert(item.texture_ref);
            match result {
                Ok(Ok(plan)) => {
                    crate::runtime::drain::note_store_route(match plan {
                        StreamPlan::Surface(_) => "par_prefetch_planned",
                        StreamPlan::RefView(_) => "par_prefetch_planned_ref_view",
                    });
                    plans.push(((task_id, item.texture_ref), plan));
                }
                Ok(Err(bail)) => {
                    crate::runtime::drain::note_store_route(prefetch::bail_route(bail))
                }
                Err(_) => crate::runtime::drain::note_store_route("par_prefetch_panic"),
            }
        }
        prefetch::merge_into_frame(plans);
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
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut table = crate::qemu::host_ops::ReimsVgpuHostOps::null();
        unsafe extern "C" fn nothing(_ctx: *mut std::os::raw::c_void) {}
        table.thread_attach = Some(nothing);
        table.thread_detach = Some(nothing);
        table.map_pages_stable = 1;
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
                            .expect("fixture resource"),
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
            prep.ahead(&state, di);
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
    /// host cache each answer came from and the mappings' epochs — is left
    /// where the serial ladder leaves it.
    #[test]
    fn chunked_answers_equal_serial() {
        let _mode = test_mode(PassPrepMode::On);
        let refs = [2u32, 3, 4, 5, 6];
        let pattern: Vec<u32> = vec![
            0, 1, 2, 0, 3, 4, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1,
        ];
        let (mut planned_state, mut planned_host) = multi_ref_fixture(&refs);
        let (mut serial_state, mut serial_host) = multi_ref_fixture(&refs);
        let planned_binds = binds_for(&planned_state, &planned_host, &pattern, &refs);
        let planned_tables = tables_of(&planned_binds);
        let mut prep =
            PassPrep::open_with_chunk(&planned_state, 1, planned_tables, 4).expect("opens");
        for di in 0..pattern.len() {
            prep.ahead(&planned_state, di);
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
        for (i, _) in refs.iter().enumerate() {
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
        prep.ahead(&state, 0);
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

    /// A ref-view plan consumed by one chunk's bind is planned again by the
    /// next chunk that binds the ref — from the witness state the consumed
    /// plan's commit left — and serves again (design H6's removal).
    #[test]
    fn consumed_ref_view_is_replanned() {
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
        prep.ahead(&state, 0);
        assert_eq!(delta("par_prefetch_planned_ref_view", before_planned), 1);
        // Draw 0 consumes the view plan: the commit moves the witness.
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
        // view ref is consumed but not bound here, so it is *not* replanned yet.
        prep.ahead(&state, 2);
        assert_eq!(
            delta("par_prefetch_planned_ref_view", before_planned),
            1,
            "no bind asked for the view ref, so no replan"
        );
        // Chunk 2 = draws 4..6: binds the view ref (consumed, so new again)
        // beside the second cache ref — the chunk forks and replans the view
        // ref from the witness's post-commit state.
        prep.ahead(&state, 4);
        assert_eq!(
            delta("par_prefetch_planned_ref_view", before_planned),
            2,
            "the consumed view ref is replanned"
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
            "the replanned view ref serves again"
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
            },
            DrawTables {
                fragment: &[],
                vertex: &[],
            },
        ];
        assert!(
            PassPrep::open_with_chunk(&bare, 1, two, 4).is_none(),
            "no pool, no preparation"
        );
    }
}
