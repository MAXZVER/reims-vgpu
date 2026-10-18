//! TEMP (lab, Opus 2026-10-18, `REIMS_VGPU_LAB_MERGE_DRAWS=on`): runs of draws that differ only in where one
//! vertex buffer starts, executed as one prepared draw plus `vkCmdDrawIndexed` calls that move `vertexOffset`.
//!
//! Mish's Agent Monitor page (and Core Animation pattern fills in general: `CA::OGL::fill_pattern_rect` /
//! `emit_quad`) draws quads one at a time: four new vertices in a ring buffer, the vertex buffer's offset moved to
//! them, the same six indices drawn. Census b756: 1.44 M of 1.65 M neighbouring draws (87 %) differed from the
//! previous one in that offset alone. Each paid the whole per-draw path - binds resolved, the vertex bytes staged,
//! the request built - for one quad.
//!
//! **Why `vertexOffset` is the same draw.** Metal's `vertex_id` on an indexed draw is the index plus `baseVertex`,
//! and Vulkan's `gl_VertexIndex` is the index plus `vertexOffset`; the translated shader reads its vertex buffer at
//! `vertex_id * stride` from the bound offset. A sub-draw whose buffer starts `d * stride` bytes later therefore
//! reads exactly the same bytes with the buffer bound at the run's first offset and `vertexOffset` moved by `d`.
//! That holds only when the moving buffer is the *only* thing the vertex stage indexes by `vertex_id`: any other
//! per-vertex stream would be shifted too. The engine learns that shape per pipeline object the first time it
//! prepares one of its indexed draws ([`learn`]); the walk merges only for a pipeline whose shape it knows
//! ([`shape`]), and only offsets that move by whole elements.
//!
//! The key is the retained resolved pipeline (a weak pointer into `pipeline_resolve`'s registry, which both the walk
//! and the draw see), not its serializer ref alone: a ref the guest deletes and reuses names another pipeline, and a
//! dead weak pointer is not a match.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::backend::vulkan::pipeline_resolve::ResolvedRenderPipeline;

/// Sub-draws a merged draw may carry after its own.
pub const MAX_SUBS: usize = 1024;

/// Sub-draws with their own buffers (each loads its own bytes when the record is prepared) a record may carry:
/// a record is prepared by one worker, so piling every tile of a frame into one record serialises what the parallel
/// preparation spread over the pool (b775: Maps 56 -> 33 fps). `REIMS_VGPU_LAB_MERGE_HEAVY` (default 16). Latched.
pub fn max_heavy_subs() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("REIMS_VGPU_LAB_MERGE_HEAVY").ok().and_then(|v| v.parse().ok()).unwrap_or(16))
}

/// On by default since b777 (plan page 62 vs 50 fps idle, 71 vs 49 dragged; Maps +17 %; frames match);
/// `REIMS_VGPU_LAB_MERGE_DRAWS=off` turns it off. Latched.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = !matches!(std::env::var("REIMS_VGPU_LAB_MERGE_DRAWS").as_deref(), Ok("off" | "0"));
        crate::observe::off(format!("lab_merge_draws on={}", u8::from(on)));
        on
    })
}

/// The one vertex buffer a pipeline's vertex stage indexes by `vertex_id`, and the stride it does so with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub index: u32,
    pub stride: u64,
}

/// One sub-draw's own device-buffer binds, where they differ from its record's.
#[derive(Clone, Debug)]
pub struct SubBinds {
    pub sub: u32,
    pub vertex: Vec<crate::runtime::draw::BufferBind>,
    pub fragment: Vec<crate::runtime::draw::BufferBind>,
    /// Stage-in buffers the sub-draw reads instead of the record's (whole other binds, not a move).
    pub vstage: Vec<crate::runtime::draw::BufferBind>,
    /// The sub-draw's own index buffer range.
    pub index: Option<crate::runtime::draw::IndexedDrawInfo>,
}

impl PartialEq for SubBinds {
    fn eq(&self, other: &Self) -> bool {
        let key = |v: &Vec<crate::runtime::draw::BufferBind>| v.iter().map(|b| (b.index, b.buffer_ref, b.offset)).collect::<Vec<_>>();
        let ik = |i: &Option<crate::runtime::draw::IndexedDrawInfo>| {
            i.as_ref().map(|i| (i.index_type, i.index_count, i.index_buffer_ref, i.index_buffer_offset, i.base_vertex))
        };
        self.sub == other.sub
            && key(&self.vertex) == key(&other.vertex)
            && key(&self.fragment) == key(&other.fragment)
            && key(&self.vstage) == key(&other.vstage)
            && ik(&self.index) == ik(&other.index)
    }
}
impl Eq for SubBinds {}

/// Bytes a merged record's index window may span (the index-bounded vertex fetch reads gathered windows up to
/// 64 KiB on the CPU).
pub const MAX_WINDOW_BYTES: u64 = 64 * 1024;

pub use crate::backend::vulkan::engine::types::LabSubDraw as SubDraw;

/// The sub-draws a merged indexed draw carries after its own.
///
/// The record's `IndexedDrawInfo::index_count` is widened to [`Self::window`] so the index bind covers every
/// sub-draw's indices. The window may hold other bytes between the ranges (Core Animation writes each quad's
/// indices beside its vertices), so the vertex bound is taken per range ([`vertex_max`]), never over the window;
/// the record itself still draws [`Self::head_count`] indices from the window's start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergedSubs {
    /// The shape the vertex-buffer moves were folded under; `None` when no sub-draw moved a vertex buffer.
    pub shape: Option<Shape>,
    /// The stage-in buffer index the sub-draws' `vb_delta` moves (rebind path; exclusive with `shape`).
    pub rebind_index: Option<u32>,
    pub subs: Vec<SubDraw>,
    pub head_count: u32,
    /// Indexed: the record's own firstIndex into the window (a later sub-draw's indices may lie before it).
    pub head_first: u32,
    pub window: u32,
    /// The sub-draws that bind other device buffers than the record (sparse, by sub-draw).
    pub overrides: Vec<SubBinds>,
    /// The record's own scissor once a sub-draw carries another (the record's `scissors` then hold their union).
    pub head_scissor: Option<[u32; 4]>,
    /// The record's own stencil reference once a sub-draw carries another.
    pub head_stencil: Option<(u32, u32)>,
    /// Not indexed: `[first, end)` of the vertices the record and its sub-draws draw (the vertex fetch bound).
    pub span: (u32, u32),
}

type Learned = HashMap<(u32, u32), (Weak<ResolvedRenderPipeline>, Option<Shape>)>;

fn with_learned<R>(f: impl FnOnce(&mut Learned) -> R) -> R {
    static MAP: Mutex<Option<Learned>> = Mutex::new(None);
    let mut g = MAP.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(HashMap::new))
}

/// What the engine found preparing one of `pipeline_ref`'s indexed draws: the shape, or `None` when the vertex
/// stage indexes no buffer or more than one by `vertex_id`, or a stride is unknown.
pub fn learn(task_id: u32, pipeline_ref: u32, object: &Arc<ResolvedRenderPipeline>, shape: Option<Shape>) {
    with_learned(|map| {
        if map.len() > 4096 {
            map.retain(|_, (w, _)| w.strong_count() > 0);
        }
        map.insert((task_id, pipeline_ref), (Arc::downgrade(object), shape));
    });
}

/// Whether the engine already taught this pipeline object's shape (learning is once per object).
pub fn learned(task_id: u32, pipeline_ref: u32, object: &Arc<ResolvedRenderPipeline>) -> bool {
    with_learned(|map| {
        map.get(&(task_id, pipeline_ref))
            .is_some_and(|(weak, _)| std::ptr::eq(weak.as_ptr(), Arc::as_ptr(object)) && weak.strong_count() > 0)
    })
}

/// The learned shape for this pipeline object, when the engine has prepared one of its draws and found it mergeable.
pub fn shape(task_id: u32, pipeline_ref: u32, object: &Arc<ResolvedRenderPipeline>) -> Option<Shape> {
    with_learned(|map| {
        let (weak, shape) = map.get(&(task_id, pipeline_ref))?;
        if !std::ptr::eq(weak.as_ptr(), Arc::as_ptr(object)) || weak.strong_count() == 0 {
            return None;
        }
        *shape
    })
}

/// The merge shape of a pipeline: its vertex stage indexes exactly one vertex buffer by `vertex_id`, with one
/// stride, and nothing it reads per vertex is beyond the reflection's sight. `None` otherwise.
///
/// Per-vertex stage-in attributes index their buffer by `vertex_id` at the layout's stride; device buffers do so
/// wherever the reflected footprint has a `VertexIndex` term. A bound buffer whose footprint is unknown or unbounded
/// could be indexed by `vertex_id` in ways the reflection cannot show, so it rules the pipeline out.
pub fn pipeline_shape(
    attributes: &[crate::runtime::decode::resource::VertexAttribute],
    reflection: &metal2vulkan::reflect::ShaderReflection,
    vertex_buffers: &[crate::runtime::draw::BufferBind],
) -> Result<Shape, &'static str> {
    use metal2vulkan::reflect::{BufferIndexSource, ResourceKind};
    use std::collections::BTreeMap;

    let mut dep: BTreeMap<u32, Option<u64>> = BTreeMap::new();
    let mut add = |index: u32, stride: u64| {
        let e = dep.entry(index).or_insert(Some(stride));
        if *e != Some(stride) {
            *e = None;
        }
    };
    for a in attributes {
        // MTLVertexStepFunction: 0 constant, 1 per vertex (the default), 2 per instance, 3+ tessellation.
        match a.step_function_ordinal(1) {
            0 | 2 => {}
            1 => add(
                a.buffer_index,
                u64::from(crate::runtime::draw::bind_attribute_stride(vertex_buffers, a.buffer_index, a.stride)),
            ),
            _ => return Err("merge_shape_step"),
        }
    }
    for b in &reflection.bindings {
        if b.kind != ResourceKind::Buffer || !vertex_buffers.iter().any(|v| v.index == b.metal_index) {
            continue;
        }
        let Some(fp) = b.footprint.as_ref() else {
            return Err("merge_shape_nofp");
        };
        if fp.has_unbounded_access {
            return Err("merge_shape_unbounded");
        }
        for access in &fp.strided_accesses {
            for t in &access.terms {
                if t.source == BufferIndexSource::VertexIndex {
                    add(b.metal_index, t.stride);
                }
            }
        }
    }
    match dep.len() {
        0 => return Err("merge_shape_deps0"),
        1 => {}
        _ => return Err("merge_shape_depsn"),
    }
    let (index, stride) = dep.into_iter().next().ok_or("merge_shape_deps0")?;
    let stride = stride.filter(|s| *s > 0).ok_or("merge_shape_stride")?;
    Ok(Shape { index, stride })
}

/// Whether `index` is read by the pipeline's vertex stage only through per-vertex stage-in attributes - Vulkan
/// vertex input bindings, whose bound offset a sub-draw may move without touching `vertex_id` - and never as a
/// device buffer.
///
/// This is the rebind path: the guest moved the buffer's offset, and so does the sub-draw. Nothing about
/// `vertex_id` changes, so neither the stride nor quad alignment matters. Constant and per-instance steps are
/// excluded (the engine rebuilds constant streams with a base-instance prefix).
pub fn stage_in_only(object: &ResolvedRenderPipeline, index: u32) -> bool {
    use metal2vulkan::reflect::ResourceKind;
    let mut any = false;
    for a in object.desc.vertex_attributes.iter().filter(|a| a.buffer_index == index) {
        if a.step_function_ordinal(1) != 1 {
            return false;
        }
        any = true;
    }
    any && !object
        .vertex
        .reflection
        .bindings
        .iter()
        .any(|b| b.kind == ResourceKind::Buffer && b.metal_index == index)
}

/// The largest vertex a merged record's draws can fetch: each range's largest index plus its own vertexOffset.
/// `max_of(first, count)` is the largest index of `count` indices from `first` in the window, plus the record's
/// baseVertex, or `None` when unknown (then nothing is bounded).
pub fn vertex_max(m: &MergedSubs, max_of: impl Fn(u32, u32) -> Option<u64>) -> Option<u64> {
    let mut best = max_of(m.head_first, m.head_count)?;
    for s in m.subs.iter().filter(|s| !s.own_index) {
        best = best.max(max_of(s.first, s.count)?.checked_add_signed(i64::from(s.vertex_delta))?);
    }
    Some(best)
}

/// TEMP (lab, MERGE_DRAWS diagnosis): what [`pipeline_shape`] saw, for the once-per-pipeline log line.
pub fn describe(
    attributes: &[crate::runtime::decode::resource::VertexAttribute],
    reflection: &metal2vulkan::reflect::ShaderReflection,
    vertex_buffers: &[crate::runtime::draw::BufferBind],
) -> String {
    let attrs: Vec<String> = attributes
        .iter()
        .map(|a| format!("{}@{}/s{}/f{}", a.buffer_index, a.offset, a.stride, a.step_function_ordinal(1)))
        .collect();
    let vbs: Vec<String> = vertex_buffers.iter().map(|v| format!("{}:{}+{}/{:?}", v.index, v.buffer_ref, v.offset, v.attribute_stride)).collect();
    let binds: Vec<String> = reflection
        .bindings
        .iter()
        .filter(|b| b.kind == metal2vulkan::reflect::ResourceKind::Buffer)
        .map(|b| match &b.footprint {
            None => format!("{}:nofp", b.metal_index),
            Some(fp) => format!(
                "{}:unb{}/st{:?}/acc{:?}",
                b.metal_index,
                u8::from(fp.has_unbounded_access),
                fp.static_ranges.len(),
                fp.strided_accesses.iter().map(|a| a.terms.iter().map(|t| (format!("{:?}", t.source), t.stride)).collect::<Vec<_>>()).collect::<Vec<_>>()
            ),
        })
        .collect();
    format!("attrs=[{}] vbs=[{}] bufs=[{}]", attrs.join(" "), vbs.join(" "), binds.join(" "))
}
