//! TEMP (lab, WALK_AHEAD, design journal/2026-10-10-frame-region-design.md and
//! 2026-10-10-speculative-walk-design.md): a stream's render walk on a pool
//! helper, ahead of its packet's execution, and its use by that execution.
//!
//! The walk is the same code: the render handlers are generic over
//! [`WalkState`], whose drain implementation is today's calls and whose
//! [`SpecWalk`] implementation answers only from what is already declared,
//! constructed and latched - the warm answers, read without the writes the
//! drain's calls make on a cold one. A cold answer, a non-render segment, an
//! indirect draw or a refused frame abandons the walk (`None`), and the
//! execution walks the stream itself, exactly as without this.
//!
//! What a spec walk read is its [`Witness`]: every name it resolved with the
//! resource it found, and every backing it answered with the mapping state that
//! answer rests on. The execution uses the walk only when every one of those
//! still holds - checked against the state the drain's own walk would read at
//! that moment, so a packet between the two that renamed, rebuilt or retired
//! one of them sends the stream back to the drain.
//!
//! Fences are the one write a render walk makes that is not a cache: they are
//! recorded and replayed, in order, by the execution that uses the walk.

use std::sync::Arc;

use super::{ExecResult, StreamAccum};
use crate::model::{DeviceState, TaskResource};
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::objects;
use crate::runtime::plan::event_sync::{Domain as FenceDomain, FenceAction};
use reims_vgpu_protocol::segment::{SegmentBody, SegmentKind, SegmentStream};

/// What a render walk does to the device beyond reading it.
pub(super) trait WalkState {
    fn shared(&self) -> &DeviceState;
    fn walk_resource<M: HostMemory>(
        &mut self,
        host: &M,
        task_id: u32,
        obj_ref: u32,
    ) -> Option<Arc<TaskResource>>;
    fn walk_mapper_ref_texture<M: HostMemory>(
        &mut self,
        host: &M,
        task_id: u32,
        obj_ref: u32,
    ) -> Option<u32>;
    fn walk_mapper_ref_texture_resource(
        &mut self,
        task_id: u32,
        obj_ref: u32,
        resource: &TaskResource,
    ) -> Option<u32>;
    fn walk_backing<M: HostMemory>(&mut self, host: &M, surface_id: u32) -> bool;
    fn walk_fence(&mut self, task_id: u32, domain: FenceDomain, fence_ref: u32, action: FenceAction);
    /// Whether an indirect draw may run here (it reads guest argument memory
    /// a preceding writeback may still owe).
    fn walk_indirect_allowed(&mut self) -> bool;
}

impl WalkState for DeviceState {
    fn shared(&self) -> &DeviceState {
        self
    }

    fn walk_resource<M: HostMemory>(
        &mut self,
        host: &M,
        task_id: u32,
        obj_ref: u32,
    ) -> Option<Arc<TaskResource>> {
        objects::resolve_resource(self, host, task_id, obj_ref).ok()
    }

    fn walk_mapper_ref_texture<M: HostMemory>(
        &mut self,
        host: &M,
        task_id: u32,
        obj_ref: u32,
    ) -> Option<u32> {
        objects::resolve_mapper_ref_texture(self, host, task_id, obj_ref)
    }

    fn walk_mapper_ref_texture_resource(
        &mut self,
        task_id: u32,
        obj_ref: u32,
        resource: &TaskResource,
    ) -> Option<u32> {
        objects::resolve_mapper_ref_texture_resource(self, task_id, obj_ref, resource)
    }

    fn walk_backing<M: HostMemory>(&mut self, host: &M, surface_id: u32) -> bool {
        objects::resolve_backing(self, host, surface_id)
    }

    fn walk_fence(&mut self, task_id: u32, domain: FenceDomain, fence_ref: u32, action: FenceAction) {
        let _ = crate::runtime::fence_exec::execute_fence(self, task_id, domain, fence_ref, action);
    }

    fn walk_indirect_allowed(&mut self) -> bool {
        true
    }
}

/// One resolved name a spec walk read.
#[derive(Debug, Clone)]
struct NameRead {
    task_id: u32,
    obj_ref: u32,
    /// `None`: the namespace held no constructed resource (the walk aborted on
    /// it, so a finished walk has none of these).
    resource: Option<usize>,
}

/// One backing a spec walk answered `true` for, and what that rested on.
#[derive(Debug, Clone)]
struct BackingRead {
    surface_id: u32,
    owner_task: u32,
    map_generation: u32,
    pages: usize,
}

/// Everything a spec walk's answers rest on.
#[derive(Debug, Clone, Default)]
pub(crate) struct Witness {
    names: Vec<NameRead>,
    backings: Vec<BackingRead>,
    /// The names generation when a probe found no backing for a surface
    /// (guest object lists read): it must not have moved.
    no_backing_gen: Option<u64>,
}

impl Witness {
    /// Whether every read still answers the same, against `state` as the
    /// drain's own walk would read it now.
    fn holds(&self, state: &DeviceState) -> bool {
        self.names.iter().all(|n| {
            let now = objects::resolve_resource_peek(state, n.task_id, n.obj_ref)
                .map(|r| Arc::as_ptr(&r) as usize);
            now == n.resource
        }) && self.no_backing_gen.is_none_or(|g| state.names_gen() == g)
            && self.backings.iter().all(|b| {
            state.mappings.get(&b.surface_id).is_some_and(|m| {
                m.owner_task_hint == b.owner_task
                    && m.map_generation == b.map_generation
                    && m.page_entries.len() == b.pages
                    && m.mapped
                    && m.has_geom
            })
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SpecFence {
    task_id: u32,
    domain: FenceDomain,
    fence_ref: u32,
    action: FenceAction,
}

/// A walk on a helper: shared state, warm answers only.
pub(super) struct SpecWalk<'a> {
    st: &'a DeviceState,
    aborted: Option<&'static str>,
    fences: Vec<SpecFence>,
    witness: Witness,
}

impl SpecWalk<'_> {
    fn abort(&mut self, why: &'static str) {
        if self.aborted.is_none() {
            self.aborted = Some(why);
        }
    }

    fn note_name(&mut self, task_id: u32, obj_ref: u32, resource: Option<&Arc<TaskResource>>) {
        self.witness.names.push(NameRead {
            task_id,
            obj_ref,
            resource: resource.map(|r| Arc::as_ptr(r) as usize),
        });
    }
}

impl WalkState for SpecWalk<'_> {
    fn shared(&self) -> &DeviceState {
        self.st
    }

    fn walk_resource<M: HostMemory>(
        &mut self,
        _host: &M,
        task_id: u32,
        obj_ref: u32,
    ) -> Option<Arc<TaskResource>> {
        let found = objects::resolve_resource_peek(self.st, task_id, obj_ref);
        if found.is_none() {
            self.abort("cold_resource");
        }
        self.note_name(task_id, obj_ref, found.as_ref());
        found
    }

    fn walk_mapper_ref_texture<M: HostMemory>(
        &mut self,
        host: &M,
        task_id: u32,
        obj_ref: u32,
    ) -> Option<u32> {
        let resource = self.walk_resource(host, task_id, obj_ref)?;
        self.walk_mapper_ref_texture_resource(task_id, obj_ref, &resource)
    }

    fn walk_mapper_ref_texture_resource(
        &mut self,
        _task_id: u32,
        _obj_ref: u32,
        resource: &TaskResource,
    ) -> Option<u32> {
        match objects::resolve_mapper_ref_texture_resource_peek(resource) {
            Some(answer) => answer,
            None => {
                self.abort("cold_mapper_ref");
                None
            }
        }
    }

    fn walk_backing<M: HostMemory>(&mut self, host: &M, surface_id: u32) -> bool {
        match objects::resolve_backing_peek(self.st, host, surface_id) {
            Some(true) => {
                if let Some(m) = self.st.mappings.get(&surface_id) {
                    self.witness.backings.push(BackingRead {
                        surface_id,
                        owner_task: m.owner_task_hint,
                        map_generation: m.map_generation,
                        pages: m.page_entries.len(),
                    });
                }
                true
            }
            Some(false) => {
                self.witness.no_backing_gen.get_or_insert(self.st.names_gen());
                false
            }
            None => {
                self.abort("cold_backing");
                false
            }
        }
    }

    fn walk_fence(&mut self, task_id: u32, domain: FenceDomain, fence_ref: u32, action: FenceAction) {
        self.fences.push(SpecFence {
            task_id,
            domain,
            fence_ref,
            action,
        });
    }

    fn walk_indirect_allowed(&mut self) -> bool {
        self.abort("indirect_draw");
        false
    }
}

/// One stream walked on a helper.
#[derive(Debug)]
pub(crate) struct SpecStream {
    acc: StreamAccum,
    out: ExecResult,
    fences: Vec<SpecFence>,
    witness: Witness,
}

/// Walk one stream's render records under the shared state, or `None` when
/// anything in it needs the drain (see the module doc). Census routes
/// `walk_ahead_abort_<why>` say what stopped one.
pub(crate) fn walk_stream_spec<M: HostMemory + HostOps>(
    st: &DeviceState,
    host: &M,
    task_id: u32,
    stream: &[u8],
) -> Option<SpecStream> {
    let mut segments = SegmentStream::new(stream).ok()?;
    let mut spec = SpecWalk {
        st,
        aborted: None,
        fences: Vec::new(),
        witness: Witness::default(),
    };
    let mut out = ExecResult::default();
    let mut acc = StreamAccum::default();
    for framed in segments.by_ref() {
        let Ok(framed) = framed else {
            crate::runtime::drain::note_store_route("walk_ahead_abort_frame");
            return None;
        };
        let (kind, commands) = match framed.body {
            SegmentBody::Encoder { kind, commands } => (kind, commands),
            SegmentBody::ProtectionEnvelope { .. } => continue,
        };
        if !matches!(kind, SegmentKind::Render) {
            crate::runtime::drain::note_store_route("walk_ahead_abort_segment");
            return None;
        }
        super::walk_segment_records(kind, framed.commands_offset, commands, |op, cmd| {
            if spec.aborted.is_none() {
                // TEMP (lab, FRAME_GATE): a helper's walk gives way to a
                // closing gate (the drain walks this stream itself).
                if crate::runtime::par::gate::job_should_stop() {
                    spec.aborted = Some("gate_closing");
                    return;
                }
                super::handle_render_record(&mut spec, host, task_id, op, cmd, &mut out, &mut acc);
            }
        });
        if let Some(why) = spec.aborted {
            crate::runtime::drain::note_store_route(match why {
                "cold_resource" => "walk_ahead_abort_cold_resource",
                "cold_mapper_ref" => "walk_ahead_abort_cold_mapper_ref",
                "cold_backing" => "walk_ahead_abort_cold_backing",
                "indirect_draw" => "walk_ahead_abort_indirect",
                "gate_closing" => "walk_ahead_abort_gate_closing",
                _ => "walk_ahead_abort_other",
            });
            return None;
        }
    }
    Some(SpecStream {
        acc,
        out,
        fences: spec.fences,
        witness: spec.witness,
    })
}

/// The stream's accumulator from a spec walk, when its witness holds against
/// `state` now: the walk's other effects (the result fields it wrote, the
/// fences it recorded) are applied to `out` and `state` here, so the caller
/// goes straight to the finish. `None`: walk it on the drain.
pub(super) fn take_walk(
    state: &mut DeviceState,
    spec: SpecStream,
    out: &mut ExecResult,
) -> Option<StreamAccum> {
    if !spec.witness.holds(state) {
        crate::runtime::drain::note_store_route("walk_ahead_stale");
        return None;
    }
    let SpecStream {
        acc,
        out: walked,
        fences,
        ..
    } = spec;
    for r in walked.texture_refs {
        if !out.texture_refs.contains(&r) {
            out.texture_refs.push(r);
        }
    }
    for m in walked.mapper_ref_texture_mappings {
        if !out.mapper_ref_texture_mappings.contains(&m) {
            out.mapper_ref_texture_mappings.push(m);
        }
    }
    out.buffer_unbinds += walked.buffer_unbinds;
    out.texture_unbinds += walked.texture_unbinds;
    out.sampler_unbinds += walked.sampler_unbinds;
    out.saw_draw |= walked.saw_draw;
    for f in fences {
        let _ = crate::runtime::fence_exec::execute_fence(state, f.task_id, f.domain, f.fence_ref, f.action);
    }
    crate::runtime::drain::note_store_route("walk_ahead_used");
    Some(acc)
}

/// TEMP (lab, FRAME_GATE): [`take_walk`] under the shared state: the
/// accumulator and the fences still to apply (exclusive, [`apply_fences`]).
pub(super) fn take_walk_shared(
    state: &DeviceState,
    spec: SpecStream,
    out: &mut ExecResult,
) -> Option<(StreamAccum, Vec<SpecFence>)> {
    if !spec.witness.holds(state) {
        crate::runtime::drain::note_store_route("walk_ahead_stale");
        return None;
    }
    let SpecStream {
        acc,
        out: walked,
        fences,
        ..
    } = spec;
    for r in walked.texture_refs {
        if !out.texture_refs.contains(&r) {
            out.texture_refs.push(r);
        }
    }
    for m in walked.mapper_ref_texture_mappings {
        if !out.mapper_ref_texture_mappings.contains(&m) {
            out.mapper_ref_texture_mappings.push(m);
        }
    }
    out.buffer_unbinds += walked.buffer_unbinds;
    out.texture_unbinds += walked.texture_unbinds;
    out.sampler_unbinds += walked.sampler_unbinds;
    out.saw_draw |= walked.saw_draw;
    crate::runtime::drain::note_store_route("walk_ahead_used");
    if !fences.is_empty() {
        crate::runtime::drain::note_store_route("walk_ahead_fences");
    }
    Some((acc, fences))
}

/// TEMP (lab, FRAME_GATE): the fences a shared take left to the drain.
pub(super) fn apply_fences(state: &mut DeviceState, fences: Vec<SpecFence>) {
    for f in fences {
        let _ = crate::runtime::fence_exec::execute_fence(state, f.task_id, f.domain, f.fence_ref, f.action);
    }
}

/// TEMP (lab, WALK_AHEAD_CHECK): compare a helper's walk with the drain's own
/// walk of the same stream (which ran, and is the one used). Census
/// `walk_ahead_check_same` / `_diff` / `_stale` (the witness did not hold, so
/// the walk would not have been used).
pub(super) fn check_walk(state: &DeviceState, spec: SpecStream, real: &StreamAccum) {
    if !spec.witness.holds(state) {
        crate::runtime::drain::note_store_route("walk_ahead_check_stale");
        return;
    }
    if format!("{:?}", spec.acc) == format!("{real:?}") {
        crate::runtime::drain::note_store_route("walk_ahead_check_same");
    } else {
        crate::runtime::drain::note_store_route("walk_ahead_check_diff");
        if crate::observe::first_sight("walk_ahead_check_diff", 0) {
            crate::observe::fail(format!(
                "walk_ahead_check_diff (a helper's walk and the drain's disagree) spec={:?} real={real:?}",
                spec.acc
            ));
        }
    }
}
