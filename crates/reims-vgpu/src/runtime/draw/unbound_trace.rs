//! Why a fragment texture a shader statically uses reached the draw unbound,
//! and what this device then sampled in its place.
//!
//! `frag_neutral_texture_substituted` (in `draw::vulkan`) says *that* a 1x1
//! neutral image stood in for a slot. It cannot say *why* the slot was empty:
//! a guest that never bound it, a `setFragmentTexture:nil` that cleared it, a
//! bind that went to the vertex table, or state the previous encoder held and
//! this one never re-bound. Those are different findings with different
//! owners, and the v9 Tahoe boot printed 4 372 of them with nothing to tell
//! them apart.
//!
//! This module carries the evidence from the decode walk to the draw:
//!
//! - [`StreamBindLog`] records every raw bind record of one render stream
//!   (bounded, [`STREAM_LOG_CAP`]), tagged with how many draws the stream had
//!   recorded when it arrived, so "since the last draw" is a range and not a
//!   copy per draw.
//! - [`enter_draw`] parks a per-draw context on the drain thread for exactly
//!   the duration of one `encode_draw_chain` call. Draw preparation is
//!   synchronous on that thread, which is what lets the substitution site read
//!   it without a field on every request type.
//! - [`note_substitution`] classifies each substituted slot into a
//!   [`SlotFate`] and counts it (`frag_neutral_tex_reason_*`), always.
//! - [`trace_draw`] writes one bounded `frag_unbound_trace` line for a draw
//!   that substituted, rate-limited per (pipeline, pass class): the first
//!   [`FIRST_PER_KEY`] per key and then one in [`SAMPLE_EVERY`], under a hard
//!   [`MAX_LINES`] per process. Icon-sized passes and material/backdrop passes
//!   are separate keys (`class=icon|material`), so the large passes cannot be
//!   starved by the thousands of icon passes ahead of them.
//!
//! **Observation only.** Nothing here changes what is bound, except the
//! opt-in [`neutral_texel`] probe, whose default is the zero texel the
//! substitution always used.
//!
//! `REIMS_VGPU_TRACE_UNBOUND_BINDS=0` turns off the raw-bind log and the trace
//! line (the per-reason census still counts, as `untraced` where the log would
//! have been needed).

#![cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use reims_vgpu_protocol::render::ShaderStage;

use super::{BindTable, TextureBind};

/// Raw bind records kept per render stream. A stream that binds more than this
/// keeps the first ones and counts the rest as dropped; the classifier then
/// says `log_overflow` rather than claiming a slot was never bound.
pub(crate) const STREAM_LOG_CAP: usize = 512;
/// Trace lines per (pipeline, pass class) before sampling starts.
pub(crate) const FIRST_PER_KEY: u32 = 4;
/// After the first lines of a key, one draw in this many is traced.
pub(crate) const SAMPLE_EVERY: u32 = 256;
/// Hard ceiling on trace lines for the whole process.
pub(crate) const MAX_LINES: u32 = 4000;
/// Distinct keys the limiter remembers. Past it, draws of new keys are only
/// ever sampled.
const MAX_KEYS: usize = 8192;
/// Both pass dimensions at or under this make the pass `class=icon`.
const ICON_MAX_EDGE: u32 = 256;
/// Entries printed per list on one trace line.
const LIST_CAP: usize = 32;

/// Whether the raw-bind log and the trace line run. Default on.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let (state, value) = crate::config::read(crate::config::TRACE_UNBOUND_BINDS);
        match state {
            crate::config::Switch::Off => {
                crate::observe::fail("unbound_bind_trace_disabled");
                false
            }
            crate::config::Switch::Unrecognized => {
                crate::observe::fail(format!(
                    "unbound_bind_trace_env_unrecognized value={} (trace stays on)",
                    value.unwrap_or_default()
                ));
                true
            }
            crate::config::Switch::On | crate::config::Switch::Unset => true,
        }
    })
}

/// The texel the neutral stand-in carries.
///
/// Zero unless `REIMS_VGPU_NEUTRAL_TEXTURE_PROBE` names another value. Zero is
/// what Apple GPUs return for a read of an unbound texture slot and what this
/// device has always bound, so it stays the default; the probe exists only to
/// A/B whether a missing glyph *depends* on the stand-in (a glyph that turns
/// magenta or reappears white under the probe was sampling it; one that stays
/// missing was not).
pub(crate) fn neutral_texel() -> [f64; 4] {
    static TEXEL: OnceLock<[f64; 4]> = OnceLock::new();
    *TEXEL.get_or_init(|| {
        use crate::config::Choice;
        match crate::config::choice(
            crate::config::NEUTRAL_TEXTURE_PROBE,
            &["zero", "white", "magenta"],
        ) {
            Choice::Unset | Choice::Named("zero") => [0.0; 4],
            Choice::Named("white") => {
                crate::observe::fail("neutral_texture_probe value=white (diagnostic, not a fix)");
                [1.0; 4]
            }
            Choice::Named(_) => {
                crate::observe::fail("neutral_texture_probe value=magenta (diagnostic, not a fix)");
                [1.0, 0.0, 1.0, 1.0]
            }
            Choice::Refused(raw) => {
                crate::observe::fail(format!(
                    "neutral_texture_probe_refused value={raw} (zero kept)"
                ));
                [0.0; 4]
            }
        }
    })
}

/// Which argument table a raw bind record wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BindKind {
    Buffer,
    Texture,
    Sampler,
}

impl BindKind {
    fn letter(self) -> char {
        match self {
            Self::Buffer => 'B',
            Self::Texture => 'T',
            Self::Sampler => 'S',
        }
    }
}

/// One slot of one bind record, as the guest sent it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawBind {
    /// Draws the stream had recorded when this bind arrived. A draw whose own
    /// tag is `k` saw every entry tagged `<= k`; the ones tagged exactly `k`
    /// arrived after the previous recorded draw.
    pub draw_tag: u32,
    pub kind: BindKind,
    pub stage: ShaderStage,
    pub index: u32,
    /// The wire ref. Zero is the guest clearing the slot.
    pub object_ref: u32,
}

/// Every raw bind record of one render stream, bounded.
#[derive(Clone, Debug, Default)]
pub(crate) struct StreamBindLog {
    entries: Vec<RawBind>,
    dropped: u32,
}

impl StreamBindLog {
    /// Record one bind record's slots. A no-op when the trace is off, so the
    /// decode walk pays one cached load per record.
    pub(crate) fn record(
        &mut self,
        draw_tag: u32,
        kind: BindKind,
        stage: ShaderStage,
        first: u32,
        refs: impl Iterator<Item = u32>,
    ) {
        if !enabled() {
            return;
        }
        for (i, object_ref) in refs.enumerate() {
            if self.entries.len() >= STREAM_LOG_CAP {
                self.dropped = self.dropped.saturating_add(1);
                continue;
            }
            self.entries.push(RawBind {
                draw_tag,
                kind,
                stage,
                index: first.saturating_add(i as u32),
                object_ref,
            });
        }
    }

    /// The log as the draws of this stream will read it. `None` when the
    /// trace is off. Called once per stream, never per draw.
    pub(crate) fn snapshot(&self) -> Option<Arc<StreamBindLog>> {
        enabled().then(|| Arc::new(self.clone()))
    }

    #[cfg(test)]
    pub(crate) fn entries(&self) -> &[RawBind] {
        &self.entries
    }
}

/// Where a traced draw came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawOrigin {
    /// A decoded draw record; carries its tag.
    Direct(u32),
    /// An ICB execute, which inherits the stream's tables.
    Icb,
}

/// What the substitution site needs to know about the draw it is preparing.
#[derive(Clone, Debug)]
pub(crate) struct DrawBindContext {
    pub log: Option<Arc<StreamBindLog>>,
    pub origin: DrawOrigin,
    /// The previous stream's final fragment texture table in the same
    /// submission. Metal resets encoder state per encoder, so a slot that
    /// was bound there and not here is *not* a bind this device lost; the
    /// fate exists to tell a guest that relied on it from one that never had it.
    pub prior_frag_textures: BindTable<TextureBind>,
    pub stream_seq: u32,
}

thread_local! {
    static CONTEXT: RefCell<Option<DrawBindContext>> = const { RefCell::new(None) };
}

/// Clears the parked context when the draw it describes is done.
pub(crate) struct DrawContextGuard {
    armed: bool,
}

impl Drop for DrawContextGuard {
    fn drop(&mut self) {
        if self.armed {
            CONTEXT.with(|c| *c.borrow_mut() = None);
        }
    }
}

/// Park `ctx` for the draw about to be encoded on this thread. A no-op guard
/// when the trace is off.
pub(crate) fn enter_draw(ctx: impl FnOnce() -> DrawBindContext) -> DrawContextGuard {
    if !enabled() {
        return DrawContextGuard { armed: false };
    }
    let ctx = ctx();
    CONTEXT.with(|c| *c.borrow_mut() = Some(ctx));
    DrawContextGuard { armed: true }
}

fn with_context<R>(f: impl FnOnce(Option<&DrawBindContext>) -> R) -> R {
    CONTEXT.with(|c| f(c.borrow().as_ref()))
}

/// Why one statically-used fragment texture slot reached the draw unbound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotFate {
    /// The draw's table holds the slot. Not a substitution.
    Bound,
    /// No fragment texture record of this stream named the slot.
    NeverBound,
    /// The last fragment texture record naming the slot carried ref zero.
    ClearedByNil,
    /// Only the vertex table was bound at this index.
    VertexStageOnly,
    /// The previous encoder of this submission had it; this one never did.
    PriorEncoderHad(u32),
    /// The raw log saw a live ref for the slot that the draw's table does not
    /// hold: a decode-side loss, and the one fate that would be a bug here.
    BoundRefLost(u32),
    /// The log overflowed before it could see the slot.
    LogOverflow,
    /// No context (trace off, or a draw prepared outside `finish_stream`).
    Untraced,
}

impl SlotFate {
    pub(crate) fn slug(self) -> &'static str {
        match self {
            Self::Bound => "bound",
            Self::NeverBound => "never_bound",
            Self::ClearedByNil => "cleared_by_nil",
            Self::VertexStageOnly => "vertex_stage_only",
            Self::PriorEncoderHad(_) => "never_bound_prior_encoder_had",
            Self::BoundRefLost(_) => "dropped_bound_ref_lost",
            Self::LogOverflow => "log_overflow",
            Self::Untraced => "untraced",
        }
    }

    fn route(self) -> &'static str {
        match self {
            Self::Bound => "frag_neutral_tex_reason_bound",
            Self::NeverBound => "frag_neutral_tex_reason_never_bound",
            Self::ClearedByNil => "frag_neutral_tex_reason_cleared_by_nil",
            Self::VertexStageOnly => "frag_neutral_tex_reason_vertex_stage_only",
            Self::PriorEncoderHad(_) => "frag_neutral_tex_reason_never_bound_prior_encoder_had",
            Self::BoundRefLost(_) => "frag_neutral_tex_reason_dropped_bound_ref_lost",
            Self::LogOverflow => "frag_neutral_tex_reason_log_overflow",
            Self::Untraced => "frag_neutral_tex_reason_untraced",
        }
    }
}

impl std::fmt::Display for SlotFate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PriorEncoderHad(r) => write!(f, "never_bound(prior_encoder_ref={r:#x})"),
            Self::BoundRefLost(r) => write!(f, "dropped(reason=bound_ref_lost ref={r:#x})"),
            Self::Bound => f.write_str("bound"),
            other => f.write_str(other.slug()),
        }
    }
}

/// Classify one fragment texture slot against the draw's table and the
/// stream's raw log.
pub(crate) fn classify(
    ctx: Option<&DrawBindContext>,
    fragment_textures: &[TextureBind],
    index: u32,
) -> SlotFate {
    if fragment_textures
        .iter()
        .any(|t| t.index == index && t.texture_ref != 0)
    {
        return SlotFate::Bound;
    }
    let Some(ctx) = ctx else {
        return SlotFate::Untraced;
    };
    let prior = || {
        ctx.prior_frag_textures
            .iter()
            .find(|t| t.index == index && t.texture_ref != 0)
            .map(|t| t.texture_ref)
    };
    let Some(log) = ctx.log.as_ref() else {
        return prior().map_or(SlotFate::Untraced, SlotFate::PriorEncoderHad);
    };
    let seen = |e: &&RawBind| match ctx.origin {
        DrawOrigin::Direct(tag) => e.draw_tag <= tag,
        DrawOrigin::Icb => true,
    };
    let last_frag = log
        .entries
        .iter()
        .filter(seen)
        .filter(|e| {
            e.kind == BindKind::Texture && e.stage == ShaderStage::Fragment && e.index == index
        })
        .last();
    if let Some(e) = last_frag {
        return if e.object_ref == 0 {
            SlotFate::ClearedByNil
        } else {
            SlotFate::BoundRefLost(e.object_ref)
        };
    }
    if log.entries.iter().filter(seen).any(|e| {
        e.kind == BindKind::Texture
            && e.stage == ShaderStage::Vertex
            && e.index == index
            && e.object_ref != 0
    }) {
        return SlotFate::VertexStageOnly;
    }
    if let Some(r) = prior() {
        return SlotFate::PriorEncoderHad(r);
    }
    if log.dropped > 0 {
        return SlotFate::LogOverflow;
    }
    SlotFate::NeverBound
}

/// Whether the shader's AIR asks `air.is_null_texture*`, keyed by the cached
/// shader's reflection identity and module size.
///
/// metal2vulkan lowers that query to a constant `false` for every real entry
/// texture (`passes/air_calls/dispatch_texture.rs` `lower_is_null_texture`):
/// only values it synthesized from `air.get_null_texture_*` answer `true`. In
/// Metal an unbound slot *is* the null texture, so a shader that branches on
/// it takes its "texture present" path here and samples the stand-in. That
/// population is the one where the neutral value is load-bearing.
///
/// The value holds a `Weak` to the reflection it was keyed by, so an address
/// reused after an eviction reads as unknown rather than as another shader's
/// answer.
type IsNullEntry = (std::sync::Weak<metal2vulkan::reflect::ShaderReflection>, bool);
fn is_null_registry() -> &'static Mutex<HashMap<(usize, usize), IsNullEntry>> {
    static REG: OnceLock<Mutex<HashMap<(usize, usize), IsNullEntry>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Remember whether one freshly translated fragment shader queries
/// `is_null_texture`. Once per translation, never per draw.
pub(crate) fn note_translated_fragment(
    reflection: &Arc<metal2vulkan::reflect::ShaderReflection>,
    spirv_len: usize,
    air: &[u8],
) {
    const NEEDLE: &[u8] = b"air.is_null_texture";
    let queries = air.windows(NEEDLE.len()).any(|w| w == NEEDLE);
    if queries {
        crate::runtime::drain::note_store_route("frag_shader_queries_is_null_texture");
    }
    let mut reg = is_null_registry().lock().unwrap_or_else(|e| e.into_inner());
    if reg.len() >= MAX_KEYS {
        reg.retain(|_, (weak, _)| weak.strong_count() > 0);
    }
    if reg.len() < MAX_KEYS {
        reg.insert(
            (Arc::as_ptr(reflection) as usize, spirv_len),
            (Arc::downgrade(reflection), queries),
        );
    }
}

fn is_null_query(
    reflection: &Arc<metal2vulkan::reflect::ShaderReflection>,
    spirv_len: usize,
) -> Option<bool> {
    is_null_registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(Arc::as_ptr(reflection) as usize, spirv_len))
        .filter(|(weak, _)| weak.upgrade().is_some_and(|live| Arc::ptr_eq(&live, reflection)))
        .map(|&(_, queries)| queries)
}

/// Count one performed substitution by why the slot was empty, by slot, and
/// by whether the shader can tell (see [`note_translated_fragment`]). Always
/// on; one thread-local read and three counter bumps.
pub(crate) fn note_substitution(
    fragment_textures: &[TextureBind],
    index: u32,
    reflection: &Arc<metal2vulkan::reflect::ShaderReflection>,
    spirv_len: usize,
) -> SlotFate {
    let fate = with_context(|ctx| classify(ctx, fragment_textures, index));
    crate::runtime::drain::note_store_route(fate.route());
    crate::runtime::drain::note_store_route(match index {
        0 => "frag_neutral_tex_slot_0",
        1 => "frag_neutral_tex_slot_1",
        2 => "frag_neutral_tex_slot_2",
        3 => "frag_neutral_tex_slot_3",
        _ => "frag_neutral_tex_slot_ge4",
    });
    crate::runtime::drain::note_store_route(match is_null_query(reflection, spirv_len) {
        Some(true) => "frag_neutral_tex_shader_queries_is_null",
        Some(false) => "frag_neutral_tex_shader_no_is_null_query",
        None => "frag_neutral_tex_shader_is_null_unknown",
    });
    fate
}

/// Pass class for the limiter and the line.
pub(crate) fn pass_class(w: u32, h: u32) -> &'static str {
    if w <= ICON_MAX_EDGE && h <= ICON_MAX_EDGE {
        "icon"
    } else {
        "material"
    }
}

/// The rate limiter behind [`trace_draw`], as a value so it can be tested
/// without the process-global one.
#[derive(Debug, Default)]
pub(crate) struct Limiter {
    per_key: HashMap<(u32, &'static str), u32>,
    unkeyed: u32,
    lines: u32,
}

impl Limiter {
    /// `Some((sample, seen))` when this draw gets a line.
    pub(crate) fn admit(&mut self, pipe: u32, class: &'static str) -> Option<(&'static str, u32)> {
        if self.lines >= MAX_LINES {
            return None;
        }
        let seen = if let Some(n) = self.per_key.get_mut(&(pipe, class)) {
            *n = n.saturating_add(1);
            *n
        } else if self.per_key.len() < MAX_KEYS {
            self.per_key.insert((pipe, class), 1);
            1
        } else {
            self.unkeyed = self.unkeyed.saturating_add(1);
            return (self.unkeyed % SAMPLE_EVERY == 1).then(|| {
                self.lines += 1;
                ("sampled_unkeyed", self.unkeyed)
            });
        };
        let sample = if seen <= FIRST_PER_KEY {
            "first"
        } else if (seen - FIRST_PER_KEY) % SAMPLE_EVERY == 0 {
            "sampled"
        } else {
            return None;
        };
        self.lines += 1;
        Some((sample, seen))
    }
}

fn limiter() -> &'static Mutex<Limiter> {
    static L: OnceLock<Mutex<Limiter>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(Limiter::default()))
}

fn table<T>(items: &[T], f: impl Fn(&T) -> String) -> String {
    let mut parts: Vec<String> = items.iter().take(LIST_CAP).map(f).collect();
    if items.len() > LIST_CAP {
        parts.push(format!("+{}", items.len() - LIST_CAP));
    }
    format!("{{{}}}", parts.join(","))
}

fn raw_list<'a>(entries: impl Iterator<Item = &'a RawBind>) -> String {
    let all: Vec<&RawBind> = entries.collect();
    let mut parts: Vec<String> = all
        .iter()
        .take(LIST_CAP)
        .map(|e| {
            format!(
                "{}.{}{}={:#x}",
                if e.stage == ShaderStage::Fragment { 'f' } else { 'v' },
                e.kind.letter(),
                e.index,
                e.object_ref
            )
        })
        .collect();
    if all.len() > LIST_CAP {
        parts.push(format!("+{}", all.len() - LIST_CAP));
    }
    format!("[{}]", parts.join(" "))
}

/// Write the bounded `frag_unbound_trace` line for one draw that is about to
/// substitute `unbound` (non-empty). Cold path: reached only by draws the
/// guard already flagged, and then only when the limiter admits them.
pub(crate) fn trace_draw(
    req: &super::DrawEncodeRequest,
    reflection: &Arc<metal2vulkan::reflect::ShaderReflection>,
    spirv_len: usize,
    unbound: &[u32],
    w: u32,
    h: u32,
) {
    if unbound.is_empty() || !enabled() {
        return;
    }
    let class = pass_class(w, h);
    let Some((sample, seen)) = limiter()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .admit(req.pipeline_ref, class)
    else {
        return;
    };
    use metal2vulkan::reflect::ResourceKind;
    let line = with_context(|ctx| {
        let declared: Vec<String> = reflection
            .bindings
            .iter()
            .filter(|b| {
                !matches!(
                    b.kind,
                    ResourceKind::ThreadgroupBuffer | ResourceKind::KernelStageInput
                )
            })
            .take(LIST_CAP)
            .map(|b| {
                let ty = b.type_name.as_deref().unwrap_or("");
                let state = match b.kind {
                    ResourceKind::Texture | ResourceKind::TextureArray => {
                        format!("={}", classify(ctx, &req.fragment_textures, b.metal_index))
                    }
                    ResourceKind::Sampler => {
                        if req
                            .fragment_samplers
                            .iter()
                            .any(|s| s.index == b.metal_index && s.sampler_ref != 0)
                        {
                            "=bound".to_string()
                        } else {
                            "=unbound(default_sampler)".to_string()
                        }
                    }
                    ResourceKind::Buffer => {
                        if req
                            .fragment_buffers
                            .iter()
                            .any(|x| x.index == b.metal_index && x.buffer_ref != 0)
                        {
                            "=bound".to_string()
                        } else {
                            "=unbound".to_string()
                        }
                    }
                    ResourceKind::StaticSampler => "=constexpr".to_string(),
                    _ => String::new(),
                };
                format!("{:?}{}({ty}){state}", b.kind, b.metal_index)
            })
            .collect();
        let (origin, since, stream_frag_tex, dropped, prior, seq) = match ctx {
            None => (
                "none".to_string(),
                "[]".to_string(),
                "[]".to_string(),
                0,
                "{}".to_string(),
                0,
            ),
            Some(c) => {
                let origin = match c.origin {
                    DrawOrigin::Direct(t) => format!("direct draw_tag={t}"),
                    DrawOrigin::Icb => "icb".to_string(),
                };
                let (since, frag, dropped) = match c.log.as_ref() {
                    None => ("[]".to_string(), "[]".to_string(), 0),
                    Some(log) => {
                        let since = match c.origin {
                            DrawOrigin::Direct(t) => {
                                raw_list(log.entries.iter().filter(|e| e.draw_tag == t))
                            }
                            DrawOrigin::Icb => raw_list(log.entries.iter()),
                        };
                        let frag = raw_list(log.entries.iter().filter(|e| {
                            e.kind == BindKind::Texture && e.stage == ShaderStage::Fragment
                        }));
                        (since, frag, log.dropped)
                    }
                };
                let prior = table(&c.prior_frag_textures, |t| {
                    format!("{}:{:#x}", t.index, t.texture_ref)
                });
                (origin, since, frag, dropped, prior, c.stream_seq)
            }
        };
        let fcs: Vec<String> = reflection
            .function_constants
            .iter()
            .take(LIST_CAP)
            .map(|fc| format!("{}:{}", fc.index, fc.name))
            .collect();
        let is_null = match is_null_query(reflection, spirv_len) {
            Some(true) => "yes",
            Some(false) => "no",
            None => "unknown",
        };
        let unbound_fates: Vec<String> = unbound
            .iter()
            .map(|i| format!("tex{i}:{}", classify(ctx, &req.fragment_textures, *i)))
            .collect();
        format!(
            "frag_unbound_trace class={class} pipe={} task={} pass={w}x{h} sample={sample} \
             seen={seen} origin={origin} stream_seq={seq} entry={} fcs=[{}] \
             argbuf_fields={} is_null_query={is_null} unbound=[{}] declared=[{}] \
             raw_since_last_draw={since} raw_stream_frag_tex={stream_frag_tex} \
             log_dropped={dropped} enc_f_tex={} enc_v_tex={} enc_f_smp={} enc_f_buf={} \
             enc_v_buf={} prior_encoder_f_tex={prior} viewports={} scissors={} \
             instances={} verts={}",
            req.pipeline_ref,
            req.task_id,
            reflection.entry_point.as_deref().unwrap_or("?"),
            fcs.join(","),
            reflection.argument_buffer_fields.len(),
            unbound_fates.join(","),
            declared.join(" "),
            table(&req.fragment_textures, |t| format!(
                "{}:{:#x}{}",
                t.index,
                t.texture_ref,
                if t.resource.is_some() { "" } else { "(unresolved)" }
            )),
            table(&req.vertex_textures, |t| format!("{}:{:#x}", t.index, t.texture_ref)),
            table(&req.fragment_samplers, |s| format!("{}:{:#x}", s.index, s.sampler_ref)),
            table(&req.fragment_buffers, |b| format!("{}:{:#x}", b.index, b.buffer_ref)),
            table(&req.vertex_buffers, |b| format!("{}:{:#x}", b.index, b.buffer_ref)),
            table(&req.viewports, |v| format!(
                "{}+{}:{}x{}",
                v[0], v[1], v[2], v[3]
            )),
            table(&req.scissors, |s| format!("{s:?}")),
            req.instance_count,
            req.vertex_count,
        )
    });
    crate::observe::fail(line);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex(index: u32, texture_ref: u32) -> TextureBind {
        TextureBind {
            index,
            texture_ref,
            resource: None,
        }
    }

    fn ctx(log: StreamBindLog, origin: DrawOrigin, prior: Vec<TextureBind>) -> DrawBindContext {
        DrawBindContext {
            log: Some(Arc::new(log)),
            origin,
            prior_frag_textures: Arc::new(prior),
            stream_seq: 0,
        }
    }

    fn push(log: &mut StreamBindLog, tag: u32, kind: BindKind, stage: ShaderStage, idx: u32, r: u32) {
        log.entries.push(RawBind {
            draw_tag: tag,
            kind,
            stage,
            index: idx,
            object_ref: r,
        });
    }

    /// Each way a slot can be empty gets its own fate, and a slot the table
    /// holds is never reported as anything but bound.
    #[test]
    fn every_way_a_slot_is_empty_is_named_apart() {
        let f = ShaderStage::Fragment;
        let v = ShaderStage::Vertex;
        let mut log = StreamBindLog::default();
        push(&mut log, 0, BindKind::Texture, f, 0, 0x10);
        push(&mut log, 0, BindKind::Texture, f, 2, 0x20);
        push(&mut log, 1, BindKind::Texture, f, 2, 0);
        push(&mut log, 0, BindKind::Texture, v, 3, 0x30);
        push(&mut log, 0, BindKind::Texture, f, 5, 0x50);
        let c = ctx(log, DrawOrigin::Direct(1), vec![tex(4, 0x40)]);
        let table = vec![tex(0, 0x10)];

        assert_eq!(classify(Some(&c), &table, 0), SlotFate::Bound);
        assert_eq!(classify(Some(&c), &table, 1), SlotFate::NeverBound);
        assert_eq!(classify(Some(&c), &table, 2), SlotFate::ClearedByNil);
        assert_eq!(classify(Some(&c), &table, 3), SlotFate::VertexStageOnly);
        assert_eq!(classify(Some(&c), &table, 4), SlotFate::PriorEncoderHad(0x40));
        assert_eq!(classify(Some(&c), &table, 5), SlotFate::BoundRefLost(0x50));
        assert_eq!(classify(None, &table, 1), SlotFate::Untraced);
    }

    /// A bind that arrives after the draw is not evidence about that draw.
    #[test]
    fn a_later_bind_does_not_reach_back_into_an_earlier_draw() {
        let mut log = StreamBindLog::default();
        push(&mut log, 2, BindKind::Texture, ShaderStage::Fragment, 1, 0);
        let early = ctx(log.clone(), DrawOrigin::Direct(1), vec![]);
        let late = ctx(log, DrawOrigin::Direct(2), vec![]);
        assert_eq!(classify(Some(&early), &[], 1), SlotFate::NeverBound);
        assert_eq!(classify(Some(&late), &[], 1), SlotFate::ClearedByNil);
    }

    /// The log is bounded and says so rather than claiming "never bound".
    #[test]
    fn an_overflowed_log_names_itself() {
        let mut log = StreamBindLog::default();
        log.record(
            0,
            BindKind::Buffer,
            ShaderStage::Fragment,
            0,
            std::iter::repeat(7).take(STREAM_LOG_CAP + 3),
        );
        assert_eq!(log.entries().len(), STREAM_LOG_CAP);
        assert_eq!(log.dropped, 3);
        let c = ctx(log, DrawOrigin::Direct(0), vec![]);
        assert_eq!(classify(Some(&c), &[], 1), SlotFate::LogOverflow);
    }

    /// A range record lands on consecutive slots from `first`.
    #[test]
    fn a_range_record_is_logged_slot_by_slot() {
        let mut log = StreamBindLog::default();
        log.record(3, BindKind::Texture, ShaderStage::Fragment, 1, [0xa, 0, 0xc].into_iter());
        let idx: Vec<(u32, u32)> = log.entries().iter().map(|e| (e.index, e.object_ref)).collect();
        assert_eq!(idx, vec![(1, 0xa), (2, 0), (3, 0xc)]);
        assert!(log.entries().iter().all(|e| e.draw_tag == 3));
    }

    /// First lines per key, then one in `SAMPLE_EVERY`, and keys are
    /// separate per pass class so large passes are not starved.
    #[test]
    fn the_limiter_is_bounded_per_key_and_per_class() {
        let mut l = Limiter::default();
        let admitted: u32 = (0..(FIRST_PER_KEY + 2 * SAMPLE_EVERY))
            .filter(|_| l.admit(37, "icon").is_some())
            .count() as u32;
        assert_eq!(admitted, FIRST_PER_KEY + 2);
        assert_eq!(l.admit(37, "material"), Some(("first", 1)));
        let mut capped = Limiter {
            lines: MAX_LINES,
            ..Default::default()
        };
        assert_eq!(capped.admit(1, "icon"), None);
    }

    #[test]
    fn every_fate_has_its_own_route() {
        let all = [
            SlotFate::Bound,
            SlotFate::NeverBound,
            SlotFate::ClearedByNil,
            SlotFate::VertexStageOnly,
            SlotFate::PriorEncoderHad(1),
            SlotFate::BoundRefLost(1),
            SlotFate::LogOverflow,
            SlotFate::Untraced,
        ];
        let routes: std::collections::BTreeSet<_> = all.iter().map(|f| f.route()).collect();
        assert_eq!(routes.len(), all.len());
        assert_eq!(pass_class(128, 128), "icon");
        assert_eq!(pass_class(736, 352), "material");
    }
}
