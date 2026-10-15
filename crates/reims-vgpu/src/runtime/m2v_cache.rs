//! Process-global metal2vulkan SPIR-V cache (AIR bytes → SPIR-V).
//!
//! Product Linux draws call m2v on the **doorbell MMIO vCPU** (sync drain under
//! BQL — see `runtime/mmio.rs` CONTROL_FIFO / child doorbell). Live OFF logs
//! showed the same pipelines re-translated dozens of times per boot (e.g.
//! pipe=39 fragment SPIR-V ~400 KiB × 28). That holds guest CPUs long enough
//! for `pmap_flush_tlbs` **IPI timeout** panics (WindowServer in
//! `processExecIndirect` / `submitOnChannel`).
//!
//! The cache key is the AIR blob itself plus its stage — not the pipeline object
//! id, which recycles, and not a hash of the AIR, which can collide. A content
//! hash narrows the bucket and `Slot::is` decides the hit; `ShaderId` carries
//! the argument for why the digest is not allowed to.
//! Measure-only hit/miss counters for fail-log census.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use metal2vulkan::passes::Stage;
use metal2vulkan::reflect::ShaderReflection;

use crate::observe::Decline;

type FragmentRelocationCache = HashMap<(bool, bool, bool), Option<Arc<ShaderVariant>>>;
type M2vResult<T> = Result<T, M2vCacheDecline>;

/// A specific failure while caching, translating, or post-processing AIR.
///
/// Raw tool/IO/layout text stays payload; the reason itself is stable and
/// registered so render, compute, and the async worker name the same check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum M2vCacheDecline {
    VertexScratchWrite {
        detail: String,
    },
    FragmentScratchWrite {
        detail: String,
    },
    KernelScratchWrite {
        detail: String,
    },
    VertexTranslate {
        detail: String,
    },
    FragmentTranslate {
        detail: String,
    },
    KernelTranslate {
        detail: String,
    },
    ReflectionDatalayoutMissing {
        stage: &'static str,
    },
    ReflectionMalformed {
        stage: &'static str,
        violations: usize,
    },
    LayoutRepair {
        stage: &'static str,
        reason: crate::runtime::spirv_layout::SpirvLayoutDecline,
    },
    TranslationPending {
        stage: &'static str,
    },
    KernelLocalSizeZero {
        local_size: [u32; 3],
    },
    KernelLocalSizeMismatch {
        requested: [u32; 3],
        reflected: Option<[u32; 3]>,
    },
}

impl M2vCacheDecline {
    /// Whether a second attempt at the same AIR could answer differently.
    ///
    /// Every entry in this cache is keyed by the AIR blob's content, so a
    /// refusal that is *about the AIR* is reached again by every later ask and
    /// is worth remembering: a module whose datalayout is missing, whose layout
    /// repair fails, or whose requested threadgroup is degenerate refuses the
    /// same way forever.
    ///
    /// The scratch writes are not about the AIR. `translate_air` and
    /// `translate_kernel_air` each begin by writing the blob to a fixed path
    /// under [`tmp_dir`], and that write fails for reasons belonging to the host
    /// filesystem at that instant — no space, no descriptors, a transient I/O
    /// error. Remembering one turns "the host could not spare a scratch file
    /// just then" into "this shader never renders again", and because the cache
    /// is unbounded and nothing evicts, "again" means for the life of the
    /// process. It is the same rule
    /// [`crate::backend::vulkan::engine::types::DrawError::out_of_memory`]
    /// states for the object caches.
    ///
    /// The translate declines are deliberately *not* here even though
    /// metal2vulkan also touches the filesystem. Their detail is an opaque
    /// string from the tool, so telling a disk failure from a malformed module
    /// would mean matching on its prose — and re-running a full translation on
    /// every draw of a genuinely untranslatable shader costs far more than the
    /// scratch write does. The split is by what the variant *names*, not by what
    /// its message happens to say.
    fn is_transient(&self) -> bool {
        match self {
            Self::VertexScratchWrite { .. }
            | Self::FragmentScratchWrite { .. }
            | Self::KernelScratchWrite { .. } => true,
            Self::VertexTranslate { .. }
            | Self::FragmentTranslate { .. }
            | Self::KernelTranslate { .. }
            | Self::ReflectionDatalayoutMissing { .. }
            | Self::ReflectionMalformed { .. }
            | Self::LayoutRepair { .. }
            | Self::TranslationPending { .. }
            | Self::KernelLocalSizeZero { .. }
            | Self::KernelLocalSizeMismatch { .. } => false,
        }
    }
}

/// Hand a stored failure to its caller, and drop it if a later ask could get a
/// different answer.
///
/// This is where the retry is armed, and it is the only point in the cache's
/// life cycle where arming one is safe. The async admission
/// ([`ensure_cached_async_keyed`]) cannot do it: its caller re-polls the same
/// guest packet until it answers `true`, so an arm that removed the entry and
/// re-queued would translate, fail, remove, re-queue — and the packet at the
/// channel head would never advance. Here the error is already on its way to
/// the caller, so this draw fails whatever we do; all that changes is that the
/// *next* draw finds no entry and translates again.
///
/// The cost while the host stays unable is one re-translation per draw of that
/// shader. The alarm is fail-visible and `fail_once`-deduped by key, so a
/// persistent one names itself once rather than per attempt.
fn forget_if_transient(cache: &mut Cache, id: ShaderId<'_>, error: &M2vCacheDecline) {
    if !error.is_transient() {
        return;
    }
    cache.forget(id);
    crate::observe::Emit::decline("m2v_transient_failure_forgotten", error)
        .field("key", id.digest)
        .fail_once(id.digest);
}

fn log_token(detail: &str) -> String {
    detail.replace(char::is_whitespace, "_")
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Vertex => "vertex",
        Stage::Fragment => "fragment",
        Stage::Kernel => "kernel",
    }
}

fn translate_decline(stage: Stage, detail: String) -> M2vCacheDecline {
    match stage {
        Stage::Vertex => M2vCacheDecline::VertexTranslate { detail },
        Stage::Fragment => M2vCacheDecline::FragmentTranslate { detail },
        Stage::Kernel => M2vCacheDecline::KernelTranslate { detail },
    }
}

fn scratch_write_decline(stage: Stage, detail: String) -> M2vCacheDecline {
    match stage {
        Stage::Vertex => M2vCacheDecline::VertexScratchWrite { detail },
        Stage::Fragment => M2vCacheDecline::FragmentScratchWrite { detail },
        Stage::Kernel => M2vCacheDecline::KernelScratchWrite { detail },
    }
}

impl Decline for M2vCacheDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::VertexScratchWrite { .. } => "m2v_vertex_scratch_write",
            Self::FragmentScratchWrite { .. } => "m2v_fragment_scratch_write",
            Self::KernelScratchWrite { .. } => "m2v_kernel_scratch_write",
            Self::VertexTranslate { .. } => "m2v_vertex_translate",
            Self::FragmentTranslate { .. } => "m2v_fragment_translate",
            Self::KernelTranslate { .. } => "m2v_kernel_translate",
            Self::ReflectionDatalayoutMissing { .. } => "m2v_reflection_datalayout_missing",
            Self::ReflectionMalformed { .. } => "m2v_reflection_malformed",
            Self::LayoutRepair { reason, .. } => reason.slug(),
            Self::TranslationPending { .. } => "m2v_translation_pending_at_sync_boundary",
            Self::KernelLocalSizeZero { .. } => "m2v_kernel_local_size_zero",
            Self::KernelLocalSizeMismatch { .. } => "m2v_kernel_local_size_mismatch",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::VertexScratchWrite { detail } => vec![
                ("stage", "vertex".to_string()),
                ("file", "v.air".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::FragmentScratchWrite { detail } => vec![
                ("stage", "fragment".to_string()),
                ("file", "f.air".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::KernelScratchWrite { detail } => vec![
                ("stage", "kernel".to_string()),
                ("file", "k.air".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::VertexTranslate { detail } => vec![
                ("stage", "vertex".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::FragmentTranslate { detail } => vec![
                ("stage", "fragment".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::KernelTranslate { detail } => vec![
                ("stage", "kernel".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::LayoutRepair { stage, reason } => {
                let mut fields = vec![("stage", (*stage).to_string())];
                fields.extend(reason.fields());
                fields
            }
            Self::ReflectionDatalayoutMissing { stage } | Self::TranslationPending { stage } => {
                vec![("stage", (*stage).to_string())]
            }
            Self::ReflectionMalformed { stage, violations } => vec![
                ("stage", (*stage).to_string()),
                ("violations", violations.to_string()),
            ],
            Self::KernelLocalSizeZero { local_size } => vec![
                ("tg_x", local_size[0].to_string()),
                ("tg_y", local_size[1].to_string()),
                ("tg_z", local_size[2].to_string()),
            ],
            Self::KernelLocalSizeMismatch {
                requested,
                reflected,
            } => vec![
                (
                    "requested",
                    format!("{},{},{}", requested[0], requested[1], requested[2]),
                ),
                (
                    "reflected",
                    reflected.map_or_else(
                        || "none".to_string(),
                        |size| format!("{},{},{}", size[0], size[1], size[2]),
                    ),
                ),
            ],
        }
    }
}

crate::observe::decline_display!(M2vCacheDecline);

impl std::error::Error for M2vCacheDecline {}

/// A translated shader: the SPIR-V bytes we hand to the Vulkan engine, plus the
/// `metal2vulkan` reflection facade derived from the SAME parsed AIR metadata
/// (descriptor bindings, texture shapes, vertex builtins, datalayout, …). The
/// reflection is the single source of truth for stage-interface facts so the
/// consumer never re-parses AIR or re-walks the emitted SPIR-V. `spirv` is the
/// post-`repair_layout` module (byte-identical to what the plain cache returned).
pub struct CachedShader {
    pub spirv: Vec<u8>,
    pub reflection: Arc<ShaderReflection>,
    /// The same module as u32 words, materialized once — draw paths clone the
    /// `Arc`, never re-collect per draw (was a full-module copy ×2 per draw).
    pub words: Arc<Vec<u32>>,
    /// The module in the translator's own binding numbering, and what a walk of
    /// it answers. Separate from `words` only because `words` predates
    /// [`ShaderVariant`] and half the crate reads it.
    base: OnceLock<Arc<ShaderVariant>>,
    /// Fragment binding-relocation variants keyed by
    /// `(separate_sampled, buf_collide)` — the relocation is a pure function of
    /// the module and those two flags, so each variant is computed once per
    /// shader lifetime instead of mutating a fresh copy per draw.
    frag_reloc: Mutex<FragmentRelocationCache>,
}

/// One binding numbering of a translated module, beside the reflected
/// interface transformed into *that* numbering.
///
/// The pair is one struct because the second is only true of the first.
/// Relocation renumbers a module's descriptor bindings. The reflected sampler
/// interface is transformed by the same variant key and stored here, so a
/// caller cannot pair words from one numbering with resources from another.
pub struct ShaderVariant {
    /// The module, in this variant's numbering.
    pub words: Arc<Vec<u32>>,
    /// The typed sampler descriptors reflection declares, transformed into
    /// this variant's numbering. Constexpr state stays attached to its binding.
    ///
    /// Derived here rather than at the draw because the answer depends only on
    /// the shader and its relocation key. Reflection is the authority; the
    /// former implementation rediscovered the same interface by walking every
    /// SPIR-V instruction, twice per draw before this value was cached.
    ///
    /// # What it removed
    ///
    /// Twelve interleaved driven macos-13 sustained-animation boots, two pinned
    /// binaries differing only by this and by
    /// [`crate::backend::vulkan::pipeline_resolve::VertexBindPlan`], scored over the
    /// fast population. Per draw, median over busy census windows, then mean and
    /// range across boots:
    ///
    /// ```text
    ///                before (n=4)              after (n=6)
    /// reflect_us     0.340 [0.332..0.345]      0.021 [0.020..0.021]
    /// sampled_us     1.911 [1.861..1.972]      1.640 [1.574..1.691]
    /// engine_us      6.235 [6.065..6.338]      6.208 [6.033..6.283]
    /// ```
    ///
    /// `reflect_us` is **−94 %** and `sampled_us` **−14 %**, both fully
    /// disjoint. `engine_us` — a phase neither change touches — does not move,
    /// which is the control that says the two arms are comparable at all.
    ///
    /// `draw_us/draw` overall reads 13.33 [13.05..13.61] against 13.03
    /// [12.82..13.44]: −2.3 %, ranges overlapping. `present_hz` is 114.25
    /// against 114.16 — no movement, which is exactly what a 2 % per-draw change
    /// is supposed to look like here. **This bought a phase, not a frame**, and
    /// the elasticity beside `crate::runtime::drain::census::VBL_REPORT_EARLY`
    /// says it could not have bought a frame at that size.
    ///
    /// A first attempt at this reading ran the two pins in sequence rather than
    /// interleaved, and reported `engine_us` **+18 %** with `gpu occupancy` up
    /// and `present_hz` down — the GPU clocking down between the two groups,
    /// with nothing in the log to say so. Interleaving is not optional for a pin
    /// comparison on this host.
    pub samplers: Arc<[crate::runtime::spirv_bind::ReflectedSamplerDescriptor]>,
    /// Uniform-constant descriptor bindings the executable module statically
    /// uses, in this variant's binding numbering.
    ///
    /// Vulkan requires each of these to appear in the pipeline layout. The
    /// engine checks that relation for every draw because a missing binding can
    /// crash a host driver during pipeline creation, but the left-hand set is
    /// immutable shader state: deriving it from the full module at every draw
    /// reconstructed state already retained here. Keeping it beside `words`
    /// also makes relocation agreement structural rather than a call-site
    /// convention.
    pub used_descriptor_bindings: Arc<[u32]>,
}

impl ShaderVariant {
    fn of(
        words: Arc<Vec<u32>>,
        samplers: Arc<[crate::runtime::spirv_bind::ReflectedSamplerDescriptor]>,
    ) -> Arc<Self> {
        let used_descriptor_bindings = crate::runtime::spirv_bind::declared_binding_numbers(&words)
            .into_iter()
            .filter(|binding| {
                crate::runtime::spirv_bind::descriptor_static_use(&words, *binding).is_violation()
            })
            .collect::<Vec<_>>()
            .into();
        Arc::new(Self {
            words,
            samplers,
            used_descriptor_bindings,
        })
    }
}

impl CachedShader {
    /// Materialize a freshly translated module, in the device's binding
    /// numbering rather than the translator's.
    ///
    /// [`crate::runtime::spirv_bind::widen_sampled_bands`] runs exactly here,
    /// once per shader on the translate miss path, because this is the one point
    /// every consumer of a module goes through — both `spirv` and `words`, and
    /// therefore every fragment relocation variant derived from `words`. Doing it
    /// per draw would be a module rewrite on the hot path; doing it in only one
    /// of the two representations would let the compute rail (which reads
    /// `spirv`) and the render rail (which reads `words`) disagree about what
    /// binding a texture has.
    ///
    /// `spirv` is rebuilt from the widened words for that reason: it is no longer
    /// byte-identical to what the translator returned, and the bytes and the
    /// words must not be allowed to drift apart.
    pub fn new(spirv: Vec<u8>, reflection: Arc<ShaderReflection>) -> Self {
        let mut words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let widened = crate::runtime::spirv_bind::widen_sampled_bands(&mut words);
        let spirv = if widened == 0 {
            // Nothing moved, so the translator's bytes already are the device's.
            spirv
        } else {
            words.iter().flat_map(|w| w.to_le_bytes()).collect()
        };
        Self {
            spirv,
            reflection,
            words: Arc::new(words),
            base: OnceLock::new(),
            frag_reloc: Mutex::new(HashMap::new()),
        }
    }

    /// This module in the binding numbering the two stage-collision flags name,
    /// with the walks of that numbering beside it — see [`ShaderVariant`].
    ///
    /// `separate_sampled` relocates sampled-resource bindings first (archive
    /// order), then `buf_collide` relocates the buffer band — matching the
    /// historical per-draw mutation order exactly. Cached per variant, and
    /// `(false, false)` is a variant like any other rather than a short-circuit,
    /// because it is the one a vertex shader always takes and its walk is worth
    /// caching for exactly the same reason.
    pub fn variant(&self, separate_sampled: bool, buf_collide: bool) -> Arc<ShaderVariant> {
        self.variant_ms(separate_sampled, buf_collide, false)
            .expect("a variant without the multisample demotion always exists")
    }

    /// [`Self::variant`], and with `ms_demote` every multisampled image made single-sample
    /// ([`crate::runtime::spirv_bind::demote_multisampled_images`]) for a draw whose multisampled binds hold
    /// single-sample content. `None` only for a demotion the module cannot take; cached like the others.
    pub fn variant_ms(
        &self,
        separate_sampled: bool,
        buf_collide: bool,
        ms_demote: bool,
    ) -> Option<Arc<ShaderVariant>> {
        if !separate_sampled && !buf_collide && !ms_demote {
            return Some(
                self.base
                .get_or_init(|| {
                    ShaderVariant::of(
                        self.words.clone(),
                        crate::runtime::spirv_bind::reflected_sampler_descriptors(
                            &self.reflection,
                            false,
                        )
                        .into(),
                    )
                })
                .clone());
        }
        let mut cache = self.frag_reloc.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .entry((separate_sampled, buf_collide, ms_demote))
            .or_insert_with(|| {
                let mut w: Vec<u32> = (*self.words).clone();
                if ms_demote {
                    w = crate::runtime::spirv_bind::demote_multisampled_images(&w)?;
                    crate::observe::verbose(|| "linux_m2v frag_ms_demoted".to_string());
                }
                if separate_sampled {
                    let n = crate::runtime::spirv_bind::offset_fragment_sampled_resource_bindings(
                        &mut w,
                    );
                    crate::observe::verbose(|| format!("linux_m2v frag_sampled_reloc n={n}"));
                }
                if buf_collide {
                    let n = crate::runtime::spirv_bind::offset_fragment_buffer_bindings(&mut w);
                    crate::observe::verbose(|| format!("linux_m2v frag_buf_reloc n={n}"));
                }
                Some(ShaderVariant::of(
                    Arc::new(w),
                    crate::runtime::spirv_bind::reflected_sampler_descriptors(
                        &self.reflection,
                        separate_sampled,
                    )
                    .into(),
                ))
            })
            .clone()
    }
}

/// Translated shaders, keyed by the content hash of the AIR the guest supplied.
///
/// **Deliberately unbounded.** The key is content, so the live entry count is the
/// number of *distinct* shaders the guest has ever compiled — a property of the
/// guest's own program set, not of how long the device has run. A driven x86
/// boot with a window-drag probe against Safari settles at 75 and stays there
/// for the rest of the run, read off the `object_cache_levels` census; an idle
/// boot reaches the same. That is the same bound a real driver's pipeline cache
/// has, and it is why no reclaim rule is needed here.
///
/// It used to hold 256 entries and evict in insertion order. Both halves were
/// wrong for this workload. Insertion order makes the *first* shader compiled —
/// the compositor's, drawn every frame for the life of the boot — the first
/// victim, so crossing the cap evicts the hot set and nothing else. And a miss is
/// not free: the module header above records why re-translating on the doorbell
/// vCPU holds guest CPUs long enough to trip `pmap_flush_tlbs` IPI-timeout
/// panics. A cap whose crossing degrades the guest is a failure mode bought for
/// a bound the workload never approached.
#[derive(Default)]
struct Cache {
    /// `hash(stage_tag || air_bytes)` → every shader filed under that digest.
    ///
    /// A bucket holds more than one entry only on a digest collision, and
    /// [`Slot::is`] is what decides a hit — see [`ShaderId`] for why the digest
    /// is not allowed to.
    entries: HashMap<u64, Vec<Slot>>,
    hits: u64,
    misses: u64,
    async_queue: VecDeque<TranslationTask>,
    async_worker_running: bool,
    /// The last AIR each `(task, pipeline_ref, stage_tag)` was translated
    /// under.
    ///
    /// **Not a cache — a census.** Pipeline references are per task: two tasks
    /// routinely name the same integer for unrelated pipelines, so a key
    /// without the task reads one task's pipeline as the other's "change" (57
    /// of the 63 refs a driven macos-15 boot reported were named in two or
    /// more tasks at the same moment). The task is part of the key for that
    /// reason, and [`forget_task_identities`] ends a task's entries with the
    /// task, because the guest reuses task ids.
    ///
    /// What is left after that is one task naming different AIR under a ref it
    /// used before: a delete followed by a reuse of the number, which is
    /// legitimate guest behaviour. The only caller asks after
    /// `pipeline_resolve::translations_ready` answered "not retained", so the
    /// retained pipeline registry holds nothing for that ref when this fires
    /// and no stale shader is served. The line is a census on the off
    /// channel, not a failure, and nothing reads this map to decide anything.
    ///
    /// The value is the digest of the AIR alone: a kernel's cache digest also
    /// carries its dispatch geometry, and one kernel dispatched at two
    /// threadgroup sizes is two translations of one AIR, not a change.
    last_identity: HashMap<(u32, u32, u8), u64>,
}

#[derive(Clone)]
enum Entry {
    Loading,
    Ready(Arc<CachedShader>),
    Failed(M2vCacheDecline),
}

/// One cached translation and the identity it was filed under.
///
/// The AIR is retained. That is the whole cost of the identity compare, and it
/// is a good deal smaller than [`Cache`]'s own doc once implied: a
/// [`CachedShader`] already holds the module as `spirv` bytes *and* as `words`,
/// plus a relocated word copy per fragment variant, so one copy of the AIR that
/// produced them is a fraction of the entry rather than a doubling of it.
struct Slot {
    stage: u8,
    /// `Some` for a kernel, whose LocalSize is baked into the SPIR-V, so one
    /// AIR blob dispatched at two geometries is two shaders. `None` for a render
    /// stage, which has no such parameter.
    local_size: Option<[u32; 3]>,
    air: Arc<[u8]>,
    entry: Entry,
}

impl Slot {
    /// The full identity compare. This alone decides a hit.
    fn is(&self, id: ShaderId<'_>) -> bool {
        self.stage == id.stage && self.local_size == id.local_size && *self.air == *id.air
    }
}

/// What makes two translations the same translation: the whole of what the
/// guest supplied, borrowed for a lookup.
///
/// # The digest narrows the bucket and decides nothing
///
/// [`Cache::entries`] used to be keyed on `digest` alone and hold no copy of the
/// AIR, so a lookup that landed on an entry returned it — there was no
/// confirmation step. Two distinct AIR blobs colliding would hand the guest a
/// shader it never compiled, silently and with no failure line, which is a worse
/// outcome than any eviction this device can make.
///
/// That was argued as acceptable against the birthday bound — `n² / 2^65` for
/// the 75 distinct shaders a driven boot settles at, about `2e-16`. The argument
/// is arithmetically right and it is the wrong shape: it prices a failure mode
/// instead of removing one, and the price it quotes is not one this device is
/// able to observe if it is ever wrong. Retaining the AIR removes the class
/// outright, which is what the old doc itself named as the answer, and it is
/// cheaper than a wider hash because a wider hash only moves the exponent.
///
/// It is also the shape the rest of this crate uses:
/// [`crate::model::content_cache`] buckets by a `u64` prefilter and decides on
/// `CacheEntry::matches`, and [`crate::backend::blob`] buckets a shader by its
/// digest and decides on the retained bytes.
///
/// This doc used to close by calling itself "the one digest-keyed cache in the
/// crate that trusted its key", on the strength of a sweep run when it was
/// fixed. **The sweep was wrong by three, and how it went wrong is the useful
/// part.** `backend::metal::cache`'s `BlobKey` was a digest beside the blob's
/// *length*, and its own doc argued that carrying the length made it an
/// identity — so a reader auditing for "digest alone" read the length as the
/// confirming compare and moved on. It is not one. It makes a collision need
/// equal lengths, which narrows the population by a factor and removes nothing.
/// Three caches keyed on it. When auditing this class, the question is not
/// whether the key has a second field; it is whether **anything retained the
/// bytes**.
///
/// Borrowed rather than owned because a lookup happens per pipeline build and an
/// owned key would allocate a copy of the AIR to throw away on every hit. Only
/// [`Cache::put`] takes ownership, once per distinct shader.
#[derive(Clone, Copy)]
struct ShaderId<'a> {
    digest: u64,
    stage: u8,
    local_size: Option<[u32; 3]>,
    air: &'a [u8],
}

impl<'a> ShaderId<'a> {
    fn render(stage: Stage, air: &'a [u8]) -> Self {
        Self {
            digest: air_key(stage, air),
            stage: stage_tag(stage),
            local_size: None,
            air,
        }
    }

    fn kernel(air: &'a [u8], local_size: [u32; 3]) -> Self {
        Self {
            digest: air_key_kernel(air, local_size),
            stage: stage_tag(Stage::Kernel),
            local_size: Some(local_size),
            air,
        }
    }

    /// The digest of the AIR this identity was built from, without the dispatch
    /// geometry a kernel's cache digest also carries.
    ///
    /// A render stage's cache digest already is that. A kernel pays one more
    /// hash of its AIR for it, on a cache miss only.
    fn air_digest(&self) -> u64 {
        match self.local_size {
            None => self.digest,
            Some(_) => air_key(Stage::Kernel, self.air),
        }
    }
}

impl Cache {
    /// The entry filed under `id`, if this cache holds one.
    fn find(&self, id: ShaderId<'_>) -> Option<&Entry> {
        self.entries
            .get(&id.digest)?
            .iter()
            .find(|s| s.is(id))
            .map(|s| &s.entry)
    }

    /// File `entry` under `id`, replacing whatever that identity held.
    ///
    /// Replacing rather than pushing is the `Loading` -> `Ready`/`Failed`
    /// transition, which is the common case: the admission puts `Loading` and
    /// the worker puts the result under the same identity. Pushing there would
    /// leave the `Loading` slot in front of the `Ready` one, where `find`
    /// reaches it first and every later ask reports the translation still
    /// pending.
    fn put(&mut self, id: ShaderId<'_>, air: &Arc<[u8]>, entry: Entry) {
        let bucket = self.entries.entry(id.digest).or_default();
        match bucket.iter_mut().find(|s| s.is(id)) {
            Some(slot) => slot.entry = entry,
            None => bucket.push(Slot {
                stage: id.stage,
                local_size: id.local_size,
                air: Arc::clone(air),
                entry,
            }),
        }
    }

    /// Record the AIR `(task_id, pipeline_ref, id.stage)` is now translated
    /// under, and return `(previous, now)` when that differs from the last one
    /// recorded. See [`Cache::last_identity`] for what the answer means.
    fn note_identity(
        &mut self,
        task_id: u32,
        pipeline_ref: u32,
        id: ShaderId<'_>,
    ) -> Option<(u64, u64)> {
        let now = id.air_digest();
        let previous = self
            .last_identity
            .insert((task_id, pipeline_ref, id.stage), now)?;
        (previous != now).then_some((previous, now))
    }

    /// End [`Self::last_identity`]'s records for one task.
    fn forget_task_identities(&mut self, task_id: u32) {
        self.last_identity
            .retain(|&(task, _, _), _| task != task_id);
    }

    /// Drop the entry filed under `id`, and the bucket with it if it was the
    /// last — so a forgotten transient failure leaves nothing behind to walk.
    fn forget(&mut self, id: ShaderId<'_>) {
        let std::collections::hash_map::Entry::Occupied(mut bucket) = self.entries.entry(id.digest)
        else {
            return;
        };
        bucket.get_mut().retain(|s| !s.is(id));
        if bucket.get().is_empty() {
            bucket.remove();
        }
    }

    /// Live entries across every bucket.
    ///
    /// The level `object_cache_levels` publishes. It sums rather than counting
    /// buckets because those differ exactly when a digest collides, which is the
    /// event this cache now survives and must not under-report.
    fn len(&self) -> usize {
        self.entries.values().map(Vec::len).sum()
    }
}

struct TranslationTask {
    stage: Stage,
    kernel_local_size: Option<[u32; 3]>,
    /// Shared with the [`Slot`] the admission filed, so the identity the worker
    /// completes under is the same bytes the admission was asked about and not a
    /// second copy that could differ.
    air: Arc<[u8]>,
    pipeline_ref: u32,
}

impl TranslationTask {
    fn id(&self) -> ShaderId<'_> {
        match self.kernel_local_size {
            Some(local_size) => ShaderId::kernel(&self.air, local_size),
            None => ShaderId::render(self.stage, &self.air),
        }
    }
}

fn global() -> &'static Mutex<Cache> {
    static C: std::sync::OnceLock<Mutex<Cache>> = std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(Cache::default()))
}

fn stage_tag(stage: Stage) -> u8 {
    match stage {
        Stage::Vertex => 1,
        Stage::Fragment => 2,
        Stage::Kernel => 3,
    }
}

/// The bucket digest for a non-kernel shader: 64 bits of SipHash over stage +
/// AIR.
///
/// # This narrows the search and decides nothing
///
/// It is not the cache key. [`ShaderId`] is, and [`Slot::is`] compares the AIR
/// itself on every hit, so a collision here costs a two-entry walk and cannot
/// cost correctness. Read [`ShaderId`] before changing this function: what it
/// must be is *stable within one process*, and what it must not be is *trusted*.
///
/// This doc used to argue the opposite — that the digest was the identity, and
/// that the birthday bound (`n² / 2^65`, about `2e-16` at the 75 distinct
/// shaders a driven boot settles at) made the resulting wrong-shader hazard
/// acceptable. The arithmetic was right. Pricing a silent failure this device
/// cannot observe was not, and the same doc already named retaining the AIR as
/// the answer.
///
/// `air.hash()` writes a length prefix before the bytes, per `Hash for [u8]`, so
/// two blobs of different lengths are not merely unlikely to share a bucket —
/// the length is in the digest too. That is now a statement about walk length
/// rather than about correctness.
fn air_key(stage: Stage, air: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    stage_tag(stage).hash(&mut h);
    air.hash(&mut h);
    h.finish()
}

/// The bucket digest for a kernel, which includes LocalSize because the SPIR-V
/// workgroup size is baked in at translate.
///
/// `local_size` is in the digest *and* in [`ShaderId`], and it has to be in both
/// for different reasons: in the identity because one AIR blob dispatched at two
/// geometries is genuinely two shaders, and in the digest so the two do not
/// share a bucket and pay a walk on every hit. Leaving it out of the identity
/// would be the wrong-shader hazard [`air_key`] describes, except reachable by a
/// guest doing something entirely ordinary rather than by chance.
fn air_key_kernel(air: &[u8], local_size: [u32; 3]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    stage_tag(Stage::Kernel).hash(&mut h);
    air.hash(&mut h);
    local_size.hash(&mut h);
    h.finish()
}

/// Process-local temp dir for m2v path-based translate (reused across draws).
fn tmp_dir() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let p = std::env::temp_dir().join(format!("reims-vgpu-m2v-cache-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&p);
        p
    })
    .clone()
}

fn translation_lock() -> &'static Mutex<()> {
    static LOCK: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Directory [`capture_air`] writes into.
fn air_capture_dir() -> std::path::PathBuf {
    std::path::PathBuf::from("/tmp/reims-vgpu-air")
}

/// Write one translated AIR blob beside the size of the SPIR-V it produced.
///
/// The name is the join this exists for. A driver quarantine line and a
/// validator refusal both quote the emitted module's **word count**, and
/// neither can name the AIR that produced it — so the count goes in the file
/// name and a handover is a lookup rather than another boot.
///
/// Observation only, on both the guest path and the failure path: every write
/// here is best-effort, and a failure is reported once and then ignored. This
/// runs after the translation it describes, so it cannot change what that
/// translation produced or whether it was cached.
fn capture_air(air: &[u8], stage: Stage, spirv: &[u8]) {
    if !matches!(
        crate::config::switch(crate::config::AIR_CAPTURE),
        crate::config::Switch::On
    ) {
        return;
    }
    let dir = air_capture_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        crate::observe::fail(format!(
            "air_capture reason=mkdir_failed dir={} err={e}",
            dir.display()
        ));
        return;
    }
    // The AIR digest disambiguates two shaders whose modules happen to be the
    // same length; the word count is what a quarantine or validator line
    // quotes, so it leads.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    air.hash(&mut hasher);
    let digest = hasher.finish();
    let words = spirv.len() / 4;
    let name = format!("{}-w{words}-{digest:016x}.air", stage_name(stage));
    let path = dir.join(&name);
    if path.exists() {
        return;
    }
    match std::fs::write(&path, air) {
        Ok(()) => crate::observe::fail(format!(
            "air_capture ok stage={} air_bytes={} spirv_words={words} file={name}",
            stage_name(stage),
            air.len()
        )),
        Err(e) => crate::observe::fail(format!(
            "air_capture reason=write_failed file={name} err={e}"
        )),
    }
}

/// TEMP lab: the on-disk translation cache, `REIMS_VGPU_LAB_M2V_DISK=<dir>`.
///
/// A translation is 5-50 ms and a cold 5K boot misses ~340 of them, each one a
/// stall the guest sees as a slow first draw or a slow login. The in-memory
/// [`Cache`] forgets all of it at process exit; this keeps the translator's raw
/// output between boots, one file per shader.
///
/// # What is stored, and what is not
///
/// The translator's own output — the SPIR-V bytes and the [`ShaderReflection`]
/// `metal2vulkan` returned — and nothing downstream of it. [`finish_translated`]
/// (layout repair, reflection validation, band widening) runs on a loaded entry
/// exactly as it runs on a fresh translation, so the reims-side post-processing
/// stays live code: editing it takes effect on the next boot without the stamp
/// having to know about it.
///
/// # A hit is byte-exact
///
/// The file name is the stage and the 64-bit [`air_key`], and that digest decides
/// nothing — the entry carries the whole AIR (and a kernel's LocalSize) and a
/// load compares them against what the guest supplied, for the same reason
/// [`ShaderId`] does. A collision or a stale file is a miss. A checksum over the
/// whole file turns a torn or bit-rotted entry into a miss too, because the
/// SPIR-V bytes are the one part nothing else would compare.
///
/// The digest does have to be the same in the next process: `air_key` hashes with
/// `DefaultHasher::new()`, whose keys are fixed, so it is for one build of the
/// standard library. A build with another one is another executable and so
/// another stamp, and a digest that moved anyway would only cost a miss.
///
/// # What invalidates it
///
/// The [`Stamp`] in the header. `metal2vulkan` is patched in from a sibling
/// checkout that is edited constantly, so its crate version means nothing. The
/// build script digests what produces a translation instead: the translator's
/// sources when they are that checkout, the workspace `Cargo.lock` (the pinned
/// rev and every crate it builds with), and the toolchain. With the reflection
/// ABI version and the `METAL2VULKAN_*` environment (several of those toggles
/// change what is emitted) that is the stamp. It used to be the executable's
/// size and mtime, so every relink of QEMU — a change anywhere in it, or the
/// same binary copied — made the next boot translate every shader cold.
///
/// The stamp is a header field rather than part of the file name so a rebuilt
/// translator overwrites the entries it invalidates instead of leaving a second
/// generation beside them: the directory holds at most one file per distinct
/// shader.
///
/// # Cost
///
/// A lookup happens only where a translation was about to run — an in-memory
/// miss — and runs before the translation lock, so a hit never queues behind
/// another thread's translation. The write follows a fresh translation, once per
/// shader.
///
/// A disk failure never fails a translation. It is counted, reported once
/// through [`crate::observe::fail`], and the translator's own result is used.
mod disk {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    use super::{log_token, M2vResult, ShaderId, ShaderReflection};

    /// The lab switch: a directory, or `off`. Unset uses [`default_dir`].
    pub(super) const ENV: &str = "REIMS_VGPU_LAB_M2V_DISK";

    const MAGIC: [u8; 8] = *b"RVGPUM2V";

    /// The framing below and the JSON inside it. Bump on any change to either.
    const SCHEMA: u32 = 1;

    /// Magic, schema, stamp, stage, LocalSize flag and value, three lengths.
    const HEADER_LEN: usize = 8 + 4 + STAMP_LEN + 1 + 1 + 12 + 3 * 8;
    const STAMP_LEN: usize = 4 + 8 + 8 + 8;
    const TRAILER_LEN: usize = 8;

    /// The translator's raw output: SPIR-V bytes and the reflection beside them.
    pub(super) type Raw = (Vec<u8>, ShaderReflection);

    /// What a translation was produced by. Entries made under a different stamp
    /// are not trusted — see the module doc.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct Stamp {
        pub reflection_version: u32,
        /// The translator's sources and locked dependencies (`build.rs`).
        pub translator: u64,
        /// `rustc -V`: the entry names are `DefaultHasher` digests (`build.rs`).
        pub toolchain: u64,
        /// Digest of every `METAL2VULKAN_*` variable in the environment.
        pub env_hash: u64,
    }

    impl Stamp {
        /// This build's stamp, or why there is none.
        ///
        /// Without one nothing could tell a fresh entry from a stale one, so the
        /// cache stays off rather than guess.
        pub(super) fn current() -> Result<Self, String> {
            let digest = |name: &str, hex: &str| {
                u64::from_str_radix(hex, 16).map_err(|e| format!("{name}: {e}"))
            };
            let translator = digest(
                "translator digest",
                env!("REIMS_VGPU_M2V_TRANSLATOR_DIGEST"),
            )?;
            let toolchain = digest("toolchain digest", env!("REIMS_VGPU_M2V_TOOLCHAIN_DIGEST"))?;

            let mut vars: Vec<(String, String)> = std::env::vars_os()
                .map(|(k, v)| {
                    (
                        k.to_string_lossy().into_owned(),
                        v.to_string_lossy().into_owned(),
                    )
                })
                .filter(|(k, _)| k.starts_with("METAL2VULKAN_"))
                .collect();
            vars.sort();
            let mut env = Vec::new();
            for (k, v) in vars {
                env.extend_from_slice(k.as_bytes());
                env.push(b'=');
                env.extend_from_slice(v.as_bytes());
                env.push(0);
            }
            Ok(Self {
                reflection_version: metal2vulkan::reflect::REFLECTION_VERSION,
                translator,
                toolchain,
                env_hash: checksum(&env),
            })
        }
    }

    /// Why an entry that exists was not used.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Stale {
        Magic,
        Schema,
        Stamp,
        Stage,
        LocalSize,
        Length,
        Air,
        Checksum,
        Reflection,
        Io,
    }

    impl Stale {
        pub(super) fn as_str(self) -> &'static str {
            match self {
                Self::Magic => "magic",
                Self::Schema => "schema",
                Self::Stamp => "stamp",
                Self::Stage => "stage",
                Self::LocalSize => "local_size",
                Self::Length => "length",
                Self::Air => "air",
                Self::Checksum => "checksum",
                Self::Reflection => "reflection",
                Self::Io => "io",
            }
        }
    }

    /// The counters the census line carries.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(super) struct Census {
        pub hits: u64,
        pub misses: u64,
        pub stale: u64,
        pub write_failures: u64,
    }

    pub(super) struct DiskCache {
        pub(super) dir: PathBuf,
        stamp: Stamp,
        hits: AtomicU64,
        misses: AtomicU64,
        stale: AtomicU64,
        write_failures: AtomicU64,
        error_reported: AtomicBool,
    }

    /// Where the cache lives when the switch is unset: `m2v-cache` beside the
    /// running executable. Unit tests get none, so they never touch the disk
    /// unless they ask to.
    #[cfg(not(test))]
    fn default_dir() -> Option<PathBuf> {
        Some(std::env::current_exe().ok()?.parent()?.join("m2v-cache"))
    }

    #[cfg(test)]
    fn default_dir() -> Option<PathBuf> {
        None
    }

    /// The process's cache: the switch's directory, `m2v-cache` beside the
    /// executable when it is unset, `None` for `off` (or when the executable
    /// could not be stamped). Latched on first use.
    pub(super) fn global() -> Option<&'static DiskCache> {
        static DISK: OnceLock<Option<DiskCache>> = OnceLock::new();
        DISK.get_or_init(|| {
            let dir = match std::env::var_os(ENV) {
                Some(v) if v == "off" || v == "0" => return None,
                Some(v) if !v.is_empty() => PathBuf::from(v),
                // On by default: `m2v-cache` beside the executable. A cold boot's
                // ~340 translations at 5-50 ms each, serialized on one lock, were
                // most of a slow login and of every first-time animation.
                _ => default_dir()?,
            };
            match Stamp::current() {
                Ok(stamp) => {
                    crate::observe::off(format!(
                        "linux_m2v_disk enabled dir={} schema={SCHEMA} reflection={} translator={:016x} \
                         toolchain={:016x} env={:016x}",
                        log_token(&dir.display().to_string()),
                        stamp.reflection_version,
                        stamp.translator,
                        stamp.toolchain,
                        stamp.env_hash,
                    ));
                    Some(DiskCache::new(dir, stamp))
                }
                Err(e) => {
                    crate::observe::fail(format!(
                        "linux_m2v_disk reason=no_stamp err={} (cache stays off: without an \
                         executable stamp a stale entry cannot be told from a fresh one)",
                        log_token(&e)
                    ));
                    None
                }
            }
        })
        .as_ref()
    }

    /// Run `translate` for an in-memory miss, answering from disk when `disk`
    /// holds this exact shader and writing the result when it does not.
    ///
    /// Everything the caller does with the result is the same on both paths;
    /// that is the point of caching the translator's output rather than ours.
    pub(super) fn obtain(
        disk: Option<&DiskCache>,
        id: ShaderId<'_>,
        translate: impl FnOnce() -> M2vResult<Raw>,
    ) -> M2vResult<Raw> {
        let Some(disk) = disk else {
            return translate();
        };
        if let Some(raw) = disk.load(id) {
            return Ok(raw);
        }
        let raw = translate()?;
        disk.store(id, &raw.0, &raw.1);
        Ok(raw)
    }

    fn tag_name(tag: u8) -> &'static str {
        match tag {
            1 => "vertex",
            2 => "fragment",
            3 => "kernel",
            _ => "unknown",
        }
    }

    impl DiskCache {
        pub(super) fn new(dir: PathBuf, stamp: Stamp) -> Self {
            Self {
                dir,
                stamp,
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                stale: AtomicU64::new(0),
                write_failures: AtomicU64::new(0),
                error_reported: AtomicBool::new(false),
            }
        }

        pub(super) fn census(&self) -> Census {
            Census {
                hits: self.hits.load(Ordering::Relaxed),
                misses: self.misses.load(Ordering::Relaxed),
                stale: self.stale.load(Ordering::Relaxed),
                write_failures: self.write_failures.load(Ordering::Relaxed),
            }
        }

        /// One file per shader, named by stage and the bucket digest.
        pub(super) fn path(&self, id: ShaderId<'_>) -> PathBuf {
            self.dir
                .join(format!("{}-{:016x}.m2v", tag_name(id.stage), id.digest))
        }

        /// The stored translation of exactly this shader, or `None`.
        pub(super) fn load(&self, id: ShaderId<'_>) -> Option<Raw> {
            let started = Instant::now();
            let path = self.path(id);
            let result = match std::fs::read(&path) {
                Ok(bytes) => decode(&self.stamp, &bytes, id).map_err(Some),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(None),
                Err(e) => {
                    self.report_once("read", &path, &e.to_string());
                    Err(Some(Stale::Io))
                }
            };
            let us = started.elapsed().as_micros();
            match result {
                Ok(raw) => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    self.line("hit", id, &format!("spv={} us={us}", raw.0.len()));
                    Some(raw)
                }
                Err(None) => {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    self.line("miss", id, &format!("us={us}"));
                    None
                }
                Err(Some(why)) => {
                    self.stale.fetch_add(1, Ordering::Relaxed);
                    self.line("stale", id, &format!("why={} us={us}", why.as_str()));
                    None
                }
            }
        }

        /// Persist a fresh translation. Never fails the caller.
        pub(super) fn store(&self, id: ShaderId<'_>, spirv: &[u8], reflection: &ShaderReflection) {
            let started = Instant::now();
            match self.try_store(id, spirv, reflection) {
                Ok(bytes) => {
                    let us = started.elapsed().as_micros();
                    self.line("stored", id, &format!("bytes={bytes} us={us}"));
                }
                Err((op, detail)) => {
                    self.write_failures.fetch_add(1, Ordering::Relaxed);
                    self.report_once(op, &self.path(id), &detail);
                    let us = started.elapsed().as_micros();
                    self.line("write_fail", id, &format!("op={op} us={us}"));
                }
            }
        }

        fn try_store(
            &self,
            id: ShaderId<'_>,
            spirv: &[u8],
            reflection: &ShaderReflection,
        ) -> Result<usize, (&'static str, String)> {
            let json = serde_json::to_vec(reflection).map_err(|e| ("serialize", e.to_string()))?;
            // JSON cannot carry a non-finite float: serde_json writes `null` for
            // one and then refuses to read it back, which would make this entry a
            // stale file on every boot and a write on every boot after. Prove the
            // round trip now, while the original is in hand, and keep such a
            // shader out of the directory instead.
            match serde_json::from_slice::<ShaderReflection>(&json) {
                Ok(back) if back == *reflection => {}
                Ok(_) => return Err(("roundtrip", "reflection changed in transit".into())),
                Err(e) => return Err(("roundtrip", e.to_string())),
            }
            let bytes = encode(&self.stamp, id, spirv, &json);

            std::fs::create_dir_all(&self.dir).map_err(|e| ("mkdir", e.to_string()))?;
            // Written beside the destination and renamed over it, so a reader —
            // this process, or the other VM's — sees the old file or the whole new
            // one. The name carries the pid and a sequence so two processes (or an
            // inline and a background translation of the same shader) cannot
            // share a temporary.
            static SEQ: AtomicU64 = AtomicU64::new(0);
            let path = self.path(id);
            let tmp = path.with_extension(format!(
                "{}.{}.tmp",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&tmp, &bytes).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                ("write", e.to_string())
            })?;
            std::fs::rename(&tmp, &path).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                ("rename", e.to_string())
            })?;
            Ok(bytes.len())
        }

        /// The first disk error of the process, on the always-on channel. Every
        /// one is counted; one line is enough to say the cache is not working.
        fn report_once(&self, op: &str, path: &std::path::Path, detail: &str) {
            if !self.error_reported.swap(true, Ordering::Relaxed) {
                crate::observe::fail(format!(
                    "linux_m2v_disk reason=io_error op={op} file={} err={} (the translation is \
                     used regardless; further disk errors are counted, not logged)",
                    log_token(&path.display().to_string()),
                    log_token(detail)
                ));
            }
        }

        /// The census line: one per disk event, each carrying the running
        /// counters, so the last one in a log is the boot's total.
        fn line(&self, event: &str, id: ShaderId<'_>, detail: &str) {
            let c = self.census();
            crate::observe::off(format!(
                "linux_m2v_disk {event} stage={} key={:016x} air={} {detail} disk_hits={} \
                 disk_misses={} disk_stale={} disk_write_fail={}",
                tag_name(id.stage),
                id.digest,
                id.air.len(),
                c.hits,
                c.misses,
                c.stale,
                c.write_failures
            ));
        }
    }

    /// Integrity check, not a security one: every step is a bijection of the
    /// running value, so corruption confined to one 8-byte lane always changes
    /// the result, and the length is mixed in so truncation does too.
    pub(super) fn checksum(bytes: &[u8]) -> u64 {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ bytes.len() as u64;
        let mut lanes = bytes.chunks_exact(8);
        for c in &mut lanes {
            let lane = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            h = (h ^ lane).wrapping_mul(PRIME);
        }
        let rest = lanes.remainder();
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            h = (h ^ u64::from_le_bytes(last)).wrapping_mul(PRIME);
        }
        h ^ (h >> 32)
    }

    /// The entry file for `id`: header, AIR, SPIR-V, reflection JSON, checksum
    /// of everything before it. All integers little-endian.
    ///
    /// ```text
    /// magic[8] schema:u32
    /// stamp: reflection_version:u32 translator:u64 toolchain:u64 env_hash:u64
    /// stage:u8 has_local_size:u8 local_size:[u32;3]
    /// air_len:u64 spirv_len:u64 json_len:u64
    /// air spirv json
    /// checksum:u64
    /// ```
    pub(super) fn encode(stamp: &Stamp, id: ShaderId<'_>, spirv: &[u8], json: &[u8]) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(HEADER_LEN + id.air.len() + spirv.len() + json.len() + TRAILER_LEN);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&SCHEMA.to_le_bytes());
        out.extend_from_slice(&stamp.reflection_version.to_le_bytes());
        out.extend_from_slice(&stamp.translator.to_le_bytes());
        out.extend_from_slice(&stamp.toolchain.to_le_bytes());
        out.extend_from_slice(&stamp.env_hash.to_le_bytes());
        out.push(id.stage);
        out.push(u8::from(id.local_size.is_some()));
        for d in id.local_size.unwrap_or([0; 3]) {
            out.extend_from_slice(&d.to_le_bytes());
        }
        for len in [id.air.len(), spirv.len(), json.len()] {
            out.extend_from_slice(&(len as u64).to_le_bytes());
        }
        out.extend_from_slice(id.air);
        out.extend_from_slice(spirv);
        out.extend_from_slice(json);
        let sum = checksum(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        out
    }

    struct Reader<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let end = self.pos.checked_add(n)?;
            let slice = self.bytes.get(self.pos..end)?;
            self.pos = end;
            Some(slice)
        }

        fn u8(&mut self) -> Option<u8> {
            self.take(1).map(|b| b[0])
        }

        fn u32(&mut self) -> Option<u32> {
            self.take(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }

        fn u64(&mut self) -> Option<u64> {
            self.take(8)
                .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        }
    }

    /// An entry's translation, if the file is a whole, current entry for exactly
    /// `id`; otherwise the first reason it is not. Total over every byte string.
    pub(super) fn decode(stamp: &Stamp, bytes: &[u8], id: ShaderId<'_>) -> Result<Raw, Stale> {
        let mut r = Reader { bytes, pos: 0 };
        let truncated = Stale::Length;
        if bytes.len() < HEADER_LEN + TRAILER_LEN {
            // Too short to hold a header: say *what* it is if the prefix we have
            // is not even ours, so a foreign file and a torn one read differently.
            return Err(if bytes.get(..MAGIC.len()) == Some(&MAGIC[..]) {
                truncated
            } else {
                Stale::Magic
            });
        }
        if r.take(MAGIC.len()) != Some(&MAGIC[..]) {
            return Err(Stale::Magic);
        }
        if r.u32() != Some(SCHEMA) {
            return Err(Stale::Schema);
        }
        let stored = Stamp {
            reflection_version: r.u32().ok_or(truncated)?,
            translator: r.u64().ok_or(truncated)?,
            toolchain: r.u64().ok_or(truncated)?,
            env_hash: r.u64().ok_or(truncated)?,
        };
        if stored != *stamp {
            return Err(Stale::Stamp);
        }
        if r.u8().ok_or(truncated)? != id.stage {
            return Err(Stale::Stage);
        }
        let has_local = r.u8().ok_or(truncated)?;
        let local = [
            r.u32().ok_or(truncated)?,
            r.u32().ok_or(truncated)?,
            r.u32().ok_or(truncated)?,
        ];
        let stored_local = match has_local {
            0 if local == [0; 3] => None,
            1 => Some(local),
            _ => return Err(Stale::LocalSize),
        };
        if stored_local != id.local_size {
            return Err(Stale::LocalSize);
        }
        let mut lens = [0usize; 3];
        for len in &mut lens {
            *len = usize::try_from(r.u64().ok_or(truncated)?).map_err(|_| truncated)?;
        }
        let [air_len, spirv_len, json_len] = lens;
        // The lengths are the file's own claim: they are believed only once they
        // account for every byte, and nothing is allocated from them.
        let payload = air_len
            .checked_add(spirv_len)
            .and_then(|n| n.checked_add(json_len))
            .ok_or(truncated)?;
        if HEADER_LEN
            .checked_add(payload)
            .and_then(|n| n.checked_add(TRAILER_LEN))
            != Some(bytes.len())
        {
            return Err(truncated);
        }
        let air = r.take(air_len).ok_or(truncated)?;
        if air != id.air {
            return Err(Stale::Air);
        }
        let spirv = r.take(spirv_len).ok_or(truncated)?;
        let json = r.take(json_len).ok_or(truncated)?;
        let body_end = r.pos;
        let sum = r.u64().ok_or(truncated)?;
        if sum != checksum(&bytes[..body_end]) {
            return Err(Stale::Checksum);
        }
        let reflection =
            serde_json::from_slice::<ShaderReflection>(json).map_err(|_| Stale::Reflection)?;
        Ok((spirv.to_vec(), reflection))
    }
}

fn translate_air(air: &[u8], stage: Stage) -> M2vResult<CachedShader> {
    let id = ShaderId::render(stage, air);
    let (spirv, reflection) = disk::obtain(disk::global(), id, || translate_air_raw(air, stage))?;
    capture_air(air, stage, &spirv);
    finish_translated(spirv, reflection, stage)
}

/// The translator's own output for a render stage: no cache, no post-processing.
fn translate_air_raw(air: &[u8], stage: Stage) -> M2vResult<disk::Raw> {
    // metal2vulkan's tool boundary uses fixed scratch names inside `tmp_dir`.
    // Serialize sync and background calls so their AIR/LLVM/SPIR-V files never
    // alias one another.
    let _guard = translation_lock().lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tmp_dir();
    let name = match stage {
        Stage::Vertex => "v.air",
        Stage::Fragment => "f.air",
        Stage::Kernel => "k.air",
    };
    let path = tmp.join(name);
    std::fs::write(&path, air).map_err(|e| scratch_write_decline(stage, e.to_string()))?;
    // Reflected translate: byte-identical SPIR-V PLUS the stage-interface facade.
    // `reflection.datalayout` carries the source `target datalayout` the sanitizer
    // strips, so the post-emit ABI reconciliation below no longer re-reads `k.ll`.
    metal2vulkan::translate_reflected(path.to_str().unwrap_or(name), stage, &tmp)
        .map_err(|e| translate_decline(stage, e.to_string()))
}

fn translate_kernel_air(air: &[u8], local_size: [u32; 3]) -> M2vResult<CachedShader> {
    let id = ShaderId::kernel(air, local_size);
    let (spirv, reflection) = disk::obtain(disk::global(), id, || {
        translate_kernel_air_raw(air, local_size)
    })?;
    capture_air(air, Stage::Kernel, &spirv);
    if reflection.local_size != Some(local_size) {
        return Err(M2vCacheDecline::KernelLocalSizeMismatch {
            requested: local_size,
            reflected: reflection.local_size,
        });
    }
    finish_translated(spirv, reflection, Stage::Kernel)
}

/// The translator's own output for a kernel at `local_size`: no cache, no
/// post-processing, and no check that the reflection agrees with the request —
/// that one runs on a stored translation too.
fn translate_kernel_air_raw(air: &[u8], local_size: [u32; 3]) -> M2vResult<disk::Raw> {
    let _guard = translation_lock().lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tmp_dir();
    let path = tmp.join("k.air");
    std::fs::write(&path, air).map_err(|e| M2vCacheDecline::KernelScratchWrite {
        detail: e.to_string(),
    })?;
    let opts = metal2vulkan::passes::TransformOptions {
        kernel_local_size: local_size,
        ..Default::default()
    };
    metal2vulkan::translate_reflected_with_options(
        path.to_str().unwrap_or("k.air"),
        Stage::Kernel,
        &tmp,
        opts,
    )
    .map_err(|e| M2vCacheDecline::KernelTranslate {
        detail: e.to_string(),
    })
}

/// Post-emit ABI reconciliation + package the translated shader with its reflection.
fn finish_translated(
    spirv: Vec<u8>,
    reflection: ShaderReflection,
    stage: Stage,
) -> M2vResult<CachedShader> {
    validate_reflection(&reflection, stage)?;
    let spirv = repair_layout(&reflection, spirv, stage)?;
    // Once-per-translate: surface any runtime function constants this shader
    // declares. metal2vulkan folds them to their disabled default and the paravirt
    // stream carries no MTLFunctionConstantValues, so the FC-disabled variant is
    // what we run; this makes that reliance measurable (which shaders use FCs)
    // without touching translation or rendering. Silent for FC-free shaders.
    crate::runtime::spirv_bind::log_folded_function_constants(&reflection);
    Ok(CachedShader::new(spirv, Arc::new(reflection)))
}

/// Once-per-translate (miss path) well-formedness guard on the AIR-derived
/// reflection. This is the always-on regression proxy for the reflection-fed hot
/// path: it validates the reflection's ABI version and its internal
/// sampled-vs-storage consistency without a second walk of the SPIR-V. Must read
/// zero on a healthy boot.
fn validate_reflection(reflection: &ShaderReflection, stage: Stage) -> M2vResult<()> {
    // pipeline_ref is telemetry only here; the stage tag localizes the shader.
    let pipe = stage_tag(stage) as u32;
    let violations = crate::runtime::spirv_bind::census_reflection_wellformed(reflection, pipe);
    if violations != 0 {
        return Err(M2vCacheDecline::ReflectionMalformed {
            stage: stage_name(stage),
            violations,
        });
    }
    Ok(())
}

fn repair_layout(
    reflection: &ShaderReflection,
    spirv: Vec<u8>,
    stage: Stage,
) -> M2vResult<Vec<u8>> {
    // The datalayout is the reflection's — single source of truth, no `k.ll` re-read.
    // A reflected translate from unsanitized AIR always populates it; its absence
    // is a genuine gap (fail-visible), matching the prior `air_datalayout_missing`.
    let datalayout =
        reflection
            .datalayout
            .as_deref()
            .ok_or(M2vCacheDecline::ReflectionDatalayoutMissing {
                stage: stage_name(stage),
            })?;
    let (spirv, stats) =
        crate::runtime::spirv_layout::repair_llvm_vector_alloc_offsets_from_datalayout(
            datalayout, &spirv,
        )
        .map_err(|reason| M2vCacheDecline::LayoutRepair {
            stage: stage_name(stage),
            reason,
        })?;
    if stats.members != 0 {
        crate::observe::fail(format!(
            "linux_m2v_layout_repair stage={stage:?} reason=llvm_vector_alloc_stride structs={} members={}",
            stats.structs, stats.members
        ));
    }
    Ok(spirv)
}

/// Start translating a render stage without holding protocol state or the
/// sole FIFO scheduler. Returns true when the content is already resolved
/// (success or deterministic failure), false while the background worker owns
/// it. Callers keep the guest packet at the channel head and retry on poll.
///
/// `task_id` and `pipeline_ref` name the guest pipeline that is asking. They
/// feed only the content-change census ([`Cache::last_identity`]): pipeline
/// references are per task, so the reference alone is not an identity.
pub fn ensure_cached_async(air: &[u8], stage: Stage, task_id: u32, pipeline_ref: u32) -> bool {
    ensure_cached_async_keyed(
        ShaderId::render(stage, air),
        stage,
        None,
        task_id,
        pipeline_ref,
    )
}

/// Kernel counterpart to [`ensure_cached_async`]. LocalSize is part of both
/// the translation options and cache key, so two dispatch geometries can never
/// alias one another.
pub fn ensure_cached_kernel_async(
    air: &[u8],
    local_size: [u32; 3],
    task_id: u32,
    pipeline_ref: u32,
) -> bool {
    if local_size.contains(&0) {
        return true;
    }
    ensure_cached_async_keyed(
        ShaderId::kernel(air, local_size),
        Stage::Kernel,
        Some(local_size),
        task_id,
        pipeline_ref,
    )
}

/// End the content-change census's memory of one task's pipeline references.
///
/// Called when the guest deletes or redefines the task: the guest reuses task
/// ids, and a new incarnation naming different AIR under an old number is not
/// a change of anything.
pub(crate) fn forget_task_identities(task_id: u32) {
    global()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .forget_task_identities(task_id);
}

/// End the census's memory of every task, with the device that owned them.
pub(crate) fn forget_all_identities() {
    let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
    c.last_identity.clear();
}

fn ensure_cached_async_keyed(
    id: ShaderId<'_>,
    stage: Stage,
    kernel_local_size: Option<[u32; 3]>,
    task_id: u32,
    pipeline_ref: u32,
) -> bool {
    let mut start_worker = false;
    let changed: Option<(u64, u64)>;
    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        match c.find(id) {
            // A stored failure resolves the ask, transient or not, and that is
            // load-bearing: the caller holds the guest packet at the channel
            // head and re-polls until this answers `true`, so an arm that
            // re-queued instead would never let the packet advance. The retry
            // is armed where the error is *consumed* — see the `Entry::Failed`
            // arm of [`translate_cached_reflected`] — which costs the failing
            // draw its failure and leaves the next draw a clean cache.
            Some(Entry::Ready(_)) | Some(Entry::Failed(_)) => return true,
            Some(Entry::Loading) => return false,
            None => {}
        }
        // A miss for a ref this task has already translated something else
        // under. See `Cache::last_identity`: a census of the guest reusing a
        // reference number for different AIR, not a stale-shader alarm.
        changed = c.note_identity(task_id, pipeline_ref, id);
        let air: Arc<[u8]> = Arc::from(id.air);
        c.put(id, &air, Entry::Loading);
        c.misses = c.misses.saturating_add(1);
        c.async_queue.push_back(TranslationTask {
            stage,
            kernel_local_size,
            air,
            pipeline_ref,
        });
        if !c.async_worker_running {
            c.async_worker_running = true;
            start_worker = true;
        }
    }
    let dims = kernel_local_size
        .map(|d| format!(" tg=[{},{},{}]", d[0], d[1], d[2]))
        .unwrap_or_default();
    // Queue census: fires once per cold shader on the normal path — OFF, not a
    // curated failure. The `done` line below stays fail-visible on translation
    // failure (a shader that will not render).
    crate::observe::off(format!(
        "linux_m2v_async queued pipe={pipeline_ref} stage={stage:?}{dims} air={}",
        id.air.len()
    ));
    // Once per `(task, ref, stage)` rather than once per event. OFF, not a
    // failure: the only caller asks after the retained pipeline registry
    // answered "not retained" for this `(task, ref)`, so the guest naming
    // different AIR under a number it used before (a delete, then a reuse) has
    // nothing stale to be served and loses no work. The magnitude still shows
    // as `seen:m2v_pipe_content_changed` in `refusal_census`.
    if let Some((previous, now)) = changed {
        if crate::observe::first_sight(
            "m2v_pipe_content_changed",
            (u64::from(task_id) << 40) | (u64::from(pipeline_ref) << 8) | u64::from(id.stage),
        ) {
            crate::observe::off(format!(
                "linux_m2v reason=m2v_pipe_content_changed task={task_id} pipe={pipeline_ref} \
                 stage={stage:?}{dims} air={} was={previous:#x} now={now:#x} (the guest named \
                 different AIR under a reference this task used before; nothing was retained \
                 for it when it was asked)",
                id.air.len()
            ));
        }
    }
    if start_worker {
        std::thread::spawn(async_worker);
    }
    false
}

fn async_worker() {
    loop {
        let task = {
            let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
            match c.async_queue.pop_front() {
                Some(task) => task,
                None => {
                    c.async_worker_running = false;
                    return;
                }
            }
        };
        let result = match task.kernel_local_size {
            Some(local_size) => translate_kernel_air(&task.air, local_size),
            None => translate_air(&task.air, task.stage),
        };
        let (hits, misses, detail, failure) = {
            let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
            let (detail, failure) = match result {
                Ok(shader) => {
                    let len = shader.spirv.len();
                    // Report the vertex builtins the translated shader uses.
                    // instance_index presence proves an instanced draw's
                    // per-instance indexing survived AIR->SPIR-V; its absence on a
                    // shader driven by an instanced draw would localize a dropped
                    // [[instance_id]]. Load-bearing for instanced-draw diagnosis.
                    // The reflection facade reports it from the SAME parsed AIR
                    // roles the emitter decorated from — single source of truth.
                    if task.stage == Stage::Vertex {
                        let vb = shader.reflection.vertex_builtins.unwrap_or_default();
                        // Cross-check the AIR-derived reflection against the emitted
                        // SPIR-V decorations, once per translate (miss path only), so a
                        // translator regression that drops a builtin between AIR meta and
                        // emit is fail-visible instead of silent. Quiet on a healthy boot.
                        let emit_instance = spirv_uses_builtin(&shader.spirv, 43);
                        let emit_vertex = spirv_uses_builtin(&shader.spirv, 42);
                        if emit_instance != vb.uses_instance_index
                            || emit_vertex != vb.uses_vertex_index
                        {
                            crate::observe::fail(format!(
                                "m2v_reflect_divergence pipe={} stage=Vertex kind=vertex_builtins reflect_instance={} emit_instance={} reflect_vertex={} emit_vertex={}",
                                task.pipeline_ref,
                                vb.uses_instance_index as u8, emit_instance as u8,
                                vb.uses_vertex_index as u8, emit_vertex as u8
                            ));
                        }
                        // Decoded vertex-builtin usage census (not an error);
                        // route off() so it leaves the curated real-error view.
                        // The genuine reflect-vs-emit divergence above stays
                        // fail()-visible.
                        crate::observe::off(format!(
                            "m2v_builtins pipe={} stage=Vertex instance_index={} vertex_index={}",
                            task.pipeline_ref,
                            vb.uses_instance_index as u8,
                            vb.uses_vertex_index as u8
                        ));
                    }
                    c.put(task.id(), &task.air, Entry::Ready(Arc::new(shader)));
                    (format!("ok spv={len}"), None)
                }
                Err(e) => {
                    c.put(task.id(), &task.air, Entry::Failed(e.clone()));
                    ("fail".to_string(), Some(e))
                }
            };
            (c.hits, c.misses, detail, failure)
        };
        let dims = task
            .kernel_local_size
            .map(|d| format!(" tg=[{},{},{}]", d[0], d[1], d[2]))
            .unwrap_or_default();
        // A successful translate is census (OFF); a failed one means the shader
        // will not render — stay fail-visible.
        if let Some(error) = failure {
            let mut emit = crate::observe::Emit::decline("linux_m2v_async_done", &error)
                .field("pipe", task.pipeline_ref)
                .field("hits", hits)
                .field("misses", misses);
            if let Some([x, y, z]) = task.kernel_local_size {
                emit = emit.field("tg_x", x).field("tg_y", y).field("tg_z", z);
            }
            emit.fail_once(task.id().digest);
        } else {
            let done = format!(
                "linux_m2v_async done pipe={} stage={:?}{dims} {detail} hits={hits} misses={misses}",
                task.pipeline_ref, task.stage
            );
            crate::observe::off(done);
        }
    }
}

/// Translate `air` for `stage`, returning cached SPIR-V when AIR matches a prior
/// translate. Logs `linux_m2v_translate` on miss and `linux_m2v_translate_hit`
/// on hit (pipe id is telemetry only).
/// Does the SPIR-V module decorate any id with `BuiltIn <builtin>`? Scans
/// `OpDecorate` (opcode 71) instructions for a `BuiltIn` (decoration 11) whose
/// value equals `builtin` (e.g. 42 = VertexIndex, 43 = InstanceIndex).
/// Bounds-checked word walk over the little-endian SPIR-V stream (header skipped).
fn spirv_uses_builtin(spv: &[u8], builtin: u32) -> bool {
    if spv.len() < 20 || !spv.len().is_multiple_of(4) {
        return false;
    }
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut i = 5; // skip header
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = words[i] & 0xffff;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        // OpDecorate = 71: [target, decoration, operands...]; BuiltIn = 11.
        if opcode == 71 && word_count >= 4 && words[i + 2] == 11 && words[i + 3] == builtin {
            return true;
        }
        i += word_count;
    }
    false
}

/// Count a translation taken inline because the async worker had it queued but
/// unfinished when an encode needed it, and name the first per `(ref, stage)`.
///
/// Not a failure — the draw is served — but it is the population whose size says
/// how often admission let a packet through ahead of its own translation.
fn note_loading_translated_inline(pipeline_ref: u32, stage: &'static str) {
    crate::runtime::drain::note_store_route("m2v_loading_translated_inline");
    let stage_key = match stage {
        "vertex" => 1u64,
        "fragment" => 2,
        _ => 3,
    };
    if crate::observe::first_sight(
        "m2v_loading_translated_inline",
        u64::from(pipeline_ref) << 8 | stage_key,
    ) {
        crate::observe::off(format!(
            "m2v_loading_translated_inline pipe={pipeline_ref} stage={stage} (queued for the \
             async worker and needed now; translated on the encoding thread instead of \
             dropping the draw)"
        ));
    }
}

/// Translate `air` for `stage`, returning the whole [`CachedShader`] (SPIR-V +
/// reflection) as a shared handle, so a consumer reads stage-interface facts
/// (texture shapes, vertex builtins, descriptor bindings) from the reflection
/// instead of re-walking the emitted SPIR-V. The returned `Arc` is a clone of the
/// cached entry — a warm hit performs no allocation beyond the refcount bump.
pub fn translate_cached_reflected(
    air: &[u8],
    stage: Stage,
    pipeline_ref: u32,
) -> M2vResult<Arc<CachedShader>> {
    let id = ShaderId::render(stage, air);
    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        match c.find(id).cloned() {
            Some(Entry::Ready(shader)) => {
                c.hits = c.hits.saturating_add(1);
                let hits = c.hits;
                let misses = c.misses;
                drop(c);
                // A cache HIT is the per-draw hot path (≈2×/draw) and carries only a
                // cumulative counter — logging it always-on flooded /tmp/reims-vgpu-fail.log
                // (~1.1M lines / boot, drowning real failures) and paid a file write
                // per draw on the render path. Verbose-gated only; the miss/`ok` line
                // below stays always-on so cache lifecycle is still visible.
                crate::observe::verbose(|| {
                    format!(
                        "linux_m2v_translate_hit pipe={pipeline_ref} stage={stage:?} spv={} hits={hits} misses={misses}",
                        shader.spirv.len()
                    )
                });
                return Ok(shader);
            }
            Some(Entry::Failed(e)) => {
                forget_if_transient(&mut c, id, &e);
                return Err(e);
            }
            // Queued for the async worker but not done. The draw asking now is
            // past every point that could have deferred it, so refusing costs
            // the guest the draw: the device then stores the clear colour, and
            // macOS 26's rounded-corner tiles and widget layers came out opaque
            // black this way whenever a pipeline ref was re-pointed at new AIR
            // just before a packet binding it was admitted. Translate here, as a
            // miss does; the worker's later `put` of the same bytes is a no-op.
            Some(Entry::Loading) => note_loading_translated_inline(pipeline_ref, stage_name(stage)),
            None => {}
        }
    }

    let shader = Arc::new(translate_air(air, stage)?);

    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        c.misses = c.misses.saturating_add(1);
        c.put(id, &Arc::from(air), Entry::Ready(Arc::clone(&shader)));
        let hits = c.hits;
        let misses = c.misses;
        drop(c);
        crate::observe::off(format!(
            "linux_m2v_translate ok pipe={pipeline_ref} stage={stage:?} v_spv_or_f={} hits={hits} misses={misses}",
            shader.spirv.len()
        ));
    }
    Ok(shader)
}

/// Translate a **compute** AIR kernel with explicit LocalSize (threadgroup dims),
/// returning the whole [`CachedShader`] (SPIR-V + reflection). See
/// [`translate_cached_reflected`].
///
/// Wallpaper CI dispatches use non-default tg sizes (e.g. 16×16×1). Default
/// m2v LocalSize is 64×1×1 — wrong for those grids. Cache key includes LocalSize.
pub fn translate_cached_kernel_reflected(
    air: &[u8],
    local_size: [u32; 3],
    pipeline_ref: u32,
) -> M2vResult<Arc<CachedShader>> {
    if local_size.contains(&0) {
        return Err(M2vCacheDecline::KernelLocalSizeZero { local_size });
    }
    let id = ShaderId::kernel(air, local_size);
    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        match c.find(id).cloned() {
            Some(Entry::Ready(shader)) => {
                c.hits = c.hits.saturating_add(1);
                let hits = c.hits;
                let misses = c.misses;
                drop(c);
                // Verbose-gated (see `translate_cached_reflected`): a compute-kernel cache hit
                // is the hot path and only carries a cumulative counter.
                crate::observe::verbose(|| {
                    format!(
                        "linux_m2v_translate_hit pipe={pipeline_ref} stage=Kernel tg=[{},{},{}] spv={} hits={hits} misses={misses}",
                        local_size[0], local_size[1], local_size[2], shader.spirv.len()
                    )
                });
                return Ok(shader);
            }
            Some(Entry::Failed(e)) => {
                forget_if_transient(&mut c, id, &e);
                return Err(e);
            }
            // See `translate_cached_reflected`: a dispatch that reaches a
            // queued kernel translates it rather than being dropped.
            Some(Entry::Loading) => note_loading_translated_inline(pipeline_ref, "kernel"),
            None => {}
        }
    }

    let shader = Arc::new(translate_kernel_air(air, local_size)?);

    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        c.misses = c.misses.saturating_add(1);
        c.put(id, &Arc::from(air), Entry::Ready(Arc::clone(&shader)));
        let hits = c.hits;
        let misses = c.misses;
        drop(c);
        crate::observe::off(format!(
            "linux_m2v_translate ok pipe={pipeline_ref} stage=Kernel tg=[{},{},{}] spv={} hits={hits} misses={misses}",
            local_size[0], local_size[1], local_size[2], shader.spirv.len()
        ));
    }
    Ok(shader)
}

/// Snapshot counters for tests.
///
/// The census does not come through here — the `linux_m2v_translate ok` line
/// above reads `hits`/`misses` off the lock it already holds. This is only how a
/// test asks the same question from outside, which is why it cannot be
/// `#[cfg(test)]`: `tests/reflection_adoption.rs` is a separate crate.
pub fn stats() -> (u64, u64, usize) {
    let c = global().lock().unwrap_or_else(|e| e.into_inner());
    (c.hits, c.misses, c.len())
}

/// Test isolation.
#[cfg(test)]
pub fn reset_for_test() {
    let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
    c.entries.clear();
    c.hits = 0;
    c.misses = 0;
    c.async_queue.clear();
    c.async_worker_running = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The content-change census names one task reusing a reference number for
    /// different AIR, and nothing else.
    ///
    /// It is checked at the map rather than through the log line, because the
    /// line is latched per key for a whole process and a unit suite is one
    /// process: a test that asserted on the line would pass or fail on whether
    /// some earlier test happened to use the same key. A private [`Cache`]
    /// keeps the state this test's own.
    ///
    /// The population it used to name was a defect of the key. Pipeline
    /// references are per task, so with the task left out, two tasks' unrelated
    /// pipelines under one number read as one pipeline changing (57 of the 63
    /// refs a driven macos-15 boot reported), and a kernel dispatched at two
    /// threadgroup sizes read as its AIR changing.
    #[test]
    fn only_one_task_reusing_a_ref_for_different_air_is_a_content_change() {
        let mut c = Cache::default();
        const REF: u32 = 30;
        let a = ShaderId::render(Stage::Fragment, b"air-one");
        let b = ShaderId::render(Stage::Fragment, b"air-two-longer");
        assert_ne!(a.digest, b.digest, "the fixture needs two identities");

        assert_eq!(
            c.note_identity(1, REF, a),
            None,
            "the first translation of a ref has nothing to differ from"
        );
        assert_eq!(
            c.note_identity(1, REF, a),
            None,
            "the same AIR asked for twice is not a change"
        );
        assert_eq!(
            c.note_identity(3, REF, b),
            None,
            "another task's pipeline under the same number is not this task's change"
        );
        assert_eq!(
            c.note_identity(1, REF, a),
            None,
            "and it did not disturb the first task's record"
        );
        assert_eq!(
            c.note_identity(1, REF, b),
            Some((a.digest, b.digest)),
            "different AIR under one task's ref is the change the census names"
        );
        assert_eq!(
            c.note_identity(1, REF, ShaderId::render(Stage::Vertex, b"air-one")),
            None,
            "the stage is part of the key: a vertex stage is not the fragment stage changing"
        );
    }

    /// One kernel dispatched at two threadgroup sizes is two translations of
    /// one AIR, which is not a change; different AIR under the ref is.
    #[test]
    fn a_kernel_dispatched_at_two_geometries_is_not_a_content_change() {
        let mut c = Cache::default();
        let air = b"kernel-air";
        let small = ShaderId::kernel(air, [8, 8, 1]);
        let large = ShaderId::kernel(air, [16, 16, 1]);
        assert_ne!(small.digest, large.digest, "the cache keeps the two apart");
        assert_eq!(small.air_digest(), large.air_digest());

        assert_eq!(c.note_identity(2, 9, small), None);
        assert_eq!(c.note_identity(2, 9, large), None);
        let other = ShaderId::kernel(b"other-kernel-air", [16, 16, 1]);
        assert_eq!(
            c.note_identity(2, 9, other),
            Some((large.air_digest(), other.air_digest()))
        );
    }

    /// The guest reuses task ids, so a task's records end with the task.
    #[test]
    fn forgetting_a_task_ends_only_that_tasks_records() {
        let mut c = Cache::default();
        let a = ShaderId::render(Stage::Fragment, b"air-one");
        let b = ShaderId::render(Stage::Fragment, b"air-two-longer");
        assert_eq!(c.note_identity(1, 5, a), None);
        assert_eq!(c.note_identity(3, 5, a), None);

        c.forget_task_identities(1);

        assert_eq!(
            c.note_identity(1, 5, b),
            None,
            "a new incarnation of task 1 naming other AIR under ref 5 is not a change"
        );
        assert_eq!(
            c.note_identity(3, 5, b),
            Some((a.digest, b.digest)),
            "task 3 was not forgotten"
        );
    }

    /// A variant's reflected sampler interface must agree with the executable
    /// module in *that variant's* numbering.
    ///
    /// This is the invariant [`ShaderVariant`] exists to make unrepresentable,
    /// and it is worth a test because the failure it replaces was silent: the
    /// draw path used to walk `f_words` at one point in the function and take
    /// the relocation flags at another, so a variant's answer and its numbering
    /// only agreed because two call sites happened to be passed the same pair.
    /// A relocated module renumbers its descriptor bindings, so an answer taken
    /// for the wrong variant is a real sampler bound at the wrong number.
    #[test]
    fn every_variants_reflected_sampler_bindings_match_the_executable_module() {
        let translator_base = metal2vulkan::reflect::SAMPLER_BINDING_BASE;
        let words = crate::runtime::spirv_bind::test_module_with_samplers(&[
            translator_base + 3,
            translator_base + 7,
            translator_base + 11,
        ]);
        let spirv: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let shader = synth_shader_with_samplers(Stage::Fragment, spirv, &[3, 7, 11]);

        // The fixture is only useful if the walk finds something in it. Not
        // asserted against the numbers the fixture was built with:
        // `CachedShader::new` runs `widen_sampled_bands` over every module, so
        // the base variant is already in the device's numbering rather than the
        // translator's, and that is the numbering every consumer sees.
        assert_eq!(
            shader.variant(false, false).samplers.len(),
            3,
            "the fixture's three samplers survive band widening"
        );

        for (separate_sampled, buf_collide) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let v = shader.variant(separate_sampled, buf_collide);
            assert_eq!(
                v.samplers
                    .iter()
                    .map(|sampler| sampler.binding)
                    .collect::<Vec<_>>(),
                crate::runtime::spirv_bind::sampler_bindings(&v.words),
                "variant({separate_sampled}, {buf_collide})"
            );
        }
    }

    /// A variant is computed once and handed out by pointer after that, which is
    /// what makes the walk a per-shader cost rather than a per-draw one.
    #[test]
    fn a_variant_is_memoized_rather_than_rewalked() {
        let words = crate::runtime::spirv_bind::test_module_with_samplers(&[5]);
        let spirv: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let shader = synth_shader(Stage::Fragment, spirv);
        for (separate_sampled, buf_collide) in [(false, false), (true, true)] {
            let first = shader.variant(separate_sampled, buf_collide);
            let again = shader.variant(separate_sampled, buf_collide);
            assert!(
                Arc::ptr_eq(&first, &again),
                "variant({separate_sampled}, {buf_collide}) was rebuilt"
            );
        }
    }

    /// Static use belongs to the executable variant, not to each draw that
    /// binds it. A declaration alone is legal to omit from a Vulkan layout;
    /// only an instruction reference enters the retained guard set.
    #[test]
    fn a_variant_retains_only_statically_used_descriptor_bindings() {
        let unused = crate::runtime::spirv_bind::test_support::module_with_descriptor(33, false);
        let used = crate::runtime::spirv_bind::test_support::module_with_descriptor(34, true);

        let unused_shader = synth_shader(
            Stage::Fragment,
            unused.iter().flat_map(|word| word.to_le_bytes()).collect(),
        );
        let used_shader = synth_shader(
            Stage::Fragment,
            used.iter().flat_map(|word| word.to_le_bytes()).collect(),
        );

        assert!(unused_shader
            .variant(false, false)
            .used_descriptor_bindings
            .is_empty());
        assert_eq!(
            used_shader
                .variant(false, false)
                .used_descriptor_bindings
                .as_ref(),
            &[34]
        );
    }

    /// A minimal `CachedShader` wrapping raw bytes with an empty reflection —
    /// enough to prime the cache in unit tests that never call metal2vulkan.
    fn synth_shader(stage: Stage, spirv: Vec<u8>) -> Arc<CachedShader> {
        use metal2vulkan::reflect::{
            DescriptorLayout, ShaderReflection, ShaderStage, REFLECTION_VERSION,
        };
        let stage = match stage {
            Stage::Vertex => ShaderStage::Vertex,
            Stage::Fragment => ShaderStage::Fragment,
            Stage::Kernel => ShaderStage::Kernel,
        };
        Arc::new(CachedShader::new(
            spirv,
            Arc::new(ShaderReflection {
                reflection_version: REFLECTION_VERSION,
                stage,
                entry_point: None,
                bindings: vec![],
                argument_buffer_fields: vec![],
                vertex_attributes: vec![],
                varyings: vec![],
                render_targets: vec![],
                depth_members: vec![],
                depth_qualifier: None,
                stencil_members: vec![],
                local_size: (stage == ShaderStage::Kernel).then_some([1, 1, 1]),
                max_work_group_size: None,
                vertex_builtins: None,
                tessellation: None,
                imageblock_layouts: vec![],
                implicit_imageblock_attachments: vec![],
                fragment_imageblock: None,
                datalayout: None,
                descriptor_layout: DescriptorLayout::default(),
                kernel_dispatch: None,
                runtime_sampler_specializations: vec![],
                runtime_storage_image_specializations: vec![],
                function_constants: vec![],
            }),
        ))
    }

    fn synth_shader_with_samplers(
        stage: Stage,
        spirv: Vec<u8>,
        metal_indices: &[u32],
    ) -> Arc<CachedShader> {
        use metal2vulkan::reflect::{
            DescriptorLayout, DescriptorLocation, ResourceBinding, ResourceKind, ShaderReflection,
            ShaderStage, REFLECTION_VERSION, RESOURCE_DESCRIPTOR_SET, SAMPLER_BINDING_BASE,
        };
        let reflected_stage = match stage {
            Stage::Vertex => ShaderStage::Vertex,
            Stage::Fragment => ShaderStage::Fragment,
            Stage::Kernel => ShaderStage::Kernel,
        };
        let bindings = metal_indices
            .iter()
            .copied()
            .map(|metal_index| ResourceBinding {
                kind: ResourceKind::Sampler,
                metal_index,
                descriptor: Some(DescriptorLocation {
                    set: RESOURCE_DESCRIPTOR_SET,
                    binding: SAMPLER_BINDING_BASE + metal_index,
                    count: 1,
                }),
                param_index: None,
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                footprint: None,
                type_layout: None,
                type_name: None,
                texture_shape: None,
                embedded_source: None,
                access: None,
                static_sampler: None,
            })
            .collect();
        Arc::new(CachedShader::new(
            spirv,
            Arc::new(ShaderReflection {
                reflection_version: REFLECTION_VERSION,
                stage: reflected_stage,
                entry_point: None,
                bindings,
                argument_buffer_fields: vec![],
                vertex_attributes: vec![],
                varyings: vec![],
                render_targets: vec![],
                depth_members: vec![],
                depth_qualifier: None,
                stencil_members: vec![],
                local_size: (reflected_stage == ShaderStage::Kernel).then_some([1, 1, 1]),
                max_work_group_size: None,
                vertex_builtins: None,
                tessellation: None,
                imageblock_layouts: vec![],
                implicit_imageblock_attachments: vec![],
                fragment_imageblock: None,
                datalayout: None,
                descriptor_layout: DescriptorLayout::default(),
                kernel_dispatch: None,
                runtime_sampler_specializations: vec![],
                runtime_storage_image_specializations: vec![],
                function_constants: vec![],
            }),
        ))
    }

    #[test]
    fn malformed_reflection_is_refused_before_it_reaches_a_binding_path() {
        let shader = synth_shader(Stage::Fragment, Vec::new());
        let mut reflection = (*shader.reflection).clone();
        reflection.reflection_version = reflection.reflection_version.wrapping_add(1);
        assert!(matches!(
            validate_reflection(&reflection, Stage::Fragment),
            Err(M2vCacheDecline::ReflectionMalformed {
                stage: "fragment",
                violations: 1,
            })
        ));
    }

    /// `CachedShader::new` widens the translator's bands, and `variant`
    /// relocates on top of that — in that order, for every representation.
    #[test]
    fn variant_words_match_direct_relocation_and_cache() {
        use crate::runtime::spirv_bind::{
            FRAG_BUFFER_BINDING_OFFSET, FRAG_SAMPLED_RESOURCE_BINDING_OFFSET,
            M2V_SAMPLER_BINDING_BASE, M2V_TEXTURE_BINDING_BASE, SAMPLED_TAIL_WIDEN_OFFSET,
        };

        // Minimal module: 5-word header + three OpDecorate Binding instructions
        // in the TRANSLATOR's numbering, which is what a fresh translate hands
        // over. One buffer, one texture, and the top of the translator's sampler
        // band — the maximum source the widen has to carry.
        const TOP_SAMPLER: u32 = M2V_SAMPLER_BINDING_BASE + 31;
        let decorate = |id: u32, binding: u32| vec![(4u32 << 16) | 71, id, 33, binding];
        let mut words: Vec<u32> = vec![0x0723_0203, 0x0001_0000, 0, 100, 0];
        words.extend(decorate(7, 3));
        words.extend(decorate(8, M2V_TEXTURE_BINDING_BASE + 8));
        words.extend(decorate(9, TOP_SAMPLER));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let shader = synth_shader(Stage::Fragment, bytes);

        // The translator's v22 layout is already wide, so the compatibility
        // pass leaves all three bindings unchanged.
        let mut widened = words.clone();
        let moved = crate::runtime::spirv_bind::widen_sampled_bands(&mut widened);
        assert_eq!(moved, 0, "the translator and device layouts agree");
        assert_eq!(*shader.words, widened);
        assert_eq!(shader.words[8], 3);
        assert_eq!(shader.words[12], M2V_TEXTURE_BINDING_BASE + 8);
        assert_eq!(shader.words[16], TOP_SAMPLER + SAMPLED_TAIL_WIDEN_OFFSET);
        // The bytes must not be allowed to disagree with the words.
        let from_bytes: Vec<u32> = shader
            .spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(from_bytes, widened);

        // No flags → the base Arc, un-relocated.
        let base = shader.variant(false, false).words.clone();
        assert!(Arc::ptr_eq(&base, &shader.words));

        // Both flags → sampled reloc first, then buffer band, matching the
        // historical per-draw mutation order.
        let mut expect = widened.clone();
        let n = crate::runtime::spirv_bind::offset_fragment_sampled_resource_bindings(&mut expect);
        assert_eq!(n, 2);
        let n = crate::runtime::spirv_bind::offset_fragment_buffer_bindings(&mut expect);
        assert_eq!(n, 1);
        let both = shader.variant(true, true).words.clone();
        assert_eq!(*both, expect);
        assert_eq!(both[8], 3 + FRAG_BUFFER_BINDING_OFFSET);
        assert_eq!(
            both[12],
            M2V_TEXTURE_BINDING_BASE + 8 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        assert_eq!(
            both[16],
            TOP_SAMPLER + SAMPLED_TAIL_WIDEN_OFFSET + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        // Every relocated band stays clear of every un-relocated one.
        assert!(both[8] > crate::runtime::spirv_bind::COLOR_INPUT_BINDING_BASE);
        assert!(both[12] > both[8] && both[16] > both[12]);

        // Second call returns the cached variant (same allocation), and the
        // stored (widened) module is never mutated by a relocation.
        let again = shader.variant(true, true).words.clone();
        assert!(Arc::ptr_eq(&both, &again));
        assert_eq!(*shader.words, widened);
    }

    #[test]
    fn air_key_differs_by_stage_and_bytes() {
        let a = b"same-air-bytes";
        assert_ne!(air_key(Stage::Vertex, a), air_key(Stage::Fragment, a));
        assert_ne!(air_key(Stage::Vertex, a), air_key(Stage::Vertex, b"other"));
        assert_eq!(air_key(Stage::Vertex, a), air_key(Stage::Vertex, a));
        assert_ne!(air_key_kernel(a, [16, 16, 1]), air_key_kernel(a, [8, 8, 1]));
    }

    /// Each `local_size` component alone changes the kernel key.
    ///
    /// The assertion above varies two components at once, so a digest that fed
    /// only `local_size[0]` would satisfy it. That matters here more than the
    /// usual thoroughness argument: the translator bakes the workgroup size into
    /// the SPIR-V, so a key blind to one axis returns a shader compiled for a
    /// different one — the guest's dispatch runs, produces wrong results, and
    /// nothing anywhere reports a miss, because it was a *hit*.
    ///
    /// Also pinned: the same triple keys the same, so this cannot pass by
    /// accident on a key that simply changes every call.
    #[test]
    fn every_local_size_axis_is_in_the_kernel_key() {
        let air = b"kernel-air";
        let base = [8u32, 8, 8];
        assert_eq!(air_key_kernel(air, base), air_key_kernel(air, base));
        for axis in 0..3 {
            let mut moved = base;
            moved[axis] += 1;
            assert_ne!(
                air_key_kernel(air, base),
                air_key_kernel(air, moved),
                "local_size[{axis}] does not reach the key: {base:?} vs {moved:?}"
            );
        }
    }

    /// Every decline is classified, and the split is by what the variant names.
    ///
    /// The scratch writes are the host filesystem at one instant; everything
    /// else is a property of the AIR blob this cache is keyed by, and is reached
    /// again by every later ask.
    #[test]
    fn only_the_scratch_writes_are_transient() {
        let detail = || "ENOSPC".to_string();
        for transient in [
            M2vCacheDecline::VertexScratchWrite { detail: detail() },
            M2vCacheDecline::FragmentScratchWrite { detail: detail() },
            M2vCacheDecline::KernelScratchWrite { detail: detail() },
        ] {
            assert!(transient.is_transient(), "{transient:?} is about the host");
        }
        for permanent in [
            M2vCacheDecline::VertexTranslate { detail: detail() },
            M2vCacheDecline::FragmentTranslate { detail: detail() },
            M2vCacheDecline::KernelTranslate { detail: detail() },
            M2vCacheDecline::ReflectionDatalayoutMissing { stage: "vertex" },
            M2vCacheDecline::ReflectionMalformed {
                stage: "vertex",
                violations: 1,
            },
            M2vCacheDecline::TranslationPending { stage: "vertex" },
            M2vCacheDecline::KernelLocalSizeZero {
                local_size: [0, 1, 1],
            },
        ] {
            assert!(!permanent.is_transient(), "{permanent:?} is about the AIR");
        }
    }

    /// A scratch-write failure fails the draw that met it and then gets out of
    /// the way. The cache is unbounded and nothing evicts, so before this the
    /// entry answered every later draw of that shader for the life of the
    /// process — a full stop for one moment when the host had no room for a
    /// temp file.
    #[test]
    fn a_transient_failure_is_forgotten_once_it_has_been_answered() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"air-whose-scratch-write-failed";
        let id = ShaderId::render(Stage::Vertex, air);
        let stored = M2vCacheDecline::VertexScratchWrite {
            detail: "No space left on device".to_string(),
        };
        global()
            .lock()
            .unwrap()
            .put(id, &Arc::from(&air[..]), Entry::Failed(stored.clone()));

        // The admission still resolves, so the guest packet advances rather
        // than re-polling a translation that keeps failing.
        assert!(
            ensure_cached_async(air, Stage::Vertex, 3, 7),
            "a stored failure must resolve the ask, or the packet never advances"
        );

        // This draw gets the real reason, unchanged.
        let Err(err) = translate_cached_reflected(air, Stage::Vertex, 7) else {
            panic!("a stored failure still fails its draw");
        };
        assert_eq!(err, stored);

        // And the next draw finds a clean cache and translates again.
        assert!(
            global().lock().unwrap().find(id).is_none(),
            "the transient failure outlived the instant it described"
        );
        reset_for_test();
    }

    /// A draw that reaches a shader the async worker has queued but not finished
    /// translates it itself rather than being refused with
    /// `m2v_translation_pending_at_sync_boundary`. The refusal made the device
    /// store the clear colour — opaque black corner tiles and widget layers on
    /// macOS 26 whenever a pipeline ref was re-pointed at new AIR.
    #[test]
    fn a_draw_reaching_a_queued_translation_translates_instead_of_refusing() {
        let _guard = test_lock();
        reset_for_test();
        // Not real AIR, so the inline translation fails — and that failure is
        // the proof it ran: a refusal would have been `TranslationPending`.
        let air = b"air-queued-for-the-worker";
        let id = ShaderId::render(Stage::Fragment, air);
        global()
            .lock()
            .unwrap()
            .put(id, &Arc::from(&air[..]), Entry::Loading);

        let Err(err) = translate_cached_reflected(air, Stage::Fragment, 9) else {
            panic!("garbage AIR cannot translate");
        };
        assert!(
            !matches!(err, M2vCacheDecline::TranslationPending { .. }),
            "a queued translation refused the draw: {err:?}"
        );
        let id = ShaderId::kernel(air, [8, 8, 1]);
        global()
            .lock()
            .unwrap()
            .put(id, &Arc::from(&air[..]), Entry::Loading);
        let Err(err) = translate_cached_kernel_reflected(air, [8, 8, 1], 9) else {
            panic!("garbage AIR cannot translate");
        };
        assert!(
            !matches!(err, M2vCacheDecline::TranslationPending { .. }),
            "a queued kernel translation refused the dispatch: {err:?}"
        );
        reset_for_test();
    }

    /// Two shaders that land in one bucket each resolve to their own entry.
    ///
    /// The bucket is forced rather than found: this cache keys on a 64-bit
    /// digest and nobody has a pair of AIR blobs that collide under it, so the
    /// only way to drive the collision path is to file both under one digest and
    /// then ask through the real lookup. That is exactly the state a natural
    /// collision would produce.
    ///
    /// Before the AIR was retained, `find` returned on digest equality alone, so
    /// the second ask here would have been answered with the first shader's
    /// SPIR-V — a shader the guest never compiled, handed over silently and with
    /// no failure line. The probability was small; the failure was
    /// unobservable, which is the part no birthday bound prices.
    #[test]
    fn two_shaders_in_one_bucket_each_resolve_to_their_own() {
        let _guard = test_lock();
        reset_for_test();

        let first = b"the-air-that-got-there-first";
        let second = b"a-different-shader-entirely";
        let id = ShaderId::render(Stage::Vertex, first);
        // The second identity, forced into the first one's bucket. Everything
        // but `digest` is the second shader's own.
        let collided = ShaderId {
            digest: id.digest,
            ..ShaderId::render(Stage::Vertex, second)
        };
        assert_ne!(
            ShaderId::render(Stage::Vertex, second).digest,
            id.digest,
            "the two blobs do not collide naturally; the bucket below is forced"
        );

        {
            let mut c = global().lock().unwrap();
            c.put(
                id,
                &Arc::from(&first[..]),
                Entry::Ready(synth_shader(Stage::Vertex, vec![1, 1, 1, 1])),
            );
            c.put(
                collided,
                &Arc::from(&second[..]),
                Entry::Ready(synth_shader(Stage::Vertex, vec![2, 2, 2, 2])),
            );
            assert_eq!(
                c.entries.get(&id.digest).map(Vec::len),
                Some(2),
                "both are filed under one digest, which is the whole setup"
            );
            assert_eq!(c.len(), 2, "and the level counts entries, not buckets");
        }

        let got = |i| match global().lock().unwrap().find(i) {
            Some(Entry::Ready(s)) => s.spirv.clone(),
            other => panic!("expected a ready entry, got {}", other.is_some()),
        };
        assert_eq!(got(id), vec![1, 1, 1, 1], "the first blob got its own");
        assert_eq!(
            got(collided),
            vec![2, 2, 2, 2],
            "the second blob got its own, not the one sharing its digest"
        );

        // And forgetting one leaves the other where it is, rather than taking
        // the whole bucket with it.
        global().lock().unwrap().forget(id);
        assert!(global().lock().unwrap().find(id).is_none());
        assert_eq!(got(collided), vec![2, 2, 2, 2]);
        reset_for_test();
    }

    /// The converse, so the test above cannot pass by forgetting everything. A
    /// module that cannot be translated is reached again by every later ask, and
    /// re-running metal2vulkan per draw to rediscover that would cost far more
    /// than the answer is worth.
    #[test]
    fn a_failure_about_the_module_is_kept() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"air-that-cannot-be-translated";
        let id = ShaderId::render(Stage::Vertex, air);
        let stored = M2vCacheDecline::VertexTranslate {
            detail: "unsupported instruction".to_string(),
        };
        global()
            .lock()
            .unwrap()
            .put(id, &Arc::from(&air[..]), Entry::Failed(stored.clone()));

        let Err(err) = translate_cached_reflected(air, Stage::Vertex, 7) else {
            panic!("an untranslatable module still fails its draw");
        };
        assert_eq!(err, stored);
        assert!(
            matches!(
                global().lock().unwrap().find(id),
                Some(Entry::Failed(kept)) if *kept == stored
            ),
            "a verdict about the AIR is reached again, so it is kept"
        );
        reset_for_test();
    }

    /// The kernel lookup is a second copy of the same `Entry::Failed` arm, so it
    /// gets the same question asked of it.
    #[test]
    fn the_kernel_lookup_forgets_a_transient_failure_too() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"kernel-air-whose-scratch-write-failed";
        let local_size = [16u32, 16, 1];
        let id = ShaderId::kernel(air, local_size);
        let stored = M2vCacheDecline::KernelScratchWrite {
            detail: "No space left on device".to_string(),
        };
        global()
            .lock()
            .unwrap()
            .put(id, &Arc::from(&air[..]), Entry::Failed(stored.clone()));

        let Err(err) = translate_cached_kernel_reflected(air, local_size, 7) else {
            panic!("a stored failure still fails its dispatch");
        };
        assert_eq!(err, stored);
        assert!(global().lock().unwrap().find(id).is_none());
        reset_for_test();
    }

    #[test]
    fn cache_hit_skips_second_lookup_path() {
        let _guard = test_lock();
        reset_for_test();
        // Inject without metal2vulkan: put entry then get_or via public API by
        // priming the map through the same key logic.
        let air = b"synthetic-air-for-cache-unit";
        {
            let mut c = global().lock().unwrap();
            // fake SPIR-V magic-ish
            c.put(
                ShaderId::render(Stage::Vertex, air),
                &Arc::from(&air[..]),
                Entry::Ready(synth_shader(Stage::Vertex, vec![0x03, 0x02, 0x23, 0x07])),
            );
        }
        // One hit total, carrying the bytes plus the (empty) reflection.
        let shader = translate_cached_reflected(air, Stage::Vertex, 99).expect("hit");
        assert_eq!(shader.spirv, vec![0x03, 0x02, 0x23, 0x07]);
        assert!(shader.reflection.vertex_builtins.is_none());
        let (hits, misses, n) = stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 0);
        assert_eq!(n, 1);
        reset_for_test();
    }

    #[test]
    fn spirv_builtin_scan_finds_instance_index() {
        // Minimal module: 5-word header, then OpDecorate(71) %id BuiltIn(11) InstanceIndex(43).
        // word0 = (wordCount<<16)|opcode = (4<<16)|71.
        let words: [u32; 9] = [0x07230203, 0x00010000, 0, 1, 0, (4 << 16) | 71, 5, 11, 43];
        let mut spv = Vec::new();
        for w in words {
            spv.extend_from_slice(&w.to_le_bytes());
        }
        assert!(spirv_uses_builtin(&spv, 43)); // InstanceIndex present
        assert!(!spirv_uses_builtin(&spv, 42)); // VertexIndex absent
        assert!(!spirv_uses_builtin(&[], 43)); // empty is safe
    }

    #[test]
    fn async_cache_state_distinguishes_pending_from_resolved_failure() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"synthetic-async-cache-state";
        let id = ShaderId::render(Stage::Fragment, air);
        {
            let mut c = global().lock().unwrap();
            c.put(id, &Arc::from(&air[..]), Entry::Loading);
        }
        assert!(!ensure_cached_async(air, Stage::Fragment, 3, 7));
        assert!(global().lock().unwrap().async_queue.is_empty());

        global().lock().unwrap().put(
            id,
            &Arc::from(&air[..]),
            Entry::Failed(M2vCacheDecline::FragmentTranslate {
                detail: "synthetic failure".into(),
            }),
        );
        assert!(ensure_cached_async(air, Stage::Fragment, 3, 7));
        assert_eq!(
            // `.err()` rather than `unwrap_err()`: the success arm is an
            // `Arc<CachedShader>`, which carries no `Debug`.
            translate_cached_reflected(air, Stage::Fragment, 7)
                .err()
                .expect("a Failed entry translates to its decline"),
            M2vCacheDecline::FragmentTranslate {
                detail: "synthetic failure".into()
            }
        );
        reset_for_test();
    }

    /// The shader a boot compiles first is the compositor's, and it is drawn
    /// every frame for the life of the boot. Under the entry cap this cache used
    /// to carry it was also the first eviction victim, because the order was
    /// insertion rather than use — so crossing the bound threw away exactly the
    /// entry that was still hot. Drive far past the old 256-entry bound and
    /// assert the first key is still resolved.
    #[test]
    fn a_hot_first_shader_survives_far_past_the_old_entry_cap() {
        let _guard = test_lock();
        reset_for_test();
        let hot = b"the-compositor-shader-compiled-first".to_vec();
        {
            let mut c = global().lock().unwrap();
            c.put(
                ShaderId::render(Stage::Fragment, &hot),
                &Arc::from(&hot[..]),
                Entry::Ready(synth_shader(Stage::Fragment, vec![1, 2, 3, 4])),
            );
        }
        // Four times the retired bound, so a cap of any nearby size would have
        // reached the first key several times over.
        for i in 0..1024u32 {
            let air = format!("cold-shader-{i}").into_bytes();
            let mut c = global().lock().unwrap();
            c.put(
                ShaderId::render(Stage::Fragment, &air),
                &Arc::from(&air[..]),
                Entry::Ready(synth_shader(Stage::Fragment, vec![0, 0, 0, 0])),
            );
        }
        assert!(
            ensure_cached_async(&hot, Stage::Fragment, 3, 1),
            "the first-compiled shader is still resolved after 1024 later ones"
        );
        assert_eq!(
            global().lock().unwrap().len(),
            1025,
            "every distinct shader is retained; nothing is evicted for count"
        );
        reset_for_test();
    }

    #[test]
    fn async_kernel_cache_uses_local_size_key() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"synthetic-kernel-async-state";
        {
            let mut c = global().lock().unwrap();
            c.put(
                ShaderId::kernel(air, [16, 16, 1]),
                &Arc::from(&air[..]),
                Entry::Loading,
            );
        }
        assert!(!ensure_cached_kernel_async(air, [16, 16, 1], 3, 20));
        assert!(global().lock().unwrap().async_queue.is_empty());
        assert!(ensure_cached_kernel_async(air, [0, 16, 1], 3, 20));
        reset_for_test();
    }

    #[test]
    fn every_cache_decline_has_a_unique_log_safe_reason_and_fields() {
        let all = [
            M2vCacheDecline::VertexScratchWrite {
                detail: "permission denied".into(),
            },
            M2vCacheDecline::FragmentScratchWrite {
                detail: "permission denied".into(),
            },
            M2vCacheDecline::KernelScratchWrite {
                detail: "permission denied".into(),
            },
            M2vCacheDecline::VertexTranslate {
                detail: "tool failed".into(),
            },
            M2vCacheDecline::FragmentTranslate {
                detail: "tool failed".into(),
            },
            M2vCacheDecline::KernelTranslate {
                detail: "tool failed".into(),
            },
            M2vCacheDecline::ReflectionDatalayoutMissing { stage: "fragment" },
            M2vCacheDecline::ReflectionMalformed {
                stage: "fragment",
                violations: 1,
            },
            M2vCacheDecline::LayoutRepair {
                stage: "kernel",
                reason:
                    crate::runtime::spirv_layout::SpirvLayoutDecline::DataLayoutVectorAlignmentMissing,
            },
            M2vCacheDecline::TranslationPending { stage: "vertex" },
            M2vCacheDecline::KernelLocalSizeZero {
                local_size: [0, 16, 1],
            },
            M2vCacheDecline::KernelLocalSizeMismatch {
                requested: [16, 16, 1],
                reflected: Some([8, 8, 1]),
            },
        ];
        let mut slugs = Vec::new();
        for decline in all {
            assert!(decline
                .slug()
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
            for (key, value) in decline.fields() {
                assert!(!key.contains(char::is_whitespace));
                assert!(!value.contains(char::is_whitespace), "{key}={value}");
            }
            slugs.push(decline.slug());
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, 12, "the m2v cache decline census moved");
        assert_eq!(before, slugs.len(), "duplicate m2v cache decline");
    }

    // ---- the on-disk translation cache (`mod disk`) ---------------------------
    //
    // Tested at the cache's own boundary: a `DiskCache` over a scratch directory,
    // with the translator replaced by a closure, because the property under test
    // is what a second boot is handed and not what the translator produces. The
    // process-global `disk::global()` is latched from the environment and is not
    // touched here.

    /// A scratch directory removed on drop. Not created up front: the first
    /// `store` has to make it, and that is part of what is under test.
    struct DiskDir(std::path::PathBuf);

    impl DiskDir {
        fn new(tag: &str) -> Self {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "reims-vgpu-m2v-disk-test-{}-{n}-{tag}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for DiskDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn stamp() -> disk::Stamp {
        disk::Stamp {
            reflection_version: metal2vulkan::reflect::REFLECTION_VERSION,
            translator: 0x0123_4567_89ab_cdef,
            toolchain: 0x0fed_cba9_8765_4321,
            env_hash: 0x1234,
        }
    }

    /// A reflection with enough in it that a lossy round trip would show: a
    /// sampler binding carrying a static sampler whose floats are not integers,
    /// a datalayout, an entry point, and for a kernel its LocalSize.
    fn sample_reflection(kernel: bool) -> ShaderReflection {
        use metal2vulkan::reflect::{
            DescriptorLayout, DescriptorLocation, ResourceBinding, ResourceKind,
            SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerCoordinates,
            SamplerFilter, SamplerMipFilter, SamplerReduction, ShaderStage, StaticSamplerState,
            REFLECTION_VERSION, RESOURCE_DESCRIPTOR_SET, SAMPLER_BINDING_BASE,
        };
        let metal_index = 3;
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            stage: if kernel {
                ShaderStage::Kernel
            } else {
                ShaderStage::Fragment
            },
            entry_point: Some("shader_main".to_string()),
            bindings: vec![ResourceBinding {
                // The kind that owns a static sampler state: reflection
                // validation refuses the state on any other.
                kind: ResourceKind::StaticSampler,
                metal_index,
                descriptor: Some(DescriptorLocation {
                    set: RESOURCE_DESCRIPTOR_SET,
                    binding: SAMPLER_BINDING_BASE + metal_index,
                    count: 1,
                }),
                param_index: None,
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                footprint: None,
                type_layout: None,
                type_name: Some("sampler".to_string()),
                texture_shape: None,
                embedded_source: None,
                access: None,
                static_sampler: Some(StaticSamplerState {
                    min_filter: SamplerFilter::Linear,
                    mag_filter: SamplerFilter::Nearest,
                    mip_filter: SamplerMipFilter::Linear,
                    address_mode_s: SamplerAddressMode::Repeat,
                    address_mode_t: SamplerAddressMode::ClampToEdge,
                    address_mode_r: SamplerAddressMode::MirroredRepeat,
                    coordinates: SamplerCoordinates::Normalized,
                    compare_function: SamplerCompareFunction::None,
                    max_anisotropy: 4,
                    lod_min_clamp: 0.1,
                    lod_max_clamp: 65504.0,
                    border_color: SamplerBorderColor::OpaqueBlack,
                    reduction: SamplerReduction::WeightedAverage,
                    lod_bias: -0.333_333_3,
                    raw_words: [u64::MAX, 0x0123_4567_89ab_cdef],
                }),
            }],
            argument_buffer_fields: vec![],
            vertex_attributes: vec![],
            varyings: vec![],
            render_targets: vec![],
            depth_members: vec![],
            depth_qualifier: None,
            stencil_members: vec![],
            local_size: kernel.then_some([16, 16, 1]),
            max_work_group_size: None,
            vertex_builtins: None,
            tessellation: None,
            imageblock_layouts: vec![],
            implicit_imageblock_attachments: vec![],
            fragment_imageblock: None,
            datalayout: Some(
                "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64\
                 -v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128\
                 -v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
                    .to_string(),
            ),
            descriptor_layout: DescriptorLayout::default(),
            kernel_dispatch: None,
            runtime_sampler_specializations: vec![],
            runtime_storage_image_specializations: vec![],
            function_constants: vec![],
        }
    }

    /// A SPIR-V module the post-processing accepts, so `finish_translated` can
    /// run on what the disk hands back.
    fn sample_spirv() -> Vec<u8> {
        let base = metal2vulkan::reflect::SAMPLER_BINDING_BASE;
        crate::runtime::spirv_bind::test_module_with_samplers(&[base + 3])
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    fn entry_bytes(id: ShaderId<'_>, kernel: bool) -> Vec<u8> {
        let json = serde_json::to_vec(&sample_reflection(kernel)).unwrap();
        disk::encode(&stamp(), id, &sample_spirv(), &json)
    }

    /// The point of the cache: a second boot is handed the first boot's
    /// translator output, byte for byte, without running the translator, and the
    /// post-processing that follows cannot tell it from a fresh translation.
    ///
    /// The second boot is a new `DiskCache` over the same directory: nothing is
    /// shared with the first but the files, which is what a new process has.
    #[test]
    fn a_translation_stored_by_one_boot_is_handed_to_the_next_unchanged() {
        let _guard = test_lock();
        reset_for_test();
        let dir = DiskDir::new("roundtrip");
        let air = b"air-of-a-fragment-shader-worth-keeping";
        let id = ShaderId::render(Stage::Fragment, air);
        let translations = std::cell::Cell::new(0u32);
        let translate = || -> M2vResult<disk::Raw> {
            translations.set(translations.get() + 1);
            Ok((sample_spirv(), sample_reflection(false)))
        };

        let first = disk::DiskCache::new(dir.0.clone(), stamp());
        let fresh = disk::obtain(Some(&first), id, translate).unwrap();
        assert_eq!(translations.get(), 1, "an empty directory translates");
        assert_eq!(
            first.census(),
            disk::Census {
                hits: 0,
                misses: 1,
                stale: 0,
                write_failures: 0
            }
        );
        assert_eq!(
            first.path(id).file_name().unwrap().to_str().unwrap(),
            format!("fragment-{:016x}.m2v", air_key(Stage::Fragment, air)),
            "one file per entry, named by stage and air_key"
        );
        assert!(first.path(id).is_file(), "the translation reached the disk");

        let second = disk::DiskCache::new(dir.0.clone(), stamp());
        let loaded = disk::obtain(Some(&second), id, translate).unwrap();
        assert_eq!(translations.get(), 1, "a hit must not run the translator");
        assert_eq!(second.census().hits, 1);
        assert_eq!(second.census().misses, 0);
        assert_eq!(loaded.0, fresh.0, "identical SPIR-V bytes");
        assert_eq!(loaded.1, fresh.1, "identical reflection");

        // Same input to the post-processing, same output from it.
        let from_fresh = finish_translated(fresh.0, fresh.1, Stage::Fragment).unwrap();
        let from_disk = finish_translated(loaded.0, loaded.1, Stage::Fragment).unwrap();
        assert_eq!(from_disk.spirv, from_fresh.spirv);
        assert_eq!(*from_disk.words, *from_fresh.words);
        assert_eq!(*from_disk.reflection, *from_fresh.reflection);
        reset_for_test();
    }

    /// A kernel's LocalSize is part of its identity, on disk as in memory: one
    /// AIR blob at two geometries is two entries, and a stored one is not an
    /// answer for the other even if the digests were to meet.
    #[test]
    fn a_kernel_entry_answers_only_the_local_size_it_was_translated_at() {
        let dir = DiskDir::new("kernel");
        let air = b"air-of-a-kernel";
        let cache = disk::DiskCache::new(dir.0.clone(), stamp());
        let id = ShaderId::kernel(air, [16, 16, 1]);
        cache.store(id, &sample_spirv(), &sample_reflection(true));
        assert!(cache.load(id).is_some());

        let other = ShaderId::kernel(air, [8, 8, 1]);
        assert_ne!(
            other.digest, id.digest,
            "distinct geometries, distinct files"
        );
        assert!(cache.load(other).is_none());

        // Forced into the same file: the entry's own LocalSize decides.
        let bytes = entry_bytes(id, true);
        let forged = ShaderId {
            local_size: Some([8, 8, 1]),
            ..id
        };
        assert_eq!(
            disk::decode(&stamp(), &bytes, forged).err(),
            Some(disk::Stale::LocalSize)
        );
        let not_a_kernel = ShaderId {
            local_size: None,
            ..id
        };
        assert_eq!(
            disk::decode(&stamp(), &bytes, not_a_kernel).err(),
            Some(disk::Stale::LocalSize)
        );
    }

    /// The digest names the file and decides nothing: other AIR under the same
    /// key is a miss, and so is the same bytes asked for as another stage.
    #[test]
    fn other_air_under_the_same_key_is_a_miss() {
        let dir = DiskDir::new("air");
        let cache = disk::DiskCache::new(dir.0.clone(), stamp());
        let id = ShaderId::render(Stage::Fragment, b"the-air-that-was-translated");
        cache.store(id, &sample_spirv(), &sample_reflection(false));
        assert!(cache.load(id).is_some());

        // The same digest — so the same file — over different bytes. One byte
        // differing, a longer blob that begins with the stored one, a shorter
        // prefix of it, and nothing at all.
        let colliding_air: [&[u8]; 4] = [
            b"the-air-that-was-translateD",
            b"the-air-that-was-translated-and-then-some",
            b"the-air",
            b"",
        ];
        for air in colliding_air {
            let colliding = ShaderId { air, ..id };
            assert!(cache.load(colliding).is_none(), "{air:?}");
        }
        assert_eq!(
            cache.census().stale,
            4,
            "all four found the file and refused it"
        );

        // Another stage under that digest names another file, and the entry
        // itself also says what stage it is.
        let as_vertex = ShaderId { stage: 1, ..id };
        assert!(cache.load(as_vertex).is_none());
        let bytes = entry_bytes(id, false);
        assert_eq!(
            disk::decode(&stamp(), &bytes, as_vertex).err(),
            Some(disk::Stale::Stage)
        );
        assert_eq!(
            disk::decode(&stamp(), &bytes, ShaderId { air: b"x", ..id }).err(),
            Some(disk::Stale::Air)
        );
    }

    /// Any difference in what produced an entry makes it a miss, and the miss is
    /// repaired in place: the rebuilt binary's translation replaces the file, so
    /// the directory keeps one entry per shader rather than one per build.
    #[test]
    fn an_entry_from_another_build_is_a_miss_and_is_replaced() {
        let dir = DiskDir::new("stamp");
        let id = ShaderId::render(Stage::Vertex, b"air-for-the-stamp-test");
        let old = disk::DiskCache::new(dir.0.clone(), stamp());
        old.store(id, &sample_spirv(), &sample_reflection(false));
        assert!(old.load(id).is_some());

        let others = [
            disk::Stamp {
                translator: stamp().translator ^ 1,
                ..stamp()
            },
            disk::Stamp {
                toolchain: stamp().toolchain ^ 1,
                ..stamp()
            },
            disk::Stamp {
                reflection_version: stamp().reflection_version + 1,
                ..stamp()
            },
            disk::Stamp {
                env_hash: stamp().env_hash ^ 1,
                ..stamp()
            },
        ];
        for other in others {
            let rebuilt = disk::DiskCache::new(dir.0.clone(), other);
            assert!(rebuilt.load(id).is_none(), "{other:?}");
            assert_eq!(rebuilt.census().stale, 1, "{other:?}");
            assert_eq!(rebuilt.census().hits, 0, "{other:?}");
        }
        assert!(
            old.load(id).is_some(),
            "refusing an entry must not damage it"
        );

        // The newer build translates, writes, and from then on owns the file.
        let newer = disk::DiskCache::new(dir.0.clone(), others[0]);
        newer.store(id, &sample_spirv(), &sample_reflection(false));
        assert!(newer.load(id).is_some());
        assert!(old.load(id).is_none(), "the file is the newer build's now");
        let files = std::fs::read_dir(&dir.0).unwrap().count();
        assert_eq!(
            files, 1,
            "one file per shader, and no temporaries left behind"
        );
    }

    /// The build's stamp exists and is stable, which is what lets a second
    /// process find the first one's entries at all.
    #[test]
    fn the_build_has_a_stable_stamp() {
        let a = disk::Stamp::current().expect("the build script stamps every build");
        let b = disk::Stamp::current().unwrap();
        assert_eq!(a, b);
        assert_eq!(
            a.reflection_version,
            metal2vulkan::reflect::REFLECTION_VERSION
        );
        assert!(a.translator != 0 && a.toolchain != 0);
    }

    /// A torn, truncated, foreign or bit-rotted file is a miss. It is also never
    /// a panic and never a hit — the SPIR-V bytes are the one part nothing else
    /// compares, so every single-byte change must be caught by *something*.
    #[test]
    fn a_corrupt_or_truncated_entry_is_a_miss_not_a_panic() {
        let air = b"air-for-the-corruption-test";
        for (id, kernel) in [
            (ShaderId::render(Stage::Fragment, air), false),
            (ShaderId::kernel(air, [16, 16, 1]), true),
        ] {
            let good = entry_bytes(id, kernel);
            assert!(
                disk::decode(&stamp(), &good, id).is_ok(),
                "the fixture is a valid entry"
            );

            for n in 0..good.len() {
                assert!(
                    disk::decode(&stamp(), &good[..n], id).is_err(),
                    "a {n}-byte prefix of a {}-byte entry was accepted",
                    good.len()
                );
            }
            let mut padded = good.clone();
            padded.push(0);
            assert_eq!(
                disk::decode(&stamp(), &padded, id).err(),
                Some(disk::Stale::Length),
                "bytes beyond the checksum"
            );
            for at in 0..good.len() {
                let mut flipped = good.clone();
                flipped[at] ^= 0xff;
                assert!(
                    disk::decode(&stamp(), &flipped, id).is_err(),
                    "flipping byte {at} of {} went unnoticed",
                    good.len()
                );
            }
        }

        // Lengths that claim more than the file has, or more than a usize.
        let id = ShaderId::render(Stage::Fragment, air);
        let good = entry_bytes(id, false);
        // Header: magic 8, schema 4, stamp 28, stage 1, flag 1, local 12.
        let lengths_at = 8 + 4 + 28 + 1 + 1 + 12;
        for lengths in [[u64::MAX; 3], [u64::MAX, 1, 1], [1 << 62, 1 << 62, 1 << 62]] {
            let mut huge = good.clone();
            for (i, len) in lengths.iter().enumerate() {
                let at = lengths_at + 8 * i;
                huge[at..at + 8].copy_from_slice(&len.to_le_bytes());
            }
            assert_eq!(
                disk::decode(&stamp(), &huge, id).err(),
                Some(disk::Stale::Length)
            );
        }

        // A frame that is intact end to end around a reflection that is not
        // JSON the type reads: the checksum cannot catch it, the parse does.
        for json in [&b"not json"[..], &b"{}"[..], &b""[..], &b"null"[..]] {
            let framed = disk::encode(&stamp(), id, &sample_spirv(), json);
            assert_eq!(
                disk::decode(&stamp(), &framed, id).err(),
                Some(disk::Stale::Reflection),
                "{json:?}"
            );
        }
    }

    /// The same, through the file path: an entry on disk that is garbage, empty,
    /// or cut short is a counted miss, and a fresh translation then repairs it.
    #[test]
    fn a_damaged_file_is_repaired_by_the_next_translation() {
        let dir = DiskDir::new("damaged");
        let id = ShaderId::render(Stage::Fragment, b"air-for-the-damaged-file-test");
        let cache = disk::DiskCache::new(dir.0.clone(), stamp());
        std::fs::create_dir_all(&dir.0).unwrap();
        let good = entry_bytes(id, false);
        for damaged in [
            Vec::new(),
            b"not an entry at all".to_vec(),
            good[..good.len() / 2].to_vec(),
        ] {
            std::fs::write(cache.path(id), &damaged).unwrap();
            let translations = std::cell::Cell::new(0u32);
            let raw = disk::obtain(Some(&cache), id, || {
                translations.set(translations.get() + 1);
                Ok((sample_spirv(), sample_reflection(false)))
            })
            .unwrap();
            assert_eq!(
                translations.get(),
                1,
                "a damaged entry falls back to the translator"
            );
            assert_eq!(raw.1, sample_reflection(false));
            assert!(
                cache.load(id).is_some(),
                "and the write that follows repairs it"
            );
        }
        assert_eq!(cache.census().stale, 3);
        assert_eq!(cache.census().write_failures, 0);
    }

    /// A disk that will not take the write does not fail the translation: the
    /// translator's result still comes back, and the failure is counted.
    #[test]
    fn a_disk_that_cannot_be_written_does_not_fail_the_translation() {
        let dir = DiskDir::new("unwritable");
        // The cache directory is a file, so neither reads nor mkdir can work.
        std::fs::write(&dir.0, b"in the way").unwrap();
        let cache = disk::DiskCache::new(dir.0.clone(), stamp());
        let id = ShaderId::render(Stage::Fragment, b"air-for-the-unwritable-test");
        let raw = disk::obtain(Some(&cache), id, || {
            Ok((sample_spirv(), sample_reflection(false)))
        })
        .expect("the translation succeeded; the disk is not its business");
        assert_eq!(raw.0, sample_spirv());
        let census = cache.census();
        assert_eq!(census.write_failures, 1);
        assert_eq!(census.hits, 0);

        // And a translator failure is still the translator's: nothing is stored.
        let failing = ShaderId::render(Stage::Fragment, b"air-that-does-not-translate");
        let err = disk::obtain(Some(&cache), failing, || {
            Err(M2vCacheDecline::FragmentTranslate {
                detail: "no".into(),
            })
        });
        assert_eq!(
            err.err(),
            Some(M2vCacheDecline::FragmentTranslate {
                detail: "no".into()
            })
        );
        assert_eq!(cache.census().write_failures, 1);
    }

    /// A reflection JSON cannot carry stays out of the directory: `serde_json`
    /// writes `null` for a non-finite float and cannot read it back, so storing
    /// it would make a stale file of it on every boot.
    #[test]
    fn a_reflection_that_does_not_survive_json_is_not_stored() {
        let dir = DiskDir::new("nonfinite");
        let cache = disk::DiskCache::new(dir.0.clone(), stamp());
        let id = ShaderId::render(Stage::Fragment, b"air-with-an-infinite-lod");
        let mut reflection = sample_reflection(false);
        if let Some(sampler) = reflection.bindings[0].static_sampler.as_mut() {
            sampler.lod_max_clamp = f32::INFINITY;
        }
        cache.store(id, &sample_spirv(), &reflection);
        assert_eq!(cache.census().write_failures, 1);
        assert!(!cache.path(id).exists(), "nothing was written");
    }

    /// With the switch unset the cache is not there: the translator runs and
    /// nothing else happens.
    #[test]
    fn with_the_switch_unset_the_translator_runs_and_nothing_is_stored() {
        if std::env::var_os(disk::ENV).is_none_or(|v| v.is_empty()) {
            assert!(disk::global().is_none());
        }
        let id = ShaderId::render(Stage::Fragment, b"air-with-no-disk");
        let raw =
            disk::obtain(None, id, || Ok((sample_spirv(), sample_reflection(false)))).unwrap();
        assert_eq!(raw.1, sample_reflection(false));
    }
}

/// TEMP (lab): the m2v disk cache's directory when it is on, for sibling records
/// keyed on module bytes (`spirv_bind::validate_memo`).
pub(crate) fn disk_dir() -> Option<&'static std::path::Path> {
    disk::global().map(|d| d.dir.as_path())
}
