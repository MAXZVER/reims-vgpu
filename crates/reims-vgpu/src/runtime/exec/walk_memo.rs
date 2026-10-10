//! TEMP (lab, `REIMS_VGPU_LAB_WALK_MEMO`, default on): what one stream's walk
//! has already resolved, so a bind does not ask the object namespace for it
//! again.
//!
//! # The claim, and why it holds
//!
//! A bind record resolves its object through [`crate::runtime::objects`]:
//! the namespace's name lookup, the retained-resource registry, and for a
//! texture the mapper and backing probes. A stream binds the same few objects
//! over and over — a driven 5K desktop made 40 000 such memo-hit resolutions a
//! second for 800 streams — and each repeat costs two mutexes and a tree walk to
//! reach the answer it gave a few microseconds earlier.
//!
//! Within **one** `walk_stream` nothing can change that answer. A retained
//! resource leaves the registry only through `DeviceState::delete_object` and
//! `delete_task`, which the drain calls from the packet handlers of the delete
//! commands and from nowhere in the record walk; a resolution that succeeded
//! once is therefore the resolution every later bind of the same reference in the
//! same stream would get. The memo lives in the stream's accumulator, so it is
//! born and dropped with the stream and cannot outlive a delete between two
//! streams.
//!
//! **Only successes are kept for a resource.** A failed construction is retried
//! on the next bind, as the registry's own contract says ("a descriptor that is
//! still being published can succeed on retry"), so a failure takes the original
//! path every time and counts what it always counted.
//!
//! **The backing probe's answer is kept either way**, and the reason is its
//! cost, not its contract. For a texture that is not a mapper-ref texture the
//! probe walks every live task's object list through the guest's page tables to
//! learn that no task lists a surface at that slot, and it learns it again at
//! every bind of the texture. The answer is a statement about the guest's object
//! lists at the moment of the probe; a stream's records are walked within
//! microseconds of each other and the guest cannot order a write of its own
//! between two of them, so what the first bind saw is what a second would. A
//! publish that lands in that window reaches the next stream's probe instead,
//! exactly as one landing a microsecond after any single probe always did.
//!
//! **The counters stay the same.** A registry hit counts
//! `task_resource_reexamine_deferred` (its interval had not elapsed); the memo
//! counts the same route for the hit it answers, once per skipped resolution,
//! and posts them at the end of the stream. The interval is 100 ms and a stream
//! is microseconds, so the second resolution of a reference in a stream is
//! always one the interval defers.
//!
//! `=off` restores the per-bind resolution, and the unchanged table writes of
//! `apply_binds`, exactly as they were.

use crate::model::TaskResource;
use std::sync::Arc;

/// How the memo is switched: `off`, on (the default), or `alt` — on for the
/// even four-second blocks of the device clock and off for the odd ones, decided
/// once per packet, so one boot measures both arms against the same host load.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    On,
    Alt,
}

fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("REIMS_VGPU_LAB_WALK_MEMO").as_deref() {
            Ok("off") => Mode::Off,
            Ok("alt") => Mode::Alt,
            _ => Mode::On,
        },
    )
}

thread_local! {
    /// `alt`'s answer for the packet being executed.
    static ALT_NOW: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Called once at the start of a packet's execution: in `alt` mode, latch which
/// arm this packet takes.
#[inline]
pub(super) fn begin_packet() {
    if mode() == Mode::Alt {
        let block = crate::observe::elapsed_ms() / 4000;
        ALT_NOW.with(|now| now.set(block % 2 == 0));
    }
}

/// Whether the memo is on.
#[inline]
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST.with(std::cell::Cell::get) {
        return on;
    }
    match mode() {
        Mode::Off => false,
        Mode::On => true,
        Mode::Alt => ALT_NOW.with(std::cell::Cell::get),
    }
}

#[cfg(test)]
thread_local! {
    static TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with [`on`] answering `state` on this thread.
#[cfg(test)]
pub(crate) fn with<R>(state: bool, f: impl FnOnce() -> R) -> R {
    let before = TEST.with(|c| c.replace(Some(state)));
    let out = f();
    TEST.with(|c| c.set(before));
    out
}

/// How many distinct references one stream memoises. A stream binding more
/// than this resolves the rest the way it always did; the bound is the one
/// linear scan's length and not a cache policy.
const MAX_REFS: usize = 48;

/// One stream's resolved references.
#[derive(Clone, Debug, Default)]
pub(super) struct StreamMemo {
    resources: Vec<(u32, Arc<TaskResource>)>,
    /// What this stream's backing probe answered for a reference, either way.
    backing: Vec<(u32, bool)>,
    /// Resolutions this memo answered, not yet counted.
    replayed: u64,
}

impl StreamMemo {
    /// The retained resource this stream already resolved for `obj_ref`, counting
    /// `resolutions` skipped registry lookups against it.
    #[inline]
    pub(super) fn resource(&mut self, obj_ref: u32, resolutions: u64) -> Option<Arc<TaskResource>> {
        let hit = self
            .resources
            .iter()
            .find(|(r, _)| *r == obj_ref)
            .map(|(_, resource)| Arc::clone(resource))?;
        self.replayed += resolutions;
        Some(hit)
    }

    /// Keep a resolution that succeeded.
    #[inline]
    pub(super) fn remember(&mut self, obj_ref: u32, resource: &Arc<TaskResource>) {
        if self.resources.len() < MAX_REFS && !self.resources.iter().any(|(r, _)| *r == obj_ref) {
            self.resources.push((obj_ref, Arc::clone(resource)));
        }
    }

    /// What this stream's backing probe already answered for `obj_ref`.
    #[inline]
    pub(super) fn backing(&self, obj_ref: u32) -> Option<bool> {
        self.backing
            .iter()
            .find(|(r, _)| *r == obj_ref)
            .map(|(_, backed)| *backed)
    }

    /// Keep what a backing probe answered.
    #[inline]
    pub(super) fn remember_backing(&mut self, obj_ref: u32, backed: bool) {
        if self.backing.len() < MAX_REFS && self.backing(obj_ref).is_none() {
            self.backing.push((obj_ref, backed));
        }
    }

    /// Post the resolutions answered here, as the route a registry hit counts.
    pub(super) fn flush(&mut self) {
        if self.replayed != 0 {
            crate::runtime::drain::note_store_route_n(
                "task_resource_reexamine_deferred",
                self.replayed,
            );
            self.replayed = 0;
        }
    }
}

/// TEMP (lab, live `BACKING_XMEMO=on`, else `REIMS_VGPU_LAB_BACKING_XMEMO=on`;
/// default off): a texture's positive backing-probe answer kept across streams.
///
/// The probe ([`crate::runtime::objects::resolve_backing`]) reads the owning
/// task's object-list entry and the backing descriptor through the guest's page
/// tables on every stream that binds the texture - b165 (Weather light scene)
/// ~6 % of the drain's busy samples. On the warm answer it writes nothing new
/// (the owner hint it records is the one already there, the object is already
/// inserted). This keeps that answer, for the drain thread, while nothing it
/// stands on moved: the names generation (`LAB_NAMES_GEN`: every lifecycle
/// change, object insert, mapping rebuild and non-EXEC packet - a delete or a
/// re-point of the surface is a packet), the mapping's `map_generation`, and the
/// mapping still mapped with its pages, geometry and owner. The window this
/// adds over the per-stream memo is the same claim as that memo's (a probe is a
/// snapshot of the guest's lists; a write landing after it reaches a later
/// probe), stretched from one stream to the streams before the next names
/// change. Only `true` is kept.
pub(crate) fn backing_xmemo_on() -> bool {
    if cfg!(test) {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("BACKING_XMEMO") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_BACKING_XMEMO").as_deref() == Ok("on"))
}

#[derive(Clone, Copy)]
struct XBacking {
    names_gen: u64,
    map_generation: u32,
    owner: u32,
}

thread_local! {
    static XMEMO: std::cell::RefCell<reims_vgpu_core::fasthash::FxHashMap<u32, XBacking>> =
        std::cell::RefCell::new(reims_vgpu_core::fasthash::FxHashMap::default());
}

/// What the mapping says now, if it is still a warm backing.
fn xbacking_now(state: &crate::model::DeviceState, surface_id: u32) -> Option<XBacking> {
    let m = state.mappings.get(&surface_id)?;
    (m.mapped && m.has_geom && !m.page_entries.is_empty()).then(|| XBacking {
        names_gen: state.names_gen(),
        map_generation: m.map_generation,
        owner: m.owner_task_hint,
    })
}

/// Whether a kept answer for `surface_id` still holds.
pub(super) fn backing_xmemo_hit(state: &crate::model::DeviceState, surface_id: u32) -> bool {
    if crate::runtime::par::planning() {
        return false;
    }
    let Some(now) = xbacking_now(state, surface_id) else {
        return false;
    };
    let hit = XMEMO.with(|m| {
        m.borrow().get(&surface_id).is_some_and(|k| {
            k.names_gen == now.names_gen && k.map_generation == now.map_generation && k.owner == now.owner
        })
    });
    crate::runtime::drain::note_store_route(if hit { "backing_xmemo_hit" } else { "backing_xmemo_miss" });
    hit
}

/// Keep a positive answer the probe just gave for `surface_id`.
pub(super) fn backing_xmemo_remember(state: &crate::model::DeviceState, surface_id: u32) {
    if crate::runtime::par::planning() {
        return;
    }
    let Some(now) = xbacking_now(state, surface_id) else {
        return;
    };
    XMEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= 4096 {
            m.clear();
        }
        m.insert(surface_id, now);
    });
}
