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
/// arm this packet takes; under `WALK_XMEMO`, latch the packet's session (see
/// [`latch_session`]).
#[inline]
pub(super) fn begin_packet(state: &crate::model::DeviceState) {
    if mode() == Mode::Alt {
        let block = crate::observe::elapsed_ms() / 4000;
        ALT_NOW.with(|now| now.set(block % 2 == 0));
    }
    if walk_xmemo_on() {
        latch_session(state);
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

    /// [`Self::resource`] with the batch behind it (TEMP lab, `WALK_XMEMO`):
    /// a stream's own map first, then the batch's, so the first bind of a
    /// reference in the 2nd..7th CB of a doorbell batch answers from what the
    /// 1st resolved. A batch answer is remembered into the stream's map too,
    /// and counts the same `resolutions` the stream's own repeat does — the
    /// skipped work is the same resolution either way.
    #[inline]
    pub(super) fn resource_x(
        &mut self,
        state: &crate::model::DeviceState,
        task_id: u32,
        obj_ref: u32,
        resolutions: u64,
    ) -> Option<Arc<TaskResource>> {
        if let Some(hit) = self.resource(obj_ref, resolutions) {
            return Some(hit);
        }
        if walk_xmemo_on() {
            if let Some(hit) = xmemo_resource(state, task_id, obj_ref) {
                self.replayed += resolutions;
                self.remember(obj_ref, &hit);
                return Some(hit);
            }
        }
        None
    }

    /// [`Self::remember`] with the batch behind it (TEMP lab, `WALK_XMEMO`):
    /// a resolution this stream paid is the batch's answer too.
    #[inline]
    pub(super) fn remember_x(
        &mut self,
        state: &crate::model::DeviceState,
        task_id: u32,
        obj_ref: u32,
        resource: &Arc<TaskResource>,
    ) {
        if walk_xmemo_on() {
            xmemo_resource_keep(state, task_id, obj_ref, resource);
        }
        self.remember(obj_ref, resource);
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

// ---------------------------------------------------------------------------
// TEMP (lab, `WALK_XMEMO`): the batch's resolve answers
// ---------------------------------------------------------------------------

/// TEMP (lab, live `WALK_XMEMO=on|off`, else `REIMS_VGPU_LAB_WALK_XMEMO=on|1`;
/// default off): the resolve answers one doorbell's **batch** of command
/// buffers shares, so the first bind of an object in the 2nd..7th CB of a
/// batch does not pay the resolution the 1st already paid.
///
/// The stream memo above lives exactly one stream — one command buffer — and a
/// driven 5K canvas sends the doorbell batches of ~7 CBs that bind the same
/// few objects (`db/cb = 1:7.3`, n338; `task_resource_reexamine_deferred`
/// 222 k/s against 5.1 k re-examinations says the repeats are already the
/// stream-internal ones). This keeps three answers for the drain thread across
/// the streams of one batch: the retained resource a bind resolved
/// ([`StreamMemo`]'s own question), the name an admission's object-namespace
/// lookup issued, and the storage key that admission derived from the guest's
/// object list and descriptor.
///
/// # Why the guards are the names generation and the session
///
/// Both answers can change only through a lifecycle change — a declare, a
/// delete, a mapping rebuild, a non-EXEC packet — and every one of those bumps
/// [`DeviceState::names_gen`], which is read here as an atomic before every
/// lookup (the same guard `RES_XMEMO` already stands the resource answer on;
/// a mismatch empties all three maps rather than serving one stale entry).
/// A reset bumps it too, for the same reason at the largest scale: it empties
/// every namespace wholesale while the counter is process-global, and task
/// ids and object refs are reused across the boundary. The session is the
/// second term because a `SessionModel` generation change closes the names
/// without any lifecycle event at all; it arrives as the admission packet's
/// own parameter there, and is latched once per packet on the walk side (a
/// generation cannot move under a packet the drain holds shared state for).
///
/// # Only successes are kept
///
/// A failed construction is retried on the next ask — "a descriptor that is
/// still being published can succeed on retry" is the registry's own contract
/// — so a `None` takes the original path every time, exactly as the stream
/// memo's resource rule. The storage key is kept for the same reason the
/// resource is: it is derived from the descriptor bytes a resolution already
/// snapshotted, and `note_stale_task_resource`'s witness re-reads those bytes
/// on its interval either way.
///
/// Counters `walk_xmemo_hit`/`walk_xmemo_miss` (the walk side) and
/// `walk_xmemo_ingress_hit`/`walk_xmemo_ingress_miss` (admission) say whether
/// a batch repeats its resolutions at all; a canvas hit-rate under 50 % means
/// the lever is dead and the arm should not win its A/B.
pub(crate) fn walk_xmemo_on() -> bool {
    #[cfg(test)]
    if let Some(on) = XTEST.with(std::cell::Cell::get) {
        return on;
    }
    if let Some(on) = crate::runtime::lab_live::on("WALK_XMEMO") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("REIMS_VGPU_LAB_WALK_XMEMO").as_deref(),
            Ok("on" | "1")
        )
    })
}

#[cfg(test)]
thread_local! {
    static XTEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with [`walk_xmemo_on`] answering `state` on this thread.
#[cfg(test)]
pub(crate) fn with_walk_xmemo<R>(state: bool, f: impl FnOnce() -> R) -> R {
    let before = XTEST.with(|c| c.replace(Some(state)));
    let out = f();
    XTEST.with(|c| c.set(before));
    out
}

/// What the three maps below stand on. `PartialEq` on the pair rather than the
/// struct's fields one at a time, so a reader of a mismatch sees "the world
/// moved" rather than "one term moved".
#[derive(Clone, Copy, PartialEq, Eq)]
struct XGuard {
    names_gen: u64,
    session: reims_vgpu_core::identity::SessionGeneration,
}

/// The batch's answers. One cell for the three maps so one guard mismatch
/// retires all of them together — the events that invalidate one (a lifecycle
/// change, a reset) invalidate the other two by the same step.
struct WalkXmemo {
    guard: Option<XGuard>,
    /// The walk side: `(task, obj_ref)` → the retained resource a bind
    /// resolved. `Arc` rather than `Weak` (RES_XMEMO's choice) because the
    /// session guard, not the registry's drop, is what retires it here.
    resources: reims_vgpu_core::fasthash::FxHashMap<(u32, u32), Arc<TaskResource>>,
    /// Admission's object-namespace answer for `(task, obj_ref)`.
    names: reims_vgpu_core::fasthash::FxHashMap<
        (u32, u32),
        reims_vgpu_core::identity::ResourceId,
    >,
    /// Admission's storage answer for `(task, obj_ref)`.
    storage:
        reims_vgpu_core::fasthash::FxHashMap<(u32, u32), reims_vgpu_core::access::ResourceKey>,
}

impl WalkXmemo {
    fn clear(&mut self, guard: XGuard) {
        self.guard = Some(guard);
        self.resources.clear();
        self.names.clear();
        self.storage.clear();
    }
}

thread_local! {
    static WALK_XMEMO: std::cell::RefCell<WalkXmemo> = std::cell::RefCell::new(WalkXmemo {
        guard: None,
        resources: reims_vgpu_core::fasthash::FxHashMap::default(),
        names: reims_vgpu_core::fasthash::FxHashMap::default(),
        storage: reims_vgpu_core::fasthash::FxHashMap::default(),
    });
}

thread_local! {
    /// The session of the packet this thread is walking. Reading it takes the
    /// session's write lock, so it is latched once per packet (see
    /// [`latch_session`]) and every lookup compares against the latch; a
    /// thread with no latch (the switch turned on under a walk already in
    /// flight) simply does not use the walk-side map.
    static PACKET_SESSION: std::cell::Cell<
        Option<reims_vgpu_core::identity::SessionGeneration>,
    > = const { std::cell::Cell::new(None) };
}

/// Latch the session of the packet this thread is about to walk. Called once
/// per packet on the drain ([`begin_packet`]) and once per helper walk
/// (`walk_stream_spec`), because a reset cannot move under a packet the walker
/// holds shared state for.
pub(crate) fn latch_session(state: &crate::model::DeviceState) {
    PACKET_SESSION.with(|s| s.set(Some(state.session_generation())));
}

/// The guard the walk side stands on now, or `None` when this thread has not
/// latched a packet session (the memo is not in play there and the ask is not
/// counted).
fn walk_guard(state: &crate::model::DeviceState) -> Option<XGuard> {
    PACKET_SESSION.with(|s| s.get()).map(|session| XGuard {
        names_gen: state.names_gen(),
        session,
    })
}

/// The guard the admission side stands on now: the session arrives as the
/// packet's own parameter there, so no latch is needed.
fn ingress_guard(
    state: &crate::model::DeviceState,
    session: reims_vgpu_core::identity::SessionGeneration,
) -> XGuard {
    XGuard {
        names_gen: state.names_gen(),
        session,
    }
}

/// Retire the maps when the world moved, and answer through `f` either way.
fn asked<T>(
    guard: XGuard,
    route_hit: &'static str,
    route_miss: &'static str,
    f: impl FnOnce(&mut WalkXmemo) -> Option<T>,
) -> Option<T> {
    WALK_XMEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.guard != Some(guard) {
            m.clear(guard);
        }
        let hit = f(&mut m);
        crate::runtime::drain::note_store_route(if hit.is_some() {
            route_hit
        } else {
            route_miss
        });
        hit
    })
}

/// The batch's retained resource for `(task_id, obj_ref)`, walk side.
pub(super) fn xmemo_resource(
    state: &crate::model::DeviceState,
    task_id: u32,
    obj_ref: u32,
) -> Option<Arc<TaskResource>> {
    let guard = walk_guard(state)?;
    asked(guard, "walk_xmemo_hit", "walk_xmemo_miss", |m| {
        m.resources.get(&(task_id, obj_ref)).cloned()
    })
}

/// Keep a walk-side answer that succeeded.
pub(super) fn xmemo_resource_keep(
    state: &crate::model::DeviceState,
    task_id: u32,
    obj_ref: u32,
    resource: &Arc<TaskResource>,
) {
    let Some(guard) = walk_guard(state) else {
        return;
    };
    WALK_XMEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.guard != Some(guard) {
            m.clear(guard);
        }
        if m.resources.len() >= 4096 {
            m.resources.clear();
        }
        m.resources.insert((task_id, obj_ref), Arc::clone(resource));
        // The answer was resolved *before* this keep read the guard, so a
        // lifecycle change landing in between would file an old-world answer
        // under the new generation — one stale hit until the next bump. The
        // guard is re-read after the insert and the cell cleared on movement;
        // the ask side's own mismatch check stays as the second half.
        if let Some(now) = walk_guard(state) {
            if now != guard {
                m.clear(now);
            }
        }
    });
}

/// The batch's name for `(task_id, obj_ref)`, admission side.
pub(crate) fn xmemo_name(
    state: &crate::model::DeviceState,
    session: reims_vgpu_core::identity::SessionGeneration,
    task_id: u32,
    obj_ref: u32,
) -> Option<reims_vgpu_core::identity::ResourceId> {
    let guard = ingress_guard(state, session);
    asked(guard, "walk_xmemo_ingress_hit", "walk_xmemo_ingress_miss", |m| {
        m.names.get(&(task_id, obj_ref)).copied()
    })
}

/// Keep an admission-side name that succeeded.
pub(crate) fn xmemo_name_keep(
    state: &crate::model::DeviceState,
    session: reims_vgpu_core::identity::SessionGeneration,
    task_id: u32,
    obj_ref: u32,
    name: reims_vgpu_core::identity::ResourceId,
) {
    let guard = ingress_guard(state, session);
    WALK_XMEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.guard != Some(guard) {
            m.clear(guard);
        }
        if m.names.len() >= 4096 {
            m.names.clear();
        }
        m.names.insert((task_id, obj_ref), name);
    });
}

/// The batch's storage key for `(task_id, obj_ref)`, admission side.
pub(crate) fn xmemo_storage(
    state: &crate::model::DeviceState,
    session: reims_vgpu_core::identity::SessionGeneration,
    task_id: u32,
    obj_ref: u32,
) -> Option<reims_vgpu_core::access::ResourceKey> {
    let guard = ingress_guard(state, session);
    asked(guard, "walk_xmemo_ingress_hit", "walk_xmemo_ingress_miss", |m| {
        m.storage.get(&(task_id, obj_ref)).copied()
    })
}

/// Keep an admission-side storage key that succeeded.
pub(crate) fn xmemo_storage_keep(
    state: &crate::model::DeviceState,
    session: reims_vgpu_core::identity::SessionGeneration,
    task_id: u32,
    obj_ref: u32,
    key: reims_vgpu_core::access::ResourceKey,
) {
    let guard = ingress_guard(state, session);
    WALK_XMEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.guard != Some(guard) {
            m.clear(guard);
        }
        if m.storage.len() >= 4096 {
            m.storage.clear();
        }
        m.storage.insert((task_id, obj_ref), key);
    });
}
