//! What can be said about a GVA render target's guest pages since the Store
//! that published them.
//!
//! # Why this exists
//!
//! A mapper-ref-texture render Store publishes into a mapping, and the device can ask
//! whether anything has written that mapping since — the token lives on
//! [`crate::model::MappingEntry`] and
//! [`crate::runtime::mapper::mapping_guest_write_verdict`] reads it. That
//! witness is what licenses the mapper-ref-texture attachment LOAD elision, which serves a
//! `MTLLoadActionLoad` straight out of the engine resident instead of reading a
//! whole frame back out of guest memory: one driven boot elided **11 484** seeds
//! against 94 uploaded.
//!
//! A **GVA** (normal-texture) render target has no mapping and so had no witness, and
//! its LOAD reads the attachment's full span out of guest pages every time —
//! `load_seed_ok_color` 4 949 per driven Safari-drag boot, blocking on a
//! guest-write settle for 2 923 of them, because those are exactly the pages the
//! previous Store has just written. This is the missing half.
//!
//! # Two writers, two questions, and neither may be dropped
//!
//! [`GvaWriteReach`] answers both:
//!
//! * **The guest CPU** wrote the span. Only the hypervisor's dirty bitmap can
//!   see this, through a token registered over the target's pages.
//! * **This device** wrote the span through some other rail — a compute
//!   writeback, a blit, a mapper-ref-texture write whose pages alias it. The hypervisor
//!   cannot see those; [`crate::runtime::host_writes`] is the record that can,
//!   and aliasing across id namespaces is real rather than theoretical (see that
//!   module's own doc).
//!
//! What this module does **not** answer is whether a draw has landed in the
//! resident since the Store. That is the engine's `content_epoch`, and the
//! caller compares it, because only the caller knows which resident it means.
//!
//! # Why the key is the whole identity
//!
//! A token registered over one page list says nothing about another, and the
//! guest recycles GVAs. The mapper-ref-texture twin handles this with an explicit
//! `guest_write_token_gen != map_generation` check that a future writer has to
//! remember. Here the page-set hash *is part of the key*
//! (`draw::vulkan::gva_span_alloc_generation`, the same value the engine
//! registry keys the resident on), so a target whose page list has moved cannot
//! be found under the key its token was armed under. The wrong answer is
//! unreachable rather than guarded; the stale entry becomes an orphan, which is
//! a resource cost the eviction below collects.
//!
//! # Why not [`crate::runtime::gather_witness`]
//!
//! It already owns a bounded, token-holding map keyed by `GatherKey::TaskGva`,
//! and reusing it was the obvious move. Two things stop it, and the first is a
//! live hazard rather than an inelegance. Its entry carries the
//! sampled-content generation the engine binds retained images on, and it keys
//! on `(task_id, gva)` with the *sampled* span — which for a compositing layer
//! that is both rendered into and sampled is the same key at a different span,
//! so the two rails would re-point each other's entries every frame and spend
//! each other's generations. Second, its stamp is overwritten on every bind, so
//! "since the previous gather" cannot express "since the Store".
//!
//! What is shared instead is everything with no such coupling: the
//! [`GuestWriteVerdict`] spelling, the release rail through
//! `DeviceState::retired_guest_write_tokens`, and the eviction shape.
//!
//! # The rule every consumer must obey
//!
//! Everything undecidable answers as written. A host with no dirty bitmap, a
//! token still inside its arming window, a target no Store has stamped, a
//! host-write record that has aged out — all refuse, and a refusal costs a
//! re-read rather than a wrong frame. The one answer that must never be
//! invented is [`GvaWriteReach::Quiet`].

use crate::model::DeviceState;
use crate::runtime::host::HostOps;
use crate::runtime::host_writes::HostWriteVerdict;
use crate::runtime::mapper::GuestWriteVerdict;
use std::collections::BTreeMap;

/// Which GVA render target a witness entry is about: the guest span the rail's
/// resident stands for, spelled without the rail's own type so `DeviceState`
/// stays backend-neutral.
///
/// `Backend::gva_witness_key` is the only constructor a product path may use,
/// so the arm site and the payment site cannot drift into naming different
/// targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct GvaTargetKey {
    pub gva: u64,
    /// Hash of the resolved guest page set — `gva_span_alloc_generation`. Part
    /// of the key, which is what makes a moved page list a different target
    /// rather than a stale answer. Never 0 for a usable key: 0 is that
    /// function's "the walk was short".
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    /// Channel order of the resident this key was taken from.
    ///
    /// It is an order and not the resident's whole format because this key
    /// names a **guest span**, and every format a rail's resident can hold
    /// today maps onto one of the two orders one-for-one. That stops being true
    /// the moment a render target may be wider than eight bits per channel:
    /// RGBA8 and half-float RGBA are one order and two residents, and they
    /// would share a key here. Whoever widens the range a rail's resident
    /// format may take widens this with it.
    pub bgra: bool,
}

#[derive(Debug)]
struct Entry {
    /// Host tracking token over this target's pages. Never 0: an entry is only
    /// inserted once the host has armed one.
    token: u64,
    /// The pages the token watches, kept so the host-write half can be asked
    /// page-exactly rather than device-globally.
    gpas: Vec<u64>,
    /// `guest_write_gen` as it stood at the Store. 0 means "no usable stamp" —
    /// the arming window — and can never equal a live generation, because the
    /// host's first readable one is 1. Meaningless while `pending` is set.
    gen_at_store: u64,
    /// TEMP (lab, `REIMS_VGPU_LAB_ASYNC_BASELINE`): the synced read of
    /// `gen_at_store` is still out on the baseline thread. Until it resolves the
    /// completion stamp of the packet that made this Store is held, so the guest
    /// has not written the target since — see [`crate::runtime::write_baseline`].
    pending: Option<crate::runtime::write_baseline::PendingGen>,
    /// TEMP (lab, `REIMS_VGPU_LAB_GVAW_LAGGED`): a writer may use
    /// `gen_at_store` ([`armed`], [`written_pages`]). False for an eager
    /// Store's lagged stamp, which may predate guest writes made before the
    /// Store: a reader may count those against the Store (a re-read), a writer
    /// may not. A deferral's lagged stamp ([`note_store_deferral`]) is usable.
    exact: bool,
    /// `host_writes.epoch()` as it stood at the Store, read *after* the Store
    /// recorded its own write so the Store does not invalidate itself.
    host_epoch_at_store: u64,
    /// Store ordinal, for the eviction order.
    last_seen: u64,
    /// TEMP (lab, PRIVATE_GPU): stamped quiet by contract — the target is
    /// `MTLStorageModePrivate`, `token` is 0 and nothing is tracked. See
    /// [`note_store_contract_quiet`].
    contract_quiet: bool,
}

impl Entry {
    /// The generation at the Store, or `None` while its read is still out.
    fn gen(&self) -> Option<u64> {
        match &self.pending {
            Some(p) => p.get(),
            None => Some(self.gen_at_store),
        }
    }

    /// Whether the Store left a usable stamp: a resolved non-zero generation,
    /// or a read still out (it resolves before the guest can write the target).
    fn stamped(&self) -> bool {
        self.gen().map_or(true, |g| g != 0)
    }
}

/// Per-device witness state: one entry per GVA render target stamped.
#[derive(Default, Debug)]
pub struct GvaStoreWitness {
    entries: BTreeMap<GvaTargetKey, Entry>,
    stores: u64,
    /// TEMP (lab, ADMIT_AHEAD): the target each `(task, texture_ref)`'s last
    /// deferred Store stamped. See [`Self::note_ref_target`].
    by_ref: std::collections::HashMap<(u32, u32), GvaTargetKey>,
}

/// Upper bound on tracked GVA targets.
///
/// A hypervisor harvest bound and not a memory one, for the same reason
/// `gather_witness::MAX_TRACKED_WINDOWS` is and out of the same budget: the shim
/// walks every page of every tracked set on the BQL thread at each register
/// write that hands the device work, so each armed target adds its page count to
/// a cost the whole VM pays. The two constants are halves of one budget and the
/// basis is stated in both.
///
/// The population is the compositing layers a desktop keeps live at once, which
/// is the same order as the sampled working set that sized the other bound.
/// Overflow evicts the least recently stored target rather than dropping the
/// map, so a working set past this degrades one target at a time instead of
/// un-arming every live one at once. `gvaw_evict` counts it, so a boot says
/// whether the bound binds rather than leaving it assumed.
const MAX_TRACKED_TARGETS: usize = 128;

/// TEMP (lab): the bound [`note_store`] evicts at. 128 unless
/// `REIMS_VGPU_LAB_GVAW_MAX` names another (16..=4096); test builds keep
/// [`MAX_TRACKED_TARGETS`]. Mission Control stores past 128 live targets and
/// evicted 233 a second, each an untrack and a fresh track through the
/// hypervisor (~12 % of its draw path); the harvest walk the bound was sized
/// against is the on-demand sync now, which reads a set only when asked.
fn max_tracked_targets() -> usize {
    if cfg!(test) {
        return MAX_TRACKED_TARGETS;
    }
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_GVAW_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(MAX_TRACKED_TARGETS)
            .clamp(16, 4096)
    })
}

impl GvaStoreWitness {
    /// Detach every tracking token, for release through
    /// [`HostOps::untrack_guest_writes`].
    ///
    /// Returns them rather than releasing them because this type has no
    /// `HostOps`, exactly as `gather_witness::GatherWitness::take_tokens` does;
    /// the crate's one rail for host state it cannot free itself is
    /// `DeviceState::retired_guest_write_tokens`.
    pub fn take_tokens(&mut self) -> Vec<u64> {
        let tokens = self
            .entries
            .values()
            .map(|e| e.token)
            .filter(|t| *t != 0)
            .collect();
        self.entries.clear();
        tokens
    }

    /// How many targets are armed. For the tests that bound the map.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// TEMP (lab, `REIMS_VGPU_LAB_ADMIT_AHEAD`): remember which target the
    /// deferred Store of `(task_id, texture_ref)` stamped, so an admission batch
    /// can name exactly that target's set before the next Store of the same
    /// texture reads it. A hint index: a stale or evicted pairing names nothing.
    pub fn note_ref_target(&mut self, task_id: u32, texture_ref: u32, key: GvaTargetKey) {
        if self.by_ref.len() >= 4096 && !self.by_ref.contains_key(&(task_id, texture_ref)) {
            self.by_ref.clear();
        }
        self.by_ref.insert((task_id, texture_ref), key);
    }

    /// TEMP (lab, PRIVATE_GPU census): whether a deferred Store of `(task_id,
    /// texture_ref)` has stamped a target this witness still holds.
    pub fn stored_ref(&self, task_id: u32, texture_ref: u32) -> bool {
        self.by_ref
            .get(&(task_id, texture_ref))
            .is_some_and(|key| self.entries.contains_key(key))
    }

    /// TEMP (lab, `REIMS_VGPU_LAB_ADMIT_AHEAD`): the tracking token of the target
    /// the last deferred Store of `(task_id, texture_ref)` stamped, if it is
    /// still tracked.
    pub fn token_for_ref(&self, task_id: u32, texture_ref: u32) -> Option<u64> {
        let key = self.by_ref.get(&(task_id, texture_ref))?;
        self.entries.get(key).map(|e| e.token).filter(|t| *t != 0)
    }

    fn evict_oldest(&mut self, retired: &mut Vec<u64>) {
        let Some(victim) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_seen)
            .map(|(k, _)| *k)
        else {
            return;
        };
        if let Some(e) = self.entries.remove(&victim) {
            crate::runtime::drain::note_store_route("gvaw_evict");
            if e.token != 0 {
                retired.push(e.token);
            }
        }
    }
}

/// Record what the two witnesses say at the moment a Store publishes this
/// target, registering its pages for tracking the first time.
///
/// `gpas` must be the target's **complete** page list — the same walk the Store
/// built its destination runs from. A partial list would have the host watch
/// part of the span and report unwritten for the rest, which is the one answer
/// that must never be invented; `store_gva_frame` refuses a short walk before it
/// reaches here, and this refuses an empty one.
///
/// **Call this after the Store has recorded its own write** through
/// `DeviceState::note_host_wrote_pages`. The host-write epoch captured here is
/// what [`reach`] compares against, and capturing it first would have every
/// target permanently invalidated by its own Store.
///
/// The twin of [`crate::runtime::mapper::stamp_guest_write_gen`], with counters
/// named apart on purpose: a rail whose registration silently never happens is
/// indistinguishable from a guest that writes every frame, and the two want
/// opposite fixes.
pub fn note_store<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    key: GvaTargetKey,
    gpas: &[u64],
) {
    stamp(state, host, key, gpas, Read::Synced);
}

/// [`note_store`] for a Store about to be deferred: the writer's arming.
///
/// TEMP (lab, `REIMS_VGPU_LAB_GVAW_LAGGED`, on unless `=off`): an armed target
/// whose resource the guest has never declared a CPU write to (`undeclared`,
/// `BufferWriteGens::declared`) is re-stamped from the lagged generation.
/// Nearly every GVA Store is deferred (n83: ~23k a boot, 76 % superseded
/// before payment, 0-1 merges), and the synced read here was a foreground sync
/// of the target's pages each time — store routing halved on transitions
/// without it (n84 fga2: 48-59 -> 25 us a chain).
///
/// What the lagged baseline changes: [`written_pages`] may also name pages
/// the guest wrote between the set's last sync and this Store, and a merge
/// keeps the guest's bytes there. The merge runs only after the guest declares
/// a write since the Store. A CPU write made before the Store is declared to
/// the device by the next command buffer that uses the object — the Store's
/// own, if it came first — so an object with no declaration at all has had no
/// such write the device was not told of, and for it the superset is the exact
/// set. An object with declarations keeps the synced read. `=all` (measurement
/// only) takes the lagged read for every object.
pub fn note_store_deferral<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    key: GvaTargetKey,
    gpas: &[u64],
    undeclared: bool,
) {
    let mode = lab_lagged_mode();
    let lagged = (mode == 2 || (mode == 1 && undeclared))
        && state
            .gva_store_witness
            .entries
            .get(&key)
            .is_some_and(Entry::stamped);
    // A deferral is armed from the draw path, inside the packet that rendered
    // the target, which is what lets its synced read wait for that packet's
    // stamp instead of running here.
    let read = if lagged { Read::Lagged } else { Read::Deferred };
    stamp(state, host, key, gpas, read);
    if lagged {
        crate::runtime::drain::note_store_route("lab_gvaw_deferral_lagged");
        if let Some(e) = state.gva_store_witness.entries.get_mut(&key) {
            e.exact = true;
        }
    }
}

/// TEMP (lab, `REIMS_VGPU_LAB_GVAW_LAGGED`, on unless `=off`): whether an eager
/// Store re-stamps an armed target from the generation its set was last synced
/// to rather than syncing it now.
fn lab_lagged_enabled() -> bool {
    lab_lagged_mode() != 0
}

/// 0 `off`, 1 eager Stores and undeclared deferrals (default), 2 `all`
/// (measurement, see [`note_store_deferral`]).
fn lab_lagged_mode() -> u8 {
    if cfg!(test) {
        return 1;
    }
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("REIMS_VGPU_LAB_GVAW_LAGGED").as_deref() {
            Ok("off") => 0,
            Ok("all") => 2,
            _ => 1,
        },
    )
}

/// [`note_store`] for a Store that landed the whole frame in the guest's pages
/// now (nothing deferred), which leaves only readers to answer.
///
/// TEMP (lab, `REIMS_VGPU_LAB_GVAW_LAGGED`): an already armed target is
/// re-stamped from the lagged generation. The synced read was a foreground
/// dirty sync of the target's pages at every Store — 95 % of this function's
/// cost and ~10 % of the drain on fullscreen transitions (n81) — and the next
/// reader syncs the same set again anyway. An older baseline only ever moves
/// a reader towards "written" (a re-read), never towards a false `Quiet`:
/// generations are monotonic. A first Store still reads synced, so a new target
/// leaves its arming window at once.
pub fn note_store_eager<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    key: GvaTargetKey,
    gpas: &[u64],
) {
    let lagged = lab_lagged_enabled()
        && state
            .gva_store_witness
            .entries
            .get(&key)
            .is_some_and(Entry::stamped);
    let read = if lagged { Read::Lagged } else { Read::Synced };
    stamp(state, host, key, gpas, read);
}

/// How [`stamp`] reads the generation at the Store.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Read {
    Synced,
    /// TEMP (lab, `REIMS_VGPU_LAB_GVAW_LAGGED`).
    Lagged,
    /// TEMP (lab, `REIMS_VGPU_LAB_ASYNC_BASELINE`): synced, on the baseline
    /// thread when the host has one. Only for a Store made inside the packet
    /// whose completion stamp is still to come.
    Deferred,
}

fn stamp<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    key: GvaTargetKey,
    gpas: &[u64],
    read: Read,
) {
    let lagged = read == Read::Lagged;
    if key.generation == 0 || gpas.is_empty() {
        crate::runtime::drain::note_store_route("gvaw_unnamed_alloc");
        return;
    }
    let page_size = state.page_size() as usize;
    // Re-arming an existing key means the same page list by construction — the
    // hash is in the key — so the token is reused and only the stamps move.
    // A contract-quiet entry (TEMP lab PRIVATE_GPU) holds no token: a Store
    // that stamps its key the ordinary way tracks the pages afresh.
    let token = match state.gva_store_witness.entries.get(&key).filter(|e| e.token != 0) {
        Some(e) => e.token,
        None => {
            while state.gva_store_witness.entries.len() >= max_tracked_targets() {
                let mut retired = Vec::new();
                state.gva_store_witness.evict_oldest(&mut retired);
                state.retired_guest_write_tokens.extend(retired);
            }
            match host.track_guest_writes(gpas, page_size) {
                Some(t) => t,
                None => {
                    // No dirty bitmap, or the host declined this set. Counted so
                    // a boot can tell a witness that never armed from a guest
                    // that writes every frame.
                    crate::runtime::drain::note_store_route("gvaw_untracked");
                    return;
                }
            }
        }
    };
    // A re-Store while the previous Store's read is still out keeps that read:
    // its stamp is held, so the guest has written nothing since either Store,
    // and the lagged answer for a set not synced yet would be the arming window.
    let inherited = match state.gva_store_witness.entries.get(&key) {
        Some(e) if read != Read::Synced && e.gen().is_none() => e.pending.clone(),
        _ => None,
    };
    let lagged = lagged && inherited.is_none();
    let pending = match inherited {
        Some(p) => {
            crate::runtime::drain::note_store_route("lab_gvaw_armed_inherited");
            Some(p)
        }
        None if read == Read::Deferred => host.guest_write_gen_deferred(token),
        None => None,
    };
    let read = if pending.is_some() {
        crate::runtime::drain::note_store_route("lab_gvaw_armed_async");
        None
    } else if lagged {
        host.guest_write_gen_lagged(token)
    } else {
        host.guest_write_gen(token)
    };
    let gen_at_store = match read {
        None if pending.is_some() => 0,
        Some(g) => {
            crate::runtime::drain::note_store_route(if lagged {
                "lab_gvaw_armed_lagged"
            } else {
                "gvaw_armed"
            });
            g
        }
        None => {
            // The host holds the pages but its report is not yet a fact about
            // the guest — the arming window. 0 fails the currency test rather
            // than passing it by default.
            crate::runtime::drain::note_store_route("gvaw_unarmed");
            0
        }
    };
    state.gva_store_witness.stores = state.gva_store_witness.stores.wrapping_add(1);
    let last_seen = state.gva_store_witness.stores;
    let host_epoch_at_store = state.host_writes.epoch();
    state.gva_store_witness.entries.insert(
        key,
        Entry {
            token,
            gpas: gpas.to_vec(),
            gen_at_store,
            pending,
            exact: !lagged,
            host_epoch_at_store,
            last_seen,
            contract_quiet: false,
        },
    );
}

/// TEMP (lab, `REIMS_VGPU_LAB_PRIVATE_GPU`, live `PRIVATE_GPU=on|off`, off by
/// default): a deferred Store into a target the contract gives the guest CPU no
/// access to at all.
///
/// `MTLStorageModePrivate`: the guest CPU can neither read nor write the
/// texture's memory, so there is no guest write for the witness to watch and
/// nothing to sync — the hypervisor set this would arm (a fresh one per new
/// target key, synced inside its arming window: p033/p034 GEN_PROF, ~19k synced
/// reads a phase at `stamp`, 1.5-2.2 s of the drain) can only ever read clean.
/// The entry is stamped quiet by contract instead: no token, a non-zero stamp,
/// exact. [`reach`] answers from the host-write half alone and
/// [`written_pages`] names none. The entry leaves with its resource
/// ([`forget_contract_ref`], from `retire_gva_resource`), so pages a later,
/// CPU-visible object reuses under the same key are tracked again.
pub fn note_store_contract_quiet(
    state: &mut DeviceState,
    key: GvaTargetKey,
    gpas: &[u64],
    task_id: u32,
    texture_ref: u32,
) {
    if key.generation == 0 || gpas.is_empty() {
        crate::runtime::drain::note_store_route("gvaw_unnamed_alloc");
        return;
    }
    if let Some(old) = state.gva_store_witness.entries.get(&key) {
        if old.token != 0 {
            state.retired_guest_write_tokens.push(old.token);
        }
    } else {
        while state.gva_store_witness.entries.len() >= max_tracked_targets() {
            let mut retired = Vec::new();
            state.gva_store_witness.evict_oldest(&mut retired);
            state.retired_guest_write_tokens.extend(retired);
        }
    }
    state.gva_store_witness.stores = state.gva_store_witness.stores.wrapping_add(1);
    let last_seen = state.gva_store_witness.stores;
    let host_epoch_at_store = state.host_writes.epoch();
    state.gva_store_witness.entries.insert(
        key,
        Entry {
            token: 0,
            gpas: gpas.to_vec(),
            gen_at_store: 1,
            pending: None,
            exact: true,
            host_epoch_at_store,
            last_seen,
            contract_quiet: true,
        },
    );
    state.gva_store_witness.note_ref_target(task_id, texture_ref, key);
    crate::runtime::drain::note_store_route("lab_gvaw_contract_quiet");
}

/// TEMP (lab, PRIVATE_GPU): the resource `(task_id, texture_ref)` is gone; a
/// contract-quiet entry its last Store left is dropped with it, so whatever
/// reuses those pages under the same key is not taken for the private texture.
pub fn forget_contract_ref(state: &mut DeviceState, task_id: u32, texture_ref: u32) {
    let w = &mut state.gva_store_witness;
    let Some(key) = w.by_ref.remove(&(task_id, texture_ref)) else {
        return;
    };
    if w.entries.get(&key).is_some_and(|e| e.contract_quiet) {
        w.entries.remove(&key);
        crate::runtime::drain::note_store_route("lab_gvaw_contract_forgotten");
    }
}

/// Forget every target whose pages this task owned, handing back their tokens.
///
/// A task teardown takes its page tables with it, so nothing in it names
/// anything any more. There is no task in the key — see [`GvaTargetKey`] — so
/// this takes the page list instead: an entry every one of whose pages the
/// caller names as gone is gone. Called with the task's retired page set.
pub fn retire_pages(state: &mut DeviceState, gone: &[u64]) {
    if gone.is_empty() {
        return;
    }
    let doomed: Vec<GvaTargetKey> = state
        .gva_store_witness
        .entries
        .iter()
        .filter(|(_, e)| e.gpas.iter().any(|p| gone.contains(p)))
        .map(|(k, _)| *k)
        .collect();
    for k in doomed {
        if let Some(e) = state.gva_store_witness.entries.remove(&k) {
            if e.token != 0 {
                state.retired_guest_write_tokens.push(e.token);
            }
        }
    }
}

/// What both witnesses say about `key`'s pages since the Store that stamped
/// them.
///
/// Every variant but [`GvaWriteReach::Quiet`] means "assume written". They are
/// kept apart because "this rail never got started", "the guest rewrites this
/// target every frame" and "the record aged out" are the same refusal and three
/// completely different findings — the same reason the mapper-ref-texture twin splits its
/// own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GvaWriteReach {
    /// Neither the guest nor another rail of this device has written these
    /// pages since the Store. The only answer that licenses reusing the
    /// resident, and only in combination with the caller's own epoch test.
    Quiet,
    /// The hypervisor's verdict, when it is not clean.
    Guest(GuestWriteVerdict),
    /// This device's own record, when it is not quiet.
    Host(HostWriteVerdict),
}

impl GvaWriteReach {
    /// Census route naming this answer, so a boot can say which half refused.
    /// The two halves keep their own vocabularies rather than being folded into
    /// one, because they are repaired differently.
    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "gvaw_quiet",
            Self::Guest(GuestWriteVerdict::Clean) => "gvaw_guest_clean",
            Self::Guest(GuestWriteVerdict::NoMapping) => "gvaw_no_entry",
            Self::Guest(GuestWriteVerdict::NoStamp) => "gvaw_guest_no_stamp",
            Self::Guest(GuestWriteVerdict::Wrote) => "gvaw_guest_wrote",
            Self::Guest(GuestWriteVerdict::Unreadable) => "gvaw_guest_unreadable",
            Self::Host(HostWriteVerdict::Quiet) => "gvaw_host_quiet",
            Self::Host(HostWriteVerdict::Overlap) => "gvaw_host_overlap",
            Self::Host(HostWriteVerdict::Unnamed) => "gvaw_host_unnamed",
        }
    }

    /// True only for [`Self::Quiet`], spelled once so a caller cannot express
    /// the question as "not one of the variants I remembered".
    pub fn is_quiet(self) -> bool {
        matches!(self, Self::Quiet)
    }
}

/// Ask both witnesses about `key`.
///
/// The guest half is asked first because it is a word; the host-write half
/// indexes the target's page epochs and is only reached when the guest half is
/// clean.
pub fn reach<H: HostOps>(state: &DeviceState, host: &H, key: GvaTargetKey) -> GvaWriteReach {
    let Some(e) = state.gva_store_witness.entries.get(&key) else {
        return GvaWriteReach::Guest(GuestWriteVerdict::NoMapping);
    };
    // TEMP (lab, PRIVATE_GPU): no guest write is possible; only this device's own
    // writes are left to ask about.
    let gen = if e.contract_quiet { None } else { e.gen() };
    match gen {
        // TEMP (lab): the Store's read is still out, so its packet's stamp is
        // held and the guest has not written the target since the Store. Only
        // the host-write half is left to ask.
        None => crate::runtime::drain::note_store_route(if e.contract_quiet {
            "lab_gvaw_reach_contract"
        } else {
            "lab_gvaw_reach_pending"
        }),
        Some(0) => return GvaWriteReach::Guest(GuestWriteVerdict::NoStamp),
        Some(gen_at_store) => {
            if let Some(answer) = guest_half(host, e, gen_at_store) {
                return answer;
            }
        }
    }
    let host_verdict = state
        .host_writes
        .wrote_any_since(e.host_epoch_at_store, &e.gpas);
    match host_verdict {
        HostWriteVerdict::Quiet => GvaWriteReach::Quiet,
        other => GvaWriteReach::Host(other),
    }
}

/// The guest half of [`reach`] against a resolved stamp: `None` when clean.
fn guest_half<H: HostOps>(host: &H, e: &Entry, gen_at_store: u64) -> Option<GvaWriteReach> {
    match host.guest_write_gen(e.token) {
        Some(g) if g == gen_at_store => None,
        Some(_) => {
            // TEMP (lab): how often a lagged stamp reads written, against a synced one.
            crate::runtime::drain::note_store_route(if e.exact {
                "lab_gvaw_reach_wrote_exact"
            } else {
                "lab_gvaw_reach_wrote_lagged"
            });
            Some(GvaWriteReach::Guest(GuestWriteVerdict::Wrote))
        }
        None => Some(GvaWriteReach::Guest(GuestWriteVerdict::Unreadable)),
    }
}

/// Whether this device can name what the guest writes to `key` from now on.
///
/// True exactly when [`written_pages`] can answer, which is the question a
/// **writer** has to ask before it defers a frame: a deferred Store is only
/// recoverable from a guest write if the extent of that write is nameable
/// afterwards. A reader asks [`reach`] instead — it is allowed to give up and
/// re-read, and a writer that gives up has already lost the frame.
///
/// False for a target no Store has stamped and for one still inside the
/// hypervisor's arming window, where the generation reads back 0 and no page
/// stamp can be compared against it.
pub fn armed(state: &DeviceState, key: GvaTargetKey) -> bool {
    state
        .gva_store_witness
        .entries
        .get(&key)
        .is_some_and(|e| e.stamped() && e.exact)
}

/// Which of `key`'s pages the hypervisor has observed the guest write since the
/// Store that stamped it, in ascending GPA order.
///
/// The extent behind a `GvaWriteReach::Guest(GuestWriteVerdict::Wrote)`, for the
/// one caller that has something to do with it other than refuse: a deferred
/// writeback deciding which of the frame's bytes the guest's own memory keeps
/// instead of the device's. [`reach`] answers *whether*; this answers *where*,
/// and the two must not be collapsed — a reader reusing a host-side copy is
/// wrong if any page moved, while a writer landing a frame is wrong only on the
/// pages that did.
///
/// `None` is "the host cannot name them" — no dirty bitmap, an unarmed token, a
/// target this device never stamped, or more written pages than the host will
/// enumerate — and every caller must read it as "assume the guest wrote all of
/// them", which is the same direction [`reach`] refuses in. An empty `Some` is a
/// real answer and says the harvest saw no page of this target move.
pub fn written_pages<H: HostOps>(
    state: &DeviceState,
    host: &H,
    key: GvaTargetKey,
) -> Option<Vec<u64>> {
    let Some(e) = state.gva_store_witness.entries.get(&key) else {
        crate::runtime::drain::note_store_route("gvaw_pages_no_entry");
        return None;
    };
    if e.contract_quiet {
        // TEMP (lab, PRIVATE_GPU): the guest has no access to these pages.
        crate::runtime::drain::note_store_route("lab_gvaw_pages_contract");
        return Some(Vec::new());
    }
    let gen_at_store = match (e.gen(), &e.pending) {
        (Some(g), _) => g,
        (None, Some(p)) => {
            crate::runtime::drain::note_store_route("lab_gvaw_pages_joined");
            p.wait()
        }
        (None, None) => unreachable!("only a pending read is unresolved"),
    };
    if gen_at_store == 0 {
        crate::runtime::drain::note_store_route("gvaw_pages_unarmed");
        return None;
    }
    if !e.exact {
        crate::runtime::drain::note_store_route("lab_gvaw_pages_lagged");
        return None;
    }
    let answer = host.guest_written_pages(e.token, gen_at_store);
    crate::runtime::drain::note_store_route(match answer {
        Some(_) => "gvaw_pages_named",
        None => "gvaw_pages_unreadable",
    });
    answer
}

/// How far back the host-write record is being asked to reach for `key`, banded.
///
/// `host_writes`'s own doc asks for exactly this distribution before anyone
/// resizes its ring: "a reach that is usually 70 and a reach that is usually
/// 7000 both produce [an `Aged`-dominated] reading, and only one of them is
/// fixed by a bigger ring". This rail asks over a Store-to-LOAD interval, which
/// is the longest reach in the device, so it is the one that should supply the
/// bands.
pub fn note_host_reach(state: &DeviceState, key: GvaTargetKey) {
    let Some(e) = state.gva_store_witness.entries.get(&key) else {
        return;
    };
    let reach = state
        .host_writes
        .epoch()
        .saturating_sub(e.host_epoch_at_store);
    crate::runtime::drain::note_store_route(if reach < 64 {
        "gvaw_reach_lt64"
    } else if reach < 512 {
        "gvaw_reach_lt512"
    } else if reach < 4096 {
        "gvaw_reach_lt4k"
    } else {
        "gvaw_reach_ge4k"
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::host::FakeHost;

    /// The x86 guest page, to match the `DeviceState` these tests build.
    const PAGE: u64 = 1u64 << crate::model::PAGE_SHIFT_X86;

    fn device() -> DeviceState {
        DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86)
    }

    fn key(generation: u64) -> GvaTargetKey {
        GvaTargetKey {
            gva: 0x1_0000,
            generation,
            width: 64,
            height: 64,
            bgra: true,
        }
    }

    fn armed(state: &mut DeviceState, host: &mut FakeHost, k: GvaTargetKey, gpas: &[u64]) {
        note_store(state, host, k, gpas);
    }

    /// The whole contract of the guest half: a Store stamps, a quiet guest reads
    /// back `Quiet`, and a guest write into a tracked page turns it into a
    /// refusal. A false `Quiet` is a stale frame that then persists, so the
    /// write case is the one that matters.
    #[test]
    fn a_guest_write_after_the_store_stops_the_target_reading_quiet() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE, 4 * PAGE];
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);

        host.guest_wrote_page(4 * PAGE);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::Wrote)
        );

        // A fresh Store re-stamps, so one guest write does not disable the
        // target for the rest of the boot.
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);
    }

    /// TEMP (lab, PRIVATE_GPU): a Private target's Store tracks nothing and arms
    /// the writer with no pages to name; readers read quiet until this device
    /// itself writes the pages; the resource's retirement drops the entry, and
    /// an ordinary Store of the same key afterwards tracks the pages afresh.
    #[test]
    fn a_contract_quiet_store_tracks_nothing_and_leaves_with_its_resource() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE, 4 * PAGE];
        note_store_contract_quiet(&mut state, key(0xabc), &pages, 7, 42);
        assert!(super::armed(&state, key(0xabc)));
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);
        assert_eq!(written_pages(&state, &host, key(0xabc)), Some(Vec::new()));

        let mut state = device();
        note_store_contract_quiet(&mut state, key(0xabc), &pages, 7, 42);
        assert_eq!(state.gva_store_witness.token_for_ref(7, 42), None);
        assert_eq!(
            state.gva_store_witness.take_tokens(),
            Vec::<u64>::new(),
            "no hypervisor set was armed"
        );

        let mut state = device();
        note_store_contract_quiet(&mut state, key(0xabc), &pages, 7, 42);
        forget_contract_ref(&mut state, 7, 42);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoMapping)
        );
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);
        host.guest_wrote_page(4 * PAGE);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::Wrote)
        );
    }

    /// TEMP (lab, `REIMS_VGPU_LAB_GVAW_LAGGED`): an eager re-stamp from the
    /// lagged generation still licenses readers — quiet until the guest writes,
    /// written after — but no longer arms a writer, whose page merge needs the
    /// Store's exact baseline. A synced Store arms it again.
    #[test]
    fn an_eager_restamp_serves_readers_but_not_writers() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE, 4 * PAGE];
        note_store_eager(&mut state, &mut host, key(0xabc), &pages);
        assert!(
            super::armed(&state, key(0xabc)),
            "a first Store reads synced"
        );

        note_store_eager(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);
        assert!(!super::armed(&state, key(0xabc)));
        assert_eq!(written_pages(&state, &host, key(0xabc)), None);

        host.guest_wrote_page(4 * PAGE);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::Wrote)
        );

        armed(&mut state, &mut host, key(0xabc), &pages);
        assert!(super::armed(&state, key(0xabc)));
        assert_eq!(written_pages(&state, &host, key(0xabc)), Some(Vec::new()));
    }

    /// TEMP (lab, `REIMS_VGPU_LAB_GVAW_LAGGED`): a deferral re-stamp of an
    /// undeclared object arms the writer and names the pages written after it;
    /// a declared object's deferral still reads synced.
    #[test]
    fn an_undeclared_deferral_restamp_arms_the_writer() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE, 4 * PAGE];
        note_store_deferral(&mut state, &mut host, key(0xabc), &pages, true);
        assert!(
            super::armed(&state, key(0xabc)),
            "a first Store reads synced"
        );

        note_store_deferral(&mut state, &mut host, key(0xabc), &pages, true);
        assert!(super::armed(&state, key(0xabc)));
        assert_eq!(written_pages(&state, &host, key(0xabc)), Some(Vec::new()));
        host.guest_wrote_page(4 * PAGE);
        assert_eq!(
            written_pages(&state, &host, key(0xabc)),
            Some(vec![4 * PAGE])
        );

        note_store_deferral(&mut state, &mut host, key(0xabc), &pages, false);
        assert!(super::armed(&state, key(0xabc)));
        assert_eq!(written_pages(&state, &host, key(0xabc)), Some(Vec::new()));
    }

    /// A write by another rail of this device is invisible to the hypervisor,
    /// and it is the second half. `host_writes` aliasing across id namespaces is
    /// real, so a page-exact ask is what separates "something wrote my span"
    /// from "something wrote somewhere".
    #[test]
    fn this_devices_own_write_into_the_span_stops_it_reading_quiet() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE, 4 * PAGE];
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);

        // Somewhere else: still quiet, because the ask is page-exact.
        state.note_host_wrote_pages(vec![9 * PAGE]);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);

        state.note_host_wrote_pages(vec![4 * PAGE]);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Host(HostWriteVerdict::Overlap)
        );
    }

    /// A writer that could not name its pages invalidates everything older —
    /// reading that as quiet is the wrong-frame direction.
    #[test]
    fn an_unnamed_host_write_refuses_rather_than_reading_quiet() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE];
        armed(&mut state, &mut host, key(0xabc), &pages);
        state.note_host_wrote_guest_ram();
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Host(HostWriteVerdict::Unnamed)
        );
    }

    /// A Store-to-LOAD interval is the longest reach in the device, and it is
    /// now answered **exactly** rather than refused for age.
    ///
    /// This used to assert `Aged` — the ring's horizon was what a deep reach
    /// hit, and refusing was the only safe answer available. The page-keyed
    /// record has no horizon, so depth alone no longer costs a re-gather. The
    /// half that must not change is the second one: a deep reach that really did
    /// land in these pages still refuses.
    #[test]
    fn a_reach_past_the_rings_horizon_is_answered_by_the_pages_and_not_by_its_depth() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE];

        armed(&mut state, &mut host, key(0xabc), &pages);
        for i in 0..256u64 {
            state.note_host_wrote_pages(vec![(100 + i) * PAGE]);
        }
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Quiet,
            "256 writes to other pages do not touch this window"
        );

        armed(&mut state, &mut host, key(0xabc), &pages);
        state.note_host_wrote_pages(vec![3 * PAGE]);
        for i in 0..256u64 {
            state.note_host_wrote_pages(vec![(100 + i) * PAGE]);
        }
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Host(HostWriteVerdict::Overlap),
            "and a write that did land here is not forgotten behind 256 later ones"
        );
    }

    /// The Store records its own write into these pages, so a witness that
    /// captured the host epoch before that would find its own Store in the ring
    /// and refuse forever. This is the ordering bug that would make the rail
    /// read zero on every boot with nothing saying why.
    #[test]
    fn a_store_does_not_invalidate_its_own_stamp() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE];
        // The order `store_gva_frame` uses: record the write, then stamp.
        state.note_host_wrote_pages(pages.to_vec());
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);
    }

    /// The guest recycling a GVA must not inherit the previous allocation's
    /// answer. The page-set hash is in the key, so the old entry is not merely
    /// out-voted — it cannot be reached at all.
    #[test]
    fn a_recycled_address_cannot_reach_the_old_allocations_entry() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE];
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(
            reach(&state, &host, key(0xdef)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoMapping),
            "a different page set at the same address is a different target"
        );
    }

    /// A host that cannot watch pages arms nothing, and a token still inside its
    /// arming window stamps 0. Neither may read as quiet by default.
    #[test]
    fn an_unwatchable_host_and_an_unarmed_token_both_refuse() {
        let mut state = device();
        let mut blind = FakeHost::new();
        blind.guest_writes_unobservable = true;
        armed(&mut state, &mut blind, key(0xabc), &[3 * PAGE]);
        assert_eq!(
            reach(&state, &blind, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoMapping),
            "nothing is inserted when the host cannot arm"
        );

        let mut warming = FakeHost::new();
        warming.guest_write_startup_window = true;
        armed(&mut state, &mut warming, key(0xabc), &[3 * PAGE]);
        assert_eq!(
            reach(&state, &warming, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoStamp)
        );
    }

    /// Past the bound the least recently stored target is evicted and its token
    /// handed back, rather than the map growing without limit or being dropped
    /// whole. A leaked token has the whole VM walking its pages at every
    /// doorbell for the life of the process.
    #[test]
    fn the_bound_evicts_the_coldest_target_and_hands_back_its_token() {
        let mut state = device();
        let mut host = FakeHost::new();
        for i in 0..(MAX_TRACKED_TARGETS as u64 + 4) {
            let k = GvaTargetKey {
                gva: 0x1000 * (i + 1),
                generation: 0xabc,
                width: 8,
                height: 8,
                bgra: true,
            };
            armed(&mut state, &mut host, k, &[(i + 3) * PAGE]);
        }
        assert!(state.gva_store_witness.len() <= MAX_TRACKED_TARGETS);
        assert_eq!(
            state.retired_guest_write_tokens.len(),
            4,
            "one token handed back per eviction"
        );
        assert_eq!(
            reach(
                &state,
                &host,
                GvaTargetKey {
                    gva: 0x1000,
                    generation: 0xabc,
                    width: 8,
                    height: 8,
                    bgra: true
                }
            ),
            GvaWriteReach::Guest(GuestWriteVerdict::NoMapping),
            "the coldest target is the one gone"
        );
    }

    /// Every armed token is handed back on reset. Anything else leaves the host
    /// dirty-logging those pages for the life of the process.
    #[test]
    fn every_token_is_handed_back_when_the_witness_is_taken() {
        let mut state = device();
        let mut host = FakeHost::new();
        armed(&mut state, &mut host, key(0xabc), &[3 * PAGE]);
        armed(
            &mut state,
            &mut host,
            GvaTargetKey {
                gva: 0x2_0000,
                ..key(0xabc)
            },
            &[4 * PAGE],
        );
        let freed = state.gva_store_witness.take_tokens();
        assert_eq!(freed.len(), 2);
        assert_eq!(state.gva_store_witness.len(), 0);
    }

    /// Pages a task teardown reclaimed take their targets with them: a GVA in a
    /// dead task names nothing, and its token must not outlive it.
    #[test]
    fn retiring_a_pages_target_hands_back_only_that_targets_token() {
        let mut state = device();
        let mut host = FakeHost::new();
        armed(&mut state, &mut host, key(0xabc), &[3 * PAGE]);
        let other = GvaTargetKey {
            gva: 0x2_0000,
            ..key(0xabc)
        };
        armed(&mut state, &mut host, other, &[4 * PAGE]);
        retire_pages(&mut state, &[3 * PAGE]);
        assert_eq!(state.retired_guest_write_tokens.len(), 1);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoMapping)
        );
        assert_eq!(
            reach(&state, &host, other),
            GvaWriteReach::Quiet,
            "another target's entry is untouched"
        );
    }

    /// A host whose deferred reads answer only when the test resolves them —
    /// the baseline thread, held still.
    struct DeferHost {
        inner: FakeHost,
        out: std::cell::RefCell<Vec<(u64, std::sync::Arc<std::sync::atomic::AtomicU64>)>>,
    }

    impl DeferHost {
        fn new() -> Self {
            Self {
                inner: FakeHost::new(),
                out: Default::default(),
            }
        }

        /// Answer every read still out with what a synced read says now.
        fn resolve(&self) {
            for (token, slot) in self.out.borrow_mut().drain(..) {
                let g = self.inner.guest_write_gen(token).unwrap_or(0);
                slot.store(g, std::sync::atomic::Ordering::Release);
            }
        }
    }

    impl HostOps for DeferHost {
        fn mono_ns(&self) -> u64 {
            0
        }
        fn enqueue(&mut self, action: crate::runtime::host::HostAction) {
            self.inner.enqueue(action);
        }
        fn schedule_bh(&mut self) {}
        fn track_guest_writes(&mut self, gpas: &[u64], page_size: usize) -> Option<u64> {
            self.inner.track_guest_writes(gpas, page_size)
        }
        fn untrack_guest_writes(&mut self, token: u64) {
            self.inner.untrack_guest_writes(token);
        }
        fn guest_write_gen(&self, token: u64) -> Option<u64> {
            self.inner.guest_write_gen(token)
        }
        fn guest_write_gen_lagged(&self, token: u64) -> Option<u64> {
            self.inner.guest_write_gen_lagged(token)
        }
        fn guest_write_gen_deferred(
            &self,
            token: u64,
        ) -> Option<crate::runtime::write_baseline::PendingGen> {
            let (pending, slot) = crate::runtime::write_baseline::PendingGen::unresolved();
            self.out.borrow_mut().push((token, slot));
            Some(pending)
        }
        fn guest_written_pages(&self, token: u64, since_gen: u64) -> Option<Vec<u64>> {
            self.inner.guest_written_pages(token, since_gen)
        }
    }

    /// A deferral's read runs off the drain: the target is armed at once, a
    /// reader while the read is out answers quiet (the packet's stamp is held,
    /// so the guest has not written it), and once the read resolves the usual
    /// generation test is back — a guest write after it reads written.
    #[test]
    fn a_deferred_read_arms_at_once_and_resolves_to_the_ordinary_test() {
        let mut state = device();
        let mut host = DeferHost::new();
        let pages = [3 * PAGE, 4 * PAGE];
        note_store_deferral(&mut state, &mut host, key(0xabc), &pages, false);
        assert!(super::armed(&state, key(0xabc)), "a read still out arms the writer");
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);

        host.resolve();
        assert!(super::armed(&state, key(0xabc)));
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);
        assert_eq!(written_pages(&state, &host, key(0xabc)), Some(vec![]));

        host.inner.guest_wrote_page(pages[1] + 8);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::Wrote)
        );
        assert_eq!(written_pages(&state, &host, key(0xabc)), Some(vec![pages[1]]));
    }

    /// A re-Store while the first read is still out keeps that read rather
    /// than taking the lagged answer of a set that has not been synced yet.
    #[test]
    fn a_restore_while_the_read_is_out_inherits_it() {
        let mut state = device();
        let mut host = DeferHost::new();
        let pages = [3 * PAGE];
        note_store_deferral(&mut state, &mut host, key(0xabc), &pages, false);
        note_store_deferral(&mut state, &mut host, key(0xabc), &pages, true);
        assert_eq!(host.out.borrow().len(), 1, "one read, inherited");
        assert!(super::armed(&state, key(0xabc)));
        host.resolve();
        assert!(super::armed(&state, key(0xabc)));
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);
    }

    /// A read that resolves unarmed is the arming window, exactly as a synced
    /// one: no writer licence, and readers refuse.
    #[test]
    fn a_read_that_resolves_unarmed_licenses_nothing() {
        let mut state = device();
        let mut host = DeferHost::new();
        note_store_deferral(&mut state, &mut host, key(0xabc), &[3 * PAGE], false);
        for (_, slot) in host.out.borrow_mut().drain(..) {
            slot.store(0, std::sync::atomic::Ordering::Release);
        }
        assert!(!super::armed(&state, key(0xabc)));
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoStamp)
        );
        assert_eq!(written_pages(&state, &host, key(0xabc)), None);
    }
}
