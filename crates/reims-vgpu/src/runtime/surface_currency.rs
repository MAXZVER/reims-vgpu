//! Whether a host-side copy of a mapper-ref-texture surface's pixels is still
//! that surface's content.
//!
//! # The question
//!
//! A mapper-ref-texture surface's pixels are plain guest RAM. The guest CPU
//! stores into them with no device operation at all, so nothing in this
//! device's own command stream marks the moment a host-side copy — a render
//! resident, the [`crate::runtime::surface_cache`] frame, an attachment LOAD
//! seed — stopped being the surface. The hypervisor's dirty bitmap is the only
//! witness, and [`crate::runtime::mapper::mapping_guest_write_verdict`] is how
//! this device reads it.
//!
//! Every consumer of a host-side copy has to ask this before serving one, and
//! there is exactly one right answer per `(mapping, window)` — so it is asked
//! here, once, and each rail names the outcome under its own census keys.
//!
//! # Why it is two stages and not one
//!
//! The tracking token covers the mapping's whole page list, and its generation
//! moves for a write to any page in it. A backing allocation is bigger than the
//! plane a bind samples: `mapper_ref_texture_sample_window` reports a `base_off`
//! precisely because the pixels do not start at offset 0, and an allocation can
//! carry a second plane and end padding past `span_end`. Refusing on the coarse
//! answer alone discarded whole 1920x1080 compositor scanouts whose pixels the
//! GPU had rendered and nothing had touched — measured live as a black desktop
//! at 17 Hz, against 120 Hz and a painted one on the same boot script with the
//! narrowing in place.
//!
//! So the coarse stage decides whether the page-list walk is worth paying for,
//! and the walk decides the answer. The walk is paid on the minority of asks
//! the coarse stage flags.
//!
//! # Which answers are evidence
//!
//! Only [`crate::runtime::mapper::GuestWriteVerdict::Wrote`] is evidence of a
//! guest write. `NoStamp` says "nobody asked the host to watch these pages",
//! which is a statement about this device's arming and not about the guest; on
//! the boot that first measured the ladder it was 14 092 of 14 396 cache binds,
//! so refusing on it would turn every host-side copy off on the strength of a
//! rail that was never armed.
//!
//! The narrowing fails closed in the other direction. Everything the host
//! cannot answer exactly — no token, no enumerable page list, no resolvable
//! sample window, or written GPAs this mapping does not own — is
//! [`SurfaceCurrency::WroteUnknown`], which does not serve. Serving a stale copy
//! is a wrong frame that is then *held*, because the pass composites onto the
//! seed and its Store publishes the composite back over the guest's pages;
//! re-reading the guest's pages costs a copy.

use crate::model::DeviceState;
use crate::protocol::pixel_format;
use crate::runtime::host::HostOps;
use crate::runtime::mapper::{self, GuestWriteVerdict};
use std::collections::BTreeMap;

/// What the host's dirty tracking says about one pixel window of one mapping,
/// since the Store that stamped this device's copies of it.
///
/// The serving rule is [`Self::serves`] and lives on the type, so a consumer
/// cannot spell it a fourth way: the three call sites that predate this each
/// wrote their own `!matches!(…)` and one of them had already drifted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfaceCurrency {
    /// The host has no evidence of a guest write to this allocation since the
    /// stamp. Carries the verdict that established it, because "the guest never
    /// rewrites its surfaces" and "the witness was never armed" are the same
    /// answer here and completely different findings.
    Unwritten(GuestWriteVerdict),
    /// The guest wrote the allocation, but every written page falls outside this
    /// pixel window — a header, a sibling plane, or end padding.
    WroteElsewhere,
    /// The guest wrote pages inside this pixel window. Carries the
    /// mapping-offset ranges the guest now owns, ascending and merged, which is
    /// exactly the `skip` list a merge that must preserve the guest's stores
    /// needs, and the generation those ranges were listed against.
    WrotePixels(GuestOwned),
    /// The guest wrote the allocation and the host cannot name where.
    /// Indistinguishable from [`Self::WrotePixels`] to a consumer that must be
    /// right, but it cannot be merged either — there is no page list to
    /// preserve.
    WroteUnknown,
}

/// Where the guest wrote, and the generation observed before that list was
/// taken.
///
/// A consumer that adopts the ranges into its copy stamps `seen` — never a
/// generation read afterwards, which could fold in a write the list did not
/// name (see [`mapper::GuestWriteObservation`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestOwned {
    pub ranges: Vec<(u64, u64)>,
    pub seen: mapper::GuestWriteObservation,
}

impl GuestOwned {
    /// Ranges with no observation behind them: commits as "no stamp".
    pub fn unobserved(ranges: Vec<(u64, u64)>) -> Self {
        Self {
            ranges,
            seen: mapper::GuestWriteObservation::default(),
        }
    }
}

/// The evidence a consumer requires of the witness before it will serve a
/// host-side copy.
///
/// Named at the ask rather than assumed, because the two standards are both
/// correct and choosing wrongly is invisible: on a pathway where the witness
/// never arms, every answer is
/// [`GuestWriteVerdict::NoStamp`], and the two standards then mean "serve
/// every time" and "serve never".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurrencyStandard {
    /// Absence of evidence to the contrary: serve unless the host actually
    /// watched a write land in this window.
    ///
    /// For a consumer that has other corroboration, or whose miss costs
    /// content the guest asked for. `NoStamp` says "nobody asked the host to
    /// watch these pages", which is a statement about this device's arming and
    /// not about the guest; on the boot that first measured the Vulkan sampled
    /// ladder it was 14 092 of 14 396 cache binds, so refusing on it would turn
    /// the rung off on the strength of a rail that was never armed.
    NoContraryEvidence,
    /// Positive evidence: the host watched these pages across the window and
    /// saw nothing written.
    ///
    /// For a consumer whose only check this is. An attachment LOAD seed served
    /// from a host-side copy has no rung under it to correct a stale serve —
    /// the pass composites onto the seed and its Store publishes the composite
    /// back over the guest's pages, so a wrong frame is *held*. Its miss costs
    /// a guest re-read and nothing else, which is the trade
    /// [`NoContraryEvidence`](Self::NoContraryEvidence) cannot make.
    WatchedAndUnwritten,
}

impl SurfaceCurrency {
    /// Whether a host-side copy of this window may be served as the surface's
    /// content, under the evidence standard the consumer states.
    ///
    /// [`Self::WroteElsewhere`] serves under both: the guest wrote the
    /// allocation and not these pixels, and discarding the copy on that is the
    /// 17 Hz black desktop the module doc records. The standards differ only on
    /// the answers that are not about the guest at all — no mapping, no stamp,
    /// an unreadable token — which
    /// [`CurrencyStandard::NoContraryEvidence`] admits and
    /// [`CurrencyStandard::WatchedAndUnwritten`] does not.
    pub fn serves(&self, standard: CurrencyStandard) -> bool {
        match standard {
            CurrencyStandard::NoContraryEvidence => {
                matches!(self, Self::Unwritten(_) | Self::WroteElsewhere)
            }
            CurrencyStandard::WatchedAndUnwritten => matches!(
                self,
                Self::Unwritten(GuestWriteVerdict::Clean) | Self::WroteElsewhere
            ),
        }
    }

    /// The coarse verdict this answer was reached under.
    ///
    /// Derived rather than stored twice: every variant but [`Self::Unwritten`]
    /// exists only because the verdict was
    /// [`GuestWriteVerdict::Wrote`], so a second field holding it could
    /// contradict the variant. Census that reports "which state was a copy
    /// served under" reads it here, so the coarse column keeps its meaning while
    /// the serving decision uses the narrowed one.
    pub fn verdict(&self) -> GuestWriteVerdict {
        match self {
            Self::Unwritten(verdict) => *verdict,
            Self::WroteElsewhere | Self::WrotePixels(_) | Self::WroteUnknown => {
                GuestWriteVerdict::Wrote
            }
        }
    }

    /// The mapping-offset ranges the guest's own stores now own, when the host
    /// could name them.
    ///
    /// Only [`Self::WrotePixels`] has them. [`Self::WroteUnknown`] deliberately
    /// answers `None` rather than an empty slice: an empty skip list reads as
    /// "the guest owns nothing", which is the opposite of what that variant
    /// means.
    pub fn guest_owned_ranges(&self) -> Option<&[(u64, u64)]> {
        match self {
            Self::WrotePixels(owned) => Some(owned.ranges.as_slice()),
            _ => None,
        }
    }

    /// The ranges together with the generation they were listed against, for a
    /// consumer that adopts them into its copy and must stamp what it saw.
    pub fn guest_owned(&self) -> Option<&GuestOwned> {
        match self {
            Self::WrotePixels(owned) => Some(owned),
            _ => None,
        }
    }
}

/// One currency answer's immutable inputs.
///
/// The observation epoch ([`HostOps::dirty_observation_epoch`]) lives on
/// [`SurfaceCurrencyCache`] rather than here: a guest doorbell invalidates every
/// entry at once, while a different geometry or a new Store is an identity
/// change for just this answer. (`dirty_sync_epoch` is not part of the key at
/// all since ABI v23: it is a profiling counter.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CurrencyKey {
    mapping_id: u32,
    map_generation: u32,
    content_generation: u32,
    surface_content_epoch: u32,
    guest_write_token: u64,
    guest_write_token_gen: u32,
    guest_write_gen_at_store: u64,
    width: u32,
    height: u32,
}

impl CurrencyKey {
    /// A cache answer is meaningful only for a live token and a Store stamp
    /// made against this exact page-list incarnation.
    fn from_mapping(state: &DeviceState, mapping_id: u32, width: u32, height: u32) -> Option<Self> {
        let m = state.mappings.get(&mapping_id)?;
        (m.guest_write_token != 0
            && m.guest_write_gen_at_store != 0
            && m.guest_write_token_gen == m.map_generation)
            .then_some(Self {
                mapping_id,
                map_generation: m.map_generation,
                content_generation: m.content_generation,
                surface_content_epoch: m.surface_content_epoch,
                guest_write_token: m.guest_write_token,
                guest_write_token_gen: m.guest_write_token_gen,
                guest_write_gen_at_store: m.guest_write_gen_at_store,
                width,
                height,
            })
    }
}

/// Currency answers observed during one host dirty-observation (doorbell)
/// epoch.
///
/// The cache never applies a drain boundary as proof that a dirty observation
/// did not happen. A background harvest may complete while one drain is still
/// active, and serving its previous `Clean` result after that is a stale-frame
/// bug. The QEMU epoch is the one witness with the required meaning.
///
/// Since ABI v23 that epoch is the guest doorbell epoch
/// ([`HostOps::dirty_observation_epoch`]), not the sync/harvest epoch. Every
/// entry was taken by [`observe_dirty_stably`] between two equal readings of
/// it, so the answer is complete for every doorbell up to `epoch`: a background
/// harvest inside the same doorbell epoch cannot falsify it (and no longer
/// forces a miss), while any new doorbell retires all of it.
#[derive(Debug, Default)]
pub struct SurfaceCurrencyCache {
    epoch: u64,
    entries: BTreeMap<CurrencyKey, SurfaceCurrency>,
}

/// What [`SurfaceCurrencyCache::lookup`] found, so each outcome can be counted
/// under its own route.
enum CacheLookup {
    /// An answer stably observed in this exact doorbell epoch.
    Hit(SurfaceCurrency),
    /// This key had an answer, but from an earlier doorbell epoch: a guest
    /// doorbell invalidated it (`surface_currency_doorbell_inval`).
    DoorbellInvalidated,
    /// No answer for this key (`surface_currency_cache_miss`).
    Miss,
}

impl SurfaceCurrencyCache {
    fn lookup(&mut self, epoch: u64, key: CurrencyKey) -> CacheLookup {
        if epoch == 0 || epoch < self.epoch {
            // No observation epoch, or one older than the entries held: those
            // entries belong to a later doorbell and say nothing about this ask.
            return CacheLookup::Miss;
        }
        if self.epoch != epoch {
            let had = self.entries.contains_key(&key);
            self.epoch = epoch;
            self.entries.clear();
            return if had {
                CacheLookup::DoorbellInvalidated
            } else {
                CacheLookup::Miss
            };
        }
        match self.entries.get(&key) {
            Some(verdict) => CacheLookup::Hit(verdict.clone()),
            None => CacheLookup::Miss,
        }
    }

    fn insert(&mut self, epoch: u64, key: CurrencyKey, verdict: SurfaceCurrency) {
        if epoch == 0 || epoch < self.epoch {
            // Never cache without an observation epoch, and never roll the
            // cache back to an epoch a later doorbell has already superseded.
            return;
        }
        if self.epoch != epoch {
            self.epoch = epoch;
            self.entries.clear();
        }
        self.entries.insert(key, verdict);
    }

    pub(crate) fn forget_mapping(&mut self, mapping_id: u32) {
        self.entries.retain(|key, _| key.mapping_id != mapping_id);
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.epoch = 0;
    }
}

/// Run a dirty-tracking query bracketed by [`HostOps::dirty_observation_epoch`].
///
/// Reads the observation epoch `e0`, runs `query`, and reads it again as `e1`.
/// Equal readings mean no guest doorbell landed during the query, so its answer
/// is complete for every doorbell up to `e0`: returns `Some((e0, answer))`.
///
/// If the epoch moved, records `retry_route` and runs `query` once more,
/// reading `e2` after it. `e1 == e2` returns `Some((e1, answer))`. If it moved
/// again, records `fallback_route` and returns `None`; the caller must then
/// answer conservatively and cache nothing (`SurfaceCurrency::WroteUnknown`,
/// or an unvouched gather).
///
/// An epoch of 0 (no observation epoch) is stable by this test; callers never
/// use 0 as a cache key.
pub fn observe_dirty_stably<M: HostOps + ?Sized, T>(
    host: &M,
    retry_route: &'static str,
    fallback_route: &'static str,
    mut query: impl FnMut(&M) -> T,
) -> Option<(u64, T)> {
    let e0 = host.dirty_observation_epoch();
    let first = query(host);
    let e1 = host.dirty_observation_epoch();
    if e0 == e1 {
        return Some((e0, first));
    }
    crate::runtime::drain::note_store_route(retry_route);
    let second = query(host);
    let e2 = host.dirty_observation_epoch();
    if e1 == e2 {
        return Some((e1, second));
    }
    crate::runtime::drain::note_store_route(fallback_route);
    None
}

/// Ask the hypervisor's witness whether this device's host-side copies of
/// `mapping_id`'s `width` x `height` pixel window are still that window.
///
/// `width`/`height` are the geometry the consumer is about to serve at, not the
/// mapping's latched geometry, because the window they resolve is what decides
/// [`SurfaceCurrency::WroteElsewhere`].
pub fn surface_currency<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> SurfaceCurrency {
    if !state.mappings.contains_key(&mapping_id) {
        return SurfaceCurrency::Unwritten(GuestWriteVerdict::NoMapping);
    }

    let key = CurrencyKey::from_mapping(state, mapping_id, width, height);
    let epoch_before = host.dirty_observation_epoch();
    if let Some(key) = key {
        let lookup = state
            .currency_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .lookup(epoch_before, key);
        match lookup {
            // A doorbell that landed after the lookup can carry a guest store.
            // Check it after dropping the cache lock, then re-evaluate rather
            // than serving the old verdict across that doorbell.
            CacheLookup::Hit(cached) if host.dirty_observation_epoch() == epoch_before => {
                crate::runtime::drain::note_store_route("surface_currency_stable_hit");
                return cached;
            }
            CacheLookup::Hit(_) | CacheLookup::DoorbellInvalidated => {
                crate::runtime::drain::note_store_route("surface_currency_doorbell_inval");
            }
            CacheLookup::Miss => {
                crate::runtime::drain::note_store_route("surface_currency_cache_miss");
            }
        }
    }

    // The verdict and its narrowing are one observation: both read the dirty
    // witness, and both must describe the same doorbell epoch.
    let observed = observe_dirty_stably(
        host,
        "surface_currency_retry",
        "surface_currency_conservative_fallback",
        |h| {
            let verdict = {
                let _span = crate::runtime::resolve_profile::Span::open(
                    crate::runtime::resolve_profile::Part::CurrencyVerdict,
                );
                mapper::mapping_guest_write_verdict(state, h, mapping_id)
            };
            if !matches!(verdict, GuestWriteVerdict::Wrote) {
                SurfaceCurrency::Unwritten(verdict)
            } else {
                narrow_to_window(state, h, mapping_id, width, height)
            }
        },
    );
    let Some((stable_epoch, currency)) = observed else {
        // Doorbells kept landing through the retry: no epoch this answer could
        // be held under. Conservative, and never cached.
        return SurfaceCurrency::WroteUnknown;
    };

    if let Some(key) = key {
        // Held under the epoch the observation was stable across — never one
        // read later, which could postdate a doorbell the answer did not see.
        state
            .currency_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(stable_epoch, key, currency.clone());
    }

    currency
}

/// The second stage: where the writes landed relative to the pixel window.
fn narrow_to_window<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> SurfaceCurrency {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return SurfaceCurrency::WroteUnknown;
    };
    let format = if m.format != 0 {
        m.format
    } else {
        pixel_format::MTL_FORMAT_BGRA8_UNORM
    };
    let Some((base_off, _bpr, span_end)) =
        crate::runtime::mapping_write::mapper_ref_texture_sample_window(m, width, height, format)
    else {
        return SurfaceCurrency::WroteUnknown;
    };
    // Observed before the list: `guest_written_pages` names every page moved
    // past the stamp up to at least this generation, so a copy that adopts the
    // list may stamp this and nothing later.
    let seen = {
        let _span = crate::runtime::resolve_profile::Span::open(
            crate::runtime::resolve_profile::Part::CurrencyObservation,
        );
        mapper::observe_current_guest_write_gen(state, host, mapping_id)
    };
    let Some(pages) = ({
        let _span = crate::runtime::resolve_profile::Span::open(
            crate::runtime::resolve_profile::Part::WrittenPages,
        );
        host.guest_written_pages(m.guest_write_token, m.guest_write_gen_at_store)
    }) else {
        return SurfaceCurrency::WroteUnknown;
    };
    let ranges = {
        let _span = crate::runtime::resolve_profile::Span::open(
            crate::runtime::resolve_profile::Part::PageOffsets,
        );
        mapper::mapping_offsets_of_pages(state, mapping_id, &pages)
    };
    if ranges.is_empty() {
        // The set-wide generation moved, so some page of this list was written,
        // yet none of them mapped back to an offset. That is a disagreement
        // between the token and the page list this call resolved against, not a
        // finding about the guest.
        return SurfaceCurrency::WroteUnknown;
    }
    if ranges_touch_window(&ranges, base_off, span_end) {
        SurfaceCurrency::WrotePixels(GuestOwned { ranges, seen })
    } else {
        SurfaceCurrency::WroteElsewhere
    }
}

/// Whether any mapping-offset range intersects `[base_off, span_end)`.
///
/// Both bounds half-open, so a range that abuts the plane's end is outside it.
fn ranges_touch_window(ranges: &[(u64, u64)], base_off: u64, span_end: u64) -> bool {
    ranges
        .iter()
        .any(|&(start, end)| start < span_end && end > base_off)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coarse stage admits everything that is not positive evidence.
    ///
    /// Both directions are asserted deliberately. Refusing more than `Wrote`
    /// would be just as wrong: `NoStamp` means this device never armed the
    /// witness, and turning every host-side copy off on that answer would send
    /// consumers to the guest's pages for surfaces whose content the deferred
    /// writeback rail has not landed there yet.
    #[test]
    fn only_a_watched_guest_write_refuses_a_host_side_copy() {
        for verdict in [
            GuestWriteVerdict::Clean,
            GuestWriteVerdict::NoMapping,
            GuestWriteVerdict::NoStamp,
            GuestWriteVerdict::Unreadable,
        ] {
            assert!(
                SurfaceCurrency::Unwritten(verdict).serves(CurrencyStandard::NoContraryEvidence),
                "{verdict:?} is not evidence of a guest write and must not refuse a copy"
            );
        }
        for state in [
            SurfaceCurrency::WrotePixels(GuestOwned::unobserved(vec![(0, 4096)])),
            SurfaceCurrency::WroteUnknown,
        ] {
            for standard in [
                CurrencyStandard::NoContraryEvidence,
                CurrencyStandard::WatchedAndUnwritten,
            ] {
                assert!(
                    !state.serves(standard),
                    "{state:?} is a write this device cannot rule out of the window"
                );
            }
        }
        for standard in [
            CurrencyStandard::NoContraryEvidence,
            CurrencyStandard::WatchedAndUnwritten,
        ] {
            assert!(
                SurfaceCurrency::WroteElsewhere.serves(standard),
                "a write outside the window leaves the window intact under {standard:?}"
            );
        }
    }

    /// The stricter standard admits only positive evidence, and the three
    /// answers it drops are exactly the ones that are about this device's arming
    /// rather than about the guest.
    ///
    /// This is the whole reason the standard is a parameter. A consumer whose
    /// only check this is cannot spend `NoStamp` — on a pathway where the
    /// hypervisor's witness never arms, every ask answers `NoStamp` and
    /// `NoContraryEvidence` degenerates into "serve whatever the cache holds".
    #[test]
    fn the_strict_standard_spends_only_a_watched_clean_answer() {
        assert!(SurfaceCurrency::Unwritten(GuestWriteVerdict::Clean)
            .serves(CurrencyStandard::WatchedAndUnwritten));
        for verdict in [
            GuestWriteVerdict::NoMapping,
            GuestWriteVerdict::NoStamp,
            GuestWriteVerdict::Unreadable,
        ] {
            assert!(
                !SurfaceCurrency::Unwritten(verdict).serves(CurrencyStandard::WatchedAndUnwritten),
                "{verdict:?} is not a watched answer and must not satisfy the strict standard"
            );
        }
    }

    /// Only the variant that can name the guest's bytes offers a skip list. An
    /// empty slice from `WroteUnknown` would read as "the guest owns nothing",
    /// which is the opposite of what it means.
    #[test]
    fn a_skip_list_comes_only_from_named_pages() {
        assert_eq!(
            SurfaceCurrency::WrotePixels(GuestOwned::unobserved(vec![(0, 4096)]))
                .guest_owned_ranges(),
            Some(&[(0u64, 4096u64)][..])
        );
        assert_eq!(SurfaceCurrency::WroteUnknown.guest_owned_ranges(), None);
        assert_eq!(SurfaceCurrency::WroteElsewhere.guest_owned_ranges(), None);
        assert_eq!(
            SurfaceCurrency::Unwritten(GuestWriteVerdict::Clean).guest_owned_ranges(),
            None
        );
    }

    /// The narrowing itself, which is what keeps the coarse stage from being
    /// ruinous. A backing allocation is bigger than the plane a bind samples —
    /// pixels start at `base_off` and padding follows `span_end` — and the
    /// tracking token's generation moves for a write to any page of it.
    #[test]
    fn a_guest_write_outside_the_window_leaves_the_window_current() {
        // A 1920x1080 BGRA8 plane one page into its allocation.
        const BASE: u64 = 4096;
        const END: u64 = BASE + 1920 * 1080 * 4;
        // The header page before the plane is not the pixels.
        assert!(!ranges_touch_window(&[(0, 4096)], BASE, END));
        // Nor is padding after it.
        assert!(!ranges_touch_window(&[(END + 4096, END + 8192)], BASE, END));
        // Abutting the end exactly is still outside — both bounds half-open.
        assert!(!ranges_touch_window(&[(END, END + 4096)], BASE, END));
        // One page anywhere inside the plane is the whole finding.
        assert!(ranges_touch_window(&[(4_198_400, 4_202_496)], BASE, END));
        // A range straddling the plane's first byte counts.
        assert!(ranges_touch_window(&[(0, 8192)], BASE, END));
        // Outside ranges do not mask an inside one.
        assert!(ranges_touch_window(
            &[(0, 4096), (4_198_400, 4_202_496), (END, END + 4096)],
            BASE,
            END
        ));
        assert!(!ranges_touch_window(&[], BASE, END));
    }

    #[test]
    fn surface_currency_reuses_every_verdict_only_within_the_same_dirty_observation_epoch() {
        use crate::model::DeviceId;
        use crate::protocol::gva::PAGE_SHIFT_X86;
        use crate::runtime::host::FakeHost;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();

        let mid = 42;
        assert!(state.map_surface(mid));
        let m = state.mappings.get_mut(&mid).unwrap();
        m.has_geom = true;
        m.width = 64;
        m.height = 64;
        m.format = pixel_format::MTL_FORMAT_BGRA8_UNORM;

        let pages: Vec<u64> = (0..4).map(|i| 0x1000_0000 + i * 4096).collect();
        let token = host.track_guest_writes(&pages, 4096).unwrap();
        let gen = host.guest_write_gen(token).unwrap();

        m.guest_write_token = token;
        m.guest_write_token_gen = m.map_generation;
        m.guest_write_gen_at_store = gen;

        // The first clean answer spends one generation query and is retained
        // under the host's current dirty observation (doorbell) epoch.
        let v1 = surface_currency(&state, &host, mid, 64, 64);
        assert_eq!(v1, SurfaceCurrency::Unwritten(GuestWriteVerdict::Clean));
        let clean_queries = host.guest_write_gen_calls.get();

        // A repeated bind of the same surface/window does not ask the host.
        assert_eq!(surface_currency(&state, &host, mid, 64, 64), v1);
        assert_eq!(host.guest_write_gen_calls.get(), clean_queries);

        // A background sync inside the same doorbell epoch is not a new
        // observation: it is profiling, and must not force a redundant query.
        host.advance_dirty_sync_epoch();
        assert_eq!(surface_currency(&state, &host, mid, 64, 64), v1);
        assert_eq!(host.guest_write_gen_calls.get(), clean_queries);

        // A doorbell ends that guarantee even if a drain happened not to end.
        // The next bind must re-evaluate; treating a drain epoch as a
        // substitute here would serve a pre-doorbell clean verdict.
        host.advance_dirty_observation_epoch();
        assert_eq!(surface_currency(&state, &host, mid, 64, 64), v1);
        assert_eq!(host.guest_write_gen_calls.get(), clean_queries + 1);

        // A guest write returns WroteUnknown for this deliberately minimal
        // mapping, and that refusal is cached too: omitting it makes each
        // repeated bind redo the WHPX generation and page-list work.
        host.guest_wrote_page(0x1000_0000);
        assert_eq!(
            surface_currency(&state, &host, mid, 64, 64),
            SurfaceCurrency::WroteUnknown
        );
        let wrote_queries = host.guest_write_gen_calls.get();
        assert_eq!(
            surface_currency(&state, &host, mid, 64, 64),
            SurfaceCurrency::WroteUnknown
        );
        assert_eq!(host.guest_write_gen_calls.get(), wrote_queries);

        // Geometry is part of the key, so a sibling plane cannot reuse the
        // first window's result even inside the same doorbell epoch.
        assert_eq!(
            surface_currency(&state, &host, mid, 128, 128),
            SurfaceCurrency::WroteUnknown
        );
        // A wrote verdict also needs one generation observation to narrow its
        // page list, so it may take more than the clean coarse-stage query. The
        // cache contract is that a same-window repeat takes none.
        let geometry_queries = host.guest_write_gen_calls.get();
        assert!(geometry_queries > wrote_queries);
        assert_eq!(
            surface_currency(&state, &host, mid, 128, 128),
            SurfaceCurrency::WroteUnknown
        );
        assert_eq!(host.guest_write_gen_calls.get(), geometry_queries);
    }

    /// A clean mapping tracked over four x86 pages, stamped at the current
    /// generation: the fixture every observation-epoch test below starts from.
    fn clean_tracked_mapping(
        state: &mut DeviceState,
        host: &mut crate::runtime::host::FakeHost,
        mid: u32,
    ) {
        assert!(state.map_surface(mid));
        let pages: Vec<u64> = (0..4).map(|i| 0x1000_0000 + i * 4096).collect();
        let token = host.track_guest_writes(&pages, 4096).unwrap();
        let gen = host.guest_write_gen(token).unwrap();
        let m = state.mappings.get_mut(&mid).unwrap();
        m.has_geom = true;
        m.width = 64;
        m.height = 64;
        m.format = pixel_format::MTL_FORMAT_BGRA8_UNORM;
        m.guest_write_token = token;
        m.guest_write_token_gen = m.map_generation;
        m.guest_write_gen_at_store = gen;
    }

    /// The bug the observation epoch exists to close: a guest doorbell that
    /// carries a store must retire a cached clean verdict at once, before any
    /// background harvest has advanced the sync epoch.
    #[test]
    fn doorbell_invalidates_cached_clean_verdict_before_background_harvest() {
        use crate::model::DeviceId;
        use crate::protocol::gva::PAGE_SHIFT_X86;
        use crate::runtime::drain::store_route_count;
        use crate::runtime::host::FakeHost;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let mid = 43;
        clean_tracked_mapping(&mut state, &mut host, mid);

        let v1 = surface_currency(&state, &host, mid, 64, 64);
        assert_eq!(v1, SurfaceCurrency::Unwritten(GuestWriteVerdict::Clean));
        let initial_obs = host.dirty_observation_epoch();
        let sync_before = host.dirty_sync_epoch();
        let queries = host.guest_write_gen_calls.get();
        let inval_before = store_route_count("surface_currency_doorbell_inval");

        // The doorbell lands with a store; the harvest has not run (the sync
        // epoch is not what tells the cache).
        host.note_doorbell_and_dirty_page(0x1000_0000);
        host.dirty_sync_epoch.set(sync_before);
        assert_ne!(host.dirty_observation_epoch(), initial_obs);
        assert_eq!(host.dirty_sync_epoch(), sync_before);

        let v2 = surface_currency(&state, &host, mid, 64, 64);
        assert_ne!(
            v2,
            SurfaceCurrency::Unwritten(GuestWriteVerdict::Clean),
            "a pre-doorbell clean verdict was served after the doorbell"
        );
        assert_eq!(v2, SurfaceCurrency::WroteUnknown);
        assert!(
            host.guest_write_gen_calls.get() > queries,
            "the witness was not re-asked"
        );
        assert!(store_route_count("surface_currency_doorbell_inval") > inval_before);
    }

    /// Background harvests inside one unchanged doorbell epoch must not cost a
    /// single further dirty query: that was the false-miss half of keying on
    /// the sync epoch.
    #[test]
    fn background_harvest_within_unchanged_doorbell_epoch_hits_cache_without_dirty_queries() {
        use crate::model::DeviceId;
        use crate::protocol::gva::PAGE_SHIFT_X86;
        use crate::runtime::drain::store_route_count;
        use crate::runtime::host::FakeHost;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let mid = 44;
        clean_tracked_mapping(&mut state, &mut host, mid);

        let v1 = surface_currency(&state, &host, mid, 64, 64);
        assert_eq!(v1, SurfaceCurrency::Unwritten(GuestWriteVerdict::Clean));
        let gen_queries = host.guest_write_gen_calls.get();
        let page_queries = host.guest_written_pages_calls.get();
        let obs = host.dirty_observation_epoch();
        let hits_before = store_route_count("surface_currency_stable_hit");

        for _ in 0..8 {
            host.advance_dirty_sync_epoch();
            assert_eq!(surface_currency(&state, &host, mid, 64, 64), v1);
        }
        assert_eq!(host.dirty_observation_epoch(), obs);
        assert_eq!(host.guest_write_gen_calls.get(), gen_queries);
        assert_eq!(host.guest_written_pages_calls.get(), page_queries);
        assert!(store_route_count("surface_currency_stable_hit") >= hits_before + 8);
    }

    /// The bracket itself: one retry when a doorbell lands mid-query, and a
    /// conservative `None` when one lands in the retry too.
    #[test]
    fn epoch_movement_during_query_retries_once_then_returns_conservative_uncached() {
        use crate::runtime::host::FakeHost;
        use std::cell::Cell;

        let host = FakeHost::new();
        let calls = Cell::new(0u32);

        // Stable: one query, answered under the epoch it was read in.
        let e = host.dirty_observation_epoch();
        let stable = observe_dirty_stably(&host, "test_obs_retry", "test_obs_fallback", |_| {
            calls.set(calls.get() + 1);
            7u32
        });
        assert_eq!(stable, Some((e, 7)));
        assert_eq!(calls.get(), 1);

        // Moves on the first attempt only: retried once, answered under the
        // epoch the retry was stable across.
        calls.set(0);
        let retried = observe_dirty_stably(&host, "test_obs_retry", "test_obs_fallback", |h| {
            let c = calls.get() + 1;
            calls.set(c);
            if c == 1 {
                h.advance_dirty_observation_epoch();
            }
            c
        });
        assert_eq!(calls.get(), 2);
        assert_eq!(retried, Some((host.dirty_observation_epoch(), 2)));

        // Moves on both attempts: no third query, and no answer.
        calls.set(0);
        let fell_back = observe_dirty_stably(&host, "test_obs_retry", "test_obs_fallback", |h| {
            calls.set(calls.get() + 1);
            h.advance_dirty_observation_epoch();
            42u32
        });
        assert_eq!(calls.get(), 2);
        assert!(fell_back.is_none());
    }

    /// A host whose doorbell epoch moves on every generation read: the worst
    /// case for the bracket, and the one that must reach the conservative arm.
    struct DoorbellStorm<'a> {
        inner: &'a crate::runtime::host::FakeHost,
    }

    impl HostOps for DoorbellStorm<'_> {
        fn mono_ns(&self) -> u64 {
            0
        }
        fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}
        fn schedule_bh(&mut self) {}
        fn guest_write_gen(&self, token: u64) -> Option<u64> {
            self.inner.advance_dirty_observation_epoch();
            self.inner.guest_write_gen(token)
        }
        fn guest_written_pages(&self, token: u64, since_gen: u64) -> Option<Vec<u64>> {
            self.inner.guest_written_pages(token, since_gen)
        }
        fn dirty_observation_epoch(&self) -> u64 {
            self.inner.dirty_observation_epoch()
        }
    }

    /// Through `surface_currency`: an epoch that never holds still answers
    /// `WroteUnknown` and caches nothing, so the next stable ask re-queries
    /// and gets the real (clean) answer.
    #[test]
    fn surface_currency_falls_back_to_uncached_wrote_unknown_when_the_epoch_never_settles() {
        use crate::model::DeviceId;
        use crate::protocol::gva::PAGE_SHIFT_X86;
        use crate::runtime::drain::store_route_count;
        use crate::runtime::host::FakeHost;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let mid = 45;
        clean_tracked_mapping(&mut state, &mut host, mid);

        let retry_before = store_route_count("surface_currency_retry");
        let fallback_before = store_route_count("surface_currency_conservative_fallback");
        let storm = DoorbellStorm { inner: &host };
        assert_eq!(
            surface_currency(&state, &storm, mid, 64, 64),
            SurfaceCurrency::WroteUnknown
        );
        assert!(store_route_count("surface_currency_retry") > retry_before);
        assert!(store_route_count("surface_currency_conservative_fallback") > fallback_before);

        // Nothing was cached: the settled host is asked again and answers clean.
        let queries = host.guest_write_gen_calls.get();
        assert_eq!(
            surface_currency(&state, &host, mid, 64, 64),
            SurfaceCurrency::Unwritten(GuestWriteVerdict::Clean)
        );
        assert!(host.guest_write_gen_calls.get() > queries);
    }
}
