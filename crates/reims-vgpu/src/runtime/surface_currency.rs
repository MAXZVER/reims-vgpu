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

/// Ask the hypervisor's witness whether this device's host-side copies of
/// `mapping_id`'s `width` x `height` pixel window are still that window.
///
/// `width`/`height` are the geometry the consumer is about to serve at, not the
/// mapping's latched geometry, because the window they resolve is what decides
/// [`SurfaceCurrency::WroteElsewhere`].
///
/// TEMP (lab, MONSTER_PREWARM): with the switch on, the ask is tranche-scoped —
/// see [`surface_currency_memo`]. The full ask below is unchanged and is what a
/// served entry was filled by, so the two can never disagree about how an
/// answer is reached.
pub fn surface_currency<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> SurfaceCurrency {
    if lab_monster_prewarm() {
        return surface_currency_memo(state, host, mapping_id, width, height);
    }
    surface_currency_ask(state, host, mapping_id, width, height)
}

/// The unmemoized ask: one hypervisor verdict, and on `Wrote` the page
/// enumeration and offset walk that narrow it to this window.
fn surface_currency_ask<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> SurfaceCurrency {
    let verdict = mapper::mapping_guest_write_verdict(state, host, mapping_id);
    let lab_miss = mapper::lab_v_take_verdict_miss(mapping_id); // TEMP (lab, verify)
    if !matches!(verdict, GuestWriteVerdict::Wrote) {
        return SurfaceCurrency::Unwritten(verdict);
    }
    let currency = narrow_to_window(state, host, mapping_id, width, height);
    if matches!(currency, SurfaceCurrency::WroteUnknown) && width >= 512 {
        // TEMP (lab, black-stripe hunt): the whole window is then re-read from guest pages.
        lab_note_wrote_pixels(mapping_id, width, height, &[], &[], 0);
    }
    if lab_miss {
        // TEMP (lab, verify): the lagged generation would have served this window's
        // host-side copy. Whether that was a stale serve depends on where the
        // unstated write landed.
        crate::runtime::drain::note_store_route(match currency {
            SurfaceCurrency::WrotePixels(_) => "lab_v_cur_miss_pixels",
            SurfaceCurrency::WroteUnknown => "lab_v_cur_miss_unknown",
            SurfaceCurrency::WroteElsewhere => "lab_v_cur_miss_elsewhere",
            SurfaceCurrency::Unwritten(_) => "lab_v_cur_miss_unwritten",
        });
    }
    currency
}

/// TEMP (lab, `REIMS_VGPU_LAB_MONSTER_PREWARM=on`, live `MONSTER_PREWARM=on|off`,
/// off unless `=on`).
///
/// # What the field data said the cold-bind class pays per bind
///
/// The n264/r0b instrumented boot (`lab_slow_tranche_phase`, journal
/// 2026-10-07-monster-tranche.md) read the relayout class (95 tranches, 1.85 s
/// of a 6.9 s slow-tranche boot) at **48 % `c_sampled` + 27 % `c_seed`** of its
/// chain time — and the seed *provides* were only 827 for the whole boot, ~1.3
/// ms each, far too few to own the bars. What recurs is the ask itself: this
/// function runs per sampled bind ([`sample_plan::surface_top`]) and per
/// attachment LOAD seed door, and for a surface the guest is actively writing
/// every ask pays the `Wrote` arm — a synced generation read plus a
/// `guest_written_pages` enumeration plus the offset walk over the surface's
/// page list. `seedx_currency_us` alone read **741 ms** for the boot.
///
/// # The memo, and why its freshness contract is the planner's
///
/// Within one tranche, binds of the same `(mapping, window)` re-ask the same
/// question against the same stamp. The memo holds the first ask's answer with
/// the mapping row it was reached under and the **lagged** generation read
/// right after it — exactly the pair the sampled-source plans carry as
/// `sample_plan::SampleReads::observed_gen` — and serves it again only while
/// all three hold:
///
/// * the tranche epoch has not moved (a new doorbell tranche re-asks once);
/// * every mapping row field the answer rests on is unchanged — a Store's
///   stamp, a token retirement, a page-list repoint each change one and are
///   each a different question;
/// * the host's lagged generation still equals the one read after the fill, so
///   no sync has observed a guest write since. A write no sync has observed
///   yet is a race the guest owns: the serial ask at that moment would be
///   served the same epoch's answer by the tracker (the planner's own
///   wording), and a later sync that does observe it moves the lagged value
///   and the next ask re-asks.
///
/// A served entry skips the synced read and, on the `Wrote` arm, the page
/// enumeration — the parts the field data priced. Anything else re-asks, and
/// the fresh answer replaces the entry, so a serve is never older than the
/// last full ask under the same row.
///
/// The memo is per thread: the drain thread and each pool worker pay their own
/// first ask per `(mapping, window)` per tranche, and no lock is added to a
/// path that holds none today.
fn surface_currency_memo<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> SurfaceCurrency {
    use crate::runtime::drain::note_store_route;
    let epoch = PREWARM_TRANCHE_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    let row = MemoRow::of(state, mapping_id);
    let key = (mapping_id, width, height);
    let lagged_now = (row.guest_write_token != 0)
        .then(|| host.guest_write_gen_lagged(row.guest_write_token))
        .flatten();
    let mut served = None;
    MEMO.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(entry) = m.get(&key) {
            if entry.epoch == epoch && entry.row == row && entry.lagged_then == lagged_now {
                note_store_route("prewarm_currency_serve");
                served = Some(entry.answer.clone());
            } else if entry.epoch != epoch {
                note_store_route("prewarm_currency_new_tranche");
                m.remove(&key);
            } else if entry.row != row {
                note_store_route("prewarm_currency_row_moved");
                m.remove(&key);
            } else {
                note_store_route("prewarm_currency_lagged_moved");
                m.remove(&key);
            }
        }
    });
    if let Some(answer) = served {
        return answer;
    }
    let answer = surface_currency_ask(state, host, mapping_id, width, height);
    // The lagged read comes after the fill, as the planner's does, so "no sync
    // observed a new write since the fill" is the exact claim a later serve
    // makes. The row is re-read for the same reason a stamp between the ask
    // and the insert must not vouch for the answer just reached.
    let row_after = MemoRow::of(state, mapping_id);
    let lagged_after = (row_after.guest_write_token != 0)
        .then(|| host.guest_write_gen_lagged(row_after.guest_write_token))
        .flatten();
    if row_after == row {
        note_store_route("prewarm_currency_ask");
        MEMO.with(|m| {
            m.borrow_mut().insert(
                key,
                MemoEntry {
                    epoch,
                    row,
                    lagged_then: lagged_after,
                    answer: answer.clone(),
                },
            );
        });
    } else {
        // The row moved during the ask (a concurrent stamp this thread did not
        // make does not happen on the drain thread, but a worker planning
        // beside it can read one): refuse to vouch and let the next ask fill.
        note_store_route("prewarm_currency_fill_refused_row_moved");
    }
    answer
}

/// TEMP (lab, MONSTER_PREWARM): the tranche the memo's entries were filled in.
/// Bumped once per drain tranche by [`prewarm_new_tranche`]; entries carry the
/// epoch they were filled at and never serve under a later one.
static PREWARM_TRANCHE_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// TEMP (lab, MONSTER_PREWARM): start a new tranche — every memo entry, on
/// every thread, stops serving. Called from the drain loop's tranche start, so
/// the answer a doorbell's first binds reach is always asked for that tranche.
pub fn prewarm_new_tranche() {
    PREWARM_TRANCHE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// TEMP (lab, MONSTER_PREWARM): the mapping-row fields an answer rests on.
/// Every one of them is a field the ask itself reads, and every one that can
/// change mid-tranche — the Store's stamp, a token retirement, a repoint, the
/// stated generation — changes the question the memo would be answering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemoRow {
    map_generation: u32,
    width: u32,
    height: u32,
    format: u16,
    guest_write_token: u64,
    guest_write_token_gen: u32,
    guest_write_gen_at_store: u64,
    guest_stated_gen: u32,
    guest_stated_at_stamp: u32,
}

impl MemoRow {
    /// The mapping's own current row; every field 0 when the mapping is gone,
    /// which is itself a row a memoized answer must not serve under.
    fn of(state: &DeviceState, mapping_id: u32) -> Self {
        let m = state.mappings.get(&mapping_id);
        Self {
            map_generation: m.map_or(0, |m| m.map_generation),
            width: m.map_or(0, |m| m.width),
            height: m.map_or(0, |m| m.height),
            format: m.map_or(0, |m| m.format),
            guest_write_token: m.map_or(0, |m| m.guest_write_token),
            guest_write_token_gen: m.map_or(0, |m| m.guest_write_token_gen),
            guest_write_gen_at_store: m.map_or(0, |m| m.guest_write_gen_at_store),
            guest_stated_gen: m.map_or(0, |m| m.guest_stated_gen),
            guest_stated_at_stamp: m.map_or(0, |m| m.guest_stated_at_stamp),
        }
    }
}

/// TEMP (lab, MONSTER_PREWARM): one asked-and-held answer.
#[derive(Clone, Debug)]
struct MemoEntry {
    epoch: u64,
    row: MemoRow,
    lagged_then: Option<u64>,
    answer: SurfaceCurrency,
}

thread_local! {
    /// TEMP (lab, MONSTER_PREWARM): this thread's held answers. Per thread
    /// because the drain thread and each pool worker ask independently; the
    /// epoch is shared so all of them retire together at the tranche boundary.
    static MEMO: std::cell::RefCell<std::collections::HashMap<(u32, u32, u32), MemoEntry>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// TEMP (lab, MONSTER_PREWARM): see [`surface_currency_memo`]. Off under test —
/// the memo itself is tested directly against [`surface_currency_ask`] so the
/// gate cannot hide a divergence between them.
fn lab_monster_prewarm() -> bool {
    if cfg!(test) {
        return false;
    }
    if let Some(on) = crate::runtime::lab_live::on("MONSTER_PREWARM") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = matches!(
            std::env::var("REIMS_VGPU_LAB_MONSTER_PREWARM").as_deref(),
            Ok("on")
        );
        crate::observe::off(format!("lab_monster_prewarm on={}", u8::from(on)));
        on
    })
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
    let seen = mapper::observe_current_guest_write_gen(state, host, mapping_id);
    let Some(pages) = host.guest_written_pages(m.guest_write_token, m.guest_write_gen_at_store)
    else {
        return SurfaceCurrency::WroteUnknown;
    };
    let ranges = mapper::mapping_offsets_of_pages(state, mapping_id, &pages);
    if ranges.is_empty() {
        // The set-wide generation moved, so some page of this list was written,
        // yet none of them mapped back to an offset. That is a disagreement
        // between the token and the page list this call resolved against, not a
        // finding about the guest.
        return SurfaceCurrency::WroteUnknown;
    }
    if ranges_touch_window(&ranges, base_off, span_end) {
        lab_note_wrote_pixels(mapping_id, width, height, &pages, &ranges, span_end - base_off);
        SurfaceCurrency::WrotePixels(GuestOwned { ranges, seen })
    } else {
        SurfaceCurrency::WroteElsewhere
    }
}

/// TEMP (lab, black-stripe hunt): one line per guest-written-pixels verdict on a
/// surface at least 512 wide (WebKit tiles), at most 3000 a boot: which pages the
/// witness says the guest wrote since the device's own Store, against the
/// window's size. A partial list on a GPU-rendered tile is the candidate for the
/// black rows (guest pages overlaid onto the resident).
fn lab_note_wrote_pixels(
    mapping_id: u32,
    width: u32,
    height: u32,
    pages: &[u64],
    ranges: &[(u64, u64)],
    window_bytes: u64,
) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static LEFT: AtomicU32 = AtomicU32::new(3000);
    if width < 512 || LEFT.fetch_sub(1, Ordering::Relaxed) == 0 {
        return;
    }
    let bytes: u64 = ranges.iter().map(|&(s, e)| e - s).sum();
    let first: Vec<String> = ranges
        .iter()
        .take(6)
        .map(|&(s, e)| format!("{:x}-{:x}", s, e))
        .collect();
    crate::observe::off(format!(
        "lab_wrote_pixels mid={mapping_id} {width}x{height} pages={} ranges={} bytes={bytes} window={window_bytes} first={}",
        pages.len(),
        ranges.len(),
        first.join(",")
    ));
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

    /// TEMP (lab, MONSTER_PREWARM): the fixture every memo test stands on — a
    /// mapped surface with a live token, stamped the way a Store leaves it, so
    /// the first currency ask answers `Unwritten(Clean)`.
    fn stamped_surface() -> (
        crate::model::DeviceState,
        crate::runtime::host::FakeHost,
        u64, /* the page GPA a guest write lands in */
    ) {
        use crate::model::{DeviceId, DeviceState};
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::host::{FakeHost, HostOps};

        let page_shift = crate::model::PAGE_SHIFT_X86;
        let mut state = DeviceState::new(DeviceId(1), page_shift);
        let mut host = FakeHost::new();
        let page = state.page_size() as u64;
        assert!(state.map_surface(7));
        let gpa = 0x40 * page;
        {
            let m = state.mappings.get_mut(&7).expect("mapped above");
            m.page_entries =
                vec![(((gpa >> page_shift) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let token = crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, 7)
            .expect("FakeHost observes guest writes");
        let stamped = host.guest_write_gen(token).expect("a live token has one");
        state
            .mappings
            .get_mut(&7)
            .expect("mapped above")
            .guest_write_gen_at_store = stamped;
        (state, host, gpa)
    }

    /// TEMP (lab, MONSTER_PREWARM): the second bind of the same window inside
    /// one tranche is the whole point of the memo — the field data priced the
    /// re-ask (a synced generation read, and on the `Wrote` arm a page
    /// enumeration) at the top of the cold-bind class's `c_sampled`+`c_seed`.
    ///
    /// The served answer must be the ask's own answer, and the second call must
    /// not pay for one: `ask` stays at one while `serve` moves.
    #[test]
    fn a_second_bind_of_the_same_window_serves_the_first_asks_answer() {
        let route = |k: &str| crate::runtime::drain::store_route_count(k);
        let ask0 = route("prewarm_currency_ask");
        let serve0 = route("prewarm_currency_serve");

        let (state, host, _gpa) = stamped_surface();
        let first = surface_currency_memo(&state, &host, 7, 64, 32);
        let second = surface_currency_memo(&state, &host, 7, 64, 32);

        assert_eq!(
            first, second,
            "a serve must repeat the ask's answer exactly"
        );
        assert_eq!(
            surface_currency_ask(&state, &host, 7, 64, 32),
            first,
            "the memo cannot disagree with the unmemoized ask on the same state"
        );
        assert_eq!(route("prewarm_currency_ask"), ask0 + 1, "one ask fills");
        assert!(
            route("prewarm_currency_serve") >= serve0 + 1,
            "the second bind serves"
        );

        // A different window of the same mapping is a different question: it
        // fills its own entry rather than borrowing this one's.
        surface_currency_memo(&state, &host, 7, 128, 32);
        assert_eq!(route("prewarm_currency_ask"), ask0 + 2, "per (mid, w, h)");
    }

    /// TEMP (lab, MONSTER_PREWARM): a guest write between two binds is the one
    /// event a memo must never fold over. The lagged generation moves with the
    /// host's observation of the write, the entry is dropped, and the second
    /// bind pays a fresh ask that answers `Wrote` — the same answer a boot
    /// without the memo would have reached.
    #[test]
    fn a_guest_write_between_binds_drops_the_held_answer() {
        use crate::runtime::host::HostOps;
        let route = |k: &str| crate::runtime::drain::store_route_count(k);
        let ask0 = route("prewarm_currency_ask");

        let (state, mut host, gpa) = stamped_surface();
        let first = surface_currency_memo(&state, &host, 7, 64, 32);
        assert!(matches!(first, SurfaceCurrency::Unwritten(_)));

        host.guest_wrote_page(gpa);
        let second = surface_currency_memo(&state, &host, 7, 64, 32);
        assert!(
            !first.serves(CurrencyStandard::NoContraryEvidence)
                || !second.serves(CurrencyStandard::NoContraryEvidence),
            "a write the host observed must reach the second bind: {second:?}"
        );
        assert_eq!(
            route("prewarm_currency_ask"),
            ask0 + 2,
            "the write re-asked"
        );
        assert!(
            route("prewarm_currency_lagged_moved") >= 1,
            "and says why the serve was refused"
        );
    }

    /// TEMP (lab, MONSTER_PREWARM): a Store's stamp between two binds changes
    /// the question — the stamp names the generation the answer is "since" — so
    /// the row check must refuse the held answer even though the token and its
    /// generation did not move.
    #[test]
    fn a_restamp_between_binds_drops_the_held_answer() {
        use crate::runtime::host::HostOps;
        let route = |k: &str| crate::runtime::drain::store_route_count(k);
        let ask0 = route("prewarm_currency_ask");

        let (mut state, host, _gpa) = stamped_surface();
        let token = state.mappings[&7].guest_write_token;
        let _ = surface_currency_memo(&state, &host, 7, 64, 32);

        // What `commit_guest_write_stamp` leaves behind: the same token, a new
        // generation the answer is held against.
        let restamped = host.guest_write_gen(token).expect("still live");
        state.mappings.get_mut(&7).unwrap().guest_write_gen_at_store = restamped + 7;
        surface_currency_memo(&state, &host, 7, 64, 32);

        assert_eq!(route("prewarm_currency_ask"), ask0 + 2, "a stamp re-asks");
        assert!(
            route("prewarm_currency_row_moved") >= 1,
            "and names the row, not the token, as what moved"
        );
    }

    /// TEMP (lab, MONSTER_PREWARM): a new tranche retires every held answer —
    /// the doorbell's first binds ask for themselves, so the memo can never
    /// carry an answer across the boundary the drain's accounting draws.
    #[test]
    fn a_new_tranche_retires_every_held_answer() {
        let route = |k: &str| crate::runtime::drain::store_route_count(k);
        let ask0 = route("prewarm_currency_ask");

        let (state, host, _gpa) = stamped_surface();
        let _ = surface_currency_memo(&state, &host, 7, 64, 32);
        prewarm_new_tranche();
        surface_currency_memo(&state, &host, 7, 64, 32);

        assert_eq!(
            route("prewarm_currency_ask"),
            ask0 + 2,
            "the tranche re-asked"
        );
        assert!(
            route("prewarm_currency_new_tranche") >= 1,
            "the retirement is named"
        );
    }
}
