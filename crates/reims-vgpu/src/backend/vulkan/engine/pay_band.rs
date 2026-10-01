//! The rows of a resident that its writes since the last payment could have
//! changed, and the pages of the packed frame those rows live in.
//!
//! # Why
//!
//! A presented frame is paid back into guest RAM by comparing the whole resident
//! against a shadow of what the guest's pages last received (`page_diff`): one
//! workgroup per 4 KiB page, 10 800 of them at 5K. The guest protocol names no
//! damage rectangle, but this device knows every write it recorded into the
//! resident since that payment, and on a window drag most of the frame was not
//! among them. The compare costs GPU time in proportion to the pages it is
//! dispatched over, so the pages no write could have reached need not be asked.
//!
//! # The invariant, and where it lives
//!
//! A [`RowBand`] says: *outside these rows, the image is byte-for-byte what it was
//! at the last payment.* [`PayDamage`] holds one per resident slot and is
//! widened by every door that writes the image. The slot is the owner because the
//! slot is the image: a new image is born [`PayDamage::Unknown`], and a resident
//! replaced under the same identity gets a new slot, so no band can outlive the
//! pixels it was kept for.
//!
//! Anything this module cannot account for is **full**: a write whose extent is
//! unknown, a clear or seed that rewrites the attachment, a slot nobody has paid,
//! a shadow other than the one the band was kept against. A band is only ever
//! narrower than the frame when every write since the payment is a known set of
//! rows. [`PayDamage::band_for`] is the one reader and compares the shadow's
//! payment epoch with the one the band stands against.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use super::page_diff::PAGE_BYTES;

/// Bytes per texel of the packed frame a payment compares: the page-diff kernel
/// reads four-byte scanout colour only, and `read_target_changed_pages_with`
/// declines every other resident before it asks for a band.
pub(crate) const PACKED_TEXEL_BYTES: u64 = 4;

/// The rows `[y0, y1)` of a frame that writes could have reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowBand {
    /// No row: the write rasterized nothing.
    Empty,
    /// Rows `y0..y1`, `y0 < y1`. Only [`Self::rows`] builds one, so an empty or
    /// inverted range is [`Self::Empty`] rather than a band that disagrees about
    /// whether it holds anything.
    Rows { y0: u32, y1: u32 },
    /// Every row, or a write whose extent is not known. The answer to doubt.
    Full,
}

impl RowBand {
    /// The rows `y0..y1`; no rows when the range is empty or inverted.
    pub(crate) fn rows(y0: u32, y1: u32) -> Self {
        if y0 < y1 {
            Self::Rows { y0, y1 }
        } else {
            Self::Empty
        }
    }

    /// The band holding everything either one does. A single interval: two
    /// disjoint writes cost the rows between them, which is the price of not
    /// keeping a list per slot.
    pub(crate) fn union(self, other: Self) -> Self {
        match (self, other) {
            (Self::Full, _) | (_, Self::Full) => Self::Full,
            (Self::Empty, band) | (band, Self::Empty) => band,
            (Self::Rows { y0: a0, y1: a1 }, Self::Rows { y0: b0, y1: b1 }) => Self::Rows {
                y0: a0.min(b0),
                y1: a1.max(b1),
            },
        }
    }

    /// [`Self::Full`] when these rows are every row of a `height`-row frame, so
    /// that "reached everything" has one spelling whichever way it was arrived at
    /// (a scissor the size of the target is as much a full write as a clear).
    pub(crate) fn normalized(self, height: u32) -> Self {
        match self {
            Self::Rows { y0: 0, y1 } if y1 >= height => Self::Full,
            band => band,
        }
    }

    /// How many of a `height`-row frame's rows this covers.
    pub(crate) fn covered_rows(self, height: u32) -> u32 {
        match self {
            Self::Empty => 0,
            Self::Full => height,
            Self::Rows { y0, y1 } => y1.min(height).saturating_sub(y0.min(height)),
        }
    }

    /// The pages of a packed `width` x `height` frame that overlap these rows,
    /// the partial pages at either edge included — a page is the compare's unit,
    /// and one holding a single band row is compared whole.
    pub(crate) fn pages(self, width: u32, height: u32) -> Range<u32> {
        let total = frame_pages(width, height);
        match self {
            Self::Empty => 0..0,
            Self::Full => 0..total,
            Self::Rows { y0, y1 } => {
                let (y0, y1) = (y0.min(height), y1.min(height));
                if y0 >= y1 {
                    return 0..0;
                }
                let pitch = u64::from(width) * PACKED_TEXEL_BYTES;
                let first = u64::from(y0) * pitch / PAGE_BYTES;
                let end = (u64::from(y1) * pitch).div_ceil(PAGE_BYTES);
                (first as u32)..(end.min(u64::from(total)) as u32)
            }
        }
    }
}

/// The rows one draw's render pass can have changed in its primary attachment.
///
/// The scissors bound the fragments only of a pass that **loaded** the target
/// and copied nothing into it first: a clear rewrites the render area (the whole
/// target), a seed copy fills the whole target, and a multisample resolve writes
/// the whole resolve attachment whatever the primitives covered. Each of those
/// is every row. `scissored` is asked only for the pass that is bounded by it.
pub(crate) fn pass_rows(
    loaded_target: bool,
    seed_copied: bool,
    resolves: bool,
    scissored: impl FnOnce() -> RowBand,
) -> RowBand {
    if loaded_target && !seed_copied && !resolves {
        scissored()
    } else {
        RowBand::Full
    }
}

/// The rows a set of copy rectangles, each given as `(y, height)` in texels,
/// land in. A rectangle above the image is a copy this does not understand, and
/// every row.
pub(crate) fn rows_of_rects(rects: impl IntoIterator<Item = (i32, u32)>) -> RowBand {
    rects
        .into_iter()
        .map(|(y, height)| match u32::try_from(y) {
            Ok(y) => RowBand::rows(y, y.saturating_add(height)),
            Err(_) => RowBand::Full,
        })
        .fold(RowBand::Empty, RowBand::union)
}

/// Pages in a packed `width` x `height` frame; the last may be partial.
pub(crate) fn frame_pages(width: u32, height: u32) -> u32 {
    (u64::from(width) * u64::from(height) * PACKED_TEXEL_BYTES).div_ceil(PAGE_BYTES) as u32
}

/// How a resident's image stands against the frame its page-diff shadow holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PayDamage {
    /// Nothing is known: a new image, a write nobody accounted for, or a payment
    /// that did not finish. The next payment compares every page.
    Unknown,
    /// The image equals the frame paid as `epoch`, except in `band`.
    Since { epoch: u64, band: RowBand },
}

impl PayDamage {
    /// The state of an image nobody has paid.
    pub(crate) const NEW: Self = Self::Unknown;

    /// A write into the image, reaching `band`. A write of every row — or one
    /// whose rows are not known — leaves nothing to narrow by.
    pub(crate) fn note_write(&mut self, band: RowBand) {
        *self = match (*self, band) {
            (Self::Since { epoch, band: held }, written) if written != RowBand::Full => {
                Self::Since {
                    epoch,
                    band: held.union(written),
                }
            }
            _ => Self::Unknown,
        };
    }

    /// The rows that may differ from the shadow whose payment epoch is `epoch`
    /// (`0`: the shadow was never paid). Everything unless the band was kept
    /// against exactly that payment.
    pub(crate) fn band_for(self, epoch: u64) -> RowBand {
        match self {
            Self::Since { epoch: held, band } if epoch != 0 && held == epoch => band,
            _ => RowBand::Full,
        }
    }

    /// The payment `epoch` has just made the shadow equal to the image.
    pub(crate) fn paid(&mut self, epoch: u64) {
        *self = Self::Since {
            epoch,
            band: RowBand::Empty,
        };
    }

    /// A payment began and may not have finished; the next one starts over.
    pub(crate) fn forget(&mut self) {
        *self = Self::Unknown;
    }
}

/// A payment epoch no earlier one had. `0` is "never paid".
pub(crate) fn next_epoch() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The most dispatches one payment records. Forced pages scattered over the
/// frame would otherwise cost one dispatch each; past this the nearest ranges
/// are joined, which compares the pages between them too and answers the same.
pub(crate) const MAX_DISPATCH_RANGES: usize = 16;

/// The page ranges one payment dispatches the compare over: the band's pages and
/// every page the caller must land whatever the compare says (`force`, which the
/// kernel also copies out). Ascending, disjoint, at most
/// [`MAX_DISPATCH_RANGES`]; empty when there is nothing to compare.
// A frame's one range of pages is a list of one range, not `0..total` pages.
#[allow(clippy::single_range_in_vec_init)]
pub(crate) fn dispatch_ranges(
    band: RowBand,
    width: u32,
    height: u32,
    force: &[u32],
) -> Vec<Range<u32>> {
    let total = frame_pages(width, height);
    let mut ranges: Vec<Range<u32>> = Vec::new();
    if band == RowBand::Full {
        return if total == 0 { ranges } else { vec![0..total] };
    }
    let band_pages = band.pages(width, height);
    if !band_pages.is_empty() {
        ranges.push(band_pages);
    }
    ranges.extend(force.iter().filter(|&&p| p < total).map(|&p| p..p + 1));
    ranges.sort_by_key(|r| r.start);
    let mut merged: Vec<Range<u32>> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match merged.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    while merged.len() > MAX_DISPATCH_RANGES {
        let join = (0..merged.len() - 1)
            .min_by_key(|&i| merged[i + 1].start - merged[i].end)
            .expect("more than one range");
        merged[join].end = merged[join + 1].end;
        merged.remove(join + 1);
    }
    merged
}

/// What one payment does with the band: the rows it narrows by, and the page
/// ranges it dispatches over.
///
/// The band says which rows a write since the last payment could have reached;
/// it speaks for a compare record (`prev`'s bytes, the hash words) only while
/// that record stands for the frame the last payment left. A payment that
/// rebuilds a record from this frame (`every_page`: a new shadow or `prev` copy,
/// a new or grown hash buffer, a record the last payment did not refresh) takes
/// every page whatever the band says, so its band is [`RowBand::Full`] and the
/// registry is not even asked. Otherwise a page outside the band was not written
/// since the last payment, the record of it is still right, and no compare need
/// look at it — which is why `PAY_BAND` and `PAY_HASH` compose: the same pages
/// are skipped whether the record is the `prev` frame or a hash.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BandPlan {
    /// The rows the compare may find changed (every row for a payment that takes
    /// every page, and with the gate off).
    pub band: RowBand,
    /// The page ranges to dispatch, with the gate `on` and only when every
    /// compare that runs reads the resident in place (`reads_image`); `None`
    /// dispatches every page.
    pub ranges: Option<Vec<Range<u32>>>,
}

impl BandPlan {
    /// `kept` reads the resident's band against the shadow's payment epoch; it
    /// is asked only when the band can narrow anything.
    pub(crate) fn new(
        mode: PayBandMode,
        every_page: bool,
        reads_image: bool,
        kept: impl FnOnce() -> RowBand,
        (width, height): (u32, u32),
        force: &[u32],
    ) -> Self {
        let band = if mode.tracks() && !every_page {
            kept()
        } else {
            RowBand::Full
        };
        let ranges = (mode == PayBandMode::On && reads_image)
            .then(|| dispatch_ranges(band, width, height, force));
        Self { band, ranges }
    }
}

/// What `REIMS_VGPU_LAB_PAY_BAND` asks of a payment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PayBandMode {
    /// Compare every page, as before. Nothing is tracked.
    Off,
    /// Compare the band's pages and the forced ones.
    On,
    /// Compare every page, as `Off` does, and count the changed pages the band
    /// did not contain. The census, not the payment, answers to the band.
    Verify,
}

impl PayBandMode {
    /// Unset and anything unrecognised is [`Self::Off`]: a typo narrows nothing.
    pub(crate) fn parse(raw: Option<&str>) -> Self {
        match raw.map(str::trim) {
            Some("on") => Self::On,
            Some("verify") => Self::Verify,
            _ => Self::Off,
        }
    }

    /// Whether widening a band is worth doing at all.
    pub(crate) fn tracks(self) -> bool {
        self != Self::Off
    }
}

/// The census of one payment's band, through the drain's store routes (counts
/// per second, summed over payments).
///
/// Coverage is `pay_band_rows / pay_band_height_rows` (a frame with no band to
/// narrow by counts as every row) and `pay_band_pages_dispatched /
/// pay_band_pages_total` for what the compare was actually dispatched over.
/// `pay_band_frames` is the payments that had a record to compare against;
/// a payment that takes every page (the first of a shadow, which copies the
/// whole frame, or one that seeds a hash record) is `pay_band_fresh`.
pub(crate) fn note_census(
    mode: PayBandMode,
    fresh: bool,
    image_kernel: bool,
    band: RowBand,
    dispatched: Option<&[Range<u32>]>,
    (width, height): (u32, u32),
    pages: u32,
) {
    use crate::runtime::drain::{note_store_route, note_store_route_n};
    if fresh {
        note_store_route("pay_band_fresh");
        return;
    }
    note_store_route("pay_band_frames");
    if band == RowBand::Full {
        note_store_route("pay_band_full");
    }
    if mode == PayBandMode::On && !image_kernel {
        // A band, but a kernel that compares a copy of the whole frame.
        note_store_route("pay_band_buffer_kernel");
    }
    note_store_route_n("pay_band_rows", u64::from(band.covered_rows(height)));
    note_store_route_n("pay_band_height_rows", u64::from(height));
    note_store_route_n(
        "pay_band_pages_band",
        u64::from(band.pages(width, height).len() as u32),
    );
    let compared: u64 = dispatched.map_or(u64::from(pages), |ranges| {
        ranges.iter().map(|r| u64::from(r.end - r.start)).sum()
    });
    note_store_route_n("pay_band_pages_dispatched", compared);
    note_store_route_n("pay_band_pages_total", u64::from(pages));
    COMPARED_PAGES.fetch_add(compared, Ordering::Relaxed);
    FRAME_PAGES.fetch_add(u64::from(pages), Ordering::Relaxed);
}

static COMPARED_PAGES: AtomicU64 = AtomicU64::new(0);
static FRAME_PAGES: AtomicU64 = AtomicU64::new(0);

/// `(pages the compare was dispatched over, pages of the frames it was asked
/// about)` summed over every tracked, non-fresh payment since the process began;
/// what the census reports per second, kept for the tests that read it back.
pub(crate) fn compared_pages_total() -> (u64, u64) {
    (
        COMPARED_PAGES.load(Ordering::Relaxed),
        FRAME_PAGES.load(Ordering::Relaxed),
    )
}

static OUTSIDE_CHANGED: AtomicU64 = AtomicU64::new(0);

/// `verify` found `pages` changed outside their payment's band: writes the band
/// did not account for. Counted in the census and, for the tests that must read
/// it back, in a process total.
pub(crate) fn note_outside_changed(pages: u64) {
    OUTSIDE_CHANGED.fetch_add(pages, Ordering::Relaxed);
    crate::runtime::drain::note_store_route_n("pay_band_outside_changed", pages);
}

/// Pages `verify` has found changed outside their band since the process began.
pub(crate) fn outside_changed_total() -> u64 {
    OUTSIDE_CHANGED.load(Ordering::Relaxed)
}

/// Pages of `changed` (ascending) that lie outside `inside`, and are not pages
/// the caller forced. Forced pages are excluded because the host pre-marks them
/// as changed whatever the image holds, so a mark on one proves nothing about
/// the band.
pub(crate) fn changed_outside(changed: &[u32], inside: &Range<u32>, force: &[u32]) -> u64 {
    changed
        .iter()
        .filter(|&&p| !inside.contains(&p) && force.binary_search(&p).is_err())
        .count() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_that_hold_nothing_are_empty() {
        assert_eq!(RowBand::rows(5, 5), RowBand::Empty);
        assert_eq!(RowBand::rows(9, 4), RowBand::Empty);
        assert_eq!(RowBand::rows(4, 9), RowBand::Rows { y0: 4, y1: 9 });
    }

    #[test]
    fn a_union_takes_the_hull_and_full_swallows_everything() {
        let a = RowBand::rows(10, 20);
        let b = RowBand::rows(40, 50);
        assert_eq!(a.union(b), RowBand::rows(10, 50));
        assert_eq!(b.union(a), RowBand::rows(10, 50));
        assert_eq!(a.union(RowBand::rows(12, 15)), a);
        assert_eq!(RowBand::Empty.union(a), a);
        assert_eq!(a.union(RowBand::Empty), a);
        assert_eq!(RowBand::Empty.union(RowBand::Empty), RowBand::Empty);
        assert_eq!(a.union(RowBand::Full), RowBand::Full);
        assert_eq!(RowBand::Full.union(RowBand::Empty), RowBand::Full);
    }

    #[test]
    fn rows_reaching_every_row_are_full() {
        assert_eq!(RowBand::rows(0, 480).normalized(480), RowBand::Full);
        assert_eq!(RowBand::rows(0, 900).normalized(480), RowBand::Full);
        assert_eq!(RowBand::rows(0, 479).normalized(480), RowBand::rows(0, 479));
        assert_eq!(RowBand::rows(1, 480).normalized(480), RowBand::rows(1, 480));
        assert_eq!(RowBand::Empty.normalized(480), RowBand::Empty);
        assert_eq!(RowBand::Full.normalized(480), RowBand::Full);
    }

    #[test]
    fn covered_rows_are_clipped_to_the_frame() {
        assert_eq!(RowBand::Empty.covered_rows(100), 0);
        assert_eq!(RowBand::Full.covered_rows(100), 100);
        assert_eq!(RowBand::rows(10, 30).covered_rows(100), 20);
        assert_eq!(RowBand::rows(90, 400).covered_rows(100), 10);
        assert_eq!(RowBand::rows(120, 400).covered_rows(100), 0);
    }

    #[test]
    fn only_a_loading_unseeded_unresolved_pass_is_bounded_by_its_scissors() {
        let scissored = || RowBand::rows(40, 60);
        assert_eq!(
            pass_rows(true, false, false, scissored),
            RowBand::rows(40, 60)
        );
        // A clear (or an undefined load), a seed copy and a resolve each rewrite
        // the whole attachment, and the scissors are not even asked.
        for (loaded, seeded, resolves) in [
            (false, false, false),
            (true, true, false),
            (true, false, true),
            (false, true, true),
        ] {
            let asked = std::cell::Cell::new(false);
            let band = pass_rows(loaded, seeded, resolves, || {
                asked.set(true);
                RowBand::rows(40, 60)
            });
            assert_eq!(band, RowBand::Full, "{loaded} {seeded} {resolves}");
            assert!(!asked.get());
        }
    }

    #[test]
    fn copy_rectangles_land_in_the_hull_of_their_rows() {
        assert_eq!(rows_of_rects([]), RowBand::Empty);
        assert_eq!(rows_of_rects([(10, 4)]), RowBand::rows(10, 14));
        assert_eq!(
            rows_of_rects([(10, 4), (200, 1), (50, 0)]),
            RowBand::rows(10, 201)
        );
        // A rectangle with no height lands nothing.
        assert_eq!(rows_of_rects([(7, 0)]), RowBand::Empty);
        // One that starts above the image is not understood.
        assert_eq!(rows_of_rects([(10, 4), (-1, 4)]), RowBand::Full);
    }

    #[test]
    fn a_band_before_any_payment_is_full() {
        let mut d = PayDamage::NEW;
        d.note_write(RowBand::rows(0, 4));
        assert_eq!(d.band_for(7), RowBand::Full);
        assert_eq!(PayDamage::NEW.band_for(0), RowBand::Full);
    }

    #[test]
    fn writes_since_a_payment_union_and_a_payment_resets_them() {
        let mut d = PayDamage::NEW;
        d.paid(7);
        assert_eq!(d.band_for(7), RowBand::Empty);
        d.note_write(RowBand::rows(100, 120));
        d.note_write(RowBand::rows(300, 310));
        d.note_write(RowBand::Empty);
        assert_eq!(d.band_for(7), RowBand::rows(100, 310));
        d.paid(8);
        assert_eq!(d.band_for(8), RowBand::Empty);
    }

    #[test]
    fn a_write_of_unknown_extent_widens_to_full_until_the_next_payment() {
        let mut d = PayDamage::NEW;
        d.paid(3);
        d.note_write(RowBand::rows(1, 2));
        d.note_write(RowBand::Full);
        assert_eq!(d.band_for(3), RowBand::Full);
        // A narrow write afterwards does not bring the band back: the image is
        // no longer the frame paid as 3 anywhere.
        d.note_write(RowBand::rows(1, 2));
        assert_eq!(d.band_for(3), RowBand::Full);
        d.paid(4);
        assert_eq!(d.band_for(4), RowBand::Empty);
    }

    #[test]
    fn a_band_is_only_read_against_the_payment_it_was_kept_for() {
        let mut d = PayDamage::NEW;
        d.paid(5);
        d.note_write(RowBand::rows(1, 2));
        // The shadow has been paid since by another resident, or replaced.
        assert_eq!(d.band_for(6), RowBand::Full);
        assert_eq!(d.band_for(0), RowBand::Full);
        assert_eq!(d.band_for(5), RowBand::rows(1, 2));
    }

    #[test]
    fn an_unfinished_payment_forgets_the_band() {
        let mut d = PayDamage::NEW;
        d.paid(5);
        d.forget();
        assert_eq!(d.band_for(5), RowBand::Full);
    }

    #[test]
    fn epochs_never_repeat_and_are_never_zero() {
        let a = next_epoch();
        let b = next_epoch();
        assert!(a != 0 && b != 0 && a != b);
    }

    #[test]
    fn rows_map_to_the_pages_that_hold_them() {
        // 1024 texels x 4 bytes is exactly one page per row.
        assert_eq!(RowBand::rows(3, 5).pages(1024, 100), 3..5);
        assert_eq!(RowBand::rows(0, 1).pages(1024, 100), 0..1);
        assert_eq!(RowBand::rows(99, 100).pages(1024, 100), 99..100);
        // Rows past the frame clip; nothing is left when the band starts past it.
        assert_eq!(RowBand::rows(98, 500).pages(1024, 100), 98..100);
        assert_eq!(RowBand::rows(100, 500).pages(1024, 100), 0..0);
        assert_eq!(RowBand::Empty.pages(1024, 100), 0..0);
        assert_eq!(RowBand::Full.pages(1024, 100), 0..100);
    }

    #[test]
    fn a_band_edge_inside_a_page_takes_the_whole_page() {
        // 1000 texels = 4000 bytes a row, so row 1 is bytes 4000..8000: it starts
        // in page 0 and ends in page 1, and row 2 (8000..12000) in pages 1 and 2.
        assert_eq!(RowBand::rows(1, 2).pages(1000, 64), 0..2);
        assert_eq!(RowBand::rows(2, 3).pages(1000, 64), 1..3);
        // Two rows: 4000..12000 -> pages 0..3.
        assert_eq!(RowBand::rows(1, 3).pages(1000, 64), 0..3);
        // A row that ends exactly on a page boundary does not take the next page.
        // 1024 texels: row 2 is 8192..12288, pages 2..3.
        assert_eq!(RowBand::rows(2, 3).pages(1024, 8), 2..3);
        // Rows narrower than a page share one: 100 texels = 400 bytes a row,
        // ten and a bit rows per page.
        assert_eq!(RowBand::rows(0, 1).pages(100, 100), 0..1);
        assert_eq!(RowBand::rows(10, 11).pages(100, 100), 0..2);
        assert_eq!(RowBand::rows(11, 12).pages(100, 100), 1..2);
    }

    #[test]
    fn the_last_partial_page_is_in_the_band_that_reaches_the_last_row() {
        // 100 x 100 x 4 = 40 000 bytes = 9.77 pages -> 10 pages.
        assert_eq!(frame_pages(100, 100), 10);
        assert_eq!(RowBand::rows(99, 100).pages(100, 100), 9..10);
        assert_eq!(RowBand::Full.pages(100, 100), 0..10);
    }

    #[test]
    fn a_5k_frame_has_the_pages_the_payment_dispatches_today() {
        assert_eq!(frame_pages(5120, 2160), 10_800);
        // One 5K row is 20 480 bytes: five whole pages.
        assert_eq!(RowBand::rows(0, 1).pages(5120, 2160), 0..5);
        assert_eq!(RowBand::rows(2159, 2160).pages(5120, 2160), 10_795..10_800);
    }

    #[test]
    fn dispatch_covers_the_band_and_every_forced_page() {
        // One page per row.
        assert_eq!(
            dispatch_ranges(RowBand::rows(10, 20), 1024, 100, &[]),
            vec![10..20]
        );
        // A forced page inside the band adds nothing; one outside is its own range.
        assert_eq!(
            dispatch_ranges(RowBand::rows(10, 20), 1024, 100, &[12, 50, 51, 99]),
            vec![10..20, 50..52, 99..100]
        );
        // A forced page touching the band joins it.
        assert_eq!(
            dispatch_ranges(RowBand::rows(10, 20), 1024, 100, &[20, 9]),
            vec![9..21]
        );
        // No band, no forced pages: nothing to compare.
        assert!(dispatch_ranges(RowBand::Empty, 1024, 100, &[]).is_empty());
        // No band but a forced page: that page alone.
        assert_eq!(
            dispatch_ranges(RowBand::Empty, 1024, 100, &[42]),
            vec![42..43]
        );
        // Forced pages past the frame are ignored.
        assert!(dispatch_ranges(RowBand::Empty, 1024, 100, &[100, 500]).is_empty());
    }

    #[test]
    fn a_full_band_dispatches_the_whole_frame_once() {
        assert_eq!(
            dispatch_ranges(RowBand::Full, 1024, 100, &[3, 4, 90]),
            vec![0..100]
        );
        assert!(dispatch_ranges(RowBand::Full, 0, 0, &[]).is_empty());
    }

    #[test]
    fn scattered_forced_pages_are_joined_at_the_smallest_gaps() {
        // Eighteen isolated pages, mostly 10 apart, one pair only 2 apart.
        let mut force: Vec<u32> = (0..17).map(|i| i * 10).collect();
        force.push(162); // 160 and 162: gap of 1 page between them.
        let ranges = dispatch_ranges(RowBand::Empty, 1024, 400, &force);
        assert_eq!(ranges.len(), MAX_DISPATCH_RANGES);
        // Every forced page is still covered.
        for p in &force {
            assert!(ranges.iter().any(|r| r.contains(p)), "page {p} lost");
        }
        // The pair with the smallest gap is the one that was joined.
        assert!(ranges.iter().any(|r| r.contains(&160) && r.contains(&162)));
        // Ascending and disjoint.
        for w in ranges.windows(2) {
            assert!(w[0].end <= w[1].start);
        }
    }

    #[test]
    fn the_gate_reads_on_and_verify_and_nothing_else() {
        assert_eq!(PayBandMode::parse(None), PayBandMode::Off);
        assert_eq!(PayBandMode::parse(Some("off")), PayBandMode::Off);
        assert_eq!(PayBandMode::parse(Some("1")), PayBandMode::Off);
        assert_eq!(PayBandMode::parse(Some("ON")), PayBandMode::Off);
        assert_eq!(PayBandMode::parse(Some("on")), PayBandMode::On);
        assert_eq!(PayBandMode::parse(Some(" verify ")), PayBandMode::Verify);
        assert!(!PayBandMode::Off.tracks());
        assert!(PayBandMode::On.tracks() && PayBandMode::Verify.tracks());
    }

    /// The plan for a 1024x100 frame (one page per row) of a payment whose band
    /// is rows 10..20, and whether the registry was asked for it.
    fn plan(
        mode: PayBandMode,
        every_page: bool,
        reads_image: bool,
        force: &[u32],
    ) -> (BandPlan, bool) {
        let asked = std::cell::Cell::new(false);
        let plan = BandPlan::new(
            mode,
            every_page,
            reads_image,
            || {
                asked.set(true);
                RowBand::rows(10, 20)
            },
            (1024, 100),
            force,
        );
        (plan, asked.get())
    }

    #[test]
    fn with_the_gate_off_the_band_is_not_asked_and_nothing_narrows() {
        let (p, asked) = plan(PayBandMode::Off, false, true, &[3]);
        assert!(!asked);
        assert_eq!(p.band, RowBand::Full);
        assert_eq!(p.ranges, None);
    }

    #[test]
    fn verify_reads_the_band_but_dispatches_everything() {
        let (p, asked) = plan(PayBandMode::Verify, false, true, &[]);
        assert!(asked);
        assert_eq!(p.band, RowBand::rows(10, 20));
        assert_eq!(p.ranges, None, "verify compares every page");
    }

    #[test]
    fn on_dispatches_the_band_and_the_forced_pages_of_an_image_reading_compare() {
        // The same plan whichever record the compare keeps (`prev` or hashes):
        // the band does not know which.
        let (p, asked) = plan(PayBandMode::On, false, true, &[50]);
        assert!(asked);
        assert_eq!(p.band, RowBand::rows(10, 20));
        assert_eq!(p.ranges, Some(vec![10..20, 50..51]));
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn a_payment_that_rebuilds_a_record_takes_every_page_and_never_asks_for_a_band() {
        // A hash record being seeded, or a `prev` being copied: the stored band
        // speaks for neither, and every page is compared and landed.
        for mode in [PayBandMode::On, PayBandMode::Verify] {
            let (p, asked) = plan(mode, true, true, &[50]);
            assert!(!asked, "{mode:?}");
            assert_eq!(p.band, RowBand::Full, "{mode:?}");
            assert_eq!(
                p.ranges,
                (mode == PayBandMode::On).then(|| vec![0..100]),
                "{mode:?}"
            );
        }
    }

    #[test]
    fn a_buffer_kernel_is_never_started_part_way() {
        let (p, asked) = plan(PayBandMode::On, false, false, &[]);
        assert!(asked, "the band is still kept and counted");
        assert_eq!(p.band, RowBand::rows(10, 20));
        assert_eq!(p.ranges, None);
    }

    #[test]
    fn changed_pages_outside_the_band_ignore_forced_ones() {
        let inside = 10..20;
        assert_eq!(changed_outside(&[10, 15, 19], &inside, &[]), 0);
        assert_eq!(changed_outside(&[9, 10, 20, 30], &inside, &[]), 3);
        // Forced pages are marked by the host, not found by the compare.
        assert_eq!(changed_outside(&[9, 10, 20, 30], &inside, &[9, 30]), 1);
        assert_eq!(changed_outside(&[], &inside, &[]), 0);
    }
}
