//! TEMP (lab, `REIMS_VGPU_LAB_STRIPE_HUNT=on`, default off): per-present
//! accounting for the one-frame black-row ("stripe") artifact class.
//!
//! # What this hunts
//!
//! The class: for exactly one presented frame, a cluster of fully-black
//! horizontal rows inside an otherwise-correct frame (ps0b frame 100245: 29
//! runs, y 466-703, x 1038-1522, all texels exactly zero), self-healed the next
//! frame. The audit (`lab/journal/2026-10-05-stripe-hunt.md`) ranked two
//! co-equal candidate families and this module names both per present:
//!
//! 1. **Unseeded partial-preserving passes** — a pass whose declared load
//!    preserves, that found no seed door and no resident, keys "no seed" and
//!    leaves every texel outside its damage scissor holding undefined/zero
//!    content (`draw_partial_preserving_unseeded`, the census's own "defect
//!    arm"). One presented frame of those rows is the artifact; the pass's own
//!    Store re-establishes the resident, so the *next* frame preserves and the
//!    artifact heals. The instrument records the scissor **y-ranges** of such
//!    draws on display-sized guest-backed targets.
//! 2. **Torn adoption / false-quiet skips of guest mid-update rows** — the
//!    overlay adopting page-granular guest-owned ranges at an instant inside
//!    the guest's update, and gather binds vouched on a harvest that predated a
//!    guest store. The instrument records overlay row-ranges (the ranges the
//!    guest owned, in rows) and the vouched-skip/fresh-gather balance.
//!
//! Beside them, three context columns make one repro boot's fail log enough to
//! pin the path without a second instrument: declared DontCare/Clear pixels
//! (what the pass *asked* for), batch-tail reopens (a clear/covers-draw pair
//! split across presents), and resident re-create/resample events (the fresh
//! zero-image population).
//!
//! # The rules it obeys
//!
//! `runtime/census/README.md`: measuring is allowed, branching on the
//! measurement is not. Every entry point here is write-only accounting behind
//! the `on()` gate; nothing reads a value back to decide what to present,
//! decode or execute. All output is `observe::off` (the lab channel), one line
//! per present, so a normal boot sees zero lines.
//!
//! The line is per present and not deduplicated: a repro boot is its
//! denominator. Every bounded list carries its overflow count so a full window
//! still totals.
//!
//! # Reading the line
//!
//! ```text
//! OFF stripe_hunt present mid=1 5120x2160 dc_px=0 clear_px=0 \
//!   unseeded=2 lost_px=430000 rows=[481-489,502-526] mids=[91,100] more=0 \
//!   elide=12 elide_px=26214400 provided=0 skip=31 skip_kb=36 gather=0 \
//!   overlay=1 overlay_rows=[932-1406] reopen=2 recreate=3 t=100245
//! ```
//!
//! Join on `t=` against the frame timestamps. A stripe frame whose present
//! line carries `unseeded rows=` matching the black rows convicts family 1;
//! `overlay_rows=` or a `skip=`-heavy window with `gather=0` on the text
//! layer's surface convicts family 2.

use std::sync::Mutex;
use std::sync::OnceLock;

/// `true` only with `REIMS_VGPU_LAB_STRIPE_HUNT=on`. Read once; every hook
/// checks this before touching the accumulator, so the off-mode cost of the
/// whole module is one boolean load per hook site.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_STRIPE_HUNT").as_deref() == Ok("on"))
}

/// How many packed y-ranges / mids a window keeps before counting the rest.
const RANGE_CAP: usize = 8;

/// The accounting for one present-to-present window. All fields are deltas
/// reset by [`Self::take_present_line`]; nothing here survives a present.
#[derive(Default)]
struct Hunt {
    /// Declared DontCare px on colour-0 (the widened preserve population).
    dc_px: u64,
    /// Declared Clear px on colour-0.
    clear_px: u64,
    /// TEMP (iter 3, white stripes): px of **partial-scissor CLEAR passes
    /// whose clear colour is white** — the only CPU-visible member of the
    /// white-row family (a guest clear + a cover that lands late leaves the
    /// cleared rows white). The other white source — a recycled image whose
    /// stale rows were a white surface, read by an unseeded preserve — lives
    /// GPU-side and cannot be seen here without a content probe the census
    /// rules forbid; the frame scanners own that half.
    clear_white_px: u64,
    /// Partial preserving-unseeded draws on guest-backed targets.
    unseeded: u32,
    /// Texels those draws left holding undefined content (attachment minus
    /// scissor), the unit the census prices the same arm in.
    unseeded_lost_px: u64,
    /// Clamped scissor y-ranges of those draws (start, end-exclusive), merged
    /// and packed; `unseeded_more` counts the ones that did not fit.
    unseeded_rows: Vec<(u32, u32)>,
    unseeded_more: u32,
    /// The mapping ids the unseeded draws targeted, deduplicated, capped.
    unseeded_mids: Vec<u32>,
    /// Mapper-ref seed elisions (resident carried the LOAD without a seed) and
    /// the px they covered.
    elide: u32,
    elide_px: u64,
    /// Seeds actually provided (the elide refused — the torn-adoption window).
    provided: u32,
    /// Gather binds vouched quiet (retained image served, no gather) and the
    /// KB they covered, beside fresh gathers — the false-quiet population.
    skip: u32,
    skip_kb: u64,
    gather: u32,
    /// Overlay adoptions of guest-owned ranges: count, and the row-ranges laid
    /// over the resident, in target rows.
    overlay: u32,
    overlay_rows: Vec<(u32, u32)>,
    overlay_more: u32,
    /// Batch-tail flush-then-reopens since the previous present.
    reopen: u32,
    /// Resident re-create/resample/evict events.
    recreate: u32,
    /// TEMP (lab, OVERLAY_GUARD): adoptions the post-read guard dropped — the
    /// suspected stripe mechanism caught in the act, one per drop.
    guard_drop: u32,
}

impl Hunt {
    /// The all-zero window, as a `const fn` so the process-wide static can be
    /// built without `Default`'s non-const trait fn.
    const fn zero() -> Self {
        Self {
            dc_px: 0,
            clear_px: 0,
            clear_white_px: 0,
            unseeded: 0,
            unseeded_lost_px: 0,
            unseeded_rows: Vec::new(),
            unseeded_more: 0,
            unseeded_mids: Vec::new(),
            elide: 0,
            elide_px: 0,
            provided: 0,
            skip: 0,
            skip_kb: 0,
            gather: 0,
            overlay: 0,
            overlay_rows: Vec::new(),
            overlay_more: 0,
            reopen: 0,
            recreate: 0,
            guard_drop: 0,
        }
    }

    /// Merge `(start, end)` into `rows`, keeping the list sorted, coalescing
    /// ranges that touch, and counting (into `more`) what does not fit the
    /// cap. Row indices are clamped to `height` by the callers.
    fn push_range(rows: &mut Vec<(u32, u32)>, more: &mut u32, start: u32, end: u32) {
        if start >= end {
            return;
        }
        // Touching or overlapping neighbours merge; the guest's glyph-row
        // updates arrive as many 1-page ranges that often sit adjacent.
        let mut merged = (start, end);
        let mut i = 0;
        while i < rows.len() {
            let (rs, re) = rows[i];
            if merged.0 <= re && rs <= merged.1 {
                merged = (merged.0.min(rs), merged.1.max(re));
                rows.remove(i);
            } else {
                i += 1;
            }
        }
        if rows.len() >= RANGE_CAP {
            *more += 1;
            return;
        }
        let at = rows.partition_point(|&(s, _)| s <= merged.0);
        rows.insert(at, merged);
    }

    /// Byte ranges within a mapping window, as (start_row, end_row) pairs over
    /// a stride of `bpr` bytes per row. A 4 KiB page over a 3880-byte row is
    /// 1.05 rows — the arithmetic that makes a page-granular guest write read
    /// as "one row plus change" in the artifact.
    fn byte_ranges_to_rows(ranges: &[(u64, u64)], base_off: u64, bpr: u64) -> Vec<(u32, u32)> {
        if bpr == 0 {
            return Vec::new();
        }
        let mut out: Vec<(u32, u32)> = Vec::new();
        for &(start, end) in ranges {
            if end <= base_off {
                continue;
            }
            let s = start.saturating_sub(base_off);
            let e = end - base_off;
            let (row_s, row_e) = ((s / bpr) as u32, (e.div_ceil(bpr)) as u32);
            Self::push_range(&mut out, &mut 0, row_s, row_e);
        }
        out
    }

    /// One partial preserving-unseeded draw on a guest-backed target. Only
    /// targets at least 256 px wide are recorded: the artifact is a display
    /// layer, and the ambient population of tiny scratch targets would bury it
    /// (702 events that second, 6.8 M texels, mostly small). 256, not 512 —
    /// og12's convicted strip was 486 px wide, and a threshold above the
    /// artifact's own width hid exactly the target the line exists to name.
    fn note_unseeded_partial(
        &mut self,
        scissor_y: u32,
        scissor_h: u32,
        target_w: u32,
        target_h: u32,
        lost_px: u64,
        mid: u32,
    ) {
        self.unseeded += 1;
        self.unseeded_lost_px += lost_px;
        if target_w >= 256 {
            let s = scissor_y.min(target_h);
            let e = (scissor_y.saturating_add(scissor_h)).min(target_h);
            Self::push_range(&mut self.unseeded_rows, &mut self.unseeded_more, s, e);
        }
        if !self.unseeded_mids.contains(&mid) && self.unseeded_mids.len() < RANGE_CAP {
            self.unseeded_mids.push(mid);
        }
    }

    /// A colour-0 load action was declared. `dontcare`/`clear` say which
    /// population the `px` belong to.
    fn note_load_declared(&mut self, dontcare: bool, clear: bool, px: u64) {
        if dontcare {
            self.dc_px += px;
        }
        if clear {
            self.clear_px += px;
        }
    }

    /// A partial-scissor pass declared CLEAR with a white clear colour. The
    /// px are the whole attachment's: the cover's share is the scissor union's,
    /// which the census already names.
    fn note_clear_partial_white(&mut self, px: u64) {
        self.clear_white_px += px;
    }

    /// The mapper-ref seed gate answered: an elision with its covered px, or a
    /// refusal that is about to read a seed.
    fn note_seed_gate(&mut self, elided: bool, px: u64) {
        if elided {
            self.elide += 1;
            self.elide_px += px;
        } else {
            self.provided += 1;
        }
    }

    /// A gather bind's witness verdict: a vouch served `span` bytes without
    /// reading them; a refusal gathered them.
    fn note_gather_bind(&mut self, vouched: bool, span: u64) {
        if vouched {
            self.skip += 1;
            self.skip_kb += span / 1024;
        } else {
            self.gather += 1;
        }
    }

    /// One guest-owned overlay adoption: the byte ranges the guest wrote,
    /// converted to target rows at the surface's stride.
    fn note_overlay(&mut self, ranges: &[(u64, u64)], base_off: u64, bpr: u64) {
        self.overlay += 1;
        let rows = Self::byte_ranges_to_rows(ranges, base_off, bpr);
        for &(s, e) in &rows {
            Self::push_range(&mut self.overlay_rows, &mut self.overlay_more, s, e);
        }
    }

    fn note_reopen(&mut self) {
        self.reopen += 1;
    }

    fn note_recreate(&mut self) {
        self.recreate += 1;
    }

    fn note_guard_drop(&mut self) {
        self.guard_drop += 1;
    }

    /// Render the per-present line and reset every delta. The reset is the
    /// emit: what follows is the next window's account, whatever order the
    /// hooks ran in.
    fn take_present_line(&mut self, mid: u32, w: u32, h: u32) -> String {        let fmt_ranges = |rows: &[(u32, u32)]| {
            if rows.is_empty() {
                "none".to_string()
            } else {
                rows.iter()
                    .map(|(s, e)| format!("{s}-{e}"))
                    .collect::<Vec<_>>()
                    .join(",")
            }
        };
        let fmt_mids = |mids: &[u32]| {
            if mids.is_empty() {
                "none".to_string()
            } else {
                mids.iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            }
        };
        let line = format!(
            "stripe_hunt present mid={mid} {w}x{h} dc_px={} clear_px={} clear_white_px={} \
             unseeded={} lost_px={} rows=[{}] rows_more={} mids=[{}] \
             elide={} elide_px={} provided={} \
             skip={} skip_kb={} gather={} \
             overlay={} overlay_rows=[{}] overlay_rows_more={} \
             reopen={} recreate={} guard_drop={} t={}",
            self.dc_px,
            self.clear_px,
            self.clear_white_px,
            self.unseeded,
            self.unseeded_lost_px,
            fmt_ranges(&self.unseeded_rows),
            self.unseeded_more,
            fmt_mids(&self.unseeded_mids),
            self.elide,
            self.elide_px,
            self.provided,
            self.skip,
            self.skip_kb,
            self.gather,
            self.overlay,
            fmt_ranges(&self.overlay_rows),
            self.overlay_more,
            self.reopen,
            self.recreate,
            self.guard_drop,
            crate::observe::elapsed_us() / 1000,
        );
        *self = Hunt::default();
        line
    }
}

/// The process-wide window. One mutex, held only for the length of each hook's
/// arithmetic; every hook is called from the drain or draw path and must never
/// take another lock while holding this one.
fn window() -> &'static Mutex<Hunt> {
    static WINDOW: Mutex<Hunt> = Mutex::new(Hunt::zero());
    &WINDOW
}

/// Hook: a colour-0 load action was declared. `dontcare`/`clear` say which
/// population the `px` belong to.
pub fn note_load_declared(dontcare: bool, clear: bool, px: u64) {
    if !on() {
        return;
    }
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note_load_declared(dontcare, clear, px);
}

/// Hook (iter 3, white stripes): a partial-scissor pass declared CLEAR with a
/// white clear colour — the CPU-visible half of the white-row family.
pub fn note_clear_partial_white(px: u64) {
    if !on() {
        return;
    }
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note_clear_partial_white(px);
}

/// Hook: a partial draw whose declared load preserves found no seed and no
/// resident — the census's defect arm. `mid` is 0 for a task-GVA target.
pub fn note_unseeded_partial(
    scissor_y: u32,
    scissor_h: u32,
    target_w: u32,
    target_h: u32,
    lost_px: u64,
    mid: u32,
) {
    if !on() {
        return;
    }
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note_unseeded_partial(scissor_y, scissor_h, target_w, target_h, lost_px, mid);
}

/// Hook: the mapper-ref seed gate answered. `elided` with `px` covered means
/// the resident carried the LOAD; `elided == false` is a seed provided (the
/// window where torn guest bytes can be adopted).
pub fn note_seed_gate(elided: bool, px: u64) {
    if !on() {
        return;
    }
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note_seed_gate(elided, px);
}

/// Hook: a gather bind's witness verdict. `vouched` with `span` bytes means a
/// retained image was served without reading the guest (the false-quiet
/// population); otherwise a fresh gather of `span` bytes ran.
pub fn note_gather_bind(vouched: bool, span: u64) {
    if !on() {
        return;
    }
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note_gather_bind(vouched, span);
}

/// Hook: an overlay laid guest-owned byte `ranges` (mapping-window offsets)
/// over the resident of a `bpr`-strided target at `base_off`.
pub fn note_overlay(ranges: &[(u64, u64)], base_off: u64, bpr: u64) {
    if !on() {
        return;
    }
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note_overlay(ranges, base_off, bpr);
}

/// Hook: a batch-tail flush-then-reopen (the open batch was full and the next
/// draw claimed its own slot). Counted between presents so a clear/covers-draw
/// pair split across a present can be seen as the split it was.
pub fn note_reopen() {
    if !on() {
        return;
    }
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note_reopen();
}

/// Hook: a resident was re-created / resampled / evicted — the fresh
/// zero-image population a subsequent unseeded pass reads.
pub fn note_recreate() {
    if !on() {
        return;
    }
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note_recreate();
}

/// Hook (lab, OVERLAY_GUARD): an overlay adoption was dropped because the
/// guest-write witness moved during its read — the suspected stripe mechanism
/// caught in the act, named on the present it preceded.
pub fn note_guard_drop() {
    if !on() {
        return;
    }
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .note_guard_drop();
}

/// Hook: the drain presented `mid` at `w`x`h`. Emits the window's line on the
/// lab channel and resets the accounting.
pub fn note_present(mid: u32, w: u32, h: u32) {
    if !on() {
        return;
    }
    let line = {
        let mut win = window().lock().unwrap_or_else(|e| e.into_inner());
        win.take_present_line(mid, w, h)
    };
    crate::observe::off(line);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hunt() -> Hunt {
        Hunt::default()
    }

    #[test]
    fn ranges_merge_touching_neighbours_and_count_overflow() {
        let mut h = hunt();
        // Three adjacent 1-row ranges become one; a distant range stays.
        h.note_unseeded_partial(10, 1, 1024, 2000, 100, 7);
        h.note_unseeded_partial(11, 1, 1024, 2000, 100, 7);
        h.note_unseeded_partial(12, 1, 1024, 2000, 100, 7);
        h.note_unseeded_partial(400, 4, 1024, 2000, 100, 7);
        assert_eq!(h.unseeded_rows, vec![(10, 13), (400, 404)]);
        assert_eq!(h.unseeded_more, 0);
        // Beyond the cap: counted, not stored, and totals keep accruing.
        for y in (500..600).step_by(2) {
            h.note_unseeded_partial(y, 1, 1024, 2000, 50, 7);
        }
        assert_eq!(h.unseeded_rows.len(), RANGE_CAP);
        assert_eq!(h.unseeded_more, 50 - (RANGE_CAP as u32 - 2));
        assert_eq!(h.unseeded, 4 + 50);
        assert_eq!(h.unseeded_lost_px, 400 + 50 * 50);
        assert_eq!(h.unseeded_mids, vec![7]);
    }

    #[test]
    fn small_targets_count_but_record_no_rows() {
        let mut h = hunt();
        // A 64x64 scratch target: counted and priced, no y-range (it is not
        // the artifact's shape and would flood the line).
        h.note_unseeded_partial(0, 8, 64, 64, 4096 - 512, 9);
        assert_eq!(h.unseeded, 1);
        assert_eq!(h.unseeded_lost_px, 3584);
        assert!(h.unseeded_rows.is_empty());
        // A 486-px-wide layer — og12's convicted strip — is named: the
        // threshold must sit below the artifact's own width or the line
        // hides exactly the target it exists to name.
        h.note_unseeded_partial(10, 29, 486, 238, 100_000, 91);
        assert_eq!(h.unseeded_rows, vec![(10, 39)]);
    }

    #[test]
    fn byte_ranges_become_rows_at_the_stride() {
        // 3880-byte rows (a 970-px BGRA8 row): a 4 KiB page is 1.05 rows, the
        // arithmetic behind the artifact's 1-px run texture. Two adjacent
        // pages straddle the row boundary and merge into one 0-3 row range.
        let ranges = [(0, 4096), (4096, 8192), (20000, 24000)];
        let rows = Hunt::byte_ranges_to_rows(&ranges, 0, 3880);
        assert_eq!(rows, vec![(0, 3), (5, 7)]);
        // base_off shifts the window: ranges below it are dropped.
        let rows = Hunt::byte_ranges_to_rows(&ranges, 10000, 3880);
        assert_eq!(rows, vec![(2, 4)]);
        // A zero stride cannot answer and says so with nothing.
        assert!(Hunt::byte_ranges_to_rows(&ranges, 0, 0).is_empty());
    }

    #[test]
    fn the_present_line_carries_the_account_and_resets_it() {
        let mut h = hunt();
        h.note_load_declared(true, false, 1_000);
        h.note_load_declared(false, true, 2_000);
        h.note_clear_partial_white(3_000);
        h.note_unseeded_partial(466, 238, 970, 476, 430_000, 91);
        h.note_seed_gate(true, 11_010_000);
        h.note_seed_gate(false, 0);
        h.note_gather_bind(true, 4096);
        h.note_gather_bind(false, 8192);
        h.note_overlay(&[(0, 3880), (7760, 11640)], 0, 3880);
        h.note_reopen();
        h.note_reopen();
        h.note_recreate();
        let line = h.take_present_line(1, 5120, 2160);
        assert!(line.contains("mid=1 5120x2160"), "{line}");
        assert!(line.contains("dc_px=1000 clear_px=2000"), "{line}");
        assert!(line.contains("clear_white_px=3000"), "{line}");
        assert!(line.contains("unseeded=1 lost_px=430000"), "{line}");
        // The scissor reaches past the target and is clamped to its height.
        assert!(line.contains("rows=[466-476]"), "{line}");
        assert!(line.contains("mids=[91]"), "{line}");
        assert!(line.contains("elide=1 elide_px=11010000 provided=1"), "{line}");
        assert!(line.contains("skip=1 skip_kb=4 gather=1"), "{line}");
        assert!(line.contains("overlay=1 overlay_rows=[0-1,2-3]"), "{line}");
        assert!(line.contains("reopen=2 recreate=1"), "{line}");
        // The reset: a second present says zero, not a running total.
        let again = h.take_present_line(1, 5120, 2160);
        assert!(again.contains("unseeded=0 lost_px=0"), "{again}");
        assert!(again.contains("skip=0 skip_kb=0 gather=0"), "{again}");
        assert!(again.contains("rows=[none]"), "{again}");
    }

    #[test]
    fn overlay_rows_merge_across_adoptions() {
        let mut h = hunt();
        // 4 KiB over 3880-byte rows is rows 0-2; the second adoption starts
        // inside row 1, so the two merge.
        h.note_overlay(&[(0, 4096)], 0, 3880);
        h.note_overlay(&[(3880, 8000)], 0, 3880);
        assert_eq!(h.overlay, 2);
        assert_eq!(h.overlay_rows, vec![(0, 3)]);
    }
}
