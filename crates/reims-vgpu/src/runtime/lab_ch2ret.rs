//! TEMP (lab, `REIMS_VGPU_LAB_CH2_RETURN_CNT=1`, off by default): how many
//! `CmdMapMemory2` (`0x39`) packets re-map the exact interval a *previous*
//! `CmdUnmapMemory` (`0x22`) of the same task tore down, and whether the pages
//! under that interval came back the same.
//!
//! # The question this sizes
//!
//! Steady-canvas ch2 is a pure IOSurface lifecycle conveyor (journal
//! 2026-10-16-ch2ops): ~29 map + ~29 unmap + ~29 delete per frame — WindowServer
//! rotates a window's whole surface set every frame, so the guest tears down
//! and re-imports the same ~29 slots 60 times a second. The lever this measures
//! is the guest-side **import cache**: elide the `(unmap + deleteResource)_t-1
//! → map_t` pair for a returning surface. Its host-side safety clause is "if a
//! map after an elided unmap names *different* pages, take the normal map
//! path". So the two numbers that decide whether the lever is worth building
//! are: what fraction of maps are returning (`map_returning / map_total`), and
//! of those, how many carry the same pages they had last time
//! (`pages_equal` vs `pages_differ`). Estimate at stake: −0.8–1.5 ms/frame of
//! WindowServer budget if nearly all maps return with equal pages.
//!
//! # Which page set the comparison is against, and why
//!
//! The obvious reading — "resolve the pages at unmap time and compare at map
//! time" — is not the honest instrument, because the two sides of the protocol
//! are not symmetrically ordered ([`crate::runtime::range_coverage`]): the
//! guest wires a range *before* submitting its map, so the map-side walk is
//! fully covered twenty times in twenty; it *starts* unwiring only when the
//! unmap is already readable, and wins that race about nineteen times in
//! twenty, so a walk at unmap time usually finds nothing to compare. This
//! module therefore remembers the page fingerprint **as resolved at the
//! interval's last map** — which is exactly the set the import cache would be
//! keeping host views over — and the unmap only marks the slot as torn down, so
//! a later map of the same `(task, gva, len)` counts as returning. A returning
//! map whose remembered fingerprint is absent (the interval's last map predates
//! the gate being on, or the task was not resolvable) is counted as
//! `pages_unknown` rather than guessed at either way.
//!
//! The fingerprint is order-independent (`count`, wrapping `sum`, `xor` over
//! the distinct resolved GPAs), so it is exact for equality against a
//! re-resolution of the same set and needs no per-interval page storage.
//!
//! # Shape
//!
//! Observation only: nothing here feeds a decision, and with the gate off the
//! whole thing is one `enabled()` load per map/unmap packet. The per-task rings
//! live on [`crate::model::DeviceState`] beside `map_audit`/`node_guard`, under
//! the lock the drain already holds; the counters are process-wide atomics and
//! print-and-reset once per second into the census line
//! `OFF reims-ch2ret map_total=… map_returning=… pages_equal=… …` from the
//! same note call, like `lab_stamp_lat` does. Each ring holds the last
//! [`RING_CAP`] intervals seen for its task; a returning map whose slot was
//! evicted is simply not counted as returning, which undercounts rather than
//! lies — `ring_evict` on the census line says whether the cap was ever the
//! limiting factor.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::model::DeviceState;
use crate::runtime::host::HostMemory;

/// How many recent intervals one task's ring holds. A frame rotates ~29
/// surfaces and a returning map is normally one frame behind its unmap, so
/// the reuse distance is tens of slots; 128 leaves headroom for a burst of
/// transitions, and `ring_evict` reports if it was ever not enough.
const RING_CAP: usize = 128;

/// TEMP (lab): `REIMS_VGPU_LAB_CH2_RETURN_CNT=1` (also accepts `on`), off by
/// default. Read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_CH2_RETURN_CNT").is_ok_and(|v| v == "1" || v == "on")
    })
}

/// Order-independent fingerprint of a resolved set of distinct page GPAs.
///
/// Two resolutions of the same set agree on all three words; two different
/// sets agreeing on all three is a 96-bit coincidence this diagnostic does not
/// expect to see and a caller cannot act on anyway.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct PagesFp {
    n: u32,
    sum: u64,
    xor: u64,
}

impl PagesFp {
    fn of_set(pages: &std::collections::HashSet<u64>) -> Self {
        let mut fp = Self::default();
        for &gpa in pages {
            fp.n = fp.n.saturating_add(1);
            fp.sum = fp.sum.wrapping_add(gpa);
            fp.xor ^= gpa;
        }
        fp
    }

    /// Whether this fingerprint actually names pages, rather than "nothing was
    /// resolved under this interval".
    const fn known(self) -> bool {
        self.n != 0
    }
}

/// One remembered `(gva, len)` interval of a task.
#[derive(Debug)]
struct RetEntry {
    gva: u64,
    len: u64,
    /// The pages the interval resolved to at its **last map** — the set an
    /// import cache would still be holding views over. Default (unknown) when
    /// only an unmap was seen for the interval.
    fp: PagesFp,
    /// Set by an unmap, cleared by the map that refreshes the entry.
    unmapped: bool,
}

/// Per-task ring of recently mapped/unmapped intervals, for the census.
///
/// Held on `DeviceState` beside `map_audit` for the same reason: the drain
/// path that mutates it is already behind the device lock, a task id may name
/// a task this device has no entry for (which is itself worth ringing), and a
/// reused id must not inherit a ring that describes somebody else's address
/// space.
#[derive(Default, Debug)]
pub struct RetTrack {
    ring: VecDeque<RetEntry>,
}

// Counters since the last census line. Print-and-reset, so each line reads as
// per-second rates exactly like `lab_stamp_lat`'s do.
static MAP_TOTAL: AtomicU64 = AtomicU64::new(0);
static UNMAP_TOTAL: AtomicU64 = AtomicU64::new(0);
static MAP_RETURNING: AtomicU64 = AtomicU64::new(0);
static PAGES_EQUAL: AtomicU64 = AtomicU64::new(0);
static PAGES_DIFFER: AtomicU64 = AtomicU64::new(0);
static PAGES_UNKNOWN: AtomicU64 = AtomicU64::new(0);
static RING_EVICT: AtomicU64 = AtomicU64::new(0);
static LAST_PRINT_MS: AtomicU64 = AtomicU64::new(0);

/// Record one decoded map/unmap notice and maybe print the census line.
///
/// Measure only: this returns nothing, mutates nothing outside its own ring
/// and counters, and must never influence the packet's own handling. Called
/// from `apply_map_family` after the notice decode and before the op is
/// applied to the model, which is the position the other map-side observers
/// (`map_audit`, `node_guard`, `range_coverage`) stand in for the same reason:
/// it is the last moment the range's pre-packet state is unambiguous.
pub fn note<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    length: u64,
    unmap: bool,
) {
    if !enabled() {
        return;
    }
    // The map side resolves the pages the interval *now* has (the guest wired
    // them before submitting the packet); the unmap side deliberately does
    // not walk — nineteen times in twenty the guest has already unwired, so
    // the remembered map-time fingerprint is the honest comparison base.
    let fp = if unmap {
        PagesFp::default()
    } else {
        PagesFp::of_set(&crate::runtime::gva_mem::task_gva_page_gpa_set(
            host,
            &state.tasks,
            task_id,
            gva,
            length,
            state.page_shift,
        ))
    };
    let track = state.ch2ret.entry(task_id).or_default();
    if unmap {
        UNMAP_TOTAL.fetch_add(1, Relaxed);
        match track.ring.iter_mut().rev().find(|e| e.gva == gva) {
            Some(e) => {
                // A different length under the same base is a new interval
                // squatting the slot: the old pages are not a comparison base
                // for it, so the fingerprint resets to unknown.
                if e.len != length {
                    e.len = length;
                    e.fp = PagesFp::default();
                }
                e.unmapped = true;
            }
            None => push_evict(track, RetEntry { gva, len: length, fp, unmapped: true }),
        }
    } else {
        MAP_TOTAL.fetch_add(1, Relaxed);
        match track.ring.iter_mut().rev().find(|e| e.gva == gva) {
            Some(e) => {
                if e.unmapped && e.len == length {
                    MAP_RETURNING.fetch_add(1, Relaxed);
                    if e.fp.known() && fp.known() {
                        if e.fp == fp {
                            PAGES_EQUAL.fetch_add(1, Relaxed);
                        } else {
                            PAGES_DIFFER.fetch_add(1, Relaxed);
                        }
                    } else {
                        PAGES_UNKNOWN.fetch_add(1, Relaxed);
                    }
                }
                // Refresh whatever the verdict was: this map is now the pages
                // an import cache would hold for the interval.
                e.len = length;
                e.fp = fp;
                e.unmapped = false;
            }
            None => push_evict(track, RetEntry { gva, len: length, fp, unmapped: false }),
        }
    }
    maybe_print();
}

/// Push an entry, evicting the oldest-seen one past the cap and counting the
/// eviction so the census can say whether the cap was ever limiting.
fn push_evict(track: &mut RetTrack, entry: RetEntry) {
    if track.ring.len() >= RING_CAP {
        track.ring.pop_front();
        RING_EVICT.fetch_add(1, Relaxed);
    }
    track.ring.push_back(entry);
}

/// Print the census line if a second has elapsed since the last one.
///
/// Driven by the packets themselves, like `lab_stamp_lat`: a second with no
/// map/unmap prints nothing, and a second with traffic always gets its line
/// within one packet past the boundary.
fn maybe_print() {
    let ms = crate::observe::elapsed_us() / 1000;
    let last = LAST_PRINT_MS.load(Relaxed);
    if ms.saturating_sub(last) < 1000
        || LAST_PRINT_MS
            .compare_exchange(last, ms, Relaxed, Relaxed)
            .is_err()
    {
        return;
    }
    let mt = MAP_TOTAL.swap(0, Relaxed);
    let um = UNMAP_TOTAL.swap(0, Relaxed);
    let ret = MAP_RETURNING.swap(0, Relaxed);
    let eq = PAGES_EQUAL.swap(0, Relaxed);
    let df = PAGES_DIFFER.swap(0, Relaxed);
    let un = PAGES_UNKNOWN.swap(0, Relaxed);
    let ev = RING_EVICT.swap(0, Relaxed);
    // Derived percentages, rounded, guarded: the line is read by eye and by
    // grep, and `ret_pct` against a silent second is not a datum.
    let ret_pct = (ret * 100)
        .checked_div(mt)
        .map_or_else(|| "-".to_string(), |p| p.to_string());
    let eq_pct = (eq * 100)
        .checked_div(eq + df)
        .map_or_else(|| "-".to_string(), |p| p.to_string());
    crate::observe::off(format!(
        "reims-ch2ret map_total={mt} unmap_total={um} map_returning={ret} \
         ret_pct={ret_pct} pages_equal={eq} pages_differ={df} pages_unknown={un} \
         ring_evict={ev} eq_pct={eq_pct}"
    ));
}
