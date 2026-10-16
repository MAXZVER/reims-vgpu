//! Which door moved the host window's source epoch.
//!
//! `ResourcePools::invalidate_window_sources` is the one place
//! `WINDOW_SOURCE_EPOCH` moves, and four call sites reach it for four different
//! reasons. A stale-source present (`host_window_slate reason=slate_source_stale`,
//! or now `host_window_stale_hold`) says the epoch moved under a published frame
//! but not which of the four moved it, and the four have different fixes. This
//! counts them. Logging only: no guard changes behaviour because of it.
//!
//! Emitted from the per-second census as
//! `window_source_invalidate removed=.. format=.. access=.. overflow=..`, and
//! only for a second in which one of them moved.

use std::sync::atomic::{AtomicU64, Ordering};

/// The four doors into `invalidate_window_sources`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InvalidateSite {
    /// A published resident left the registry (`unregister_resident`).
    Removed = 0,
    /// A published resident was re-declared under another format.
    Format = 1,
    /// A published resident's access state changed.
    Access = 2,
    /// The publish set overflowed `WINDOW_PUBLISHED_MAX`.
    Overflow = 3,
}

const SITES: usize = 4;

static COUNTS: [AtomicU64; SITES] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Count one invalidation at `site`.
pub(crate) fn note(site: InvalidateSite) {
    COUNTS[site as usize].fetch_add(1, Ordering::Relaxed);
}

/// Totals since boot, without resetting. For tests and deltas.
#[cfg(test)]
pub(crate) fn snapshot() -> [u64; SITES] {
    std::array::from_fn(|i| COUNTS[i].load(Ordering::Relaxed))
}

/// Take this window's counts and reset them.
pub(crate) fn take_window() -> [u64; SITES] {
    std::array::from_fn(|i| COUNTS[i].swap(0, Ordering::Relaxed))
}

/// The census line, or `None` for a window in which nothing moved.
pub(crate) fn line(counts: [u64; SITES]) -> Option<String> {
    if counts.iter().all(|&n| n == 0) {
        return None;
    }
    Some(format!(
        "window_source_invalidate removed={} format={} access={} overflow={}",
        counts[InvalidateSite::Removed as usize],
        counts[InvalidateSite::Format as usize],
        counts[InvalidateSite::Access as usize],
        counts[InvalidateSite::Overflow as usize],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_site_counts_on_its_own_field() {
        for (site, ix) in [
            (InvalidateSite::Removed, 0),
            (InvalidateSite::Format, 1),
            (InvalidateSite::Access, 2),
            (InvalidateSite::Overflow, 3),
        ] {
            let before = snapshot();
            note(site);
            let after = snapshot();
            for i in 0..SITES {
                assert_eq!(
                    after[i] - before[i],
                    u64::from(i == ix),
                    "{site:?} moved field {i}"
                );
            }
        }
    }

    #[test]
    fn the_line_is_silent_when_nothing_moved_and_names_every_site_otherwise() {
        assert_eq!(line([0, 0, 0, 0]), None);
        assert_eq!(
            line([1, 2, 3, 4]).as_deref(),
            Some("window_source_invalidate removed=1 format=2 access=3 overflow=4")
        );
    }
}