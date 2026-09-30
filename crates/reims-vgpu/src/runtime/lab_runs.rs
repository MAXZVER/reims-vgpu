//! TEMP (lab): exact fixes for the per-bind cost of a sampled window backed by a
//! mapping's guest pages (the `runs_big` route of `resolve_sampled_source`).
//! `REIMS_VGPU_LAB_RUNS_FIX=on` turns them on, `=verify` runs them beside the
//! original computation and counts every disagreement (the answer returned is
//! the original's), anything else keeps today's behaviour.
//!
//! Two fixes, neither a dirty-tracking shortcut - each returns exactly what the
//! code it short-cuts returns:
//!
//! * [`crate::runtime::host_writes::HostWrites::wrote_any_since`] answers
//!   `Quiet` when `since` is the current epoch, without walking the window. Every
//!   stored cell and `unnamed_at` are at most the current epoch, so the walk it
//!   skips can only reach `Quiet` (`wrote_any_since_at_the_current_epoch_is_the_
//!   full_verdict` is the proof, run over arbitrary write sequences).
//! * A per-mapping cache of the window a bind resolves: the window's page list
//!   and its coalesced runs, both `Arc`'d, keyed on the mapping's
//!   `map_generation` (every writer of `page_entries` that changes the list bumps
//!   it), the control-layout key (the other side of the control-page collision
//!   check) and the window's own arithmetic. A hit skips the whole-mapping page
//!   `Vec`, the run memo's SipHash over the window, the run list clone and the
//!   page-list clone.
//!
//! Counters: `lab_rf_hw_shortcut` / `lab_rf_hw_full` (asks answered by the epoch
//! shortcut / by the walk), `lab_rf_wc_hit` / `lab_rf_wc_miss` (window cache),
//! `lab_rf_wc_evict`, and in verify mode `lab_rf_verify_ok` / `lab_rf_verify_bad`.

use std::sync::Arc;

use crate::runtime::guest_ram::GuestRun;

/// Which arm of the experiment this process is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// Today's behaviour.
    Off,
    /// The exact fixes.
    On,
    /// The fixes beside the original computation; the original's answer stands.
    Verify,
}

pub fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("REIMS_VGPU_LAB_RUNS_FIX").as_deref() {
        Ok("on") => Mode::On,
        Ok("verify") => Mode::Verify,
        _ => Mode::Off,
    })
}

/// Windows remembered per mapping. A mapping is sampled through one or two
/// windows (a plane, a sub-rectangle); more than this thrashes and says so.
pub const WINDOWS_PER_MAPPING: usize = 4;

/// One resolved window of one mapping, valid while every field of its key still
/// holds.
#[derive(Clone, Debug)]
pub struct WindowCacheEntry {
    /// `MappingEntry::map_generation` the page list was read at.
    pub map_generation: u32,
    /// `mapper::control_layout_key` the control-page collision check passed at.
    pub control_key: u64,
    /// Pages in the whole mapping (the window's bounds are checked against it).
    pub mapping_pages: usize,
    pub base_off: u64,
    pub span: u64,
    pub page: u64,
    /// The window's own page list, in window order.
    pub gpas: Arc<Vec<u64>>,
    /// The window's coalesced host runs.
    pub runs: Arc<Vec<GuestRun>>,
}

/// What identifies a window for the cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WindowKey {
    pub map_generation: u32,
    pub control_key: u64,
    pub mapping_pages: usize,
    pub base_off: u64,
    pub span: u64,
    pub page: u64,
}

impl WindowCacheEntry {
    pub fn matches(&self, k: &WindowKey) -> bool {
        self.map_generation == k.map_generation
            && self.control_key == k.control_key
            && self.mapping_pages == k.mapping_pages
            && self.base_off == k.base_off
            && self.span == k.span
            && self.page == k.page
    }
}

/// Look `key` up in a mapping's windows.
pub fn lookup(cache: &[WindowCacheEntry], key: &WindowKey) -> Option<(Arc<Vec<u64>>, Arc<Vec<GuestRun>>)> {
    cache
        .iter()
        .find(|e| e.matches(key))
        .map(|e| (Arc::clone(&e.gpas), Arc::clone(&e.runs)))
}

/// Remember a window. Entries of a stale generation are dropped first (they can
/// never match again), then the oldest if the mapping is still over its share.
pub fn insert(cache: &mut Vec<WindowCacheEntry>, key: WindowKey, gpas: Arc<Vec<u64>>, runs: Arc<Vec<GuestRun>>) {
    cache.retain(|e| {
        e.map_generation == key.map_generation
            && e.control_key == key.control_key
            && e.mapping_pages == key.mapping_pages
            && !(e.base_off == key.base_off && e.span == key.span && e.page == key.page)
    });
    if cache.len() >= WINDOWS_PER_MAPPING {
        crate::runtime::drain::note_store_route("lab_rf_wc_evict");
        cache.remove(0);
    }
    cache.push(WindowCacheEntry {
        map_generation: key.map_generation,
        control_key: key.control_key,
        mapping_pages: key.mapping_pages,
        base_off: key.base_off,
        span: key.span,
        page: key.page,
        gpas,
        runs,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(generation: u32, control: u64, base_off: u64) -> WindowKey {
        WindowKey {
            map_generation: generation,
            control_key: control,
            mapping_pages: 8,
            base_off,
            span: 4096,
            page: 4096,
        }
    }

    fn window(tag: u64) -> (Arc<Vec<u64>>, Arc<Vec<GuestRun>>) {
        (Arc::new(vec![tag * 4096]), Arc::new(vec![GuestRun::in_mapping(0x1000, 4096, 0, 4096).unwrap()]))
    }

    #[test]
    fn a_hit_shares_the_arcs_and_any_key_field_that_moves_misses() {
        let mut cache = Vec::new();
        let (g, r) = window(1);
        insert(&mut cache, key(3, 7, 0), Arc::clone(&g), Arc::clone(&r));
        let (hg, hr) = lookup(&cache, &key(3, 7, 0)).expect("same key hits");
        assert!(Arc::ptr_eq(&hg, &g) && Arc::ptr_eq(&hr, &r), "a hit hands back the stored Arcs");
        assert!(lookup(&cache, &key(4, 7, 0)).is_none(), "map_generation moved");
        assert!(lookup(&cache, &key(3, 8, 0)).is_none(), "control layout moved");
        assert!(lookup(&cache, &key(3, 7, 4096)).is_none(), "a different window of the mapping");
        let mut k = key(3, 7, 0);
        k.mapping_pages = 9;
        assert!(lookup(&cache, &k).is_none(), "the mapping's page count moved");
    }

    #[test]
    fn a_new_generation_drops_the_stale_windows_and_the_mapping_keeps_a_bounded_share() {
        let mut cache = Vec::new();
        for i in 0..WINDOWS_PER_MAPPING as u64 {
            let (g, r) = window(i);
            insert(&mut cache, key(1, 1, i * 4096), g, r);
        }
        assert_eq!(cache.len(), WINDOWS_PER_MAPPING);
        let (g, r) = window(99);
        insert(&mut cache, key(1, 1, 99 * 4096), g, r);
        assert_eq!(cache.len(), WINDOWS_PER_MAPPING, "the oldest window made room");
        assert!(lookup(&cache, &key(1, 1, 0)).is_none(), "the oldest went");
        let (g, r) = window(5);
        insert(&mut cache, key(2, 1, 0), g, r);
        assert_eq!(cache.len(), 1, "a generation bump retires every window of the old list");
        assert!(lookup(&cache, &key(1, 1, 99 * 4096)).is_none());
        assert!(lookup(&cache, &key(2, 1, 0)).is_some());
    }

    #[test]
    fn re_resolving_the_same_window_replaces_it_instead_of_growing() {
        let mut cache = Vec::new();
        let (g, r) = window(1);
        insert(&mut cache, key(1, 1, 0), g, r);
        let (g2, r2) = window(2);
        insert(&mut cache, key(1, 1, 0), Arc::clone(&g2), r2);
        assert_eq!(cache.len(), 1);
        assert!(Arc::ptr_eq(&lookup(&cache, &key(1, 1, 0)).unwrap().0, &g2));
    }
}
