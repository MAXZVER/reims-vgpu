//! Tests for mapper revalidation: when a captured page plan is re-checked.
//!
//! Out of line for the reason the sibling `runtime/` modules that already do
//! this have: colocated, this module and `tests` were together 3,038 of
//! `mapper.rs`'s 4,199 lines — 72% — so the IOSurface mapper itself was the
//! quarter of the file that was hardest to find.

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_X86};
use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
use crate::runtime::host::FakeHost;

#[test]
fn page_table_revalidation_slow_proxy_threshold_is_explicit() {
    assert!(!revalidate_timing_is_slow(REVALIDATE_SLOW_US - 1));
    assert!(revalidate_timing_is_slow(REVALIDATE_SLOW_US));
}

#[test]
fn fragmented_run_import_slow_proxy_threshold_is_explicit() {
    assert!(!mapping_run_import_is_slow(MAPPING_RUN_IMPORT_SLOW_US - 1));
    assert!(mapping_run_import_is_slow(MAPPING_RUN_IMPORT_SLOW_US));
}

#[test]
fn revalidate_fail_closed_without_internal_and_empty_pages() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    state.map_surface(2);
    // Mapped but no MappingInternal and no pages → not writable.
    assert!(!revalidate_mapping_pages(&mut state, &host, 2));
}

#[test]
fn revalidate_reason_disambiguates_the_miss() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    // Unknown id → the mapping was never created / already forgotten.
    assert_eq!(
        revalidate_mapping_reason(&mut state, &host, 7),
        Some("revalidate_gone")
    );
    // Mapped, no MappingInternal, no page list. The resolve never ran, so
    // this says nothing about whether the pages exist — which is exactly
    // what the old shared `revalidate_no_pages` slug hid behind a comment
    // calling it a benign (re)wire gap.
    state.map_surface(2);
    assert_eq!(
        revalidate_mapping_reason(&mut state, &host, 2),
        Some("revalidate_no_internal")
    );
    // Unmapped on entry is caught before the resolve and keeps its own slug.
    state.map_surface(3);
    state.mappings.get_mut(&3).unwrap().mapped = false;
    assert_eq!(
        revalidate_mapping_reason(&mut state, &host, 3),
        Some("revalidate_unmapped")
    );
    // A condemned backing is empty for a REASON the guest gave us
    // (DeleteIOSurfaceBacking2 stashed the page list), which is a different
    // answer from "the page list happens to be empty" and must not share its
    // slug — a deferred window flushing here has nothing safe to write
    // through, rather than nothing resolved yet.
    state.map_surface(5);
    state.mappings.get_mut(&5).unwrap().page_entries =
        vec![(0x200 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    assert!(state.condemn_surface_backing(5));
    assert_eq!(
        revalidate_mapping_reason(&mut state, &host, 5),
        Some("revalidate_condemned")
    );
    // A resolvable static page list → success (None).
    state.map_surface(4);
    state.mappings.get_mut(&4).unwrap().page_entries =
        vec![(0x100 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    assert_eq!(revalidate_mapping_reason(&mut state, &host, 4), None);
}

#[test]
fn surface_page_collision_detects_only_distinct_live_alias() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    let gpa = |pfn: u64| pfn << PAGE_SHIFT_X86;
    // Two distinct live surfaces on disjoint pages → no collision.
    state.map_surface(10);
    state.map_surface(20);
    state.mappings.get_mut(&10).unwrap().page_entries = vec![entry(0x100), entry(0x101)];
    state.mappings.get_mut(&20).unwrap().page_entries = vec![entry(0x200), entry(0x201)];
    assert_eq!(first_surface_page_collision(&state, 10), None);
    assert_eq!(first_surface_page_collision(&state, 20), None);
    // Surface 20 rewires onto a page surface 10 still owns → collision,
    // reported against the other owner (10) at the shared GPA.
    state.mappings.get_mut(&20).unwrap().page_entries = vec![entry(0x101), entry(0x201)];
    assert_eq!(
        first_surface_page_collision(&state, 20),
        Some((gpa(0x101), 10))
    );
    // A surface never collides with itself.
    assert_eq!(
        first_surface_page_collision(&state, 10),
        Some((gpa(0x101), 20))
    );
    // If the other owner is unmapped, the alias is legitimate (handoff) →
    // no collision.
    state.unmap_surface(10);
    assert_eq!(first_surface_page_collision(&state, 20), None);
    // Empty / unmapped self → None.
    state.mappings.get_mut(&20).unwrap().page_entries.clear();
    assert_eq!(first_surface_page_collision(&state, 20), None);
}

#[test]
fn reprieve_with_aliasing_peer_is_a_detected_collision() {
    // The condemn/reprieve corruptor precondition: a mapping's backing was
    // deleted (condemn stashed its pages), the guest handed the SAME
    // physical pages to another live surface, but this mapping's page table
    // still resolves to them — so the resolve fingerprints identical and
    // REPRIEVES (pages_changed == false, no map_generation bump). The rewire
    // wrong-PFN guard is gated on pages_changed and would never run; the
    // reprieve-path guard must catch it. This asserts both halves the branch
    // composes fire together on that exact state.
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    let gpa = |pfn: u64| pfn << PAGE_SHIFT_X86;

    // Mapping 3: condemned, its stashed pages == the plan it re-adopts.
    state.map_surface(3);
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.page_entries = vec![entry(0x300), entry(0x301)];
        m.map_generation = 4;
    }
    assert!(state.condemn_surface_backing(3));
    // The re-walked plan matches the condemned fingerprint → reprieve.
    let condemned = state.mappings.get(&3).unwrap().condemned_entries.clone();
    let plan = vec![entry(0x300), entry(0x301)];
    let (pages_changed, incarnation_changed, reprieved) =
        plan_adoption_decision(condemned.as_deref(), &[], &plan);
    assert!(
        reprieved,
        "same plan as condemned fingerprint must reprieve"
    );
    assert!(!pages_changed, "reprieve must not see a page change");
    assert!(!incarnation_changed);

    // Re-adopt the plan (as the resolve would) and stand up a DIFFERENT live
    // surface (20) that now also owns page 0x301 — the guest recycled it.
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.page_entries = plan.clone();
        m.condemned_entries = None;
    }
    state.map_surface(20);
    {
        let m = state.mappings.get_mut(&20).unwrap();
        m.mapped = true;
        m.page_entries = vec![entry(0x301), entry(0x999)];
    }
    // The reprieve-path guard's detector fires: mapping 3's re-adopted page
    // 0x301 is also owned by live surface 20 — the wrong-PFN write vector the
    // rewire-only guard would have missed (pages_changed was false).
    assert_eq!(
        first_surface_page_collision(&state, 3),
        Some((gpa(0x301), 20))
    );
}

#[test]
fn surface_page_collision_invalidates_mapping_fail_closed() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let entry = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    let gpa = |pfn: u64| pfn << PAGE_SHIFT_X86;
    const MID: u32 = 0x0CA;
    const OWNER: u32 = 0x0BE;

    state.map_surface(MID);
    {
        let m = state.mappings.get_mut(&MID).unwrap();
        m.mapped = true;
        m.map_generation = 7;
        m.page_entries = vec![entry(0x777), entry(0x778)];
        m.page_table_kva = 0xABC0;
    }
    state.map_surface(OWNER);
    {
        let m = state.mappings.get_mut(&OWNER).unwrap();
        m.mapped = true;
        m.page_entries = vec![entry(0x778)];
    }

    let (shared_gpa, owner) = first_surface_page_collision(&state, MID).expect("must detect alias");
    assert_eq!((shared_gpa, owner), (gpa(0x778), OWNER));

    fail_closed_surface_page_collision(&mut state, MID, shared_gpa, owner, 2, "test");
    let m = state.mappings.get(&MID).unwrap();
    assert!(m.mapped, "surface stays mapped but unresolved");
    assert!(
        m.page_entries.is_empty(),
        "known-bad page plan must be cleared"
    );
    assert_eq!(m.page_table_kva, 0);
    assert_eq!(
        m.map_generation, 8,
        "generation bump makes any deferred writeback fail closed"
    );
    assert_eq!(first_surface_page_collision(&state, MID), None);
}

#[test]
fn revalidate_accepts_static_page_list_without_internal() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    state.map_surface(4);
    {
        let m = state.mappings.get_mut(&4).unwrap();
        m.page_entries = vec![(0x100 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(revalidate_mapping_pages(&mut state, &host, 4));
}

#[test]
fn mapping_io_still_rejects_non_ram_page_at_map_boundary() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mid = 6;
    assert!(state.map_surface(mid));
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.page_entries = vec![(0x7f000 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    let mut byte = [0u8; 1];
    assert!(!read_mapping_bytes(
        &mut state, &mut host, mid, 0, &mut byte,
    ));
    let vouched = vouch_mapping_pages_verdict(&mut state, &host, mid)
        .1
        .expect("no walk to contradict");
    assert!(!write_mapping_bytes(
        &mut state,
        &mut host,
        mid,
        0,
        &[1],
        &vouched
    ));
}

#[test]
fn invalidate_mapping_pages_bumps_map_generation_and_clears() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.map_surface(5);
    let import_id = {
        let m = state.mappings.get_mut(&5).unwrap();
        m.page_entries = vec![1];
        m.contig_ptr = 0xdead;
        m.contig_len = 4096;
        m.contig_import = Some(std::sync::Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(0xdead_0000, 4096, 4096)
                .expect("synthetic aligned import"),
        ));
        m.contig_import.as_ref().unwrap().id()
    };
    let gen0 = state.mappings.get(&5).unwrap().map_generation;
    assert!(state.invalidate_mapping_pages(5));
    let m = state.mappings.get(&5).unwrap();
    assert!(m.page_entries.is_empty());
    assert_eq!(m.contig_ptr, 0);
    assert!(m.contig_import.is_none());
    assert!(m.map_generation != gen0);
    assert_eq!(state.retired_views, vec![(0xdead, 4096)]);
    assert_eq!(state.retired_guest_imports, vec![import_id]);
}

/// Invalidating a mapping's pages drops the host-side copy of those pages.
///
/// The cached frame is keyed by `(mapping_id, geometry)` with no
/// `map_generation` in it, so the bump that disqualifies the resident, the
/// contiguous view and every armed window leaves this entry addressable and
/// still holding pixels read through the page list that just stopped being
/// the surface's. The sampled ladder's host-cache rung then serves it: its
/// only currency test is the guest-write witness, and the token this call
/// retires makes that witness answer `NoStamp`, which is deliberately not
/// treated as evidence of a guest write.
///
/// Without the drop this asserts nothing about a rung — it asserts that a
/// copy of invalidated pages is still readable, which is the whole defect.
#[test]
fn invalidate_mapping_pages_drops_the_host_cache_of_those_pages() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.map_surface(5);
    {
        let m = state.mappings.get_mut(&5).unwrap();
        m.page_entries = vec![1];
        m.has_geom = true;
        m.width = 2;
        m.height = 2;
    }
    crate::runtime::surface_cache::store(&mut state, 5, 2, 2, vec![0xab; 2 * 2 * 4]);
    assert!(
        crate::runtime::surface_cache::get_shared(&state, 5, 2, 2).is_some(),
        "the cache entry under test was never stored"
    );

    state.invalidate_mapping_pages(5);

    assert!(
        crate::runtime::surface_cache::get_shared(&state, 5, 2, 2).is_none(),
        "a host copy of the invalidated pages is still being served"
    );
}

/// A host refusal is derived once per page list, not once per call, and a new
/// page list re-asks the host.
///
/// Before this cache every call on a fragmented mapping rebuilt the page-GPA
/// vector, rescanned it for runs, and emitted a line — 471 757 of them in
/// one 2 900 s boot, the only prefix ever to trip `log_flood_detected`. The
/// line count is therefore the assertion: repeated calls must add none, and
/// the magnitude the old line carried must still be readable as `served=`.
#[test]
fn a_host_refusal_is_cached_but_fragmentation_is_still_offered_to_it() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let entry =
        |gpa: u64| ((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT | PAGE_ENTRY_VALID;
    // Non-adjacent guest pages — never one packed run.
    let (gpa0, gpa1, gpa2) = (0x1000_0000u64, 0x2000_0000u64, 0x3000_0000u64);
    for gpa in [gpa0, gpa1, gpa2] {
        host.map_range(gpa, page as usize, 0);
    }
    let mid = 9u32;
    state.map_surface(mid);
    state.mappings.get_mut(&mid).unwrap().page_entries = vec![entry(gpa0), entry(gpa1)];

    let cap = crate::observe::FailCapture::start();
    let lines = || -> Vec<String> {
        cap.lines()
            .into_iter()
            .filter(|l| l.starts_with("OFF contig_view_refused"))
            .collect()
    };
    for _ in 0..16 {
        assert!(
            ensure_contig_view(&mut state, &mut host, mid).is_none(),
            "this fixture models a host that declines scattered aliases"
        );
    }
    let first = lines();
    assert_eq!(
        first.len(),
        1,
        "16 calls on one page list must derive (and say) the verdict once: {first:?}"
    );
    assert!(
        first[0].contains(" pages=2 physical_runs=2 "),
        "the derived line keeps its shape: {}",
        first[0]
    );

    // A different page list is a different verdict: the generation bump that
    // retires `contig_ptr` must also retire this.
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.page_entries = vec![entry(gpa0), entry(gpa1), entry(gpa2)];
        DeviceState::bump_map_generation(m);
    }
    assert!(ensure_contig_view(&mut state, &mut host, mid).is_none());
    let after = lines();
    assert_eq!(
        after.len(),
        2,
        "a new page list must re-derive and re-report: {after:?}"
    );
    assert!(
        after[1].contains(" pages=3 physical_runs=3 "),
        "the second line describes the second list: {}",
        after[1]
    );

    // Magnitude survives deduplication: `served` counts every fragmented
    // answer, cached ones included, so it advanced by the 16 calls between
    // the two derivations.
    let served = |l: &str| -> u64 {
        l.rsplit_once("served=")
            .and_then(|(_, v)| v.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .expect("line carries served=")
    };
    assert_eq!(
        served(&after[1]) - served(&after[0]),
        16,
        "served must count cached answers too: {after:?}"
    );
    assert_eq!(
        host.map_pages_calls, 2,
        "each page-list generation must reach the host exactly once"
    );
}

/// A host without a scattered-alias primitive declines the full view;
/// `write_mapping_bytes` still lands bytes via maximal packed runs.
#[test]
fn multi_import_fragmented_mapping_write() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    // Two non-adjacent guest pages (gap in GPA → not one packed map_pages).
    let gpa0 = 0x1000_0000u64;
    let gpa1 = 0x2000_0000u64;
    host.map_range(gpa0, page as usize, 0);
    host.map_range(gpa1, page as usize, 0);
    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 9u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.page_entries = vec![
            (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    assert!(
        ensure_contig_view(&mut state, &mut host, mid).is_none(),
        "fragmented list must not pack under strict_linux_map"
    );
    let payload = b"FRAG-MULTI-IMPORT-OK!!!!"; // 24 bytes
    let vouched = vouch_mapping_pages_verdict(&mut state, &host, mid)
        .1
        .expect("no walk to contradict");
    assert!(write_mapping_bytes(
        &mut state, &mut host, mid, 0, payload, &vouched
    ));
    // Second page offset = page_size.
    let mut hi = [0u8; 8];
    assert!(read_mapping_bytes(
        &mut state, &mut host, mid, page, &mut hi
    ));
    // Write only touched page 0; page 1 still zero.
    assert_eq!(hi, [0u8; 8]);
    let mut lo = [0u8; 24];
    assert!(read_mapping_bytes(&mut state, &mut host, mid, 0, &mut lo));
    assert_eq!(&lo[..], &payload[..]);
    // Cross-page write spanning the gap.
    let cross = vec![0xABu8; 16];
    let off = page - 8;
    assert!(write_mapping_bytes(
        &mut state, &mut host, mid, off, &cross, &vouched
    ));
    let mut check = [0u8; 16];
    assert!(read_mapping_bytes(
        &mut state, &mut host, mid, off, &mut check
    ));
    assert_eq!(check, [0xABu8; 16]);
}

/// The mapping rail's writes reach `observe::footprint`, and claim exactly
/// the frames they wrote.
///
/// This is the positive control the footprint's own unit tests cannot be:
/// those drive the bit set directly, so they prove the container works and
/// say nothing about whether any rail is wired to it. A footprint fed by no
/// rail reports an empty set, and an empty set answers "this device never
/// wrote that page" to every question — the exoneration that costs nothing
/// to produce and cannot be told from a real one.
///
/// The negatives are the load-bearing half. A mapping's page list is a
/// scatter, so the tempting implementation — mark `[gpa_lo, gpa_hi]` over
/// the write's span — would claim the 64 Ki frames between these two pages,
/// none of which this device can reach through this mapping. Every one of
/// them would then read as a hit for the rest of the boot, against a guest
/// panic that had nothing to do with us.
#[test]
fn a_mapping_write_marks_its_own_frames_and_not_the_gap_between_them() {
    use crate::observe::footprint;

    let _fp = footprint::exclusive_for_tests();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x1000_0000u64;
    let gpa1 = 0x2000_0000u64;
    host.map_range(gpa0, page as usize, 0);
    host.map_range(gpa1, page as usize, 0);
    let mid = 9u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.page_entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    let vouched = vouch_mapping_pages_verdict(&mut state, &host, mid)
        .1
        .expect("no walk to contradict");

    // 24 bytes at offset 0: page 0 only.
    assert!(write_mapping_bytes(
        &mut state,
        &mut host,
        mid,
        0,
        b"FOOTPRINT-FIRST-PAGE!!!!",
        &vouched
    ));
    assert!(footprint::wrote_gpa(gpa0), "the destination must be marked");
    assert!(
        !footprint::wrote_gpa(gpa1),
        "a write that never reached page 1 must not claim it"
    );
    assert!(
        !footprint::wrote_gpa((gpa0 + gpa1) / 2),
        "the hull between the two pages belongs to whoever the guest gave it \
         to; claiming it would make every later `pn` in that range a hit"
    );
    assert_eq!(footprint::counts(), (1, 0));

    // Cross-page: now both, and still nothing between them.
    assert!(write_mapping_bytes(
        &mut state,
        &mut host,
        mid,
        page - 8,
        &[0xABu8; 16],
        &vouched
    ));
    assert!(footprint::wrote_gpa(gpa1), "the second page is reached now");
    assert!(!footprint::wrote_gpa((gpa0 + gpa1) / 2));
    assert_eq!(
        footprint::counts(),
        (2, 0),
        "two frames total, not the span between them"
    );
}

/// The same claim for the contiguous-view fast path, which is the one
/// production takes.
///
/// The two paths mark through different code — the fast path resolves the
/// destination from the mapping's page list because it has only a host
/// pointer and an offset, the slow path from each packed run it maps — so a
/// control over one says nothing about the other. The fast path is where
/// nearly all of the ~100 000 mapping writes a driven boot makes actually
/// go, so an unmarked one there is most of the footprint missing.
#[test]
fn a_contiguous_mapping_write_marks_only_the_pages_its_offset_reaches() {
    use crate::observe::footprint;

    let _fp = footprint::exclusive_for_tests();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page = 1u64 << PAGE_SHIFT_X86;
    // Adjacent, so `ensure_contig_view` packs them into one view.
    let gpa0 = 0x3000_0000u64;
    let gpa1 = gpa0 + page;
    host.map_range(gpa0, 2 * page as usize, 0);
    let mid = 11u32;
    state.map_surface(mid);
    {
        let m = state.mappings.get_mut(&mid).unwrap();
        m.page_entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    let (_, _, pages) = ensure_contig_view_with_pages(&mut state, &mut host, mid)
        .expect("the fixture must take the fast path or it is testing the other one");
    assert_eq!(&*pages, &[gpa0, gpa1]);
    let (_, _, reused) = ensure_contig_view_with_pages(&mut state, &mut host, mid)
        .expect("the retained view remains live");
    assert!(
        std::sync::Arc::ptr_eq(&pages, &reused),
        "resource synchronization must retain the admitted footprint, not rebuild it"
    );
    let vouched = vouch_mapping_pages_verdict(&mut state, &host, mid)
        .1
        .expect("no walk to contradict");

    // Entirely inside page 1: the offset, not the base, decides the frame.
    assert!(write_mapping_bytes(
        &mut state,
        &mut host,
        mid,
        page + 16,
        &[0x5Au8; 32],
        &vouched
    ));
    assert!(footprint::wrote_gpa(gpa1));
    assert!(
        !footprint::wrote_gpa(gpa0),
        "marking from the mapping's base rather than the write's offset \
         would claim page 0, which this write never touched"
    );
    assert_eq!(footprint::counts(), (1, 0));
}

/// A guest-backed resident publishes the page footprint retained at admission,
/// not a second decode of the mapping that happened to carry the same identity
/// earlier. The byte window must still select within that retained scatter.
#[test]
fn an_admitted_page_footprint_marks_its_window_without_filling_scatter_gaps() {
    use crate::observe::footprint;

    let _fp = footprint::exclusive_for_tests();
    let page = 1u64 << PAGE_SHIFT_X86;
    let pages: std::sync::Arc<[u64]> = [0x1200_0000u64, 0x3200_0000u64].into();
    let retained =
        crate::runtime::guest_ram::GuestPageFootprint::new(std::sync::Arc::clone(&pages), page)
            .expect("non-empty page footprint");
    note_physical_page_write_footprint(&retained, page - 8, 16);

    assert!(footprint::wrote_gpa(pages[0] + page - 8));
    assert!(footprint::wrote_gpa(pages[1]));
    assert!(
        !footprint::wrote_gpa((pages[0] + pages[1]) / 2),
        "the retained allocation is still a scatter, never its physical hull"
    );
    assert_eq!(footprint::counts(), (2, 0));
}

/// The worker-pool arm of a mapping write ([`par_land_write_in`]) lands exactly
/// the bytes the serial run walk lands.
mod par_land {
    use super::*;
    use std::sync::Arc;

    const PAGES: usize = 96;
    const BASE: u64 = 0x4000_0000;

    unsafe extern "C" fn attach(_ctx: *mut std::os::raw::c_void) {}
    unsafe extern "C" fn detach(_ctx: *mut std::os::raw::c_void) {}

    fn pool(workers: usize) -> Arc<crate::runtime::par::Pool> {
        let mut table = crate::qemu::host_ops::ReimsVgpuHostOps::null();
        table.thread_attach = Some(attach);
        table.thread_detach = Some(detach);
        let settings = crate::runtime::par::Settings {
            workers,
            worker_spin: std::time::Duration::ZERO,
            join_spin: std::time::Duration::from_micros(50),
        };
        Arc::new(
            crate::runtime::par::Pool::new(&table, &settings).expect("the attach pair is offered"),
        )
    }

    /// One RAM range of `PAGES` pages, and a mapping whose page list visits it
    /// in `order`, so the walk meets single pages and packed runs alike.
    fn fixture(order: &[usize], with_pool: bool) -> (DeviceState, FakeHost, u32) {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.stable_map_pages = true;
        // A host whose map aliases packed runs only, as the Windows shim's does:
        // a scattered list has no single view, so the serial arm walks the runs
        // too instead of writing through a bounce copy of the whole list.
        host.strict_linux_map = true;
        let page = 1usize << PAGE_SHIFT_X86;
        host.map_range(BASE, PAGES * page, 0x5a);
        let mid = 11u32;
        state.map_surface(mid);
        let first_pfn = (BASE >> PAGE_SHIFT_X86) as u32;
        state.mappings.get_mut(&mid).unwrap().page_entries = order
            .iter()
            .map(|&i| ((first_pfn + i as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
        if with_pool {
            state.install_par_pool(pool(3));
        }
        (state, host, mid)
    }

    /// Pairs swapped in the first half, identity in the second: singles, then
    /// one long packed run.
    fn scrambled() -> Vec<usize> {
        (0..PAGES)
            .map(|i| match (i < PAGES / 2, i % 2) {
                (true, 0) => i + 1,
                (true, _) => i - 1,
                _ => i,
            })
            .collect()
    }

    fn ram(host: &FakeHost) -> Vec<u8> {
        let mut out = vec![0u8; PAGES << PAGE_SHIFT_X86];
        host.read_gpa(BASE, &mut out).expect("the fixture's range");
        out
    }

    /// Land `only` of a patterned buffer at `off` through both arms and return
    /// the two RAM images and what the pool arm answered.
    fn land_both(
        order: &[usize],
        off: u64,
        only: &[(u64, u64)],
    ) -> (Vec<u8>, Vec<u8>, Option<bool>) {
        let page = 1u64 << PAGE_SHIFT_X86;
        let len = (PAGES as u64 * page - off - 300) as usize;
        let buf: Vec<u8> = (0..len).map(|i| (i * 31 + i / 4096) as u8).collect();
        let (mut s_state, mut s_host, mid) = fixture(order, false);
        assert!(copy_mapping_runs(
            &mut s_state,
            &mut s_host,
            mid,
            off,
            RunCopy::Write(&buf),
            Some(only),
            "par_land_test"
        ));
        let (mut p_state, mut p_host, mid) = fixture(order, true);
        let gpas = mapping_page_gpas(&mut p_state, &mut p_host, mid).expect("a valid list");
        let runs = reims_vgpu_paging::runs::contig_page_runs(&gpas, page);
        let answer = par_land_write_in(
            true,
            &p_state,
            &mut p_host,
            mid,
            off,
            &buf,
            Some(only),
            &gpas,
            &runs,
            "par_land_test",
        );
        (ram(&s_host), ram(&p_host), answer)
    }

    fn selection() -> Vec<(u64, u64)> {
        let page = 1u64 << PAGE_SHIFT_X86;
        vec![
            (100, 5000),
            (3 * page + 17, 9 * page),
            (20 * page, 20 * page + 1),
            (25 * page + 9, 90 * page + 7),
        ]
    }

    /// The device's own landing hands the guest-write witness exactly the pages
    /// it touched - whole pages, in the mapping's own order, each once - so a
    /// hashed set takes them as its baseline instead of reporting them as the
    /// guest's at its next sync.
    #[test]
    fn a_device_landing_hands_the_witness_the_pages_it_touched() {
        let order = scrambled();
        let (mut state, mut host, mid) = fixture(&order, false);
        let token = ensure_guest_write_token(&mut state, &mut host, mid).expect("tracked");
        let page = 1u64 << PAGE_SHIFT_X86;
        let end = PAGES as u64 * page - 300;
        rebase_device_write(&state, &host, mid, 100, end, Some(&selection()));
        let expected: Vec<u64> = [0usize, 1, 3, 4, 5, 6, 7, 8, 20]
            .into_iter()
            .chain(25..=90)
            .map(|i| BASE + order[i] as u64 * page)
            .collect();
        assert_eq!(*host.rebased.borrow(), vec![(token, expected)]);
    }

    /// A token built for an earlier page list watches pages the surface no
    /// longer owns: nothing is handed to it.
    #[test]
    fn a_landing_under_a_moved_page_list_hands_over_nothing() {
        let (mut state, mut host, mid) = fixture(&scrambled(), false);
        ensure_guest_write_token(&mut state, &mut host, mid).expect("tracked");
        state.mappings.get_mut(&mid).unwrap().map_generation += 1;
        rebase_device_write(&state, &host, mid, 0, 4096, None);
        assert!(host.rebased.borrow().is_empty());
    }

    #[test]
    fn a_pooled_landing_writes_what_the_serial_walk_writes() {
        let (serial, pooled, answer) = land_both(&scrambled(), 100, &selection());
        assert_eq!(answer, Some(true), "the selection is large enough to fork");
        assert!(serial == pooled, "the two arms landed different bytes");
        // And the write did something: the untouched pattern is gone.
        assert!(serial.iter().any(|&b| b != 0x5a));
    }

    /// A page listed twice makes the last writer matter, so the pooled arm copies
    /// in walk order and still lands what the serial walk lands.
    #[test]
    fn a_page_listed_twice_lands_in_walk_order() {
        let mut order = scrambled();
        order[70] = 7;
        let (serial, pooled, answer) = land_both(&order, 100, &selection());
        assert_eq!(answer, Some(true));
        // What walk order means, computed directly: every selected byte of
        // position i lands in page order[i], later positions over earlier ones.
        let page = 1usize << PAGE_SHIFT_X86;
        let off = 100usize;
        let len = PAGES * page - off - 300;
        let buf: Vec<u8> = (0..len).map(|i| (i * 31 + i / 4096) as u8).collect();
        let mut want = vec![0x5au8; PAGES * page];
        for (i, &pg) in order.iter().enumerate() {
            for (lo, hi) in selection() {
                let (lo, hi) = (lo as usize, hi as usize);
                let a = lo.max(i * page).max(off);
                let b = hi.min((i + 1) * page).min(off + len);
                for m in a..b {
                    want[pg * page + (m - i * page)] = buf[m - off];
                }
            }
        }
        let bad = |img: &[u8]| -> Vec<usize> {
            (0..PAGES)
                .filter(|&p| img[p * page..(p + 1) * page] != want[p * page..(p + 1) * page])
                .collect()
        };
        assert!(
            bad(&serial).is_empty() && bad(&pooled).is_empty(),
            "serial off at {:?}, pooled off at {:?}",
            bad(&serial),
            bad(&pooled)
        );
    }

    #[test]
    fn a_small_selection_stays_on_the_caller() {
        let page = 1u64 << PAGE_SHIFT_X86;
        let (_, _, answer) = land_both(&scrambled(), 0, &[(page, 3 * page)]);
        assert_eq!(answer, None);
    }
}
