//! TEMP (lab, `REIMS_VGPU_LAB_PASS_INVARIANTS=off|on`): the pass invariants of
//! one render pass, computed once on the commit of draw 0 and reused by the
//! records after it (design 2026-10-03 §6 C3 / §8.5, "thin commit").
//!
//! A serialized Metal render stream is one render pass: its attachment set is
//! fixed while pipeline, binds, and draw arguments change per record
//! (`exec::render_pass_attachment_template`). But the identities that describe
//! that fixed set were being re-derived on every record:
//!
//! - `gva_alloc_generation` — the color0 GVA target's page-set walk (and a
//!   task-local resource's declaration), once per draw at the top of
//!   `try_metal2vulkan_draw`;
//! - `render_chain_identity` — the Surface/Gva identity the seed, alias and
//!   chain rails key on, re-read from `state.mappings` at four sites per draw;
//! - `mapper_ref_texture_render_identity` (what
//!   `mapper_ref_texture_store_identity` answers under its writeback gate and
//!   what pairs the guest backing import), twice more;
//! - `mapper_ref_texture_guest_target_backing` — the sample window plus the
//!   checked contig import, whose `ensure_contig_view` re-resolves the
//!   mapping's page tables on every record.
//!
//! This module carries those answers from draw 0 to the records after it:
//! draw 0's commit records them at the canonical point (after the backing's
//! revalidation, so a mapping generation it advanced is the generation the
//! identities record), the exec chain loop harvests the record into the
//! attachment template, and every later record clones them out of the
//! template. Reuse is guarded, not assumed:
//!
//! - the **request half** — color0's identity-bearing fields must still equal
//!   the key the record was resolved from. The template copies color0
//!   verbatim, so a mismatch is impossible by construction and is asserted
//!   (`debug_assert!`); the runtime check under it is the fallback, not a
//!   second opinion;
//! - the **state half** — a Surface identity's mapping must still carry the
//!   `map_generation` the identity was resolved at. Anything device-side that
//!   rewired the mapping mid-pass (a resolve under another path advancing the
//!   generation) trips this and the site recomputes exactly as the serial
//!   path would, censused as `pass_inv_recomputed_fallback`.
//!
//! A Gva identity's generation is the draw-0 page-set hash and has no cheap
//! state-side witness — re-deriving it is the walk it stands in for. Within
//! one pass nothing rewrites a task's page tables (map/unmap are packets the
//! same drain thread serializes outside this EXEC), which is the same
//! stability `gva_resource_generation`'s `pending_writebacks` memo already
//! gives the `texture_ref != 0` form of the walk on the serial path.
//!
//! `off` (the default) is byte-for-byte the serial path: nothing records, so
//! every helper sees `None` and calls the function it replaced.

use super::*;

use crate::backend::vulkan::engine::{GuestTargetMemory, TargetIdentity};

/// Whether records 2+ reuse the pass invariants draw 0's commit recorded.
/// `REIMS_VGPU_LAB_PASS_INVARIANTS`, read once: `on` enables, anything else
/// (including unset) is the serial path.
static MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[cfg(test)]
static TEST_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub(crate) fn pass_inv_mode() -> bool {
    #[cfg(test)]
    {
        // Tests cannot re-latch the environment's OnceLock per arm, so a
        // process-wide override stands in front of it; the guard `test_mode`
        // returns clears it, keeping the tests order-independent. Unit tests
        // keep the serial path unless they ask.
        match TEST_MODE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    // On by default (n205, on the PASS_PREP-on base): fullscreen transitions
    // 19/10 % frames >25/>50 ms with the invariants against 26/12 % without,
    // stripes 0; `=off` restores the per-record recomputation.
    *MODE.get_or_init(|| {
        !matches!(
            std::env::var("REIMS_VGPU_LAB_PASS_INVARIANTS").as_deref(),
            Ok("off")
        )
    })
}

/// Force the mode for the lifetime of the returned guard (tests only).
#[cfg(test)]
pub(crate) fn test_mode(on: bool) -> TestModeGuard {
    TEST_MODE.store(
        if on { 2 } else { 1 },
        std::sync::atomic::Ordering::Relaxed,
    );
    TestModeGuard
}

/// Clears the [`test_mode`] override. Never outlives its guard's frame.
#[cfg(test)]
pub(crate) struct TestModeGuard;

#[cfg(test)]
impl Drop for TestModeGuard {
    fn drop(&mut self) {
        TEST_MODE.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The color0 request fields the pass invariants are a function of.
///
/// Everything the four hoisted answers read out of the request, and nothing
/// they do not: `load_action` is absent because records 2+ force LOAD while
/// the identities never consult it, and the seeds are absent because they are
/// per-record content, not per-pass identity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PassTargetKey {
    pub mapping_id: u32,
    pub texture_ref: u32,
    pub target_gva: u64,
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    pub store_action: u16,
}

impl PassTargetKey {
    fn of(req: &DrawEncodeRequest) -> Self {
        let c0 = req.colors.first();
        Self {
            mapping_id: c0.map_or(0, |c| c.mapping_id),
            texture_ref: c0.map_or(0, |c| c.texture_ref),
            target_gva: c0.map_or(0, |c| c.target_gva),
            row_stride: c0.map_or(0, |c| c.row_stride),
            width: c0.map_or(0, |c| c.width),
            height: c0.map_or(0, |c| c.height),
            format: c0.map_or(0, |c| c.format),
            store_action: c0.map_or(0, |c| c.store_action),
        }
    }

    /// Whether `req`'s color0 is still the color0 this key was resolved from.
    fn matches(&self, req: &DrawEncodeRequest) -> bool {
        *self == Self::of(req)
    }
}

/// The pass invariants of one render pass, recorded on the commit of draw 0.
///
/// Carried on [`DrawEncodeRequest::pass_invariants`] (an out-field on draw 0,
/// template cargo on the records after it), so the values cross the
/// backend-neutral exec loop as plain data the way `gva_alloc_gen` already
/// does.
#[derive(Clone, Debug)]
pub(crate) struct PassInvariants {
    /// The color0 the answers below were resolved from.
    pub key: PassTargetKey,
    /// [`super::gva_alloc_generation`]'s answer for draw 0: the page-set hash
    /// (or resource generation) every `TargetIdentity::Gva` this pass builds
    /// keys on, via `req.gva_alloc_gen`.
    pub gva_alloc_gen: u64,
    /// [`super::render_chain_identity`]'s answer: the identity the chain,
    /// alias and resident rails name the pass's target by.
    pub chain_identity: Option<TargetIdentity>,
    /// [`super::mapper_ref_texture_render_identity`]'s answer: the same
    /// identity gated on a publishing store — what
    /// `mapper_ref_texture_store_identity` returns under `writeback_guest`
    /// and what the guest backing import is paired with.
    pub render_identity: Option<TargetIdentity>,
    /// [`super::mapper_ref_texture_guest_target_backing`]'s answer: the
    /// sample window and the checked import over the mapping's pages. The
    /// import is an `Arc`; cloning this is pointer work, which is the point.
    pub guest_backing: Option<GuestTargetMemory>,
}

impl PassInvariants {
    /// Whether a Surface identity still names the mapping's current
    /// incarnation — one `HashMap` get and a `u32` compare, priced to sit in
    /// front of the lookups the recompute it guards would do.
    fn surface_generation_current(
        state: &DeviceState,
        identity: &Option<TargetIdentity>,
    ) -> bool {
        match identity {
            Some(TargetIdentity::Surface {
                id, generation, ..
            }) => state
                .mappings
                .get(id)
                .is_some_and(|m| u64::from(m.map_generation) == *generation),
            // A Gva generation is the draw-0 page-set hash: re-deriving it is
            // the walk it stands in for, and nothing inside one pass rewrites
            // a task's page tables (map/unmap packets are drain-serialized
            // outside this EXEC). No state-side witness, by design.
            _ => true,
        }
    }

    /// Whether the record this request describes may reuse these invariants.
    fn reusable(&self, state: &DeviceState, req: &DrawEncodeRequest) -> bool {
        // The request half is the structural guarantee: the attachment
        // template copies color0 verbatim, so a mismatch is a wiring defect
        // and the assert fires in debug; the checked `return false` under it
        // is the fallback that keeps a release build correct anyway.
        debug_assert!(
            self.key.matches(req),
            "PASS_INVARIANTS: draw-0 key {key:?} does not describe this record's color0",
            key = self.key
        );
        if !self.key.matches(req) {
            return false;
        }
        Self::surface_generation_current(state, &self.chain_identity)
            && Self::surface_generation_current(state, &self.render_identity)
    }
}

/// Record the pass invariants from the state of draw 0's commit.
///
/// Called at the canonical point — after `mapper_ref_texture_guest_backing`,
/// whose `ensure_contig_view` revalidation is the one step inside a draw that
/// can advance the mapping's generation — so the identities record the
/// post-revalidation incarnation, exactly the one the serial sites below the
/// backing call read. `guest_backing` is the value already resolved for this
/// draw; the identities are read here, once, for the pass.
pub(super) fn record_from_commit(
    state: &DeviceState,
    req: &DrawEncodeRequest,
    guest_backing: Option<GuestTargetMemory>,
) -> PassInvariants {
    let render_identity = super::mapper_ref_texture_render_identity(state, req);
    let chain_identity = super::render_chain_identity(state, req);
    // The render identity is the chain identity gated on a publishing store
    // action (`mapper_ref_texture_render_identity`): where it exists at all,
    // the two must be the same slot, or the backing pairing below has been
    // resolved against a different namespace than the one it claims.
    debug_assert!(
        render_identity.is_none() || render_identity == chain_identity,
        "the render identity is the chain identity gated on a publishing store: \
         {render_identity:?} vs {chain_identity:?}"
    );
    PassInvariants {
        key: PassTargetKey::of(req),
        gva_alloc_gen: req.gva_alloc_gen,
        chain_identity,
        render_identity,
        guest_backing,
    }
}

/// The pass's [`super::render_chain_identity`]: reused when the guard holds,
/// recomputed serially when it does not or nothing was recorded.
pub(super) fn chain_identity(
    state: &DeviceState,
    req: &DrawEncodeRequest,
) -> Option<TargetIdentity> {
    if let Some(inv) = req.pass_invariants.as_ref() {
        if inv.reusable(state, req) {
            return inv.chain_identity.clone();
        }
        crate::runtime::drain::note_store_route("pass_inv_recomputed_fallback");
    }
    super::render_chain_identity(state, req)
}

/// The pass's [`super::mapper_ref_texture_render_identity`], same contract as
/// [`chain_identity`].
pub(super) fn render_identity(
    state: &DeviceState,
    req: &DrawEncodeRequest,
) -> Option<TargetIdentity> {
    if let Some(inv) = req.pass_invariants.as_ref() {
        if inv.reusable(state, req) {
            return inv.render_identity.clone();
        }
        crate::runtime::drain::note_store_route("pass_inv_recomputed_fallback");
    }
    super::mapper_ref_texture_render_identity(state, req)
}

/// The pass's [`super::mapper_ref_texture_store_identity`]. The writeback gate
/// is the record's role in the packet, not an invariant — draw 0 answered it
/// `false`, the last record answers it `true`, and the identity under it is
/// the same slot either way — so the gate is asked per record and only the
/// identity under it is reused.
pub(super) fn store_identity(
    state: &DeviceState,
    req: &DrawEncodeRequest,
    writeback_guest: bool,
) -> Option<TargetIdentity> {
    if !writeback_guest {
        return None;
    }
    render_identity(state, req)
}

/// The pass's [`super::mapper_ref_texture_guest_target_backing`]. Reuse skips
/// the per-record `ensure_contig_view` revalidation — that walk is the work
/// this hoist removes — and the generation guard above it is what detects a
/// mapping that any other path re-resolved mid-pass.
pub(super) fn guest_backing<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
) -> Option<GuestTargetMemory> {
    if let Some(inv) = req.pass_invariants.as_ref() {
        if inv.reusable(state, req) {
            // Whole-answer reuse: the import and the identity it was paired
            // with stay one decision, never a cached pointer beside a newly
            // rewired surface.
            return inv.guest_backing.clone();
        }
        crate::runtime::drain::note_store_route("pass_inv_recomputed_fallback");
    }
    super::mapper_ref_texture_guest_target_backing(state, host, req)
}

/// The pass's [`super::gva_alloc_generation`], answered at the top of every
/// record. This is the one site every record passes through, so it is where
/// the reuse census counts.
pub(super) fn gva_alloc_gen<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
) -> u64 {
    if let Some(inv) = req.pass_invariants.as_ref() {
        if inv.reusable(state, req) {
            crate::runtime::drain::note_store_route("pass_inv_reused_n");
            return inv.gva_alloc_gen;
        }
        crate::runtime::drain::note_store_route("pass_inv_recomputed_fallback");
    }
    super::gva_alloc_generation(state, host, req)
}

#[cfg(all(test, feature = "backend-vulkan"))]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::drain::store_route_count;
    use crate::runtime::host::FakeHost;
    use reims_vgpu_protocol::pass_action::{MTL_LOAD_ACTION_LOAD, MTL_STORE_ACTION_STORE};

    fn delta(route: &str, before: u64) -> u64 {
        store_route_count(route).saturating_sub(before)
    }

    /// One guest page-table entry pointing GVA page 1 at `pfn`, on a task the
    /// GVA walker accepts (the shape `vulkan_split_tests` uses).
    fn map_one_gva_page(host: &mut FakeHost, pfn: u32) {
        use crate::protocol::endian::st32;
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        let page = 1u64 << PAGE_SHIFT_X86;
        for gpa in [2 * page, 3 * page, 4 * page, 5 * page] {
            host.map_range(gpa, page as usize, 0);
        }
        let mut dir = [0u8; 8];
        st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(2 * page, &dir).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        host.write_gpa(3 * page + 4, &pte).unwrap();
    }

    /// A GVA render target whose color0 span is one guest page at GVA 0x1000.
    fn one_page_gva_request() -> DrawEncodeRequest {
        DrawEncodeRequest {
            task_id: 1,
            colors: vec![ColorRtRequest {
                slot: 0,
                texture_ref: 7,
                mapping_id: 0,
                target_gva: 0x1000,
                row_stride: 32,
                width: 8,
                height: 8,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// A mapper-ref-texture fixture whose surface has real geometry, one
    /// mapped page, and a host that claims stable page aliasing, so the guest
    /// backing import resolves instead of declining on capability.
    fn mapper_ref_fixture() -> (DeviceState, FakeHost) {
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        crate::runtime::guest_ram_map::reset();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.stable_map_pages = true;
        let mid = 18u32;
        let gpa = 0x4200_0000u64;
        host.map_range(gpa, 1 << PAGE_SHIFT_X86, 0);
        assert!(state.map_surface(mid));
        state.mappings.get_mut(&mid).unwrap().page_entries = vec![
            ((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT | PAGE_ENTRY_VALID,
        ];
        assert!(state.set_mapping_geom(mid, 16, 16, MTL_FORMAT_BGRA8_UNORM));
        crate::runtime::guest_ram::latch_import_limits(1 << PAGE_SHIFT_X86, 1 << 30, 1 << 30);
        (state, host)
    }

    /// A mapper-ref draw request with a publishing store action, so the render
    /// identity (and under it the guest backing) resolves.
    fn mapper_ref_request() -> DrawEncodeRequest {
        DrawEncodeRequest {
            task_id: 1,
            pipeline_ref: 4,
            vertex_count: 3,
            colors: vec![ColorRtRequest {
                slot: 0,
                texture_ref: 7,
                mapping_id: 18,
                width: 16,
                height: 16,
                load_action: MTL_LOAD_ACTION_LOAD,
                store_action: MTL_STORE_ACTION_STORE,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// Early (draw-0 record) and late (per-draw serial) answers agree on the
    /// GVA fixture: the recorded generation is the walk's, the chain identity
    /// derived from it equals a fresh one, and reuse serves the same values
    /// without recomputing.
    #[test]
    fn early_record_equals_late_recompute_on_a_gva_target() {
        let _mode = test_mode(true);
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        map_one_gva_page(&mut host, 4);
        state.define_task(1, 0x1_0000, 2);

        let mut req = one_page_gva_request();
        req.gva_alloc_gen = super::super::gva_alloc_generation(&mut state, &mut host, &req);
        assert_ne!(req.gva_alloc_gen, 0, "fixture must walk a full page set");

        let inv = record_from_commit(&state, &req, None);
        assert_eq!(inv.key, PassTargetKey::of(&req));
        assert_eq!(inv.gva_alloc_gen, req.gva_alloc_gen);
        assert_eq!(
            inv.chain_identity,
            super::super::render_chain_identity(&state, &req),
            "the recorded chain identity is the one a late per-draw call answers"
        );
        assert!(inv.render_identity.is_none(), "a GVA target has no render identity");
        assert!(inv.guest_backing.is_none());

        // Reuse: same request, unchanged state → the hoisted values stand.
        req.pass_invariants = Some(inv);
        let before = store_route_count("pass_inv_reused_n");
        assert_eq!(
            gva_alloc_gen(&mut state, &mut host, &req),
            req.gva_alloc_gen,
        );
        assert_eq!(delta("pass_inv_reused_n", before), 1);
        assert_eq!(
            chain_identity(&state, &req),
            super::super::render_chain_identity(&state, &req),
            "reuse and a fresh call name the same resident"
        );
    }

    /// Early and late agree on the mapper-ref fixture too — including the
    /// guest backing import, whose window and footprint the record carries
    /// whole — and the store identity answers the writeback gate per record
    /// off the one hoisted render identity.
    #[test]
    fn early_record_equals_late_recompute_on_a_mapper_ref_target() {
        let _mode = test_mode(true);
        let (mut state, mut host) = mapper_ref_fixture();
        let mut req = mapper_ref_request();

        req.gva_alloc_gen = super::super::gva_alloc_generation(&mut state, &mut host, &req);
        let serial_backing = super::super::mapper_ref_texture_guest_target_backing(
            &mut state, &mut host, &req,
        );
        let inv = record_from_commit(&state, &req, serial_backing.clone());
        assert_eq!(
            inv.chain_identity,
            super::super::render_chain_identity(&state, &req)
        );
        assert_eq!(
            inv.render_identity,
            super::super::mapper_ref_texture_render_identity(&state, &req)
        );
        assert!(inv.render_identity.is_some(), "a publishing store resolves one");

        // The store identity follows the record's role in the packet, not the
        // template: draw 0 (no writeback) declines, the last record (writeback)
        // names the same slot the record holds.
        assert_eq!(store_identity(&state, &req, false), None);
        assert_eq!(
            store_identity(&state, &req, true),
            super::super::mapper_ref_texture_store_identity(&state, &req, true)
        );

        // The backing round-trips through the record and serves reuse.
        req.pass_invariants = Some(inv);
        assert_eq!(
            guest_backing(&mut state, &mut host, &req).map(|b| b.backing),
            serial_backing.map(|b| b.backing),
            "reused backing keeps the window and import the serial call built"
        );
        crate::runtime::guest_ram::forget_import_limits();
    }

    /// A mapping generation that moves between records trips the state guard
    /// and every identity site recomputes serially — censused, never silently
    /// stale. The backing falls with the identities: a cached pointer paired
    /// with a retired alias is the one shape the pairing rule forbids.
    #[test]
    fn a_mid_pass_generation_move_falls_back_to_the_serial_answer() {
        let _mode = test_mode(true);
        let (mut state, mut host) = mapper_ref_fixture();
        let mut req = mapper_ref_request();
        req.gva_alloc_gen = super::super::gva_alloc_generation(&mut state, &mut host, &req);
        let backing =
            super::super::mapper_ref_texture_guest_target_backing(&mut state, &mut host, &req);
        let inv = record_from_commit(&state, &req, backing);
        let stale = inv.chain_identity.clone();
        req.pass_invariants = Some(inv);

        // Whatever a mid-pass resolve was: it advanced the mapping's
        // generation, which is the only device-side witness this rail has.
        state.mappings.get_mut(&18).unwrap().map_generation += 1;

        let before = store_route_count("pass_inv_recomputed_fallback");
        let fresh = chain_identity(&state, &req);
        assert_eq!(delta("pass_inv_recomputed_fallback", before), 1);
        assert_ne!(
            fresh, stale,
            "the fallback must answer the new generation, not the recorded one"
        );
        assert_eq!(
            fresh,
            super::super::render_chain_identity(&state, &req),
            "the fallback is the serial answer"
        );
        assert_eq!(
            render_identity(&state, &req),
            super::super::mapper_ref_texture_render_identity(&state, &req),
            "the render identity site falls back the same way"
        );
        assert_eq!(
            delta("pass_inv_recomputed_fallback", before),
            2,
            "each guarded site recomputes and says so"
        );
        assert_eq!(
            guest_backing(&mut state, &mut host, &req).map(|b| b.backing),
            super::super::mapper_ref_texture_guest_target_backing(&mut state, &mut host, &req)
                .map(|b| b.backing),
            "the backing site falls back to the serial resolution"
        );
        assert_eq!(
            delta("pass_inv_recomputed_fallback", before),
            3,
            "the backing site recomputes and says so"
        );
        crate::runtime::guest_ram::forget_import_limits();
    }

    /// The request half of the guard: a key that no longer describes the
    /// record's color0 is a template defect, and the predicate (not the
    /// helper, whose debug_assert is the wiring alarm) reports it so the
    /// fallback has a test.
    #[test]
    fn a_key_that_no_longer_describes_color0_is_detectable() {
        let req = one_page_gva_request();
        let key = PassTargetKey::of(&req);
        assert!(key.matches(&req));
        let mut drifted = req.clone();
        drifted.colors[0].width += 8;
        assert!(
            !key.matches(&drifted),
            "a geometry change is a different attachment set"
        );
        let mut load_only = req.clone();
        load_only.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
        assert!(
            key.matches(&load_only),
            "the per-record load action is not part of the pass identity"
        );
    }

    /// Off is off: with nothing recorded, every helper is the function it
    /// replaced, and no pass census moves.
    #[test]
    fn with_no_record_every_helper_is_the_serial_call() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        map_one_gva_page(&mut host, 4);
        state.define_task(1, 0x1_0000, 2);
        let req = one_page_gva_request();
        assert!(req.pass_invariants.is_none());

        let before_fallback = store_route_count("pass_inv_recomputed_fallback");
        let before_reused = store_route_count("pass_inv_reused_n");
        assert_eq!(
            gva_alloc_gen(&mut state, &mut host, &req),
            super::super::gva_alloc_generation(&mut state, &mut host, &req)
        );
        assert_eq!(
            chain_identity(&state, &req),
            super::super::render_chain_identity(&state, &req)
        );
        assert_eq!(
            store_identity(&state, &req, true),
            super::super::mapper_ref_texture_store_identity(&state, &req, true)
        );
        assert_eq!(delta("pass_inv_recomputed_fallback", before_fallback), 0);
        assert_eq!(delta("pass_inv_reused_n", before_reused), 0);
    }
}
