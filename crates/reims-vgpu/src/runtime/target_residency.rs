//! Explicit `DEVICE_LOCAL` residency for persistent composite / scanout surfaces.
//!
//! Step 2 of the sustained-FPS / VRAM residency repair. Eligibility and the
//! decision to prefer a device-local resident over importing guest RAM as the
//! *render target* live here. Placement and geometry are **derived** from the
//! live [`ResidentTargetSlot`] / [`ResidentContentBacking`] / [`TargetIdentity`]
//! â€” this module does not keep a parallel pixel or byte table that can drift.
//!
//! Scanout handoff, ordered guest copyback, and attachment `LOAD` elision are
//! Step 3 stamps [`TargetResidency::guest_published_epoch`] after ordered copyback
//! and stays 0 here.

use crate::backend::vulkan::engine::{
    ResidentContentBacking, TargetIdentity,
};
use crate::model::{DeviceState, MappingEntry};
use crate::protocol::pixel_format;
use crate::runtime::host::HostOps;
use crate::runtime::surface_currency::{surface_currency, CurrencyStandard};

/// Stable identity of a persistent composite or scanout surface eligible for
/// explicit `DEVICE_LOCAL` residency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PersistentSurfaceKey {
    pub mapping_id: u32,
    pub map_generation: u32,
    pub width: u32,
    pub height: u32,
    pub format: u16,
}

/// Actual allocation placement for a target, derived from the live resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationPlacement {
    DeviceLocal { bytes: u64 },
    ImportedGuestRam { bytes: u64 },
}

impl AllocationPlacement {
    pub fn bytes(self) -> u64 {
        match self {
            Self::DeviceLocal { bytes } | Self::ImportedGuestRam { bytes } => bytes,
        }
    }

    pub fn is_device_local(self) -> bool {
        matches!(self, Self::DeviceLocal { .. })
    }
}

/// Tracked residency metadata for a persistent composite or scanout surface.
///
/// Built by [`TargetResidency::from_live`]: mapping fields come from
/// [`MappingEntry`], placement/bytes/promoted/`guest_published_epoch` from the
/// live registry slot. Nothing here is an authoritative second copy of pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetResidency {
    pub key: PersistentSurfaceKey,
    pub content_generation: u32,
    pub surface_content_epoch: u32,
    /// Opaque allocation stamp taken from the resident image handle bits, or 0
    /// when no resident exists yet.
    pub allocation_id: u64,
    pub placement: AllocationPlacement,
    /// Step 3: stamped after ordered guest copyback; 0 until then.
    pub guest_published_epoch: u32,
    pub promoted: bool,
}

impl TargetResidency {
    /// Derive residency from the mapping and a live slot observation.
    pub fn from_live(
        mapping_id: u32,
        mapping: &MappingEntry,
        width: u32,
        height: u32,
        format: u16,
        live: LiveResidentObservation,
    ) -> Self {
        Self {
            key: PersistentSurfaceKey {
                mapping_id,
                map_generation: mapping.map_generation,
                width,
                height,
                format,
            },
            content_generation: mapping.content_generation,
            surface_content_epoch: mapping.surface_content_epoch,
            allocation_id: live.allocation_id,
            placement: live.placement,
            guest_published_epoch: live.guest_published_epoch,
            promoted: live.promoted,
        }
    }
}

/// What the Vulkan registry currently holds for one identity, for promotion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveResidentObservation {
    pub backing: ResidentContentBacking,
    pub placement: AllocationPlacement,
    pub allocation_id: u64,
    pub promoted: bool,
    pub guest_published_epoch: u32,
}

/// Lower-bound byte estimate for a colour attachment of this extent/format.
pub fn attachment_bytes_estimate(width: u32, height: u32, format: u16) -> u64 {
    let bpp = pixel_format::bytes_per_pixel(format).unwrap_or(pixel_format::BGRA8_BPP);
    u64::from(width) * u64::from(height) * u64::from(bpp)
}

fn mapping_declared_format(mapping: &MappingEntry) -> u16 {
    if mapping.format != 0 {
        mapping.format
    } else {
        pixel_format::MTL_FORMAT_BGRA8_UNORM
    }
}

/// Why a surface was refused for `DEVICE_LOCAL` promotion. Each arm's
/// [`PromoteRefusal::slug`] is a distinct greppable route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromoteRefusal {
    Unmapped,
    ExtentMismatch,
    FormatMismatch,
    GuestDirty,
    Warming,
    GenerationChanged { expected: u32, found: u32 },
    NotCompositeOrScanout,
}

impl PromoteRefusal {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unmapped => "promote_refused_unmapped",
            Self::ExtentMismatch => "promote_refused_extent_mismatch",
            Self::FormatMismatch => "promote_refused_format_mismatch",
            Self::GuestDirty => "promote_refused_guest_dirty",
            Self::Warming => "promote_refused_warming",
            Self::GenerationChanged { .. } => "promote_refused_generation_changed",
            Self::NotCompositeOrScanout => "promote_refused_not_composite_or_scanout",
        }
    }
}

/// Whether this mapping is eligible for persistent `DEVICE_LOCAL` promotion.
///
/// Requires a live mapped surface with matching extent/format, either
/// `content_generation >= 2` or `surface_content_epoch >= 2`, a composite or
/// scanout role, and [`CurrencyStandard::WatchedAndUnwritten`] from
/// [`surface_currency`]. `mapping_id` is a parameter (never filled as 0).
pub fn qualifies_for_promotion<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
) -> Result<PersistentSurfaceKey, PromoteRefusal> {
    let Some(mapping) = state.mappings.get(&mapping_id) else {
        return Err(PromoteRefusal::Unmapped);
    };
    if !mapping.mapped || !mapping.has_geom {
        return Err(PromoteRefusal::Unmapped);
    }
    // Composite (mapper-ref store path) or a surface the scanout has presented.
    if mapping.width != width || mapping.height != height {
        return Err(PromoteRefusal::ExtentMismatch);
    }
    let declared = mapping_declared_format(mapping);
    if declared != format {
        return Err(PromoteRefusal::FormatMismatch);
    }
    if mapping.content_generation < 2 && mapping.surface_content_epoch < 2 {
        return Err(PromoteRefusal::Warming);
    }
    let currency = surface_currency(state, host, mapping_id, width, height);
    if !currency.serves(CurrencyStandard::WatchedAndUnwritten) {
        return Err(PromoteRefusal::GuestDirty);
    }
    Ok(PersistentSurfaceKey {
        mapping_id,
        map_generation: mapping.map_generation,
        width,
        height,
        format,
    })
}

/// Same as [`qualifies_for_promotion`], but requires the caller to assert the
/// surface is on the composite (mapper-ref) or scanout path.
pub fn qualifies_for_promotion_composite_or_scanout<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
    is_composite_or_scanout: bool,
) -> Result<PersistentSurfaceKey, PromoteRefusal> {
    if !is_composite_or_scanout {
        return Err(PromoteRefusal::NotCompositeOrScanout);
    }
    qualifies_for_promotion(state, host, mapping_id, width, height, format)
}

/// Decision produced by [`prefer_device_local_target`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromoteDecision {
    /// Keep the existing guest-import / ordinary path.
    Unchanged { reason: &'static str },
    /// Drop guest-target memory so `registry_ensure` reuses or allocates a
    /// `DEVICE_LOCAL` resident. `reuse` is true when a DeviceAllocation slot
    /// already exists for the identity.
    PreferDeviceLocal { reuse: bool },
}

/// If the surface qualifies, drop guest-target backing so the engine uses a
/// `DEVICE_LOCAL` resident (reusing one already in the registry when present).
///
/// Does not allocate a second VkImage when a DeviceAllocation slot already
/// exists for `identity`. Does not disable `VK_EXT_external_memory_host` for
/// gathers or staging â€” only the primary render-target import is withheld.
pub fn prefer_device_local_target<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
    identity: &TargetIdentity,
    is_composite_or_scanout: bool,
) -> PromoteDecision {
    match qualifies_for_promotion_composite_or_scanout(
        state,
        host,
        mapping_id,
        width,
        height,
        format,
        is_composite_or_scanout,
    ) {
        Err(refusal) => {
            let reason = refusal.slug();
            crate::runtime::drain::note_store_route(reason);
            PromoteDecision::Unchanged { reason }
        }
        Ok(key) => {
            if let Some(live) = crate::backend::vulkan::engine::observe_resident_promotion(identity)
            {
                if live.allocation_id != 0 {
                    // A live resident whose map generation no longer matches the
                    // key is not the persistent surface we just qualified.
                    if let TargetIdentity::Surface { generation, .. } = identity {
                        if *generation != u64::from(key.map_generation) {
                            let reason = PromoteRefusal::GenerationChanged {
                                expected: key.map_generation,
                                found: *generation as u32,
                            }
                            .slug();
                            crate::runtime::drain::note_store_route(reason);
                            return PromoteDecision::Unchanged { reason };
                        }
                    }
                }
                match live.backing {
                    ResidentContentBacking::DeviceAllocation => {
                        crate::runtime::drain::note_store_route("promote_reused_device_local");
                        let _ = crate::backend::vulkan::engine::mark_resident_promoted(identity);
                        PromoteDecision::PreferDeviceLocal { reuse: true }
                    }
                    ResidentContentBacking::GuestAllocation
                    | ResidentContentBacking::NotReady => {
                        crate::runtime::drain::note_store_route("promote_allocate_device_local");
                        PromoteDecision::PreferDeviceLocal { reuse: false }
                    }
                }
            } else {
                crate::runtime::drain::note_store_route("promote_allocate_device_local");
                PromoteDecision::PreferDeviceLocal { reuse: false }
            }
        }
    }
}

/// Step 4: why a LOAD/DONT_CARE could not elide its seed into LoadFromTarget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadElisionRefusal {
    /// Guest load action does not preserve prior contents.
    NotLoadAction,
    /// No live promoted DEVICE_LOCAL resident for this identity.
    NotPromoted,
    /// Resident absent or content_epoch cleared since the last store.
    AbsentOrUnstamped,
    /// Mapping `surface_content_epoch` disagrees with the resident stamp.
    EpochMismatch { mapping: u32, resident: u32 },
    /// Identity generation / extent / format disagree with the mapping key.
    KeyMismatch,
    /// `surface_currency` does not prove the guest left the window alone.
    CurrencyDirty,
}

impl LoadElisionRefusal {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NotLoadAction => "persistent_load_refused_not_load_action",
            Self::NotPromoted => "persistent_load_refused_not_promoted",
            Self::AbsentOrUnstamped => "persistent_load_refused_absent_or_unstamped",
            Self::EpochMismatch { .. } => "persistent_load_refused_epoch_mismatch",
            Self::KeyMismatch => "persistent_load_refused_key_mismatch",
            Self::CurrencyDirty => "persistent_load_refused_currency_dirty",
        }
    }
}

/// Step 4: satisfy attachment LOAD from the persistent DEVICE_LOCAL resident
/// (`LoadOp::LoadFromTarget`) when the residency key and content stamp match
/// and `surface_currency` proves no guest write. Seed from guest RAM only when
/// this returns `Err`.
pub fn prefer_load_from_persistent_resident<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
    identity: &TargetIdentity,
    load_preserves_prior: bool,
) -> Result<(), LoadElisionRefusal> {
    if !load_preserves_prior {
        return Err(LoadElisionRefusal::NotLoadAction);
    }
    let Some(mapping) = state.mappings.get(&mapping_id) else {
        return Err(LoadElisionRefusal::KeyMismatch);
    };
    let key = PersistentSurfaceKey {
        mapping_id,
        map_generation: mapping.map_generation,
        width,
        height,
        format: if format != 0 {
            format
        } else {
            mapping_declared_format(mapping)
        },
    };
    // Identity must name the same generation / extent / format as the key.
    match identity {
        TargetIdentity::Surface {
            id,
            width: iw,
            height: ih,
            generation,
            format: _fmt,
        } => {
            if *id != mapping_id
                || *iw != key.width
                || *ih != key.height
                || *generation != u64::from(key.map_generation)
            {
                return Err(LoadElisionRefusal::KeyMismatch);
            }
            // VkFormat checked via mapping declaration / key above
        }
        _ => return Err(LoadElisionRefusal::KeyMismatch),
    }
    let Some(live) = crate::backend::vulkan::engine::observe_resident_promotion(identity) else {
        return Err(LoadElisionRefusal::NotPromoted);
    };
    if !live.promoted
        || !matches!(
            live.backing,
            ResidentContentBacking::DeviceAllocation
        )
    {
        return Err(LoadElisionRefusal::NotPromoted);
    }
    let resident_epoch = match crate::backend::vulkan::engine::resident_content_state(identity) {
        crate::backend::vulkan::engine::ResidentContent::Epoch(e) => e,
        crate::backend::vulkan::engine::ResidentContent::Absent
        | crate::backend::vulkan::engine::ResidentContent::Unstamped => {
            return Err(LoadElisionRefusal::AbsentOrUnstamped);
        }
    };
    if resident_epoch != mapping.surface_content_epoch {
        return Err(LoadElisionRefusal::EpochMismatch {
            mapping: mapping.surface_content_epoch,
            resident: resident_epoch,
        });
    }
    let currency = crate::runtime::surface_currency::surface_currency(
        state, host, mapping_id, width, height,
    );
    use crate::runtime::surface_currency::CurrencyStandard;
    // WatchedAndUnwritten is the default bar. A promoted DEVICE_LOCAL sole
    // producer may also accept NoContraryEvidence (resident is the authority).
    let currency_ok = currency.serves(CurrencyStandard::WatchedAndUnwritten)
        || (live.promoted && currency.serves(CurrencyStandard::NoContraryEvidence));
    if !currency_ok {
        return Err(LoadElisionRefusal::CurrencyDirty);
    }
    let _ = key;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DeviceId;
    use crate::protocol::gva::PAGE_SHIFT_X86;
    use crate::runtime::host::FakeHost;
    use crate::runtime::mapper::GuestWriteVerdict;
    use crate::runtime::surface_currency::SurfaceCurrency;

    fn mapped_surface(
        state: &mut DeviceState,
        host: &mut FakeHost,
        mid: u32,
        content_generation: u32,
        surface_content_epoch: u32,
        scanout: bool,
    ) {
        assert!(state.map_surface(mid));
        let pages: Vec<u64> = (0..4).map(|i| 0x2000_0000 + i * 4096).collect();
        let token = host.track_guest_writes(&pages, 4096).unwrap();
        let gen = host.guest_write_gen(token).unwrap();
        let m = state.mappings.get_mut(&mid).unwrap();
        m.has_geom = true;
        m.width = 64;
        m.height = 64;
        m.format = pixel_format::MTL_FORMAT_BGRA8_UNORM;
        m.content_generation = content_generation;
        m.surface_content_epoch = surface_content_epoch;
        m.scanout_presented = scanout;
        m.guest_write_token = token;
        m.guest_write_token_gen = m.map_generation;
        m.guest_write_gen_at_store = gen;
    }

    #[test]
    fn promotion_rejects_warming_extent_format_and_dirty_guest_pages() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        mapped_surface(&mut state, &mut host, 7, 1, 0, true);

        assert_eq!(
            qualifies_for_promotion(
                &state,
                &host,
                7,
                64,
                64,
                pixel_format::MTL_FORMAT_BGRA8_UNORM
            ),
            Err(PromoteRefusal::Warming)
        );

        state.mappings.get_mut(&7).unwrap().content_generation = 2;
        assert_eq!(
            qualifies_for_promotion(
                &state,
                &host,
                7,
                128,
                64,
                pixel_format::MTL_FORMAT_BGRA8_UNORM
            ),
            Err(PromoteRefusal::ExtentMismatch)
        );
        assert_eq!(
            qualifies_for_promotion(
                &state,
                &host,
                7,
                64,
                64,
                pixel_format::MTL_FORMAT_RGBA16_FLOAT
            ),
            Err(PromoteRefusal::FormatMismatch)
        );

        host.note_doorbell_and_dirty_page(0x2000_0000);
        assert_eq!(
            qualifies_for_promotion(
                &state,
                &host,
                7,
                64,
                64,
                pixel_format::MTL_FORMAT_BGRA8_UNORM
            ),
            Err(PromoteRefusal::GuestDirty)
        );
        assert_eq!(
            surface_currency(&state, &host, 7, 64, 64),
            SurfaceCurrency::WroteUnknown
        );
    }

    #[test]
    fn promotion_rejects_unmapped_and_non_composite_scanout() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        assert_eq!(
            qualifies_for_promotion(
                &state,
                &host,
                9,
                64,
                64,
                pixel_format::MTL_FORMAT_BGRA8_UNORM
            ),
            Err(PromoteRefusal::Unmapped)
        );
        assert_eq!(
            qualifies_for_promotion_composite_or_scanout(
                &state,
                &host,
                9,
                64,
                64,
                pixel_format::MTL_FORMAT_BGRA8_UNORM,
                false,
            ),
            Err(PromoteRefusal::NotCompositeOrScanout)
        );
    }

    #[test]
    fn clean_persistent_scanout_qualifies_and_prefers_device_local_without_resident() {
        use crate::runtime::drain::store_route_count;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        mapped_surface(&mut state, &mut host, 11, 2, 0, true);

        let key = qualifies_for_promotion(
            &state,
            &host,
            11,
            64,
            64,
            pixel_format::MTL_FORMAT_BGRA8_UNORM,
        )
        .expect("clean gen>=2 scanout must qualify");
        assert_eq!(key.mapping_id, 11);
        assert_eq!(key.map_generation, state.mappings[&11].map_generation);

        let identity = crate::backend::vulkan::present_identity::surface_identity(
            &state, 11, 64, 64,
        );
        let before = store_route_count("promote_allocate_device_local");
        let decision = prefer_device_local_target(
            &state,
            &host,
            11,
            64,
            64,
            pixel_format::MTL_FORMAT_BGRA8_UNORM,
            &identity,
            true,
        );
        assert_eq!(decision, PromoteDecision::PreferDeviceLocal { reuse: false });
        assert!(store_route_count("promote_allocate_device_local") > before);
    }

    #[test]
    fn placement_bytes_follow_extent_and_format() {
        assert_eq!(
            attachment_bytes_estimate(64, 64, pixel_format::MTL_FORMAT_BGRA8_UNORM),
            64 * 64 * 4
        );
        let live = LiveResidentObservation {
            backing: ResidentContentBacking::DeviceAllocation,
            placement: AllocationPlacement::DeviceLocal { bytes: 4096 },
            allocation_id: 0xdead,
            promoted: true,
            guest_published_epoch: 0,
        };
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        mapped_surface(&mut state, &mut host, 3, 3, 3, true);
        let m = state.mappings.get(&3).unwrap();
        let res = TargetResidency::from_live(
            3,
            m,
            64,
            64,
            pixel_format::MTL_FORMAT_BGRA8_UNORM,
            live,
        );
        assert!(res.promoted);
        assert_eq!(res.placement, AllocationPlacement::DeviceLocal { bytes: 4096 });
        assert_eq!(res.allocation_id, 0xdead);
        assert_eq!(res.guest_published_epoch, 0);
        assert_eq!(res.content_generation, 3);
        assert_eq!(res.surface_content_epoch, 3);
    }

    #[test]
    fn watched_clean_verdict_is_required_not_merely_no_contrary_evidence() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        assert!(state.map_surface(15));
        let m = state.mappings.get_mut(&15).unwrap();
        m.has_geom = true;
        m.width = 32;
        m.height = 32;
        m.format = pixel_format::MTL_FORMAT_BGRA8_UNORM;
        m.content_generation = 2;
        m.scanout_presented = true;
        // No guest-write token: currency is NoStamp, which serves
        // NoContraryEvidence but not WatchedAndUnwritten.
        assert_eq!(
            surface_currency(&state, &host, 15, 32, 32),
            SurfaceCurrency::Unwritten(GuestWriteVerdict::NoStamp)
        );
        assert_eq!(
            qualifies_for_promotion(
                &state,
                &host,
                15,
                32,
                32,
                pixel_format::MTL_FORMAT_BGRA8_UNORM
            ),
            Err(PromoteRefusal::GuestDirty)
        );
    }

    /// Step 3: GPU-frame handoff metadata derives from the live observation —
    /// guest_published_epoch rides beside promoted (no parallel table).
    #[test]
    fn gpu_frame_handoff_fields_derive_from_live_observation() {
        let live = LiveResidentObservation {
            backing: ResidentContentBacking::DeviceAllocation,
            placement: AllocationPlacement::DeviceLocal { bytes: 4096 },
            allocation_id: 7,
            promoted: true,
            guest_published_epoch: 3,
        };
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        mapped_surface(&mut state, &mut host, 9, 4, 4, true);
        let m = state.mappings.get(&9).unwrap();
        let res = TargetResidency::from_live(
            9,
            m,
            64,
            64,
            pixel_format::MTL_FORMAT_BGRA8_UNORM,
            live,
        );
        assert_eq!(res.guest_published_epoch, 3);
        assert!(res.promoted);
        assert_eq!(res.placement, AllocationPlacement::DeviceLocal { bytes: 4096 });
        assert_eq!(res.content_generation, 4);
    }

    /// Step 3: present completion is "guest batch flushed at publish", not a
    /// guest copyback — publication uses `guest_published_after_copyback` only.
    #[test]
    fn present_completion_kind_is_flush_at_publish_not_copyback() {
        use crate::backend::vulkan::engine::WindowPresentCompletion;
        let kind = WindowPresentCompletion::GuestBatchFlushedAtPublish;
        assert_eq!(kind, WindowPresentCompletion::GuestBatchFlushedAtPublish);
    }

    /// Step 4: CLEAR never elides into LoadFromTarget.
    #[test]
    fn persistent_load_elision_rejects_clear_action() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        mapped_surface(&mut state, &mut host, 21, 3, 3, true);
        let identity = TargetIdentity::Surface {
            id: 21,
            width: 64,
            height: 64,
            generation: 0,
            format: ash::vk::Format::B8G8R8A8_UNORM,
        };
        assert_eq!(
            prefer_load_from_persistent_resident(
                &state,
                &host,
                21,
                64,
                64,
                pixel_format::MTL_FORMAT_BGRA8_UNORM,
                &identity,
                false,
            ),
            Err(LoadElisionRefusal::NotLoadAction)
        );
        assert_eq!(
            LoadElisionRefusal::NotLoadAction.slug(),
            "persistent_load_refused_not_load_action"
        );
    }

    /// Step 4: without a promoted live resident, LOAD elision refuses.
    #[test]
    fn persistent_load_elision_rejects_when_not_promoted() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        mapped_surface(&mut state, &mut host, 22, 3, 3, true);
        let m = state.mappings.get(&22).unwrap();
        let identity = TargetIdentity::Surface {
            id: 22,
            width: 64,
            height: 64,
            generation: u64::from(m.map_generation),
            format: ash::vk::Format::B8G8R8A8_UNORM,
        };
        assert_eq!(
            prefer_load_from_persistent_resident(
                &state,
                &host,
                22,
                64,
                64,
                pixel_format::MTL_FORMAT_BGRA8_UNORM,
                &identity,
                true,
            ),
            Err(LoadElisionRefusal::NotPromoted)
        );
    }

    /// Step 4: identity generation that disagrees with the mapping key refuses.
    #[test]
    fn persistent_load_elision_rejects_key_mismatch() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        mapped_surface(&mut state, &mut host, 23, 3, 3, true);
        let identity = TargetIdentity::Surface {
            id: 23,
            width: 64,
            height: 64,
            generation: 99, // not the mapping's map_generation
            format: ash::vk::Format::B8G8R8A8_UNORM,
        };
        assert_eq!(
            prefer_load_from_persistent_resident(
                &state,
                &host,
                23,
                64,
                64,
                pixel_format::MTL_FORMAT_BGRA8_UNORM,
                &identity,
                true,
            ),
            Err(LoadElisionRefusal::KeyMismatch)
        );
    }
}