//! Planning a sampled-source bind away from the drain thread, and taking the
//! plan on it.
//!
//! # Why
//!
//! Resolving a sampled source is the largest single phase of a draw and it runs
//! on the one drain thread, serially, for every distinct texture of every draw
//! (`sampled_phase resolve_us`). The top of the surface ladder — a ready
//! resident, or the host cache — is also the part most binds end on, and in its
//! warm state it only *reads* device state. That makes it plannable on another
//! thread ahead of the draw that needs it.
//!
//! # The contract
//!
//! Commit order and every guest-observable effect stay exactly the serial
//! ones. A plan is a *proposal*:
//!
//! - It is built from state it only reads ([`plan_sampled_source`] takes
//!   `&DeviceState`), and it refuses — [`SampleBail`] — at every point where the
//!   serial ladder would mutate: a cold resource, a cold backing, a lease that
//!   would have to be acquired, a resident the guest wrote over (merge/overlay),
//!   and every rung below the host cache.
//! - It carries the reads it rests on ([`SampleReads`]). [`commit_sampled_plan`]
//!   re-checks them at the point where the serial ladder would have resolved
//!   the bind itself; any difference and the plan is dropped and the serial
//!   ladder runs, unchanged.
//! - It carries the census routes the serial ladder would have noted, and the
//!   commit notes them — so a planned bind and a serial bind count the same.
//!
//! The decision itself is not duplicated: [`surface_top`] is the one function
//! both the serial ladder and the plan call for the resident rungs, and the
//! host-cache rung reads the same cache entry through the same accessor.
//!
//! # Why the guest-write witness can be re-checked without a sync
//!
//! The plan asked [`surface_currency`], which asked the hypervisor for the
//! token's write generation. The commit asks for the generation *as of the
//! last sync* ([`HostOps::guest_write_gen_lagged`]), which never waits. If no
//! sync has observed a new write since the plan, the lagged value is the one
//! the plan saw — and the serial ladder asking at that moment would be served
//! the same epoch's answer by the tracker. A sync that saw a write moves the
//! value, and the plan is dropped. A guest store no sync has observed yet is a
//! race the guest itself owns: the serial ladder could not see it either.

use super::*;
use crate::backend::vulkan::engine::{ResidentContentBacking, TargetIdentity};

/// Why the top of the surface ladder produced no [`SurfaceTop`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TopUnavailable {
    /// The mapping is absent or has no geometry; the serial ladder skips the
    /// surface rungs entirely for it.
    NoGeometry,
    /// The resident backing could not be answered without acquiring a lease.
    /// Only a plan's backing query returns this; the serial one always answers.
    BackingUnanswered,
}

/// What the resident rungs decided for one bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TopOutcome {
    /// A guest-allocation resident, bound directly; the currency is not asked.
    Direct,
    /// A ready device resident the guest has not replaced.
    Resident,
    /// A ready resident the view cannot bind (it remaps channels): the byte
    /// rungs serve it.
    Swizzled,
    /// A ready resident the guest wrote over: refused, merged, fallen through.
    Refused,
    /// No ready resident: the byte rungs serve it.
    NotReady,
}

/// The resident rungs' answer, and everything below them needs from it.
pub(super) struct SurfaceTop {
    pub(super) w: u32,
    pub(super) h: u32,
    pub(super) resident_id: TargetIdentity,
    pub(super) resident_backing: ResidentContentBacking,
    /// `None` only for [`TopOutcome::Direct`], which is taken before the
    /// currency is asked.
    pub(super) currency: Option<SurfaceCurrency>,
    pub(super) outcome: TopOutcome,
    /// The census routes the decision implies, in the order the serial ladder
    /// noted them.
    pub(super) routes: Vec<&'static str>,
}

impl SurfaceTop {
    pub(super) fn resident_ready(&self) -> bool {
        self.resident_backing != ResidentContentBacking::NotReady
    }

    /// The coarse verdict the rung census reports under.
    pub(super) fn guest_write(&self) -> GuestWriteVerdict {
        self.currency
            .as_ref()
            .map_or(GuestWriteVerdict::Clean, SurfaceCurrency::verdict)
    }

    /// Whether the guest replaced the bytes a host-side copy holds. The ladder
    /// under the resident reads the guest's own pages, so a refusal there is
    /// corrected below: the permissive standard.
    pub(super) fn guest_replaced(&self) -> bool {
        self.currency
            .as_ref()
            .is_some_and(|c| !c.serves(CurrencyStandard::NoContraryEvidence))
    }
}

/// Push a sample rung and its guest-write census column.
fn push_rung(routes: &mut Vec<&'static str>, rung: &'static str, guest_write: GuestWriteVerdict) {
    routes.push(rung);
    if let Some(gw) = sample_rung_gw_route(rung, guest_write) {
        routes.push(gw);
    }
}

/// The resident rungs of the surface ladder, from state they only read.
///
/// The one decision both the serial ladder ([`resolve_sampled_source`]) and a
/// plan ([`plan_sampled_source`]) take. `backing` answers the resident's
/// content backing: the serial ladder may acquire a lease to answer, a plan may
/// only read a warm one and returns `None` otherwise.
pub(super) fn surface_top<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mid: u32,
    may_bind_resident: bool,
    backing: impl FnOnce(&TargetIdentity) -> Option<ResidentContentBacking>,
) -> Result<SurfaceTop, TopUnavailable> {
    let (w, h) = match state.mappings.get(&mid) {
        Some(m) if m.has_geom && m.width > 0 && m.height > 0 => (m.width, m.height),
        _ => return Err(TopUnavailable::NoGeometry),
    };
    // Compute the resident-surface identity once and reuse it for
    // both the readiness check and the direct bind.
    let resident_id = crate::backend::vulkan::present_identity::surface_identity(state, mid, w, h);
    // `content_ready` only. The obvious strengthening — also require
    // the resident's `content_epoch` to match the mapping's, as the
    // attachment LOAD elision does — was tried and reverted, because
    // `content_epoch` is not free for this rung to reinterpret. The
    // deferred flush uses the same field as its own identity check
    // ("is this still the resident my window was armed on"), so
    // withholding the stamp to disqualify a resident *here* made
    // every later window on that resident report
    // `resident_epoch_drift` and drop its frame: `deferred_flush_lost`
    // went from 17 to 3 161 on one boot and the screen stayed black.
    //
    // Separating the two meanings is worth doing and is not this
    // change. What guards this rung today is the guest-write witness
    // below, which is the witness the LOAD elision's epoch pair
    // cannot supply anyway.
    let resident_backing = backing(&resident_id).ok_or(TopUnavailable::BackingUnanswered)?;
    let mut routes = Vec::with_capacity(4);
    match resident_backing {
        ResidentContentBacking::GuestAllocation => routes.push("t11sample_ready_guest_allocation"),
        ResidentContentBacking::DeviceAllocation => {
            routes.push("t11sample_ready_device_allocation")
        }
        ResidentContentBacking::NotReady => {}
    }
    // A render attachment and a sampled binding name the same
    // texture resource. When that resource's storage is the guest
    // allocation itself, a guest CPU write changes the resource;
    // it cannot make a second resident copy stale because there is
    // no second copy. Bind it directly before entering the currency
    // ladder, which exists for device-allocation mirrors.
    //
    // A non-identity view still falls through. The target bind
    // cannot carry that view's channel remap yet, so treating it as
    // direct would preserve the bytes and sample the wrong logical
    // channels.
    if guest_allocation_sample_is_direct(resident_backing, may_bind_resident) {
        routes.push("t11rung_resident");
        return Ok(SurfaceTop {
            w,
            h,
            resident_id,
            resident_backing,
            currency: None,
            outcome: TopOutcome::Direct,
            routes,
        });
    }
    // What the hypervisor can say about the guest's own stores into
    // this surface since the Store that produced our copies of it.
    // Asked once here and used by every rung below, because every
    // rung below either serves a host-side copy — whose currency
    // this is the only witness for — or reads the pages and does
    // not care. The host-cache rung used to ask it privately; the
    // resident rung above it asked nothing at all.
    //
    // Two stages: the token's generation says whether the guest
    // wrote the *allocation*, and only when it did is the page list
    // enumerated to say whether it wrote the *pixels*. The second
    // stage costs a page-list walk and is paid on the minority of
    // binds the first stage flags.
    let currency = surface_currency(state, host, mid, w, h);
    // The coarse column the rung census reports under. Taken from
    // the same answer the serving decision uses, so the two cannot
    // describe different asks.
    let guest_write = currency.verdict();
    if let Some(route) = match &currency {
        SurfaceCurrency::Unwritten(_) => None,
        SurfaceCurrency::WrotePixels(_) => Some("t11sample_gw_wrote_pixels"),
        SurfaceCurrency::WroteElsewhere => Some("t11sample_gw_wrote_elsewhere"),
        SurfaceCurrency::WroteUnknown => Some("t11sample_gw_wrote_unknown"),
    } {
        routes.push(route);
    }
    // Every rung under this one reads the guest's own pages, so a
    // serve this rung refuses is corrected below it rather than
    // held: the ladder takes the permissive standard.
    let guest_replaced = !currency.serves(CurrencyStandard::NoContraryEvidence);
    let resident_ready = resident_backing != ResidentContentBacking::NotReady;
    // A ready resident target is authoritative after a product
    // Store — but only while nothing has replaced the bytes it is a
    // copy of. A mapper-ref-texture surface's pages are plain guest RAM and the
    // guest CPU stores into them with no device operation, so a
    // resident produced for one tenant of the surface keeps claiming
    // to hold its pixels after the guest has painted different ones
    // there. This is the same question `mapper_ref_texture_guest_wrote_since_store`
    // asks before the attachment LOAD elision reuses a resident, and
    // the same one the host-cache rung below asks before serving its
    // copy. This rung asked neither, and it is the largest of the
    // three: one 14-round Finder boot measured 92 730 binds here
    // against the cache rung's 14 396.
    //
    // It also sits *above* both rungs that read the guest's own
    // pages, so a stale bind here is never corrected by anything
    // below it — the wrong image is held for as long as the guest
    // keeps re-binding the same surface, which is what "renders
    // correctly for a few frames then stays corrupted" looks like
    // from inside the guest.
    //
    // Refusing is not enough on its own, because neither copy is the
    // surface once both sides have written it. The refusal merges
    // them into the guest's pages first — see
    // [`merge_guest_writes_into_pages`] — and every rung below then
    // reads a surface that holds both halves.
    // A bind whose view remaps channels cannot take a resident
    // directly — the engine hands the swizzle to the image view and
    // the direct bind has none — so it falls straight to a byte rung
    // that can apply it. Counted on its own route and *outside* the
    // refusal below: nothing was replaced and there is nothing to
    // merge, so borrowing `_refused` would report a repaint that did
    // not happen and run a merge for it.
    let outcome = if resident_ready && !guest_replaced && !may_bind_resident {
        push_rung(&mut routes, "t11rung_resident_swizzled", guest_write);
        TopOutcome::Swizzled
    } else if resident_ready && !guest_replaced {
        push_rung(&mut routes, "t11rung_resident", guest_write);
        TopOutcome::Resident
    } else if resident_ready {
        push_rung(&mut routes, "t11rung_resident_refused", guest_write);
        TopOutcome::Refused
    } else {
        TopOutcome::NotReady
    };
    Ok(SurfaceTop {
        w,
        h,
        resident_id,
        resident_backing,
        currency: Some(currency),
        outcome,
        routes,
    })
}

/// Why a bind was not planned. Every arm is served by the serial ladder on the
/// drain thread, unchanged; the arm names which step would have mutated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleBail {
    /// Texture ref 0.
    NullRef,
    /// An opcode-9 buffer-backed texture: a CPU byte load.
    BufferTexture,
    /// A serialized ref-texture view that must be materialized.
    RefTextureView,
    /// Not a surface-backed sample (linear texture, or unclassified).
    NotSurface,
    /// A mapper-ref-texture resource not yet registered to its mapping.
    ColdMapperRef,
    /// A surface whose backing pages are not resolved yet.
    ColdBacking,
    /// A mapping with no geometry.
    NoGeometry,
    /// The resident backing needs a lease acquired.
    ColdLease,
    /// A ready resident the guest wrote over: merge/overlay mutate.
    GuestReplaced,
    /// The host cache holds no entry for the surface: the page rungs serve it.
    NoHostCache,
    /// The guest replaced the cache's bytes: the page rungs serve it.
    LowerRung,
    /// A ref-texture view whose zero-copy rail declines the plane (format,
    /// swizzle, window, floor or alias): the CPU loader serves it.
    ZeroCopyDeclined,
    /// The surface owes a writeback; paying it writes guest pages.
    OwedFrame,
    /// The mapping's window is not in its cache, or needs the mutating resolve.
    ColdWindow,
    /// The witness does not track this window with this page set yet.
    ColdWitness,
    /// A rung above the planned one would answer the bind (a GVA resident for
    /// the linear rail, the mapping's direct image for the mapper-ref rail):
    /// the serial ladder serves a different shape there, so the plan declines.
    UpperRung,
    /// The task's packed allocation for this reference is not retained (or is
    /// retained under another window): the serial rail would resolve or copy.
    ColdPacked,
    /// The memo fill this bind would carry exceeds the chunk's byte budget;
    /// the serial memo has no budget and fills on the drain thread.
    MemoBudget,
}

impl SampleBail {
    pub(crate) fn slug(self) -> &'static str {
        match self {
            Self::NullRef => "null_ref",
            Self::BufferTexture => "buffer_texture",
            Self::RefTextureView => "ref_texture_view",
            Self::NotSurface => "not_surface",
            Self::ColdMapperRef => "cold_mapper_ref",
            Self::ColdBacking => "cold_backing",
            Self::NoGeometry => "no_geometry",
            Self::ColdLease => "cold_lease",
            Self::GuestReplaced => "guest_replaced",
            Self::NoHostCache => "no_host_cache",
            Self::LowerRung => "lower_rung",
            Self::ZeroCopyDeclined => "zero_copy_declined",
            Self::OwedFrame => "owed_frame",
            Self::ColdWindow => "cold_window",
            Self::ColdWitness => "cold_witness",
            Self::UpperRung => "upper_rung",
            Self::ColdPacked => "cold_packed",
            Self::MemoBudget => "memo_budget",
        }
    }
}

/// Why a plan was dropped at commit. The serial ladder serves the bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleInvalid {
    /// The commit is for a different bind than the plan.
    Key,
    /// The ref resolves to a different retained resource now.
    Resource,
    /// The mapping's geometry, generation or witness fields moved.
    Mapping,
    /// The resident's backing changed.
    Backing,
    /// A sync observed a guest write since the plan.
    GuestWrite,
    /// The host cache entry was replaced.
    HostCache,
    /// The mapping's cached window is not the one the plan read.
    Window,
    /// The witness entry or its readings moved since the plan.
    Witness,
    /// A writeback debt was armed on the surface since the plan.
    Debt,
    /// A rung above the planned one became servable since the plan.
    UpperRung,
    /// The retained packed allocation the plan borrowed moved or vanished.
    Packed,
}

impl SampleInvalid {
    pub(crate) fn slug(self) -> &'static str {
        match self {
            Self::Key => "key",
            Self::Resource => "resource",
            Self::Mapping => "mapping",
            Self::Backing => "backing",
            Self::GuestWrite => "guest_write",
            Self::HostCache => "host_cache",
            Self::Window => "window",
            Self::Witness => "witness",
            Self::Debt => "debt",
            Self::UpperRung => "upper_rung",
            Self::Packed => "packed",
        }
    }
}

/// The mapping fields a plan's answer depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MappingReads {
    map_generation: u32,
    surface_content_epoch: u32,
    width: u32,
    height: u32,
    format: u16,
    has_geom: bool,
    guest_write_token: u64,
    guest_write_gen_at_store: u64,
    guest_write_token_gen: u32,
    guest_stated_gen: u32,
    guest_stated_at_stamp: u32,
}

impl MappingReads {
    fn of(state: &DeviceState, mid: u32) -> Option<Self> {
        state.mappings.get(&mid).map(|m| Self {
            map_generation: m.map_generation,
            surface_content_epoch: m.surface_content_epoch,
            width: m.width,
            height: m.height,
            format: m.format,
            has_geom: m.has_geom,
            guest_write_token: m.guest_write_token,
            guest_write_gen_at_store: m.guest_write_gen_at_store,
            guest_write_token_gen: m.guest_write_token_gen,
            guest_stated_gen: m.guest_stated_gen,
            guest_stated_at_stamp: m.guest_stated_at_stamp,
        })
    }
}

/// Everything a plan's answer rests on, re-checked at commit.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SampleReads {
    mapping: MappingReads,
    backing: ResidentContentBacking,
    /// The token's write generation as of the last sync, taken right after the
    /// plan asked the currency. `None` when the currency was not asked
    /// (a direct guest-allocation bind) or the token is unarmed.
    observed_gen: Option<u64>,
    /// The host cache entry's generation, for a host-cache serve.
    host_gen: Option<u64>,
}

/// What a planned bind serves.
#[derive(Clone)]
enum PlannedServe {
    Target(TargetIdentity),
    HostCache(std::sync::Arc<Vec<u8>>),
}

/// A planned sampled-source bind: the serve a top rung chose, the reads it rests
/// on, and the census routes it owes.
#[derive(Clone)]
pub(crate) struct SamplePlan {
    task_id: u32,
    texture_ref: u32,
    may_bind_resident: bool,
    resource: std::sync::Arc<crate::model::TaskResource>,
    mid: u32,
    w: u32,
    h: u32,
    serve: PlannedServe,
    reads: SampleReads,
    routes: Vec<&'static str>,
    /// The surface's resident identity, whose backing the commit re-reads.
    resident_id: TargetIdentity,
    /// Whether the resource owns the resident's lease (else the backing was
    /// the engine's own answer, as it is on the serial ladder).
    owns_resident: bool,
    /// Whether the backing came from a warm lease (else from the engine's
    /// readiness query, which a plan takes only when it says "not ready").
    lease_warm: bool,
    /// A host-cache serve fell past a resident that was not ready: the serial
    /// ladder reports that absence there, so the commit does too.
    absent_resident: bool,
}

impl SamplePlan {
    pub(crate) fn key(&self) -> (u32, u32) {
        (self.task_id, self.texture_ref)
    }
}

/// Plan the sampled source for one bind from state this call only reads.
///
/// Returns the plan for a bind the top of the surface ladder serves — a direct
/// or ready resident, or the host cache — and a [`SampleBail`] for every other
/// bind, which the serial ladder then serves on the drain thread.
/// TEMP (lab, `REIMS_VGPU_LAB_PREP_SYNC`): the guest-write token of the mapping
/// a planned bind of `resource` would ask about — [`plan_sampled_source`]'s own
/// classification, read only — when that token watches the mapping's current
/// page list. `None` for anything the plan would not reach the surface rungs
/// with.
pub(super) fn plan_guest_write_token(
    state: &DeviceState,
    texture_ref: u32,
    resource: &crate::model::TaskResource,
) -> Option<u64> {
    let entry = resource.entry;
    let mid = if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
        reims_vgpu_wire::device_desc::ref_texture_header(&resource.descriptor)
            .ok()
            .map(|t5| t5.surface_id.get())
            .filter(|sid| *sid != 0)?
    } else if entry.object_type == objects::OBJECT_TYPE_BACKING {
        texture_ref
    } else if entry.object_type == crate::runtime::decode::resource::OBJECT_TYPE_MAPPER_REF_TEXTURE
    {
        resource.registered_mapper_ref_texture_mapping()?
    } else {
        return None;
    };
    state
        .mappings
        .get(&mid)
        .filter(|m| m.guest_write_token != 0 && m.guest_write_token_gen == m.map_generation)
        .map(|m| m.guest_write_token)
}

pub(crate) fn plan_sampled_source<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    may_bind_resident: bool,
) -> Result<SamplePlan, SampleBail> {
    if texture_ref == 0 {
        return Err(SampleBail::NullRef);
    }
    if buffer_texture_descriptor(state, host, task_id, texture_ref, Some(&resource)).is_some() {
        return Err(SampleBail::BufferTexture);
    }
    // The same classification as the serial ladder, taking only warm answers.
    let entry = resource.entry;
    let mid = if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
        let desc = &resource.descriptor;
        let sid = reims_vgpu_wire::device_desc::ref_texture_header(desc)
            .map(|t5| t5.surface_id.get())
            .unwrap_or(0);
        if sid == 0 {
            return Err(SampleBail::NotSurface);
        }
        if let Some(view) = objects::decode_ref_texture_view(desc) {
            let needs_materialization = state
                .mappings
                .get(&sid)
                .map(|m| {
                    ref_texture_view_requires_materialization(
                        m.has_geom, m.width, m.height, m.format, view,
                    )
                })
                .unwrap_or(true);
            if needs_materialization {
                return Err(SampleBail::RefTextureView);
            }
        }
        sid
    } else if entry.object_type == objects::OBJECT_TYPE_BACKING {
        texture_ref
    } else if entry.object_type == crate::runtime::decode::resource::OBJECT_TYPE_MAPPER_REF_TEXTURE
    {
        resource
            .registered_mapper_ref_texture_mapping()
            .ok_or(SampleBail::ColdMapperRef)?
    } else {
        return Err(SampleBail::NotSurface);
    };
    // `ensure_surface_for_texture_bind`'s warm test; its cold arm mutates.
    let warm_backing = state
        .mappings
        .get(&mid)
        .is_some_and(|m| m.mapped && !m.page_entries.is_empty() && m.has_geom);
    if !warm_backing {
        return Err(SampleBail::ColdBacking);
    }
    let owns_resident = resource_type_owns_surface_resident(entry.object_type);
    // The serial ladder's backing query without its acquiring step. A warm
    // lease answers as it does there. Without one, the engine is asked whether
    // the resident is ready at all: "not ready" is the answer the acquiring
    // path reaches too (it retains only a content-ready slot) and it retains
    // nothing, so the plan may take it, noting the route that path notes.
    // "Ready" would make the serial ladder acquire a lease, which a plan must
    // not, so that bind is refused.
    let mut lease_warm = false;
    let mut acquire_would_miss = false;
    let top = surface_top(state, host, mid, may_bind_resident, |identity| {
        if !owns_resident {
            return Some(crate::backend::vulkan::engine::resident_content_backing(
                identity,
            ));
        }
        if let Some(warm) = resource_lease::resident_target_backing_warm(&resource, identity) {
            lease_warm = true;
            return Some(warm);
        }
        (crate::backend::vulkan::engine::resident_content_backing(identity)
            == ResidentContentBacking::NotReady)
            .then(|| {
                acquire_would_miss = true;
                ResidentContentBacking::NotReady
            })
    })
    .map_err(|unavailable| match unavailable {
        TopUnavailable::NoGeometry => SampleBail::NoGeometry,
        TopUnavailable::BackingUnanswered => SampleBail::ColdLease,
    })?;
    let mapping = MappingReads::of(state, mid).ok_or(SampleBail::NoGeometry)?;
    let observed_gen = (top.currency.is_some() && mapping.guest_write_token != 0)
        .then(|| host.guest_write_gen_lagged(mapping.guest_write_token))
        .flatten();
    let guest_write = top.guest_write();
    let guest_replaced = top.guest_replaced();
    let resident_ready = top.resident_ready();
    let SurfaceTop {
        w,
        h,
        resident_id,
        resident_backing,
        outcome,
        mut routes,
        ..
    } = top;
    if acquire_would_miss {
        routes.push("resident_resource_unavailable");
    }
    let (serve, host_gen, absent_resident) = match outcome {
        TopOutcome::Direct | TopOutcome::Resident => {
            (PlannedServe::Target(resident_id.clone()), None, false)
        }
        TopOutcome::Refused => return Err(SampleBail::GuestReplaced),
        TopOutcome::Swizzled | TopOutcome::NotReady => {
            if guest_replaced {
                return Err(SampleBail::LowerRung);
            }
            let (bgra, host_gen) =
                crate::runtime::surface_cache::get_shared_with_gen(state, mid, w, h)
                    .ok_or(SampleBail::NoHostCache)?;
            push_rung(&mut routes, "t11rung_host_cache", guest_write);
            (
                PlannedServe::HostCache(bgra),
                Some(host_gen),
                !resident_ready,
            )
        }
    };
    Ok(SamplePlan {
        task_id,
        texture_ref,
        may_bind_resident,
        resource,
        mid,
        w,
        h,
        serve,
        reads: SampleReads {
            mapping,
            backing: resident_backing,
            observed_gen,
            host_gen,
        },
        routes,
        resident_id,
        owns_resident,
        lease_warm,
        absent_resident,
    })
}

/// Re-check a plan's reads at the point the serial ladder would resolve the bind.
///
/// `Err` drops the plan for this bind; the caller then runs
/// [`resolve_sampled_source`], which is exactly what it would have run without
/// one. Validation reads only, so a plan may be validated again for a later bind.
pub(super) fn validate_sampled_plan<M: HostOps>(
    state: &DeviceState,
    host: &M,
    plan: &SamplePlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
) -> Result<(), SampleInvalid> {
    if plan.task_id != task_id
        || plan.texture_ref != texture_ref
        || plan.may_bind_resident != may_bind_resident
    {
        return Err(SampleInvalid::Key);
    }
    if !resource.is_some_and(|r| std::sync::Arc::ptr_eq(r, &plan.resource)) {
        return Err(SampleInvalid::Resource);
    }
    if MappingReads::of(state, plan.mid) != Some(plan.reads.mapping) {
        return Err(SampleInvalid::Mapping);
    }
    // The backing through the same source the plan read it from: a warm lease
    // must still be warm with the same backing (an expired one would make the
    // serial ladder acquire again), and an engine "not ready" must still be
    // "not ready" with no lease acquired meanwhile.
    let identity = &plan.resident_id;
    let engine_now = || crate::backend::vulkan::engine::resident_content_backing(identity);
    let backing_now = if !plan.owns_resident {
        Some(engine_now())
    } else if plan.lease_warm {
        resource_lease::resident_target_backing_warm(&plan.resource, identity)
    } else {
        resource_lease::resident_target_backing_warm(&plan.resource, identity)
            .is_none()
            .then(engine_now)
    };
    if backing_now != Some(plan.reads.backing) {
        return Err(SampleInvalid::Backing);
    }
    if let Some(gen_) = plan.reads.observed_gen {
        if host.guest_write_gen_lagged(plan.reads.mapping.guest_write_token) != Some(gen_) {
            return Err(SampleInvalid::GuestWrite);
        }
    }
    if let Some(host_gen) = plan.reads.host_gen {
        let now =
            crate::runtime::surface_cache::get_shared_with_gen(state, plan.mid, plan.w, plan.h)
                .map(|(_, g)| g);
        if now != Some(host_gen) {
            return Err(SampleInvalid::HostCache);
        }
    }
    Ok(())
}

/// Take a validated plan: its serve, and (with `note`) the census routes and
/// the reclaim report the serial ladder would have made for the same bind.
/// Taking changes no device state, which is why one plan serves every bind of
/// its ref for as long as it validates.
pub(super) fn take_sampled_plan(
    state: &DeviceState,
    plan: &SamplePlan,
    note: bool,
) -> (u32, u32, u32, SampledSourceRequest) {
    if note {
        for route in &plan.routes {
            crate::runtime::drain::note_store_route(route);
        }
        if plan.absent_resident {
            note_sampled_resident_absence(plan.mid, plan.w, plan.h, &plan.resident_id);
        }
    }
    let (w, h, mid) = (plan.w, plan.h, plan.mid);
    match &plan.serve {
        PlannedServe::Target(identity) => {
            let format = super::surface_sample_format(state, mid, identity);
            (
                w,
                h,
                mid,
                SampledSourceRequest::Target(identity.clone(), format),
            )
        }
        PlannedServe::HostCache(bgra) => {
            let host_gen = plan.reads.host_gen.unwrap_or(0);
            (
                w,
                h,
                mid,
                host_cache_request(state, mid, bgra.clone(), host_gen),
            )
        }
    }
}

/// Validate and take a plan in one step.
#[cfg(test)]
pub(super) fn commit_sampled_plan<M: HostOps>(
    state: &DeviceState,
    host: &M,
    plan: &SamplePlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
) -> Result<(u32, u32, u32, SampledSourceRequest), SampleInvalid> {
    validate_sampled_plan(
        state,
        host,
        plan,
        task_id,
        texture_ref,
        resource,
        may_bind_resident,
    )?;
    Ok(take_sampled_plan(state, plan, true))
}

/// Why the zero-copy rail's backend import cannot serve a plane, so the guest
/// runs do: the two answers `mapped_sampled_source` reaches without changing
/// anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImportGate {
    /// No stable page alias, no contiguous view for this page list (refused and
    /// remembered for this generation), or a view with no footprint.
    NoView,
    /// A contiguous view exists and the import map stands on a refusal; the
    /// serial rail notes `zc_packed_alias_standing_refusal` on its way past.
    StandingRefusal,
}

/// A ref-texture view's plane as the zero-copy rail derives it from the mapping
/// row, and the import gate in front of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RefViewPlane {
    native: TexelLayout,
    sampled_vk_format: ash::vk::Format,
    base_off: u64,
    span: u64,
    row_length_texels: u32,
    import: ImportGate,
}

/// `try_ref_texture_sample_zero_copy`'s plane for `view` over `mid`, derived the
/// same way from state this call only reads. `None` wherever that rail would
/// decline, or would have to resolve or import something first.
fn ref_view_plane<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mid: u32,
    view: objects::RefTextureView,
) -> Option<RefViewPlane> {
    let (w, h) = (view.width, view.height);
    if w == 0 || h == 0 || view.depth != 1 {
        return None;
    }
    let m = state.mappings.get(&mid)?;
    // `ensure_resolved_for_scanout` answers from the row alone only without a
    // `mapping_internal`; with one it may re-resolve, which only the drain does.
    if !m.mapped || m.page_entries.is_empty() || m.mapping_internal != 0 {
        return None;
    }
    let (native, bpp) = match translate::pixel::sampled_pixels(view.pixel_format) {
        Ok((layout, _decline, components)) if pixel_format::swizzle_is_identity(&components) => {
            (layout, layout.bytes_per_texel())
        }
        _ => return None,
    };
    let (base_off, bpr_u32, _span_end) = crate::runtime::mapping_write::ref_texture_sample_window(
        m,
        view.plane_index,
        w,
        h,
        view.pixel_format,
    )?;
    let sampled_vk_format = translate::pixel::translate(view.pixel_format).ok()?.vk;
    let (span, row_length_texels) = strided_window_extent(w, h, bpp as u64, bpr_u32 as u64)?;
    let import = if !host.map_pages_stable()
        || (m.contig_ptr == 0 && m.contig_refused_gen == Some(m.map_generation))
        || (m.contig_ptr != 0 && m.contig_footprint.is_none())
    {
        ImportGate::NoView
    } else if m.contig_ptr != 0
        && matches!(
            crate::runtime::guest_ram_map::standing_refusal_resolved(),
            Some(Some(_))
        )
    {
        ImportGate::StandingRefusal
    } else {
        return None;
    };
    Some(RefViewPlane {
        native,
        sampled_vk_format,
        base_off,
        span,
        row_length_texels,
        import,
    })
}

/// A ref-texture view's plane planned onto the zero-copy rail: the window the
/// mapping's cache holds and the gather witness's decision for it, taken off the
/// drain thread from state the plan only reads.
///
/// The serial counterpart is the materialization arm of [`resolve_sampled_source`]
/// through [`try_ref_texture_sample_zero_copy`]. The witness decision moves the
/// witness when it is applied; the commit hands back the repeat for the next
/// bind of the same ref (design H6's removal), committed from the post state.
#[derive(Clone)]
pub(crate) struct RefViewPlan {
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    mid: u32,
    view: objects::RefTextureView,
    /// [`crate::runtime::mapper::mapping_window_key_readonly`] when planned.
    window_key: (u32, u64, usize),
    plane: RefViewPlane,
    gpas: std::sync::Arc<Vec<u64>>,
    runs: std::sync::Arc<Vec<crate::backend::vulkan::engine::GuestRun>>,
    gather: crate::runtime::gather_witness::PlannedGather,
    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): whether a commit has proven this
    /// plan by serving a bind from it — set on the repeat [`commit_ref_view_plan`]
    /// hands back, because surviving one bind is the evidence the next binds of
    /// the key will read a window worth copying ([`attach_staged`] gates on it;
    /// a fresh plan's window died unbound with 99.4 % of the eager copies).
    proven: bool,
    /// TEMP (lab, PASS_STAGE, design 8.4): the plan's gather window, staged
    /// into a worker arena by [`attach_staged`]; the serve wraps it and the
    /// engine's CPU-gather arm reads the window instead of gathering.
    staged: Option<crate::backend::vulkan::engine::pass_stage::StagedSlice>,
}

/// Plan the zero-copy bind of a ref-texture view that needs materializing.
///
/// Every step the serial rail takes that could write anything — resolving the
/// surface, paying an owed frame, building an import, filling the window cache,
/// arming the witness — is a [`SampleBail`] here, so a plan exists only for a
/// bind on which the serial rail would have done none of them.
pub(crate) fn plan_ref_view_zero_copy<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
) -> Result<RefViewPlan, SampleBail> {
    if resource.entry.object_type != objects::OBJECT_TYPE_REF_TEXTURE {
        return Err(SampleBail::NotSurface);
    }
    let desc = &resource.descriptor;
    let mid = reims_vgpu_wire::device_desc::ref_texture_header(desc)
        .map(|t5| t5.surface_id.get())
        .unwrap_or(0);
    let view = objects::decode_ref_texture_view(desc).ok_or(SampleBail::NotSurface)?;
    if mid == 0 {
        return Err(SampleBail::NotSurface);
    }
    // `ensure_surface_for_texture_bind`'s warm test; its cold arm mutates.
    let m = state.mappings.get(&mid).ok_or(SampleBail::ColdBacking)?;
    if !(m.mapped && !m.page_entries.is_empty() && m.has_geom) {
        return Err(SampleBail::ColdBacking);
    }
    if !ref_texture_view_requires_materialization(m.has_geom, m.width, m.height, m.format, view) {
        return Err(SampleBail::NotSurface);
    }
    let window_key = crate::runtime::mapper::mapping_window_key_readonly(state, mid)
        .ok_or(SampleBail::ColdWindow)?;
    let plane = ref_view_plane(state, host, mid, view).ok_or(SampleBail::ZeroCopyDeclined)?;
    if !state.pending_writebacks.is_empty() && state.pending_writebacks.owes(mid) {
        return Err(SampleBail::OwedFrame);
    }
    if !sampled_gather_floor_admits(SampledFloorRail::RefTexture, plane.native, plane.span)
        || !guest_run_alias_available(host)
    {
        return Err(SampleBail::ZeroCopyDeclined);
    }
    if crate::runtime::lab_runs::mode() != crate::runtime::lab_runs::Mode::On {
        return Err(SampleBail::ColdWindow);
    }
    let (gpas, runs) =
        crate::runtime::mapper::mapping_window_cached(state, mid, plane.base_off, plane.span)
            .ok_or(SampleBail::ColdWindow)?;
    let gather = crate::runtime::gather_witness::plan_gather(
        state,
        host,
        crate::runtime::gather_witness::GatherRail::RefTexture,
        crate::runtime::gather_witness::GatherKey::Mapping {
            mid,
            base_off: plane.base_off,
        },
        crate::runtime::gather_witness::GatherWindow {
            gpas: &gpas,
            runs: &runs,
            span: plane.span,
            page_size: state.page_size() as usize,
        },
    )
    .ok_or(SampleBail::ColdWitness)?;
    Ok(RefViewPlan {
        task_id,
        texture_ref,
        resource,
        mid,
        view,
        window_key,
        plane,
        gpas,
        runs,
        gather,
        proven: false,
        staged: None,
    })
}

impl RefViewPlan {
    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): a commit proved this plan and no
    /// worker has staged its window yet — the restage job's selection, and the
    /// one plan shape [`attach_staged`] will copy under the default rule.
    pub(crate) fn staging_pending(&self) -> bool {
        self.proven && self.staged.is_none()
    }

    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): whether a worker's window is
    /// attached — the restage job's success test.
    pub(crate) fn has_staged_window(&self) -> bool {
        self.staged.is_some()
    }
}

/// TEMP (lab, PASS_STAGE, design 8.4): wrap a guest-runs serve whose window a
/// worker staged, as the staged request the engine resolves like the
/// guest-runs bind it stands for. `unreachable` on any other variant because
/// every caller built the guest-runs tuple on the line above.
fn wrap_staged(
    request: SampledSourceRequest,
    staged: &crate::backend::vulkan::engine::pass_stage::StagedSlice,
) -> SampledSourceRequest {
    let SampledSourceRequest::GuestRuns(
        src,
        native,
        format,
        planes,
        identity,
        vouch,
        evidence,
        components,
    ) = request
    else {
        unreachable!("a staged wrap is applied to a guest-runs serve");
    };
    SampledSourceRequest::Staged(Box::new(super::StagedSampledRequest {
        src,
        native,
        format,
        planes,
        identity,
        vouch,
        evidence,
        components,
        staged: staged.clone(),
    }))
}

/// TEMP (lab, PASS_STAGE, STAGED_PROVEN): the gather window a plan would
/// stage, for the skipped-copy census; `None` on the plans staging never
/// touches (the surface ladder's own answers, the memo's bytes).
fn staged_window_span(plan: &super::prefetch::StreamPlan) -> Option<u64> {
    match plan {
        super::prefetch::StreamPlan::RefView(p) => Some(p.plane.span),
        super::prefetch::StreamPlan::Linear(p) => Some(p.span),
        super::prefetch::StreamPlan::MapperRef(p) => Some(p.span),
        _ => None,
    }
}

/// TEMP (lab, PASS_STAGE, design 8.4): stage the gather window of a
/// guest-runs plan into the worker's arena -- the copy the engine's
/// CPU-gather arm (`write_staging_from_runs`) would have paid, moved to the
/// planning thread. Plans without a gather window are untouched; an arena
/// refusal leaves the plan as it was (the serial gather serves).
///
/// TEMP (lab, STAGED_PROVEN): under the default rule the window is copied
/// only for a **proven** plan — the repeat a commit handed back after serving
/// one bind, offered here by the next chunk's restage job. A fresh plan's
/// copy was speculation that died with the plan: over the cssscrollglassv
/// window 99.4 % of staged windows were never read (planned 9582, hit 54,
/// 121 GB of copies), because the witness churn a chunk of draws produces
/// invalidates the plan between its planning and its bind. A first bind loses
/// nothing by the gate: unstaged, its commit serves the plain guest-runs
/// request and the engine gathers serially, exactly as it did before the
/// staging rail existed. `REIMS_VGPU_LAB_STAGED_PROVEN=off` restores the
/// eager attach of every plan.
pub(crate) fn attach_staged(
    plan: &mut super::prefetch::StreamPlan,
    stager: &super::pass_stage::Stager,
) {
    if crate::runtime::par::staging::staged_proven() {
        let speculative = match plan {
            super::prefetch::StreamPlan::RefView(p) => !p.proven,
            super::prefetch::StreamPlan::Linear(p) => !p.proven,
            super::prefetch::StreamPlan::MapperRef(p) => !p.proven,
            _ => false,
        };
        if speculative {
            crate::runtime::drain::note_store_route("pass_stage_sampled_skipped_speculative");
            if let Some(span) = staged_window_span(plan) {
                // The direct measurement of the bytes this rule did not copy:
                // skipped windows x their span, against the arena census's
                // allocated_bytes for what it did.
                crate::runtime::drain::note_store_route_n("pass_stage_sampled_skipped_bytes", span);
            }
            return;
        }
    }
    let attached = match plan {
        super::prefetch::StreamPlan::RefView(p) => stager
            .stage_runs(&p.runs, 0, p.plane.span)
            .map(|staged| p.staged = Some(staged))
            .is_some(),
        super::prefetch::StreamPlan::Linear(p) => stager
            .stage_runs(&p.packed_runs, p.layout_offset, p.span)
            .map(|staged| p.staged = Some(staged))
            .is_some(),
        super::prefetch::StreamPlan::MapperRef(p) => stager
            .stage_runs(&p.runs, 0, p.span)
            .map(|staged| p.staged = Some(staged))
            .is_some(),
        _ => false,
    };
    if attached {
        crate::runtime::drain::note_store_route("pass_stage_sampled_planned");
        if crate::runtime::par::staging::staged_proven() {
            crate::runtime::drain::note_store_route("pass_stage_sampled_staged_proven");
        }
    }
}

/// Re-check a ref-view plan at the point the serial rail would run, and with
/// `apply` take it: the field-window note and the routes the serial rail makes,
/// the witness update, and the plane's guest-page window.
///
/// Without `apply` nothing is written or noted, and the answer is the plan's
/// prediction with no page window, for a shadow comparison against the serial
/// rail that runs next. With `apply` the pair's second element is the plan for
/// the next bind of the same ref — the gather repeat, committed from the state
/// this commit left (design H6's removal: one plan per key serves the pass).
pub(crate) fn commit_ref_view_plan<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    plan: &RefViewPlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    apply: bool,
) -> Result<((u32, u32, u32, SampledSourceRequest), Option<RefViewPlan>), SampleInvalid> {
    if plan.task_id != task_id || plan.texture_ref != texture_ref {
        return Err(SampleInvalid::Key);
    }
    if !resource.is_some_and(|r| std::sync::Arc::ptr_eq(r, &plan.resource)) {
        return Err(SampleInvalid::Resource);
    }
    let mid = plan.mid;
    let row_holds = state.mappings.get(&mid).is_some_and(|m| {
        m.mapped
            && !m.page_entries.is_empty()
            && m.has_geom
            && ref_texture_view_requires_materialization(
                m.has_geom, m.width, m.height, m.format, plan.view,
            )
    });
    if !row_holds
        || crate::runtime::mapper::mapping_window_key_readonly(state, mid) != Some(plan.window_key)
        || ref_view_plane(&*state, &*host, mid, plan.view) != Some(plan.plane)
    {
        return Err(SampleInvalid::Mapping);
    }
    if !state.pending_writebacks.is_empty() && state.pending_writebacks.owes(mid) {
        return Err(SampleInvalid::Debt);
    }
    let (base_off, span) = (plan.plane.base_off, plan.plane.span);
    let cached = crate::runtime::mapper::mapping_window_cached(state, mid, base_off, span);
    if !cached.is_some_and(|(gpas, runs)| {
        std::sync::Arc::ptr_eq(&gpas, &plan.gpas) && std::sync::Arc::ptr_eq(&runs, &plan.runs)
    }) {
        return Err(SampleInvalid::Window);
    }
    if !crate::runtime::gather_witness::gather_plan_holds(state, &*host, &plan.gather, &plan.gpas) {
        return Err(SampleInvalid::Witness);
    }
    let view = plan.view;
    let source = |pages, outcome: crate::runtime::gather_witness::GatherOutcome| {
        (
            view.width,
            view.height,
            mid,
            SampledSourceRequest::GuestRuns(
                crate::backend::vulkan::engine::GuestRunSource {
                    runs: std::sync::Arc::clone(&plan.runs),
                    source_offset: 0,
                    total_len: span,
                    row_length_texels: plan.plane.row_length_texels,
                    pages,
                    direct_image: None,
                },
                plan.plane.native,
                plan.plane.sampled_vk_format,
                1,
                Some(LinearSampleIdentity::from(outcome.identity)),
                outcome.vouch,
                outcome.evidence,
                pixel_format::swizzle_identity(),
            ),
        )
    };
    // TEMP (lab, PASS_STAGE): the staged window, on every arm the closure
    // below serves.
    let wrap = |served: (u32, u32, u32, SampledSourceRequest)| match plan.staged.as_ref() {
        Some(st) => (served.0, served.1, served.2, wrap_staged(served.3, st)),
        None => served,
    };
    if !apply {
        return Ok((wrap(source(None, plan.gather.predicted())), None));
    }
    // What the serial rail notes before it decides how to carry the plane.
    if let Some(bpp) = crate::protocol::pixel_format::bytes_per_pixel(view.pixel_format) {
        if let Some((field_off, bpr, _)) = state.mappings.get(&mid).and_then(|m| {
            crate::runtime::mapping_write::ref_texture_sample_window(
                m,
                view.plane_index,
                view.width,
                view.height,
                view.pixel_format,
            )
        }) {
            crate::runtime::scanout::note_sampled_surface_field_window(
                &*state,
                &*host,
                mid,
                texture_ref,
                "ref_texture_view",
                crate::runtime::scanout::SampledFieldWindow {
                    width: view.width,
                    height: view.height,
                    format: u32::from(view.pixel_format),
                    base_off: field_off,
                    bpr,
                    bpp,
                },
            );
        }
    }
    if plan.plane.import == ImportGate::StandingRefusal {
        crate::runtime::drain::note_store_route("zc_packed_alias_standing_refusal");
    }
    crate::runtime::drain::note_store_route("lab_rf_wc_hit");
    let (outcome, repeat) = crate::runtime::gather_witness::commit_gather_repeatable(
        state,
        &*host,
        &plan.gather,
        &plan.gpas,
        &plan.runs,
    )
    .ok_or(SampleInvalid::Witness)?;
    let page = state.page_size();
    let pages = guest_page_window(host, &plan.gpas, page, base_off % page, span);
    let mut next = plan.clone();
    next.gather = repeat;
    // TEMP (lab, STAGED_PROVEN): the repeat survived this commit, so its
    // window is the one worth staging (the restage job this hands to).
    next.proven = true;
    Ok((wrap(source(pages, outcome)), Some(next)))
}

/// Whether the GVA-resident rung that sits above the linear zero-copy rail
/// (`try_gva_resident_sample`) may serve this bind, from state this call only
/// reads — the question a linear plan must answer before it plans the rung
/// below, because a bind that rung serves is a `Target` and the plan would
/// answer `GuestRuns`.
///
/// Conservative by construction: every case the read-only ledger and witness
/// cannot prove a refusal for answers `true`, and the plan bails to the serial
/// ladder that runs the rung itself. The provable refusals are the two the
/// serial rung takes without mutating:
///
/// * a declared plane whose window the store witness will not call quiet
///   (`reach` answers anything but its `Quiet` verdict) — the serial rung
///   refuses with `Wrote` and falls through to the linear rail;
/// * `may_bind_resident == false`, which the caller checks itself and the
///   serial ladder honours by skipping the rung entirely.
///
/// The unprovable cases: an undeclared plane (the serial declaration may mint
/// a generation or keep one — a kept generation can already have witness
/// state), and a quiet witness (the serial rung asks readiness and may serve).
fn linear_resident_rung_above_may_serve<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    row_stride: u32,
    w: u32,
    h: u32,
    format: u16,
) -> bool {
    // The span the identity is declared at, spelled as `gva_span_identity`
    // spells it: row_stride × height.
    let span_bytes = u64::from(row_stride).saturating_mul(u64::from(h));
    let key = crate::runtime::writeback_debt::GvaResourceKey {
        task_id,
        texture_ref,
    };
    let Some(generation) =
        crate::runtime::writeback_debt::gva_plane_generation(state, key, gva, span_bytes)
    else {
        // Never declared, declared at another span, or declared without
        // walkable pages: the serial path's declaration may keep a generation
        // the witness already tracks, which this cannot rule out.
        return true;
    };
    let identity = crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva,
        width: w,
        height: h,
        generation,
        format: gva_resident_format(format),
    };
    // An unpaid Store over this window makes the resident authoritative; the
    // serial rung's answer then turns on readiness alone.
    if crate::backend::vulkan::gva_window(&identity).is_some_and(|window| {
        crate::runtime::writeback_debt::gva_resident_authoritative(state, window)
    }) {
        return true;
    }
    let Some(witness_key) = crate::backend::vulkan::gva_witness_key(&identity) else {
        return true;
    };
    // Quiet is the one answer the serial rung carries past; every other verdict
    // is a refusal that lands it on this plan's rail.
    crate::runtime::gva_store_witness::reach(state, host, witness_key).is_quiet()
}

/// A linear texture's zero-copy bind planned off the drain thread: the warm
/// branch of [`try_linear_sample_zero_copy`], which is the whole of that
/// function's packed-allocation arms.
///
/// What the serial rail does before those arms that a plan cannot: pay the
/// texture's and the allocation's debts (the plan refuses them, `OwedFrame`),
/// and answer the GVA-resident rung above (the plan refuses the binds that rung
/// may serve, `UpperRung`). What it does that a plan repeats at commit: the
/// two debt predicates, and the identity of the retained packed allocation.
///
/// The direct single-plane arm ([`direct_linear_sample_from_packed`]) moves
/// nothing at all — its plan serves every bind of the ref that still
/// validates. The multi-plane gather arm arms the witness through
/// [`crate::runtime::gather_witness::plan_gather`], so a commit hands back the
/// repeat for the next bind of the same key (design H6's removal).
#[derive(Clone)]
pub(crate) struct LinearPlan {
    task_id: u32,
    texture_ref: u32,
    may_bind_resident: bool,
    resource: std::sync::Arc<crate::model::TaskResource>,
    /// The level-0 window the serial rail derives from the same descriptor.
    gva: u64,
    w: u32,
    h: u32,
    layout_offset: u64,
    row_stride: u64,
    span: u64,
    row_length_texels: u32,
    native: TexelLayout,
    sampled_vk_format: ash::vk::Format,
    native_components: pixel_format::SwizzlePlan,
    planes: u32,
    /// The allocation the packed resource retains — what the commit re-asks
    /// `packed_available` with.
    allocation: BufferBacking,
    /// The retained packed allocation's identity, re-checked at commit.
    packed_head: u64,
    packed_runs: std::sync::Arc<Vec<crate::backend::vulkan::engine::GuestRun>>,
    packed_pages: std::sync::Arc<Vec<crate::runtime::guest_ram_map::GuestWindowRun>>,
    /// The gather arm's witness window, cut from the packed allocation exactly
    /// as the serial rail cuts it; `None` on the direct arm.
    gather: Option<crate::runtime::gather_witness::PlannedGather>,
    witness_gpas: Vec<u64>,
    witness_run: Option<crate::backend::vulkan::engine::GuestRun>,
    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): see [`RefViewPlan::proven`].
    proven: bool,
    /// TEMP (lab, PASS_STAGE, design 8.4): see [`RefViewPlan::staged`].
    staged: Option<crate::backend::vulkan::engine::pass_stage::StagedSlice>,
}

/// Plan the linear zero-copy bind of one texture ref from state this call only
/// reads. Every point where the serial rail would mutate — resolve a packed
/// allocation, pay a debt, answer the resident rung above — is a
/// [`SampleBail`] and the serial rail serves the bind unchanged.
pub(crate) fn plan_linear_zero_copy<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    may_bind_resident: bool,
) -> Result<LinearPlan, SampleBail> {
    use crate::runtime::decode::resource::Descriptor;
    if !matches!(
        resource.entry.object_type,
        crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE
            | crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS
    ) {
        return Err(SampleBail::NotSurface);
    }
    let tex = match resource.decoded() {
        Ok(Descriptor::Texture(tex)) => tex,
        _ => return Err(SampleBail::NotSurface),
    };
    // The same format gates the serial rail applies, quiet here so the serial
    // rail's decline routes are noted by the serial rail that takes them.
    let declared_format = tex
        .declared_pixel_format()
        .ok_or(SampleBail::ZeroCopyDeclined)?;
    let (native, sampled_vk_format, native_components) =
        match translate::pixel::sampled_pixels(declared_format) {
            Ok((layout, _decline, components))
                if crate::backend::vulkan::engine::supports_sampled_layout_bind(layout) =>
            {
                let format = translate::pixel::translate(declared_format)
                    .ok()
                    .ok_or(SampleBail::ZeroCopyDeclined)?
                    .vk;
                (layout, format, components)
            }
            _ => return Err(SampleBail::ZeroCopyDeclined),
        };
    let (gva, layout) = tex
        .level_gva(0, state.page_shift)
        .ok_or(SampleBail::ZeroCopyDeclined)?;
    let (w, h) = (layout.width, layout.height);
    if w == 0 || h == 0 {
        return Err(SampleBail::ZeroCopyDeclined);
    }
    let (span, row_length_texels) =
        strided_level_extent(layout, native.block()).ok_or(SampleBail::ZeroCopyDeclined)?;
    let planes = layout.planes();
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return Err(SampleBail::ZeroCopyDeclined);
    }
    // The rung above (`try_gva_resident_sample`): a bind it serves is a Target
    // and must stay the serial ladder's answer.
    if may_bind_resident
        && linear_resident_rung_above_may_serve(
            state,
            host,
            task_id,
            texture_ref,
            gva,
            u32::try_from(layout.row_stride).map_err(|_| SampleBail::ZeroCopyDeclined)?,
            w,
            h,
            tex.pixel_format,
        )
    {
        return Err(SampleBail::UpperRung);
    }
    // The plan never pays (H7): both debts the serial rail pays before it
    // touches the allocation are refusals here.
    if crate::runtime::writeback_debt::owes_texture(state, task_id, texture_ref) {
        return Err(SampleBail::OwedFrame);
    }
    let allocation_gva = tex
        .allocation_base_gva(state.page_shift)
        .filter(|_| tex.allocation_size != 0)
        .ok_or(SampleBail::ColdPacked)?;
    let allocation = BufferBacking {
        gva: allocation_gva,
        size: tex.allocation_size,
    };
    if state
        .pending_writebacks
        .owes_gva_overlapping(task_id, allocation.gva, allocation.size)
    {
        return Err(SampleBail::OwedFrame);
    }
    // Copied out before the resource moves: the construction below cannot
    // borrow the decode it is built from.
    let (layout_offset, row_stride) = (layout.offset, layout.row_stride);
    // The warm packed allocation. `ensure_packed_resource` resolves (mutates)
    // on a miss, so a plan takes only what `bound_buffers` already holds.
    let packed = state
        .bound_buffers
        .packed_available(task_id, texture_ref, allocation.gva, allocation.size)
        .ok_or(SampleBail::ColdPacked)?;
    // The direct single-plane arm: pure construction from the borrowed
    // allocation, verified feasible here and rebuilt at commit.
    if planes == 1 {
        if direct_linear_sample_from_packed(
            packed,
            layout_offset,
            row_stride,
            span,
            row_length_texels,
            native,
            sampled_vk_format,
            native_components,
            resource.lifetime_ref(),
        )
        .is_none()
        {
            return Err(SampleBail::ColdPacked);
        }
        return Ok(LinearPlan {
            task_id,
            texture_ref,
            may_bind_resident,
            resource,
            gva,
            w,
            h,
            layout_offset,
            row_stride,
            span,
            row_length_texels,
            native,
            sampled_vk_format,
            native_components,
            planes,
            allocation,
            packed_head: packed.head,
            packed_runs: std::sync::Arc::clone(&packed.runs),
            packed_pages: std::sync::Arc::clone(&packed.pages),
            gather: None,
            witness_gpas: Vec::new(),
            witness_run: None,
            proven: false,
            staged: None,
        });
    }
    // The multi-plane gather arm, over the import's own coordinates exactly as
    // the serial rail cuts them.
    if !sampled_gather_floor_pure(native, span) {
        return Err(SampleBail::ZeroCopyDeclined);
    }
    let page = state.page_size();
    let packed_offset = packed
        .head
        .checked_add(layout_offset)
        .ok_or(SampleBail::ColdPacked)?;
    let first_page = usize::try_from(packed_offset / page).map_err(|_| SampleBail::ColdPacked)?;
    let head_off = packed_offset % page;
    let page_count = usize::try_from(head_off.checked_add(span).ok_or(SampleBail::ColdPacked)?)
        .map_err(|_| SampleBail::ColdPacked)?
        .div_ceil(page as usize);
    let witness_gpas = packed
        .gpas
        .get(
            first_page
                ..first_page
                    .checked_add(page_count)
                    .ok_or(SampleBail::ColdPacked)?,
        )
        .ok_or(SampleBail::ColdPacked)?
        .to_vec();
    let witness_run = crate::backend::vulkan::engine::GuestRun::in_mapping(
        packed.import.host_base(),
        packed.import.len(),
        packed_offset,
        span,
    )
    .ok_or(SampleBail::ColdPacked)?;
    let witness_runs = [witness_run];
    let gather = crate::runtime::gather_witness::plan_gather(
        state,
        host,
        crate::runtime::gather_witness::GatherRail::Linear,
        crate::runtime::gather_witness::GatherKey::TaskGva { task_id, gva },
        crate::runtime::gather_witness::GatherWindow {
            gpas: &witness_gpas,
            runs: &witness_runs,
            span,
            page_size: page as usize,
        },
    )
    .ok_or(SampleBail::ColdWitness)?;
    Ok(LinearPlan {
        task_id,
        texture_ref,
        may_bind_resident,
        resource,
        gva,
        w,
        h,
        layout_offset,
        row_stride,
        span,
        row_length_texels,
        native,
        sampled_vk_format,
        native_components,
        planes,
        allocation,
        packed_head: packed.head,
        packed_runs: std::sync::Arc::clone(&packed.runs),
        packed_pages: std::sync::Arc::clone(&packed.pages),
        gather: Some(gather),
        witness_gpas,
        witness_run: Some(witness_run),
        proven: false,
        staged: None,
    })
}

impl LinearPlan {
    /// The serve both arms answer, from the witness outcome the gather arm
    /// produced (or the direct arm's witness-free request).
    fn source(
        &self,
        request: crate::backend::vulkan::engine::GuestRunSource,
        outcome: Option<crate::runtime::gather_witness::GatherOutcome>,
    ) -> (u32, u32, u32, SampledSourceRequest) {
        (
            self.w,
            self.h,
            0,
            SampledSourceRequest::GuestRuns(
                request,
                self.native,
                self.sampled_vk_format,
                self.planes,
                outcome.map(|seen| LinearSampleIdentity::from(seen.identity)),
                outcome.map_or(crate::runtime::gather_witness::GatherVouch::Fresh, |seen| {
                    seen.vouch
                }),
                outcome.map_or(None, |seen| seen.evidence),
                self.native_components,
            ),
        )
    }

    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): see [`RefViewPlan::staging_pending`].
    pub(crate) fn staging_pending(&self) -> bool {
        self.proven && self.staged.is_none()
    }

    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): see
    /// [`RefViewPlan::has_staged_window`].
    pub(crate) fn has_staged_window(&self) -> bool {
        self.staged.is_some()
    }
}

/// Re-check a linear plan at the point the serial rail would run
/// [`try_linear_sample_zero_copy`], and with `apply` take it: the debt
/// predicates, the rung above, the packed allocation's identity, and (gather
/// arm) the witness plan. The Ok pair's second element is the plan for the
/// next bind of the same ref — the direct arm's own plan again (it moved
/// nothing), the gather arm's repeat (design H6).
///
/// Without `apply` nothing is written or noted, and the answer is the plan's
/// prediction with no witness outcome applied, for a shadow comparison against
/// the serial rail that runs next.
pub(crate) fn commit_linear_plan<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    plan: &LinearPlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
    apply: bool,
) -> Result<((u32, u32, u32, SampledSourceRequest), Option<LinearPlan>), SampleInvalid> {
    if plan.task_id != task_id
        || plan.texture_ref != texture_ref
        || plan.may_bind_resident != may_bind_resident
    {
        return Err(SampleInvalid::Key);
    }
    if !resource.is_some_and(|r| std::sync::Arc::ptr_eq(r, &plan.resource)) {
        return Err(SampleInvalid::Resource);
    }
    // The predicates the plan refused through, repeated: a debt armed since the
    // plan is the drain's to pay, on the serial rail.
    if crate::runtime::writeback_debt::owes_texture(state, task_id, texture_ref)
        || state.pending_writebacks.owes_gva_overlapping(
            task_id,
            plan.allocation.gva,
            plan.allocation.size,
        )
    {
        return Err(SampleInvalid::Debt);
    }
    // The rung above, repeated: it became servable since the plan (a Store
    // armed the witness, a resident turned ready) — the serial answer changed
    // shape above this rail.
    let tex = match plan.resource.decoded() {
        Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => tex,
        _ => return Err(SampleInvalid::Resource),
    };
    if may_bind_resident
        && linear_resident_rung_above_may_serve(
            &*state,
            &*host,
            task_id,
            texture_ref,
            plan.gva,
            u32::try_from(plan.row_stride).map_err(|_| SampleInvalid::Resource)?,
            plan.w,
            plan.h,
            tex.pixel_format,
        )
    {
        return Err(SampleInvalid::UpperRung);
    }
    let packed = state
        .bound_buffers
        .packed_available(
            task_id,
            texture_ref,
            plan.allocation.gva,
            plan.allocation.size,
        )
        .ok_or(SampleInvalid::Packed)?;
    if plan.packed_head != packed.head
        || !std::sync::Arc::ptr_eq(&plan.packed_runs, &packed.runs)
        || !std::sync::Arc::ptr_eq(&plan.packed_pages, &packed.pages)
    {
        return Err(SampleInvalid::Packed);
    }
    // Cloned out before the witness commit takes `state` mutably: the serve
    // below names the same allocation the identity check just proved.
    let (packed_runs, packed_pages) = (
        std::sync::Arc::clone(&packed.runs),
        std::sync::Arc::clone(&packed.pages),
    );
    let guest_runs = crate::backend::vulkan::engine::GuestRunSource {
        runs: packed_runs,
        source_offset: plan.layout_offset,
        total_len: plan.span,
        row_length_texels: plan.row_length_texels,
        pages: Some(packed_pages),
        direct_image: None,
    };
    if plan.gather.is_none() {
        // The direct arm: rebuilt from the same borrowed allocation, moving
        // nothing — the plan serves again for the next bind of the ref.
        let request = direct_linear_sample_from_packed(
            packed,
            plan.layout_offset,
            plan.row_stride,
            plan.span,
            plan.row_length_texels,
            plan.native,
            plan.sampled_vk_format,
            plan.native_components,
            plan.resource.lifetime_ref(),
        )
        .ok_or(SampleInvalid::Packed)?;
        if apply && plan.span < SAMPLED_GATHER_MIN_BYTES {
            crate::runtime::drain::note_store_route("zc_lin_direct_under_floor_span");
        }
        let SampledSourceRequest::GuestRuns(source, ..) = request else {
            return Err(SampleInvalid::Packed);
        };
        // TEMP (lab, PASS_STAGE): the staged window, when one was attached.
        let served = plan.source(source, None);
        // TEMP (lab, STAGED_PROVEN): the direct arm moved nothing, and this
        // commit served the bind — the plan it hands back is proven (its
        // window, once a worker stages it, serves every later bind).
        let mut next = plan.clone();
        next.proven = true;
        return Ok((wrap_linear(plan, served), apply.then_some(next)));
    }
    let gather = plan.gather.as_ref().expect("checked above");
    let run = plan.witness_run.expect("the gather arm carries its run");
    if !apply {
        return Ok((
            wrap_linear(plan, plan.source(guest_runs, Some(gather.predicted()))),
            None,
        ));
    }
    let (outcome, repeat) = crate::runtime::gather_witness::commit_gather_repeatable(
        state,
        &*host,
        gather,
        &plan.witness_gpas,
        std::slice::from_ref(&run),
    )
    .ok_or(SampleInvalid::Witness)?;
    let mut next = plan.clone();
    next.gather = Some(repeat);
    // TEMP (lab, STAGED_PROVEN): see [`commit_ref_view_plan`].
    next.proven = true;
    Ok((
        wrap_linear(plan, plan.source(guest_runs, Some(outcome))),
        Some(next),
    ))
}

/// TEMP (lab, PASS_STAGE): a linear plan's serve, wrapped with its staged
/// window when one was attached (see [`attach_staged`]).
fn wrap_linear(
    plan: &LinearPlan,
    served: (u32, u32, u32, SampledSourceRequest),
) -> (u32, u32, u32, SampledSourceRequest) {
    match plan.staged.as_ref() {
        Some(st) => (served.0, served.1, served.2, wrap_staged(served.3, st)),
        None => served,
    }
}

/// The surface id a sampled ref names, by the same classification
/// [`plan_sampled_source`] applies — the ref-texture's descriptor surface, the
/// backing ref's own id, or a mapper-ref-texture's registered mapping.
fn sampled_surface_of(
    resource: &std::sync::Arc<crate::model::TaskResource>,
    texture_ref: u32,
) -> Result<u32, SampleBail> {
    let entry = resource.entry;
    if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
        let sid = reims_vgpu_wire::device_desc::ref_texture_header(&resource.descriptor)
            .map(|t5| t5.surface_id.get())
            .unwrap_or(0);
        return (sid != 0).then_some(sid).ok_or(SampleBail::NotSurface);
    }
    if entry.object_type == objects::OBJECT_TYPE_BACKING {
        return Ok(texture_ref);
    }
    if entry.object_type == crate::runtime::decode::resource::OBJECT_TYPE_MAPPER_REF_TEXTURE {
        return resource
            .registered_mapper_ref_texture_mapping()
            .ok_or(SampleBail::ColdMapperRef);
    }
    Err(SampleBail::NotSurface)
}

/// Whether the mapping's direct image — the arm of
/// [`try_mapper_ref_texture_sample_zero_copy`] that resolves (or borrows) the
/// mapping's contiguous import — may serve this bind, from state this call only
/// reads. Provable refusals only: an unstable host (the arm answers `None`
/// before it resolves anything) and a contig view this page-list generation has
/// already refused (the cached refusal answers without a host call — and a
/// *live* view outranks the cached refusal, exactly as
/// `ensure_contig_view_with_pages` orders the two). Everything else — no view
/// yet, a view with no import, an import the alignment gate would rebuild — is
/// the serial rail's to resolve, so the plan declines.
fn mapper_direct_rung_above_may_serve<M: HostOps>(state: &DeviceState, host: &M, mid: u32) -> bool {
    if !host.map_pages_stable() {
        return false;
    }
    let Some(m) = state.mappings.get(&mid) else {
        return false;
    };
    if m.contig_ptr != 0 {
        // A view the serial resolve would borrow or rebuild an import over;
        // either way the direct arm answers before the gather arm runs.
        return true;
    }
    m.contig_refused_gen != Some(m.map_generation)
}

/// A mapper-ref surface's zero-copy bind planned off the drain thread: the
/// window+gather arm of [`try_mapper_ref_texture_sample_zero_copy`] over the
/// mapping's cached window ([`crate::runtime::mapper::mapping_window_cached`])
/// and [`crate::runtime::gather_witness::plan_gather`] on
/// `GatherRail::MapperRefTexture`.
///
/// The direct image arm above it is refused wherever it may serve
/// ([`SampleBail::UpperRung`], conservatively — resolving it mutates), the
/// mapping's owed frame is refused ([`SampleBail::OwedFrame`]; the serial rail
/// pays before it looks at the pages), and a cold window cache is refused
/// ([`SampleBail::ColdWindow`]; the serial resolve fills it). A commit hands
/// back the repeat for the next bind of the same window.
#[derive(Clone)]
pub(crate) struct MapperRefPlan {
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    mid: u32,
    w: u32,
    h: u32,
    base_off: u64,
    bpr: u64,
    span: u64,
    row_length_texels: u32,
    native: TexelLayout,
    sampled_vk_format: ash::vk::Format,
    window_key: (u32, u64, usize),
    gpas: std::sync::Arc<Vec<u64>>,
    runs: std::sync::Arc<Vec<crate::backend::vulkan::engine::GuestRun>>,
    gather: crate::runtime::gather_witness::PlannedGather,
    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): see [`RefViewPlan::proven`].
    proven: bool,
    /// TEMP (lab, PASS_STAGE, design 8.4): see [`RefViewPlan::staged`].
    staged: Option<crate::backend::vulkan::engine::pass_stage::StagedSlice>,
}

/// Plan the mapper-ref zero-copy bind of one sampled ref. Every point where
/// the serial rail would mutate — pay the mapping's debt, resolve the contig
/// import, fill the window cache, arm the witness — is a [`SampleBail`] and
/// the serial rail serves the bind unchanged.
pub(crate) fn plan_mapper_ref_zero_copy<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
) -> Result<MapperRefPlan, SampleBail> {
    let mid = sampled_surface_of(&resource, texture_ref)?;
    let m = state.mappings.get(&mid).ok_or(SampleBail::ColdBacking)?;
    if !(m.mapped && !m.page_entries.is_empty() && m.has_geom) {
        return Err(SampleBail::ColdBacking);
    }
    let (w, h) = (m.width, m.height);
    if w == 0 || h == 0 {
        return Err(SampleBail::NoGeometry);
    }
    let format = if m.format != 0 {
        m.format
    } else {
        pixel_format::MTL_FORMAT_BGRA8_UNORM
    };
    // The same four-byte-colour + identity-swizzle gate the serial rail applies.
    let native = match translate::pixel::sampled_pixels(format) {
        Ok((layout, _decline, components))
            if layout.is_four_byte_color() && pixel_format::swizzle_is_identity(&components) =>
        {
            layout
        }
        _ => return Err(SampleBail::ZeroCopyDeclined),
    };
    let sampled_vk_format = translate::pixel::translate(format)
        .ok()
        .ok_or(SampleBail::ZeroCopyDeclined)?
        .vk;
    let (base_off, bpr_u32, _) =
        crate::runtime::mapping_write::mapper_ref_texture_sample_window(m, w, h, format)
            .ok_or(SampleBail::ZeroCopyDeclined)?;
    let bpr = u64::from(bpr_u32);
    let (span, row_length_texels) =
        strided_window_extent(w, h, native.bytes_per_texel() as u64, bpr)
            .ok_or(SampleBail::ZeroCopyDeclined)?;
    // The plan never pays (H7).
    if crate::runtime::writeback_debt::owes_mapping(state, mid) {
        return Err(SampleBail::OwedFrame);
    }
    // The direct image arm above: refused wherever it may serve or resolve.
    if mapper_direct_rung_above_may_serve(state, host, mid) {
        return Err(SampleBail::UpperRung);
    }
    if !sampled_gather_floor_pure(native, span) {
        return Err(SampleBail::ZeroCopyDeclined);
    }
    if crate::runtime::lab_runs::mode() != crate::runtime::lab_runs::Mode::On {
        return Err(SampleBail::ColdWindow);
    }
    let window_key = crate::runtime::mapper::mapping_window_key_readonly(state, mid)
        .ok_or(SampleBail::ColdWindow)?;
    let (gpas, runs) = crate::runtime::mapper::mapping_window_cached(state, mid, base_off, span)
        .ok_or(SampleBail::ColdWindow)?;
    if !guest_run_alias_available(host) {
        return Err(SampleBail::ZeroCopyDeclined);
    }
    let gather = crate::runtime::gather_witness::plan_gather(
        state,
        host,
        crate::runtime::gather_witness::GatherRail::MapperRefTexture,
        crate::runtime::gather_witness::GatherKey::Mapping { mid, base_off },
        crate::runtime::gather_witness::GatherWindow {
            gpas: &gpas,
            runs: &runs,
            span,
            page_size: state.page_size() as usize,
        },
    )
    .ok_or(SampleBail::ColdWitness)?;
    Ok(MapperRefPlan {
        task_id,
        texture_ref,
        resource,
        mid,
        w,
        h,
        base_off,
        bpr,
        span,
        row_length_texels,
        native,
        sampled_vk_format,
        window_key,
        gpas,
        runs,
        gather,
        proven: false,
        staged: None,
    })
}

impl MapperRefPlan {
    fn source(
        &self,
        pages: Option<std::sync::Arc<Vec<crate::runtime::guest_ram_map::GuestWindowRun>>>,
        outcome: crate::runtime::gather_witness::GatherOutcome,
    ) -> (u32, u32, u32, SampledSourceRequest) {
        (
            self.w,
            self.h,
            self.mid,
            SampledSourceRequest::GuestRuns(
                crate::backend::vulkan::engine::GuestRunSource {
                    runs: std::sync::Arc::clone(&self.runs),
                    source_offset: 0,
                    total_len: self.span,
                    row_length_texels: self.row_length_texels,
                    pages,
                    direct_image: None,
                },
                self.native,
                self.sampled_vk_format,
                1,
                Some(LinearSampleIdentity::from(outcome.identity)),
                outcome.vouch,
                outcome.evidence,
                pixel_format::swizzle_identity(),
            ),
        )
    }

    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): see [`RefViewPlan::staging_pending`].
    pub(crate) fn staging_pending(&self) -> bool {
        self.proven && self.staged.is_none()
    }

    /// TEMP (lab, PASS_STAGE, STAGED_PROVEN): see
    /// [`RefViewPlan::has_staged_window`].
    pub(crate) fn has_staged_window(&self) -> bool {
        self.staged.is_some()
    }
}

/// Whether the surface ladder's resident rung — the rung every mapper-ref and
/// memo plan falls past, because the resident was absent (or the guest had
/// replaced it) when the ladder planned — has become servable since the plan:
/// a warm lease where there was none, or a readiness the registry has
/// published since (a render the recorder was still holding when the pool
/// planned).
///
/// This is the readiness nondeterminism the design names: a plan reads the
/// pre-state of a recorder job, the serial ladder the post-state, and "both
/// are versions, validate decides". The surface plans re-check their backing
/// for exactly this; the page-rung plans must too, or a bind whose serial
/// answer became a `Target` keeps answering `GuestRuns`/`Bytes`.
///
/// Cheap by design: the readiness side table is versioned and lock-free, and
/// the lease is the resource's own map. The currency question is *not* asked
/// here — that is the sync hypervisor query the plan exists to avoid; when
/// this fires the bind yields to the serial ladder, which pays it and serves
/// whichever rung it serves.
fn surface_resident_appeared(
    state: &DeviceState,
    resource: &std::sync::Arc<crate::model::TaskResource>,
    mid: u32,
    w: u32,
    h: u32,
) -> bool {
    let identity = crate::backend::vulkan::present_identity::surface_identity(state, mid, w, h);
    if resource_type_owns_surface_resident(resource.entry.object_type)
        && resource_lease::resident_target_backing_warm(resource, &identity).is_some()
    {
        return true;
    }
    crate::backend::vulkan::engine::resident_content_backing(&identity)
        != ResidentContentBacking::NotReady
}

/// Re-check a mapper-ref plan at the point the serial rail would run, and with
/// `apply` take it: the mapping row, the window key and its cached window, the
/// debt and the direct arm above, and the witness plan — which commits through
/// [`crate::runtime::gather_witness::commit_gather_repeatable`], so the Ok
/// pair's second element is the repeat for the next bind of the same window.
pub(crate) fn commit_mapper_ref_plan<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    plan: &MapperRefPlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    apply: bool,
) -> Result<((u32, u32, u32, SampledSourceRequest), Option<MapperRefPlan>), SampleInvalid> {
    if plan.task_id != task_id || plan.texture_ref != texture_ref {
        return Err(SampleInvalid::Key);
    }
    if !resource.is_some_and(|r| std::sync::Arc::ptr_eq(r, &plan.resource)) {
        return Err(SampleInvalid::Resource);
    }
    let mid = plan.mid;
    let row_holds = state.mappings.get(&mid).is_some_and(|m| {
        m.mapped
            && !m.page_entries.is_empty()
            && m.has_geom
            && m.width == plan.w
            && m.height == plan.h
    });
    let window_holds = state.mappings.get(&mid).and_then(|m| {
        let format = if m.format != 0 {
            m.format
        } else {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        };
        crate::runtime::mapping_write::mapper_ref_texture_sample_window(m, plan.w, plan.h, format)
            .map(|(base_off, bpr, _)| (base_off, u64::from(bpr)))
    });
    if !row_holds
        || window_holds != Some((plan.base_off, plan.bpr))
        || crate::runtime::mapper::mapping_window_key_readonly(state, mid) != Some(plan.window_key)
    {
        return Err(SampleInvalid::Mapping);
    }
    if crate::runtime::writeback_debt::owes_mapping(state, mid) {
        return Err(SampleInvalid::Debt);
    }
    if mapper_direct_rung_above_may_serve(&*state, &*host, mid) {
        return Err(SampleInvalid::UpperRung);
    }
    // The resident rung above the whole page ladder: the plan fell past it
    // when the surface ladder planned, and a readiness that appeared since
    // changes the serial answer's shape above this rail.
    if surface_resident_appeared(&*state, &plan.resource, mid, plan.w, plan.h) {
        return Err(SampleInvalid::UpperRung);
    }
    let cached =
        crate::runtime::mapper::mapping_window_cached(state, mid, plan.base_off, plan.span);
    if !cached.is_some_and(|(gpas, runs)| {
        std::sync::Arc::ptr_eq(&gpas, &plan.gpas) && std::sync::Arc::ptr_eq(&runs, &plan.runs)
    }) {
        return Err(SampleInvalid::Window);
    }
    if !apply {
        // TEMP (lab, PASS_STAGE): the staged window, when one was attached.
        return Ok((
            wrap_mapper(plan, plan.source(None, plan.gather.predicted())),
            None,
        ));
    }
    crate::runtime::drain::note_store_route("t11rung_zero_copy");
    let (outcome, repeat) = crate::runtime::gather_witness::commit_gather_repeatable(
        state,
        &*host,
        &plan.gather,
        &plan.gpas,
        &plan.runs,
    )
    .ok_or(SampleInvalid::Witness)?;
    let page = state.page_size();
    let pages = guest_page_window(host, &plan.gpas, page, plan.base_off % page, plan.span);
    let mut next = plan.clone();
    next.gather = repeat;
    // TEMP (lab, STAGED_PROVEN): see [`commit_ref_view_plan`].
    next.proven = true;
    Ok((wrap_mapper(plan, plan.source(pages, outcome)), Some(next)))
}

/// TEMP (lab, PASS_STAGE): a mapper-ref plan's serve, wrapped with its staged
/// window when one was attached (see [`attach_staged`]).
fn wrap_mapper(
    plan: &MapperRefPlan,
    served: (u32, u32, u32, SampledSourceRequest),
) -> (u32, u32, u32, SampledSourceRequest) {
    match plan.staged.as_ref() {
        Some(st) => (served.0, served.1, served.2, wrap_staged(served.3, st)),
        None => served,
    }
}

/// TEMP (lab, step 2): the memo-fill byte budget one planning chunk may spend,
/// `REIMS_VGPU_LAB_PASS_PREP_MEMO=KB` (default 4096 KiB). The serial memo has
/// no budget — this bounds the memcpy a chunk's workers add, not the memo
/// itself (the memo's own byte cap still governs what it retains).
pub(crate) fn memo_fill_chunk_limit() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_PASS_PREP_MEMO")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(4096)
            .saturating_mul(1024)
    })
}

/// The mapping fields a memo plan's answer depends on, re-checked at commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemoReads {
    map_generation: u32,
    width: u32,
    height: u32,
    format: u16,
    mapped: bool,
    has_geom: bool,
    page_entries: usize,
    mapping_internal: u64,
    contig_ptr: usize,
    contig_refused_gen: Option<u32>,
    guest_write_token: u64,
    content_generation: u32,
}

impl MemoReads {
    fn of(state: &DeviceState, mid: u32) -> Option<Self> {
        let m = state.mappings.get(&mid)?;
        Some(Self {
            map_generation: m.map_generation,
            width: m.width,
            height: m.height,
            format: m.format,
            mapped: m.mapped,
            has_geom: m.has_geom,
            page_entries: m.page_entries.len(),
            mapping_internal: m.mapping_internal,
            contig_ptr: m.contig_ptr,
            contig_refused_gen: m.contig_refused_gen,
            guest_write_token: m.guest_write_token,
            content_generation: m.content_generation,
        })
    }
}

/// What a planned memo bind serves: the entry the plan proved unchanged (a
/// hit, serving the memo's `Arc` and identity), or the entry the plan built
/// from the rows it read (a fill, inserted at commit).
#[derive(Clone)]
enum MemoServe {
    Hit {
        rgba: std::sync::Arc<Vec<u8>>,
        generation: u64,
        layout: TexelLayout,
    },
    Fill {
        native: Vec<u8>,
        rgba: std::sync::Arc<Vec<u8>>,
        generation: u64,
        layout: TexelLayout,
        entry_bytes: usize,
    },
}

/// A mapper-ref surface's guest-memo bind planned off the drain thread: the
/// rung under the zero-copy rail, whose serial cost is a whole-surface guest
/// read + compare per bind.
///
/// The plan reads the surface's native rows itself, through the cached window
/// runs' stable host pointers — the same bytes the serial fast path reads
/// after its settles — and compares them against the memo:
///
/// * unchanged → a hit: the memo's `Arc` and content identity, no state moved;
/// * changed or absent → a fill: converted rows plus a fresh generation,
///   carried as the fact the commit inserts into the memo (the serial rail's
///   insert, done off the drain thread), against the chunk's byte budget.
///
/// The coherence the serial rail buys with its settles is gated instead: no
/// debt for the mapping (the payment is the drain's), no submitted writeback
/// pending over the window's pages, and no device write into them since the
/// epoch taken before the read. A guest write the plan's own compare did not
/// see is bounded by the mapping token's lagged generation, re-checked at
/// commit — the same plan/commit standard every gather plan rests on (H5).
#[derive(Clone)]
pub(crate) struct MemoPlan {
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    mid: u32,
    w: u32,
    h: u32,
    stride: u32,
    span: u64,
    base_off: u64,
    sample_fmt: u16,
    window_key: (u32, u64, usize),
    gpas: std::sync::Arc<Vec<u64>>,
    runs: std::sync::Arc<Vec<crate::backend::vulkan::engine::GuestRun>>,
    reads: MemoReads,
    observed_gen: Option<u64>,
    /// `host_writes`' epoch before the plan's read; a device write since is a
    /// race the serial rail's settle would have closed.
    host_epoch: u64,
    serve: MemoServe,
}

/// Plan the memo bind of one sampled ref. `budget` is the chunk's remaining
/// memo-fill bytes; a fill that would exceed it is a [`SampleBail::MemoBudget`]
/// refusal and the serial memo fills on the drain thread.
pub(crate) fn plan_mapper_ref_memo<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    resource: std::sync::Arc<crate::model::TaskResource>,
    budget: &std::sync::atomic::AtomicU64,
) -> Result<MemoPlan, SampleBail> {
    let mid = sampled_surface_of(&resource, texture_ref)?;
    let m = state.mappings.get(&mid).ok_or(SampleBail::ColdBacking)?;
    if !(m.mapped && !m.page_entries.is_empty() && m.has_geom) {
        return Err(SampleBail::ColdBacking);
    }
    let (w, h) = (m.width, m.height);
    if w == 0 || h == 0 {
        return Err(SampleBail::NoGeometry);
    }
    let sample_fmt = effective_view_sample_format(pixel_format::MTL_FORMAT_BGRA8_UNORM, None)
        .ok_or(SampleBail::ZeroCopyDeclined)?;
    let stride = w.saturating_mul(4);
    // The serial scratch's own sizing: `host_alloc_len(stride × h)`, which the
    // memo's stored `native` is — the compare is over exactly these bytes.
    let scratch_len = host_alloc_len(u64::from(stride).saturating_mul(u64::from(h)))
        .ok_or(SampleBail::ZeroCopyDeclined)?;
    if scratch_len == 0 {
        return Err(SampleBail::ZeroCopyDeclined);
    }
    // The serial fast path's own window probe: the multi-run arm, a BGRA8
    // mapping, no `mapping_internal`, and a contig refusal this page-list
    // generation latched. Anything else takes the serial reader's other arm.
    // The probe's span is the *window* (page-rounded), not the rect: the runs
    // cover it while the compare covers the scratch.
    let (base_off, bpr, window_span, tight) =
        memo_read_window(state, mid, w, h, stride, scratch_len).ok_or(SampleBail::ColdWindow)?;
    if crate::runtime::writeback_debt::owes_mapping(state, mid) {
        return Err(SampleBail::OwedFrame);
    }
    let (gpas, runs) =
        crate::runtime::mapper::mapping_window_cached(state, mid, base_off, window_span)
            .ok_or(SampleBail::ColdWindow)?;
    // The serial reader settles before it reads; a plan may only ask that
    // nothing it would wait for is outstanding over exactly these pages.
    if crate::runtime::gather_witness::writes_pending_over(&gpas) {
        return Err(SampleBail::OwedFrame);
    }
    let host_epoch = state.host_writes.epoch();
    // The read: tight rows out of the cached runs' stable host pointers, in
    // the same order the serial fast path reads them, over a scratch-sized
    // buffer whose tail past the rect is zeroed as `prepare_memo_scratch`
    // zeroes it — the compare is over the whole scratch.
    let mut native = vec![0u8; scratch_len];
    if runs.iter().map(|r| r.len()).sum::<u64>() != window_span {
        return Err(SampleBail::ColdWindow);
    }
    let read_row = |dst: &mut [u8], mut at: u64| -> bool {
        let (mut ri, mut run_lo) = (0usize, 0u64);
        let mut done = 0usize;
        while run_lo + runs[ri].len() <= at {
            run_lo += runs[ri].len();
            ri += 1;
            if ri == runs.len() {
                return false;
            }
        }
        while done < tight {
            while run_lo + runs[ri].len() <= at {
                run_lo += runs[ri].len();
                ri += 1;
                if ri == runs.len() {
                    return false;
                }
            }
            let run = runs[ri];
            let off = at - run_lo;
            let n = ((run.len() - off) as usize).min(tight - done);
            // SAFETY: `run` is a stable host's `map_pages` alias of guest RAM,
            // valid for the life of the VM (`map_pages_stable`), and
            // `off + n <= run.len()`; `dst[done..done + n]` is in bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (run.host_ptr() as *const u8).add(off as usize),
                    dst[done..].as_mut_ptr(),
                    n,
                );
            }
            done += n;
            at += n as u64;
        }
        true
    };
    for y in 0..h as usize {
        let row = &mut native[y * stride as usize..(y + 1) * stride as usize];
        // The serial fast path indexes the same window: row `y` starts at
        // `y * bpr` inside the window the runs cover.
        if !read_row(row, y as u64 * bpr as u64) {
            return Err(SampleBail::ColdWindow);
        }
    }
    // A device write that raced the read invalidates it; the serial settle
    // would have waited it out.
    if state.host_writes.wrote_any_since(host_epoch, &gpas).wrote() {
        return Err(SampleBail::ColdWindow);
    }
    let reads = MemoReads::of(state, mid).ok_or(SampleBail::ColdBacking)?;
    // The memo's own coherence is a byte compare the plan cannot repeat at
    // commit; it rests on the mapping token's lagged generation instead (the
    // H5 standard every plan/commit pair here uses). An unarmed token leaves
    // the plan blind to guest writes the serial compare would catch, so those
    // surfaces stay serial.
    if reads.guest_write_token == 0 {
        return Err(SampleBail::ColdWindow);
    }
    let observed_gen = host.guest_write_gen_lagged(reads.guest_write_token);
    let window_key = crate::runtime::mapper::mapping_window_key_readonly(state, mid)
        .ok_or(SampleBail::ColdWindow)?;
    let key = (mid, w, h);
    // The serial rung's byte order (`lab_memo_bgra`), so a plan and the serial
    // fill it stands in for agree on what the memo holds.
    let bgra = super::lab_memo_bgra();
    let want_layout = if bgra { TexelLayout::Bgra8 } else { TexelLayout::Rgba8 };
    let serve = match state.mapper_ref_texture_memo.peek(&key) {
        Some(entry) if entry.native[..] == native[..] && entry.layout == want_layout => {
            MemoServe::Hit {
                rgba: std::sync::Arc::clone(&entry.rgba),
                generation: entry.generation,
                layout: entry.layout,
            }
        }
        _ => {
            // First sight or the native bytes changed: convert fresh, new
            // generation — the serial fill, minus its drain-thread read.
            let len = (u64::from(stride) * u64::from(h)) as usize;
            let mut rgba = vec![0u8; len];
            let converted = if bgra {
                rgba.copy_from_slice(&native[..len]);
                true
            } else {
                pixel_format::RowToRgba8::for_format(sample_fmt).is_some_and(|row_rail| {
                    (0..h as usize).all(|y| {
                        let off = y * stride as usize;
                        row_rail.convert(
                            &native[off..off + (w as usize) * 4],
                            w,
                            &mut rgba[off..off + (w as usize) * 4],
                        )
                    })
                })
            };
            if !converted {
                return Err(SampleBail::ZeroCopyDeclined);
            }
            let entry_bytes = native.len() + rgba.len();
            if take_memo_budget(budget, entry_bytes as u64).is_err() {
                return Err(SampleBail::MemoBudget);
            }
            MemoServe::Fill {
                native,
                rgba: std::sync::Arc::new(rgba),
                generation: state.next_sampled_content_generation(),
                layout: want_layout,
                entry_bytes,
            }
        }
    };
    Ok(MemoPlan {
        task_id,
        texture_ref,
        resource,
        mid,
        w,
        h,
        stride,
        // The window span: what the cached window and the runs cover.
        span: window_span,
        base_off,
        sample_fmt,
        window_key,
        gpas,
        runs,
        reads,
        observed_gen,
        host_epoch,
        serve,
    })
}

/// Charge `bytes` against the chunk's fill budget, answering what remains.
fn take_memo_budget(budget: &std::sync::atomic::AtomicU64, bytes: u64) -> Result<u64, u64> {
    budget.fetch_update(
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
        // `then`, not `then_some`: the difference is taken only when it exists.
        |left| (left >= bytes).then(|| left - bytes),
    )
}

impl MemoPlan {
    fn identity(&self) -> LinearSampleIdentity {
        // The identity key namespace the serial memo producer uses: bits 63+62
        // mark mapper-ref-texture memo content.
        LinearSampleIdentity {
            key: (1u64 << 63) | (1u64 << 62) | u64::from(self.mid),
            generation: match &self.serve {
                MemoServe::Hit { generation, .. } | MemoServe::Fill { generation, .. } => {
                    *generation
                }
            },
        }
    }

    fn source(&self) -> (u32, u32, u32, SampledSourceRequest) {
        self.source_with(Some(self.identity()))
    }

    /// The fill's shadow prediction: no identity, because the generation a
    /// fill names is minted by whichever arm runs, and either arm's identity
    /// forces the same engine upload.
    fn shadow_source(&self) -> (u32, u32, u32, SampledSourceRequest) {
        self.source_with(None)
    }

    fn source_with(
        &self,
        identity: Option<LinearSampleIdentity>,
    ) -> (u32, u32, u32, SampledSourceRequest) {
        let rgba = match &self.serve {
            MemoServe::Hit { rgba, .. } | MemoServe::Fill { rgba, .. } => {
                std::sync::Arc::clone(rgba)
            }
        };
        let layout = match &self.serve {
            MemoServe::Hit { layout, .. } | MemoServe::Fill { layout, .. } => *layout,
        };
        (
            self.w,
            self.h,
            self.mid,
            SampledSourceRequest::Bytes(
                rgba,
                identity,
                SampledByteFormat::from_source(layout, self.sample_fmt),
                crate::backend::vulkan::engine::SampledByteOrigin::SurfaceGuestFallback,
            ),
        )
    }

    /// The hit-shaped plan over a fill this commit inserted: what the next
    /// bind of the ref serves if nothing moves.
    fn hit_like(
        &self,
        rgba: std::sync::Arc<Vec<u8>>,
        generation: u64,
        layout: TexelLayout,
    ) -> Self {
        Self {
            task_id: self.task_id,
            texture_ref: self.texture_ref,
            resource: std::sync::Arc::clone(&self.resource),
            mid: self.mid,
            w: self.w,
            h: self.h,
            stride: self.stride,
            span: self.span,
            base_off: self.base_off,
            sample_fmt: self.sample_fmt,
            window_key: self.window_key,
            gpas: std::sync::Arc::clone(&self.gpas),
            runs: std::sync::Arc::clone(&self.runs),
            reads: self.reads,
            observed_gen: self.observed_gen,
            host_epoch: self.host_epoch,
            serve: MemoServe::Hit {
                rgba,
                generation,
                layout,
            },
        }
    }
}

/// Re-check a memo plan at the point the serial memo would run, and with
/// `apply` take it. A hit serves the memo's `Arc` (bumping its recency, as the
/// serial `get_touch` does); a fill inserts the entry the plan built — the
/// serial insert, whose bytes were read off the drain thread — and hands back
/// the hit-shaped plan for the next bind of the ref.
///
/// Without `apply` nothing is written; the fill's shadow prediction carries no
/// identity, because the generation a fill names is minted by whichever arm
/// runs, and either arm's identity forces the same engine upload.
pub(crate) fn commit_memo_plan<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    plan: &MemoPlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    apply: bool,
) -> Result<((u32, u32, u32, SampledSourceRequest), Option<MemoPlan>), SampleInvalid> {
    if plan.task_id != task_id || plan.texture_ref != texture_ref {
        return Err(SampleInvalid::Key);
    }
    if !resource.is_some_and(|r| std::sync::Arc::ptr_eq(r, &plan.resource)) {
        return Err(SampleInvalid::Resource);
    }
    if MemoReads::of(state, plan.mid) != Some(plan.reads) {
        return Err(SampleInvalid::Mapping);
    }
    if crate::runtime::mapper::mapping_window_key_readonly(state, plan.mid) != Some(plan.window_key)
    {
        return Err(SampleInvalid::Mapping);
    }
    if crate::runtime::writeback_debt::owes_mapping(state, plan.mid) {
        return Err(SampleInvalid::Debt);
    }
    // The resident rung above the page ladder, as the mapper-ref commit
    // re-checks it: a readiness that appeared since the plan changes the
    // serial answer's shape above this rail.
    if surface_resident_appeared(&*state, &plan.resource, plan.mid, plan.w, plan.h) {
        return Err(SampleInvalid::UpperRung);
    }
    let cached =
        crate::runtime::mapper::mapping_window_cached(state, plan.mid, plan.base_off, plan.span);
    if !cached.is_some_and(|(gpas, runs)| {
        std::sync::Arc::ptr_eq(&gpas, &plan.gpas) && std::sync::Arc::ptr_eq(&runs, &plan.runs)
    }) {
        return Err(SampleInvalid::Window);
    }
    if crate::runtime::gather_witness::writes_pending_over(&plan.gpas) {
        return Err(SampleInvalid::Debt);
    }
    if state
        .host_writes
        .wrote_any_since(plan.host_epoch, &plan.gpas)
        .wrote()
    {
        return Err(SampleInvalid::Witness);
    }
    if let Some(gen) = plan.observed_gen {
        if host.guest_write_gen_lagged(plan.reads.guest_write_token) != Some(gen) {
            return Err(SampleInvalid::GuestWrite);
        }
    }
    let key = (plan.mid, plan.w, plan.h);
    match &plan.serve {
        MemoServe::Hit {
            rgba,
            generation,
            layout,
        } => {
            // The entry the plan proved unchanged must still be the memo's
            // answer for this key — a fill in between replaced it.
            let holds = state
                .mapper_ref_texture_memo
                .peek(&key)
                .is_some_and(|entry| {
                    entry.generation == *generation && std::sync::Arc::ptr_eq(&entry.rgba, rgba)
                });
            if !holds {
                return Err(SampleInvalid::HostCache);
            }
            if !apply {
                return Ok((plan.source(), None));
            }
            let _touched = state.mapper_ref_texture_memo.get_touch(&key);
            crate::runtime::drain::note_store_route("t11rung_guest_memo");
            Ok((
                plan.source(),
                Some(plan.hit_like(rgba.clone(), *generation, *layout)),
            ))
        }
        MemoServe::Fill {
            native,
            rgba,
            generation,
            layout,
            entry_bytes,
        } => {
            if !apply {
                // No identity: the serial fill running next mints its own.
                return Ok((plan.shadow_source(), None));
            }
            state.mapper_ref_texture_memo.insert(
                key,
                crate::model::GuestLinearMemo {
                    native: native.clone(),
                    rgba: std::sync::Arc::clone(rgba),
                    layout: *layout,
                    generation: *generation,
                },
                *entry_bytes,
            );
            crate::runtime::drain::note_store_route("t11rung_guest_memo");
            // The inserted entry is what the next bind hits, under this fill's
            // own identity (minted at the plan).
            Ok((
                plan.source(),
                Some(plan.hit_like(rgba.clone(), *generation, *layout)),
            ))
        }
    }
}

/// TEMP (lab, DRAW_PAR): [`commit_ref_view_plan`]'s validation and predicted
/// serve over a shared state — what a pool worker assembles a request from.
/// The drain's commit validates again and lands the witness; the request is
/// used only if the committed outcome equals the predicted one.
pub(crate) fn predict_ref_view_plan<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    plan: &RefViewPlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
) -> Result<(u32, u32, u32, SampledSourceRequest), SampleInvalid> {
    if plan.task_id != task_id || plan.texture_ref != texture_ref {
        return Err(SampleInvalid::Key);
    }
    if !resource.is_some_and(|r| std::sync::Arc::ptr_eq(r, &plan.resource)) {
        return Err(SampleInvalid::Resource);
    }
    let mid = plan.mid;
    let row_holds = state.mappings.get(&mid).is_some_and(|m| {
        m.mapped
            && !m.page_entries.is_empty()
            && m.has_geom
            && ref_texture_view_requires_materialization(
                m.has_geom, m.width, m.height, m.format, plan.view,
            )
    });
    if !row_holds
        || crate::runtime::mapper::mapping_window_key_readonly(state, mid) != Some(plan.window_key)
        || ref_view_plane(state, host, mid, plan.view) != Some(plan.plane)
    {
        return Err(SampleInvalid::Mapping);
    }
    if !state.pending_writebacks.is_empty() && state.pending_writebacks.owes(mid) {
        return Err(SampleInvalid::Debt);
    }
    let (base_off, span) = (plan.plane.base_off, plan.plane.span);
    let cached = crate::runtime::mapper::mapping_window_cached(state, mid, base_off, span);
    if !cached.is_some_and(|(gpas, runs)| {
        std::sync::Arc::ptr_eq(&gpas, &plan.gpas) && std::sync::Arc::ptr_eq(&runs, &plan.runs)
    }) {
        return Err(SampleInvalid::Window);
    }
    if !crate::runtime::gather_witness::gather_plan_holds(state, host, &plan.gather, &plan.gpas) {
        return Err(SampleInvalid::Witness);
    }
    let view = plan.view;
    let source = |pages, outcome: crate::runtime::gather_witness::GatherOutcome| {
        (
            view.width,
            view.height,
            mid,
            SampledSourceRequest::GuestRuns(
                crate::backend::vulkan::engine::GuestRunSource {
                    runs: std::sync::Arc::clone(&plan.runs),
                    source_offset: 0,
                    total_len: span,
                    row_length_texels: plan.plane.row_length_texels,
                    pages,
                    direct_image: None,
                },
                plan.plane.native,
                plan.plane.sampled_vk_format,
                1,
                Some(LinearSampleIdentity::from(outcome.identity)),
                outcome.vouch,
                outcome.evidence,
                pixel_format::swizzle_identity(),
            ),
        )
    };
    // TEMP (lab, PASS_STAGE): the staged window, on every arm the closure
    // below serves.
    let wrap = |served: (u32, u32, u32, SampledSourceRequest)| match plan.staged.as_ref() {
        Some(st) => (served.0, served.1, served.2, wrap_staged(served.3, st)),
        None => served,
    };
    Ok(wrap(source(None, plan.gather.predicted())))
}


/// TEMP (lab, DRAW_PAR): [`commit_linear_plan`]'s validation and predicted
/// serve over a shared state; see [`predict_ref_view_plan`].
pub(crate) fn predict_linear_plan<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    plan: &LinearPlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
) -> Result<(u32, u32, u32, SampledSourceRequest), SampleInvalid> {
    if plan.task_id != task_id
        || plan.texture_ref != texture_ref
        || plan.may_bind_resident != may_bind_resident
    {
        return Err(SampleInvalid::Key);
    }
    if !resource.is_some_and(|r| std::sync::Arc::ptr_eq(r, &plan.resource)) {
        return Err(SampleInvalid::Resource);
    }
    // The predicates the plan refused through, repeated: a debt armed since the
    // plan is the drain's to pay, on the serial rail.
    if crate::runtime::writeback_debt::owes_texture(state, task_id, texture_ref)
        || state.pending_writebacks.owes_gva_overlapping(
            task_id,
            plan.allocation.gva,
            plan.allocation.size,
        )
    {
        return Err(SampleInvalid::Debt);
    }
    // The rung above, repeated: it became servable since the plan (a Store
    // armed the witness, a resident turned ready) — the serial answer changed
    // shape above this rail.
    let tex = match plan.resource.decoded() {
        Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => tex,
        _ => return Err(SampleInvalid::Resource),
    };
    if may_bind_resident
        && linear_resident_rung_above_may_serve(
            state,
            host,
            task_id,
            texture_ref,
            plan.gva,
            u32::try_from(plan.row_stride).map_err(|_| SampleInvalid::Resource)?,
            plan.w,
            plan.h,
            tex.pixel_format,
        )
    {
        return Err(SampleInvalid::UpperRung);
    }
    let packed = state
        .bound_buffers
        .packed_available(
            task_id,
            texture_ref,
            plan.allocation.gva,
            plan.allocation.size,
        )
        .ok_or(SampleInvalid::Packed)?;
    if plan.packed_head != packed.head
        || !std::sync::Arc::ptr_eq(&plan.packed_runs, &packed.runs)
        || !std::sync::Arc::ptr_eq(&plan.packed_pages, &packed.pages)
    {
        return Err(SampleInvalid::Packed);
    }
    // Cloned out before the witness commit takes `state` mutably: the serve
    // below names the same allocation the identity check just proved.
    let (packed_runs, packed_pages) = (
        std::sync::Arc::clone(&packed.runs),
        std::sync::Arc::clone(&packed.pages),
    );
    let guest_runs = crate::backend::vulkan::engine::GuestRunSource {
        runs: packed_runs,
        source_offset: plan.layout_offset,
        total_len: plan.span,
        row_length_texels: plan.row_length_texels,
        pages: Some(packed_pages),
        direct_image: None,
    };
    if plan.gather.is_none() {
        // The direct arm: rebuilt from the same borrowed allocation, moving
        // nothing — the plan serves again for the next bind of the ref.
        let request = direct_linear_sample_from_packed(
            packed,
            plan.layout_offset,
            plan.row_stride,
            plan.span,
            plan.row_length_texels,
            plan.native,
            plan.sampled_vk_format,
            plan.native_components,
            plan.resource.lifetime_ref(),
        )
        .ok_or(SampleInvalid::Packed)?;

        let SampledSourceRequest::GuestRuns(source, ..) = request else {
            return Err(SampleInvalid::Packed);
        };
        // TEMP (lab, PASS_STAGE): the staged window, when one was attached.
        let served = plan.source(source, None);
        // TEMP (lab, STAGED_PROVEN): the direct arm moved nothing, and this
        // commit served the bind — the plan it hands back is proven (its
        // window, once a worker stages it, serves every later bind).
        return Ok(wrap_linear(plan, served));
    }
    let gather = plan.gather.as_ref().expect("checked above");
    let run = plan.witness_run.expect("the gather arm carries its run");
    Ok(wrap_linear(plan, plan.source(guest_runs, Some(gather.predicted()))))
}


/// TEMP (lab, DRAW_PAR): [`commit_mapper_ref_plan`]'s validation and predicted
/// serve over a shared state; see [`predict_ref_view_plan`].
pub(crate) fn predict_mapper_ref_plan<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    plan: &MapperRefPlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
) -> Result<(u32, u32, u32, SampledSourceRequest), SampleInvalid> {
    if plan.task_id != task_id || plan.texture_ref != texture_ref {
        return Err(SampleInvalid::Key);
    }
    if !resource.is_some_and(|r| std::sync::Arc::ptr_eq(r, &plan.resource)) {
        return Err(SampleInvalid::Resource);
    }
    let mid = plan.mid;
    let row_holds = state.mappings.get(&mid).is_some_and(|m| {
        m.mapped
            && !m.page_entries.is_empty()
            && m.has_geom
            && m.width == plan.w
            && m.height == plan.h
    });
    let window_holds = state.mappings.get(&mid).and_then(|m| {
        let format = if m.format != 0 {
            m.format
        } else {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        };
        crate::runtime::mapping_write::mapper_ref_texture_sample_window(m, plan.w, plan.h, format)
            .map(|(base_off, bpr, _)| (base_off, u64::from(bpr)))
    });
    if !row_holds
        || window_holds != Some((plan.base_off, plan.bpr))
        || crate::runtime::mapper::mapping_window_key_readonly(state, mid) != Some(plan.window_key)
    {
        return Err(SampleInvalid::Mapping);
    }
    if crate::runtime::writeback_debt::owes_mapping(state, mid) {
        return Err(SampleInvalid::Debt);
    }
    if mapper_direct_rung_above_may_serve(state, host, mid) {
        return Err(SampleInvalid::UpperRung);
    }
    // The resident rung above the whole page ladder: the plan fell past it
    // when the surface ladder planned, and a readiness that appeared since
    // changes the serial answer's shape above this rail.
    if surface_resident_appeared(state, &plan.resource, mid, plan.w, plan.h) {
        return Err(SampleInvalid::UpperRung);
    }
    let cached =
        crate::runtime::mapper::mapping_window_cached(state, mid, plan.base_off, plan.span);
    if !cached.is_some_and(|(gpas, runs)| {
        std::sync::Arc::ptr_eq(&gpas, &plan.gpas) && std::sync::Arc::ptr_eq(&runs, &plan.runs)
    }) {
        return Err(SampleInvalid::Window);
    }
    Ok(wrap_mapper(plan, plan.source(None, plan.gather.predicted())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;
    use reims_vgpu_protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    /// A ref-texture (`texture_ref` 2) naming surface 71, 4x3 BGRA8, whose
    /// host cache holds one frame; the fixture of the serial ladder test
    /// `ref_texture_sample_uses_descriptor_surface_id_not_ref_collision`.
    fn fixture() -> (DeviceState, FakeHost, u32, u32) {
        use crate::protocol::endian::st32;
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
        use crate::runtime::gva_mem;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let (dir_pfn, root_pfn) = (2u32, 3u32);
        let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
        let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        let mut dir = [0u8; 8];
        st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
        st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
        assert!(host.write_gpa(dir_gpa, &dir).is_ok());
        for i in 0..3u32 {
            let pfn = 4 + i;
            host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
        }
        state.define_task(1, 0x1000, dir_pfn);
        assert!(state.set_object_list(1, 0, 32));
        let (texture_ref, surface_id, desc_gva) = (2u32, 71u32, 0x1000u64);
        let built = reims_vgpu_wire::device_desc::RefTextureBuilder::new(surface_id, 0, 0, 0)
            .with_len(objects::TYPE5_MIN_LEN);
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            desc_gva,
            built.bytes(),
            PAGE_SHIFT_X86
        )
        .is_ok());
        let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed =
            (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((objects::TYPE5_MIN_LEN as u32) << 8);
        st32(&mut list_entry, packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            list_off,
            &list_entry,
            PAGE_SHIFT_X86
        )
        .is_ok());
        assert!(state.map_surface(surface_id));
        assert!(state.set_mapping_geom(surface_id, 4, 3, MTL_FORMAT_BGRA8_UNORM));
        state.mappings.get_mut(&surface_id).unwrap().page_entries = vec![1];
        crate::runtime::surface_cache::store(
            &mut state,
            surface_id,
            4,
            3,
            [255u8, 0, 0, 255].repeat(4 * 3),
        );
        (state, host, texture_ref, surface_id)
    }

    fn resource(
        state: &DeviceState,
        host: &FakeHost,
        texture_ref: u32,
    ) -> std::sync::Arc<crate::model::TaskResource> {
        objects::resolve_resource(state, host, 1, texture_ref).expect("fixture resource")
    }

    fn bytes_of(src: &SampledSourceRequest) -> (Vec<u8>, TexelLayout) {
        match src {
            SampledSourceRequest::Bytes(bytes, _, layout, _) => (bytes.to_vec(), layout.layout()),
            _ => panic!("expected a byte serve"),
        }
    }

    #[test]
    fn a_planned_host_cache_bind_commits_to_the_serial_answer() {
        let (mut state, mut host, texture_ref, surface_id) = fixture();
        let res = resource(&state, &host, texture_ref);
        let plan = plan_sampled_source(&state, &host, 1, texture_ref, res.clone(), true)
            .expect("a warm host-cache bind plans");
        // No resident and an unarmed witness: the host-cache rung under a
        // NoStamp verdict, exactly the routes the serial ladder notes.
        assert_eq!(
            plan.routes,
            vec![
                "resident_resource_unavailable",
                "t11rung_host_cache",
                "t11rung_host_cache_gw_no_stamp"
            ]
        );
        let (w, h, mid, planned) =
            commit_sampled_plan(&state, &host, &plan, 1, texture_ref, Some(&res), true)
                .expect("nothing moved between plan and commit");
        let (sw, sh, smid, serial) =
            resolve_sampled_source(&mut state, &mut host, 1, texture_ref, Some(res), true)
                .expect("serial ladder serves the same bind");
        assert_eq!((w, h, mid), (sw, sh, smid));
        assert_eq!(mid, surface_id);
        assert_eq!(bytes_of(&planned), bytes_of(&serial));
    }

    #[test]
    fn a_replaced_host_cache_entry_drops_the_plan() {
        let (mut state, host, texture_ref, surface_id) = fixture();
        let res = resource(&state, &host, texture_ref);
        let plan = plan_sampled_source(&state, &host, 1, texture_ref, res.clone(), true).unwrap();
        crate::runtime::surface_cache::store(
            &mut state,
            surface_id,
            4,
            3,
            [0u8, 255, 0, 255].repeat(4 * 3),
        );
        assert_eq!(
            commit_sampled_plan(&state, &host, &plan, 1, texture_ref, Some(&res), true).err(),
            Some(SampleInvalid::HostCache)
        );
    }

    #[test]
    fn a_moved_mapping_drops_the_plan() {
        let (mut state, host, texture_ref, surface_id) = fixture();
        let res = resource(&state, &host, texture_ref);
        let plan = plan_sampled_source(&state, &host, 1, texture_ref, res.clone(), true).unwrap();
        state.mappings.get_mut(&surface_id).unwrap().map_generation += 1;
        assert_eq!(
            commit_sampled_plan(&state, &host, &plan, 1, texture_ref, Some(&res), true).err(),
            Some(SampleInvalid::Mapping)
        );
    }

    #[test]
    fn a_plan_is_taken_only_for_its_own_bind_and_resource() {
        let (state, host, texture_ref, _) = fixture();
        let res = resource(&state, &host, texture_ref);
        let plan = plan_sampled_source(&state, &host, 1, texture_ref, res.clone(), true).unwrap();
        assert_eq!(
            commit_sampled_plan(&state, &host, &plan, 1, texture_ref, None, true).err(),
            Some(SampleInvalid::Resource)
        );
        let plan = plan_sampled_source(&state, &host, 1, texture_ref, res.clone(), true).unwrap();
        assert_eq!(
            commit_sampled_plan(&state, &host, &plan, 1, texture_ref, Some(&res), false).err(),
            Some(SampleInvalid::Key)
        );
    }

    #[test]
    fn cold_steps_are_not_planned() {
        let (mut state, host, texture_ref, surface_id) = fixture();
        let res = resource(&state, &host, texture_ref);
        assert_eq!(
            plan_sampled_source(&state, &host, 1, 0, res.clone(), true).err(),
            Some(SampleBail::NullRef)
        );
        // An unresolved backing is the cold arm of ensure_surface_for_texture_bind.
        state
            .mappings
            .get_mut(&surface_id)
            .unwrap()
            .page_entries
            .clear();
        assert_eq!(
            plan_sampled_source(&state, &host, 1, texture_ref, res.clone(), true).err(),
            Some(SampleBail::ColdBacking)
        );
        state.mappings.get_mut(&surface_id).unwrap().page_entries = vec![1];
        // No cache entry and no resident: the page rungs serve it.
        state.host_surfaces.remove(&surface_id);
        assert_eq!(
            plan_sampled_source(&state, &host, 1, texture_ref, res, true).err(),
            Some(SampleBail::NoHostCache)
        );
    }

    #[test]
    fn a_warm_resident_plans_the_serial_target() {
        let (mut state, mut host, texture_ref, surface_id) = fixture();
        let res = resource(&state, &host, texture_ref);
        let identity =
            crate::backend::vulkan::present_identity::surface_identity(&state, surface_id, 4, 3);
        resource_lease::seed_warm_lease_for_test(
            &res,
            &identity,
            ResidentContentBacking::DeviceAllocation,
        );
        let plan = plan_sampled_source(&state, &host, 1, texture_ref, res.clone(), true)
            .expect("a warm resident plans");
        assert_eq!(
            plan.routes,
            vec![
                "t11sample_ready_device_allocation",
                "t11rung_resident",
                "t11rung_resident_gw_no_stamp"
            ]
        );
        let (_, _, _, planned) =
            commit_sampled_plan(&state, &host, &plan, 1, texture_ref, Some(&res), true).unwrap();
        let (_, _, _, serial) =
            resolve_sampled_source(&mut state, &mut host, 1, texture_ref, Some(res), true).unwrap();
        match (planned, serial) {
            (SampledSourceRequest::Target(a, fa), SampledSourceRequest::Target(b, fb)) => {
                assert_eq!(a, b);
                assert_eq!(fa, fb);
            }
            _ => panic!("both must bind the resident"),
        }
    }

    #[test]
    fn a_surface_with_no_resident_plans_the_host_cache_without_a_lease() {
        let (state, host, texture_ref, surface_id) = fixture();
        let res = resource(&state, &host, texture_ref);
        let identity =
            crate::backend::vulkan::present_identity::surface_identity(&state, surface_id, 4, 3);
        let plan = plan_sampled_source(&state, &host, 1, texture_ref, res.clone(), true).unwrap();
        assert!(plan.absent_resident && !plan.lease_warm);
        // The plan retained nothing: the resource still holds no lease.
        assert_eq!(
            resource_lease::resident_target_backing_warm(&res, &identity),
            None
        );
    }

    #[test]
    fn a_lease_acquired_after_the_plan_drops_it() {
        let (state, host, texture_ref, surface_id) = fixture();
        let res = resource(&state, &host, texture_ref);
        let plan = plan_sampled_source(&state, &host, 1, texture_ref, res.clone(), true).unwrap();
        let identity =
            crate::backend::vulkan::present_identity::surface_identity(&state, surface_id, 4, 3);
        resource_lease::seed_warm_lease_for_test(
            &res,
            &identity,
            ResidentContentBacking::DeviceAllocation,
        );
        assert_eq!(
            commit_sampled_plan(&state, &host, &plan, 1, texture_ref, Some(&res), true).err(),
            Some(SampleInvalid::Backing)
        );
    }

    /// A ref-texture (`texture_ref` 2) whose descriptor views surface 72 as
    /// sRGB BGRA8 while the surface is declared unorm BGRA8, 256x128: a view
    /// that needs materializing, over 32 guest pages scattered two apart on a
    /// host that aliases packed runs only, so no contiguous view or import
    /// exists and the zero-copy rail gathers the guest runs.
    fn view_fixture() -> (DeviceState, FakeHost, u32, u32, Vec<u64>) {
        use crate::protocol::endian::st32;
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
        use crate::runtime::gva_mem;
        use reims_vgpu_protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.stable_map_pages = true;
        host.strict_linux_map = true;
        let (dir_pfn, root_pfn) = (2u32, 3u32);
        let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
        let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        let mut dir = [0u8; 8];
        st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
        st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
        assert!(host.write_gpa(dir_gpa, &dir).is_ok());
        for i in 0..3u32 {
            let pfn = 4 + i;
            host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
        }
        state.define_task(1, 0x1000, dir_pfn);
        assert!(state.set_object_list(1, 0, 32));
        let (texture_ref, surface_id, desc_gva) = (2u32, 72u32, 0x1000u64);
        let built = reims_vgpu_wire::device_desc::RefTextureBuilder::new(
            surface_id,
            0,
            0,
            reims_vgpu_wire::device_desc::TYPE5_RECORD_TAG_COLOR_VIEW,
        )
        .geometry(MTL_FORMAT_BGRA8_UNORM_SRGB, 256, 128, 1);
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            desc_gva,
            built.bytes(),
            PAGE_SHIFT_X86
        )
        .is_ok());
        let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        let packed =
            (objects::OBJECT_TYPE_REF_TEXTURE as u32) | ((built.bytes().len() as u32) << 8);
        st32(&mut list_entry, packed);
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            list_off,
            &list_entry,
            PAGE_SHIFT_X86
        )
        .is_ok());
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpas: Vec<u64> = (0..32u64).map(|i| 0x5000_0000 + 2 * i * page).collect();
        for &gpa in &gpas {
            host.map_range(gpa, page as usize, 0x33);
        }
        assert!(state.map_surface(surface_id));
        assert!(state.set_mapping_geom(surface_id, 256, 128, MTL_FORMAT_BGRA8_UNORM));
        state.mappings.get_mut(&surface_id).unwrap().page_entries = gpas
            .iter()
            .map(|&gpa| {
                (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
        (state, host, texture_ref, surface_id, gpas)
    }

    /// The fields of a guest-run serve two fixtures can agree on (host pointers
    /// differ between fixtures, so the runs are compared by shape).
    fn runs_shape(
        served: &(u32, u32, u32, SampledSourceRequest),
    ) -> (
        u32,
        u32,
        u32,
        u64,
        u32,
        usize,
        TexelLayout,
        ash::vk::Format,
        Option<LinearSampleIdentity>,
        crate::runtime::gather_witness::GatherVouch,
    ) {
        let (w, h, mid, src) = served;
        let SampledSourceRequest::GuestRuns(source, layout, format, _, identity, vouch, _, _) =
            src
        else {
            panic!("expected a guest-run serve")
        };
        (
            *w,
            *h,
            *mid,
            source.total_len,
            source.row_length_texels,
            source.runs.len(),
            *layout,
            *format,
            *identity,
            *vouch,
        )
    }

    fn serial(
        state: &mut DeviceState,
        host: &mut FakeHost,
        texture_ref: u32,
        res: &std::sync::Arc<crate::model::TaskResource>,
    ) -> (u32, u32, u32, SampledSourceRequest) {
        resolve_sampled_source(state, host, 1, texture_ref, Some(res.clone()), true)
            .expect("the view resolves")
    }

    /// A planned ref-view bind serves what the serial rail serves, and leaves the
    /// witness where the serial rail leaves it.
    #[test]
    fn a_planned_ref_view_serves_the_serial_answer() {
        let (mut a, mut a_host, texture_ref, _, _) = view_fixture();
        let (mut b, mut b_host, _, _, _) = view_fixture();
        let (ra, rb) = (
            resource(&a, &a_host, texture_ref),
            resource(&b, &b_host, texture_ref),
        );
        // The first bind arms the witness and fills the window cache: serial only.
        let first_a = serial(&mut a, &mut a_host, texture_ref, &ra);
        let first_b = serial(&mut b, &mut b_host, texture_ref, &rb);
        assert_eq!(runs_shape(&first_a), runs_shape(&first_b));
        for _ in 0..3 {
            let plan = plan_ref_view_zero_copy(&b, &b_host, 1, texture_ref, rb.clone())
                .unwrap_or_else(|bail| panic!("a warm view plans, got {bail:?}"));
            let (planned, repeat) =
                commit_ref_view_plan(&mut b, &mut b_host, &plan, 1, texture_ref, Some(&rb), true)
                    .expect("nothing moved since the plan");
            let serial_a = serial(&mut a, &mut a_host, texture_ref, &ra);
            assert_eq!(runs_shape(&planned), runs_shape(&serial_a));
            // The repeat is the plan the next bind of the ref commits; the
            // loop re-plans from the witness anyway, which the two fixtures'
            // continued agreement below covers.
            drop(repeat);
        }
        // And the next serial bind of each agrees: the witnesses match.
        let next_a = serial(&mut a, &mut a_host, texture_ref, &ra);
        let next_b = serial(&mut b, &mut b_host, texture_ref, &rb);
        assert_eq!(runs_shape(&next_a), runs_shape(&next_b));
    }

    /// A guest store seen after the plan drops it, and a shadow commit leaves
    /// nothing behind that the serial rail would not have.
    #[test]
    fn a_ref_view_plan_yields_to_a_guest_store_and_shadows_cleanly() {
        let (mut b, mut b_host, texture_ref, _, gpas) = view_fixture();
        let rb = resource(&b, &b_host, texture_ref);
        serial(&mut b, &mut b_host, texture_ref, &rb);
        let plan = plan_ref_view_zero_copy(&b, &b_host, 1, texture_ref, rb.clone()).expect("warm");
        // Shadow: validated, predicted, nothing applied — so the plan still holds.
        let (predicted, _) =
            commit_ref_view_plan(&mut b, &mut b_host, &plan, 1, texture_ref, Some(&rb), false)
                .expect("holds");
        assert!(matches!(predicted.3, SampledSourceRequest::GuestRuns(..)));
        b_host.guest_wrote_page(gpas[5]);
        assert_eq!(
            commit_ref_view_plan(&mut b, &mut b_host, &plan, 1, texture_ref, Some(&rb), true).err(),
            Some(SampleInvalid::Witness)
        );
    }

    /// A rewired mapping (a new page-list generation) has no cached window under
    /// its new key, so the plan bails to the serial rail that re-resolves it.
    #[test]
    fn a_rewired_mapping_bails_the_ref_view_plan() {
        let (mut b, mut b_host, texture_ref, sid, _) = view_fixture();
        let rb = resource(&b, &b_host, texture_ref);
        serial(&mut b, &mut b_host, texture_ref, &rb);
        assert!(plan_ref_view_zero_copy(&b, &b_host, 1, texture_ref, rb.clone()).is_ok());
        b.mappings.get_mut(&sid).unwrap().map_generation += 1;
        assert!(matches!(
            plan_ref_view_zero_copy(&b, &b_host, 1, texture_ref, rb.clone()),
            Err(SampleBail::ColdWindow | SampleBail::ZeroCopyDeclined)
        ));
    }

    /// TEMP (lab bench): a small mapper-ref surface (60x64 BGRA8 over four
    /// scattered pages, a packed-only stable host) with no host cache and no
    /// resident, so every bind takes the guest-memo rung. Prints the per-bind
    /// cost of the whole resolve and of its parts. `--ignored` to run.
    pub(crate) fn memo_fixture() -> (DeviceState, FakeHost, u32, u32) {
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        let (mut state, mut host, texture_ref, surface_id) = fixture();
        host.stable_map_pages = true;
        host.strict_linux_map = true;
        state.host_surfaces.remove(&surface_id);
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpas: Vec<u64> = (0..4u64).map(|i| 0x5000_0000 + 2 * i * page).collect();
        for (n, &gpa) in gpas.iter().enumerate() {
            host.map_range(gpa, page as usize, 0x40 + n as u8);
        }
        assert!(state.set_mapping_geom(surface_id, 60, 64, MTL_FORMAT_BGRA8_UNORM));
        state.mappings.get_mut(&surface_id).unwrap().page_entries = gpas
            .iter()
            .map(|&gpa| {
                (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
        (state, host, texture_ref, surface_id)
    }

    /// `REIMS_VGPU_LAB_MEMO_READ`: the memo's coherence read through the window
    /// cache answers exactly what `read_mapping_bgra8` answers — on first sight,
    /// on a repeat, and after the guest writes a page — and a repeat imports
    /// nothing, where the original re-imports every run on every bind.
    #[test]
    fn the_memo_read_through_the_window_cache_is_the_original_read() {
        let (mut a, mut a_host, _, sid) = memo_fixture();
        let (mut b, mut b_host, _, _) = memo_fixture();
        let (w, h, stride) = (60u32, 64u32, 240u32);
        let read = |fast: bool, state: &mut DeviceState, host: &mut FakeHost| {
            let mut dst = vec![0xA5u8; (stride * h) as usize];
            let ok = memo_read_bgra8(fast, state, host, sid, &mut dst, stride, w, h);
            (ok, dst)
        };
        // The contiguous view is refused and cached on the first read of either arm.
        let first_fast = read(true, &mut a, &mut a_host);
        let first_orig = read(false, &mut b, &mut b_host);
        assert!(first_fast.0 && first_orig.0);
        assert_eq!(first_fast, first_orig);
        // The second fast read fills the window cache; the third is warm.
        assert_eq!(read(true, &mut a, &mut a_host), first_orig);
        let calls = a_host.map_pages_calls;
        let again_fast = read(true, &mut a, &mut a_host);
        assert_eq!(again_fast, first_orig);
        assert_eq!(
            a_host.map_pages_calls, calls,
            "a warm window imports nothing"
        );
        let calls = b_host.map_pages_calls;
        assert_eq!(read(false, &mut b, &mut b_host), first_orig);
        assert!(b_host.map_pages_calls > calls, "the original re-imports");
        // A guest write to the third page is read by both.
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa = 0x5000_0000 + 4 * page + 0x10;
        assert!(a_host.write_gpa(gpa, &[1, 2, 3, 4, 5, 6, 7]).is_ok());
        assert!(b_host.write_gpa(gpa, &[1, 2, 3, 4, 5, 6, 7]).is_ok());
        let wrote_fast = read(true, &mut a, &mut a_host);
        let wrote_orig = read(false, &mut b, &mut b_host);
        assert_ne!(wrote_fast.1, first_orig.1, "the write must be seen");
        assert_eq!(wrote_fast, wrote_orig);
        // A mapping whose page list moved goes back through the original.
        a.mappings.get_mut(&sid).unwrap().map_generation += 1;
        b.mappings.get_mut(&sid).unwrap().map_generation += 1;
        assert_eq!(
            read(true, &mut a, &mut a_host),
            read(false, &mut b, &mut b_host)
        );
    }

    /// `REIMS_VGPU_LAB_FIELD_RUNS`: a full-screen field witness whose window the
    /// zero-copy rail has resolved reads its patches from the cached runs and
    /// prints exactly the record the `read_gpa` walk prints for the same pages.
    #[test]
    fn the_field_witness_reads_a_cached_window_from_its_runs() {
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        let (w, h) = (1024u32, 1024u32);
        let page = 1u64 << PAGE_SHIFT_X86;
        let build = |mid: u32| {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
            let mut host = FakeHost::new();
            host.stable_map_pages = true;
            host.strict_linux_map = true;
            let pages = u64::from(w) * u64::from(h) * 4 / page;
            let gpas: Vec<u64> = (0..pages).map(|i| 0x1_0000_0000 + 2 * i * page).collect();
            for (n, &gpa) in gpas.iter().enumerate() {
                host.map_range(gpa, page as usize, (n * 37 % 251) as u8);
            }
            assert!(state.map_surface(mid));
            assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
            state.mappings.get_mut(&mid).unwrap().page_entries = gpas
                .iter()
                .map(|&gpa| {
                    (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
                })
                .collect();
            (state, host, gpas)
        };
        let record = |mid: u32| {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .lines()
                .filter(|l| l.contains(&format!("sampled_surface_field mid={mid} ")))
                .map(|l| {
                    let at = l.find("patches=").expect("patches");
                    let end = l.find(" (guest pages").unwrap_or(l.len());
                    l[at..end].to_string()
                })
                .collect::<Vec<_>>()
        };
        let (mid_runs, mid_walk) = (0xF1E1u32, 0xF1E2u32);
        let (mut a, mut a_host, a_gpas) = build(mid_runs);
        let (b, b_host, _) = build(mid_walk);
        let (base_off, bpr, _) = crate::runtime::mapping_write::mapper_ref_texture_sample_window(
            a.mappings.get(&mid_runs).unwrap(),
            w,
            h,
            MTL_FORMAT_BGRA8_UNORM,
        )
        .unwrap();
        let span = u64::from(bpr) * u64::from(h - 1) + u64::from(w) * 4;
        assert!(mapping_window_guest_runs(&mut a, &mut a_host, mid_runs, base_off, span).is_some());
        crate::runtime::scanout::note_sampled_surface_field(&a, &a_host, mid_runs, 9, "guest_runs");
        crate::runtime::scanout::note_sampled_surface_field(&b, &b_host, mid_walk, 9, "guest_runs");
        let (ra, rb) = (record(mid_runs), record(mid_walk));
        assert_eq!(ra.len(), 1);
        assert_eq!(ra, rb, "the runs and the walk must print the same record");
        assert!(
            ra[0].contains("painted:"),
            "fixture must exercise the stats: {}",
            ra[0]
        );
        // The runs path is the one taken: with every page refusing `read_gpa`
        // the walk reads nothing and prints nothing, while the runs still read
        // the guest's new bytes and report the change.
        for &gpa in &a_gpas {
            assert!(a_host.write_gpa(gpa, &[0xffu8; 4096]).is_ok());
            a_host.mark_non_ram(gpa, page);
        }
        crate::runtime::scanout::note_sampled_surface_field(&a, &a_host, mid_runs, 9, "guest_runs");
        let ra = record(mid_runs);
        assert_eq!(ra.len(), 2);
        assert!(
            ra[1].starts_with("patches=[white:255/0,white:255/0,white:255/0,white:255/0]"),
            "{}",
            ra[1]
        );
    }

    #[test]
    #[ignore]
    fn lab_bench_resident_bind() {
        let (mut state, mut host, texture_ref, surface_id) = fixture();
        let res = resource(&state, &host, texture_ref);
        let identity =
            crate::backend::vulkan::present_identity::surface_identity(&state, surface_id, 4, 3);
        resource_lease::seed_warm_lease_for_test(
            &res,
            &identity,
            ResidentContentBacking::DeviceAllocation,
        );
        let n = 20_000u32;
        let t = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(resolve_sampled_source(
                &mut state,
                &mut host,
                1,
                texture_ref,
                Some(res.clone()),
                true,
            ));
        }
        eprintln!(
            "lab_bench resident resolve: {:.3} us",
            t.elapsed().as_secs_f64() * 1e6 / f64::from(n)
        );
        let t = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(
                surface_top(&state, &host, surface_id, true, |identity| {
                    Some(resource_lease::resident_target_backing(&res, identity))
                })
                .ok(),
            );
        }
        eprintln!(
            "lab_bench resident surface_top: {:.3} us",
            t.elapsed().as_secs_f64() * 1e6 / f64::from(n)
        );
    }

    #[test]
    #[ignore]
    fn lab_bench_memo_bind() {
        let (mut state, mut host, texture_ref, sid) = memo_fixture();
        let res = resource(&state, &host, texture_ref);
        let first = resolve_sampled_source(
            &mut state,
            &mut host,
            1,
            texture_ref,
            Some(res.clone()),
            true,
        )
        .expect("memo rung serves");
        assert!(matches!(first.3, SampledSourceRequest::Bytes(..)));
        let n = 20_000u32;
        let time = |label: &str, f: &mut dyn FnMut()| {
            let t = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            eprintln!(
                "lab_bench {label}: {:.3} us",
                t.elapsed().as_secs_f64() * 1e6 / f64::from(n)
            );
        };
        time("resolve_sampled_source", &mut || {
            std::hint::black_box(resolve_sampled_source(
                &mut state,
                &mut host,
                1,
                texture_ref,
                Some(res.clone()),
                true,
            ));
        });
        time("surface_top", &mut || {
            std::hint::black_box(
                surface_top(&state, &host, sid, true, |identity| {
                    Some(resource_lease::resident_target_backing(&res, identity))
                })
                .ok(),
            );
        });
        time("ensure_surface", &mut || {
            std::hint::black_box(objects::ensure_surface_for_texture_bind(
                &mut state, &host, sid,
            ));
        });
        time("zero_copy", &mut || {
            std::hint::black_box(try_mapper_ref_texture_sample_zero_copy(
                &mut state,
                &mut host,
                sid,
                60,
                64,
                res.lifetime_ref(),
            ));
        });
        time("memoized", &mut || {
            std::hint::black_box(load_mapper_ref_texture_rgba_memoized(
                &mut state, &mut host, sid,
            ));
        });
        let mut dst = vec![0u8; 60 * 64 * 4];
        time("memo_read fast", &mut || {
            std::hint::black_box(memo_read_bgra8(
                true, &mut state, &mut host, sid, &mut dst, 240, 60, 64,
            ));
        });
        time("memo_read orig", &mut || {
            std::hint::black_box(memo_read_bgra8(
                false, &mut state, &mut host, sid, &mut dst, 240, 60, 64,
            ));
        });
        let mut dst = vec![0u8; 60 * 64 * 4];
        time("read_mapping_bgra8", &mut || {
            std::hint::black_box(crate::runtime::scanout::read_mapping_bgra8(
                &mut state, &mut host, sid, &mut dst, 240, 60, 64,
            ));
        });
        let mut buf = vec![0u8; 60 * 64 * 4];
        time("read_mapping_bytes", &mut || {
            std::hint::black_box(crate::runtime::mapper::read_mapping_bytes(
                &mut state, &mut host, sid, 0, &mut buf,
            ));
        });
        time("ensure_contig_view", &mut || {
            std::hint::black_box(crate::runtime::mapper::ensure_contig_view(
                &mut state, &mut host, sid,
            ));
        });
        time("flush_retired_views", &mut || {
            crate::runtime::mapper::flush_retired_views(&mut state, &mut host);
        });
        time("note_ref_backing", &mut || {
            note_sampled_ref_backing(&state, 1, texture_ref, 60, 64, sid, "bytes");
        });
    }

    /// A linear texture (`texture_ref` 2, OBJECT_TYPE_TEXTURE) over a 64x64
    /// BGRA8 level at task GVA `handle << 12`, four pages, whose packed
    /// allocation is already retained in `bound_buffers` — the warm state the
    /// serial rail's `ensure_packed_resource` answers without resolving.
    fn linear_fixture() -> (DeviceState, FakeHost, u32) {
        use crate::protocol::endian::{st16, st32, st64};
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::bound_buffers::{BoundBuffers, PackedBuffer, PackedBufferResolution};
        use crate::runtime::decode::resource::{
            list_object_entry_offset, LINEAR_DESC_HANDLE, LINEAR_DESC_SIZE, OBJECT_LIST_ENTRY_LEN,
            OBJECT_TYPE_TEXTURE, TEXTURE_DESC_BASE_LEN, TEXTURE_DESC_HEIGHT,
            TEXTURE_DESC_PIXEL_FORMAT, TEXTURE_DESC_ROW_STRIDE, TEXTURE_DESC_WIDTH,
        };
        use crate::runtime::gva_mem;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.stable_map_pages = true;
        host.strict_linux_map = true;
        let (dir_pfn, root_pfn) = (2u32, 3u32);
        let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_X86;
        let root_gpa = (root_pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        let mut dir = [0u8; 8];
        st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
        st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
        assert!(host.write_gpa(dir_gpa, &dir).is_ok());
        for i in 0..3u32 {
            let pfn = 4 + i;
            host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
        }
        state.define_task(1, 0x1000, dir_pfn);
        assert!(state.set_object_list(1, 0, 32));

        // The descriptor: handle 32 (allocation base 0x20_000), 64x64 BGRA8 at
        // 256 B/row, allocation exactly the level's four pages.
        let (texture_ref, handle, w, h, bpr) = (2u32, 32u32, 64u32, 64u32, 256u32);
        let span = u64::from(bpr) * u64::from(h);
        let mut desc = vec![0u8; TEXTURE_DESC_BASE_LEN];
        st64(&mut desc[LINEAR_DESC_SIZE..], span);
        st32(&mut desc[LINEAR_DESC_HANDLE..], handle);
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], bpr);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], w);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], h);
        st16(
            &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_BGRA8_UNORM,
        );
        let desc_gva = 0x1000u64;
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            desc_gva,
            &desc,
            PAGE_SHIFT_X86
        )
        .is_ok());
        let list_off = list_object_entry_offset(texture_ref, 32).unwrap();
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut list_entry,
            (OBJECT_TYPE_TEXTURE as u32) | ((TEXTURE_DESC_BASE_LEN as u32) << 8),
        );
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[1],
            list_off,
            &list_entry,
            PAGE_SHIFT_X86
        )
        .is_ok());

        // The allocation's pages, mapped in the task table (slots 32..36) and
        // the host, so a copied-fallback walk would resolve them too.
        let gpas: Vec<u64> = (0..4u64)
            .map(|i| ((handle + i as u32) as u64) << PAGE_SHIFT_X86)
            .collect();
        for (i, &gpa) in gpas.iter().enumerate() {
            host.map_range(gpa, 0x1000, 0x50 + i as u8);
            let mut pte = [0u8; 4];
            st32(&mut pte, (gpa >> PAGE_SHIFT_X86) as u32);
            assert!(host
                .write_gpa(root_gpa + ((handle + i as u32) as u64) * 4, &pte)
                .is_ok());
        }

        // The warm packed allocation: what `ensure_packed_resource` would have
        // retained after one serial bind, held so the plan takes the arm the
        // serial rail takes without resolving anything.
        let base = 0x7f00_0000_0000usize;
        let import = std::sync::Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(base, 0x8000, 0x1000)
                .expect("aligned allocation"),
        );
        let run = crate::runtime::guest_ram::GuestRun::in_mapping(base, 0x8000, 0, span)
            .expect("the run covers the allocation");
        state.bound_buffers.insert_packed(
            1,
            texture_ref,
            PackedBufferResolution::Available(PackedBuffer {
                gva: u64::from(handle) << PAGE_SHIFT_X86,
                size: span,
                head: 0,
                import,
                gpas: std::sync::Arc::new(gpas),
                runs: std::sync::Arc::new(vec![run]),
                pages: std::sync::Arc::new(Vec::new()),
            }),
        );
        (state, host, texture_ref)
    }

    /// The fields of a linear serve two fixtures can agree on (host pointers
    /// differ between fixtures, so the source is compared by shape).
    #[allow(clippy::type_complexity)]
    fn linear_shape(
        served: &(u32, u32, u32, SampledSourceRequest),
    ) -> (
        u32,
        u32,
        u32,
        u64,
        u32,
        u64,
        usize,
        bool,
        TexelLayout,
        Option<LinearSampleIdentity>,
        crate::runtime::gather_witness::GatherVouch,
    ) {
        let (w, h, mid, src) = served;
        let SampledSourceRequest::GuestRuns(source, layout, _vk, planes, identity, vouch, _, _) =
            src
        else {
            panic!("expected a linear guest-run serve");
        };
        (
            *w,
            *h,
            *mid,
            source.total_len,
            source.row_length_texels,
            source.source_offset,
            *planes as usize,
            source.direct_image.is_some(),
            *layout,
            *identity,
            *vouch,
        )
    }

    /// A planned linear zero-copy bind commits to the serial answer (the direct
    /// single-plane arm, off the warm packed allocation), and — that arm moving
    /// nothing — the plan the commit hands back serves the next bind too.
    #[test]
    fn a_planned_linear_bind_commits_to_the_serial_answer() {
        let (mut a, mut a_host, texture_ref) = linear_fixture();
        let (mut b, mut b_host, _) = linear_fixture();
        let (ra, rb) = (
            resource(&a, &a_host, texture_ref),
            resource(&b, &b_host, texture_ref),
        );
        let mut plan = plan_linear_zero_copy(&a, &a_host, 1, texture_ref, ra.clone(), false)
            .unwrap_or_else(|bail| panic!("a warm linear texture plans, got {bail:?}"));
        for _ in 0..2 {
            let (planned, repeat) = commit_linear_plan(
                &mut a,
                &mut a_host,
                &plan,
                1,
                texture_ref,
                Some(&ra),
                false,
                true,
            )
            .expect("nothing moved since the plan");
            let serial = resolve_sampled_source(
                &mut b,
                &mut b_host,
                1,
                texture_ref,
                Some(rb.clone()),
                false,
            )
            .expect("the serial rail serves the same bind");
            assert_eq!(linear_shape(&planned), linear_shape(&serial));
            assert!(
                matches!(&serial.3, SampledSourceRequest::GuestRuns(s, ..) if s.direct_image.is_some()),
                "the serial rail serves the direct arm this plan planned"
            );
            plan = repeat.expect("the direct arm serves again");
        }
    }

    /// A writeback debt armed under the texture's own GVA resource key is the
    /// one refusal the plan takes before anything else: paying it writes guest
    /// pages, which is the drain's to do (design H7).
    #[test]
    fn an_owed_frame_refuses_the_linear_plan_and_invalidates_a_committed_one() {
        let (mut a, mut a_host, texture_ref) = linear_fixture();
        let ra = resource(&a, &a_host, texture_ref);
        // The span the resident identity would be declared at.
        let (gva, row_stride, w, h) = (0x20_000u64, 256u32, 64u32, 64u32);
        let key = crate::runtime::writeback_debt::GvaResourceKey {
            task_id: 1,
            texture_ref,
        };
        let arm = |state: &mut DeviceState| {
            state.pending_writebacks.ensure_gva_resource(
                key,
                gva,
                u64::from(row_stride) * u64::from(h),
                Some(vec![0x28_000]),
            );
            state.pending_writebacks.arm_gva(
                key,
                crate::runtime::writeback_debt::GvaWritebackDebt {
                    gva,
                    row_stride,
                    width: w,
                    height: h,
                    format: MTL_FORMAT_BGRA8_UNORM,
                    generation: 1,
                    guest_write: Default::default(),
                    seq: 0,
                },
            );
        };
        arm(&mut a);
        assert_eq!(
            plan_linear_zero_copy(&a, &a_host, 1, texture_ref, ra.clone(), false).err(),
            Some(SampleBail::OwedFrame),
            "the plan never pays"
        );
        // The same debt armed *after* a plan invalidates it at commit, and the
        // serial rail then pays and serves.
        let (mut b, mut b_host, _) = linear_fixture();
        let rb = resource(&b, &b_host, texture_ref);
        let plan = plan_linear_zero_copy(&b, &b_host, 1, texture_ref, rb.clone(), false)
            .expect("no debt when planned");
        arm(&mut b);
        assert_eq!(
            commit_linear_plan(
                &mut b,
                &mut b_host,
                &plan,
                1,
                texture_ref,
                Some(&rb),
                false,
                true
            )
            .err(),
            Some(SampleInvalid::Debt)
        );
        let serial = resolve_sampled_source(&mut b, &mut b_host, 1, texture_ref, Some(rb), false);
        assert!(serial.is_some(), "the serial rail pays the debt and serves");
    }

    /// The GVA-resident rung above the linear rail is the serial ladder's
    /// answer wherever it may serve, so the plan refuses those binds —
    /// conservatively, including an undeclared resource whose serial
    /// declaration this read-only question cannot predict.
    #[test]
    fn the_resident_rung_above_refuses_the_linear_plan() {
        let (a, a_host, texture_ref) = linear_fixture();
        let ra = resource(&a, &a_host, texture_ref);
        assert_eq!(
            plan_linear_zero_copy(&a, &a_host, 1, texture_ref, ra, true).err(),
            Some(SampleBail::UpperRung),
            "may_bind_resident makes the rung above the serial answer"
        );
    }

    /// Without the retained packed allocation the plan refuses (`ColdPacked`)
    /// and the serial rail resolves or copies on the drain thread.
    #[test]
    fn a_cold_packed_refusal_leaves_the_serial_rail() {
        let (mut a, mut a_host, texture_ref) = linear_fixture();
        a.bound_buffers = crate::runtime::bound_buffers::BoundBuffers::default();
        let ra = resource(&a, &a_host, texture_ref);
        assert_eq!(
            plan_linear_zero_copy(&a, &a_host, 1, texture_ref, ra.clone(), false).err(),
            Some(SampleBail::ColdPacked)
        );
        // And the serial rail still serves the bind, by resolving the
        // allocation itself.
        let serial = resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra), false);
        assert!(
            serial.is_some(),
            "the serial rail resolves what the plan refuses"
        );
    }

    /// A mapper-ref surface above the gather floor (128x128 BGRA8 = 64 KiB,
    /// sixteen scattered pages, a packed-only stable host), with no host cache
    /// and no resident: one serial bind primes everything the zero-copy plan
    /// borrows — the cached refusal of a contig view, the cached window, the
    /// witness entry — and a mapping write token.
    fn mapper_ref_zero_fixture() -> (DeviceState, FakeHost, u32, u32) {
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        let (mut state, mut host, texture_ref, surface_id) = fixture();
        host.stable_map_pages = true;
        host.strict_linux_map = true;
        state.host_surfaces.remove(&surface_id);
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpas: Vec<u64> = (0..16u64).map(|i| 0x5000_0000 + 2 * i * page).collect();
        for (n, &gpa) in gpas.iter().enumerate() {
            host.map_range(gpa, page as usize, 0x40 + n as u8);
        }
        assert!(state.set_mapping_geom(surface_id, 128, 128, MTL_FORMAT_BGRA8_UNORM));
        state.mappings.get_mut(&surface_id).unwrap().page_entries = gpas
            .iter()
            .map(|&gpa| {
                (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
        // The mapping's own token: the memo rung's plan rests on its lagged
        // generation, and the zero-copy witness arms its own.
        let _ = crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, surface_id);
        (state, host, texture_ref, surface_id)
    }

    /// A planned mapper-ref zero-copy bind commits to the serial answer, and
    /// the repeat the commit hands back serves the pass's later binds of the
    /// same ref from the post state, leaving the witness where the serial
    /// observations leave it.
    #[test]
    fn a_planned_mapper_ref_bind_commits_to_the_serial_answer() {
        let (mut a, mut a_host, texture_ref, _) = mapper_ref_zero_fixture();
        let (mut b, mut b_host, _, _) = mapper_ref_zero_fixture();
        let (ra, rb) = (
            resource(&a, &a_host, texture_ref),
            resource(&b, &b_host, texture_ref),
        );
        // One serial bind on each arm primes the window and the witness.
        let first_a =
            resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra.clone()), true)
                .expect("the fixture's first bind serves the zero-copy rung");
        let first_b =
            resolve_sampled_source(&mut b, &mut b_host, 1, texture_ref, Some(rb.clone()), true)
                .expect("the serial arm primes identically");
        assert_eq!(runs_shape(&first_a), runs_shape(&first_b));

        let mut plan = plan_mapper_ref_zero_copy(&a, &a_host, 1, texture_ref, ra.clone())
            .unwrap_or_else(|bail| panic!("a warm mapper-ref window plans, got {bail:?}"));
        for _ in 0..3 {
            let (planned, repeat) =
                commit_mapper_ref_plan(&mut a, &mut a_host, &plan, 1, texture_ref, Some(&ra), true)
                    .expect("nothing moved since the plan");
            let serial =
                resolve_sampled_source(&mut b, &mut b_host, 1, texture_ref, Some(rb.clone()), true)
                    .expect("the serial rail keeps serving the rung");
            assert_eq!(runs_shape(&planned), runs_shape(&serial));
            plan = repeat.expect("the repeat serves the next bind");
        }
        // And the two witnesses keep agreeing on the next serial bind.
        let next_a = resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra), true)
            .expect("the planned arm still serves serially");
        let next_b = resolve_sampled_source(&mut b, &mut b_host, 1, texture_ref, Some(rb), true)
            .expect("the serial arm still serves");
        assert_eq!(runs_shape(&next_a), runs_shape(&next_b));
    }

    /// A guest store the host observed between plan and commit moves the
    /// witness readings, and the plan yields to the serial rail that re-reads.
    #[test]
    fn a_guest_store_drops_a_mapper_ref_plan_at_commit() {
        let (mut a, mut a_host, texture_ref, sid) = mapper_ref_zero_fixture();
        let ra = resource(&a, &a_host, texture_ref);
        let _ = resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra.clone()), true);
        let plan = plan_mapper_ref_zero_copy(&a, &a_host, 1, texture_ref, ra.clone())
            .expect("warm after the priming bind");
        // The guest writes the first page of the window, between the plan and
        // the commit.
        a_host.guest_wrote_page(0x5000_0000);
        assert_eq!(
            commit_mapper_ref_plan(&mut a, &mut a_host, &plan, 1, texture_ref, Some(&ra), true)
                .err(),
            Some(SampleInvalid::Witness)
        );
        let serial = resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra), true);
        assert!(serial.is_some(), "the serial rail re-reads and serves");
        let _ = sid;
    }

    /// A debt under the mapping is the plan's refusal (H7) and a commit's
    /// invalidation; a resolved contig view is the rung above's.
    #[test]
    fn debt_and_the_direct_rung_refuse_a_mapper_ref_plan() {
        let (mut a, mut a_host, texture_ref, sid) = mapper_ref_zero_fixture();
        let ra = resource(&a, &a_host, texture_ref);
        let _ = resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra.clone()), true);
        // The mapping's debt, armed directly on the ledger.
        a.pending_writebacks.arm(
            sid,
            crate::runtime::writeback_debt::test_resident_identity(sid, 128, 128, 1),
            128,
            128,
            1,
        );
        assert_eq!(
            plan_mapper_ref_zero_copy(&a, &a_host, 1, texture_ref, ra.clone()).err(),
            Some(SampleBail::OwedFrame)
        );
        a.pending_writebacks.take(sid);
        // A resolved contig view makes the direct image the serial answer.
        a.mappings.get_mut(&sid).unwrap().contig_ptr = 0x7f00_0000_0000;
        assert_eq!(
            plan_mapper_ref_zero_copy(&a, &a_host, 1, texture_ref, ra).err(),
            Some(SampleBail::UpperRung)
        );
    }

    /// A planned memo bind reads the surface's rows on the worker and commits
    /// to the serial memo's answer — a hit serving the memo's own `Arc` and
    /// identity — and the handed-back plan serves the next bind too. A guest
    /// store the token saw between plan and commit drops it to serial.
    #[test]
    fn a_planned_memo_bind_commits_to_the_serial_answer() {
        let budget = std::sync::atomic::AtomicU64::new(memo_fill_chunk_limit());
        let (mut a, mut a_host, texture_ref, sid) = memo_fixture();
        let (mut b, mut b_host, _, _) = memo_fixture();
        let (ra, rb) = (
            resource(&a, &a_host, texture_ref),
            resource(&b, &b_host, texture_ref),
        );
        // Tokens, then one serial bind on each arm to prime the memo.
        let _ = crate::runtime::mapper::ensure_guest_write_token(&mut a, &mut a_host, sid);
        let _ = crate::runtime::mapper::ensure_guest_write_token(&mut b, &mut b_host, sid);
        let first_a =
            resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra.clone()), true)
                .expect("the memo rung serves the fixture");
        let first_b =
            resolve_sampled_source(&mut b, &mut b_host, 1, texture_ref, Some(rb.clone()), true)
                .expect("the serial arm fills its memo identically");
        let bytes_of = |s: &(u32, u32, u32, SampledSourceRequest)| match &s.3 {
            SampledSourceRequest::Bytes(bytes, identity, fmt, _) => (
                bytes.to_vec(),
                identity.as_ref().map(|i| (i.key, i.generation)),
                fmt.layout(),
            ),
            _ => panic!("expected a byte serve"),
        };
        assert_eq!(bytes_of(&first_a), bytes_of(&first_b));
        let _ = first_a;

        let mut plan = plan_mapper_ref_memo(&a, &a_host, 1, texture_ref, ra.clone(), &budget)
            .unwrap_or_else(|bail| panic!("a primed memo plans, got {bail:?}"));
        for _ in 0..2 {
            let (planned, repeat) =
                commit_memo_plan(&mut a, &mut a_host, &plan, 1, texture_ref, Some(&ra), true)
                    .expect("nothing moved since the plan");
            let serial =
                resolve_sampled_source(&mut b, &mut b_host, 1, texture_ref, Some(rb.clone()), true)
                    .expect("the serial memo keeps serving");
            assert_eq!(bytes_of(&planned), bytes_of(&serial));
            plan = repeat.expect("the hit serves the next bind");
        }
        // A guest store the token observed drops the plan at commit.
        let plan = plan_mapper_ref_memo(&a, &a_host, 1, texture_ref, ra.clone(), &budget)
            .expect("still warm");
        let gpa = 0x5000_0000;
        a_host.guest_wrote_page(gpa);
        assert_eq!(
            commit_memo_plan(&mut a, &mut a_host, &plan, 1, texture_ref, Some(&ra), true).err(),
            Some(SampleInvalid::GuestWrite)
        );
    }

    /// A cold memo (no entry, no primed window) is not planned, a debt under
    /// the mapping is the `OwedFrame` refusal (the serial memo pays before it
    /// reads), and a fill that would exceed the chunk's budget is the
    /// `MemoBudget` refusal.
    #[test]
    fn cold_and_over_budget_memo_binds_stay_serial() {
        let (mut a, mut a_host, texture_ref, sid) = memo_fixture();
        let ra = resource(&a, &a_host, texture_ref);
        // Not primed: the window cache is cold and the token unarmed.
        assert!(matches!(
            plan_mapper_ref_memo(
                &a,
                &a_host,
                1,
                texture_ref,
                ra.clone(),
                &std::sync::atomic::AtomicU64::new(memo_fill_chunk_limit()),
            ),
            Err(SampleBail::ColdWindow | SampleBail::ColdBacking)
        ));
        // Primed by one serial bind, then planned against a spent budget.
        let _ = crate::runtime::mapper::ensure_guest_write_token(&mut a, &mut a_host, sid);
        let _ = resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra.clone()), true);
        // The mapping's debt is the one refusal before anything else: the
        // serial memo settles (pays) before it reads, and a plan never pays.
        a.pending_writebacks.arm(
            sid,
            crate::runtime::writeback_debt::test_resident_identity(sid, 60, 64, 1),
            60,
            64,
            1,
        );
        assert_eq!(
            plan_mapper_ref_memo(
                &a,
                &a_host,
                1,
                texture_ref,
                ra.clone(),
                &std::sync::atomic::AtomicU64::new(memo_fill_chunk_limit()),
            )
            .err(),
            Some(SampleBail::OwedFrame)
        );
        a.pending_writebacks.take(sid);
        // The memo holds the surface now, so a plain plan hits; spend the
        // budget and rewrite a page so the next plan must fill.
        assert!(
            plan_mapper_ref_memo(
                &a,
                &a_host,
                1,
                texture_ref,
                ra.clone(),
                &std::sync::atomic::AtomicU64::new(memo_fill_chunk_limit()),
            )
            .is_ok(),
            "the primed memo plans as a hit"
        );
        assert!(a_host.write_gpa(0x5000_0000, &[9, 9, 9, 9]).is_ok());
        assert_eq!(
            plan_mapper_ref_memo(
                &a,
                &a_host,
                1,
                texture_ref,
                ra,
                &std::sync::atomic::AtomicU64::new(1),
            )
            .err(),
            Some(SampleBail::MemoBudget),
            "a fill with no budget left is the serial memo's"
        );
    }

    /// A resident that appeared between the plan and the commit — the
    /// readiness flip a recorder job landing mid-pass makes — changes the
    /// serial answer's shape above the page rungs, and both page-rung plans
    /// yield the bind to the serial ladder. This is the fs2s1 shadow class
    /// (lab/resolve 394d5c9a: ~3 080 `par_shadow_mismatch_variant` binds,
    /// planned GuestRuns/Bytes against a serial `Target`), reproduced here by
    /// the same lease seeding the surface-plan tests use for their flip.
    #[test]
    fn a_resident_appearing_above_drops_the_page_rung_plans() {
        use crate::backend::vulkan::engine::ResidentContentBacking;
        // The zero-copy arm, above the floor.
        let (mut a, mut a_host, texture_ref, sid) = mapper_ref_zero_fixture();
        let ra = resource(&a, &a_host, texture_ref);
        let _ = resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra.clone()), true);
        let plan =
            plan_mapper_ref_zero_copy(&a, &a_host, 1, texture_ref, ra.clone()).expect("warm");
        let identity =
            crate::backend::vulkan::present_identity::surface_identity(&a, sid, 128, 128);
        resource_lease::seed_warm_lease_for_test(
            &ra,
            &identity,
            ResidentContentBacking::DeviceAllocation,
        );
        assert_eq!(
            commit_mapper_ref_plan(&mut a, &mut a_host, &plan, 1, texture_ref, Some(&ra), true)
                .err(),
            Some(SampleInvalid::UpperRung),
            "the resident that appeared is the serial answer's rung now"
        );
        assert!(matches!(
            resolve_sampled_source(&mut a, &mut a_host, 1, texture_ref, Some(ra), true),
            Some((_, _, _, SampledSourceRequest::Target(..)))
        ));

        // The memo arm, under the floor.
        let (mut b, mut b_host, texture_ref, sid) = memo_fixture();
        let rb = resource(&b, &b_host, texture_ref);
        let _ = crate::runtime::mapper::ensure_guest_write_token(&mut b, &mut b_host, sid);
        let _ = resolve_sampled_source(&mut b, &mut b_host, 1, texture_ref, Some(rb.clone()), true);
        let plan = plan_mapper_ref_memo(
            &b,
            &b_host,
            1,
            texture_ref,
            rb.clone(),
            &std::sync::atomic::AtomicU64::new(memo_fill_chunk_limit()),
        )
        .expect("the primed memo plans");
        let identity = crate::backend::vulkan::present_identity::surface_identity(&b, sid, 60, 64);
        resource_lease::seed_warm_lease_for_test(
            &rb,
            &identity,
            ResidentContentBacking::DeviceAllocation,
        );
        assert_eq!(
            commit_memo_plan(&mut b, &mut b_host, &plan, 1, texture_ref, Some(&rb), true).err(),
            Some(SampleInvalid::UpperRung),
            "the memo rung yields the same way"
        );
        assert!(matches!(
            resolve_sampled_source(&mut b, &mut b_host, 1, texture_ref, Some(rb), true),
            Some((_, _, _, SampledSourceRequest::Target(..)))
        ));
    }
}
