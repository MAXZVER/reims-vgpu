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
enum PlannedServe {
    Target(TargetIdentity),
    HostCache(std::sync::Arc<Vec<u8>>),
}

/// A planned sampled-source bind: the serve a top rung chose, the reads it rests
/// on, and the census routes it owes.
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
            let format = identity.resident_format();
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
/// witness when it is applied, so a plan serves one bind of its ref per stream;
/// the next bind of the same ref finds the entry moved and takes the serial path.
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
    if !state.pending_writebacks.is_empty() && state.pending_writebacks.get(mid).is_some() {
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
    })
}

/// Re-check a ref-view plan at the point the serial rail would run, and with
/// `apply` take it: the field-window note and the routes the serial rail makes,
/// the witness update, and the plane's guest-page window.
///
/// Without `apply` nothing is written or noted, and the answer is the plan's
/// prediction with no page window, for a shadow comparison against the serial
/// rail that runs next.
pub(crate) fn commit_ref_view_plan<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    plan: &RefViewPlan,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    apply: bool,
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
        || ref_view_plane(&*state, &*host, mid, plan.view) != Some(plan.plane)
    {
        return Err(SampleInvalid::Mapping);
    }
    if !state.pending_writebacks.is_empty() && state.pending_writebacks.get(mid).is_some() {
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
                pixel_format::swizzle_identity(),
            ),
        )
    };
    if !apply {
        return Ok(source(None, plan.gather.predicted()));
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
    let outcome =
        crate::runtime::gather_witness::commit_gather(state, &*host, &plan.gather, &plan.gpas)
            .ok_or(SampleInvalid::Witness)?;
    let page = state.page_size();
    let pages = guest_page_window(host, &plan.gpas, page, base_off % page, span);
    Ok(source(pages, outcome))
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
        let SampledSourceRequest::GuestRuns(source, layout, format, _, identity, vouch, _) = src
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
            let planned =
                commit_ref_view_plan(&mut b, &mut b_host, &plan, 1, texture_ref, Some(&rb), true)
                    .expect("nothing moved since the plan");
            let serial_a = serial(&mut a, &mut a_host, texture_ref, &ra);
            assert_eq!(runs_shape(&planned), runs_shape(&serial_a));
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
        let predicted =
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
}
