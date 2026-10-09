//! The Vulkan rail's half of the writeback: a resident's pixels into the
//! guest's own pages, with no CPU round trip.
//!
//! [`super`] owns what every rail owes the guest — the settle sites, and the
//! wait that has to happen before any host reader touches pages a GPU may still
//! be writing. That half is neutral: it asks
//! [`crate::backend::Backend::guest_writes_outstanding`] and reads the answer.
//!
//! This half is the copy itself, and it is Vulkan's alone because a resident is:
//! every entry point here takes a
//! [`crate::backend::vulkan::engine::TargetIdentity`], and a rail that holds no
//! residents has nothing to name. The three destinations are the three
//! namespaces the guest renders into — a mapper-ref-texture mapping, a guest-backed
//! surface, and a raw GVA plane — and each is reached only from that rail's own
//! draw, blit or compute path.
//!
//! Split out rather than left as eighteen `cfg`-ed items below the neutral
//! settle. The gate is the module's, once.

use super::*;
use crate::runtime::host::{HostMemory, HostOps};

/// Copy `identity`'s pixels into `mapping_id`'s guest pages.
///
/// `true` when the guest's pages hold the frame. `false` is a real loss and is
/// reported on the failure channel by the arm that refused — the caller has no
/// second copy to fall back to, because this rail never made one.
pub fn store_render_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> bool {
    let stored = store_render_frame_inner(state, host, mapping_id, identity, width, height);
    if stored {
        lab_probe_small_frame(state, host, mapping_id, width, height);
    }
    stored
}

/// TEMP (lab): what the guest will read back from a small surface (icon-sized)
/// right after this device landed a frame in it — once per mapping for an
/// all-zero landing and once for a non-zero one.
fn lab_probe_small_frame<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) {
    if u64::from(width) * u64::from(height) > 256 * 256 {
        return;
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return;
    };
    let page_size = state.page_size();
    let shift = page_size.trailing_zeros();
    let mut buf = vec![0u8; page_size as usize];
    let (mut nonzero, mut total, mut alpha_hi) = (0usize, 0usize, 0usize);
    let bpp = crate::protocol::pixel_format::bytes_per_pixel(m.format).unwrap_or(4) as usize;
    let mut unreadable = 0usize;
    for &entry in m.page_entries.iter().take(128) {
        let Some(gpa) = crate::protocol::iosurface_pages::entry_gpa_shift(entry, shift) else {
            unreadable += 1;
            continue;
        };
        if host.read_gpa(gpa, &mut buf).is_ok() {
            total += buf.len();
            nonzero += buf.iter().filter(|b| **b != 0).count();
            // texels with any non-zero byte (alpha_hi reused as "non-empty texels")
            alpha_hi += buf
                .chunks_exact(bpp.max(1))
                .filter(|t| t.iter().any(|b| *b != 0))
                .count();
        } else {
            unreadable += 1;
        }
    }
    let key = (u64::from(mapping_id) << 1) | u64::from(nonzero > 0);
    if crate::observe::first_sight("lab_small_frame", key) {
        crate::observe::fail(format!(
            "lab_small_frame mid={mapping_id} {width}x{height} fmt={:#x} pages={} unreadable={unreadable} nonzero_bytes={nonzero}/{total} nonempty_texels={alpha_hi}",
            m.format,
            m.page_entries.len()
        ));
    }
}

fn store_render_frame_inner<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> bool {
    let started = std::time::Instant::now();
    crate::runtime::drain::note_store_route("surface_flush");
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        let now = crate::observe::elapsed_us();
        // TEMP (lab, b037): which surfaces the 1.4 GB a Safari phase of render
        // flushes are - geometry, whether scanout presented it, how long since
        // the previous flush into it. Capped per boot.
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4000 {
            crate::observe::off(format!(
                "lab_flush_trace mid={mapping_id} {width}x{height} presented={} mapgen={} since_prev_ms={}",
                u8::from(m.scanout_presented),
                m.map_generation,
                if m.lab_flush_us == 0 { -1 } else { (now.saturating_sub(m.lab_flush_us) / 1000) as i64 },
            ));
        }
        m.lab_flush_us = now;
    }
    if crate::runtime::writeback_debt::pay_diff_enabled()
        && state
            .mappings
            .get(&mapping_id)
            .is_some_and(|m| m.scanout_presented)
    {
        if let Some(done) =
            store_changed_pages(state, host, mapping_id, identity, width, height, started)
        {
            return done;
        }
    }
    // The GPU writes the guest's pages directly. Tried first because when it
    // works there is nothing left to do: no staging buffer is mapped and no
    // host pass over the frame happens at all.
    match crate::runtime::mapping_write::vulkan::write_bgra8_from_resident_gpu(
        state, host, mapping_id, identity, width, height,
    ) {
        Ok(bytes) => {
            crate::runtime::drain::note_store_route("render_flush_gpu_direct");
            finish(state, mapping_id, identity, bytes as usize, started, false);
            return true;
        }
        Err(decline) => {
            // Latched per mapping as well as per reason: a host without
            // `VK_EXT_external_memory_host` declines every Store of every
            // surface, and a line each would drown the channel.
            crate::observe::Emit::decline("render_flush_gpu_declined", &decline)
                .field("mapping", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .fail_once(u64::from(mapping_id));
            crate::runtime::drain::note_store_route("render_flush_gpu_declined");
        }
    }
    if let Some(done) = store_native_frame(state, host, mapping_id, identity, width, height, started)
    {
        return done;
    }
    // The copying arms. These are the only arms on a host that cannot import
    // guest RAM, and the arm a discrete GPU takes regardless.
    //
    // Borrow the readback where it needs no transformation. The writer below is
    // declared in guest scanout order, so a resident reporting semantic RGBA8
    // owes an R/B exchange first — a whole-frame pass, and `into_bgra8` on an
    // owned copy is its home, so a non-BGRA resident takes the copy rather than
    // teaching the lease to rewrite memory it does not own.
    let bpr = width.saturating_mul(4);
    let write_started = std::time::Instant::now();
    // A refused lease is a *routing* answer, never a loss. The lease is an
    // elision of one whole-frame copy and nothing else, so whatever it declines
    // for, the copying rail below can still serve — and it is strictly the
    // better place to find out that the frame is unreadable, because it is the
    // rail that owns the loss report.
    //
    // Collapsing the refusal into `None` rather than enumerating which refusals
    // are recoverable. That enumeration is what would go stale: the lease
    // refuses a resident wider than four bytes a texel (it hands out a pointer
    // into the slot and so has nowhere to narrow one), and reading that
    // particular `Err` as fatal would lose exactly the frames `read_target`
    // exists to quantize. Any future refusal of a rail that is an optimisation
    // has the same answer.
    //
    // Measured, and only reachable one way. On a host that can import guest RAM
    // every Store lands GPU-direct and this whole tail runs zero times — six
    // driven rails, `render_flush_copied` and `render_flush_leased` exactly zero
    // on all of them. With `REIMS_VGPU_GUEST_IMPORT=off`, macos-26 produces
    // **34 lease refusals and 34 copies, one to one, and zero
    // `render_store_lost`**: every one a Surface resident too wide for the lease
    // to lend. Reading the `Err` as fatal cost 34 frames a boot on exactly the
    // host class that has no second rail, and no boot of a capable host could
    // have shown it.
    let leased = match crate::backend::vulkan::engine::read_target_leased(identity) {
        Ok(leased) => leased,
        Err(decline) => {
            // The typed fields, not `Display`. `TargetReadDecline::UnknownIdentity`
            // carries `diverges`/`asked_gen`/`held_gen` precisely because "not in
            // the registry" is two findings with opposite repairs, and formatting
            // the error dropped all three: a `REIMS_VGPU_GUEST_IMPORT=off` boot
            // produced 80 of these and 80 `render_store_lost` beside them with
            // nothing in the log saying which of the two it was.
            crate::observe::Emit::decline("render_flush_lease_declined", &decline)
                .field("mapping", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .off();
            None
        }
    };
    let (ok, frame_len) = match leased {
        Some(leased) if leased.bgra => {
            crate::runtime::drain::note_store_route("render_flush_leased");
            let len = leased.bytes().len();
            let ok = crate::runtime::mapping_write::write_bgra8_uncached(
                state,
                host,
                mapping_id,
                leased.bytes(),
                bpr,
                width,
                height,
            );
            // End the lease before anything below reaches the engine again: the
            // re-stamp in `finish` does, and a holder blocking on the engine
            // lock while a teardown waits for this lease is the deadlock
            // `LeasedFrame` forbids.
            drop(leased);
            (ok, len)
        }
        // The pool declined the lease (uncached readback memory, where reading
        // the mapping in place is the slower shape), or it refused outright, or
        // the resident is not in scanout order. Drop any leased frame first so
        // its slot is back in the pool before the second readback asks for one.
        leased => {
            drop(leased);
            crate::runtime::drain::note_store_route("render_flush_copied");
            match crate::backend::vulkan::engine::read_target(identity) {
                Ok(rb) => {
                    // A mapping is scanout-ordered eight-bit colour, so a native
                    // readback has nothing this rail can land. Named rather than
                    // reinterpreted — the bytes are a texel the mapping cannot
                    // mean.
                    let texel = rb.texel;
                    match rb.into_bgra8() {
                        Some(scanout) => {
                            // Shared rather than owned outright: the write's
                            // tail publishes this frame to the surface cache,
                            // and a cache entry holds its frame behind an `Arc`
                            // precisely so the two can name one allocation
                            // instead of copying it.
                            let bytes = std::sync::Arc::new(scanout);
                            let len = bytes.len();
                            let ok = crate::runtime::mapping_write::write_bgra8_owned(
                                state, host, mapping_id, &bytes, bpr, width, height,
                            );
                            (ok, len)
                        }
                        None => {
                            crate::observe::fail(format!(
                                "render_store_lost reason=readback_texel_not_scanout \
                                 mapping={mapping_id} geom={width}x{height} texel={texel:?}"
                            ));
                            (false, 0)
                        }
                    }
                }
                Err(e) => {
                    // Typed, for the reason the lease decline above is: this is
                    // the one line that says *why* the frame was unreadable, and
                    // `Display` on a `DrawError` prints the slug alone.
                    crate::observe::Emit::decline("render_store_lost", &e)
                        .field("mapping", mapping_id)
                        .field("geom", format!("{width}x{height}"))
                        .field("at", "resident_read")
                        .fail();
                    return false;
                }
            }
        }
    };
    crate::runtime::drain::note_readback_phase(
        crate::runtime::drain::ReadbackPhase::Write,
        write_started.elapsed().as_micros() as u64,
    );
    if !ok {
        crate::observe::fail(format!(
            "render_store_lost mapping={mapping_id} {width}x{height} reason=write_refused"
        ));
        return false;
    }
    finish(state, mapping_id, identity, frame_len, started, false);
    true
}

/// TEMP (lab, live `NATIVE_PAY=on|off`, on unless `=off`): whether a copying
/// writeback of a resident whose texels are not four-byte colour lands them as
/// they are ([`store_native_frame`]).
fn lab_native_pay() -> bool {
    if cfg!(test) {
        return true;
    }
    crate::runtime::lab_live::on("NATIVE_PAY").unwrap_or(true)
}

/// The copying writeback of a resident whose texels are not four-byte colour,
/// into a mapping declared in that same texel layout, as the resident's own
/// bytes.
///
/// The eight-bit rail read such a resident back through a lease it then
/// refused (a second readback), widened every texel to RGBA8, exchanged R and
/// B, and narrowed each row back to the mapping's texel on its way into guest
/// pages: for the 4935x1895 R8 transition mask, two readbacks and three
/// whole-frame passes over four times its bytes (p062: half of the 45 ms
/// tranche that opens every Finder full-screen animation was this payment,
/// convert 27 % of it). Here: one readback, one write of the bytes the guest
/// declared.
///
/// `None` hands the frame to the eight-bit rail untouched: a four-byte colour
/// mapping (the common case, which that rail exists for), a format with no
/// sampled layout, or a resident that does not hold the mapping's layout - the
/// last one after its readback, which is then landed by the eight-bit rail
/// from the bytes already read rather than a second readback.
fn store_native_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
    started: std::time::Instant,
) -> Option<bool> {
    if !lab_native_pay() {
        return None;
    }
    let m = state.mappings.get(&mapping_id)?;
    // The geometry the native writer lands at (a mapping that declared its own);
    // anything else is the eight-bit rail's, refusal included.
    if !m.has_geom || m.width != width || m.height != height {
        return None;
    }
    let format = crate::runtime::mapping_write::mapping_store_format(m);
    let (layout, _, _) = crate::backend::vulkan::translate::pixel::sampled_pixels(format).ok()?;
    if layout.is_four_byte_color() {
        return None;
    }
    let write_started = std::time::Instant::now();
    let read = match crate::backend::vulkan::engine::read_target_keeping(identity, Some(layout)) {
        Ok(read) => read,
        // The eight-bit rail reads again and owns the loss report.
        Err(_) => return None,
    };
    let (ok, frame_len) = match read {
        crate::backend::vulkan::engine::TargetRead::Native(bytes, _) => {
            crate::runtime::drain::note_store_route("render_flush_native");
            let stride = width.saturating_mul(u32::from(layout.bytes_per_texel()));
            let ok = crate::runtime::mapping_write::write_native_image(
                state, host, mapping_id, &bytes, stride, width, height, format,
            );
            (ok, bytes.len())
        }
        crate::backend::vulkan::engine::TargetRead::Narrowed(rb) => {
            crate::runtime::drain::note_store_route("render_flush_native_layout_differs");
            let texel = rb.texel;
            let Some(scanout) = rb.into_bgra8() else {
                crate::observe::fail(format!(
                    "render_store_lost reason=readback_texel_not_scanout \
                     mapping={mapping_id} geom={width}x{height} texel={texel:?}"
                ));
                return Some(false);
            };
            let bytes = std::sync::Arc::new(scanout);
            let len = bytes.len();
            let ok = crate::runtime::mapping_write::write_bgra8_owned(
                state,
                host,
                mapping_id,
                &bytes,
                width.saturating_mul(4),
                width,
                height,
            );
            (ok, len)
        }
    };
    crate::runtime::drain::note_readback_phase(
        crate::runtime::drain::ReadbackPhase::Write,
        write_started.elapsed().as_micros() as u64,
    );
    if !ok {
        crate::observe::fail(format!(
            "render_store_lost mapping={mapping_id} {width}x{height} reason=write_refused rail=native"
        ));
        return Some(false);
    }
    finish(state, mapping_id, identity, frame_len, started, false);
    Some(true)
}

/// Publish a Store for a resident whose attachment memory is the guest mapping.
pub fn store_guest_backed_frame(
    state: &mut DeviceState,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
    guest_store_recorded: bool,
    guest_store_footprint: Option<crate::runtime::guest_ram::GuestPageFootprint>,
) -> Result<(), crate::runtime::mapping_write::vulkan::GpuWritebackDecline> {
    let started = std::time::Instant::now();
    crate::runtime::drain::note_store_route("surface_flush");
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        m.lab_flush_us = crate::observe::elapsed_us();
    }
    let bytes = crate::runtime::mapping_write::vulkan::synchronize_guest_backed_resident(
        state,
        mapping_id,
        identity,
        width,
        height,
        guest_store_recorded,
        guest_store_footprint,
    )?;
    crate::runtime::drain::note_store_route("render_flush_gpu_direct");
    finish(state, mapping_id, identity, bytes as usize, started, true);
    Ok(())
}

/// [`store_render_frame`] for a presented framebuffer, moving only the pages the
/// GPU found changed (`engine::read_target_changed_pages`). `None` hands the
/// frame back to the whole-frame arms.
fn store_changed_pages<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
    started: std::time::Instant,
) -> Option<bool> {
    // TEMP (lab, REIMS_VGPU_LAB_PAY_SKIP=1): pay nothing. The guest's pages go
    // stale, so this is only an upper bound on what a free payment would buy
    // (drain duty and present_hz over a short window), never a mode to run.
    static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *SKIP.get_or_init(|| std::env::var("REIMS_VGPU_LAB_PAY_SKIP").is_ok_and(|v| v == "1")) {
        crate::runtime::drain::note_store_route("lab_pay_skipped");
        return Some(true);
    }
    let m = state.mappings.get(&mapping_id)?;
    let (base_off, bpr, format) =
        crate::runtime::mapping_write::resident_gpu_plane(m, width, height)?;
    if u64::from(bpr) != u64::from(width) * 4
        || (format != crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM
            && format != crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB)
    {
        crate::runtime::drain::note_store_route("pay_diff_plane_not_packed_bgra");
        return None;
    }
    let key = (m.map_generation, width, height, m.guest_bytes_seq);
    let (token, baseline) = (m.guest_write_token, m.pay_diff_gen);
    let frame_bytes = u64::from(width) * u64::from(height) * 4;
    // TEMP (lab, `REIMS_VGPU_LAB_PAY_EARLY_FLUSH`): a draw batch still recording
    // here (draws the swap's rescue drains added, or a payment not at a swap)
    // goes to the GPU before the CPU-side preparation below, not with the
    // compare after it. See `writeback_debt::lab_pay_early_flush`.
    if crate::runtime::writeback_debt::lab_pay_early_flush()
        && crate::backend::vulkan::engine::lab_batch_open()
    {
        let _lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_flush_us");
        crate::runtime::drain::note_store_route("lab_pd_early_flushed_batch");
        crate::backend::vulkan::engine::flush_batched_draws(state);
    }
    // The generation this landing will stand for, read before it: a guest store
    // racing the landing then counts as written since, and is landed next time.
    let lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_gen_us"); // TEMP (lab)
    let gen_now = if token != 0 {
        host.guest_write_gen(token).filter(|&g| g != 0)
    } else {
        None
    };
    drop(lab_span);
    let lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_force_us"); // TEMP (lab)
    let force = match (baseline, gen_now) {
        (0, _) | (_, None) => None,
        (b, Some(g)) if g == b => Some(Vec::new()),
        (b, Some(_)) => host.guest_written_pages(token, b).map(|pages| {
            let ranges =
                crate::runtime::mapper::mapping_offsets_of_pages(state, mapping_id, &pages);
            forced_frame_pages(&ranges, base_off, frame_bytes)
        }),
    };
    let force = match force {
        Some(f) => f,
        None => {
            // No witness to say which guest pages moved since the copy was
            // taken: land the whole frame and take the copy again.
            crate::runtime::drain::note_store_route("pay_diff_unwitnessed_full");
            crate::backend::vulkan::engine::page_diff_forget(mapping_id);
            Vec::new()
        }
    };
    drop(lab_span);
    crate::runtime::drain::note_store_route_n("pay_diff_pages_forced", force.len() as u64);
    // TEMP (lab, `REIMS_VGPU_LAB_PAY_SPLIT`): the vouch gates only the landing,
    // so with the split on it runs while the GPU computes the compare rather
    // than before the compare is submitted. See `vouch_after_submit`.
    let split = lab_pay_split();
    let mut vouched: Option<Option<crate::runtime::mapper::PagesVouched>> = None;
    if !split {
        let _lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_vouch_us"); // TEMP (lab)
        let (_, v) = crate::runtime::mapper::vouch_mapping_pages_verdict(state, host, mapping_id);
        v.as_ref()?;
        vouched = Some(v);
    }
    let lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_read_us"); // TEMP (lab)
    let read = crate::backend::vulkan::engine::read_target_changed_pages_with(
        identity,
        mapping_id,
        key,
        &force,
        || {
            if split {
                let _lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_vouch_us");
                vouched = Some(
                    crate::runtime::mapper::vouch_mapping_pages_verdict(state, host, mapping_id).1,
                );
            }
        },
    );
    let changed = match read {
        Ok(Some(c)) => c,
        Ok(None) => {
            crate::runtime::drain::note_store_route("pay_diff_declined");
            return None;
        }
        Err(e) => {
            crate::observe::Emit::decline("pay_diff_failed", &e)
                .field("mapping", mapping_id)
                .fail_once(u64::from(mapping_id));
            return None;
        }
    };
    drop(lab_span);
    let vouched = match vouch_after_submit(vouched) {
        VouchAfterSubmit::Vouched(v) => v,
        VouchAfterSubmit::Refused => {
            // The shadow now holds this frame and the guest's pages will not
            // receive it from here: forget it, so the next payment lands whole.
            crate::runtime::drain::note_store_route("lab_pd_split_vouch_refused");
            crate::backend::vulkan::engine::page_diff_forget(mapping_id);
            return None;
        }
        VouchAfterSubmit::NotRun => {
            // The compare answered without the hook having run (cannot happen
            // on an answered read; kept total rather than assumed).
            crate::runtime::drain::note_store_route("lab_pd_split_vouch_late");
            match crate::runtime::mapper::vouch_mapping_pages_verdict(state, host, mapping_id).1 {
                Some(v) => v,
                None => {
                    crate::backend::vulkan::engine::page_diff_forget(mapping_id);
                    return None;
                }
            }
        }
    };
    land_changed_pages(
        state, host, mapping_id, identity, width, height, base_off, gen_now, started, changed,
        vouched,
    )
}

/// The landing half of a changed-pages payment: the compare's answer, landed in
/// the guest's pages under `vouched`, and the payment's bookkeeping. Shared by
/// the synchronous path and TEMP (lab PAY_DEFER) [`land_render_frame`].
#[allow(clippy::too_many_arguments)]
fn land_changed_pages<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
    base_off: u64,
    gen_now: Option<u64>,
    started: std::time::Instant,
    changed: crate::backend::vulkan::engine::ChangedPages,
    vouched: crate::runtime::mapper::PagesVouched,
) -> Option<bool> {
    let page = 4096u64;
    let frame_len = changed.frame_bytes as usize;
    // SAFETY: `frame_ptr` is the engine's mapped page-diff host buffer, at least
    // `frame_bytes` long, written by the call above and not touched again until
    // the next payment — which runs on this same drain thread after this one.
    let frame = unsafe { std::slice::from_raw_parts(changed.frame_ptr, frame_len) };
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &p in &changed.pages {
        let lo = u64::from(p) * page;
        let hi = (lo + page).min(changed.frame_bytes);
        match runs.last_mut() {
            Some(last) if last.1 == base_off + lo => last.1 = base_off + hi,
            _ => runs.push((base_off + lo, base_off + hi)),
        }
    }
    crate::runtime::drain::note_store_route(if changed.full {
        "pay_diff_full"
    } else {
        "pay_diff_pages"
    });
    crate::runtime::drain::note_store_route_n("pay_diff_pages_landed", changed.pages.len() as u64);
    let lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_land_us"); // TEMP (lab)
    if !runs.is_empty()
        && !crate::runtime::mapper::write_mapping_bytes_only(
            state,
            host,
            mapping_id,
            base_off,
            frame,
            Some(&runs),
            &vouched,
        )
    {
        crate::backend::vulkan::engine::page_diff_forget(mapping_id);
        if let Some(m) = state.mappings.get_mut(&mapping_id) {
            m.pay_diff_gen = 0;
        }
        crate::observe::fail(format!(
            "render_store_lost mapping={mapping_id} {width}x{height} reason=pay_diff_write_refused"
        ));
        return Some(false);
    }
    drop(lab_span);
    lab_pay_verify(state, host, mapping_id, identity, base_off, changed.frame_bytes);
    let lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_tail_us"); // TEMP (lab)
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        m.pay_diff_gen = gen_now.unwrap_or(0);
    }
    state.invalidate_storage_residency_window(mapping_id, base_off, base_off + changed.frame_bytes);
    let _ = state.mark_mapping_written(mapping_id);
    if let Some(m) = state.mappings.get(&mapping_id) {
        crate::backend::vulkan::engine::page_diff_set_key(
            mapping_id,
            (m.map_generation, width, height, m.guest_bytes_seq),
        );
    }
    drop(lab_span);
    finish(state, mapping_id, identity, frame_len, started, false);
    Some(true)
}

/// TEMP (lab PAY_DEFER): a presented frame's changed-pages payment whose compare
/// is on the GPU and whose landing waits for a later call (see
/// `writeback_debt::pay_deferred`). It holds no vouch: the answer is read and
/// vouched when the compare is done, and its bytes are copied into the guest's
/// pages by the landing thread while the drain goes on (`Stage::Copying`); the
/// payment's bookkeeping runs once that copy is in.
pub struct DeferredStore {
    stage: Stage,
    pub(crate) identity: crate::backend::vulkan::engine::TargetIdentity,
    mapping_id: u32,
    width: u32,
    height: u32,
    base_off: u64,
    gen_now: Option<u64>,
    started: std::time::Instant,
}

enum Stage {
    /// The compare is submitted.
    Compare(crate::backend::vulkan::engine::CompareTicket),
    /// The changed pages are being copied into the guest's pages.
    Copying(LandJob),
    /// Consumed.
    Taken,
}

impl DeferredStore {
    /// Whether the next step would not wait: the compare has finished on the
    /// GPU, or the landing copy is in.
    pub(crate) fn ready(&self) -> bool {
        match &self.stage {
            Stage::Compare(ticket) => crate::backend::vulkan::engine::target_compare_done(ticket),
            Stage::Copying(job) => job.done(),
            Stage::Taken => true,
        }
    }
}

/// TEMP (lab PAY_DEFER): one landing copy on the landing thread — the selected
/// runs of a compare's frame-shaped answer into the mapping's packed guest view.
/// Dropping one waits for its copy: a landing abandoned half-way must not keep
/// writing guest pages behind the drain's back.
struct LandJob {
    done: std::sync::Arc<LandDone>,
    /// Mapping-linear `[lo, hi)` runs copied, for the footprint marks.
    runs: Vec<(u64, u64)>,
    /// Page runs this landing mapped itself (`(ptr, len)`), unmapped by the drain
    /// once the copy is in; empty when it wrote through the mapping's packed view.
    mapped: Vec<(usize, usize)>,
    frame_len: usize,
    pages: usize,
    full: bool,
}

#[derive(Default)]
struct LandDone {
    flag: std::sync::atomic::AtomicBool,
    lock: std::sync::Mutex<()>,
    cv: std::sync::Condvar,
}

/// `(dst, src, n)`: raw addresses, valid until the job is waited for — the
/// destination is a packed guest view the drain keeps (a map-family command
/// lands the payment first), the source the page-diff answer buffer no compare
/// reuses before the payment lands.
type LandSpans = Vec<(usize, usize, usize)>;

impl LandJob {
    fn done(&self) -> bool {
        self.done.flag.load(std::sync::atomic::Ordering::Acquire)
    }

    fn wait(&self) {
        if self.done() {
            return;
        }
        let _span = crate::runtime::chain_phase::CostSpan::new("lab_pay_defer_copy_wait_us");
        let mut g = self.done.lock.lock().unwrap_or_else(|e| e.into_inner());
        while !self.done() {
            g = self.done.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Hand `spans` to the landing thread (started on first use).
    fn spawn(
        spans: LandSpans,
        runs: Vec<(u64, u64)>,
        mapped: Vec<(usize, usize)>,
        frame_len: usize,
        pages: usize,
        full: bool,
    ) -> Self {
        use std::sync::{mpsc, Mutex, OnceLock};
        static TX: OnceLock<Mutex<mpsc::Sender<(LandSpans, std::sync::Arc<LandDone>)>>> =
            OnceLock::new();
        let done = std::sync::Arc::new(LandDone::default());
        let tx = TX.get_or_init(|| {
            let (tx, rx) = mpsc::channel::<(LandSpans, std::sync::Arc<LandDone>)>();
            std::thread::Builder::new()
                .name("reims-land".into())
                .spawn(move || {
                    while let Ok((spans, done)) = rx.recv() {
                        for (dst, src, n) in spans {
                            // SAFETY: see `LandSpans`; the ranges were bounds-checked
                            // against the view and the answer buffer by the drain.
                            unsafe {
                                crate::runtime::mapper::write_guest_bytes_raw(
                                    src as *const u8,
                                    dst as *mut u8,
                                    n,
                                )
                            };
                        }
                        let _g = done.lock.lock().unwrap_or_else(|e| e.into_inner());
                        done.flag.store(true, std::sync::atomic::Ordering::Release);
                        done.cv.notify_all();
                    }
                })
                .expect("spawn the landing thread");
            Mutex::new(tx)
        });
        let sent = tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send((spans, done.clone()));
        if let Err(mpsc::SendError((spans, _))) = sent {
            // No landing thread: copy here, so the job is as done as it would be.
            for (dst, src, n) in spans {
                // SAFETY: as on the landing thread.
                unsafe { crate::runtime::mapper::write_guest_bytes_raw(src as *const u8, dst as *mut u8, n) };
            }
            done.flag.store(true, std::sync::atomic::Ordering::Release);
        }
        Self { done, runs, mapped, frame_len, pages, full }
    }
}

impl Drop for LandJob {
    fn drop(&mut self) {
        self.wait();
        if !self.mapped.is_empty() {
            // Abandoned before `complete_landing` (a stale landing): its views stay
            // mapped. Rare, and named so it cannot grow unseen.
            crate::runtime::drain::note_store_route_n("lab_pay_defer_views_leaked", self.mapped.len() as u64);
        }
    }
}

/// TEMP (lab PAY_DEFER): what [`begin_render_frame`] did.
pub enum BeginStore {
    /// Paid now, as [`store_render_frame`] answers.
    Done(bool),
    /// The compare is submitted; [`land_render_frame`] finishes it.
    Deferred(DeferredStore),
}

/// TEMP (lab PAY_DEFER): [`store_render_frame`], stopping after the
/// changed-pages compare is submitted when that is the arm the frame takes.
pub fn begin_render_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> BeginStore {
    let started = std::time::Instant::now();
    if crate::runtime::writeback_debt::pay_diff_enabled()
        && state
            .mappings
            .get(&mapping_id)
            .is_some_and(|m| m.scanout_presented)
    {
        if let Some(d) = begin_changed_pages(state, host, mapping_id, identity, width, height, started)
        {
            crate::runtime::drain::note_store_route("lab_pay_deferred");
            return BeginStore::Deferred(d);
        }
    }
    BeginStore::Done(store_render_frame(state, host, mapping_id, identity, width, height))
}

/// TEMP (lab PAY_DEFER): the preparation of [`store_changed_pages`] and the
/// compare's submit, without its wait. `None` hands the frame to the synchronous
/// arms, which ask again.
fn begin_changed_pages<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
    started: std::time::Instant,
) -> Option<DeferredStore> {
    // TEMP (lab, REIMS_VGPU_LAB_PAY_SKIP=1): pay nothing. The guest's pages go
    // stale, so this is only an upper bound on what a free payment would buy
    // (drain duty and present_hz over a short window), never a mode to run.
    static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *SKIP.get_or_init(|| std::env::var("REIMS_VGPU_LAB_PAY_SKIP").is_ok_and(|v| v == "1")) {
        crate::runtime::drain::note_store_route("lab_pay_skipped");
        return None; // the synchronous arm pays nothing too
    }
    let m = state.mappings.get(&mapping_id)?;
    let (base_off, bpr, format) =
        crate::runtime::mapping_write::resident_gpu_plane(m, width, height)?;
    if u64::from(bpr) != u64::from(width) * 4
        || (format != crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM
            && format != crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB)
    {
        crate::runtime::drain::note_store_route("pay_diff_plane_not_packed_bgra");
        return None;
    }
    let key = (m.map_generation, width, height, m.guest_bytes_seq);
    let (token, baseline) = (m.guest_write_token, m.pay_diff_gen);
    let frame_bytes = u64::from(width) * u64::from(height) * 4;
    // TEMP (lab, `REIMS_VGPU_LAB_PAY_EARLY_FLUSH`): a draw batch still recording
    // here (draws the swap's rescue drains added, or a payment not at a swap)
    // goes to the GPU before the CPU-side preparation below, not with the
    // compare after it. See `writeback_debt::lab_pay_early_flush`.
    if crate::runtime::writeback_debt::lab_pay_early_flush()
        && crate::backend::vulkan::engine::lab_batch_open()
    {
        let _lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_flush_us");
        crate::runtime::drain::note_store_route("lab_pd_early_flushed_batch");
        crate::backend::vulkan::engine::flush_batched_draws(state);
    }
    // The generation this landing will stand for, read before it: a guest store
    // racing the landing then counts as written since, and is landed next time.
    let lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_gen_us"); // TEMP (lab)
    let gen_now = if token != 0 {
        host.guest_write_gen(token).filter(|&g| g != 0)
    } else {
        None
    };
    drop(lab_span);
    let lab_span = crate::runtime::chain_phase::CostSpan::new("lab_pd_force_us"); // TEMP (lab)
    let force = match (baseline, gen_now) {
        (0, _) | (_, None) => None,
        (b, Some(g)) if g == b => Some(Vec::new()),
        (b, Some(_)) => host.guest_written_pages(token, b).map(|pages| {
            let ranges =
                crate::runtime::mapper::mapping_offsets_of_pages(state, mapping_id, &pages);
            forced_frame_pages(&ranges, base_off, frame_bytes)
        }),
    };
    let force = match force {
        Some(f) => f,
        None => {
            // No witness to say which guest pages moved since the copy was
            // taken: land the whole frame and take the copy again.
            crate::runtime::drain::note_store_route("pay_diff_unwitnessed_full");
            crate::backend::vulkan::engine::page_diff_forget(mapping_id);
            Vec::new()
        }
    };
    drop(lab_span);
    crate::runtime::drain::note_store_route_n("pay_diff_pages_forced", force.len() as u64);
    let ticket = match crate::backend::vulkan::engine::submit_target_compare(
        identity,
        mapping_id,
        key,
        &force,
        || {},
    ) {
        Ok(Some(ticket)) => ticket,
        Ok(None) => {
            crate::runtime::drain::note_store_route("pay_diff_declined");
            return None;
        }
        Err(e) => {
            crate::observe::Emit::decline("pay_diff_failed", &e)
                .field("mapping", mapping_id)
                .fail_once(u64::from(mapping_id));
            return None;
        }
    };
    Some(DeferredStore {
        stage: Stage::Compare(ticket),
        identity: identity.clone(),
        mapping_id,
        width,
        height,
        base_off,
        gen_now,
        started,
    })
}

/// TEMP (lab PAY_DEFER): take a [`DeferredStore`] as far as it goes without
/// waiting — the compare's answer read, vouched and handed to the landing
/// thread once the GPU is done; the bookkeeping once the copy is in. `Some` is
/// [`store_render_frame`]'s answer when the payment finished.
pub fn advance_render_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    d: &mut DeferredStore,
) -> Option<bool> {
    step_render_frame(state, host, d, false)
}

/// TEMP (lab PAY_DEFER): finish a [`DeferredStore`], waiting for what it still
/// waits on.
pub fn land_render_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mut d: DeferredStore,
) -> bool {
    step_render_frame(state, host, &mut d, true).unwrap_or(false)
}

fn step_render_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    d: &mut DeferredStore,
    block: bool,
) -> Option<bool> {
    loop {
        match std::mem::replace(&mut d.stage, Stage::Taken) {
            Stage::Compare(ticket) => {
                if !block && !crate::backend::vulkan::engine::target_compare_done(&ticket) {
                    d.stage = Stage::Compare(ticket);
                    return None;
                }
                match begin_landing(state, host, d, ticket) {
                    Ok(paid) => return Some(paid),
                    Err(job) => {
                        d.stage = Stage::Copying(job);
                        if !block {
                            return None;
                        }
                    }
                }
            }
            Stage::Copying(job) => {
                if !block && !job.done() {
                    d.stage = Stage::Copying(job);
                    return None;
                }
                job.wait();
                return Some(complete_landing(state, host, d, job));
            }
            Stage::Taken => return Some(false),
        }
    }
}

/// The compare is done: read its answer and either hand the copy to the landing
/// thread (`Err`, the payment goes on) or finish the payment here (`Ok`: the
/// synchronous landing, or the whole-frame fallbacks of the synchronous payment).
fn begin_landing<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    d: &DeferredStore,
    ticket: crate::backend::vulkan::engine::CompareTicket,
) -> Result<bool, LandJob> {
    let changed = match crate::backend::vulkan::engine::finish_target_compare(&ticket) {
        Ok(changed) => changed,
        Err(e) => {
            crate::observe::Emit::decline("pay_diff_failed", &e)
                .field("mapping", d.mapping_id)
                .fail_once(u64::from(d.mapping_id));
            None
        }
    };
    drop(ticket);
    let Some(changed) = changed else {
        return Ok(whole_frame(state, host, d));
    };
    let Some(vouched) = crate::runtime::mapper::vouch_mapping_pages_verdict(state, host, d.mapping_id).1
    else {
        crate::runtime::drain::note_store_route("lab_pay_defer_vouch_refused");
        return Ok(whole_frame(state, host, d));
    };
    if let Some(job) = start_landing_copy(state, host, d, &changed, &vouched) {
        return Err(job);
    }
    match land_changed_pages(
        state, host, d.mapping_id, &d.identity, d.width, d.height, d.base_off, d.gen_now, d.started,
        changed, vouched,
    ) {
        Some(done) => {
            crate::runtime::drain::note_store_route("lab_pay_defer_landed");
            Ok(done)
        }
        None => Ok(whole_frame(state, host, d)),
    }
}

/// The synchronous payment's last resort: forget the shadow and land the frame
/// the resident holds now, whole.
fn whole_frame<M: HostMemory + HostOps>(state: &mut DeviceState, host: &mut M, d: &DeferredStore) -> bool {
    crate::runtime::drain::note_store_route("lab_pay_defer_whole");
    crate::backend::vulkan::engine::page_diff_forget(d.mapping_id);
    store_render_frame(state, host, d.mapping_id, &d.identity, d.width, d.height)
}

/// [`land_changed_pages`]' write — `write_mapping_bytes_only`'s prologue, then
/// the selected runs through the mapping's packed guest view — with the copy on
/// the landing thread. `None` when there is no packed view covering the frame
/// (or nothing to copy): the caller lands synchronously.
fn start_landing_copy<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    d: &DeferredStore,
    changed: &crate::backend::vulkan::engine::ChangedPages,
    vouched: &crate::runtime::mapper::PagesVouched,
) -> Option<LandJob> {
    let page = 4096u64;
    let mapping_id = d.mapping_id;
    let base_off = d.base_off;
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &p in &changed.pages {
        let lo = u64::from(p) * page;
        let hi = (lo + page).min(changed.frame_bytes);
        match runs.last_mut() {
            Some(last) if last.1 == base_off + lo => last.1 = base_off + hi,
            _ => runs.push((base_off + lo, base_off + hi)),
        }
    }
    if runs.is_empty() {
        return None;
    }
    let need_end = base_off + changed.frame_bytes;
    let src_base = changed.frame_ptr as usize;
    // `write_mapping_bytes_only`'s prologue, in its order.
    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingBytesWrite,
    );
    state.invalidate_storage_residency_window(mapping_id, base_off, need_end);
    if !vouched.covers(state, mapping_id) {
        return None; // the synchronous landing reports it
    }
    let mut mapped: Vec<(usize, usize)> = Vec::new();
    let view = crate::runtime::mapper::ensure_contig_view(state, host, mapping_id)
        .filter(|&(_, len)| len as u64 >= need_end);
    let spans: LandSpans = match view {
        // One packed view covers the frame: `copy_mapping_runs`' fast path.
        Some((ptr, _)) => runs
            .iter()
            .map(|&(lo, hi)| (ptr + lo as usize, src_base + (lo - base_off) as usize, (hi - lo) as usize))
            .collect(),
        // Scattered pages: map only the page runs the selection touches, as
        // `par_land_write` does, and keep them mapped until the copy is in.
        None => {
            if !host.map_pages_stable() {
                return None;
            }
            let gpas = crate::runtime::mapper::mapping_page_gpas(state, host, mapping_id)?;
            let page_size = state.page_size();
            if need_end > (gpas.len() as u64).saturating_mul(page_size) {
                return None;
            }
            crate::runtime::mapper::flush_retired_views(state, host);
            let page_runs = reims_vgpu_paging::runs::contig_page_runs(&gpas, page_size);
            let mut spans: LandSpans = Vec::new();
            let mut first = 0usize; // the first selected run that may still meet a page run
            for pr in &page_runs {
                let run_mlo = (pr.start as u64).saturating_mul(page_size);
                let run_mhi = (pr.end as u64).saturating_mul(page_size);
                while first < runs.len() && runs[first].1 <= run_mlo {
                    first += 1;
                }
                let mut ptr = None;
                let mut i = first;
                while i < runs.len() && runs[i].0 < run_mhi {
                    let (a, b) = (runs[i].0.max(run_mlo), runs[i].1.min(run_mhi));
                    if a < b {
                        let base = match ptr {
                            Some(base) => base,
                            None => {
                                let Some(base) = host.map_pages(&gpas[pr.clone()], page_size as usize)
                                else {
                                    for &(p, len) in &mapped {
                                        host.unmap_pages(p, len);
                                    }
                                    return None;
                                };
                                mapped.push((base, (pr.end - pr.start) * page_size as usize));
                                ptr = Some(base);
                                base
                            }
                        };
                        spans.push((
                            base + (a - run_mlo) as usize,
                            src_base + (a - base_off) as usize,
                            (b - a) as usize,
                        ));
                    }
                    i += 1;
                }
            }
            crate::runtime::drain::note_store_route("lab_pay_defer_copy_runs");
            spans
        }
    };
    // `copy_mapping_runs`' host-write note, before any byte moves.
    state.note_host_wrote_mapping_selected(mapping_id, &runs, base_off, need_end);
    // A draw the recorder still holds may read these pages.
    crate::backend::settle_deferred_draws_for_guest_write();
    crate::runtime::drain::note_store_route("lab_pay_defer_copy_async");
    Some(LandJob::spawn(
        spans,
        runs,
        mapped,
        changed.frame_bytes as usize,
        changed.pages.len(),
        changed.full,
    ))
}

/// The landing copy is in: [`land_changed_pages`]' bookkeeping, and the views it
/// mapped unmapped.
fn complete_landing<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    d: &DeferredStore,
    mut job: LandJob,
) -> bool {
    let mapping_id = d.mapping_id;
    for (p, len) in std::mem::take(&mut job.mapped) {
        host.unmap_pages(p, len);
    }
    for &(lo, hi) in &job.runs {
        crate::runtime::mapper::note_mapping_write_footprint(state, mapping_id, lo, hi - lo);
    }
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        m.guest_bytes_seq = m.guest_bytes_seq.wrapping_add(1);
    }
    crate::runtime::drain::note_store_route(if job.full { "pay_diff_full" } else { "pay_diff_pages" });
    crate::runtime::drain::note_store_route_n("pay_diff_pages_landed", job.pages as u64);
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        m.pay_diff_gen = d.gen_now.unwrap_or(0);
    }
    state.invalidate_storage_residency_window(
        mapping_id,
        d.base_off,
        d.base_off + job.frame_len as u64,
    );
    let _ = state.mark_mapping_written(mapping_id);
    if let Some(m) = state.mappings.get(&mapping_id) {
        crate::backend::vulkan::engine::page_diff_set_key(
            mapping_id,
            (m.map_generation, d.width, d.height, m.guest_bytes_seq),
        );
    }
    finish(state, mapping_id, &d.identity, job.frame_len, d.started, false);
    crate::runtime::drain::note_store_route("lab_pay_defer_landed");
    drop(job);
    true
}

/// TEMP (lab): `REIMS_VGPU_LAB_PAY_SPLIT=off` vouches the mapping's pages before
/// the compare is submitted, as before. On by default: the vouch is a page-table
/// walk that only the landing depends on, so it runs between the compare's
/// submit and its fence wait (`engine::read_target_changed_pages_with`). The
/// ordering that matters is unchanged — the generation and the forced pages are
/// still read before the submit, and nothing lands without a vouch. Latched.
fn lab_pay_split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("REIMS_VGPU_LAB_PAY_SPLIT").map_or(true, |v| v != "off");
        crate::observe::off(format!("pay_split on={on}"));
        on
    })
}

/// What a page-diff payment may do with its answered compare, given what the
/// vouch said (`None`: it never ran).
#[derive(Debug, PartialEq, Eq)]
enum VouchAfterSubmit<V> {
    /// Land, under this vouch.
    Vouched(V),
    /// The pages drifted: land nothing from the compare, and forget the
    /// shadow, which already stands for a frame the guest will not hold.
    Refused,
    /// The vouch has still to be taken.
    NotRun,
}

fn vouch_after_submit<V>(vouched: Option<Option<V>>) -> VouchAfterSubmit<V> {
    match vouched {
        Some(Some(v)) => VouchAfterSubmit::Vouched(v),
        Some(None) => VouchAfterSubmit::Refused,
        None => VouchAfterSubmit::NotRun,
    }
}

/// TEMP (lab): `REIMS_VGPU_LAB_PAY_VERIFY=1` — after a page-diff landing, read the
/// whole resident back and the mapping's guest pages, and count the 4 KiB pages
/// that differ (`pay_verify_pages` / `pay_verify_bad_pages`). The guest CPU can
/// write the mapping between the landing and the read, so the count is compared
/// between arms, not against zero.
fn lab_pay_verify<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    base_off: u64,
    frame_bytes: u64,
) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("REIMS_VGPU_LAB_PAY_VERIFY").is_ok_and(|v| v == "1")) {
        return;
    }
    let Ok(rb) = crate::backend::vulkan::engine::read_target(identity) else {
        return;
    };
    let Some(bgra) = rb.into_bgra8() else {
        return;
    };
    let len = (frame_bytes as usize).min(bgra.len());
    let mut guest = vec![0u8; len];
    if !crate::runtime::mapper::read_guest_owned_bytes(state, host, mapping_id, base_off, &mut guest)
    {
        crate::runtime::drain::note_store_route("pay_verify_unreadable");
        return;
    }
    let mut bad = 0u64;
    let mut pages = 0u64;
    for (g, r) in guest.chunks(4096).zip(bgra[..len].chunks(4096)) {
        pages += 1;
        if g != r {
            bad += 1;
        }
    }
    crate::runtime::drain::note_store_route_n("pay_verify_pages", pages);
    crate::runtime::drain::note_store_route_n("pay_verify_bad_pages", bad);
    if bad > 0 {
        crate::runtime::drain::note_store_route("pay_verify_bad_frames");
    }
}

/// The 4 KiB frame pages (the diff's unit, counted from `base_off`) that a set of
/// guest-written mapping-offset ranges touches, ascending. A guest page that
/// straddles two frame pages forces both.
fn forced_frame_pages(ranges: &[(u64, u64)], base_off: u64, frame_bytes: u64) -> Vec<u32> {
    const PAGE: u64 = 4096;
    let end = base_off + frame_bytes;
    let mut out: Vec<u32> = Vec::new();
    for &(s, e) in ranges {
        if e <= base_off || s >= end {
            continue;
        }
        let lo = (s.max(base_off) - base_off) / PAGE;
        let hi = (e.min(end) - base_off).div_ceil(PAGE);
        for p in lo..hi {
            if out.last() != Some(&(p as u32)) {
                out.push(p as u32);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn finish_needs_registry_handoff(guest_backed: bool) -> bool {
    !guest_backed
}

/// Hand the currency witness back to the image the frame came out of, and score
/// the write.
///
/// `write_bgra8_*` ends in `mark_mapping_written`, which advances the mapping's
/// `surface_content_epoch` — correctly, because its guest pages did change. But
/// the *pixels* did not: they are this resident's, copied out of it one
/// statement ago. Leaving the stamp behind invalidates a resident that holds
/// exactly the mapping's content, which costs the next Load its elision and
/// sends it to a CPU seed for bytes it already has.
fn finish(
    state: &mut DeviceState,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    frame_len: usize,
    started: std::time::Instant,
    guest_backed: bool,
) {
    // A copied resident needs two pieces of registry state handed back: the
    // mapping epoch says its mirror is current, and clearing sole-copy permits
    // reclaim now that the guest pages hold the pixels. An imported resident is
    // the guest allocation itself. It has no mirror epoch to stamp, and draw
    // completion already leaves it non-sole-copy, so neither registry mutation
    // applies.
    if finish_needs_registry_handoff(guest_backed) {
        if let Some(epoch) = state
            .mappings
            .get(&mapping_id)
            .map(|m| m.surface_content_epoch)
        {
            crate::backend::vulkan::engine::stamp_resident_content_epoch(identity, epoch);
        }
        crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    } else {
        crate::runtime::drain::note_store_route("shared_store_registry_handoff_elided");
    }
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Render),
        started,
    );
    crate::observe::verbose(|| format!(
        "render_store mapping={mapping_id} bytes={frame_len} us={}",
        started.elapsed().as_micros()
    ));
}

#[cfg(test)]
mod lab_vouch_after_submit_tests {
    use super::{vouch_after_submit, VouchAfterSubmit};

    #[test]
    fn a_drifted_vouch_lands_nothing_and_an_unrun_one_is_taken_late() {
        assert_eq!(
            vouch_after_submit(Some(Some(7u32))),
            VouchAfterSubmit::Vouched(7)
        );
        assert_eq!(
            vouch_after_submit::<u32>(Some(None)),
            VouchAfterSubmit::Refused
        );
        assert_eq!(vouch_after_submit::<u32>(None), VouchAfterSubmit::NotRun);
    }
}

#[cfg(test)]
mod guest_backed_finish_tests {
    use super::finish_needs_registry_handoff;

    #[test]
    fn shared_storage_has_no_mirror_or_sole_copy_handoff() {
        assert!(!finish_needs_registry_handoff(true));
        assert!(
            finish_needs_registry_handoff(false),
            "a copied resident still needs its epoch and reclaimability published"
        );
    }
}

/// Why a GVA render Store could not hand its resident straight to the guest's
/// pages, so it fell back to reading the frame back and converting it row by
/// row.
///
/// Every one of these is a routing answer and not a loss — the copying rail
/// still lands the frame — but each costs a blocking GPU→host readback of a
/// whole framebuffer plus a host pass over it, which is the largest single cost
/// this device pays. They are named individually so a boot says which check is
/// holding the volume.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GvaWritebackDecline {
    /// The guest declared a destination format that no host texel reproduces
    /// verbatim, so no image→buffer copy can produce it and `convert_rgba8_to_row`
    /// is the only route.
    ///
    /// This used to say "not four bytes of colour", and to name `RGBA16_FLOAT`
    /// as landing here *always*. Both went stale together: a resident now
    /// carries the format the guest declared rather than always being eight bits
    /// per channel, so a half-float destination can be the same bytes as the
    /// image. The rule was never a width — it is whether the destination's texel
    /// and the resident's are one layout, which is
    /// [`crate::backend::vulkan::translate::pixel::verbatim_texel`]'s question
    /// and not this doc's to restate.
    ///
    /// `R16_FLOAT` is what lands here now, twice on a driven macos-26 boot: it
    /// is renderable but deliberately not a byte-copy destination, for the
    /// reason `store_texel_order`'s own doc gives for `RG16_FLOAT`, and no
    /// compute selector names it either.
    FormatNeedsConversion { format: u16 },
    /// The resident's format is not the format the destination stores, so a
    /// byte copy would land the wrong texel. Distinct from the engine's own
    /// check of the same pair: this one is asked before the walk, so a mismatch
    /// costs no page-table work.
    ///
    /// Whole formats and not channel orders. The two differ once a render
    /// target may be wider than eight bits per channel: `RGBA16_FLOAT` and
    /// `RGBA8_UNORM` share an order and are four bytes per texel apart, and this
    /// is the arm that catches a half-float destination whose resident fell back
    /// to eight bits because the host cannot render to the wider format.
    ResidentFormatMismatch {
        held: ash::vk::Format,
        want: ash::vk::Format,
    },
    /// The guest's row pitch is not a whole number of texels, or is narrower
    /// than the frame, so there is no `bufferRowLength` that describes it.
    PitchNotTexels { row_stride: u32 },
    /// The frame's first texel does not start on a texel boundary within its
    /// page. `VkBufferImageCopy::bufferOffset` must be a multiple of the texel
    /// block size, and a copy that ignored this is undefined rather than
    /// misaligned.
    OffsetNotTexelAligned { in_page: u64 },
    /// The command resolved no destination pages before it was submitted, so
    /// there is nothing this rail is authorised to write. The copying rail
    /// treats the same answer as "unbounded" and writes anyway through its own
    /// re-walk; this rail has no second walk to fail closed on, so it declines.
    Unlicensed,
    /// The pre-submit walk did not resolve every page of the destination span,
    /// so its page list cannot be read positionally.
    /// [`crate::runtime::draw::StoreTargetPages::ordered_complete`] states what
    /// a short list would land.
    SpanIncomplete,
    /// The destination span did not become a guest-RAM reference; the inner
    /// refusal names the check, restated here for the reason
    /// `GpuWritebackDecline::GuestRefRefused` restates its own.
    GuestRefRefused {
        refusal: crate::runtime::guest_ram_map::MapRefusal,
    },
    /// The engine declined or the copy failed; the inner error names which.
    Engine {
        inner: crate::backend::vulkan::engine::DrawError,
    },
    /// The **copying** arm could not read the resident back, so neither arm
    /// landed the frame. Unlike every variant above this one, it is a loss: the
    /// direct arm has already declined by the time it can be produced, and there
    /// is no third rail.
    CopiedReadRefused {
        inner: crate::backend::vulkan::engine::DrawError,
    },
    /// The **copying** arm read the frame and could not write it into the guest
    /// pages its licence names. A loss, for the same reason as above.
    CopiedWriteRefused { err: crate::runtime::host::MemError },
}

impl crate::observe::Decline for GvaWritebackDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::FormatNeedsConversion { .. } => "gvawb_format_needs_conversion",
            Self::ResidentFormatMismatch { .. } => "gvawb_resident_format_mismatch",
            Self::PitchNotTexels { .. } => "gvawb_pitch_not_texels",
            Self::OffsetNotTexelAligned { .. } => "gvawb_offset_not_texel_aligned",
            Self::Unlicensed => "gvawb_unlicensed",
            Self::SpanIncomplete => "gvawb_span_incomplete",
            Self::GuestRefRefused { .. } => "gvawb_guest_ref_refused",
            Self::Engine { inner } => crate::observe::Decline::slug(inner),
            Self::CopiedReadRefused { .. } => "gvawb_copied_read_refused",
            Self::CopiedWriteRefused { .. } => "gvawb_copied_write_refused",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unlicensed | Self::SpanIncomplete => Vec::new(),
            Self::FormatNeedsConversion { format } => vec![("fmt", format!("{format:#x}"))],
            Self::ResidentFormatMismatch { held, want } => vec![
                ("resident", format!("{held:?}")),
                ("want", format!("{want:?}")),
            ],
            Self::PitchNotTexels { row_stride } => vec![("bpr", row_stride.to_string())],
            Self::OffsetNotTexelAligned { in_page } => vec![("in_page", in_page.to_string())],
            Self::GuestRefRefused { refusal } => {
                let mut f = vec![("via", crate::observe::Decline::slug(refusal).to_string())];
                f.extend(crate::observe::Decline::fields(refusal));
                f
            }
            Self::Engine { inner } => crate::observe::Decline::fields(inner),
            Self::CopiedReadRefused { inner } => {
                let mut f = vec![("via", crate::observe::Decline::slug(inner).to_string())];
                f.extend(crate::observe::Decline::fields(inner));
                f
            }
            Self::CopiedWriteRefused { err } => vec![("err", format!("{err:?}"))],
        }
    }
}

crate::observe::decline::decline_display!(GvaWritebackDecline);

/// Land a GVA render Store's frame in the guest's pages, GPU-direct where the
/// host allows it and by reading the resident back where it does not.
///
/// # The ladder, and why it is here rather than at one call site
///
/// The GPU-direct arm below needs a guest-RAM reference over the destination
/// pages, which is `VK_EXT_external_memory_host` or nothing. On a host without
/// that extension — and on every discrete GPU, where `linear_target_import`
/// refuses `UnsupportedTopology` — it declines every Store of every GVA target
/// with `gvawb_guest_ref_refused via=guest_ram_map_no_backend_import`.
///
/// That decline used to be a **lost frame** for one of the two callers. The
/// eager site in `draw::vulkan` carries its own copying rail (`gva_store_sync`:
/// read the resident back, then `write_gva_rgba8_within`), so it never sees
/// one; the deferred site — [`crate::runtime::writeback_debt::pay_for_texture`],
/// which pays a resource's debt at the moment the guest samples it — had
/// nothing behind the decline and reported `gvadebt_pay_lost`. A driven
/// macos-13 boot under `REIMS_VGPU_GUEST_IMPORT=off` read **27
/// `gvadebt_paid_named` and 27 `gvadebt_pay_lost`**: every deferred payment on
/// that host class lost its frame, and because the failure also released the
/// debt, the sampled rail then read guest pages the frame had never reached.
/// Thirteen of the twenty-seven were one texture of one task re-rendered over
/// and over, which on screen is a window whose content stays blank while its
/// chrome draws.
///
/// So the copying arm lives here, in the function that owns "land this GVA
/// Store", and both callers get it. A decline of the direct arm is a *cost* —
/// a blocking readback plus a host pass over the frame — and never a loss.
///
/// # What the licence is
///
/// A mapping carries its own page list and a page-table vouch licenses it; a
/// GVA carries neither, so the licence is the exact page list supplied in
/// `pages`, which **neither** arm may widen. The eager fallback captures it
/// before draw submission; deferred payment gets it from the live resource's
/// transfer backing. `pages == None` is therefore still `Unlicensed` on both
/// arms: there is no authorisation to write anywhere, and a copy is not a
/// second opinion about that.
/// # What `skip` is
///
/// Bytes from `c0.target_gva` this store may not write, because the guest's own
/// memory holds something newer there. Every eager caller passes `&[]`: a Store
/// landing at the moment it is issued has nothing to make room for. The
/// deferred caller can have something, and the empty slice is the wrong answer
/// for it — see `writeback_debt::pay_gva`.
///
/// A non-empty `skip` forces the copying rail. The GPU-direct arm copies the
/// resident into the plane as one image region and has no way to exclude pages
/// from it, so a caller that must exclude some pays the readback. That is the
/// right price: the alternative is losing one of the two writers, and it is
/// paid only when the hypervisor has actually reported a guest write into a
/// plane this device still owes a frame for.
#[allow(
    clippy::too_many_arguments,
    reason = "the store's own parameters, plus the bytes its owner may not overwrite"
)]
pub(crate) fn store_gva_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    c0: &crate::runtime::draw::ColorRtRequest,
    texture_ref: u32,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
    skip: crate::runtime::mapping_write::SkipRanges<'_>,
) -> Result<u64, GvaWritebackDecline> {
    let direct = if skip.is_empty() {
        store_gva_frame_direct(state, host, task_id, identity, c0, texture_ref, pages)
    } else {
        crate::runtime::drain::note_store_route("gva_flush_skipping");
        Err(GvaWritebackDecline::Unlicensed)
    };
    let decline = match direct {
        Ok(extent) => {
            crate::runtime::drain::note_store_route("gva_flush_gpu_direct");
            return Ok(extent);
        }
        Err(decline) => decline,
    };
    // Latched per target GVA as well as per reason, for the reason
    // `store_render_frame`'s twin latch gives: a host without the import
    // declines every Store of every target, and a line each would drown the
    // channel.
    crate::observe::Emit::decline("gva_flush_gpu_declined", &decline)
        .field("gva", format!("{:#x}", c0.target_gva))
        .field("geom", format!("{}x{}", c0.width, c0.height))
        // The destination's declared format and the resident's own. The copying
        // arm below converts by `into_rgba8` and then
        // `convert_rgba8_to_row(c0.format)`, and whether those two cancel is
        // decided entirely by this pair — so without it a channel-order defect
        // on this rail is a screenshot and no reading.
        .field("fmt", format!("{:#x}", c0.format))
        .field("resident", format!("{:?}", identity.resident_format()))
        .fail_once(c0.target_gva);
    crate::runtime::drain::note_store_route("gva_flush_gpu_declined");
    let Some(pages) = pages else {
        return Err(GvaWritebackDecline::Unlicensed);
    };
    // The blocking readback the direct arm exists to avoid. `into_rgba8` is the
    // order every GVA guest writer takes, and it exchanges or not according to
    // the order the engine reports for the image it copied — which for a target
    // the guest declared in BGRA order, as most of them are, is a whole-frame
    // pass and not the no-op this comment used to claim. Both spellings of that
    // declaration must reach the same answer; `ResidentReadSnapshot::bgra` is
    // where they do, and where they did not.
    let readback = crate::backend::vulkan::engine::read_target(identity)
        .map_err(|inner| GvaWritebackDecline::CopiedReadRefused { inner })?;
    // Two ways to land a frame, and which one is decided by what the readback
    // actually holds rather than by what this rail wishes it held.
    //
    // A native frame is the resident's own texel, carried out because no
    // eight-bit narrowing of it exists. It lands verbatim **only** when the
    // destination is that same layout — `store_texel_order`'s question, the one
    // the GPU-direct arm has always asked. That is what lets a destination with
    // no eight-bit form take the copying rail at all, so a host without the
    // guest-RAM import serves it instead of losing every frame of it.
    let extent = match readback.texel.native_layout() {
        Some(layout) => {
            if crate::protocol::pixel_format::store_texel_order(c0.format) != Some(layout) {
                return Err(GvaWritebackDecline::FormatNeedsConversion { format: c0.format });
            }
            crate::runtime::drain::note_store_route("gva_flush_copied_native");
            land_gva_frame_bytes(
                state,
                host,
                task_id,
                c0,
                texture_ref,
                crate::runtime::draw::FrameRows::Native(&readback.pixels),
                pages,
                skip,
            )?
        }
        None => {
            let Some(rgba) = readback.into_rgba8() else {
                return Err(GvaWritebackDecline::FormatNeedsConversion { format: c0.format });
            };
            land_gva_frame_bytes(
                state,
                host,
                task_id,
                c0,
                texture_ref,
                crate::runtime::draw::FrameRows::Rgba8(&rgba),
                pages,
                skip,
            )?
        }
    };
    crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    // Arm the write witness exactly as the direct arm does: the pages now hold
    // this resident's frame, texel for texel on the native arm and byte for byte
    // on the eight-bit one. Without it a later LOAD or sample never reads quiet
    // and re-reads the pages through the eight-bit seed rail — which for a float
    // target is not "a re-read" but a clamped, quantised copy of it. Only for a
    // whole-frame landing: a store that skipped ranges left guest bytes in the
    // plane that the resident does not hold.
    if skip.is_empty() {
        if let (Some(key), Some(gpas)) = (
            crate::backend::vulkan::gva_witness_key(identity),
            pages.ordered_complete(c0.target_gva, state.page_size()),
        ) {
            let gpas = gpas.to_vec();
            crate::runtime::gva_store_witness::note_store_eager(state, host, key, &gpas);
        }
    }
    crate::runtime::drain::note_store_route("gva_flush_copied");
    Ok(extent)
}

/// The copying arm's body: convert `rgba` into the destination's own texel
/// order and write it into the guest pages `pages` licenses.
///
/// Split out from [`store_gva_frame`] at the readback boundary so it can be
/// tested against real guest memory without a device — the readback is the only
/// part of the arm that needs one.
///
/// It forgets both GVA pixel caches on the way out, exactly as the direct arm
/// does and for the same reason: after this call the guest's own pages hold the
/// frame, and a stale entry left behind would serve a later sample the previous
/// Store's bytes. `write_gva_rgba8_within` records the host write itself, from
/// inside `gva_view`, so the two arms leave the same witness behind and a
/// decline is invisible to every reader of `gather_witness`.
#[allow(
    clippy::too_many_arguments,
    reason = "the destination's geometry, plus the bytes its owner may not overwrite"
)]
fn land_gva_frame_bytes<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    c0: &crate::runtime::draw::ColorRtRequest,
    texture_ref: u32,
    frame: crate::runtime::draw::FrameRows<'_>,
    pages: &crate::runtime::draw::StoreTargetPages,
    skip: crate::runtime::mapping_write::SkipRanges<'_>,
) -> Result<u64, GvaWritebackDecline> {
    // The destination extent in the destination's own bytes, which is what the
    // direct arm returns too — the two rails must agree about how much guest
    // memory a Store of this geometry lands, or a caller could tell them apart.
    let Some(tight) = crate::protocol::pixel_format::tight_row_bytes(c0.width, c0.format) else {
        return Err(GvaWritebackDecline::FormatNeedsConversion { format: c0.format });
    };
    let extent =
        u64::from(c0.height.saturating_sub(1)) * u64::from(c0.row_stride) + u64::from(tight);
    crate::runtime::draw::write_gva_frame_within_skipping(
        state,
        host,
        task_id,
        c0.target_gva,
        c0.width,
        c0.height,
        c0.row_stride,
        c0.format,
        frame,
        Some(pages.membership()),
        skip,
    )
    .map_err(|err| GvaWritebackDecline::CopiedWriteRefused { err })?;
    crate::runtime::surface_cache::forget_gva_copies(state, task_id, c0.target_gva, texture_ref);
    Ok(extent)
}

/// Copy `identity`'s pixels into the guest pages behind a normal-texture render
/// target's `target_gva`, with no host copy of the frame at any point.
///
/// The GVA twin of [`store_render_frame`]'s first arm, and worth diffing
/// against it: both end in `copy_target_to_guest_pages` and they differ only in
/// how the destination pages are named.
///
/// # Why this is the whole cost of a GVA synchronization
///
/// The arm behind this one reads the resident back to the host (a blocking
/// fence) and then writes it out again a row at a time through
/// `convert_rgba8_to_row`. On a driven desktop-compositing boot that is 59 % of
/// render Stores and most of the time the device spends waiting on a fence.
/// Everything this call declines on pays that instead.
///
/// # What it does not do
///
/// It publishes no host-side copy of the frame. The two GVA pixel caches are
/// *dropped* instead, for the same reason `store_render_frame` forgets the
/// mapping's: after this call the guest's own pages are the only place the
/// frame exists, and an entry left behind would serve a later sample the
/// previous Store's bytes. The sampled rail re-reads them from guest memory,
/// which is what `store_gva_owned`'s `guest_holds_bytes` already promised.
fn store_gva_frame_direct<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    c0: &crate::runtime::draw::ColorRtRequest,
    texture_ref: u32,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
) -> Result<u64, GvaWritebackDecline> {
    copy_resident_into_gva_plane(
        state,
        host,
        task_id,
        identity,
        &GvaPlaneDestination {
            target_gva: c0.target_gva,
            width: c0.width,
            height: c0.height,
            row_stride: c0.row_stride,
            format: c0.format,
            texture_ref,
        },
        pages,
    )
}

/// A guest-linear plane a resident's pixels may be copied into.
///
/// The five terms [`copy_resident_into_gva_plane`] needs and nothing else. It
/// exists because that rail used to take a [`crate::runtime::draw::ColorRtRequest`],
/// which describes a *render attachment* — a slot, load and store actions, a
/// clear colour, a multisample source — and a second caller with the same
/// destination and no render pass behind it could not honestly fill one in.
///
/// `texture_ref` is the resource whose host-side pixel caches this copy
/// invalidates, and is `0` where the caller has none to name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GvaPlaneDestination {
    pub target_gva: u64,
    pub width: u32,
    pub height: u32,
    pub row_stride: u32,
    /// The guest's own declaration for these bytes, which is what it will read
    /// them back as. A copy converts nothing, so this is the format the
    /// resident must already hold.
    pub format: u16,
    pub texture_ref: u32,
}

/// What a [`GvaPlaneDestination`]'s own terms imply about the bytes it names.
///
/// `extent` is the reason this is a type rather than three lines inside the copy
/// below. A caller has to walk the guest's page table before it can hand over a
/// licence, and the span it walks must be the span the copy writes — a walk one
/// row short resolves fewer pages than `ordered_complete` demands and the copy
/// declines, while a walk that is longer authorises pages the frame never
/// reaches. Deriving it in one pure place is the only way the two cannot drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GvaPlaneGeometry {
    /// The Vulkan format the resident must already hold.
    pub want: ash::vk::Format,
    pub bpt: u64,
    pub row_stride: u64,
    /// The bytes from `target_gva` this plane occupies — the extent the copy
    /// names and nothing past it. Padding after the final row belongs to the
    /// allocation but is not texels this destination was given, and the copying
    /// rail leaves it alone too. The two rails must land identical guest memory
    /// or a fallback would be visible.
    pub extent: u64,
}

impl GvaPlaneDestination {
    /// The geometry, or the typed reason this destination has none.
    ///
    /// Pure, so a caller may ask for the span it must walk and
    /// [`copy_resident_into_gva_plane`] may ask again for the copy it issues,
    /// without either restating the rule.
    pub(crate) fn geometry(&self) -> Result<GvaPlaneGeometry, GvaWritebackDecline> {
        // The destination's texel, and the whole reason this rail can exist at
        // all: a copy converts nothing, so the guest must already read these
        // bytes exactly as the resident holds them.
        //
        // Asked of every rail that creates images, not of the render Store's
        // table alone — this destination serves a compute storage output as
        // readily as a Store, and a storage image is a thing the guest neither
        // renders into nor samples. See
        // [`crate::backend::vulkan::translate::pixel::verbatim_texel`].
        let Some((want, bpt)) =
            crate::backend::vulkan::translate::pixel::verbatim_texel(self.format)
        else {
            return Err(GvaWritebackDecline::FormatNeedsConversion {
                format: self.format,
            });
        };
        let bpt = u64::from(bpt);
        let row_stride = u64::from(self.row_stride);
        if row_stride == 0
            || !row_stride.is_multiple_of(bpt)
            || row_stride < u64::from(self.width) * bpt
        {
            return Err(GvaWritebackDecline::PitchNotTexels {
                row_stride: self.row_stride,
            });
        }
        Ok(GvaPlaneGeometry {
            want,
            bpt,
            row_stride,
            extent: u64::from(self.height.saturating_sub(1)) * row_stride
                + u64::from(self.width) * bpt,
        })
    }
}

/// Copy `identity`'s pixels into the guest pages behind a linear destination,
/// with no host copy of the frame at any point.
///
/// The body of [`store_gva_frame_direct`], reachable by any caller holding a
/// resident and a licensed guest-linear plane. A render Store is one such
/// caller; a `copyFromTexture:toTexture:` whose source is an IOSurface the GPU
/// already holds and whose destination is a linear guest allocation is another,
/// and it is the same copy — the two differ only in where the geometry and the
/// page licence came from.
/// A licensed direct-to-guest-pages destination: where the bytes go, which
/// pages the licence covers, and how many bytes land.
///
/// The two rails that write guest pages from the GPU — a render Store and a
/// compute storage-image output — differ only in the image they copy *from* and
/// in which command buffer records the copy. Everything about the destination is
/// this, so it is derived once and the second rail cannot spell any of it
/// differently.
pub(crate) struct GvaPlaneLicence {
    pub target: crate::backend::vulkan::engine::GuestPageTarget,
    /// The pages walked, in guest-virtual order — what the copy is licensed
    /// over and what every witness on this path is armed against.
    pub gpas: Vec<u64>,
    /// The bytes from `target_gva` the copy will land. The caller's return
    /// value, and what the counters on this path are charged.
    pub extent: u64,
}

/// Resolve a guest-linear plane into a destination the GPU may copy into, or
/// the typed reason it may not.
///
/// `held` is the format the source image actually holds. **A copy converts
/// nothing** — neither `vkCmdCopyImageToBuffer` nor the blit behind the
/// rectangle plan performs a channel swap or a texel resize — so a source whose
/// format is not the one the guest will read these bytes back as must be refused
/// here rather than landed. That is the whole of the format contract on this
/// rail, and it is stated once so neither caller can state it differently.
///
/// The comparison is between whole formats, not channel orders. While every
/// resident was eight bits per channel an order was a complete description of
/// one, so `is_bgra() != (order == Bgra8)` decided this. It is not any more:
/// `RGBA16_FLOAT` and `RGBA8_UNORM` are both RGBA-ordered and are four bytes per
/// texel apart, so an order comparison would admit a half-float destination over
/// an eight-bit source and copy a frame of the wrong size and the wrong texel
/// into the guest's pages.
pub(crate) fn licence_gva_plane<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    held: ash::vk::Format,
    c0: &GvaPlaneDestination,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
) -> Result<GvaPlaneLicence, GvaWritebackDecline> {
    let GvaPlaneGeometry {
        want,
        bpt,
        row_stride,
        extent,
    } = c0.geometry()?;
    // On the render rail this is a healthy zero as things stand:
    // `gva_chain_identity` builds the key from this same `c0.format`, so the two
    // agree by construction and this arm is the alarm for an identity that came
    // from somewhere else. It is also the arm that catches the honest
    // disagreement — a guest format whose resident fell back to eight bits
    // because the host cannot render to it — and sends that Store down the CPU
    // conversion rail where it belongs. On the compute rail it is not a healthy
    // zero: a storage image takes its format from the specialized SPIR-V texel
    // format, which can legitimately differ from the guest surface's, and this
    // is what stops those bytes being landed raw.
    if held != want {
        return Err(GvaWritebackDecline::ResidentFormatMismatch { held, want });
    }
    let Some(pages) = pages else {
        return Err(GvaWritebackDecline::Unlicensed);
    };
    let page_size = state.page_size();
    let Some(gpas) = pages.ordered_complete(c0.target_gva, page_size) else {
        return Err(GvaWritebackDecline::SpanIncomplete);
    };
    let in_page = c0.target_gva % page_size;
    if !in_page.is_multiple_of(bpt) {
        return Err(GvaWritebackDecline::OffsetNotTexelAligned { in_page });
    }
    // The GPU is about to write these pages: an operator who narrowed the
    // import to reads sends this Store down the copying rail, the same decline
    // a host without the import takes.
    if let Some(refusal) = crate::runtime::guest_ram_map::write_refusal() {
        return Err(GvaWritebackDecline::GuestRefRefused { refusal });
    }
    let runs =
        crate::runtime::guest_ram_map::references_for_runs(host, gpas, page_size, in_page, extent)
            .map_err(|refusal| GvaWritebackDecline::GuestRefRefused { refusal })?;
    let target = crate::backend::vulkan::engine::GuestPageTarget {
        runs,
        // Checked above to divide exactly, so this is the guest's pitch and not
        // a rounding of it.
        row_length_texels: (row_stride / bpt) as u32,
        width: c0.width,
        height: c0.height,
        format: want,
        shared_backing: None,
    };
    // This device is about to write these guest pages, and the hypervisor's
    // dirty bitmap is defined not to see it. Without this record a reader
    // holding a gathered image over the same pages
    // (`crate::runtime::gather_witness`) cannot tell "nobody wrote them" from
    // "we wrote them ourselves", and vouches a retained image that no longer
    // matches the pages — a wrong frame that then persists.
    //
    // The copying rail this stands in front of records the identical fact from
    // inside `gva_view`, so the two rails leave the same witness behind and a
    // decline is invisible to every reader. Before the submit and not after it,
    // and over the whole walked span rather than the copy's extent: a spurious
    // bump costs a re-read of bytes that did not change, and the opposite error
    // hands out a stale copy. It is armed here, past the last refusal, so a
    // caller that goes on to fail its submit has over-recorded rather than
    // under-recorded — the direction that costs a re-read instead of a wrong
    // frame.
    state.note_host_wrote_pages(gpas.to_vec());
    // The boot-wide write footprint, so a panic's page number can be looked up
    // against this rail too. The mapper-ref licence marks it; this one did not,
    // and a page this rail wrote read as "never written by this device".
    // Marked over the copy's extent within the walked pages, before the
    // submit — the over-recording direction, as the mapper's own note says.
    crate::runtime::mapper::note_page_write_footprint(page_size, in_page, extent, |i| {
        gpas.get(i).map(|&gpa| Some(gpa))
    });
    Ok(GvaPlaneLicence {
        target,
        gpas: gpas.to_vec(),
        extent,
    })
}

pub(crate) fn copy_resident_into_gva_plane<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    c0: &GvaPlaneDestination,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
) -> Result<u64, GvaWritebackDecline> {
    let licence = licence_gva_plane(state, host, identity.resident_format(), c0, pages)?;
    let GvaPlaneLicence {
        target,
        gpas,
        extent,
    } = &licence;
    crate::backend::vulkan::engine::copy_target_to_guest_pages(identity, target, gpas)
        .map_err(|inner| GvaWritebackDecline::Engine { inner })?;
    let extent = *extent;
    // Nothing here leaves a host copy of the frame, so neither GVA-keyed cache
    // may go on naming one.
    crate::runtime::surface_cache::forget_gva_copies(state, task_id, c0.target_gva, c0.texture_ref);
    // The copy means this image has stopped being the only place these pixels
    // exist, so the reclaim paths may take it — the same handover
    // `store_render_frame` performs in `finish`.
    crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    // Arm the GVA write witness over the pages this Store just published, the
    // twin of `mapper::stamp_guest_write_gen` on the mapper-ref-texture rail. It is what
    // lets a later reader ask whether these pages still hold this frame without
    // reading them — see `crate::runtime::gva_store_witness`.
    //
    // After the submit, not before it: a stamp taken ahead of a copy that then
    // declines would claim the guest's pages hold a frame that never reached
    // them. And after `note_host_wrote_pages` above, because the epoch the
    // witness records is compared against that same ring — capturing it first
    // would have every target permanently invalidated by its own Store.
    //
    // The copying arm of [`store_gva_frame`] stamps too, for a whole-frame
    // landing. It used not to, on the argument that a missing stamp costs a
    // re-read and never a wrong image; that holds for eight-bit colour only. The
    // re-read is the eight-bit seed rail, so for a float target it *is* a wrong
    // image — the clamped one. The eager eight-bit `gva_store_sync` still leaves
    // none, which is safe for the formats it now carries.
    if let Some(key) = crate::backend::vulkan::gva_witness_key(identity) {
        crate::runtime::gva_store_witness::note_store_eager(state, host, key, gpas);
    }
    Ok(extent)
}

#[cfg(test)]
mod gva_copying_arm_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::draw::{ColorRtRequest, StoreTargetPages};
    use crate::runtime::gva_mem::define_task_pages_arm64e;
    use crate::runtime::host::FakeHost;

    /// The first data page `define_task_pages_arm64e` installs, as a PFN. The
    /// task's GVA page `i` is this plus `i`, which is what lets a test name a
    /// destination page and the licence for it from one number.
    const DATA_BASE_PFN: u64 = 4;
    const PAGES: u32 = 8;
    /// GVA page 1, so the destination is neither the null address
    /// `write_gva_rows_within` rejects nor the first page of the walk.
    const TARGET_GVA: u64 = 1 << PAGE_SHIFT_ARM64E;
    const W: u32 = 4;
    const H: u32 = 2;
    const BPR: u32 = 16;

    fn fixture() -> (FakeHost, DeviceState) {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        define_task_pages_arm64e(&mut host, &mut state, DATA_BASE_PFN as u32, PAGES);
        (host, state)
    }

    fn request() -> ColorRtRequest {
        ColorRtRequest {
            texture_ref: 0,
            target_gva: TARGET_GVA,
            row_stride: BPR,
            width: W,
            height: H,
            // Byte-identical with the RGBA8 the readback hands over, so the
            // assertion below is about where the bytes landed and not about the
            // conversion, which has its own tests.
            format: crate::protocol::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            store_action: reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_STORE,
            ..Default::default()
        }
    }

    fn frame() -> Vec<u8> {
        (0..(W * H * 4) as u8).map(|b| b.wrapping_add(1)).collect()
    }

    /// The page the destination lives in, as a one-entry licence.
    fn licence_for(page_index: u64) -> StoreTargetPages {
        let span = u64::from(BPR) * u64::from(H);
        StoreTargetPages::from_ordered(&[(DATA_BASE_PFN + page_index) << PAGE_SHIFT_ARM64E], span)
    }

    /// The whole point of the arm: a GVA render Store whose GPU-direct rail
    /// cannot run still reaches the guest's pages.
    ///
    /// Before this arm existed the deferred payer had nothing behind the direct
    /// decline and reported `gvadebt_pay_lost`, so these bytes never left the
    /// resident on any host without `VK_EXT_external_memory_host`.
    #[test]
    fn the_copying_arm_lands_the_frame_in_the_guests_own_pages() {
        let (mut host, mut state) = fixture();
        let rgba = frame();
        let pages = licence_for(1);
        let extent = land_gva_frame_bytes(
            &mut state,
            &mut host,
            1,
            &request(),
            0,
            crate::runtime::draw::FrameRows::Rgba8(&rgba),
            &pages,
            &[],
        )
        .expect("the licensed page is writable");
        assert_eq!(
            extent,
            u64::from(H - 1) * u64::from(BPR) + u64::from(W) * 4,
            "the extent must be the destination's own bytes, as the direct arm reports"
        );
        let gpa = (DATA_BASE_PFN + 1) << PAGE_SHIFT_ARM64E;
        for y in 0..H as usize {
            let mut got = [0u8; (W * 4) as usize];
            crate::runtime::host::HostMemory::read_gpa(
                &host,
                gpa + (y as u64) * u64::from(BPR),
                &mut got,
            )
            .expect("the destination page is guest RAM");
            let at = y * (W * 4) as usize;
            assert_eq!(
                &got[..],
                &rgba[at..at + (W * 4) as usize],
                "row {y} of the frame did not reach the guest"
            );
        }
    }

    /// The copying arm may not widen the licence the direct arm was given.
    ///
    /// A GVA carries no page list of its own, so the supplied one is the entire
    /// authorisation. A second rail that re-walked and wrote wherever the
    /// address points now would be the stale-view class this device already
    /// bounds — so the arm refuses by name and the guest's memory is untouched.
    #[test]
    fn the_copying_arm_refuses_a_page_its_licence_does_not_name() {
        let (mut host, mut state) = fixture();
        let rgba = frame();
        // A licence naming some other page of the same task.
        let pages = licence_for(5);
        let refusal = land_gva_frame_bytes(
            &mut state,
            &mut host,
            1,
            &request(),
            0,
            crate::runtime::draw::FrameRows::Rgba8(&rgba),
            &pages,
            &[],
        )
        .expect_err("the destination page is outside the licence");
        assert_eq!(
            crate::observe::Decline::slug(&refusal),
            "gvawb_copied_write_refused",
            "an unlicensed write is a named refusal, not a silent skip"
        );
        let gpa = (DATA_BASE_PFN + 1) << PAGE_SHIFT_ARM64E;
        let mut got = [0u8; (W * 4) as usize];
        crate::runtime::host::HostMemory::read_gpa(&host, gpa, &mut got)
            .expect("the destination page is guest RAM");
        assert_eq!(
            got,
            [0u8; (W * 4) as usize],
            "nothing may have been written"
        );
    }

    /// A deferred frame landing over pages the guest CPU wrote in between keeps
    /// both writers.
    ///
    /// This is the external relation `cpu_write_after_render` asks for. A
    /// `.shared` texture's storage is guest RAM and the GPU and the guest CPU
    /// are both writers of it; Metal's guarantee is per region, so a CPU write
    /// into one part of a layer the GPU rendered leaves the rest of the GPU's
    /// work standing. A writeback with no third answer had only two, and both
    /// destroy a writer: land the whole frame and the guest's bytes are gone,
    /// drop it and everything the GPU rendered is gone.
    #[test]
    fn a_landed_frame_leaves_the_bytes_the_guest_wrote_alone() {
        let (mut host, mut state) = fixture();
        let gpa = (DATA_BASE_PFN + 1) << PAGE_SHIFT_ARM64E;
        // What the guest CPU put in row 1 after the render and before this
        // payment. Distinct from every byte of the frame.
        let guest_row = [0xEFu8; (W * 4) as usize];
        crate::runtime::host::HostMemory::write_gpa(&mut host, gpa + u64::from(BPR), &guest_row)
            .expect("the destination page is guest RAM");

        let rgba = frame();
        let pages = licence_for(1);
        // Row 1's bytes, in the same coordinate system the row offsets are in.
        let skip = [(u64::from(BPR), u64::from(BPR) + u64::from(W) * 4)];
        land_gva_frame_bytes(
            &mut state,
            &mut host,
            1,
            &request(),
            0,
            crate::runtime::draw::FrameRows::Rgba8(&rgba),
            &pages,
            &skip,
        )
        .expect("the licensed page is writable");

        let mut got0 = [0u8; (W * 4) as usize];
        crate::runtime::host::HostMemory::read_gpa(&host, gpa, &mut got0)
            .expect("the destination page is guest RAM");
        assert_eq!(
            &got0[..],
            &rgba[..(W * 4) as usize],
            "the device's own row must still land"
        );

        let mut got1 = [0u8; (W * 4) as usize];
        crate::runtime::host::HostMemory::read_gpa(&host, gpa + u64::from(BPR), &mut got1)
            .expect("the destination page is guest RAM");
        assert_eq!(
            got1, guest_row,
            "the guest's own row must survive the payment"
        );
    }
}
