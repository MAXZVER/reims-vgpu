//! The page-diff payment's damage band (TEMP lab, `REIMS_VGPU_LAB_PAY_BAND`) end
//! to end on a real device.
//!
//! The payment lands into guest pages only the 4 KiB pages its compare calls
//! changed, so whatever the band narrows the compare to, the guest's frame must
//! still equal the resident's after every payment. This keeps a model of the
//! guest's pages, lands each answer into it, and checks the model against a full
//! readback of the resident — the same claim `REIMS_VGPU_LAB_PAY_VERIFY` makes in
//! a boot — and checks the set of pages each answer named.
//!
//! The assertions hold in every mode, which is the parity: run it with the gate
//! unset or `off` (the full compare), `on` (the band's pages) and `verify` (the
//! full compare, counting what the band would have missed). A run that sets
//! nothing takes `on`, the arm that has something to prove.
//!
//! The band composes with `REIMS_VGPU_LAB_PAY_HASH`, the compare that keeps a
//! hash per page in place of the previous frame: the band narrows whichever
//! record the compare keeps, and the pages it leaves out keep their stored bytes
//! or hash words. The same scenario therefore runs once per hash arm (`off`,
//! `on`, `verify`, set for the test's own thread), with the band mode of the
//! process, and a randomized sequence is paid through all three arms at once and
//! must land the same pages and bytes in each.
//!
//! The band mode is latched on the first draw of the process, so this is its own
//! test binary, and each test takes the engine lock for its whole length.

#![cfg(feature = "backend-vulkan")]

use reims_vgpu::backend::vulkan::engine::{
    self, DrawRequest, PayHash, PrimitiveTopology, ScissorResource, TargetIdentity,
};
use std::sync::{Mutex, OnceLock};

/// One 4 KiB page per row, so a row range is a page range.
const W: u32 = 1024;
const H: u32 = 64;
const PAGE: usize = 4096;
const PAGES: u32 = H;
const FRAME_BYTES: usize = (W * H * 4) as usize;

const SURFACE_TEST_FORMAT: ash::vk::Format =
    reims_vgpu::backend::vulkan::translate::pixel::SCANOUT_FORMAT;

fn engine_device() -> &'static reims_vgpu::model::DeviceState {
    static DEVICE: OnceLock<reims_vgpu::model::DeviceState> = OnceLock::new();
    DEVICE.get_or_init(|| {
        reims_vgpu::model::DeviceState::new(
            reims_vgpu::model::DeviceId(1),
            reims_vgpu_protocol::gva::PAGE_SHIFT_X86,
        )
    })
}

fn engine_test_session() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| {
            // Never share the live product logs with a concurrent boot.
            reims_vgpu::observe::redirect_logs_for_tests();
            // The gate is latched by the first draw: take the mode the run asked
            // for, and `on` when it asked for none.
            if std::env::var_os("REIMS_VGPU_LAB_PAY_BAND").is_none() {
                std::env::set_var("REIMS_VGPU_LAB_PAY_BAND", "on");
            }
            Mutex::new(())
        })
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    engine::test_reset_engine(engine_device());
    guard
}

fn skip_if_no_gpu(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("no vulkan")
        || lower.contains("load vulkan")
        || lower.contains("create_instance")
        || lower.contains("no graphics")
        || lower.contains("vk_engine_init")
}

fn engine_req(vert: &[u32], frag: &[u32]) -> DrawRequest {
    DrawRequest {
        vert_spirv: std::sync::Arc::new(vert.to_vec()),
        frag_spirv: std::sync::Arc::new(frag.to_vec()),
        width: W,
        height: H,
        vertex_count: 3,
        first_vertex: 0,
        instance_count: Some(1),
        base_instance: 0,
        primitive_topology: PrimitiveTopology(reims_vgpu_core::topology::PrimitiveType::Triangle),
        ..Default::default()
    }
}

/// A vertex and a fragment module that rasterize nothing, so a pass that clears
/// to `color_attachment`'s colour leaves exactly that colour everywhere.
#[rustfmt::skip]
fn degenerate_triangle_spirv() -> (Vec<u32>, Vec<u32>) {
    let body = |interface_decoration: [u32; 4]| {
        let mut w = interface_decoration.to_vec();
        w.extend_from_slice(&[
            0x0002_0013, 3, // OpTypeVoid
            0x0003_0021, 4, 3, // OpTypeFunction
            0x0003_0016, 5, 32, // OpTypeFloat 32
            0x0004_0017, 6, 5, 4, // OpTypeVector
            0x0004_0020, 7, 3, 6, // OpTypePointer Output
            0x0004_003b, 7, 2, 3, // OpVariable Output
            0x0004_002b, 5, 8, 0, // OpConstant 0.0
            0x0004_002b, 5, 9, 0x3f80_0000, // OpConstant 1.0
            0x0007_002c, 6, 10, 8, 8, 8, 9, // OpConstantComposite
            0x0005_0036, 3, 1, 0, 4, // OpFunction
            0x0002_00f8, 11, // OpLabel
            0x0003_003e, 2, 10, // OpStore
            0x0001_00fd, // OpReturn
            0x0001_0038, // OpFunctionEnd
        ]);
        w
    };
    let header = [0x0723_0203, 0x0001_0000, 0, 12, 0, 0x0002_0011, 1, 0x0003_000e, 0, 1];
    let main = 0x6e69_616d;
    let mut vert = header.to_vec();
    vert.extend_from_slice(&[0x0006_000f, 0, 1, main, 0, 2]);
    vert.extend(body([0x0004_0047, 2, 11, 0]));
    let mut frag = header.to_vec();
    frag.extend_from_slice(&[0x0006_000f, 4, 1, main, 0, 2]);
    frag.extend_from_slice(&[0x0003_0010, 1, 7]);
    frag.extend(body([0x0004_0047, 2, 30, 0]));
    (vert, frag)
}

/// A fullscreen triangle in magenta, from words assembled by hand (no AIR
/// fixture, so no `llvm-dis` on the host): the vertex stage places vertex `i` at
/// `(4 * (i & 1) - 1, 2 * (i & 2) - 1)`, i.e. (-1,-1), (3,-1), (-1,3), and the
/// fragment stage writes `(1, 0, 1, 1)` — exact in unorm8, and none of the
/// colours the passes clear to.
#[rustfmt::skip]
fn fullscreen_triangle_spirv() -> (Vec<u32>, Vec<u32>) {
    let header = |bound: u32| [0x0723_0203, 0x0001_0000, 0, bound, 0, 0x0002_0011, 1, 0x0003_000e, 0, 1];
    let main = 0x6e69_616d;
    // Vertex. Ids: %1 main, %2 Position (Output vec4), %3 void, %4 fn, %5 float,
    // %6 vec4, %7 ptr Output vec4, %8 0.0, %9 1.0, %12 int, %13 ptr Input int,
    // %14 VertexIndex, %15 1, %16 2, %25 -1.0, then the body's values.
    let mut vert = header(29).to_vec();
    vert.extend_from_slice(&[
        0x0007_000f, 0, 1, main, 0, 2, 14, // OpEntryPoint Vertex main %2 %14
        0x0004_0047, 2, 11, 0, // OpDecorate %2 BuiltIn Position
        0x0004_0047, 14, 11, 42, // OpDecorate %14 BuiltIn VertexIndex
        0x0002_0013, 3, // OpTypeVoid
        0x0003_0021, 4, 3, // OpTypeFunction
        0x0003_0016, 5, 32, // OpTypeFloat 32
        0x0004_0017, 6, 5, 4, // OpTypeVector
        0x0004_0020, 7, 3, 6, // OpTypePointer Output vec4
        0x0004_003b, 7, 2, 3, // OpVariable Output
        0x0004_0015, 12, 32, 1, // OpTypeInt 32 signed
        0x0004_0020, 13, 1, 12, // OpTypePointer Input int
        0x0004_003b, 13, 14, 1, // OpVariable Input
        0x0004_002b, 5, 8, 0, // OpConstant 0.0
        0x0004_002b, 5, 9, 0x3f80_0000, // OpConstant 1.0
        0x0004_002b, 5, 25, 0xbf80_0000, // OpConstant -1.0
        0x0004_002b, 12, 15, 1, // OpConstant int 1
        0x0004_002b, 12, 16, 2, // OpConstant int 2
        0x0005_0036, 3, 1, 0, 4, // OpFunction
        0x0002_00f8, 17, // OpLabel
        0x0004_003d, 12, 18, 14, // %18 = OpLoad VertexIndex
        0x0005_00c7, 12, 19, 18, 15, // %19 = %18 & 1
        0x0005_00c4, 12, 20, 19, 16, // %20 = %19 << 2
        0x0005_00c7, 12, 21, 18, 16, // %21 = %18 & 2
        0x0005_00c4, 12, 22, 21, 15, // %22 = %21 << 1
        0x0004_006f, 5, 23, 20, // %23 = float(%20)
        0x0004_006f, 5, 24, 22, // %24 = float(%22)
        0x0005_0081, 5, 26, 23, 25, // %26 = %23 + -1.0
        0x0005_0081, 5, 27, 24, 25, // %27 = %24 + -1.0
        0x0007_0050, 6, 28, 26, 27, 8, 9, // %28 = vec4(%26, %27, 0, 1)
        0x0003_003e, 2, 28, // OpStore Position
        0x0001_00fd, // OpReturn
        0x0001_0038, // OpFunctionEnd
    ]);
    // Fragment: Location 0 = (1, 0, 1, 1).
    let mut frag = header(12).to_vec();
    frag.extend_from_slice(&[
        0x0006_000f, 4, 1, main, 0, 2, // OpEntryPoint Fragment main %2
        0x0003_0010, 1, 7, // OpExecutionMode OriginUpperLeft
        0x0004_0047, 2, 30, 0, // OpDecorate %2 Location 0
        0x0002_0013, 3, // OpTypeVoid
        0x0003_0021, 4, 3, // OpTypeFunction
        0x0003_0016, 5, 32, // OpTypeFloat 32
        0x0004_0017, 6, 5, 4, // OpTypeVector
        0x0004_0020, 7, 3, 6, // OpTypePointer Output
        0x0004_003b, 7, 2, 3, // OpVariable Output
        0x0004_002b, 5, 8, 0, // OpConstant 0.0
        0x0004_002b, 5, 9, 0x3f80_0000, // OpConstant 1.0
        0x0007_002c, 6, 10, 9, 8, 9, 9, // OpConstantComposite (1, 0, 1, 1)
        0x0005_0036, 3, 1, 0, 4, // OpFunction
        0x0002_00f8, 11, // OpLabel
        0x0003_003e, 2, 10, // OpStore
        0x0001_00fd, // OpReturn
        0x0001_0038, // OpFunctionEnd
    ]);
    (vert, frag)
}

fn attachment(clear: [f64; 4]) -> engine::ColorAttachmentState {
    reims_vgpu::backend::vulkan::translate::pixel::color_attachment(
        reims_vgpu::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM,
    )
    .expect("test attachment format is renderable")
    .0
    .with_clear(clear)
}

/// Clear the whole resident to one colour: a pass that rewrites every texel.
fn clear_to(identity: &TargetIdentity, rgba: [u8; 4]) -> bool {
    let (vert, frag) = degenerate_triangle_spirv();
    let mut req = engine_req(&vert, &frag);
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;
    req.color_attachment = Some(attachment(rgba.map(|c| f64::from(c) / 255.0)));
    match engine::execute_draw_request(engine_device(), &req) {
        Ok(_) => true,
        Err(e) if skip_if_no_gpu(&e.to_string()) => {
            eprintln!("SKIP pay band: {e}");
            false
        }
        Err(e) => panic!("pay band clear: {e}"),
    }
}

/// Paint the fullscreen triangle over the resident's *current*
/// contents, clipped to one scissor per `(y0, y1)`. The primitives reach only
/// the first scissor's rows (nothing selects another viewport), the band is the
/// hull of every scissor the draw bound.
fn paint_rows(identity: &TargetIdentity, shaders: &(Vec<u32>, Vec<u32>), rows: &[(u32, u32)]) {
    let mut req = engine_req(&shaders.0, &shaders.1);
    req.target_identity = Some(identity.clone());
    req.skip_readback = true;
    req.load_from_target = true;
    req.color_attachment = Some(attachment([0.0; 4]));
    req.scissors = rows
        .iter()
        .map(|&(y0, y1)| ScissorResource {
            x: 0,
            y: y0,
            width: W,
            height: y1 - y0,
        })
        .collect();
    engine::execute_draw_request(engine_device(), &req).expect("pay band paint");
}

/// The guest's pages, as the payments have landed them, and the hash arm
/// (`REIMS_VGPU_LAB_PAY_HASH`) its payments are made through.
struct Guest(Vec<u8>, PayHash);

impl Guest {
    /// One payment: ask the engine, land what it names, return which pages.
    fn pay(
        &mut self,
        identity: &TargetIdentity,
        mapping: u32,
        key: (u32, u32, u32, u64),
        force: &[u32],
    ) -> (Vec<u32>, bool) {
        self.pay_expecting(identity, mapping, key, force, None)
    }

    /// [`Self::pay`], and when `compared` names the pages a band-narrowed compare
    /// is dispatched over, check what the compare was dispatched over in the mode
    /// this run took: those pages with the gate `on`, every page with `verify`,
    /// none counted with it off — and none for a first payment, which copies the
    /// frame rather than comparing it.
    fn pay_expecting(
        &mut self,
        identity: &TargetIdentity,
        mapping: u32,
        key: (u32, u32, u32, u64),
        force: &[u32],
        compared: Option<u64>,
    ) -> (Vec<u32>, bool) {
        let before = engine::lab_pay_band_pages();
        let mut ran = 0u32;
        let got = engine::with_pay_hash(self.1, || {
            engine::read_target_changed_pages_with(identity, mapping, key, force, || {
                ran += 1;
            })
        })
        .expect("page diff read")
        .expect("a BGRA8 resident takes the page diff");
        assert_eq!(ran, 1, "the between hook runs once per answered read");
        assert_eq!(got.frame_bytes as usize, FRAME_BYTES);
        // SAFETY: the engine's mapped `out` buffer, `frame_bytes` long, valid
        // until the next page-diff call.
        let out = unsafe { std::slice::from_raw_parts(got.frame_ptr, FRAME_BYTES) };
        for &p in &got.pages {
            let at = p as usize * PAGE;
            self.0[at..at + PAGE].copy_from_slice(&out[at..at + PAGE]);
        }
        engine::page_diff_set_key(mapping, key);
        if let Some(narrow) = compared {
            let after = engine::lab_pay_band_pages();
            let want = match (mode().as_str(), got.full) {
                (_, true) | ("off", _) => (0, 0),
                ("verify", false) => (u64::from(PAGES), u64::from(PAGES)),
                (_, false) => (narrow, u64::from(PAGES)),
            };
            assert_eq!(
                (after.0 - before.0, after.1 - before.1),
                want,
                "pages the compare was dispatched over, and pages of the frame, in mode {}",
                mode()
            );
        }
        (got.pages, got.full)
    }

    /// The guest's frame is the resident, byte for byte.
    fn assert_equals_resident(&self, identity: &TargetIdentity, when: &str) {
        let truth = engine::read_target(identity)
            .expect("read the resident")
            .into_bgra8()
            .expect("an eight-bit colour readback");
        let bad: Vec<usize> = (0..PAGES as usize)
            .filter(|&p| self.0[p * PAGE..(p + 1) * PAGE] != truth[p * PAGE..(p + 1) * PAGE])
            .collect();
        assert!(
            bad.is_empty(),
            "{when}: the guest's pages {bad:?} are not the resident's"
        );
    }
}

/// The gate as this run took it (the session sets `on` when nothing did).
fn mode() -> String {
    std::env::var("REIMS_VGPU_LAB_PAY_BAND").unwrap_or_default()
}

fn range(a: u32, b: u32) -> Vec<u32> {
    (a..b).collect()
}

/// The scenario of one hash arm. Each arm pays its own surface under its own
/// mapping, so its shadow is new (the page-diff state outlives the engine reset).
fn scenario(hash: PayHash, surface: u32) {
    let _g = engine_test_session();
    let verify_before = engine::pay_hash_verify_totals();
    let shaders = fullscreen_triangle_spirv();
    let id = TargetIdentity::Surface {
        id: surface,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mapping = surface;
    let key = (1u32, W, H, 1u64);
    let mut guest = Guest(vec![0xAB; FRAME_BYTES], hash);

    if !clear_to(&id, [51, 102, 153, 255]) {
        return; // no GPU
    }

    // A new shadow lands the whole frame.
    let (pages, full) = guest.pay(&id, mapping, key, &[]);
    assert!(full);
    assert_eq!(pages, range(0, PAGES));
    guest.assert_equals_resident(&id, "first landing");

    // Nothing written since: nothing to compare, nothing to land.
    let (pages, full) = guest.pay_expecting(&id, mapping, key, &[], Some(0));
    assert!(!full);
    assert!(
        pages.is_empty(),
        "an unchanged frame lands nothing: {pages:?}"
    );

    // A scissored draw over the loaded frame: its rows, and only them.
    paint_rows(&id, &shaders, &[(10, 20)]);
    let (pages, _) = guest.pay_expecting(&id, mapping, key, &[], Some(10));
    assert_eq!(
        pages,
        range(10, 20),
        "the scissor's rows are the changed pages"
    );
    guest.assert_equals_resident(&id, "after the first scissored draw");

    // The account starts again at each payment: a later draw elsewhere, and a
    // page the caller forces from outside its band, which must be landed with the
    // frame's own bytes though nothing there changed.
    paint_rows(&id, &shaders, &[(30, 32)]);
    let (pages, _) = guest.pay_expecting(&id, mapping, key, &[50], Some(3));
    assert_eq!(pages, vec![30, 31, 50]);
    guest.assert_equals_resident(&id, "a forced page outside the band");

    let (pages, _) = guest.pay_expecting(&id, mapping, key, &[], Some(0));
    assert!(
        pages.is_empty(),
        "the band was reset by the payment: {pages:?}"
    );

    // A repaint of rows already holding that colour: the band names them, the
    // compare still compares, and finds nothing changed.
    paint_rows(&id, &shaders, &[(10, 20)]);
    let (pages, _) = guest.pay_expecting(&id, mapping, key, &[], Some(10));
    assert!(
        pages.is_empty(),
        "identical bytes are not a change: {pages:?}"
    );

    // Two scissors bound: the band is the hull of both, and the compare finds
    // what the primitives actually reached (the first scissor's rows).
    paint_rows(&id, &shaders, &[(5, 6), (40, 42)]);
    let (pages, _) = guest.pay_expecting(&id, mapping, key, &[], Some(37));
    assert_eq!(pages, vec![5]);
    guest.assert_equals_resident(&id, "two scissors");

    // A draw that clears rewrites every texel, whatever scissors it binds.
    assert!(clear_to(&id, [204, 0, 85, 255]));
    let (pages, full) = guest.pay_expecting(&id, mapping, key, &[], Some(u64::from(PAGES)));
    assert!(!full);
    assert_eq!(pages, range(0, PAGES), "a clear is every row");
    guest.assert_equals_resident(&id, "after a clear");

    // And the band comes back narrow after it.
    paint_rows(&id, &shaders, &[(0, 1)]);
    let (pages, _) = guest.pay_expecting(&id, mapping, key, &[], Some(1));
    assert_eq!(pages, vec![0]);
    guest.assert_equals_resident(&id, "a narrow draw after a clear");

    // A shadow that is replaced (the key moved: the guest's pages were written
    // behind the payment) is a fresh one, and takes the whole frame whatever the
    // band says.
    let moved = (1u32, W, H, 2u64);
    let (pages, full) = guest.pay_expecting(&id, mapping, moved, &[], Some(0));
    assert!(full, "a shadow under another key lands the whole frame");
    assert_eq!(pages, range(0, PAGES));
    guest.assert_equals_resident(&id, "a replaced shadow");

    // `verify`'s claim, and `on`'s and `off`'s too: nothing changed outside a band.
    assert_eq!(
        engine::lab_pay_band_outside_changed(),
        0,
        "a page changed outside the rows the draws were accounted to"
    );
    // The hash `verify` arm's claim: no page one compare lands and the other
    // misses.
    let verify_after = engine::pay_hash_verify_totals();
    assert_eq!(
        (
            verify_after.1 - verify_before.1,
            verify_after.2 - verify_before.2
        ),
        (0, 0),
        "the prev and hash compares disagreed ({hash:?})"
    );
}

#[test]
fn a_payment_lands_exactly_the_pages_its_writes_changed_whatever_narrows_the_compare() {
    scenario(PayHash::Off, 7201);
}

#[test]
fn a_hash_payment_lands_exactly_the_same_pages_whatever_the_band_narrows() {
    scenario(PayHash::On, 7202);
}

#[test]
fn a_hash_verify_payment_lands_exactly_the_same_pages_whatever_the_band_narrows() {
    scenario(PayHash::Verify, 7203);
}

/// One resident paid through different hash arms in turn, with a write between
/// payments: each payment refreshes only the records its compare ran, and a
/// compare whose record the last payment did not refresh (the `prev` frame after
/// a hash payment, the hash words after a `prev` one) must take every page and
/// rebuild it, whatever band the writes since the last payment left. A band that
/// narrowed such a payment would land `out` bytes nothing had just written.
#[test]
fn a_record_that_fell_behind_is_rebuilt_whatever_the_band_says() {
    let _g = engine_test_session();
    let verify_before = engine::pay_hash_verify_totals();
    let shaders = fullscreen_triangle_spirv();
    let id = TargetIdentity::Surface {
        id: 7231,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let mapping = 7231u32;
    let key = (1u32, W, H, 1u64);
    let mut guest = Guest(vec![0xAB; FRAME_BYTES], PayHash::Off);
    if !clear_to(&id, [51, 102, 153, 255]) {
        return; // no GPU
    }
    let all = range(0, PAGES);
    // Each step: the hash arm, the rows written before the payment, the pages the
    // payment lands, and whether it takes every page. A repaint of rows already
    // holding the colour changes nothing, so a record the payment before did not
    // rebuild for those pages would show as a page landed that did not change.
    let (off, on, verify) = (PayHash::Off, PayHash::On, PayHash::Verify);
    let none: Vec<u32> = Vec::new();
    let steps = [
        (off, (0, 0), all.clone(), true, "a new shadow"),
        (off, (30, 32), vec![30, 31], false, "prev compares the band"),
        (on, (10, 20), all.clone(), true, "the hash record is new"),
        (
            on,
            (30, 32),
            none.clone(),
            false,
            "a repaint outside the seeding payment's band",
        ),
        (on, (40, 41), vec![40], false, "the hash record stands"),
        (
            off,
            (5, 6),
            all.clone(),
            true,
            "prev fell behind the hash payment",
        ),
        (
            verify,
            (50, 52),
            all.clone(),
            true,
            "the hashes fell behind the prev payment",
        ),
        (
            verify,
            (5, 6),
            none.clone(),
            false,
            "a repaint outside the seeding payment's band",
        ),
        (
            on,
            (25, 26),
            vec![25],
            false,
            "the hashes stand after verify",
        ),
        (
            verify,
            (22, 23),
            all.clone(),
            true,
            "prev fell behind the hash payment",
        ),
        (off, (60, 61), vec![60], false, "prev stands after verify"),
        (
            on,
            (3, 4),
            all.clone(),
            true,
            "the hashes fell behind the prev payment",
        ),
        (
            on,
            (60, 61),
            none,
            false,
            "a repaint outside the seeding payment's band",
        ),
    ];
    for (n, (hash, (y0, y1), want, want_full, what)) in steps.into_iter().enumerate() {
        if y0 < y1 {
            paint_rows(&id, &shaders, &[(y0, y1)]);
        }
        guest.1 = hash;
        let (pages, full) = guest.pay(&id, mapping, key, &[]);
        assert_eq!(
            full, want_full,
            "step {n} ({hash:?}, {what}): takes every page"
        );
        assert_eq!(
            pages, want,
            "step {n} ({hash:?}, {what}): lands these pages"
        );
        guest.assert_equals_resident(&id, &format!("step {n} ({hash:?}, {what})"));
    }
    assert_eq!(engine::lab_pay_band_outside_changed(), 0);
    let verify_after = engine::pay_hash_verify_totals();
    assert_eq!(
        (
            verify_after.1 - verify_before.1,
            verify_after.2 - verify_before.2
        ),
        (0, 0),
        "the prev and hash compares disagreed"
    );
}

/// One arm of the randomized differential: its own resident (a surface whose
/// generation moves when the resident is replaced), mapping and guest pages.
struct Arm {
    surface: u32,
    generation: u64,
    key: (u32, u32, u32, u64),
    guest: Guest,
}

impl Arm {
    fn id(&self) -> TargetIdentity {
        TargetIdentity::Surface {
            id: self.surface,
            width: W,
            height: H,
            generation: self.generation,
            format: SURFACE_TEST_FORMAT,
        }
    }
}

/// Pay every arm once and check that each landed what the first did, left the
/// same bytes in its guest, and that the guest equals its resident.
fn pay_all(arms: &mut [Arm], force: &[u32], step: usize, what: &str) -> (Vec<u32>, bool) {
    let mut reference: Option<(Vec<u32>, bool)> = None;
    for arm in arms.iter_mut() {
        let (id, mapping, key) = (arm.id(), arm.surface, arm.key);
        let got = arm.guest.pay(&id, mapping, key, force);
        match &reference {
            None => reference = Some(got),
            Some(want) => assert_eq!(
                &got, want,
                "step {step} ({what}): {:?} landed other pages than the first arm",
                arm.guest.1
            ),
        }
        arm.guest
            .assert_equals_resident(&id, &format!("step {step} ({what}) {:?}", arm.guest.1));
    }
    for arm in &arms[1..] {
        assert!(
            arm.guest.0 == arms[0].guest.0,
            "step {step} ({what}): {:?}'s guest bytes differ from the first arm's",
            arm.guest.1
        );
    }
    reference.expect("an arm")
}

/// The three hash arms are paid the same randomized sequence of writes (scissored
/// draws of one or two bands, clears, a resident replaced under its shadow, a
/// shadow replaced under its resident, nothing at all) with random forced pages,
/// and every payment must land the same pages, and leave the same bytes in the
/// guest, in each — and the guest must equal the resident. With the band `on`
/// this is the claim that composing the band with the hash compare (or with
/// `verify`, which runs both) moves nothing a payment lands; run under each band
/// mode it is the same test.
#[test]
fn every_hash_arm_lands_the_same_pages_and_bytes_through_a_random_sequence() {
    let _g = engine_test_session();
    let verify_before = engine::pay_hash_verify_totals();
    let shaders = fullscreen_triangle_spirv();
    let mut arms: Vec<Arm> = [
        (PayHash::Off, 7211u32),
        (PayHash::On, 7212),
        (PayHash::Verify, 7213),
    ]
    .into_iter()
    .map(|(hash, surface)| Arm {
        surface,
        generation: 1,
        key: (1, W, H, 1),
        guest: Guest(vec![0xAB; FRAME_BYTES], hash),
    })
    .collect();
    const COLORS: [[u8; 4]; 3] = [[51, 102, 153, 255], [204, 0, 85, 255], [0, 128, 0, 255]];
    let mut seed = 0x2545_F491u32;
    let mut next = move |n: u32| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) % n
    };
    for arm in &arms {
        if !clear_to(&arm.id(), COLORS[0]) {
            return; // no GPU
        }
    }

    let (pages, full) = pay_all(&mut arms, &[], 0, "first frame");
    assert!(full);
    assert_eq!(pages, range(0, PAGES));
    let census_before = engine::lab_pay_band_pages();

    for step in 1..=80usize {
        // `true` when the step leaves every arm's shadow new.
        let (what, shadow_replaced) = match next(12) {
            0..=5 => {
                let y0 = next(H - 1);
                let mut rows = vec![(y0, (y0 + 1 + next(9)).min(H))];
                if next(4) == 0 {
                    let z0 = next(H - 1);
                    rows.push((z0, (z0 + 1 + next(5)).min(H)));
                }
                for arm in &arms {
                    paint_rows(&arm.id(), &shaders, &rows);
                }
                ("painted rows", false)
            }
            6 | 7 => {
                let c = COLORS[next(3) as usize];
                for arm in &arms {
                    assert!(clear_to(&arm.id(), c));
                }
                ("cleared", false)
            }
            8 => {
                // The resident replaced under its shadow: a new image, so a new
                // slot, paid against the records of the old one.
                let c = COLORS[next(3) as usize];
                for arm in &mut arms {
                    arm.generation += 1;
                    assert!(clear_to(&arm.id(), c));
                }
                ("replaced the resident", false)
            }
            9 => {
                // The shadow replaced under its resident: the guest's pages were
                // written behind the payment (the key moved), or the landing
                // failed (`page_diff_forget`, which an unfinished payment's
                // caller does).
                if next(2) == 0 {
                    for arm in &mut arms {
                        arm.key.3 += 1;
                    }
                    ("moved the key", true)
                } else {
                    for arm in &arms {
                        engine::page_diff_forget(arm.surface);
                    }
                    ("forgot the shadow", true)
                }
            }
            _ => ("nothing", false),
        };
        let mut force: Vec<u32> = Vec::new();
        if next(3) == 0 {
            for _ in 0..1 + next(3) {
                force.push(next(PAGES));
            }
            force.sort_unstable();
            force.dedup();
        }
        let (pages, full) = pay_all(&mut arms, &force, step, what);
        if shadow_replaced {
            assert!(full, "step {step}: a shadow under another key lands whole");
            assert_eq!(pages, range(0, PAGES));
        }
        for p in &force {
            assert!(
                pages.contains(p),
                "step {step} ({what}): forced page {p} was not landed"
            );
        }
    }

    assert_eq!(
        engine::lab_pay_band_outside_changed(),
        0,
        "a page changed outside the rows the draws were accounted to"
    );
    // What the band did to the compares of the run, in the mode this run took.
    let census_after = engine::lab_pay_band_pages();
    let (dispatched, total) = (
        census_after.0 - census_before.0,
        census_after.1 - census_before.1,
    );
    match mode().as_str() {
        "on" => assert!(
            dispatched < total,
            "the band narrowed no compare: {dispatched} of {total} pages"
        ),
        "verify" => assert_eq!(dispatched, total, "verify compares every page"),
        _ => assert_eq!(
            (dispatched, total),
            (0, 0),
            "nothing is counted with the gate off"
        ),
    }
    let verify_after = engine::pay_hash_verify_totals();
    assert!(
        verify_after.0 > verify_before.0,
        "the verify arm compared pages"
    );
    assert_eq!(
        (
            verify_after.1 - verify_before.1,
            verify_after.2 - verify_before.2
        ),
        (0, 0),
        "the prev and hash compares disagreed"
    );
}
