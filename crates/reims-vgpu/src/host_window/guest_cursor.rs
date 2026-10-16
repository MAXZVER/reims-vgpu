//! The guest's own pointer glyph, shown as the host window's cursor.
//!
//! QEMU runs this device with `-display none`, so the cursor the guest sends
//! (`CmdSetCursorGlyph`, op 0x04, loaded by the drain's `load_cursor_glyph`)
//! reached no screen: the C side's `qemu_console_set_cursor` has no console to
//! draw on, and nothing in the window set a cursor, so the host window showed
//! the platform's default arrow over the guest desktop.
//!
//! The pieces, in the order a glyph travels:
//! - [`offer_cursor`] runs on the drain worker at the tranche tail (the
//!   window publisher's own call site). It converts the guest's QEMUCursor
//!   words (`0xAARRGGBB`, straight alpha) into RGBA8 **once**, and stores the
//!   result in the shared [`CursorSlot`] with `try_lock`, so the drain never
//!   waits on the window thread. A contended slot leaves the glyph pending
//!   (`host_window_cursor_publish_skipped`) and the next tranche retries.
//! - The window thread compares the slot's `seq` and the current
//!   viewport scale against what it last applied ([`CursorKey`]), scales the
//!   glyph by the same window-px-per-guest-px ratio the presenter's aspect-fit
//!   uses, and hands it to the platform as a custom cursor. Built cursors are
//!   cached per (content, scale) in [`CursorView`].
//! - Until the first glyph arrives, and whenever a guest glyph cannot be
//!   built, the window shows [`static_arrow`]: a black arrow with a white
//!   outline, hotspot at the tip, drawn in code.
//!
//! `REIMS_VGPU_WINDOW_GUEST_CURSOR=0` keeps the platform's own arrow exactly as
//! before (mode `system`); nothing is published or set.
//!
//! The platform applies a window cursor only inside the client area (winit's
//! `WM_SETCURSOR` arm on Windows hands non-client hits back to
//! `DefWindowProc`), so the desktop's own arrow returns at the window edge.
//!
//! The guest's show/hide bit is **not** mirrored. winit's `set_cursor_visible`
//! on Windows also confines the pointer to the window's centre while hidden,
//! and a transparent glyph would leave the pointer invisible inside the window
//! for as long as a drain stall delays the guest's next show. Both are worse
//! than a visible pointer while the guest briefly hides its own.

use std::sync::{Arc, Mutex, TryLockError};

/// Largest edge of a scaled cursor image, in window pixels. The guest's own
/// bound (`CURSOR_MAX_DIM`) is the same value; winit's is 2048.
pub const CURSOR_OUT_MAX_DIM: u32 = 512;

/// Fixed-point unit for [`CursorKey::scale_q`]: 64 is 1.0.
pub const SCALE_ONE: u32 = 64;
const SCALE_MIN: u32 = SCALE_ONE / 4;
const SCALE_MAX: u32 = SCALE_ONE * 8;

/// One guest cursor glyph, converted once on the drain side into what a
/// platform cursor takes: straight-alpha RGBA8, tightly packed
/// `width * height * 4`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestCursorGlyph {
    /// `CursorState::glyph_serial` of the load this came from.
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub hot_x: u32,
    pub hot_y: u32,
    /// Hash of the geometry and pixels, so a glyph the guest re-sends unchanged
    /// reuses the cursor already built for it.
    pub content: u64,
    pub rgba: Vec<u8>,
}

/// Latest-wins slot the drain writes and the window reads, beside the
/// `FrameSlot`.
pub type CursorSlot = Arc<Mutex<Option<Arc<GuestCursorGlyph>>>>;

/// An empty slot.
pub fn new_slot() -> CursorSlot {
    Arc::new(Mutex::new(None))
}

/// Why a loaded glyph was not turned into a [`GuestCursorGlyph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlyphRefusal {
    /// Zero width or height.
    Empty,
    /// The pixel words do not cover `width * height`.
    LengthMismatch { have: usize, need: usize },
    /// The hotspot is outside the glyph.
    Hotspot,
}

impl GlyphRefusal {
    pub fn slug(&self) -> &'static str {
        match self {
            Self::Empty => "glyph_empty",
            Self::LengthMismatch { .. } => "glyph_len_mismatch",
            Self::Hotspot => "glyph_hotspot",
        }
    }
}

/// QEMUCursor `0xAARRGGBB` words to RGBA8 bytes. Straight alpha in, straight
/// alpha out: no premultiply, which is what winit's `from_rgba` expects.
pub fn argb_to_rgba(argb: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(argb.len() * 4);
    for &w in argb {
        out.push((w >> 16) as u8);
        out.push((w >> 8) as u8);
        out.push(w as u8);
        out.push((w >> 24) as u8);
    }
    out
}

fn content_hash(width: u32, height: u32, hot_x: u32, hot_y: u32, rgba: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (width, height, hot_x, hot_y).hash(&mut h);
    rgba.hash(&mut h);
    h.finish()
}

impl GuestCursorGlyph {
    /// Convert one loaded glyph (`CursorState` fields).
    pub fn from_argb(
        seq: u64,
        width: u32,
        height: u32,
        hot_x: u32,
        hot_y: u32,
        argb: &[u32],
    ) -> Result<Self, GlyphRefusal> {
        if width == 0 || height == 0 {
            return Err(GlyphRefusal::Empty);
        }
        let need = (width as usize).saturating_mul(height as usize);
        if argb.len() != need {
            return Err(GlyphRefusal::LengthMismatch {
                have: argb.len(),
                need,
            });
        }
        if hot_x >= width || hot_y >= height {
            return Err(GlyphRefusal::Hotspot);
        }
        let rgba = argb_to_rgba(argb);
        Ok(Self {
            seq,
            width,
            height,
            hot_x,
            hot_y,
            content: content_hash(width, height, hot_x, hot_y, &rgba),
            rgba,
        })
    }
}

/// How the window's cursor is chosen this run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorMode {
    /// Default: the bundled arrow until the guest's first glyph, then the
    /// guest's glyph.
    Guest,
    /// `REIMS_VGPU_WINDOW_GUEST_CURSOR=0`: the platform's own arrow, as before.
    System,
}

/// Map the switch. Unset, on, or unrecognised keep the guest cursor; only an
/// explicit off restores the system arrow.
pub fn mode_from(state: crate::config::Switch) -> CursorMode {
    if matches!(state, crate::config::Switch::Off) {
        CursorMode::System
    } else {
        CursorMode::Guest
    }
}

/// Read `REIMS_VGPU_WINDOW_GUEST_CURSOR` once.
pub fn mode() -> CursorMode {
    static MODE: std::sync::OnceLock<CursorMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let (state, value) = crate::config::read(crate::config::WINDOW_GUEST_CURSOR);
        if matches!(state, crate::config::Switch::Unrecognized) {
            crate::observe::fail(format!(
                "host_window_cursor_fail reason=guest_cursor_env_unrecognized value={}",
                value.unwrap_or_default()
            ));
        }
        mode_from(state)
    })
}

/// What [`offer_cursor`] did with the current glyph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorOffer {
    /// Nothing new since the last offer (or no glyph yet).
    Unchanged,
    /// Stored in the slot; the caller wakes the window.
    Published,
    /// The window was reading the slot. The glyph stays pending and the next
    /// tranche offers it again. Route `host_window_cursor_publish_skipped`.
    Skipped,
    /// The glyph could not be converted; it is marked done so it is not
    /// retried per tranche. Fail line `host_window_cursor_fail`.
    Refused(GlyphRefusal),
    /// Kill switch off, or the window thread is gone (poisoned slot): marked
    /// done, nothing stored.
    NotWanted,
}

/// Offer the drain's current glyph to the window, never blocking.
///
/// `published` is the serial last handed over; it is advanced on every outcome
/// except [`CursorOffer::Skipped`], which is what keeps a contended glyph
/// pending rather than lost.
#[allow(clippy::too_many_arguments)]
pub fn offer_cursor(
    slot: &CursorSlot,
    published: &mut u64,
    mode: CursorMode,
    serial: u64,
    ready: bool,
    geom: (u32, u32, u32, u32),
    argb: &[u32],
) -> CursorOffer {
    if serial == 0 || !ready || serial == *published {
        return CursorOffer::Unchanged;
    }
    if mode == CursorMode::System {
        *published = serial;
        return CursorOffer::NotWanted;
    }
    let (width, height, hot_x, hot_y) = geom;
    let glyph = match GuestCursorGlyph::from_argb(serial, width, height, hot_x, hot_y, argb) {
        Ok(glyph) => glyph,
        Err(refusal) => {
            *published = serial;
            return CursorOffer::Refused(refusal);
        }
    };
    match slot.try_lock() {
        Ok(mut current) => {
            *current = Some(Arc::new(glyph));
            *published = serial;
            CursorOffer::Published
        }
        Err(TryLockError::WouldBlock) => CursorOffer::Skipped,
        Err(TryLockError::Poisoned(_)) => {
            *published = serial;
            CursorOffer::NotWanted
        }
    }
}

/// Which picture the window's cursor shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CursorSource {
    /// The bundled arrow ([`static_arrow`]).
    Static,
    /// A guest glyph, by its `seq`.
    Guest(u64),
}

/// What the window last applied: a new source or a new scale rebuilds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CursorKey {
    pub source: CursorSource,
    pub scale_q: u32,
}

/// A ratio in [`SCALE_ONE`] fixed point, clamped to 0.25x..8x.
pub fn scale_q(ratio: f64) -> u32 {
    if !ratio.is_finite() || ratio <= 0.0 {
        return SCALE_ONE;
    }
    ((ratio * SCALE_ONE as f64).round() as u32).clamp(SCALE_MIN, SCALE_MAX)
}

/// Window pixels per guest pixel through the presenter's own aspect-fit, so the
/// pointer is the size it is in the picture around it. Before the first guest
/// frame there is no guest extent and the glyph is shown 1:1.
pub fn viewport_scale_q(window: (u32, u32), guest: Option<(u32, u32)>) -> u32 {
    let Some(guest) = guest else {
        return SCALE_ONE;
    };
    let vp = crate::backend::window::viewport::aspect_fit(guest, window);
    scale_q(f64::from(vp.width.max(1)) / f64::from(guest.0.max(1)))
}

/// The key the window should be showing.
///
/// `viewport_q` scales a guest glyph (guest pixels to window pixels); `dpi_q`
/// scales the bundled arrow, which has no guest pixels behind it and is sized
/// for the host display instead.
pub fn desired_key(glyph: Option<&GuestCursorGlyph>, viewport_q: u32, dpi_q: u32) -> CursorKey {
    match glyph {
        Some(glyph) => CursorKey {
            source: CursorSource::Guest(glyph.seq),
            scale_q: viewport_q,
        },
        None => CursorKey {
            source: CursorSource::Static,
            scale_q: dpi_q,
        },
    }
}

/// A cursor image in window pixels, ready for the platform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorImage {
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub rgba: Vec<u8>,
}

/// Scale a straight-alpha RGBA image by `scale_q`.
///
/// Exact copy at 1:1. Otherwise each output pixel averages a 4x4 grid of
/// nearest source samples in premultiplied space (so a transparent pixel's
/// colour cannot bleed into the edge) and is un-premultiplied again. The
/// hotspot scales with the image and stays inside it. `None` for an empty or
/// short input.
pub fn scale_image(
    width: u32,
    height: u32,
    hot_x: u32,
    hot_y: u32,
    rgba: &[u8],
    scale_q: u32,
) -> Option<CursorImage> {
    if width == 0 || height == 0 {
        return None;
    }
    let need = (width as usize) * (height as usize) * 4;
    if rgba.len() < need {
        return None;
    }
    let s = f64::from(scale_q.max(1)) / f64::from(SCALE_ONE);
    let out_w = ((f64::from(width) * s).round() as u32).clamp(1, CURSOR_OUT_MAX_DIM);
    let out_h = ((f64::from(height) * s).round() as u32).clamp(1, CURSOR_OUT_MAX_DIM);
    let hx = ((f64::from(hot_x) * f64::from(out_w) / f64::from(width)) as u32).min(out_w - 1);
    let hy = ((f64::from(hot_y) * f64::from(out_h) / f64::from(height)) as u32).min(out_h - 1);
    let out = if out_w == width && out_h == height {
        rgba[..need].to_vec()
    } else {
        let mut out = Vec::with_capacity((out_w * out_h * 4) as usize);
        const GRID: u32 = 4;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = [0u32; 4];
                for sy in 0..GRID {
                    for sx in 0..GRID {
                        let fx = (f64::from(ox) + (f64::from(sx) + 0.5) / f64::from(GRID))
                            * f64::from(width)
                            / f64::from(out_w);
                        let fy = (f64::from(oy) + (f64::from(sy) + 0.5) / f64::from(GRID))
                            * f64::from(height)
                            / f64::from(out_h);
                        let x = (fx as u32).min(width - 1) as usize;
                        let y = (fy as u32).min(height - 1) as usize;
                        let i = (y * width as usize + x) * 4;
                        let a = u32::from(rgba[i + 3]);
                        acc[0] += u32::from(rgba[i]) * a;
                        acc[1] += u32::from(rgba[i + 1]) * a;
                        acc[2] += u32::from(rgba[i + 2]) * a;
                        acc[3] += a;
                    }
                }
                let n = GRID * GRID;
                let a_sum = acc[3];
                let alpha = (a_sum + n / 2) / n;
                if a_sum == 0 {
                    out.extend_from_slice(&[0, 0, 0, 0]);
                } else {
                    for c in &acc[..3] {
                        out.push(((c + a_sum / 2) / a_sum).min(255) as u8);
                    }
                    out.push(alpha.min(255) as u8);
                }
            }
        }
        out
    };
    Some(CursorImage {
        width: out_w as u16,
        height: out_h as u16,
        hot_x: hx as u16,
        hot_y: hy as u16,
        rgba: out,
    })
}

/// The bundled arrow's outline, in arrow units with the tip at the origin: a
/// macOS-style pointer (straight left edge, notched tail).
const ARROW: [(f64, f64); 7] = [
    (0.0, 0.0),
    (0.0, 16.0),
    (4.0, 12.5),
    (6.8, 18.5),
    (9.0, 17.5),
    (6.3, 11.7),
    (11.3, 11.7),
];
/// White outline width around the black body, in arrow units.
const ARROW_OUTLINE: f64 = 1.1;
/// Margin so the outline fits, in arrow units; also where the tip lands.
const ARROW_MARGIN: f64 = 1.5;

fn inside(poly: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut odd = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            odd = !odd;
        }
        j = i;
    }
    odd
}

fn edge_distance(poly: &[(f64, f64)], x: f64, y: f64) -> f64 {
    let mut best = f64::MAX;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (ax, ay) = poly[j];
        let (bx, by) = poly[i];
        let (dx, dy) = (bx - ax, by - ay);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 {
            (((x - ax) * dx + (y - ay) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (px, py) = (ax + t * dx - x, ay + t * dy - y);
        best = best.min((px * px + py * py).sqrt());
        j = i;
    }
    best
}

/// The bundled arrow at `scale_q` (host display scale): black body, white
/// outline, antialiased by 4x4 coverage, hotspot on the tip.
pub fn static_arrow(scale_q: u32) -> CursorImage {
    let s = f64::from(scale_q.clamp(SCALE_MIN, SCALE_MAX)) / f64::from(SCALE_ONE);
    let (max_x, max_y) = ARROW
        .iter()
        .fold((0.0f64, 0.0f64), |(mx, my), &(x, y)| (mx.max(x), my.max(y)));
    let w = (((max_x + 2.0 * ARROW_MARGIN) * s).ceil() as u32).max(1);
    let h = (((max_y + 2.0 * ARROW_MARGIN) * s).ceil() as u32).max(1);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    const GRID: u32 = 4;
    for py in 0..h {
        for px in 0..w {
            let (mut black, mut white) = (0u32, 0u32);
            for sy in 0..GRID {
                for sx in 0..GRID {
                    let x = (f64::from(px) + (f64::from(sx) + 0.5) / f64::from(GRID)) / s
                        - ARROW_MARGIN;
                    let y = (f64::from(py) + (f64::from(sy) + 0.5) / f64::from(GRID)) / s
                        - ARROW_MARGIN;
                    if inside(&ARROW, x, y) {
                        // The body, less a thin white rim on its inner edge.
                        if edge_distance(&ARROW, x, y) < ARROW_OUTLINE * 0.35 {
                            white += 1;
                        } else {
                            black += 1;
                        }
                    } else if edge_distance(&ARROW, x, y) <= ARROW_OUTLINE * 0.65 {
                        white += 1;
                    }
                }
            }
            let covered = black + white;
            let n = GRID * GRID;
            if covered == 0 {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                let c = (255 * white / covered) as u8;
                rgba.extend_from_slice(&[c, c, c, (255 * covered / n) as u8]);
            }
        }
    }
    let hot = ((ARROW_MARGIN * s) as u32).min(w.min(h) - 1) as u16;
    CursorImage {
        width: w as u16,
        height: h as u16,
        hot_x: hot,
        hot_y: hot,
        rgba,
    }
}

/// Content key for [`static_arrow`] in the [`CursorView`] cache. Guest
/// content hashes are 64-bit `DefaultHasher` values; a collision with this
/// constant would at worst show the arrow for one glyph.
pub const STATIC_CONTENT: u64 = 0x5354_4154_4943_4152;

/// Number of built platform cursors kept per window.
const CACHE_CAP: usize = 8;

/// The window thread's cursor bookkeeping, generic over the platform cursor so
/// the decisions are testable without a window.
#[derive(Debug)]
pub struct CursorView<C> {
    /// What was last applied (or last attempted: a key that failed is not
    /// retried until the source or scale moves).
    pub applied: Option<CursorKey>,
    cache: Vec<((u64, u32), C)>,
    reported: [bool; 3],
}

impl<C: Clone> Default for CursorView<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Clone> CursorView<C> {
    pub fn new() -> Self {
        Self {
            applied: None,
            cache: Vec::new(),
            reported: [false; 3],
        }
    }

    /// True when `key` differs from what was last applied.
    pub fn needs(&self, key: CursorKey) -> bool {
        self.applied != Some(key)
    }

    /// A cursor already built for `(content, scale_q)`, refreshed to newest.
    pub fn cached(&mut self, content: u64, scale_q: u32) -> Option<C> {
        let at = self.cache.iter().position(|(k, _)| *k == (content, scale_q))?;
        let entry = self.cache.remove(at);
        let cursor = entry.1.clone();
        self.cache.insert(0, entry);
        Some(cursor)
    }

    /// Keep a built cursor, evicting the oldest past the cap.
    pub fn remember(&mut self, content: u64, scale_q: u32, cursor: C) {
        self.cache.retain(|(k, _)| *k != (content, scale_q));
        self.cache.insert(0, ((content, scale_q), cursor));
        self.cache.truncate(CACHE_CAP);
    }

    /// True the first time a mode label is reported (`static`, `guest`,
    /// `system`), so `host_window_cursor mode=` is one line per mode per run.
    pub fn first_report(&mut self, label: &str) -> bool {
        let i = match label {
            "static" => 0,
            "guest" => 1,
            _ => 2,
        };
        !std::mem::replace(&mut self.reported[i], true)
    }

    #[cfg(test)]
    fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argb_words_become_straight_rgba_bytes() {
        assert_eq!(
            argb_to_rgba(&[0x80_11_22_33, 0xff_00_00_ff, 0x00_ff_ff_ff]),
            vec![0x11, 0x22, 0x33, 0x80, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0]
        );
    }

    #[test]
    fn a_glyph_converts_once_and_refuses_bad_shapes() {
        let g = GuestCursorGlyph::from_argb(3, 2, 1, 1, 0, &[0xff_01_02_03, 0]).unwrap();
        assert_eq!(g.seq, 3);
        assert_eq!(g.rgba, vec![1, 2, 3, 0xff, 0, 0, 0, 0]);
        assert_eq!(
            GuestCursorGlyph::from_argb(1, 0, 1, 0, 0, &[]),
            Err(GlyphRefusal::Empty)
        );
        assert_eq!(
            GuestCursorGlyph::from_argb(1, 2, 2, 0, 0, &[0; 3]),
            Err(GlyphRefusal::LengthMismatch { have: 3, need: 4 })
        );
        assert_eq!(
            GuestCursorGlyph::from_argb(1, 2, 2, 2, 0, &[0; 4]),
            Err(GlyphRefusal::Hotspot)
        );
        let same = GuestCursorGlyph::from_argb(9, 2, 1, 1, 0, &[0xff_01_02_03, 0]).unwrap();
        assert_eq!(g.content, same.content, "content key ignores seq");
    }

    fn offer(slot: &CursorSlot, published: &mut u64, mode: CursorMode, serial: u64) -> CursorOffer {
        offer_cursor(slot, published, mode, serial, true, (1, 1, 0, 0), &[0xff_00_00_00])
    }

    #[test]
    fn offers_publish_once_per_serial() {
        let slot = new_slot();
        let mut published = 0;
        assert_eq!(offer(&slot, &mut published, CursorMode::Guest, 0), CursorOffer::Unchanged);
        assert_eq!(offer(&slot, &mut published, CursorMode::Guest, 1), CursorOffer::Published);
        assert_eq!(published, 1);
        assert_eq!(slot.lock().unwrap().as_ref().unwrap().seq, 1);
        assert_eq!(offer(&slot, &mut published, CursorMode::Guest, 1), CursorOffer::Unchanged);
        assert_eq!(offer(&slot, &mut published, CursorMode::Guest, 2), CursorOffer::Published);
        assert_eq!(slot.lock().unwrap().as_ref().unwrap().seq, 2);
    }

    /// The drain never waits on the window: a held slot skips, keeps the glyph
    /// pending, and the next offer publishes it.
    #[test]
    fn a_contended_slot_skips_and_retries_rather_than_blocking() {
        let slot = new_slot();
        let mut published = 0;
        {
            let _held = slot.lock().unwrap();
            assert_eq!(offer(&slot, &mut published, CursorMode::Guest, 5), CursorOffer::Skipped);
            assert_eq!(published, 0, "still pending");
        }
        assert_eq!(offer(&slot, &mut published, CursorMode::Guest, 5), CursorOffer::Published);
        assert_eq!(published, 5);
    }

    #[test]
    fn the_kill_switch_publishes_nothing_and_a_refusal_is_not_retried() {
        let slot = new_slot();
        let mut published = 0;
        assert_eq!(offer(&slot, &mut published, CursorMode::System, 1), CursorOffer::NotWanted);
        assert!(slot.lock().unwrap().is_none());
        assert_eq!(published, 1);
        let short = offer_cursor(&slot, &mut published, CursorMode::Guest, 2, true, (2, 2, 0, 0), &[0]);
        assert!(matches!(short, CursorOffer::Refused(GlyphRefusal::LengthMismatch { .. })));
        assert_eq!(published, 2, "a bad glyph is marked done, not retried per tranche");
        assert_eq!(
            offer_cursor(&slot, &mut published, CursorMode::Guest, 3, false, (1, 1, 0, 0), &[0]),
            CursorOffer::Unchanged,
            "a glyph that is not ready is not offered"
        );
    }

    #[test]
    fn mode_is_guest_unless_switched_off() {
        assert_eq!(mode_from(crate::config::Switch::Off), CursorMode::System);
        assert_eq!(mode_from(crate::config::Switch::Unset), CursorMode::Guest);
        assert_eq!(mode_from(crate::config::Switch::On), CursorMode::Guest);
        assert_eq!(mode_from(crate::config::Switch::Unrecognized), CursorMode::Guest);
    }

    #[test]
    fn the_scale_follows_the_presenter_viewport() {
        assert_eq!(viewport_scale_q((1920, 1080), None), SCALE_ONE);
        assert_eq!(viewport_scale_q((1920, 1080), Some((1920, 1080))), SCALE_ONE);
        // A 4K guest in a 1080p window is drawn at half size.
        assert_eq!(viewport_scale_q((1920, 1080), Some((3840, 2160))), SCALE_ONE / 2);
        // Letterboxed: 1440x1080 guest in 1920x1080 keeps full height, 1:1.
        assert_eq!(viewport_scale_q((1920, 1080), Some((1440, 1080))), SCALE_ONE);
        assert_eq!(scale_q(100.0), SCALE_ONE * 8);
        assert_eq!(scale_q(0.01), SCALE_ONE / 4);
        assert_eq!(scale_q(f64::NAN), SCALE_ONE);
    }

    #[test]
    fn a_new_seq_or_a_new_scale_changes_the_key() {
        let g1 = GuestCursorGlyph::from_argb(1, 1, 1, 0, 0, &[0xff_00_00_00]).unwrap();
        let g2 = GuestCursorGlyph::from_argb(2, 1, 1, 0, 0, &[0xff_00_00_00]).unwrap();
        let mut view: CursorView<u32> = CursorView::new();
        let static_key = desired_key(None, 32, SCALE_ONE);
        assert_eq!(static_key.source, CursorSource::Static);
        assert_eq!(static_key.scale_q, SCALE_ONE, "the arrow follows dpi, not the viewport");
        assert!(view.needs(static_key));
        view.applied = Some(static_key);
        assert!(!view.needs(static_key));
        let k1 = desired_key(Some(&g1), SCALE_ONE, SCALE_ONE);
        assert!(view.needs(k1));
        view.applied = Some(k1);
        assert!(!view.needs(desired_key(Some(&g1), SCALE_ONE, 128)));
        assert!(view.needs(desired_key(Some(&g1), 32, SCALE_ONE)), "resize rebuilds");
        assert!(view.needs(desired_key(Some(&g2), SCALE_ONE, SCALE_ONE)), "new glyph");
    }

    #[test]
    fn the_cache_is_keyed_by_content_and_scale_and_bounded() {
        let mut view: CursorView<u32> = CursorView::new();
        view.remember(1, 64, 10);
        view.remember(1, 32, 11);
        assert_eq!(view.cached(1, 64), Some(10));
        assert_eq!(view.cached(1, 32), Some(11));
        assert_eq!(view.cached(2, 64), None);
        for i in 0..20 {
            view.remember(100 + i, 64, i as u32);
        }
        assert_eq!(view.cache_len(), CACHE_CAP);
        assert_eq!(view.cached(1, 64), None, "oldest evicted");
        assert!(view.first_report("guest"));
        assert!(!view.first_report("guest"));
        assert!(view.first_report("static"));
    }

    #[test]
    fn scaling_is_exact_at_one_and_keeps_the_hotspot_inside() {
        let rgba: Vec<u8> = (0..4 * 4 * 4).map(|i| i as u8).collect();
        let same = scale_image(4, 4, 3, 3, &rgba, SCALE_ONE).unwrap();
        assert_eq!((same.width, same.height, same.hot_x, same.hot_y), (4, 4, 3, 3));
        assert_eq!(same.rgba, rgba);
        let half = scale_image(4, 4, 3, 3, &rgba, SCALE_ONE / 2).unwrap();
        assert_eq!((half.width, half.height), (2, 2));
        assert!(half.hot_x < 2 && half.hot_y < 2);
        assert_eq!(half.rgba.len(), 2 * 2 * 4);
        let double = scale_image(4, 4, 3, 3, &rgba, SCALE_ONE * 2).unwrap();
        assert_eq!((double.width, double.height, double.hot_x, double.hot_y), (8, 8, 6, 6));
        assert!(scale_image(0, 4, 0, 0, &rgba, SCALE_ONE).is_none());
        assert!(scale_image(8, 8, 0, 0, &rgba, SCALE_ONE).is_none(), "short input");
    }

    /// Premultiplied averaging: a transparent neighbour lowers alpha but does
    /// not darken an opaque white pixel's colour.
    #[test]
    fn transparent_pixels_do_not_bleed_colour_when_downscaling() {
        let rgba = vec![255, 255, 255, 255, 0, 0, 0, 0];
        let out = scale_image(2, 1, 0, 0, &rgba, SCALE_ONE / 2).unwrap();
        assert_eq!(out.width, 1);
        assert_eq!(&out.rgba[..3], &[255, 255, 255]);
        assert!(out.rgba[3] > 100 && out.rgba[3] < 160);
    }

    #[test]
    fn the_static_arrow_is_black_with_a_white_outline_and_a_tip_hotspot() {
        let a = static_arrow(SCALE_ONE);
        assert!(a.width >= 12 && a.height >= 18, "{}x{}", a.width, a.height);
        assert_eq!(a.rgba.len(), a.width as usize * a.height as usize * 4);
        assert_eq!((a.hot_x, a.hot_y), (1, 1));
        let px = |x: usize, y: usize| {
            let i = (y * a.width as usize + x) * 4;
            [a.rgba[i], a.rgba[i + 1], a.rgba[i + 2], a.rgba[i + 3]]
        };
        // Deep inside the body: opaque black.
        assert_eq!(px(3, 9), [0, 0, 0, 255]);
        // Far corner: transparent.
        assert_eq!(px(a.width as usize - 1, 0)[3], 0);
        // Just left of the straight edge: white outline.
        let edge = px(0, 9);
        assert!(edge[3] > 0 && edge[0] > 200, "{edge:?}");
        let big = static_arrow(SCALE_ONE * 2);
        assert_eq!((big.hot_x, big.hot_y), (3, 3));
        assert!(big.width > a.width);
    }
}
