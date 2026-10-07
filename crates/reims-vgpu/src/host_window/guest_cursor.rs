//! The guest's hardware cursor on the host window ([`crate::config::HOST_CURSOR`]).
//!
//! The guest tells the display where its pointer is (the cursor position word in
//! the display's shared page, sampled at the cursor doorbell), whether it is
//! shown (`CmdDisplayCursorShow`), and what it looks like (`CmdDisplayCursorGlyph`).
//! Before this, all three went only to QEMU's console, which nobody shows, and
//! the arrow over the window was the host's own. That arrow agrees with the guest
//! only while the host pointer is what moves the guest's — the window forwards
//! its absolute position — and is missing entirely when anything else does: a
//! USB mouse or a Bluetooth pointer passed into the guest moves a cursor the host
//! cannot see.
//!
//! # Drawn by the host's cursor layer, not composited into the frame
//!
//! The window makes the guest's glyph its own cursor and, when the guest moved
//! the pointer itself, moves the host pointer to where the guest put it. The host
//! window system already owns a cursor plane: it is drawn without a present, on
//! top of whatever the swapchain holds, and never appears in a captured frame.
//! Compositing the glyph into the presented image instead would cost a present
//! per pointer move on an idle scene, add the guest's round trip to every host
//! pointer move, and need care to keep it out of the frame dumps.
//!
//! # Who moved the pointer
//!
//! Both sources must keep working: the host pointer forwarded as absolute moves
//! (the emulated tablet) and a pointer device the guest owns. The host pointer is
//! authoritative while it moves — the guest's position then trails it by the
//! guest's own round trip, and moving the host pointer back to that position
//! would fight the hand on the mouse. So the window moves the host pointer only
//! once the host pointer has been still for [`HOST_IDLE`], and drops the move
//! event its own placement generates ([`WarpGate`]) so that event neither reaches
//! the guest as input nor counts as the host pointer moving.
//!
//! # Ownership
//!
//! [`GuestCursorFeed`] is latest-wins state the device writes and the window
//! reads; one per device window, found by device id through [`publish_position`]
//! and [`publish_glyph`] so the device's MMIO and drain paths reach it without
//! taking any device lock (they already hold the device's own).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::present::WindowWakeHandle;

/// One cursor image from the guest, converted once to `0xAARRGGBB` words (the
/// console's layout, which the drain already produces).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestCursorGlyph {
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub argb: Vec<u32>,
}

/// What the window last needs to know, with a generation per half so it applies
/// only what changed.
#[derive(Debug, Clone, Default)]
pub struct CursorSnapshot {
    pub x: u16,
    pub y: u16,
    pub show: bool,
    /// Bumped on every position or visibility report; 0 until the guest's first.
    pub pos_gen: u64,
    pub glyph: Option<Arc<GuestCursorGlyph>>,
    /// Bumped on every glyph; 0 until the guest's first.
    pub glyph_gen: u64,
}

/// The device → window cursor channel of one device.
pub struct GuestCursorFeed {
    state: Mutex<CursorSnapshot>,
    wake: WindowWakeHandle,
}

pub type GuestCursorHandle = Arc<GuestCursorFeed>;

impl GuestCursorFeed {
    pub fn new(wake: WindowWakeHandle) -> GuestCursorHandle {
        Arc::new(Self {
            state: Mutex::new(CursorSnapshot::default()),
            wake,
        })
    }

    /// The guest reported its pointer. Wakes the window only when something it
    /// would act on changed: the cursor doorbell rings on every guest display
    /// interrupt, most of them with the pointer where it was.
    pub fn set_position(&self, x: u16, y: u16, show: bool) {
        let changed = match self.state.lock() {
            Ok(mut s) => {
                let changed = s.pos_gen == 0 || s.x != x || s.y != y || s.show != show;
                if changed {
                    s.x = x;
                    s.y = y;
                    s.show = show;
                    s.pos_gen = s.pos_gen.wrapping_add(1).max(1);
                }
                changed
            }
            Err(_) => false,
        };
        if changed {
            self.wake.wake();
        }
    }

    /// The guest changed the pointer's image.
    pub fn set_glyph(&self, glyph: GuestCursorGlyph) {
        let changed = match self.state.lock() {
            Ok(mut s) => {
                if s.glyph.as_deref() == Some(&glyph) {
                    false
                } else {
                    s.glyph = Some(Arc::new(glyph));
                    s.glyph_gen = s.glyph_gen.wrapping_add(1).max(1);
                    true
                }
            }
            Err(_) => false,
        };
        if changed {
            self.wake.wake();
        }
    }

    pub fn snapshot(&self) -> CursorSnapshot {
        self.state.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

/// Device id → feed, for the device paths that know only their id.
static FEEDS: Mutex<Vec<(u64, GuestCursorHandle)>> = Mutex::new(Vec::new());

/// Make `feed` the window cursor of device `id` (window start).
pub fn register(id: u64, feed: GuestCursorHandle) {
    if let Ok(mut feeds) = FEEDS.lock() {
        feeds.retain(|(owner, _)| *owner != id);
        feeds.push((id, feed));
    }
}

/// Forget device `id`'s window cursor (window stop).
pub fn unregister(id: u64) {
    if let Ok(mut feeds) = FEEDS.lock() {
        feeds.retain(|(owner, _)| *owner != id);
    }
}

fn feed_of(id: u64) -> Option<GuestCursorHandle> {
    let feeds = FEEDS.lock().ok()?;
    feeds
        .iter()
        .find(|(owner, _)| *owner == id)
        .map(|(_, f)| Arc::clone(f))
}

/// The guest reported its pointer: a no-op without a window for `id`.
pub fn publish_position(id: u64, x: u16, y: u16, show: bool) {
    if let Some(feed) = feed_of(id) {
        feed.set_position(x, y, show);
    }
}

/// The guest changed its pointer's image: a no-op without a window for `id`.
pub fn publish_glyph(id: u64, glyph: GuestCursorGlyph) {
    if let Some(feed) = feed_of(id) {
        feed.set_glyph(glyph);
    }
}

/// The largest cursor this module hands the window system (winit's own bound).
const MAX_CURSOR: u32 = 2048;

/// The glyph as non-premultiplied RGBA bytes at `scale` window pixels per guest
/// pixel, with its hot spot: `(rgba, width, height, hot_x, hot_y)`. `None` for
/// an empty glyph.
///
/// Nearest-neighbour: the guest draws its cursors for the pixel grid it renders
/// at, and an integer scale (2x for a 1080p mode on a 2160-line monitor) keeps
/// every edge sharp. The guest's words may be premultiplied; when no colour
/// channel exceeds its alpha anywhere in the glyph they are taken to be and are
/// unpremultiplied, because the window system takes straight alpha and a
/// premultiplied shadow read as straight comes out darker than the guest drew it.
pub fn glyph_rgba(glyph: &GuestCursorGlyph, scale: f64) -> Option<(Vec<u8>, u16, u16, u16, u16)> {
    let (w, h) = (u32::from(glyph.width), u32::from(glyph.height));
    if w == 0 || h == 0 || glyph.argb.len() < (w * h) as usize {
        return None;
    }
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let ow = ((f64::from(w) * scale).round() as u32).clamp(1, MAX_CURSOR);
    let oh = ((f64::from(h) * scale).round() as u32).clamp(1, MAX_CURSOR);
    let premultiplied = glyph.argb[..(w * h) as usize].iter().all(|&p| {
        let a = p >> 24;
        (p >> 16) & 0xff <= a && (p >> 8) & 0xff <= a && p & 0xff <= a
    });
    let mut rgba = Vec::with_capacity((ow * oh * 4) as usize);
    for oy in 0..oh {
        let sy = ((u64::from(oy) * u64::from(h)) / u64::from(oh)) as u32;
        for ox in 0..ow {
            let sx = ((u64::from(ox) * u64::from(w)) / u64::from(ow)) as u32;
            let p = glyph.argb[(sy * w + sx) as usize];
            let a = (p >> 24) as u8;
            let mut c = [(p >> 16) as u8, (p >> 8) as u8, p as u8];
            if premultiplied && a != 0 && a != 255 {
                for v in &mut c {
                    *v = ((u32::from(*v) * 255 + u32::from(a) / 2) / u32::from(a)).min(255) as u8;
                }
            }
            rgba.extend_from_slice(&[c[0], c[1], c[2], a]);
        }
    }
    let hx = ((f64::from(glyph.hot_x) * f64::from(ow) / f64::from(w)) as u32).min(ow - 1);
    let hy = ((f64::from(glyph.hot_y) * f64::from(oh) / f64::from(h)) as u32).min(oh - 1);
    Some((rgba, ow as u16, oh as u16, hx as u16, hy as u16))
}

/// How long the host pointer must have been still before the guest's position
/// may move it. Longer than the guest's round trip for a forwarded move (a few
/// milliseconds) so the guest's echo of the host's own motion never pulls the
/// pointer back; short enough that picking up a passed-through mouse after the
/// host one feels immediate.
pub const HOST_IDLE: Duration = Duration::from_millis(150);

/// A host move event within this many window pixels of the last placement is
/// that placement's own echo.
const ECHO_PX: f64 = 1.5;

/// How long a placement's echo is waited for.
const ECHO_TTL: Duration = Duration::from_millis(250);

/// Decides when the guest's position may move the host pointer, and recognises
/// the host move event a placement causes. Pure, so the rule is testable.
#[derive(Debug, Default)]
pub struct WarpGate {
    host_moved_at: Option<Instant>,
    placed: Option<((f64, f64), Instant)>,
}

impl WarpGate {
    /// A host pointer move arrived at `pos`. `false` when it is the echo of the
    /// window's own placement: not input, and not the host pointer moving.
    pub fn host_move(&mut self, pos: (f64, f64), now: Instant) -> bool {
        if let Some((target, at)) = self.placed {
            if now.duration_since(at) <= ECHO_TTL
                && (pos.0 - target.0).abs() <= ECHO_PX
                && (pos.1 - target.1).abs() <= ECHO_PX
            {
                self.placed = None;
                return false;
            }
        }
        self.host_moved_at = Some(now);
        true
    }

    /// Whether the guest's position `target` should move the host pointer, now
    /// at `current` (unknown before the first host event).
    pub fn should_place(
        &self,
        target: (f64, f64),
        current: Option<(f64, f64)>,
        now: Instant,
    ) -> bool {
        if self
            .host_moved_at
            .is_some_and(|at| now.duration_since(at) < HOST_IDLE)
        {
            return false;
        }
        match current {
            Some(cur) => (cur.0 - target.0).abs() > ECHO_PX || (cur.1 - target.1).abs() > ECHO_PX,
            None => true,
        }
    }

    /// The window moved the host pointer to `target`.
    pub fn placed(&mut self, target: (f64, f64), now: Instant) {
        self.placed = Some((target, now));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glyph(w: u16, h: u16, argb: Vec<u32>) -> GuestCursorGlyph {
        GuestCursorGlyph {
            width: w,
            height: h,
            hot_x: 1,
            hot_y: 1,
            argb,
        }
    }

    #[test]
    fn a_glyph_converts_to_straight_rgba_at_its_own_scale() {
        // Opaque red, transparent, half-alpha premultiplied grey (0x40 of 0x80).
        let g = glyph(3, 1, vec![0xffff_0000, 0x0000_0000, 0x8040_4040]);
        let (rgba, w, h, hx, hy) = glyph_rgba(&g, 1.0).unwrap();
        assert_eq!((w, h, hx, hy), (3, 1, 1, 0));
        assert_eq!(&rgba[0..4], &[255, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[0, 0, 0, 0]);
        // Unpremultiplied: 0x40 * 255 / 0x80 rounds to 128.
        assert_eq!(&rgba[8..12], &[128, 128, 128, 0x80]);
    }

    #[test]
    fn a_straight_alpha_glyph_is_left_alone() {
        // A colour channel above its alpha cannot be premultiplied.
        let g = glyph(2, 1, vec![0x80ff_ffff, 0xff00_00ff]);
        let (rgba, ..) = glyph_rgba(&g, 1.0).unwrap();
        assert_eq!(&rgba[0..4], &[255, 255, 255, 0x80]);
    }

    #[test]
    fn a_scaled_glyph_repeats_pixels_and_moves_its_hot_spot() {
        let g = GuestCursorGlyph {
            width: 2,
            height: 2,
            hot_x: 1,
            hot_y: 1,
            argb: vec![0xff00_0000, 0xffff_ffff, 0xffff_ffff, 0xff00_0000],
        };
        let (rgba, w, h, hx, hy) = glyph_rgba(&g, 2.0).unwrap();
        assert_eq!((w, h, hx, hy), (4, 4, 2, 2));
        // Row 0: black black white white.
        assert_eq!(
            &rgba[0..16],
            &[0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        );
    }

    #[test]
    fn the_host_pointer_wins_while_it_moves() {
        let t0 = Instant::now();
        let mut gate = WarpGate::default();
        assert!(gate.host_move((100.0, 100.0), t0));
        // The guest echoes the host's motion a few ms later: never placed.
        assert!(!gate.should_place(
            (120.0, 100.0),
            Some((100.0, 100.0)),
            t0 + Duration::from_millis(5)
        ));
        // Still host pointer, guest pointer moving on its own: placed.
        assert!(gate.should_place((300.0, 200.0), Some((100.0, 100.0)), t0 + HOST_IDLE));
    }

    #[test]
    fn a_placement_echo_is_not_host_motion() {
        let t0 = Instant::now();
        let mut gate = WarpGate::default();
        gate.placed((300.0, 200.0), t0);
        assert!(!gate.host_move((300.4, 199.6), t0 + Duration::from_millis(2)));
        // The echo did not restart the idle clock: the next guest move places.
        assert!(gate.should_place(
            (310.0, 205.0),
            Some((300.0, 200.0)),
            t0 + Duration::from_millis(3)
        ));
        // A real host move after it does.
        assert!(gate.host_move((500.0, 500.0), t0 + Duration::from_millis(4)));
        assert!(!gate.should_place(
            (310.0, 205.0),
            Some((500.0, 500.0)),
            t0 + Duration::from_millis(5)
        ));
    }

    #[test]
    fn the_feed_counts_only_what_changed() {
        let feed = GuestCursorFeed::new(super::super::present::WindowWaker::new());
        assert_eq!(feed.snapshot().pos_gen, 0);
        feed.set_position(10, 20, true);
        let first = feed.snapshot();
        assert_eq!(
            (first.x, first.y, first.show, first.pos_gen),
            (10, 20, true, 1)
        );
        // The doorbell rings with the pointer where it was: no new generation.
        feed.set_position(10, 20, true);
        assert_eq!(feed.snapshot().pos_gen, 1);
        // Hidden in place is a change.
        feed.set_position(10, 20, false);
        assert_eq!(feed.snapshot().pos_gen, 2);
        let g = glyph(1, 1, vec![0xffff_ffff]);
        feed.set_glyph(g.clone());
        feed.set_glyph(g);
        assert_eq!(feed.snapshot().glyph_gen, 1);
        feed.set_glyph(glyph(1, 1, vec![0xff00_0000]));
        assert_eq!(feed.snapshot().glyph_gen, 2);
    }

    #[test]
    fn a_published_cursor_reaches_only_its_own_devices_feed() {
        let a = GuestCursorFeed::new(super::super::present::WindowWaker::new());
        let b = GuestCursorFeed::new(super::super::present::WindowWaker::new());
        register(0xa11ce, Arc::clone(&a));
        register(0xb0b, Arc::clone(&b));
        publish_position(0xa11ce, 5, 6, true);
        assert_eq!(a.snapshot().pos_gen, 1);
        assert_eq!(b.snapshot().pos_gen, 0);
        unregister(0xa11ce);
        publish_position(0xa11ce, 7, 8, true);
        assert_eq!(
            a.snapshot().pos_gen,
            1,
            "an unregistered device reaches nothing"
        );
        unregister(0xb0b);
    }

    #[test]
    fn a_guest_position_already_under_the_pointer_is_not_placed_again() {
        let gate = WarpGate::default();
        assert!(!gate.should_place((100.0, 100.0), Some((100.5, 100.0)), Instant::now()));
    }
}
