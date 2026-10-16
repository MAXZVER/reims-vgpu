//! Stale-source hold for the host window's present.
//!
//! # What it replaces
//!
//! When the source a frame was published against has been invalidated by the
//! time the window presents it (`WINDOW_SOURCE_EPOCH` moved) and no CPU copy of
//! the frame is ready, the presenter used to acquire a swapchain image and clear
//! it to `SLATE_CLEAR`: one near-black 1920x1080 frame on screen until the next
//! publish. Measured on the v6 bump boot of 2026-10-15: 72 such frames in about
//! 17 minutes, each `frames=1`, on screen 5-59 ms (median 7). The A-on boot
//! had 25, on screen 14-151 ms. That is a flash the eye sees and a 30/60 fps
//! screen recording mostly does not.
//!
//! # What it does instead
//!
//! The verdict is taken **before** `acquire_next_image`, so a held present never
//! acquires an image it would then not present. A held present returns `Busy`,
//! which leaves the window's seq gate where it was: the previous swapchain image
//! stays on screen, and the next publish (or the next backstop retry) offers a
//! frame again. The hold is bounded by [`STALE_HOLD_MAX`]; past it the present
//! falls through to the existing slate arm, so a window can never be stuck
//! showing a held frame.
//!
//! `REIMS_VGPU_WINDOW_STALE_HOLD=0` restores the slate behaviour exactly.
#![cfg_attr(
    not(feature = "host-window"),
    allow(
        dead_code,
        reason = "the hold is consumed by the host-window presenter; the lib tests \
                  exercise it without that feature"
    )
)]

use std::time::{Duration, Instant};

/// How long one stale episode may hold the previous frame before the present
/// falls back to the slate arm.
///
/// Above the longest stale-to-fresh gap measured (151 ms, A-on boot) so a normal
/// episode never expires, and short enough that a source that never recovers is
/// answered within a quarter second rather than left on a frozen frame.
pub(crate) const STALE_HOLD_MAX: Duration = Duration::from_millis(250);

/// What the presenter does with the frame it was offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StaleVerdict {
    /// Not stale, or stale but a CPU copy covers it: present as before.
    Proceed,
    /// Stale with nothing to cover it: do not acquire, return `Busy`, keep the
    /// previous image on screen.
    Hold,
    /// The hold outlived [`STALE_HOLD_MAX`]: fall through to the slate arm.
    /// Slug `host_window_stale_hold_expired`.
    Expired { held_ms: u64 },
    /// The switch is off: fall through to the slate arm, the old behaviour.
    /// Slug `host_window_stale_hold_disabled`.
    Disabled,
}

/// Per-presenter hold state. Owned by the window thread; no lock.
#[derive(Debug, Default)]
pub(crate) struct StaleHold {
    /// When the episode in progress started holding.
    since: Option<Instant>,
    /// Whether the disabled slug has been reported this run.
    disabled_reported: bool,
}

impl StaleHold {
    /// Judge one offered frame.
    ///
    /// `stale` is the epoch comparison, `cpu_ready` whether a complete CPU copy
    /// of the frame could cover it. `now` is injected so the bound is testable.
    pub(crate) fn judge(
        &mut self,
        enabled: bool,
        stale: bool,
        cpu_ready: bool,
        now: Instant,
    ) -> StaleVerdict {
        if !stale || cpu_ready {
            // A fresh (or covered) frame ends the episode: the next stale one
            // starts a new hold rather than inheriting this one's age.
            self.since = None;
            return StaleVerdict::Proceed;
        }
        if !enabled {
            self.since = None;
            return StaleVerdict::Disabled;
        }
        let since = *self.since.get_or_insert(now);
        let held = now.saturating_duration_since(since);
        if held >= STALE_HOLD_MAX {
            // The slate this falls through to is a present, so the episode
            // ends with it; a later stale frame holds afresh.
            self.since = None;
            return StaleVerdict::Expired {
                held_ms: u64::try_from(held.as_millis()).unwrap_or(u64::MAX),
            };
        }
        StaleVerdict::Hold
    }

    /// True the first time it is asked, so the disabled slug is one line per
    /// run rather than one per stale frame.
    pub(crate) fn first_disabled_report(&mut self) -> bool {
        !std::mem::replace(&mut self.disabled_reported, true)
    }
}

/// Map the switch to whether the hold runs. Unset and on hold; off restores the
/// slate. An unrecognised value keeps the default and says so.
pub(crate) fn enabled_from(state: crate::config::Switch) -> bool {
    !matches!(state, crate::config::Switch::Off)
}

/// Read `REIMS_VGPU_WINDOW_STALE_HOLD` once.
pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let (state, value) = crate::config::read(crate::config::WINDOW_STALE_HOLD);
        if matches!(state, crate::config::Switch::Unrecognized) {
            crate::observe::fail(format!(
                "host_window_stale_hold reason=window_stale_hold_env_unrecognized value={}",
                value.unwrap_or_default()
            ));
        }
        enabled_from(state)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> Instant {
        Instant::now()
    }

    /// Stale with no CPU copy is held, not cleared.
    #[test]
    fn stale_without_cpu_copy_is_held() {
        let mut hold = StaleHold::default();
        let now = t0();
        assert_eq!(hold.judge(true, true, false, now), StaleVerdict::Hold);
        assert_eq!(
            hold.judge(true, true, false, now + Duration::from_millis(100)),
            StaleVerdict::Hold
        );
    }

    /// A covered stale frame and a fresh frame both present as before.
    #[test]
    fn fresh_or_covered_frames_proceed() {
        let mut hold = StaleHold::default();
        let now = t0();
        assert_eq!(hold.judge(true, false, false, now), StaleVerdict::Proceed);
        assert_eq!(hold.judge(true, true, true, now), StaleVerdict::Proceed);
    }

    /// The hold expires to the slate arm at the bound, and the next stale frame
    /// holds afresh rather than expiring immediately.
    #[test]
    fn the_hold_expires_to_slate_after_the_bound() {
        let mut hold = StaleHold::default();
        let now = t0();
        assert_eq!(hold.judge(true, true, false, now), StaleVerdict::Hold);
        let late = now + STALE_HOLD_MAX;
        assert_eq!(
            hold.judge(true, true, false, late),
            StaleVerdict::Expired {
                held_ms: STALE_HOLD_MAX.as_millis() as u64
            }
        );
        assert_eq!(
            hold.judge(true, true, false, late + Duration::from_millis(1)),
            StaleVerdict::Hold,
            "the slate ended the episode"
        );
    }

    /// Switch off: every stale frame goes to the slate arm, as before the hold.
    #[test]
    fn switch_off_restores_the_slate() {
        assert!(!enabled_from(crate::config::Switch::Off));
        assert!(enabled_from(crate::config::Switch::Unset));
        assert!(enabled_from(crate::config::Switch::On));
        assert!(enabled_from(crate::config::Switch::Unrecognized));
        let mut hold = StaleHold::default();
        let now = t0();
        assert_eq!(hold.judge(false, true, false, now), StaleVerdict::Disabled);
        assert_eq!(
            hold.judge(false, false, false, now),
            StaleVerdict::Proceed,
            "a fresh frame presents whatever the switch says"
        );
        assert!(hold.first_disabled_report());
        assert!(!hold.first_disabled_report());
    }

    /// A fresh publish after a hold presents, and resets the episode clock.
    #[test]
    fn a_fresh_publish_after_a_hold_presents() {
        let mut hold = StaleHold::default();
        let now = t0();
        assert_eq!(hold.judge(true, true, false, now), StaleVerdict::Hold);
        assert_eq!(
            hold.judge(true, false, false, now + Duration::from_millis(200)),
            StaleVerdict::Proceed
        );
        // 200 ms into the old episode plus 200 ms more would be past the
        // bound; a reset clock means this is a new hold, not an expiry.
        assert_eq!(
            hold.judge(true, true, false, now + Duration::from_millis(400)),
            StaleVerdict::Hold
        );
    }

    /// The verdict is taken before the acquire, so a held present never holds
    /// an acquired image it does not present. Pinned on the source because the
    /// presenter needs a device to drive.
    #[test]
    fn the_hold_is_judged_before_the_acquire() {
        const PRESENTER: &str = include_str!("window_present.rs");
        let begin = PRESENTER
            .find("pub(crate) unsafe fn begin_present(")
            .expect("begin_present exists");
        let body = &PRESENTER[begin..];
        let judge = body
            .find("self.stale_hold.judge(")
            .expect("begin_present judges the hold");
        let acquire = body
            .find("acquire_next_image(")
            .expect("begin_present acquires");
        assert!(judge < acquire, "the hold must be decided before the acquire");
        let hold_arm = body[judge..]
            .find("StaleVerdict::Hold =>")
            .expect("a hold arm");
        assert!(
            judge + hold_arm < acquire,
            "the hold arm returns before the acquire"
        );
        // The slate arm and its logging stay.
        assert!(body.contains("SlateReason::SourceStale"));
        assert!(PRESENTER.contains("const SLATE_CLEAR"));
    }
}