//! TEMP (lab): a cheap clock for the census spans.
//!
//! A 5K drag profile (in-process drain sampler, 2026-10-01) put `Instant::now`
//! at ~5.7 % of the drain thread: the chain, draw, sampled and gather phase
//! spans take it a dozen times a draw, and on this host — Windows as the
//! Hyper-V root partition — a `QueryPerformanceCounter` read costs ~150 ns.
//! [`LabInstant`] reads the invariant TSC instead and converts ticks to
//! nanoseconds with a scale calibrated once against `Instant`. It is only for
//! spans whose sums are reported: nothing paces, times out or orders work by it.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// A point on the TSC, comparable within one process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LabInstant(u64);

impl LabInstant {
    #[inline]
    pub fn now() -> Self {
        Self(ticks())
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        Self::now().saturating_duration_since(*self)
    }

    #[inline]
    pub fn saturating_duration_since(&self, earlier: Self) -> Duration {
        Duration::from_nanos(to_ns(self.0.saturating_sub(earlier.0)))
    }

    #[inline]
    pub fn duration_since(&self, earlier: Self) -> Duration {
        self.saturating_duration_since(earlier)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn ticks() -> u64 {
    // SAFETY: RDTSC has no preconditions; every x86_64 CPU has it.
    unsafe { core::arch::x86_64::_rdtsc() }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn ticks() -> u64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// Nanoseconds per tick, Q32.
fn scale() -> u64 {
    static SCALE: OnceLock<u64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        if cfg!(not(target_arch = "x86_64")) {
            return 1 << 32;
        }
        let (i0, c0) = (Instant::now(), ticks());
        while i0.elapsed() < Duration::from_millis(5) {
            std::hint::spin_loop();
        }
        let (ns, c1) = (i0.elapsed().as_nanos() as u64, ticks());
        if c1 <= c0 {
            return 1 << 32;
        }
        (((ns as u128) << 32) / u128::from(c1 - c0)) as u64
    })
}

#[inline]
fn to_ns(t: u64) -> u64 {
    ((u128::from(t) * u128::from(scale())) >> 32) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lab_span_measures_what_instant_measures() {
        let (a, i) = (LabInstant::now(), Instant::now());
        std::thread::sleep(Duration::from_millis(20));
        let (lab, real) = (a.elapsed().as_micros() as i64, i.elapsed().as_micros() as i64);
        assert!((lab - real).abs() < real / 20 + 200, "lab {lab} us against {real} us");
    }
}
