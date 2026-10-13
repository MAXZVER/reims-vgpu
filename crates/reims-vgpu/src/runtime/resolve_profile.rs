//! Bounded sampled-source diagnostics, emitted in the sampled-phase window.
//! Linear, view and generic are parent spans; do not sum them with children.
//! PageWalk excludes guest byte copying. Unknown freshness still reads bytes.
use crate::observe::phase_clock::{charge_ns, to_us};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy)]
pub(crate) enum Part {
    Classify,
    Currency,
    Resident,
    View,
    Linear,
    Generic,
    Settle,
    Compare,
    Convert,
    HostCache,
    CurrencyVerdict,
    CurrencyObservation,
    WrittenPages,
    PageOffsets,
}
const PARTS: usize = Part::PageOffsets as usize + 1;
static NS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static CALLS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static READS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static WALK_READS: AtomicU64 = AtomicU64::new(0);
static WALK_BYTES: AtomicU64 = AtomicU64::new(0);
static WALK_NS: AtomicU64 = AtomicU64::new(0);
static COPY_NS: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
thread_local! { static ACTIVE: Cell<bool> = const { Cell::new(false) }; }
pub(crate) fn active() -> bool {
    ACTIVE.with(Cell::get)
}
pub(crate) struct Scope(bool);
impl Scope {
    pub(crate) fn open() -> Self {
        Self(ACTIVE.with(|v| v.replace(true)))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|v| v.set(self.0));
    }
}
pub(crate) struct Span {
    part: Part,
    started: Option<Instant>,
}
impl Span {
    pub(crate) fn open(part: Part) -> Self {
        Self {
            part,
            started: active().then(Instant::now),
        }
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            NS[self.part as usize].fetch_add(charge_ns(started.elapsed()), Ordering::Relaxed);
            CALLS[self.part as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
}
pub(crate) fn walk_read(bytes: usize) {
    if active() {
        WALK_READS.fetch_add(1, Ordering::Relaxed);
        WALK_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}
pub(crate) fn read_done(total_ns: u64, copy_ns: u64, bytes: usize, refused: bool) {
    READS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    WALK_NS.fetch_add(total_ns.saturating_sub(copy_ns), Ordering::Relaxed);
    COPY_NS.fetch_add(copy_ns, Ordering::Relaxed);
    REFUSED.fetch_add(u64::from(refused), Ordering::Relaxed);
}
pub(crate) fn emit() {
    const NAMES: [&str; PARTS] = [
        "classify",
        "currency",
        "resident",
        "view",
        "linear",
        "generic",
        "settle",
        "compare",
        "convert",
        "host_cache",
        "currency_verdict",
        "currency_observation",
        "written_pages",
        "page_offsets",
    ];
    let mut line = String::from("sampled_subphase");
    use std::fmt::Write as _;
    for (i, name) in NAMES.iter().enumerate() {
        let n = CALLS[i].swap(0, Ordering::Relaxed);
        let us = to_us(NS[i].swap(0, Ordering::Relaxed));
        let _ = write!(line, " {name}_n={n} {name}_us={us}");
    }
    for (name, counter) in [
        ("gva_reads", &READS),
        ("gva_bytes", &BYTES),
        ("walk_reads", &WALK_READS),
        ("walk_bytes", &WALK_BYTES),
        ("read_refused", &REFUSED),
    ] {
        let _ = write!(line, " {name}={}", counter.swap(0, Ordering::Relaxed));
    }
    let _ = write!(
        line,
        " walk_us={} copy_us={}",
        to_us(WALK_NS.swap(0, Ordering::Relaxed)),
        to_us(COPY_NS.swap(0, Ordering::Relaxed))
    );
    crate::observe::off(line);
}
