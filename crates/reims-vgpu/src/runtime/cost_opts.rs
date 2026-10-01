//! Whether the exact per-packet cost reductions are on. See
//! [`crate::config::COST_OPTS`].

/// On unless the operator said `off`. Read once: the answer is asked on the hottest
/// path the device has, so it is a load and a branch.
#[inline]
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            crate::config::read(crate::config::COST_OPTS).0,
            crate::config::Switch::Off
        )
    })
}

#[cfg(test)]
thread_local! {
    static TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with [`on`] answering `state` on this thread.
#[cfg(test)]
pub(crate) fn with<R>(state: bool, f: impl FnOnce() -> R) -> R {
    let before = TEST.with(|c| c.replace(Some(state)));
    let out = f();
    TEST.with(|c| c.set(before));
    out
}
