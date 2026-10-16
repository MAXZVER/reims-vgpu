//! Persistent `VkPipelineCache` across boots: a stable per-user location, a
//! validated load, and saves that never run on the drain worker.
//!
//! # Why this exists (v10, 2026-10-15)
//!
//! The v9 Tahoe boot logged `vk_pipeline_cache_load bytes=0` on every boot and
//! paid ~6.7 s of cold pipeline compile before the desktop, then 0.2-1.3 s more
//! on each first app open. The cache *was* being saved — five landed saves in
//! the v9 log, 15 KB at t=155 s growing to 7.26 MB at t=200 s — but at t=210 s
//! the in-memory cache reached 9.24 MB, past the 8 MiB
//! `PIPELINE_CACHE_MAX_WARM_BYTES` bound, and the legacy save
//! (`DeviceContext::persist_pipeline_cache_internal`) answered by **deleting**
//! the file (`persist::Job::Discard`). A `VkPipelineCache` never shrinks, so
//! every later save in that process also discarded, and the next boot found no
//! file: `NotFound` is the silent "first boot" arm of the load, so it read
//! `bytes=0` with no decline. Tahoe's single-boot working set is larger than the
//! macos-13/macos-15 working sets (3.4-3.85 MB) the bound was sized from, so the
//! discard-and-rebuild self-heal fired on *every* boot and the rail was
//! cold-only by construction. The file also lived in `%TEMP%`, which storage
//! cleaners may empty.
//!
//! # What this arm does instead
//!
//! - **Location**: `%LOCALAPPDATA%\reims-vgpu\pipeline-cache\` (XDG cache dir
//!   or `~/.cache` elsewhere, `temp_dir()` as a named last resort), file name
//!   `<vendorID>-<deviceID>-<driverVersion>-<pipelineCacheUUID>.bin`, so a
//!   driver update starts a new file instead of feeding the old blob in.
//! - **Load**: the `VkPipelineCacheHeaderVersionOne` header is checked field by
//!   field; a refusal is `pipeline_cache_load_rejected reason=..` and the boot
//!   continues with an empty cache. The size bound was unchanged in v10; v12
//!   raised it to 50 MiB (`PIPELINE_CACHE_MAX_WARM_BYTES` in `context.rs`)
//!   after the v11 Tahoe boot1 overflowed 8 MiB at ~100 s (9772133 bytes, kept
//!   6444408 at 627 pipelines). `REIMS_VGPU_PIPELINE_CACHE_MAX_MB` narrows it.
//! - **Over the bound, keep the last good snapshot** rather than delete it. A
//!   snapshot that landed under the bound is, on a cold boot, exactly this
//!   workload's own working-set prefix (boot + early desktop), which is what the
//!   bound's measurements say is worth warming from. The stale-blob self-heal
//!   the legacy discard existed for is kept for the case it targets: a *warm*
//!   boot whose own new compiles outgrow what it loaded (the loaded blob covered
//!   less than half of what the boot needed) discards, so the next boot
//!   rebuilds. Both are named (`pipeline_cache_save_skipped reason=..`).
//! - **Saves off the drain thread**: a pipeline creation only bumps an atomic
//!   counter. The one persistence thread wakes every
//!   [`SAVE_MIN_INTERVAL`], and only when pipelines were created since its last
//!   save does it call `vkGetPipelineCacheData` (the pipeline cache is
//!   internally synchronized; no external-sync requirement applies to it unless
//!   it is created with `EXTERNALLY_SYNCHRONIZED`, which this device never
//!   does) and write the file atomically (unique temp file, then rename). The
//!   drain never takes the Engine lock for a save, never serializes the cache,
//!   and never touches the disk.
//! - **Teardown**: `DeviceContext::destroy` closes the source's gate (waiting
//!   for at most one in-flight serialize), takes a final snapshot if anything
//!   was created since the last save, and hands the bytes to the persistence
//!   thread. A QEMU that is killed never reaches destroy; the periodic save is
//!   what covers that, losing at most the last interval of compiles.
//!
//! # v13: no compiled bound (`REIMS_VGPU_PIPELINE_CACHE_UNBOUNDED`, default on)
//!
//! Nour asked for the cache to grow without the fixed 50 MiB bound. With the
//! switch on, the cap handed to [`Source`] is `usize::MAX` unless
//! `REIMS_VGPU_PIPELINE_CACHE_MAX_MB` sets a lower limit, and the
//! [`SavePolicy::V13`] policy applies: over a limit the last good file is kept
//! and **never deleted** (the discard-stale verdict is downgraded to keep-last,
//! named `stale_discard_suppressed=1`), and a save that would leave less than
//! [`DISK_FREE_FLOOR`] free on the cache's drive is skipped
//! (`pipeline_cache_save_skipped reason=low_disk_keep_last`), keeping the last
//! good file. `=0` restores the v12 bound and policy exactly
//! (`pipeline_cache_unbounded_disabled`).
//!
//! `REIMS_VGPU_PIPELINE_CACHE_PERSIST=0` keeps the legacy arm exactly
//! (`%TEMP%` path, save-on-create under the Engine lock, discard over the
//! bound) and logs `pipeline_cache_persist_disabled` once.

use ash::vk;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Minimum spacing between two saves of one cache, and the persistence
/// thread's wake period. A compile storm therefore costs at most one
/// serialize + write per interval, on the persistence thread.
pub(crate) const SAVE_MIN_INTERVAL: Duration = Duration::from_secs(10);

/// `sizeof(VkPipelineCacheHeaderVersionOne)`.
const HEADER_ONE_LEN: usize = 32;

/// Read `REIMS_VGPU_PIPELINE_CACHE_PERSIST` once. Unset/on runs this arm; off
/// restores the legacy arm and says so once. An unrecognised value keeps the
/// default and says so.
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let (state, value) = crate::config::read(crate::config::PIPELINE_CACHE_PERSIST);
        match state {
            crate::config::Switch::Off => {
                crate::observe::fail(
                    "pipeline_cache_persist_disabled reason=disabled_by_env (legacy temp-dir arm: save on create, discard over cap)",
                );
                false
            }
            crate::config::Switch::Unrecognized => {
                crate::observe::fail(format!(
                    "pipeline_cache_persist reason=pipeline_cache_persist_env_unrecognized value={}",
                    value.unwrap_or_default()
                ));
                true
            }
            _ => true,
        }
    })
}

/// v13: the least free space a save may leave on the cache's drive, counting
/// the bytes it is about to write (the atomic write holds a temp copy beside
/// the old file until the rename).
pub(crate) const DISK_FREE_FLOOR: u64 = 1 << 30;

/// v13: ceiling for `REIMS_VGPU_PIPELINE_CACHE_MAX_MB` when the compiled bound
/// is gone (1 TiB in MiB). The count still refuses 0 and junk by name.
pub(crate) const UNBOUNDED_MAX_MB_CEILING: u64 = 1 << 20;

/// Read `REIMS_VGPU_PIPELINE_CACHE_UNBOUNDED` once. Unset/on: no compiled
/// bound (v13). Off: the v12 bound and policy, said once.
pub(crate) fn unbounded_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let (state, value) = crate::config::read(crate::config::PIPELINE_CACHE_UNBOUNDED);
        match state {
            crate::config::Switch::Off => {
                crate::observe::fail(
                    "pipeline_cache_unbounded_disabled reason=disabled_by_env (v12 50 MiB bound, keep-last/discard-stale, no free-space check)",
                );
                false
            }
            crate::config::Switch::Unrecognized => {
                crate::observe::fail(format!(
                    "pipeline_cache_unbounded reason=pipeline_cache_unbounded_env_unrecognized value={} (kept on)",
                    value.unwrap_or_default()
                ));
                true
            }
            _ => true,
        }
    })
}

/// The persistence arm's cap with the compiled bound gone: an optional lower
/// limit from `REIMS_VGPU_PIPELINE_CACHE_MAX_MB`, else unbounded. Pure over
/// the parsed count so the rule is testable. Returns (cap, source slug,
/// refused raw value if any).
pub(crate) fn unbounded_cap_from(count: crate::config::Count) -> (usize, &'static str, Option<String>) {
    const MIB: usize = 1024 * 1024;
    match count {
        crate::config::Count::Narrowed(mb) => (
            usize::try_from(mb).ok().and_then(|mb| mb.checked_mul(MIB)).unwrap_or(usize::MAX),
            "env_lower_limit",
            None,
        ),
        crate::config::Count::Unset => (usize::MAX, "unbounded", None),
        crate::config::Count::Refused(raw) => (usize::MAX, "unbounded_after_refusal", Some(raw)),
    }
}

/// How a cap reads in a log line.
pub(crate) fn cap_text(cap: usize) -> String {
    if cap == usize::MAX {
        "unbounded".to_string()
    } else {
        cap.to_string()
    }
}

/// What a [`Source`] does at and over its cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SavePolicy {
    /// v12: a stale warm blob over the cap is deleted. v13: never.
    pub discard_on_oversize: bool,
    /// v13: skip a save that would leave less than this free on the drive.
    pub disk_free_floor: Option<u64>,
}

impl SavePolicy {
    /// The v10..v12 policy (and the `REIMS_VGPU_PIPELINE_CACHE_UNBOUNDED=0` arm).
    pub(crate) const V12: Self = Self {
        discard_on_oversize: true,
        disk_free_floor: None,
    };
    /// v13 default: never delete on oversize, keep 1 GiB free.
    pub(crate) const V13: Self = Self {
        discard_on_oversize: false,
        disk_free_floor: Some(DISK_FREE_FLOOR),
    };
}

/// [`decide_save`]'s verdict under a policy: a policy that never deletes turns
/// discard-stale into keep-last. Pure.
pub(crate) fn apply_policy(decision: SaveDecision, policy: SavePolicy) -> SaveDecision {
    if decision == SaveDecision::OverCapDiscardStale && !policy.discard_on_oversize {
        SaveDecision::OverCapKeepLast
    } else {
        decision
    }
}

/// Free-space verdict for one save. Pure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DiskVerdict {
    /// Enough room left after the write.
    Room,
    /// Writing would leave less than the floor; skip, keep the last file.
    Low { free: u64 },
    /// The free space could not be read; the save goes ahead, said once.
    Unknown,
}

pub(crate) fn disk_verdict(free: Option<u64>, len: usize, floor: u64) -> DiskVerdict {
    match free {
        None => DiskVerdict::Unknown,
        Some(f) if f.saturating_sub(len as u64) < floor => DiskVerdict::Low { free: f },
        Some(_) => DiskVerdict::Room,
    }
}

/// Bytes available to this user on the drive holding `path` (its nearest
/// existing ancestor), or `None` when that cannot be read.
pub(crate) fn free_bytes_near(path: &Path) -> Option<u64> {
    let probe = path.ancestors().find(|a| a.is_dir())?;
    free_bytes_at(probe)
}

#[cfg(windows)]
fn free_bytes_at(dir: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt as _;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(
            directory: *const u16,
            free_to_caller: *mut u64,
            total: *mut u64,
            total_free: *mut u64,
        ) -> i32;
    }
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut avail = 0u64;
    // SAFETY: `wide` is NUL-terminated and outlives the call; the two optional
    // out-pointers may be null per the API contract.
    let ok = unsafe {
        GetDiskFreeSpaceExW(wide.as_ptr(), &mut avail, std::ptr::null_mut(), std::ptr::null_mut())
    };
    (ok != 0).then_some(avail)
}

#[cfg(not(windows))]
fn free_bytes_at(_dir: &Path) -> Option<u64> {
    None
}

// ---------------------------------------------------------------------------
// Path derivation (pure)
// ---------------------------------------------------------------------------

/// Where the base directory came from. `TempFallback` is fail-visible: a cache
/// in the temp dir may be cleaned between boots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DirSource {
    LocalAppData,
    XdgCache,
    HomeCache,
    TempFallback,
}

/// Pick the cache directory from the given environment values. Pure so the
/// precedence is testable without touching the process environment.
pub(crate) fn cache_dir_from(
    local_appdata: Option<&Path>,
    xdg_cache_home: Option<&Path>,
    home: Option<&Path>,
    temp: &Path,
) -> (PathBuf, DirSource) {
    let usable = |p: Option<&Path>| p.filter(|p| p.is_absolute()).map(Path::to_path_buf);
    let tail = |base: PathBuf| base.join("reims-vgpu").join("pipeline-cache");
    if let Some(base) = usable(local_appdata) {
        return (tail(base), DirSource::LocalAppData);
    }
    if let Some(base) = usable(xdg_cache_home) {
        return (tail(base), DirSource::XdgCache);
    }
    if let Some(base) = usable(home) {
        return (tail(base.join(".cache")), DirSource::HomeCache);
    }
    (tail(temp.to_path_buf()), DirSource::TempFallback)
}

/// The process's cache directory, from the live environment.
pub(crate) fn cache_dir() -> (PathBuf, DirSource) {
    let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
    let local = if cfg!(windows) { var("LOCALAPPDATA") } else { None };
    cache_dir_from(
        local.as_deref(),
        var("XDG_CACHE_HOME").as_deref(),
        var("HOME").as_deref().or(var("USERPROFILE").as_deref()),
        &std::env::temp_dir(),
    )
}

/// `<vendorID>-<deviceID>-<driverVersion>-<pipelineCacheUUID hex>.bin`, all
/// lowercase hex, fixed width. The driver version is part of the key so a
/// driver update never even opens the old file.
pub(crate) fn cache_file_name(vendor: u32, device: u32, driver: u32, uuid: &[u8; 16]) -> String {
    use std::fmt::Write as _;
    let mut name = format!("{vendor:04x}-{device:04x}-{driver:08x}-");
    for b in uuid {
        let _ = write!(name, "{b:02x}");
    }
    name.push_str(".bin");
    name
}

// ---------------------------------------------------------------------------
// Header validation (pure)
// ---------------------------------------------------------------------------

/// Why a blob's `VkPipelineCacheHeaderVersionOne` was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeaderReject {
    Short,
    HeaderSize,
    HeaderVersion,
    Vendor,
    Device,
    Uuid,
}

impl HeaderReject {
    pub(crate) fn reason(self) -> &'static str {
        match self {
            Self::Short => "header_short",
            Self::HeaderSize => "header_size",
            Self::HeaderVersion => "header_version",
            Self::Vendor => "vendor_mismatch",
            Self::Device => "device_mismatch",
            Self::Uuid => "uuid_mismatch",
        }
    }
}

/// Check every field of the header against the live device. Feeding a blob
/// that `vkGetPipelineCacheData` did not produce for this device to
/// `vkCreatePipelineCache` is a valid-usage violation, so this is not left to
/// the driver.
pub(crate) fn check_header(
    blob: &[u8],
    vendor: u32,
    device: u32,
    uuid: &[u8; 16],
) -> Result<(), HeaderReject> {
    if blob.len() < HEADER_ONE_LEN {
        return Err(HeaderReject::Short);
    }
    let u32le = |off: usize| u32::from_le_bytes(blob[off..off + 4].try_into().unwrap());
    let header_size = u32le(0) as usize;
    if header_size < HEADER_ONE_LEN || header_size > blob.len() {
        return Err(HeaderReject::HeaderSize);
    }
    if u32le(4) != vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32 {
        return Err(HeaderReject::HeaderVersion);
    }
    if u32le(8) != vendor {
        return Err(HeaderReject::Vendor);
    }
    if u32le(12) != device {
        return Err(HeaderReject::Device);
    }
    if blob[16..HEADER_ONE_LEN] != uuid[..] {
        return Err(HeaderReject::Uuid);
    }
    Ok(())
}

/// Outcome of the boot-time load, for the caller's log line.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LoadOutcome {
    /// Header-valid blob within the bound.
    Loaded(Vec<u8>),
    /// No file: first boot on this device/driver.
    Absent,
    /// Refused; start empty. `reason` is the slug suffix.
    Rejected { reason: String, bytes: usize },
}

/// Read and validate the blob at `path`. Pure w.r.t. Vulkan.
pub(crate) fn load_blob(
    path: &Path,
    vendor: u32,
    device: u32,
    uuid: &[u8; 16],
    cap: usize,
) -> LoadOutcome {
    match std::fs::read(path) {
        Ok(blob) if blob.len() > cap => LoadOutcome::Rejected {
            reason: format!("too_large cap={cap}"),
            bytes: blob.len(),
        },
        Ok(blob) => match check_header(&blob, vendor, device, uuid) {
            Ok(()) => LoadOutcome::Loaded(blob),
            Err(why) => LoadOutcome::Rejected {
                reason: why.reason().to_string(),
                bytes: blob.len(),
            },
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => LoadOutcome::Absent,
        Err(e) => LoadOutcome::Rejected {
            reason: format!(
                "read errno={} io_kind={:?}",
                e.raw_os_error().map_or_else(|| "none".into(), |v| v.to_string()),
                e.kind()
            ),
            bytes: 0,
        },
    }
}

// ---------------------------------------------------------------------------
// Throttle + save decision (pure)
// ---------------------------------------------------------------------------

/// Per-cache save bookkeeping, touched only by the persistence thread (and by
/// `close` at teardown, briefly).
#[derive(Clone, Debug, Default)]
pub(crate) struct SaveThrottle {
    /// When the last snapshot was taken (landed, skipped or failed).
    pub last_attempt: Option<Instant>,
    /// Creation count at the last snapshot.
    pub created_at_last: u64,
    /// Length of the blob that last landed on disk (or was loaded).
    pub landed_len: usize,
    /// The cache crossed the bound; it only grows, so no later snapshot can
    /// come back under it. No more snapshots this process.
    pub frozen: bool,
}

/// What the throttle says about one wake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tick {
    /// Nothing new since the last snapshot.
    Idle,
    /// New pipelines, but the last snapshot was too recent.
    TooSoon,
    /// The cache is frozen over the bound.
    Frozen,
    /// Take a snapshot now.
    SaveNow,
}

impl SaveThrottle {
    pub(crate) fn decide(&self, now: Instant, created: u64, min_interval: Duration) -> Tick {
        if self.frozen {
            return Tick::Frozen;
        }
        if created <= self.created_at_last {
            return Tick::Idle;
        }
        match self.last_attempt {
            Some(prev) if now.saturating_duration_since(prev) < min_interval => Tick::TooSoon,
            _ => Tick::SaveNow,
        }
    }
}

/// What to do with one snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SaveDecision {
    /// Land these bytes.
    Write,
    /// Same length as what is on disk: nothing new to land.
    Unchanged,
    /// Over the bound on a cold (or barely-warmed) boot: keep the last landed
    /// snapshot, which is this workload's own working-set prefix.
    OverCapKeepLast,
    /// Over the bound on a warm boot that compiled more than it loaded: the
    /// loaded blob is mostly stale, so delete it and let the next boot rebuild.
    OverCapDiscardStale,
    /// v13: under the cap, but writing would leave less than
    /// [`DISK_FREE_FLOOR`] free on the drive. The last good file is kept.
    /// Never returned by [`decide_save`]; `Source::land` produces it.
    LowDiskKeepLast,
}

impl SaveDecision {
    pub(crate) fn reason(self) -> &'static str {
        match self {
            Self::Write => "write",
            Self::Unchanged => "unchanged",
            Self::OverCapKeepLast => "over_cap_keep_last",
            Self::OverCapDiscardStale => "over_cap_discard_stale",
            Self::LowDiskKeepLast => "low_disk_keep_last",
        }
    }
}

pub(crate) fn decide_save(initial_len: usize, landed_len: usize, len: usize, cap: usize) -> SaveDecision {
    if len > cap {
        // Growth this boot beyond what it loaded. A warm blob that covered the
        // workload grows little; one that grows by more than its own size was
        // mostly dead weight (a translator change, another build's shaders).
        let grown = len.saturating_sub(initial_len);
        if initial_len > 0 && grown > initial_len {
            return SaveDecision::OverCapDiscardStale;
        }
        return SaveDecision::OverCapKeepLast;
    }
    if len == landed_len {
        return SaveDecision::Unchanged;
    }
    SaveDecision::Write
}

// ---------------------------------------------------------------------------
// Atomic write (fs only)
// ---------------------------------------------------------------------------

/// Write `data` to a temp file unique to this process and call, flush it to
/// disk, then rename it over `path`. Creates the directory. On any failure the
/// temp file is removed and `path` is untouched. Returns the failing stage.
pub(crate) fn write_atomic(path: &Path, data: &[u8]) -> Result<(), (&'static str, std::io::Error)> {
    use std::io::Write as _;
    static SEQ: AtomicU64 = AtomicU64::new(0);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| ("create_dir", e))?;
    }
    let tmp = path.with_extension(format!(
        "tmp.{}.{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut f = std::fs::File::create(&tmp).map_err(|e| ("write", e))?;
        f.write_all(data).map_err(|e| ("write", e))?;
        f.sync_all().map_err(|e| ("sync", e))?;
        drop(f);
        std::fs::rename(&tmp, path).map_err(|e| ("rename", e))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn io_fields(e: &std::io::Error) -> String {
    format!(
        "errno={} io_kind={:?}",
        e.raw_os_error().map_or_else(|| "none".into(), |v| v.to_string()),
        e.kind()
    )
}

// ---------------------------------------------------------------------------
// Live source + the persistence thread
// ---------------------------------------------------------------------------

/// One device's pipeline cache, as the persistence thread sees it.
pub(crate) struct Source {
    device: ash::Device,
    cache: vk::PipelineCache,
    path: PathBuf,
    cap: usize,
    initial_len: usize,
    /// Bumped by the drain on every real pipeline creation. The only thing the
    /// hot path touches.
    created: AtomicU64,
    /// `true` while `cache` may be read. Held across `vkGetPipelineCacheData`
    /// so teardown cannot destroy the cache mid-serialize.
    alive: Mutex<bool>,
    throttle: Mutex<SaveThrottle>,
    /// v13: what to do at and over the cap ([`SavePolicy`]).
    policy: SavePolicy,
}

impl Source {
    /// The v10..v12 constructor: [`SavePolicy::V12`].
    #[allow(dead_code)]
    pub(crate) fn new(
        device: ash::Device,
        cache: vk::PipelineCache,
        path: PathBuf,
        cap: usize,
        initial_len: usize,
    ) -> Arc<Self> {
        Self::new_with_policy(device, cache, path, cap, initial_len, SavePolicy::V12)
    }

    /// v13: as [`Self::new`] with an explicit [`SavePolicy`].
    pub(crate) fn new_with_policy(
        device: ash::Device,
        cache: vk::PipelineCache,
        path: PathBuf,
        cap: usize,
        initial_len: usize,
        policy: SavePolicy,
    ) -> Arc<Self> {
        Arc::new(Self {
            device,
            cache,
            path,
            cap,
            initial_len,
            created: AtomicU64::new(0),
            alive: Mutex::new(true),
            throttle: Mutex::new(SaveThrottle {
                landed_len: initial_len,
                ..SaveThrottle::default()
            }),
            policy,
        })
    }

    /// Hot path: a pipeline was created. One relaxed atomic add; no lock, no
    /// serialize, no disk.
    pub(crate) fn note_created(&self) {
        self.created.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> Option<Result<Vec<u8>, vk::Result>> {
        let alive = self.alive.lock().unwrap_or_else(|p| p.into_inner());
        if !*alive {
            return None;
        }
        // SAFETY: `alive` is held, and `close` clears it under the same lock
        // before the cache or device is destroyed.
        Some(unsafe { self.device.get_pipeline_cache_data(self.cache) })
    }

    /// Teardown: close the gate (waits for at most one in-flight serialize)
    /// and, when pipelines were created since the last snapshot and the cache
    /// is not frozen, take a final snapshot and hand it to the persistence
    /// thread. Never touches the disk on the caller's thread. Must be called
    /// before `vkDestroyPipelineCache`.
    pub(crate) fn close(self: &Arc<Self>) {
        let created = self.created.load(Ordering::Relaxed);
        let (frozen, created_at_last) = {
            let t = self.throttle.lock().unwrap_or_else(|p| p.into_inner());
            (t.frozen, t.created_at_last)
        };
        let mut alive = self.alive.lock().unwrap_or_else(|p| p.into_inner());
        let final_blob = if *alive && !frozen && created > created_at_last {
            // SAFETY: gate held and still open.
            Some(unsafe { self.device.get_pipeline_cache_data(self.cache) })
        } else {
            None
        };
        *alive = false;
        drop(alive);
        match final_blob {
            Some(Ok(data)) => owner().submit_final(Arc::clone(self), data, created),
            Some(Err(result)) => crate::observe::fail(format!(
                "pipeline_cache_save_failed reason=get_data vk_result={} site=teardown",
                result.to_string().replace(char::is_whitespace, "_")
            )),
            None => crate::observe::off(format!(
                "pipeline_cache_save_skipped reason=teardown_nothing_new frozen={frozen} pipelines={created}"
            )),
        }
    }

    /// Persistence thread: apply one snapshot's decision. Returns the decision
    /// for tests and logs.
    fn land(&self, data: &[u8], created: u64, site: &str) -> SaveDecision {
        let mut t = self.throttle.lock().unwrap_or_else(|p| p.into_inner());
        let raw_decision = decide_save(self.initial_len, t.landed_len, data.len(), self.cap);
        // v13: a never-delete policy downgrades discard-stale to keep-last.
        let decision = apply_policy(raw_decision, self.policy);
        let discard_suppressed = decision != raw_decision;
        t.created_at_last = t.created_at_last.max(created);
        match decision {
            SaveDecision::Unchanged => {}
            SaveDecision::LowDiskKeepLast => {}
            SaveDecision::Write => {
                // Drop the bookkeeping lock across the disk write: `close`
                // only needs it briefly, and must never wait on the disk.
                let landed = t.landed_len;
                drop(t);
                // v13: keep the drive from filling. Skipped, not frozen: the
                // next tick with new pipelines asks again.
                if let Some(floor) = self.policy.disk_free_floor {
                    match disk_verdict(free_bytes_near(&self.path), data.len(), floor) {
                        DiskVerdict::Room => {}
                        DiskVerdict::Low { free } => {
                            let line = format!(
                                "pipeline_cache_save_skipped reason={} bytes={} free={free} floor={floor} kept={landed} pipelines={created} site={site} path={}",
                                SaveDecision::LowDiskKeepLast.reason(),
                                data.len(),
                                self.path.display()
                            );
                            if crate::observe::first_sight("pipeline_cache_low_disk", 0) {
                                crate::observe::fail(line);
                            } else {
                                crate::observe::off(line);
                            }
                            return SaveDecision::LowDiskKeepLast;
                        }
                        DiskVerdict::Unknown => {
                            if crate::observe::first_sight("pipeline_cache_free_space_unknown", 0) {
                                crate::observe::fail(format!(
                                    "pipeline_cache_free_space_unknown path={} (saving without the free-space check)",
                                    self.path.display()
                                ));
                            }
                        }
                    }
                }
                match write_atomic(&self.path, data) {
                    Ok(()) => {
                        self.throttle
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .landed_len = data.len();
                        crate::observe::off(format!(
                            "pipeline_cache_saved bytes={} pipelines={created} site={site} path={}",
                            data.len(),
                            self.path.display()
                        ));
                    }
                    Err((stage, e)) => {
                        if crate::observe::first_sight("pipeline_cache_save_failed", 0) {
                            crate::observe::fail(format!(
                                "pipeline_cache_save_failed reason={stage} {} bytes={} path={}",
                                io_fields(&e),
                                data.len(),
                                self.path.display()
                            ));
                        } else {
                            crate::observe::off(format!(
                                "pipeline_cache_save_failed reason={stage} {} bytes={}",
                                io_fields(&e),
                                data.len()
                            ));
                        }
                    }
                }
                return decision;
            }
            SaveDecision::OverCapKeepLast | SaveDecision::OverCapDiscardStale => {
                t.frozen = true;
                let landed = t.landed_len;
                drop(t);
                let mut line = format!(
                    "pipeline_cache_save_skipped reason={} bytes={} cap={} loaded={} kept={landed} pipelines={created}",
                    decision.reason(),
                    data.len(),
                    cap_text(self.cap),
                    self.initial_len,
                );
                if discard_suppressed {
                    line.push_str(" stale_discard_suppressed=1 (v13: oversize never deletes the file)");
                }
                if decision == SaveDecision::OverCapDiscardStale {
                    match std::fs::remove_file(&self.path) {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => line.push_str(&format!(" discard_failed {}", io_fields(&e))),
                    }
                    self.throttle
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .landed_len = 0;
                }
                crate::observe::fail(line);
            }
        }
        decision
    }

    /// Persistence thread: one periodic wake for this source.
    fn tick(&self, now: Instant) {
        let created = self.created.load(Ordering::Relaxed);
        let verdict = self
            .throttle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .decide(now, created, SAVE_MIN_INTERVAL);
        if verdict != Tick::SaveNow {
            return;
        }
        self.throttle
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .last_attempt = Some(now);
        match self.snapshot() {
            None => {}
            Some(Ok(data)) => {
                self.land(&data, created, "periodic");
            }
            Some(Err(result)) => {
                self.throttle
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .created_at_last = created;
                if crate::observe::first_sight("pipeline_cache_save_failed", 1) {
                    crate::observe::fail(format!(
                        "pipeline_cache_save_failed reason=get_data vk_result={} site=periodic",
                        result.to_string().replace(char::is_whitespace, "_")
                    ));
                }
            }
        }
    }

    fn is_alive(&self) -> bool {
        *self.alive.lock().unwrap_or_else(|p| p.into_inner())
    }
}

struct OwnerState {
    sources: Vec<Arc<Source>>,
    finals: Vec<(Arc<Source>, Vec<u8>, u64)>,
}

/// The one thread that serializes and writes pipeline caches for this arm.
pub(crate) struct Owner {
    state: Mutex<OwnerState>,
    wake: Condvar,
}

fn owner() -> &'static Owner {
    static OWNER: OnceLock<Owner> = OnceLock::new();
    OWNER.get_or_init(|| {
        let spawned = std::thread::Builder::new()
            .name("reims-vgpu-pipeline-cache".into())
            .spawn(|| owner().run());
        if let Err(e) = spawned {
            crate::observe::fail(format!(
                "pipeline_cache_save_failed reason=thread_spawn {} (no periodic saves this process)",
                io_fields(&e)
            ));
        }
        Owner {
            state: Mutex::new(OwnerState {
                sources: Vec::new(),
                finals: Vec::new(),
            }),
            wake: Condvar::new(),
        }
    })
}

/// Register a live cache with the persistence thread.
pub(crate) fn register(source: &Arc<Source>) {
    let o = owner();
    o.state
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .sources
        .push(Arc::clone(source));
    o.wake.notify_one();
}

impl Owner {
    fn submit_final(&self, source: Arc<Source>, data: Vec<u8>, created: u64) {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .finals
            .push((source, data, created));
        self.wake.notify_one();
    }

    fn run(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            let finals = std::mem::take(&mut state.finals);
            state.sources.retain(|s| s.is_alive());
            let sources = state.sources.clone();
            drop(state);
            for (source, data, created) in finals {
                source.land(&data, created, "teardown");
            }
            let now = Instant::now();
            for source in &sources {
                source.tick(now);
            }
            drop(sources);
            state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.finals.is_empty() {
                state = self
                    .wake
                    .wait_timeout(state, SAVE_MIN_INTERVAL)
                    .unwrap_or_else(|p| p.into_inner())
                    .0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UUID: [u8; 16] = [
        0x3d, 0xba, 0x65, 0xdf, 0x4c, 0xa4, 0x40, 0x2e, 0xef, 0x67, 0x0d, 0xc3, 0xbf, 0x80, 0x5a,
        0xfd,
    ];

    fn header(size: u32, version: u32, vendor: u32, device: u32, uuid: [u8; 16]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&size.to_le_bytes());
        b.extend_from_slice(&version.to_le_bytes());
        b.extend_from_slice(&vendor.to_le_bytes());
        b.extend_from_slice(&device.to_le_bytes());
        b.extend_from_slice(&uuid);
        b
    }

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("reims-pcp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_matching_header_is_accepted_and_each_field_is_named_on_mismatch() {
        let ok = header(32, 1, 0x10de, 0x2684, UUID);
        assert_eq!(check_header(&ok, 0x10de, 0x2684, &UUID), Ok(()));
        let mut with_payload = ok.clone();
        with_payload.extend_from_slice(&[7u8; 100]);
        assert_eq!(check_header(&with_payload, 0x10de, 0x2684, &UUID), Ok(()));
        assert_eq!(check_header(&ok[..31], 0x10de, 0x2684, &UUID), Err(HeaderReject::Short));
        assert_eq!(
            check_header(&header(16, 1, 0x10de, 0x2684, UUID), 0x10de, 0x2684, &UUID),
            Err(HeaderReject::HeaderSize)
        );
        assert_eq!(
            check_header(&header(4096, 1, 0x10de, 0x2684, UUID), 0x10de, 0x2684, &UUID),
            Err(HeaderReject::HeaderSize),
            "a header size past the blob is corruption"
        );
        assert_eq!(
            check_header(&header(32, 2, 0x10de, 0x2684, UUID), 0x10de, 0x2684, &UUID),
            Err(HeaderReject::HeaderVersion)
        );
        assert_eq!(check_header(&ok, 0x1002, 0x2684, &UUID), Err(HeaderReject::Vendor));
        assert_eq!(check_header(&ok, 0x10de, 0x2685, &UUID), Err(HeaderReject::Device));
        let mut other = UUID;
        other[15] ^= 1;
        assert_eq!(check_header(&ok, 0x10de, 0x2684, &other), Err(HeaderReject::Uuid));
        for r in [
            HeaderReject::Short,
            HeaderReject::HeaderSize,
            HeaderReject::HeaderVersion,
            HeaderReject::Vendor,
            HeaderReject::Device,
            HeaderReject::Uuid,
        ] {
            assert!(!r.reason().is_empty());
        }
    }

    #[test]
    fn load_distinguishes_absent_loaded_and_each_rejection() {
        let dir = scratch("load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.bin");
        assert_eq!(load_blob(&path, 1, 2, &UUID, 1 << 20), LoadOutcome::Absent);
        let mut good = header(32, 1, 1, 2, UUID);
        good.extend_from_slice(&[9u8; 64]);
        std::fs::write(&path, &good).unwrap();
        assert_eq!(load_blob(&path, 1, 2, &UUID, 1 << 20), LoadOutcome::Loaded(good.clone()));
        match load_blob(&path, 1, 3, &UUID, 1 << 20) {
            LoadOutcome::Rejected { reason, bytes } => {
                assert_eq!(reason, "device_mismatch");
                assert_eq!(bytes, good.len());
            }
            other => panic!("{other:?}"),
        }
        match load_blob(&path, 1, 2, &UUID, 40) {
            LoadOutcome::Rejected { reason, .. } => assert!(reason.starts_with("too_large")),
            other => panic!("{other:?}"),
        }
        std::fs::write(&path, b"garbage").unwrap();
        match load_blob(&path, 1, 2, &UUID, 1 << 20) {
            LoadOutcome::Rejected { reason, .. } => assert_eq!(reason, "header_short"),
            other => panic!("{other:?}"),
        }
        // A directory where the file should be is a read failure, not absence.
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir_all(&path).unwrap();
        match load_blob(&path, 1, 2, &UUID, 1 << 20) {
            LoadOutcome::Rejected { reason, .. } => assert!(reason.starts_with("read"), "{reason}"),
            other => panic!("{other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_path_is_per_user_and_keyed_by_vendor_device_driver_and_uuid() {
        let tmp = Path::new(if cfg!(windows) { r"C:\T" } else { "/tmp" });
        let la = PathBuf::from(if cfg!(windows) { r"C:\Users\n\AppData\Local" } else { "/la" });
        let xdg = PathBuf::from(if cfg!(windows) { r"C:\xdg" } else { "/xdg" });
        let home = PathBuf::from(if cfg!(windows) { r"C:\Users\n" } else { "/home/n" });
        let (d, s) = cache_dir_from(Some(&la), Some(&xdg), Some(&home), tmp);
        assert_eq!(s, DirSource::LocalAppData);
        assert_eq!(d, la.join("reims-vgpu").join("pipeline-cache"));
        let (d, s) = cache_dir_from(None, Some(&xdg), Some(&home), tmp);
        assert_eq!((d, s), (xdg.join("reims-vgpu").join("pipeline-cache"), DirSource::XdgCache));
        let (d, s) = cache_dir_from(None, None, Some(&home), tmp);
        assert_eq!(
            (d, s),
            (home.join(".cache").join("reims-vgpu").join("pipeline-cache"), DirSource::HomeCache)
        );
        // Relative values are not trusted (they would follow the cwd).
        let (_, s) = cache_dir_from(Some(Path::new("rel")), None, None, tmp);
        assert_eq!(s, DirSource::TempFallback);
        let (d, s) = cache_dir_from(None, None, None, tmp);
        assert_eq!((d, s), (tmp.join("reims-vgpu").join("pipeline-cache"), DirSource::TempFallback));

        let name = cache_file_name(0x10de, 0x2684, 0x8d2a_c000, &UUID);
        assert_eq!(name, "10de-2684-8d2ac000-3dba65df4ca4402eef670dc3bf805afd.bin");
        assert_ne!(name, cache_file_name(0x10de, 0x2684, 0x8d2a_c001, &UUID), "driver update = new file");
        assert_ne!(name, cache_file_name(0x10de, 0x2685, 0x8d2a_c000, &UUID));
        assert_ne!(name, cache_file_name(0x1002, 0x2684, 0x8d2a_c000, &UUID));
        assert_eq!(cache_file_name(1, 2, 3, &[0; 16]).len(), name.len(), "fixed width");
    }

    #[test]
    fn the_throttle_saves_only_new_work_and_at_most_once_per_interval() {
        let i = Duration::from_secs(10);
        let t0 = Instant::now();
        let mut t = SaveThrottle::default();
        assert_eq!(t.decide(t0, 0, i), Tick::Idle, "nothing created");
        assert_eq!(t.decide(t0, 3, i), Tick::SaveNow, "first save is immediate");
        t.last_attempt = Some(t0);
        t.created_at_last = 3;
        assert_eq!(t.decide(t0 + Duration::from_secs(1), 3, i), Tick::Idle);
        assert_eq!(t.decide(t0 + Duration::from_secs(1), 4, i), Tick::TooSoon);
        assert_eq!(t.decide(t0 + Duration::from_millis(9_999), 50, i), Tick::TooSoon);
        assert_eq!(t.decide(t0 + i, 50, i), Tick::SaveNow);
        assert_eq!(t.decide(t0 + Duration::from_secs(3600), 3, i), Tick::Idle, "idle forever stays idle");
        t.frozen = true;
        assert_eq!(t.decide(t0 + Duration::from_secs(3600), 99, i), Tick::Frozen);
    }

    #[test]
    fn over_the_bound_a_cold_boot_keeps_its_last_snapshot_and_a_stale_warm_boot_discards() {
        // The decision is bound-relative; pinned at the v10 bound of 8 MiB so
        // the v9 byte counts below keep their meaning. The v12 50 MiB bound is
        // pinned in `context.rs`.
        let cap = 8 << 20;
        // The v9 shape: cold boot, saves under the bound land, 9.24 MB keeps 7.26 MB.
        assert_eq!(decide_save(0, 0, 15_260, cap), SaveDecision::Write);
        assert_eq!(decide_save(0, 15_260, 15_260, cap), SaveDecision::Unchanged);
        assert_eq!(decide_save(0, 7_261_348, 9_235_124, cap), SaveDecision::OverCapKeepLast);
        // Warm boot from 7.26 MB that adds a little: keep.
        assert_eq!(decide_save(7_261_348, 7_261_348, 8_500_000, cap), SaveDecision::OverCapKeepLast);
        assert_eq!(decide_save(7_261_348, 7_261_348, 8_000_000, cap), SaveDecision::Write);
        // Warm boot whose own compiles outgrow what it loaded: stale, rebuild.
        assert_eq!(decide_save(3_000_000, 3_000_000, 8_400_000, cap), SaveDecision::OverCapDiscardStale);
        // Exactly at the bound is still under it.
        assert_eq!(decide_save(0, 0, cap, cap), SaveDecision::Write);
        // The v11 Tahoe boot1 shape under the v12 50 MiB bound: it writes; past
        // that bound the keep-last arm is unchanged.
        let v12 = 50 << 20;
        assert_eq!(decide_save(0, 6_444_408, 9_772_133, v12), SaveDecision::Write);
        assert_eq!(decide_save(0, 50_000_000, v12 + 1, v12), SaveDecision::OverCapKeepLast);
        for d in [
            SaveDecision::Write,
            SaveDecision::Unchanged,
            SaveDecision::OverCapKeepLast,
            SaveDecision::OverCapDiscardStale,
        ] {
            assert!(!d.reason().is_empty());
        }
    }

    /// v13: the unbounded cap, the optional lower limit, and the refusal.
    #[test]
    fn v13_the_cap_is_unbounded_unless_a_lower_limit_is_set() {
        use crate::config::Count;
        assert_eq!(unbounded_cap_from(Count::Unset), (usize::MAX, "unbounded", None));
        assert_eq!(
            unbounded_cap_from(Count::Narrowed(200)),
            (200 << 20, "env_lower_limit", None)
        );
        assert_eq!(
            unbounded_cap_from(Count::Refused("abc".into())),
            (usize::MAX, "unbounded_after_refusal", Some("abc".to_string()))
        );
        assert_eq!(cap_text(usize::MAX), "unbounded");
        assert_eq!(cap_text(52_428_800), "52428800");
        // A 300 MB cache, far past the v12 50 MiB bound, simply writes.
        assert_eq!(decide_save(0, 250_000_000, 300_000_000, usize::MAX), SaveDecision::Write);
        assert_eq!(decide_save(0, 300_000_000, 300_000_000, usize::MAX), SaveDecision::Unchanged);
        assert!(UNBOUNDED_MAX_MB_CEILING >= 1 << 20);
    }

    /// v13: the never-delete policy turns discard-stale into keep-last and
    /// leaves every other verdict alone; the v12 policy is unchanged.
    #[test]
    fn v13_oversize_never_deletes_the_file() {
        let cap = 8 << 20;
        let stale = decide_save(3_000_000, 3_000_000, 8_400_000, cap);
        assert_eq!(stale, SaveDecision::OverCapDiscardStale);
        assert_eq!(apply_policy(stale, SavePolicy::V13), SaveDecision::OverCapKeepLast);
        assert_eq!(apply_policy(stale, SavePolicy::V12), SaveDecision::OverCapDiscardStale);
        for d in [
            SaveDecision::Write,
            SaveDecision::Unchanged,
            SaveDecision::OverCapKeepLast,
            SaveDecision::LowDiskKeepLast,
        ] {
            assert_eq!(apply_policy(d, SavePolicy::V13), d);
            assert_eq!(apply_policy(d, SavePolicy::V12), d);
        }
        assert_eq!(SaveDecision::LowDiskKeepLast.reason(), "low_disk_keep_last");
        assert_eq!(SavePolicy::V13.disk_free_floor, Some(DISK_FREE_FLOOR));
        assert_eq!(SavePolicy::V12.disk_free_floor, None);
    }

    /// v13: a save that would leave less than the floor free is skipped.
    #[test]
    fn v13_a_save_that_would_leave_under_1_gib_free_is_skipped() {
        let floor = DISK_FREE_FLOOR;
        assert_eq!(disk_verdict(None, 10, floor), DiskVerdict::Unknown);
        assert_eq!(disk_verdict(Some(floor + 100), 100, floor), DiskVerdict::Room);
        assert_eq!(
            disk_verdict(Some(floor + 100), 101, floor),
            DiskVerdict::Low { free: floor + 100 }
        );
        assert_eq!(disk_verdict(Some(10), 100, floor), DiskVerdict::Low { free: 10 });
        assert_eq!(disk_verdict(Some(u64::MAX), usize::MAX, 0), DiskVerdict::Room);
    }

    /// v13: the free-space probe answers for a real directory (and for a
    /// not-yet-created path under it) on Windows.
    #[test]
    fn v13_free_space_is_readable_for_the_cache_drive() {
        let dir = std::env::temp_dir();
        let missing = dir.join("reims-pcp-not-created").join("x.bin");
        if cfg!(windows) {
            assert!(free_bytes_near(&dir).is_some_and(|f| f > 0));
            assert!(free_bytes_near(&missing).is_some());
        } else {
            assert_eq!(free_bytes_near(&dir), None);
        }
    }

    #[test]
    fn the_atomic_write_creates_the_directory_replaces_whole_and_leaves_no_temp() {
        let dir = scratch("atomic");
        let path = dir.join("nested").join("c.bin");
        write_atomic(&path, &[1u8; 1000]).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), vec![1u8; 1000]);
        write_atomic(&path, &[2u8; 10]).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), vec![2u8; 10], "replaced, not appended");
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "c.bin")
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        // A failing rename (target is a directory) leaves the target intact
        // and cleans its temp file.
        let blocked = dir.join("blocked");
        std::fs::create_dir_all(blocked.join("inner")).unwrap();
        let err = write_atomic(&blocked, &[3u8; 4]).unwrap_err();
        assert_eq!(err.0, "rename");
        assert!(blocked.join("inner").is_dir());
        let temps = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .count();
        assert_eq!(temps, 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
