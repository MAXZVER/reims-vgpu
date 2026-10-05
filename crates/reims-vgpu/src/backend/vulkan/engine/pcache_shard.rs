//! TEMP (lab, `REIMS_VGPU_LAB_PCACHE_SHARD=on`): a warm pipeline cache that
//! survives boots without the size pathology that keeps the monolithic blob
//! cold.
//!
//! # What the monolithic path does, and why it cannot be fixed in place
//!
//! [`super::context`] keeps one `VkPipelineCache` and persists one blob, keyed
//! by `pipelineCacheUUID`. A `VkPipelineCache` has no eviction, the blob only
//! grows across boots, and every compile consults **all** of it. Three
//! measurements, on three rails, all price the same shape:
//!
//! - macos-13 (driven, 4 boots): a 30.82 MB blob charges 4.13 ms per compile
//!   against 0.73 ms for a right-sized 3.85 MB one — the miss count barely
//!   moves, only what each compile costs;
//! - macos-15 (6 interleaved boots): 14.4 MB charges 2.11 ms against 0.52-0.57
//!   for 3.3 MB or none at all;
//! - Tahoe at 5K (n75, `REIMS_VGPU_LAB_PCACHE_KEEP`): a warm 7.4 MB blob was
//!   the best arm of its day, and one boot later an 8.0 MB blob turned 21 new
//!   pipelines into 2.7 s of compile — a "big blob makes individual compiles
//!   pathological" result that kept `PCACHE_KEEP` off.
//!
//! Tahoe's genuine working set is ~9 MiB, past the 8 MiB monolithic bound, so
//! the honest choices there were "cold every boot" or "growing blob with a
//! pathological tail". The problem is not the amount of warm data — it is that
//! **one compile's consult cost scales with every boot's accumulated data**.
//!
//! # This scheme
//!
//! Split the cache by pipeline: 64 shards, one `VkPipelineCache` each, a
//! pipeline always compiled through `shard(digest(key))`. Each shard's blob is
//! its own small file (`s%02x.bin`) under a per-device directory, described by
//! a `ledger.bin` that carries the LRU metadata across boots.
//!
//! What that buys, term by term:
//!
//! - **Bounded consult.** A compile consults its shard's blob — ~9 MiB / 64
//!   ≈ 150-300 KiB — never the whole history. The pathology above is excluded
//!   structurally rather than bounded by a size that a new working set can
//!   always outgrow.
//! - **Eviction at the right granularity.** Overflow trims the coldest
//!   *shards* (LRU on the ledger's `last_used` boot), so a rare one-boot
//!   workload costs its own shard, not the whole warm set. The monolithic
//!   discard-everything is what made every Tahoe boot cold.
//! - **Lazily loaded warm data.** A shard file is read into a
//!   `vkCreatePipelineCache` only when a pipeline in that bucket is first
//!   needed, so a boot pays for what it draws.
//! - **A single pathological group costs only itself.** A shard snapshot past
//!   [`SHARD_MAX_WARM_BYTES`] is reset — cache destroyed, file and ledger
//!   record dropped — instead of being allowed to grow into the next n75.
//!
//! # Alternatives considered
//!
//! - **LRU ledger on the monolithic blob**: a `VkPipelineCache` blob is
//!   opaque; there is no per-entry eviction, so a ledger could only re-decide
//!   the same all-or-nothing discard the cap already makes.
//! - **`VK_EXT_pipeline_cache_control`** (`FAIL_ON_PIPELINE_COMPILE_REQUIRED`):
//!   ash 0.38 carries the flag, but using it needs `pipelineCreationCacheControl`
//!   enabled at device create — churn on the off path's twin — and it yields a
//!   per-pipeline hit *signal*, which `pl_compile_us` and `pipeline_misses`
//!   already give. It cannot bound the consult size, which is the pathology.
//! - **Rely on the NVIDIA ICD's own disk cache**: the macos-13 arms price a
//!   right-sized blob at 0.73 ms/compile against 1.41 ms with no blob, on the
//!   same driver with its own cache warm — the driver cache does not cover the
//!   pipeline-level work. What no `VkPipelineCache` can carry — pipeline
//!   layouts, descriptor set layouts, render passes — is rebuilt every boot by
//!   the in-memory `ObjectCaches` either way; this scheme targets exactly the
//!   boundary that is cacheable.
//!
//! # Byte-for-byte off path
//!
//! Default (`off`/unset) changes nothing: `pipeline_cache_for` returns the
//! monolithic cache, `persist_pipeline_cache` keeps its snapshot path, and no
//! census route fires.
//!
//! # What "hit" and "miss" mean here
//!
//! A shard-level proxy, honestly named: a compile routed through a shard whose
//! cache **had warm initial data** counts `pcache_shard_hit`, through a cold
//! one `pcache_shard_miss`. The specific pipeline may still miss inside a warm
//! shard (first boot after a guest update); the true per-pipeline reading
//! stays `pl_compile_us` against `pipeline_misses`, which is what the A/B
//! should read first.

use ash::vk;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::context::{pipeline_cache_blob_compatible, submit_shard_wave};
use super::digest::Digest128;

/// How many shard buckets (and therefore files, and `VkPipelineCache` objects)
/// the warm set is split into. 64 against a ~9 MiB Tahoe working set averages
/// ~150-300 KiB a shard — far from the 2 MiB per-shard bound — while keeping
/// the per-boot file count a directory listing can still read.
pub(crate) const SHARD_COUNT: u32 = 64;

/// Largest single shard blob that may be warmed from or written back. A shard
/// averages a fraction of this; a snapshot past it means one bucket accumulated
/// a whole workload's worth of entries, which is the n75 shape at bucket
/// scale, and the shard is reset rather than allowed to grow.
const SHARD_MAX_WARM_BYTES: usize = 2 * 1024 * 1024;

/// Default total warm-set bound across all shard files. The monolithic path's
/// 8 MiB had to sit under one boot's working set or every boot went cold; the
/// shard bound only has to sit under the *LRU-hot* set, because overflow now
/// trims cold shards instead of discarding the file. 128 MiB since 2026-10-05:
/// the home rig (AMD) settled at 37 MB and trimmed every boot at the old 32
/// (one 1.25 s compile on a first Finder transition); at 128 it holds
/// trimmed=0 with 0-2 compiles over 100 ms (Opus hq057/hq058), while Intel
/// pays nothing for the headroom (32.4 MB warm, trimmed=0 at either cap,
/// compile totals and Hz identical across the cap arms — ZCode n234).
const SHARD_TOTAL_WARM_BYTES: usize = 128 * 1024 * 1024;

/// TEMP (lab): `REIMS_VGPU_LAB_PCACHE_SHARD=off` restores the byte-for-byte
/// monolithic path. On by default (n200: per-boot compile 752/683 ms with the
/// discarding monolith against 400 cold / 337-338 warm with shards, misses
/// flat, no size pathology across boots).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("REIMS_VGPU_LAB_PCACHE_SHARD") {
        Ok(v) => v != "off",
        Err(_) => true,   // unset: on (the default since n200)
    })
}

/// TEMP (lab): total shard-directory bound — [`SHARD_TOTAL_WARM_BYTES`], or
/// `REIMS_VGPU_LAB_PCACHE_MAX_MB=N` MiB (1..=256), the same knob the
/// monolithic path reads, so an A/B narrows both arms with one variable.
fn total_cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("REIMS_VGPU_LAB_PCACHE_MAX_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|mb| (1..=256).contains(mb))
            .map_or(SHARD_TOTAL_WARM_BYTES, |mb| mb * 1024 * 1024)
    })
}

/// Bucket a key digest into a shard id. The digest is a *bucketing* key (see
/// [`super::digest`] for the width rule), pre-mixed by the caller —
/// `PipelineKey::shard_digest` / `ComputePipelineKey::shard_digest`, because
/// only the key's own module knows which of its members are per-boot ordinals
/// that must not reach the bucket. Two pipelines in one shard is the normal
/// case and costs nothing.
pub(crate) fn shard_of(digest: u64) -> u32 {
    (digest % u64::from(SHARD_COUNT)) as u32
}

/// Digest the engine's own embedded kernels (scatter, page-diff) the same
/// way, so every compile site routes through the same bucket space.
pub(crate) fn words_digest(words: &[u32]) -> u64 {
    Digest128::of_u32_words(words).a
}

/// One compile was routed through a warm (`true`) or cold (`false`) shard —
/// the census hit/miss point. Free because the caller holds a mutable borrow
/// of the shard map, and because it owns no state but the process counters.
fn note_routed(warm: bool) {
    if warm {
        stats::HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        stats::MISSES.fetch_add(1, Ordering::Relaxed);
    }
}

/// On-disk home for one device's shards, beside the monolithic path's flat
/// file and keyed by the same `pipelineCacheUUID` so a GPU/driver change lands
/// in a distinct directory and never warms from a foreign blob.
pub(crate) fn disk_dir(uuid: &[u8; 16]) -> PathBuf {
    use std::fmt::Write as _;
    let mut hex = String::with_capacity(32);
    for b in uuid {
        let _ = write!(hex, "{b:02x}");
    }
    std::env::temp_dir().join(format!("reims-vgpu-vk-pcache-shards-{hex}"))
}

fn shard_path(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("s{id:02x}.bin"))
}

fn ledger_path(dir: &Path) -> PathBuf {
    dir.join("ledger.bin")
}

const LEDGER_MAGIC: [u8; 8] = *b"RVPSHL01";

/// One shard's LRU metadata, carried across boots by the ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShardRecord {
    pub id: u32,
    /// Blob length last written (or scanned) — the budget the trim spends.
    pub bytes: u64,
    /// `boot_seq` of the last boot that routed a compile through this shard.
    pub last_used: u64,
    /// How many boots warm-loaded this shard and compiled through it — the
    /// tiebreak that keeps a long-hot shard over a recently-lucky cold one.
    pub used: u64,
}

/// The ledger: a boot sequence number and one record per known shard file.
/// Corrupt or missing resets to empty (boot 1, no records); the next save wave
/// rebuilds it from the live shards, so the reset costs LRU history, never
/// warm data still on disk.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Ledger {
    pub boot_seq: u64,
    pub records: BTreeMap<u32, ShardRecord>,
}

/// Little-endian put helpers, so the format is fixed without a dependency.
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn get_u32(blob: &[u8], at: usize) -> Option<u32> {
    blob.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}

fn get_u64(blob: &[u8], at: usize) -> Option<u64> {
    blob.get(at..at + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
}

pub(crate) fn encode_ledger(ledger: &Ledger) -> Vec<u8> {
    // One record is id(4) + bytes(8) + last_used(8) + used(8) = 28 bytes.
    let mut out = Vec::with_capacity(8 + 8 + 4 + ledger.records.len() * 28);
    out.extend_from_slice(&LEDGER_MAGIC);
    put_u64(&mut out, ledger.boot_seq);
    put_u32(&mut out, ledger.records.len() as u32);
    for record in ledger.records.values() {
        put_u32(&mut out, record.id);
        put_u64(&mut out, record.bytes);
        put_u64(&mut out, record.last_used);
        put_u64(&mut out, record.used);
    }
    out
}

pub(crate) fn parse_ledger(blob: &[u8]) -> Option<Ledger> {
    if blob.len() < 20 || blob[..8] != LEDGER_MAGIC {
        return None;
    }
    let boot_seq = get_u64(blob, 8)?;
    let count = get_u32(blob, 16)? as usize;
    let mut records = BTreeMap::new();
    let mut at = 20;
    for _ in 0..count {
        // Record layout, mirroring `encode_ledger`: id(4) bytes(8)
        // last_used(8) used(8) = 28 bytes.
        let id = get_u32(blob, at)?;
        let bytes = get_u64(blob, at + 4)?;
        let last_used = get_u64(blob, at + 12)?;
        let used = get_u64(blob, at + 20)?;
        at += 28;
        if id >= SHARD_COUNT {
            return None; // an id out of range means a format we do not know
        }
        records.insert(id, ShardRecord { id, bytes, last_used, used });
    }
    // Trailing bytes mean a format change we do not understand; refuse rather
    // than guess at records we may be misreading.
    if at != blob.len() {
        return None;
    }
    Some(Ledger { boot_seq, records })
}

/// Every shard file on disk with its length, whether or not the ledger knows
/// it. The boot merge turns unknown files into `last_used = 0` records, so
/// stray data participates in the budget instead of hiding from the trim.
pub(crate) fn scan_shard_files(dir: &Path) -> Vec<(u32, u64)> {
    let mut scanned = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return scanned;
    };
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(id) = name
            .strip_prefix('s')
            .and_then(|rest| rest.strip_suffix(".bin"))
            .and_then(|hex| u32::from_str_radix(hex, 16).ok())
        else {
            continue;
        };
        if id >= SHARD_COUNT {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        scanned.push((id, meta.len()));
    }
    scanned.sort_unstable();
    scanned
}

/// Remove the coldest shard files until the ledger's byte total fits `cap`,
/// and return the victims so the caller can report each by name.
///
/// Cold = lowest `last_used` boot, tiebroken by `used` (boots the shard was
/// warm and drawn through) and then id, so the order is total and reproducible
/// across boots. `live` names shards this boot is already using; the wave-time
/// trim passes them so a save never deletes warm data out from under the
/// device, and the boot-time trim passes an empty set because nothing is live
/// yet.
///
/// Never trims below one record: a cap smaller than the working set must
/// produce a cold-but-warmable cache, not an empty directory that would make
/// every following boot boot-one. That is the failure mode the monolithic
/// bound had, and it is not being re-introduced under a lab flag.
fn ledger_total(ledger: &Ledger) -> u64 {
    ledger.records.values().map(|r| r.bytes).sum()
}

pub(crate) fn trim_over_cap(
    ledger: &mut Ledger,
    dir: &Path,
    live: &HashSet<u32>,
    cap: u64,
) -> Vec<u32> {
    let mut victims = Vec::new();
    while ledger_total(ledger) > cap && ledger.records.len() > 1 {
        let Some(coldest) = ledger
            .records
            .values()
            .filter(|r| !live.contains(&r.id))
            .min_by_key(|r| (r.last_used, r.used, r.id))
            .map(|r| r.id)
        else {
            break; // everything left is live; overflow stands until next boot
        };
        ledger.records.remove(&coldest);
        let _ = std::fs::remove_file(shard_path(dir, coldest));
        victims.push(coldest);
    }
    victims
}

/// A shard warm-read or persistence failure, on the fail-visible boundary for
/// the same reason as the monolithic [`PipelineCacheDecline`]: a shard that
/// will not load costs this boot real cold compiles, and a reader has to be
/// able to tell that from a first boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShardDecline {
    Read { errno: Option<i32>, kind: std::io::ErrorKind },
    Incompatible { shard: u32, bytes: usize },
    TooLarge { shard: u32, bytes: usize, cap: usize },
    WarmCreate { shard: u32, result: vk::Result },
    Write { errno: Option<i32>, kind: std::io::ErrorKind },
}

impl ShardDecline {
    fn read(error: &std::io::Error) -> Self {
        Self::Read { errno: error.raw_os_error(), kind: error.kind() }
    }

    fn write(error: &std::io::Error) -> Self {
        Self::Write { errno: error.raw_os_error(), kind: error.kind() }
    }
}

impl crate::observe::Decline for ShardDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Read { .. } => "pcache_shard_read",
            Self::Incompatible { .. } => "pcache_shard_incompatible",
            Self::TooLarge { .. } => "pcache_shard_too_large",
            Self::WarmCreate { .. } => "pcache_shard_warm_create",
            Self::Write { .. } => "pcache_shard_write",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Read { errno, kind } | Self::Write { errno, kind } => vec![
                ("errno", errno.map_or_else(|| "none".to_string(), |v| v.to_string())),
                ("io_kind", format!("{kind:?}")),
            ],
            Self::Incompatible { shard, bytes } => {
                vec![("shard", shard.to_string()), ("bytes", bytes.to_string())]
            }
            Self::TooLarge { shard, bytes, cap } => vec![
                ("shard", shard.to_string()),
                ("bytes", bytes.to_string()),
                ("cap", cap.to_string()),
            ],
            Self::WarmCreate { shard, result } => vec![
                ("shard", shard.to_string()),
                ("vk_result", result.to_string().replace(char::is_whitespace, "_")),
            ],
        }
    }
}

/// Read one shard's warm blob and validate it against this device. `Ok(None)`
/// covers both "no file" (first boot for this bucket, quiet) and "file
/// rejected" (declined by name, cold fallback) — the split the caller needs,
/// because only the second is a real warm failure.
pub(crate) fn warm_blob(
    dir: &Path,
    shard: u32,
    props: &vk::PhysicalDeviceProperties,
) -> Result<Option<Vec<u8>>, ShardDecline> {
    let path = shard_path(dir, shard);
    let blob = match std::fs::read(&path) {
        Ok(blob) => blob,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(ShardDecline::read(&error)),
    };
    // Size before compatibility, as the monolithic read does: the cheaper,
    // more decisive refusal wins the report, and both are cold starts.
    if blob.len() > SHARD_MAX_WARM_BYTES {
        return Err(ShardDecline::TooLarge {
            shard,
            bytes: blob.len(),
            cap: SHARD_MAX_WARM_BYTES,
        });
    }
    if pipeline_cache_blob_compatible(&blob, props) {
        Ok(Some(blob))
    } else {
        Err(ShardDecline::Incompatible { shard, bytes: blob.len() })
    }
}

/// One live shard: its `VkPipelineCache`, what it warmed from, and the
/// wave-to-wave bookkeeping.
struct ShardEntry {
    cache: vk::PipelineCache,
    /// Bytes of initial data this cache was created with; 0 = this shard is
    /// cold this boot, and compiles through it count `pcache_shard_miss`.
    warm_bytes: usize,
    /// Last snapshot length seen (loaded or written) — the growth debounce.
    saved_len: usize,
    /// Compiles routed here since the last wave; >0 means snapshot-candidate.
    compiles_since_wave: u64,
    /// Whether `used` was already advanced this boot (once per boot, by name).
    warm_hit_counted: bool,
}

/// The per-device shard state, held by `DeviceContext` behind its mutex.
pub(crate) struct ShardSet {
    dir: PathBuf,
    boot_seq: u64,
    ledger: Ledger,
    props: vk::PhysicalDeviceProperties,
    entries: HashMap<u32, ShardEntry>,
    /// Shards included in the last submitted wave whose job may not have run
    /// yet. The next wave re-includes them, so a wave replaced in the mailbox
    /// cannot drop a shard's only write — the mailbox holds one job, and the
    /// replacement is built as a superset by [`Self::save_wave`].
    awaiting: Option<(HashSet<u32>, Arc<AtomicBool>)>,
    over_cap_reported: bool,
}

impl ShardSet {
    /// Boot: read the ledger, merge in whatever files are on disk, advance
    /// the boot sequence, trim over cap, and write the ledger back so the
    /// sequence survives a SIGKILLed boot (the testing rail's normal exit).
    pub(crate) fn open(dir: &Path, props: &vk::PhysicalDeviceProperties) -> Self {
        let _ = std::fs::create_dir_all(dir);
        let mut ledger = std::fs::read(ledger_path(dir))
            .ok()
            .and_then(|blob| parse_ledger(&blob))
            .unwrap_or_default();
        for (id, len) in scan_shard_files(dir) {
            let record = ledger
                .records
                .entry(id)
                .or_insert(ShardRecord { id, bytes: 0, last_used: 0, used: 0 });
            record.bytes = record.bytes.max(len);
        }
        ledger.boot_seq += 1;
        let cap = total_cap() as u64;
        let victims = trim_over_cap(&mut ledger, dir, &HashSet::new(), cap);
        for victim in &victims {
            crate::observe::off(format!(
                "pcache_shard_trim shard={victim:02x} live_mib={}/{cap}",
                ledger.records.values().map(|r| r.bytes).sum::<u64>() / (1024 * 1024),
            ));
        }
        // Synchronous by design: this is the one ledger write on the init
        // path, and losing the boot_seq bump to a crash would reuse a boot
        // number and blur the LRU order.
        if let Err(error) = write_ledger(dir, &ledger) {
            let decline = ShardDecline::write(&error);
            crate::observe::Emit::decline("pcache_shard_write", &decline).fail_once(0);
        }
        let bytes: u64 = ledger.records.values().map(|r| r.bytes).sum();
        crate::observe::off(format!(
            "pcache_shard_boot shards={} bytes_kib={} cap_mib={} boot_seq={} trimmed={} dir={}",
            ledger.records.len(),
            bytes / 1024,
            cap / (1024 * 1024),
            ledger.boot_seq,
            victims.len(),
            dir.display(),
        ));
        stats::OPENED.fetch_add(1, Ordering::Relaxed);
        stats::TRIMMED.fetch_add(victims.len() as u64, Ordering::Relaxed);
        stats::DISK_BYTES.store(bytes, Ordering::Relaxed);
        Self {
            dir: dir.to_path_buf(),
            boot_seq: ledger.boot_seq,
            ledger,
            props: props.clone(),
            entries: HashMap::new(),
            awaiting: None,
            over_cap_reported: false,
        }
    }

    /// The `VkPipelineCache` one compile should go through, creating and
    /// warm-loading this digest's shard on first use. Also the hit/miss
    /// census point: every caller is one actual `vkCreate*Pipelines` about to
    /// happen (the in-memory `ObjectCache` filtered the real hits already).
    pub(crate) fn cache_for(&mut self, digest: u64, device: &ash::Device) -> vk::PipelineCache {
        let shard = shard_of(digest);
        if let Some(entry) = self.entries.get_mut(&shard) {
            entry.compiles_since_wave += 1;
            note_routed(entry.warm_bytes > 0);
            return entry.cache;
        }
        let warm = match warm_blob(&self.dir, shard, &self.props) {
            Ok(warm) => warm,
            Err(decline) => {
                crate::observe::Emit::decline("pcache_shard_load", &decline).fail_once(0);
                None
            }
        };
        let mut pcci = vk::PipelineCacheCreateInfo::default();
        if let Some(blob) = warm.as_deref() {
            pcci = pcci.initial_data(blob);
        }
        // Warm-create failure falls back cold, exactly as the monolithic boot
        // path does: the header matched, but the driver refused the payload,
        // and the boot must go on.
        let (cache, warm_bytes) = match unsafe { device.create_pipeline_cache(&pcci, None) } {
            Ok(cache) => (cache, warm.as_ref().map_or(0, Vec::len)),
            Err(result) => {
                let decline = ShardDecline::WarmCreate { shard, result };
                crate::observe::Emit::decline("pcache_shard_load", &decline).fail_once(0);
                let cold = unsafe {
                    device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                };
                match cold {
                    Ok(cache) => (cache, 0),
                    // Unreachable short of driver loss; the shard simply has
                    // no entry, and the next compile retries the cold create.
                    Err(_) => return vk::PipelineCache::null(),
                }
            }
        };
        if warm_bytes > 0 {
            crate::observe::off(format!(
                "pcache_shard_loaded shard={shard:02x} bytes={warm_bytes}"
            ));
            stats::LOADED.fetch_add(1, Ordering::Relaxed);
        }
        note_routed(warm_bytes > 0);
        self.entries.insert(
            shard,
            ShardEntry {
                cache,
                warm_bytes,
                saved_len: warm_bytes,
                compiles_since_wave: 1,
                warm_hit_counted: false,
            },
        );
        cache
    }

    /// Snapshot every dirty shard (plus any still awaiting an unwritten wave)
    /// and hand one `WriteShardWave` to the persistence owner: shard files,
    /// the rebuilt ledger, trim deletes. Debounced by the caller — this is
    /// the shard twin of `persist_pipeline_cache`'s snapshot stage, and shares
    /// its `REIMS_VGPU_LAB_PCACHE_SAVE_MS` interval.
    pub(crate) fn save_wave(&mut self, device: &ash::Device) {
        // The build set: every shard a compile touched since the last wave,
        // plus every shard whose write is still sitting in an unrun job (the
        // mailbox holds one job; a replacement must be a superset or that
        // shard's only write is dropped on the floor).
        let mut from_awaiting: HashSet<u32> = HashSet::new();
        let mut ids: Vec<u32> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.compiles_since_wave > 0)
            .map(|(id, _)| *id)
            .collect();
        if let Some((awaiting, ack)) = &self.awaiting {
            if !ack.load(Ordering::Relaxed) {
                for id in awaiting.iter().copied() {
                    from_awaiting.insert(id);
                    ids.push(id);
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            // Nothing to write; a finished awaiting wave can be forgotten.
            if self
                .awaiting
                .as_ref()
                .is_some_and(|(_, ack)| ack.load(Ordering::Relaxed))
            {
                self.awaiting = None;
            }
            return;
        }
        // What this wave must remember per shard even when it writes nothing:
        // the compile count (for the hit/miss routes) and warm-ness (for the
        // ledger's `used`), captured before the counters reset.
        let mut routed: Vec<(u32, u64, bool)> = Vec::new();
        let mut writes: Vec<(PathBuf, Vec<u8>, u32, u64)> = Vec::new();
        let mut deletes: Vec<PathBuf> = Vec::new();
        for id in ids {
            let Some(entry) = self.entries.get_mut(&id) else {
                continue;
            };
            if entry.compiles_since_wave > 0 {
                routed.push((id, entry.compiles_since_wave, entry.warm_bytes > 0));
            }
            entry.compiles_since_wave = 0;
            let data = match unsafe { device.get_pipeline_cache_data(entry.cache) } {
                Ok(data) => data,
                Err(result) => {
                    // The snapshot stays unsaved until a later compile in
                    // this bucket re-marks it dirty — the same recovery the
                    // monolithic write failure has.
                    let decline = ShardDecline::WarmCreate { shard: id, result };
                    crate::observe::Emit::decline("pcache_shard_save", &decline).fail_once(0);
                    continue;
                }
            };
            // One bucket outgrew its bound: reset it rather than write it.
            // The next compiles through it are cold, and only this bucket's.
            if data.len() > SHARD_MAX_WARM_BYTES {
                let decline = ShardDecline::TooLarge {
                    shard: id,
                    bytes: data.len(),
                    cap: SHARD_MAX_WARM_BYTES,
                };
                crate::observe::Emit::decline("pcache_shard_save", &decline).fail_once(0);
                unsafe { device.destroy_pipeline_cache(entry.cache, None) };
                match unsafe {
                    device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                } {
                    Ok(cache) => {
                        *entry = ShardEntry {
                            cache,
                            warm_bytes: 0,
                            saved_len: 0,
                            compiles_since_wave: 0,
                            warm_hit_counted: false,
                        };
                    }
                    Err(_) => {
                        self.entries.remove(&id);
                    }
                }
                self.ledger.records.remove(&id);
                deletes.push(shard_path(&self.dir, id));
                continue;
            }
            // Growth debounce, per shard: an unchanged length means no new
            // pipeline landed in this bucket since the last snapshot. It does
            // NOT apply to an awaiting shard — its last snapshot was submitted
            // into a job that never ran, so equal length still means unwritten
            // bytes, and dropping it here would drop the shard's only write.
            if data.len() == entry.saved_len && !from_awaiting.contains(&id) {
                continue;
            }
            entry.saved_len = data.len();
            let len = data.len() as u64;
            writes.push((shard_path(&self.dir, id), data, id, len));
        }
        // Per-shard hit/miss routes for this wave, aggregated rather than per
        // compile — a login storm would otherwise emit hundreds of
        // single-count lines inside one debounce window.
        for (id, compiles, warm) in &routed {
            let route = if *warm { "pcache_shard_hit" } else { "pcache_shard_miss" };
            crate::observe::off(format!("{route} shard={id:02x} compiles={compiles}"));
        }
        // LRU bookkeeping for every shard this boot actually used, warm or
        // not: a fully warm-hit shard never grows, and must still advance
        // `last_used` or a later trim would eat it while it was busy paying
        // off. `used` counts boots the shard was warm AND compiled through.
        let boot_seq = self.boot_seq;
        for (id, _, warm) in &routed {
            let record = self
                .ledger
                .records
                .entry(*id)
                .or_insert(ShardRecord { id: *id, bytes: 0, last_used: 0, used: 0 });
            record.last_used = boot_seq;
            if *warm {
                if let Some(entry) = self.entries.get_mut(id) {
                    if !entry.warm_hit_counted {
                        entry.warm_hit_counted = true;
                        record.used += 1;
                    }
                }
            }
        }
        for (_, _, id, len) in &writes {
            let record = self
                .ledger
                .records
                .entry(*id)
                .or_insert(ShardRecord { id: *id, bytes: 0, last_used: 0, used: 0 });
            record.bytes = *len;
        }
        // Wave-time trim: never a shard this boot loaded (live), always the
        // coldest the ledger knows. Keeps the cap honest across a growing
        // boot without deleting data the device is using.
        let live: HashSet<u32> = self.entries.keys().copied().collect();
        let cap = total_cap() as u64;
        let victims = trim_over_cap(&mut self.ledger, &self.dir, &live, cap);
        let live_bytes: u64 = self.ledger.records.values().map(|r| r.bytes).sum();
        for victim in &victims {
            crate::observe::off(format!(
                "pcache_shard_trim shard={victim:02x} live_mib={}/{}",
                live_bytes / (1024 * 1024),
                cap / (1024 * 1024),
            ));
            deletes.push(shard_path(&self.dir, *victim));
        }
        stats::TRIMMED.fetch_add(victims.len() as u64, Ordering::Relaxed);
        stats::SAVED.fetch_add(writes.len() as u64, Ordering::Relaxed);
        stats::DISK_BYTES.store(live_bytes, Ordering::Relaxed);
        if victims.is_empty() && !self.over_cap_reported && live_bytes > cap {
            self.over_cap_reported = true;
            crate::observe::off(format!(
                "pcache_shard_over_cap live_mib={}/{} (every shard is live this boot; the next boot trims)",
                live_bytes / (1024 * 1024),
                cap / (1024 * 1024),
            ));
        }
        let ack = Arc::new(AtomicBool::new(false));
        let wave_shards: HashSet<u32> = writes.iter().map(|(_, _, id, _)| *id).collect();
        submit_shard_wave(
            writes.into_iter().map(|(path, data, _, _)| (path, data)).collect(),
            Some((ledger_path(&self.dir), encode_ledger(&self.ledger))),
            deletes,
            ack.clone(),
        );
        self.awaiting = Some((wave_shards, ack));
    }

    /// Destroy every shard cache. The last wave was already submitted by the
    /// context's flush (the owner thread owns no Vulkan handles, so its
    /// in-flight file writes are unaffected by this).
    pub(crate) fn destroy(&mut self, device: &ash::Device) {
        for entry in self.entries.values() {
            unsafe { device.destroy_pipeline_cache(entry.cache, None) };
        }
        self.entries.clear();
    }
}

fn write_ledger(dir: &Path, ledger: &Ledger) -> std::io::Result<()> {
    let path = ledger_path(dir);
    // Distinct from the wave side's `ledger.tmp.<pid>`: a device-lost
    // recreate can open a new set (boot write, engine thread) while the old
    // context's last wave is still in flight (owner thread), and a shared
    // tmp name is the rename-under-each-other bug the monolithic path's doc
    // already describes.
    let tmp = dir.join(format!("ledger.boot.tmp.{}", std::process::id()));
    std::fs::write(&tmp, encode_ledger(ledger))?;
    match std::fs::rename(&tmp, &path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_file(&tmp);
            Err(error)
        }
    }
}

/// Process-lifetime census counters. The device-lost recreate path builds a
/// second `ShardSet` in one process; these deliberately do not reset, so a
/// window line reports the process, matching how the rest of the census reads.
mod stats {
    use std::sync::atomic::AtomicU64;

    pub(super) static OPENED: AtomicU64 = AtomicU64::new(0);
    pub(super) static HITS: AtomicU64 = AtomicU64::new(0);
    pub(super) static MISSES: AtomicU64 = AtomicU64::new(0);
    pub(super) static LOADED: AtomicU64 = AtomicU64::new(0);
    pub(super) static SAVED: AtomicU64 = AtomicU64::new(0);
    pub(super) static TRIMMED: AtomicU64 = AtomicU64::new(0);
    pub(super) static DISK_BYTES: AtomicU64 = AtomicU64::new(0);
}

/// One census window of the shard routes: hit/miss deltas beside levels, so
/// the worst second of a login storm can be read next to `draw_phase`'s
/// `pl_compile_us`. `None` when the lab flag is off (no line, no cost) or no
/// device ever opened a shard set.
pub(crate) fn census_window() -> Option<String> {
    if !enabled() || stats::OPENED.load(Ordering::Relaxed) == 0 {
        return None;
    }
    #[derive(Default)]
    struct Prev {
        hits: u64,
        misses: u64,
        loaded: u64,
        saved: u64,
        trimmed: u64,
    }
    static PREV: Mutex<Option<Prev>> = Mutex::new(None);
    let now = Prev {
        hits: stats::HITS.load(Ordering::Relaxed),
        misses: stats::MISSES.load(Ordering::Relaxed),
        loaded: stats::LOADED.load(Ordering::Relaxed),
        saved: stats::SAVED.load(Ordering::Relaxed),
        trimmed: stats::TRIMMED.load(Ordering::Relaxed),
    };
    let Ok(mut prev) = PREV.lock() else {
        return None;
    };
    let delta = Prev {
        hits: now.hits - prev.as_ref().map_or(0, |p| p.hits),
        misses: now.misses - prev.as_ref().map_or(0, |p| p.misses),
        loaded: now.loaded - prev.as_ref().map_or(0, |p| p.loaded),
        saved: now.saved - prev.as_ref().map_or(0, |p| p.saved),
        trimmed: now.trimmed - prev.as_ref().map_or(0, |p| p.trimmed),
    };
    *prev = Some(now);
    drop(prev);
    Some(format!(
        "pcache_shard hit={} miss={} loaded={} saved={} trim={} (deltas, not levels) \
         opens={} disk_kib={} cap_mib={}",
        delta.hits,
        delta.misses,
        delta.loaded,
        delta.saved,
        delta.trimmed,
        stats::OPENED.load(Ordering::Relaxed),
        stats::DISK_BYTES.load(Ordering::Relaxed) / 1024,
        total_cap() / (1024 * 1024),
    ))
}

#[cfg(test)]
mod pcache_shard_tests {
    use super::*;

    fn props() -> vk::PhysicalDeviceProperties {
        vk::PhysicalDeviceProperties {
            vendor_id: 0x10de,
            device_id: 0x2c02,
            pipeline_cache_uuid: [7u8; 16],
            ..vk::PhysicalDeviceProperties::default()
        }
    }

    /// A minimal valid VkPipelineCacheHeaderVersionOne: size 32, version ONE,
    /// this device's ids and UUID.
    fn valid_blob(len: usize) -> Vec<u8> {
        let mut blob = Vec::with_capacity(len.max(32));
        blob.extend_from_slice(&32u32.to_le_bytes());
        blob.extend_from_slice(&1u32.to_le_bytes());
        blob.extend_from_slice(&0x10deu32.to_le_bytes());
        blob.extend_from_slice(&0x2c02u32.to_le_bytes());
        blob.extend_from_slice(&[7u8; 16]);
        blob.resize(len.max(32), 0xEE);
        blob
    }

    fn test_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "reims-vgpu-pcache-shard-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The codec has to survive what a SIGKILLed boot can leave behind: a
    /// ledger read back must equal the one written, field for field.
    #[test]
    fn ledger_roundtrip_preserves_every_field() {
        let mut ledger = Ledger { boot_seq: 42, records: BTreeMap::new() };
        ledger.records.insert(
            3,
            ShardRecord { id: 3, bytes: 1234, last_used: 40, used: 7 },
        );
        ledger.records.insert(
            17,
            ShardRecord { id: 17, bytes: 5678, last_used: 42, used: 0 },
        );
        assert_eq!(parse_ledger(&encode_ledger(&ledger)), Some(ledger));
        assert_eq!(parse_ledger(&encode_ledger(&Ledger::default())), Some(Ledger::default()));
    }

    /// A corrupt or foreign ledger is refused outright — truncations, bad
    /// magic, trailing bytes, out-of-range ids — because half-read LRU
    /// metadata quietly deletes the wrong shard later.
    #[test]
    fn corrupt_ledger_is_refused() {
        let good = encode_ledger(&Ledger {
            boot_seq: 1,
            records: [(
                0u32,
                ShardRecord { id: 0, bytes: 8, last_used: 1, used: 1 },
            )]
            .into_iter()
            .collect(),
        });
        assert!(parse_ledger(&good).is_some());
        assert_eq!(parse_ledger(&[]), None, "empty");
        assert_eq!(parse_ledger(b"RVPSHL0X...."), None, "bad magic");
        assert_eq!(parse_ledger(&good[..good.len() - 1]), None, "truncated");
        let mut trailing = good.clone();
        trailing.push(0);
        assert_eq!(parse_ledger(&trailing), None, "trailing bytes");
        // An id past SHARD_COUNT claims a format with more shards than this
        // build knows what to do with.
        let mut foreign = Vec::new();
        foreign.extend_from_slice(&LEDGER_MAGIC);
        put_u64(&mut foreign, 1);
        put_u32(&mut foreign, 1);
        put_u32(&mut foreign, SHARD_COUNT);
        put_u64(&mut foreign, 0);
        put_u64(&mut foreign, 0);
        put_u64(&mut foreign, 0);
        assert_eq!(parse_ledger(&foreign), None, "out-of-range shard id");
    }

    /// A shard saved through the persistence owner and read back through the
    /// warm path is the roundtrip the whole scheme rests on: what one boot
    /// wrote is what the next boot warms from.
    ///
    /// The owner's mailbox holds one job process-wide, so a concurrently
    /// running test can displace this wave before it runs — the resubmit loop
    /// is the same idempotent wave until its ack observes a run, which is what
    /// makes this test independent of thread scheduling.
    #[test]
    fn shard_save_then_warm_read_roundtrips() {
        let dir = test_dir("roundtrip");
        let blob = valid_blob(96);
        let ack = Arc::new(AtomicBool::new(false));
        let wave = || {
            submit_shard_wave(
                vec![(shard_path(&dir, 9), blob.clone())],
                Some((ledger_path(&dir), encode_ledger(&Ledger::default()))),
                Vec::new(),
                ack.clone(),
            );
        };
        wave();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut last_submit = std::time::Instant::now();
        while !ack.load(Ordering::Relaxed) {
            assert!(std::time::Instant::now() < deadline, "the wave never ran");
            if last_submit.elapsed() >= std::time::Duration::from_millis(200) {
                wave();
                last_submit = std::time::Instant::now();
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(warm_blob(&dir, 9, &props()).unwrap(), Some(blob));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A damaged shard file is refused by name (the caller falls back cold)
    /// rather than fed to `vkCreatePipelineCache`, where a bad header is not
    /// an error but a valid-usage violation.
    #[test]
    fn a_corrupt_shard_is_ignored_not_fed_to_the_driver() {
        let dir = test_dir("corrupt");
        std::fs::write(shard_path(&dir, 4), b"not a pipeline cache").unwrap();
        assert_eq!(
            warm_blob(&dir, 4, &props()),
            Err(ShardDecline::Incompatible { shard: 4, bytes: 20 }),
            "a damaged file is a named refusal"
        );
        // A header from another device is equally refused.
        std::fs::write(shard_path(&dir, 5), valid_blob(64)).unwrap();
        let mut other = props();
        other.pipeline_cache_uuid = [8u8; 16];
        assert_eq!(
            warm_blob(&dir, 5, &other),
            Err(ShardDecline::Incompatible { shard: 5, bytes: 64 })
        );
        // And a missing file is the quiet first-boot case, not a failure.
        assert_eq!(warm_blob(&dir, 6, &props()), Ok(None));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A shard past the per-shard bound is declined by name — the
    /// single-giant-blob guard, which must refuse *before* the driver ever
    /// sees the bytes.
    #[test]
    fn an_oversize_shard_declines_instead_of_warming() {
        let dir = test_dir("oversize");
        std::fs::write(shard_path(&dir, 2), valid_blob(SHARD_MAX_WARM_BYTES + 1)).unwrap();
        assert_eq!(
            warm_blob(&dir, 2, &props()),
            Err(ShardDecline::TooLarge {
                shard: 2,
                bytes: SHARD_MAX_WARM_BYTES + 1,
                cap: SHARD_MAX_WARM_BYTES
            })
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Trim removes the oldest-`last_used` shards until the total fits, and
    /// the files go with the records.
    #[test]
    fn trim_removes_the_coldest_shards_until_under_cap() {
        let dir = test_dir("trim");
        let mut ledger = Ledger { boot_seq: 9, records: BTreeMap::new() };
        for (id, bytes, last_used) in [(1u32, 100u64, 3u64), (2, 100, 8), (3, 100, 1), (4, 100, 5)]
        {
            ledger.records.insert(id, ShardRecord { id, bytes, last_used, used: 0 });
            std::fs::write(shard_path(&dir, id), [0u8; 8]).unwrap();
        }
        let victims = trim_over_cap(&mut ledger, &dir, &HashSet::new(), 250);
        assert_eq!(victims, vec![3, 1], "coldest last_used first, oldest to newest");
        assert_eq!(ledger.records.keys().copied().collect::<Vec<_>>(), vec![2, 4]);
        assert!(!shard_path(&dir, 3).exists());
        assert!(!shard_path(&dir, 1).exists());
        assert!(shard_path(&dir, 2).exists());
        assert!(shard_path(&dir, 4).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A live shard is never a victim, and a cap below the working set leaves
    /// one record rather than an empty directory — the monolithic bound's
    /// delete-everything failure mode is what this must not re-introduce.
    #[test]
    fn a_cap_too_small_never_empties_the_directory_or_eats_a_live_shard() {
        let dir = test_dir("capfloor");
        let mut ledger = Ledger { boot_seq: 2, records: BTreeMap::new() };
        for (id, last_used) in [(0u32, 2u64), (1, 1)] {
            ledger
                .records
                .insert(id, ShardRecord { id, bytes: 100, last_used, used: 0 });
        }
        // Even a zero cap keeps the newest shard.
        let victims = trim_over_cap(&mut ledger, &dir, &HashSet::new(), 0);
        assert_eq!(victims, vec![1]);
        assert_eq!(ledger.records.len(), 1, "never everything");

        // And with the survivor live, no trim fires at all.
        let live: HashSet<u32> = [0].into();
        let mut ledger2 = Ledger { boot_seq: 3, records: BTreeMap::new() };
        ledger2
            .records
            .insert(0, ShardRecord { id: 0, bytes: 100, last_used: 2, used: 1 });
        assert!(trim_over_cap(&mut ledger2, &dir, &live, 0).is_empty());
        assert_eq!(ledger2.records.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Boot merges stray files the ledger does not know into `last_used = 0`
    /// records, so nothing hides from the budget; the boot sequence advances;
    /// the ledger lands back on disk with the trim applied.
    #[test]
    fn open_merges_stray_files_advances_the_boot_and_persists() {
        let dir = test_dir("open");
        std::fs::write(shard_path(&dir, 12), valid_blob(48)).unwrap();
        let set = ShardSet::open(&dir, &props());
        assert_eq!(set.boot_seq, 1, "first open is boot one");
        assert_eq!(set.ledger.records[&12].bytes, 48);
        assert_eq!(set.ledger.records[&12].last_used, 0, "a stray file is cold-ordered");
        drop(set);
        // Second boot reads the ledger the first wrote.
        let set = ShardSet::open(&dir, &props());
        assert_eq!(set.boot_seq, 2);
        let on_disk = std::fs::read(ledger_path(&dir)).unwrap();
        assert_eq!(parse_ledger(&on_disk).unwrap().boot_seq, 2);
        // A corrupt ledger resets to cold, not to a panic.
        std::fs::write(ledger_path(&dir), b"garbage").unwrap();
        let set = ShardSet::open(&dir, &props());
        assert_eq!(set.boot_seq, 1, "a reset ledger restarts the count");
        assert!(set.ledger.records.contains_key(&12), "the file still merges back in");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Shard routing spreads keys across the range and never leaves it, and
    /// the two digest helpers agree with themselves (stability is the whole
    /// contract: a pipeline must find its shard next boot).
    #[test]
    fn shard_routing_is_stable_and_in_range() {
        for digest in [0u64, 1, 63, 64, u64::MAX, 0xdead_beef] {
            assert!(shard_of(digest) < SHARD_COUNT);
        }
        assert_eq!(shard_of(7), shard_of(7));
        assert_ne!(shard_of(0), shard_of(63));
        assert_eq!(words_digest(&[1, 2, 3]), words_digest(&[1, 2, 3]));
        assert_ne!(words_digest(&[1, 2, 3]), words_digest(&[3, 2, 1]));
    }
}
