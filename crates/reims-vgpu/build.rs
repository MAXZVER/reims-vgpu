//! Product staticlib is pure Rust for protocol utilities.
//!
//! Pre-rewrite C packages lived under `host/utils/` (deleted; git history only).
//!
//! It computes two things. Whether the wire crate's oracle fixtures are on
//! disk, so `tests/wire_fixtures_reach_the_decoders.rs` stands down as `ignored`
//! rather than as `ok` when they are not. The probe is
//! `reims_vgpu_testkit::probe_wire_fixtures`, so the answer is shared with the
//! other two suites instead of decided three times.
//!
//! And the digest of what produces a shader translation, the stamp of the
//! on-disk translation cache (`runtime::m2v_cache`, module `disk`).

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    reims_vgpu_testkit::probe_wire_fixtures(&format!("{manifest}/../reims-vgpu-wire/fixtures"));
    emit_translator_digest(Path::new(&manifest));
}

/// FNV-1a, 64-bit: stable across toolchains, which `DefaultHasher` is not
/// promised to be, and enough to tell one source tree from the next.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn feed(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
        // A length suffix, so a file boundary cannot slide between two inputs.
        for b in (bytes.len() as u64).to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// `REIMS_VGPU_M2V_TRANSLATOR_DIGEST` and `REIMS_VGPU_M2V_TOOLCHAIN_DIGEST`.
///
/// The translator is metal2vulkan. When it is the path checkout the workspace
/// patches in (`../metal2vulkan` beside the workspace root), every `.rs` file
/// under its `src/` and its manifest are hashed; the workspace `Cargo.lock`
/// is hashed always, which is what names a git-pinned translator and every
/// crate it builds with. The toolchain digest is `rustc -V`: the cache keys its
/// file names with the standard library's `DefaultHasher`.
///
/// Before this the stamp was the running executable's size and mtime, so every
/// relink of QEMU — any change anywhere in it, or a copy of the same binary —
/// threw away every translation and the next boot translated ~340 shaders cold.
fn emit_translator_digest(manifest: &Path) {
    let root = manifest.join("../..");
    let mut h = Fnv::new();
    let lock = root.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    h.feed(&std::fs::read(&lock).unwrap_or_default());
    let m2v = root.join("../metal2vulkan");
    if m2v.join("Cargo.toml").is_file() {
        let src = m2v.join("src");
        println!("cargo:rerun-if-changed={}", src.display());
        println!(
            "cargo:rerun-if-changed={}",
            m2v.join("Cargo.toml").display()
        );
        let mut files = vec![m2v.join("Cargo.toml")];
        rust_files(&src, &mut files);
        files.sort();
        for f in files {
            let rel = f.strip_prefix(&m2v).unwrap_or(&f);
            h.feed(rel.to_string_lossy().replace('\\', "/").as_bytes());
            h.feed(&std::fs::read(&f).unwrap_or_default());
        }
    }
    println!(
        "cargo:rustc-env=REIMS_VGPU_M2V_TRANSLATOR_DIGEST={:016x}",
        h.0
    );

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = std::process::Command::new(rustc)
        .arg("-V")
        .output()
        .map(|o| o.stdout)
        .unwrap_or_default();
    let mut t = Fnv::new();
    t.feed(&version);
    println!(
        "cargo:rustc-env=REIMS_VGPU_M2V_TOOLCHAIN_DIGEST={:016x}",
        t.0
    );
}
