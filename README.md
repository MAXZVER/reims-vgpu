# reims-vgpu: Windows edition

[![Host: Windows 11](https://img.shields.io/badge/host-Windows%2011-0078D4)](#build-and-run-on-windows)
[![Hypervisor: QEMU + WHPX](https://img.shields.io/badge/hypervisor-QEMU%20%2B%20WHPX-E65100)](https://github.com/MAXZVER/qemu-reims-vgpu)
[![Guest: macOS 13 Ventura](https://img.shields.io/badge/guest-macOS%2013%20Ventura-555555)](#results)
[![GPU: Metal to Vulkan](https://img.shields.io/badge/GPU-Metal%20%E2%86%92%20Vulkan-AC162C)](#what-this-is)
[![Status: experimental](https://img.shields.io/badge/status-experimental-yellow)](#status-and-known-limitations)
[![License: unchanged from upstream](https://img.shields.io/badge/license-unchanged%20from%20upstream-blue)](#license)

**A macOS guest with a paravirtual GPU on a Windows 11 PC.** This fork of
[steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu) collects the work that makes
the Windows 11 + QEMU + WHPX host pathway fast. On one lab PC, a fullscreen Safari CSS animation
in a macOS 13 guest at 1920x1080 went from **5–7 fps to a median of ~93 fps**.

Nearly everything here is also offered upstream as pull requests; the fork-only commits are
listed under [What changed](#what-changed). This fork is not a replacement for upstream: it is a
place to try all of the pending Windows-host work in one build while those PRs are reviewed. The
original upstream README follows unchanged under [Upstream README](#upstream-readme).

## Latest results (2026-09-29)

These come from the same lab PC (RTX 4060, Windows 11 + WHPX), with a macOS Ventura 13.7.8
guest. The figures are median present rates in Hz. Boot-to-boot variance in the lab is large, so
read them as ranges. They were measured on the lab's development trees, which carry the changes
below (in both this repository and the QEMU fork) plus temporary diagnostics; the published
branches have been compile-checked and unit-tested, not re-benchmarked on their own.

| Guest | CSS animation | Full-screen CSS scroll | Wheel scroll |
|---|---|---|---|
| 1920x1080 | ~92–97 | ~116–118 | ~88–96 |
| 5120x2160 | ~98–106 | ~47–51 | ~63–71 |

What is new:

- **Page-diff writeback is the default.** A presented framebuffer is written back into guest RAM
  page by page. The GPU compares the frame with a copy of what was last written back, and only
  the 4 KiB pages that changed cross PCIe, together with the pages the guest CPU wrote since the
  last write-back. That second part addresses a 4 KiB page seen stuck stale for about 20 s
  during a 5K full-screen scroll. At 5K, page-diff raised CSS animation from ~93 to ~100–106,
  wheel scroll from ~64 to ~71 and window drag from ~23 to ~28–30. At 1080p nothing changed
  (82 / 117 / 92 with it against 81 / 117 / 92 without). `REIMS_VGPU_PAY_DIFF=off` writes whole
  frames back instead.
- **5K guests.** `REIMS_VGPU_DISPLAY_NATIVE=WxH` sets the panel's native mode, and the sampled
  cache holds 512 MiB instead of 128 MiB so that a 5K desktop's working set fits.
- **QEMU side: the LLP64 dirty-bitmap fix.** On Windows `long` is 32 bits, and QEMU's bitmap
  import used the pointer width for its words, so the WHPX dirty syncs dropped or misplaced bits
  in every unaligned range. The dropped bits were the stale tile pages; on full-screen scroll the
  streak detector now reads 0 stale-page streaks. The misplaced bits had been marking random
  surfaces as written, which cost a lot at 5K; fixing them was a large part of the 5K speedup.
- **QEMU side: on-demand dirty sync is the default.** The drain no longer waits for a dirty-log
  harvest per doorbell. `REIMS_VGPU_DIRTY_ONDEMAND=off` restores the per-doorbell harvest.

Known limits:

- 5K does not reach 120 Hz yet. Full-screen scrolling is the slowest case at ~47–51.
- At 5K the remaining cost is the per-doorbell hypervisor dirty-log queries, at about 300
  doorbells a second. Work on running the prefetch in parallel is in progress.
- A macOS 26 Tahoe guest has not been tested in the lab yet.

The older tables below are kept for comparison.

## What this is

[reims-vgpu](https://github.com/steelbrain/reims-vgpu) implements the QEMU device that macOS's
built-in `AppleParavirtGPU.kext` attaches to. It decodes the guest's Metal command stream on the
host and runs it through Vulkan via [`metal2vulkan`](https://github.com/steelbrain/metal2vulkan).
There is no custom kext and no guest driver to install. Upstream targets Linux/KVM and Apple
Silicon hosts. [@Hi-Jiajun](https://github.com/Hi-Jiajun) added a Windows host port upstream
([reims-vgpu#57](https://github.com/steelbrain/reims-vgpu/pull/57)).

This fork builds on that port and focuses on performance and correctness on Windows 11 with QEMU's
WHPX accelerator (Windows Hypervisor Platform) and a native Vulkan driver.

## Results

The number is the present rate: frames per second the guest actually presents, from reims-vgpu's
own census log (`present_hz`, one-second windows).

| Scenario | Before this work (same PC) | `windows` branch changes ¹ | + direct IRQ (experimental) ² |
|---|---|---|---|
| CSS keyframe animation, fullscreen Safari | 5–7 | median ~93 (max ~114) | median ~107 (max ~119) |
| Safari scroll | 5–7 | median ~78–96, mean 66–76 | re-measuring |
| Window drag | 5–7 | median ~40 | ~35–49 |

**Test conditions**

- Host: Intel Core i7-14700K, 96 GB RAM, NVIDIA GeForce RTX 4060 8 GB, Windows 11 Pro. The host
  display is 5120x2160 at 120 Hz.
- Guest: x86_64 macOS 13 Ventura (OSX-KVM OpenCore) at 1920x1080. 8 vCPUs, 12 GB RAM,
  QEMU + WHPX, `reims-vgpu-pci` with the host window, `REIMS_VGPU_GUEST_IMPORT=off`.
- Scripted workloads driven over QMP and SSH, 15–35 s per scenario, several boots per
  configuration. Ranges span boots.
- ¹ Measured on the development tree this branch was cut from. That tree also carried upstream
  PR [#78](https://github.com/steelbrain/reims-vgpu/pull/78) (dependency-graph compaction, by
  [@redvulps](https://github.com/redvulps)), which is *not* merged here, plus the QEMU-side changes
  listed [below](#qemu-side). The `windows` branch itself has been compile-checked and unit-tested.
  It has not yet been re-benchmarked on its own.
- ² Not in this branch yet.

These are single-machine lab numbers at 1080p, not 5K. A steady 120 fps is the goal and has not
been reached yet. Window drag is the weak spot.

## What changed

### reims-vgpu (merged into `windows`)

All six are open pull requests against upstream.

| PR | Change | Effect |
|---|---|---|
| [#110](https://github.com/steelbrain/reims-vgpu/pull/110) | Hold doorbell work until the QEMU shim's dirty-log harvests settle (ABI v21) | Lets QEMU harvest the dirty log off the vCPU (qemu #7) without the drain serving work before its harvest. Merge first. |
| [#105](https://github.com/steelbrain/reims-vgpu/pull/105) | Retire finished window presents on the maintenance tick | Fixes a VRAM leak: the image slab grew until Vulkan allocations failed. |
| [#106](https://github.com/steelbrain/reims-vgpu/pull/106) | Narrow GPU writes through the guest-RAM import | Two off-switches that split the import's reads from its writes. They name the write class behind the `AppleParavirtPageTable` panics ([#70](https://github.com/steelbrain/reims-vgpu/issues/70), [#71](https://github.com/steelbrain/reims-vgpu/issues/71), [#100](https://github.com/steelbrain/reims-vgpu/issues/100)). |
| [#107](https://github.com/steelbrain/reims-vgpu/pull/107) | Consume the resident merge's readback in place | Drops a whole-frame buffer fill per merge. |
| [#108](https://github.com/steelbrain/reims-vgpu/pull/108) | GPU resident overlay | Copies only the pages the guest CPU wrote onto the live GPU resident, instead of a whole-frame readback-and-merge. The biggest single win. |
| [#109](https://github.com/steelbrain/reims-vgpu/pull/109) | Write presented framebuffers back at DisplaySwap | Fixes the persistent 1-px horizontal stale-line artifacts. |

### reims-vgpu, fork only

These are in `windows` (from the `windows-5k` branch) and are not upstream pull requests yet:

- `REIMS_VGPU_DISPLAY_NATIVE=WxH` sets the panel's native mode, for 5K guests.
- The sampled cache's byte cap is 512 MiB instead of 128 MiB, which keeps ~11 full 5K surfaces
  instead of ~3.
- Page-diff writeback of presented framebuffers, on by default (`REIMS_VGPU_PAY_DIFF=off` turns
  it off), including the pages the guest CPU wrote since the last write-back.
- An indexed draw bounds its vertex binds by its largest index, instead of staging every
  vertex-indexed buffer whole.

### QEMU side

These live in the QEMU fork, not in this repository:

- [qemu-reims-vgpu#6](https://github.com/steelbrain/qemu-reims-vgpu/pull/6): keeps the timer
  resolution while QEMU's windows are occluded. Without it, Windows 11 rounds waits up to 15.6 ms,
  and the device's VBL heartbeat drops to 64 Hz.
- [qemu-reims-vgpu#7](https://github.com/steelbrain/qemu-reims-vgpu/pull/7): harvests the dirty
  log off the vCPU. Each harvest used to hold the vCPU and the BQL for ~10 ms on WHPX. It needs
  ABI v21 from #110.
- The lab QEMU also carries [@Hi-Jiajun](https://github.com/Hi-Jiajun)'s WHPX fixes
  ([qemu-reims-vgpu#4](https://github.com/steelbrain/qemu-reims-vgpu/pull/4)) and WHPX dirty-page
  tracking with the LLP64 bitmap fix
  ([qemu-reims-vgpu#10](https://github.com/steelbrain/qemu-reims-vgpu/pull/10)).
- On-demand dirty sync, the default in the QEMU fork's `windows` branch, has not been proposed
  upstream yet.

## Status and known limitations

- **Experimental.** Upstream calls itself alpha, and this fork is no further along than that.
  Expect glitches, and don't use it for anything you can't lose.
- **120 fps is not stable yet.** The CSS animation peaks at ~114–119 fps, but its median is
  ~93 fps (~107 with the experimental direct IRQ).
- **Window drag is the weak spot** at a median of ~40 fps.
- **One machine.** Only an Intel CPU with an NVIDIA RTX 4060 has been measured. Other GPUs are
  untested on this branch. Hi-Jiajun validated his WHPX fixes on an AMD Ryzen host.
- **Guest-RAM import.** On this host the import panics the guest (#70/#71/#100), so every
  number above uses `REIMS_VGPU_GUEST_IMPORT=off`. With the import on,
  `REIMS_VGPU_GUEST_IMPORT_SURFACE_WRITES=off` (#106) keeps its read side and avoids the class that
  crashed here. Whether that is faster on this host is not settled.
- **`vendor/qemu` still points at upstream's shim branch.** See
  [Build and run](#build-and-run-on-windows).
- **Only tested on macOS 13**, at 1920x1080 and, since 2026-09-29, 5120x2160. 5K does not reach
  120 Hz yet, and no macOS 26 Tahoe guest has been tried.

### Roadmap

1. A steady 120 fps at 1920x1080, including window drag.
2. A 5120x2160 guest at 120 fps. It runs now; see [Latest results](#latest-results-2026-09-29).
3. A macOS 26 Tahoe guest.

Each step goes upstream as PRs, as before.

## Build and run on Windows

> **Verification status.** The build command and QEMU command line below are the ones our lab
> runs. The lab drives them through private scripts with machine-specific paths, so steps 1–5 as
> written here have **not** been re-run end-to-end from a fresh clone. Open an issue if a step
> fails for you.

### Prerequisites

- Windows 11 with the **Windows Hypervisor Platform** optional feature enabled. QEMU's
  `-accel whpx` needs it.
- A Vulkan driver for your GPU. The lab uses the NVIDIA driver on an RTX 4060.
- [MSYS2](https://www.msys2.org/), **UCRT64** environment. Upstream's Windows port used MINGW64,
  which is untested here. This is the package set on the lab machine; it has not been trimmed to a
  minimal set:

  ```sh
  pacman -S --needed git make bison flex diffutils \
    mingw-w64-ucrt-x86_64-{gcc,binutils,pkgconf,ninja,meson,python,glib2,pixman,zstd} \
    mingw-w64-ucrt-x86_64-{libslirp,SDL2,curl,rust,vulkan-headers,vulkan-loader} \
    mingw-w64-ucrt-x86_64-{llvm-tools,spirv-tools,qemu-image-util}
  ```

  The Metal→Vulkan translator runs `llvm-dis` (from `llvm-tools`) and `spirv-val` (from
  `spirv-tools`) at runtime. Start QEMU with the UCRT64 `bin` directory on `PATH`, which also
  supplies QEMU's DLLs.
- Docker Desktop, only for building the option ROM the way we did (step 3).

### 1. Get the sources

```sh
git clone --recurse-submodules https://github.com/MAXZVER/reims-vgpu.git
cd reims-vgpu            # the default branch is `windows`
```

`vendor/qemu` still points at upstream's
[`host-reims-vgpu-vmapple`](https://github.com/steelbrain/qemu-reims-vgpu/tree/host-reims-vgpu-vmapple)
branch. At that commit, `hw/display/reims-vgpu-pci.c` does not compile on Windows (an unguarded
`munmap`), and it lacks the WHPX fixes above. Check out the companion QEMU fork in the submodule
instead:

```sh
cd vendor/qemu
git remote add maxzver https://github.com/MAXZVER/qemu-reims-vgpu.git
git fetch maxzver
git checkout <windows-integration-branch>   # named in that repository's README
cd ../..
```

### 2. Build QEMU with the reims-vgpu device

In an MSYS2 UCRT64 shell at the repository root:

```sh
REIMS_VGPU_BACKEND=vulkan scripts/qemu-build/qemu-build.sh --target x86_64 --backend vulkan
```

This builds the Rust staticlib and links it into `vendor/qemu/build/qemu-system-x86_64.exe`.
Upstream's Windows port notes that non-ASCII Windows locales need `PYTHONUTF8=1` for this step.

### 3. Build the UEFI GOP option ROM

`crates/reims-vgpu-efi/scripts/reims-vgpu-efi-rom/reims-vgpu-efi-rom.sh` produces
`crates/reims-vgpu-efi/out/reims-vgpu-gop.rom`. It needs `rustup` with the `x86_64-unknown-uefi`
target, and `python3`. We ran it in the official Rust container, from PowerShell at the repository
root:

```powershell
docker run --rm -v "${PWD}:/src" -w /src rust:1-slim bash -c "apt-get update -qq && apt-get install -y -qq python3 && bash crates/reims-vgpu-efi/scripts/reims-vgpu-efi-rom/reims-vgpu-efi-rom.sh"
```

A native Windows build of the ROM was not tried.

### 4. Prepare a macOS 13 Ventura guest

This repository ships no disk images, firmware variables or OpenCore blobs. Upstream's
[x86_64 guest steps](#x86_64-guest-on-linux-kvm) use [OSX-KVM](https://github.com/kholia/OSX-KVM)
for OpenCore, OVMF and the install. On our Windows host we installed the guest with
[dockur/macos](https://github.com/dockur/macos) under Docker Desktop (WSL2). We then booted the
disk natively under QEMU + WHPX with OSX-KVM's OpenCore image and OVMF. You end up with:

| File | Role |
|---|---|
| `OVMF_CODE_4M.fd` | UEFI firmware, read-only |
| `OVMF_VARS.fd` | UEFI variables (a copy of OSX-KVM's `OVMF_VARS-1920x1080.fd`) |
| `OpenCore.qcow2` | OpenCore boot disk |
| `macos.img` | the installed guest disk (qcow2) |
| `reims-vgpu-gop.rom` | the option ROM from step 3 |

Tips from our setup:

- We converted dockur's raw disk with `qemu-img convert -O qcow2 data.img macos.img`.
- To keep the installed disk pristine, boot a throwaway overlay made with
  `qemu-img create -f qcow2 -b macos.img -F qcow2 macos-run.qcow2`, and point the script's
  `MacHDD` drive at it.
- If OpenCore lists the installer's EFI partition first, pick the second entry, the system volume.
- Enable Remote Login in the guest if you want SSH.

### 5. Boot

`vm/boot-windows.sh`, from upstream's Windows port, boots this configuration from an MSYS2 shell.
It expects the files above in `C:/hackintosh/vm`; edit `VM_DIR` at the top of the script to use
another folder.

```sh
export REIMS_VGPU_GUEST_IMPORT=off   # what every number above was measured with
vm/boot-windows.sh
```

- The script sets `REIMS_VGPU_WINDOW=on`, so the guest appears in a host window that reims-vgpu
  owns.
- The script defaults to 16 vCPUs and 16 GB. Our lab runs the same devices with `-smp 8` and
  `-m 12G`.
- QMP listens on `127.0.0.1:4444`. The script's SSH forward (`hostfwd=tcp::2222-:22`) binds every
  host interface, so bind it to `127.0.0.1` if that matters on your network.
- reims-vgpu writes its always-on log to `/tmp/reims-vgpu-fail.log`. On Windows that means
  `\tmp\reims-vgpu-fail.log` on the drive of QEMU's working directory. Create that `\tmp` folder
  first, because the device does not create it.

## Companion QEMU fork

The QEMU side lives in **[MAXZVER/qemu-reims-vgpu](https://github.com/MAXZVER/qemu-reims-vgpu)**,
a fork of [steelbrain/qemu-reims-vgpu](https://github.com/steelbrain/qemu-reims-vgpu). That is
where the Windows QEMU work is collected: WHPX fixes, #6 and #7. Its Windows integration branch was
still being assembled when this was written, so check that repository for the current branch.

## Credits

- **[steelbrain](https://github.com/steelbrain)** wrote reims-vgpu, `metal2vulkan` and the QEMU
  shims. All of the real work is upstream; this fork only adds to it.
- **[@Hi-Jiajun](https://github.com/Hi-Jiajun)** did the Windows host port
  ([reims-vgpu#57](https://github.com/steelbrain/reims-vgpu/pull/57),
  [qemu-reims-vgpu#2](https://github.com/steelbrain/qemu-reims-vgpu/pull/2)) and the WHPX fixes
  ([qemu-reims-vgpu#3](https://github.com/steelbrain/qemu-reims-vgpu/pull/3),
  [#4](https://github.com/steelbrain/qemu-reims-vgpu/pull/4)) that got macOS guests booting under
  WHPX.
- **[@redvulps](https://github.com/redvulps)** wrote the dependency-graph compaction
  ([#78](https://github.com/steelbrain/reims-vgpu/pull/78)) that our lab tree carried.
- **The [QEMU](https://www.qemu.org/) project**, including its WHPX accelerator.
- **[OSX-KVM](https://github.com/kholia/OSX-KVM)** and **[dockur/macos](https://github.com/dockur/macos)**
  provided the guest bring-up.

## License

The license is unchanged from upstream. The repository's [LICENSE](LICENSE) file is the GNU LGPL
v3, and the upstream README below states `LGPL-3.0-or-later`. The crates' `Cargo.toml` files
declare `GPL-2.0-or-later`. The vendored QEMU carries its own licenses, mainly GPL-2.0. This fork
keeps every license file and notice as it is. Its changes are contributed under the same terms as
the files they touch.

## Legal note

Apple's macOS license permits running macOS in a virtual machine only on Apple-branded hardware.
Running a macOS guest on a Windows PC falls outside those terms. This is a research and
interoperability project about GPU virtualization. It ships no Apple software, and you are
responsible for complying with the licenses of whatever you run. It is not affiliated with,
sponsored by, or endorsed by Apple Inc. Metal and macOS are trademarks of Apple Inc.

## AI-assisted development

Development of this fork was AI-assisted. The investigation, code and PR write-ups were done with
Claude (Anthropic) under the maintainer's direction and review. Commits carry a
`Co-Authored-By: Claude` trailer. Every number on this page comes from real runs on the machine
described above.

---

# Upstream README

> Everything below is the README of
> [steelbrain/reims-vgpu](https://github.com/steelbrain/reims-vgpu), unchanged.

# reims-vgpu

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/License-LGPL%203.0%20or%20later-blue.svg)](LICENSE) [![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/D2AM9mrDgs)

> **Alpha.** This project is early and under active development. The QEMU device ABI, boot scripts,
> crate layout, backend behavior, and supported host/guest pathways may change without a stable
> compatibility guarantee. Treat it as research-quality: useful for experimentation and bring-up,
> not a frozen virtualization product.

reims-vgpu is an experimental virtual GPU for macOS guests. It aims to let macOS running inside a
VM use accelerated graphics instead of a basic framebuffer, while keeping the guest operating system
unchanged.

macOS already includes a paravirtual GPU driver named `AppleParavirtGPU.kext`.
reims-vgpu provides the QEMU device that driver attaches to, then decodes the guest's GPU command
stream on the host and executes it through Metal (TODO) or Vulkan, with Vulkan translation handled
by [`metal2vulkan`](https://github.com/steelbrain/metal2vulkan). There is no custom macOS kext and
no guest driver to install.

Contributions are welcome. I am especially interested in collaborating with developers who want to
work on correctness, visual glitches, synchronization bugs, command-stream decoding, Metal/Vulkan
translation, and making more host/guest combinations reliable.

![reims-vgpu running an arm64 macOS 13 Ventura guest desktop on an Apple Silicon host](assets/readme/reims-vgpu-macos-arm64-desktop.png)

*arm64 macOS 13 Ventura guest on an Apple Silicon host.*

![reims-vgpu running an x86_64 macOS 13 Ventura guest desktop on a Linux host](assets/readme/reims-vgpu-macos-x86-desktop.png)

*x86_64 macOS 13 Ventura guest on a Linux host.*

## Three pathways

`crates/reims-vgpu` targets the following host/guest/backend combinations. Agents pick the pathway
their unit of work is on.

| Pathway | Host | Guest | Device attach | Backend | Boot |
|---|---|---|---|---|---|
| **x86 macOS / Linux Vulkan** | Linux x86_64 (KVM) | x86_64 macOS Metal guest | PCI `reims-vgpu-pci` | host **Vulkan** via `metal2vulkan` | `vm/boot-x86.sh` |
| **arm64 macOS / macOS Metal** | Apple Silicon macOS (HVF) | arm64 macOS Metal guest (`vmapple`) | sysbus MMIO `reims-vgpu-mmio` | host **Metal** | `vm/boot-arm64.sh` |
| **arm64 macOS / macOS Vulkan** | Apple Silicon macOS (HVF) | arm64 macOS Metal guest (`vmapple`) | sysbus MMIO `reims-vgpu-mmio` | host **Vulkan** via `metal2vulkan` through MoltenVK | `vm/boot-arm64.sh` |

- QEMU device shims: `vendor/qemu` tracks
  [`steelbrain/qemu-reims-vgpu@host-reims-vgpu-vmapple`](https://github.com/steelbrain/qemu-reims-vgpu/tree/host-reims-vgpu-vmapple)
  (thin C — QOM/MMIO/IRQ/console/HostOps only)
- Product logic: `crates/reims-vgpu` (decode + device model + Metal/Vulkan backends)
- Wire layouts: `crates/reims-vgpu-wire` (derived serializer views/parsers; decode uses these as the layout authority for covered records)
- Vulkan translator dependency: public `steelbrain/metal2vulkan` Git crate. On macOS, the Vulkan
  host backend runs through MoltenVK.
- VM lifecycle: `vm/` (snapshot-revert; arm and x86 guest boot scripts)

## Getting started

This tree ships **boot scripts and the device**, not a ready-made macOS disk image. Guest disks,
firmware vars, and OpenCore blobs are private/gitignored under `vm/`. Pick a pathway, provision a
guest once, freeze a golden snapshot, then use the snapshot-revert boots for day-to-day work.
macOS 13 Ventura is the recommended guest release for bring-up.

### x86_64 guest on Linux (KVM)

1. **Host prep.** You need KVM (`/dev/kvm`), a working NVIDIA (or other) Vulkan stack for the product
   backend, and build deps for the in-tree QEMU (`scripts/qemu-build/qemu-build.sh --target x86_64
   --backend vulkan`).

2. **Generate OpenCore, OVMF, and a guest disk with [OSX-KVM](https://github.com/kholia/OSX-KVM).**
   **macOS 13 is recommended**.Follow that project’s docs to fetch recovery media, build OpenCore,
   and install macOS under QEMU+KVM. The point of this step is only to produce a
   **working, post-Setup-Assistant guest** plus the usual OpenCore/OVMF pieces — not to stay on
   OSX-KVM’s long-term launcher.

3. **Drop the artifacts where this repo expects them** (paths are the defaults in `vm/boot-x86.sh`;
   override with env if you prefer):

   | Artifact | Default location |
   |---|---|
   | Guest system disk | `vm/disks/macos.img` |
   | OpenCore boot disk | `vm/disks/OpenCore.qcow2` |
   | OVMF code | `vm/ovmf/OVMF_CODE_4M.fd` |
   | OVMF vars template | `vm/ovmf/OVMF_VARS-1920x1080.fd` |

   Finish install in the guest: enable Remote Login, install your SSH key, turn off sleep/screensaver
   as you like. Host SSH is typically `localhost:2222` → guest `:22` (see `vm/boot-x86.sh`).

4. **Capture the first immutable snapshot.** Guests are organised into **rails** — one rail per guest
   OS line (`macos-11` … `macos-26`), each with a snapshot history of its own under
   `vm/disks/rails/<rail>/snapshots/`. Create the rail's directory, then from a clean guest state
   (logged in, network/SSH known-good) shut down cleanly while booting in capture mode:

   ```bash
   mkdir -p vm/disks/rails/macos-15
   vm/boot-x86.sh --rail macos-15 --capture --device vmware-svga
   # clean shutdown from inside the guest → new label under
   # vm/disks/rails/macos-15/snapshots/, and that rail's snapshots/current points at it
   ```

   Every later boot clones the selected rail's `snapshots/current` (COW when possible) and **throws
   the clone away** on exit, so wedges and hard kills never poison the golden image.

   Importing a guest built elsewhere is the same shape without the boot — drop
   `{macos.img,OpenCore.qcow2,OVMF_VARS.fd}` (plus `OVMF_CODE.fd` if that guest was installed under
   a different OVMF build) into `vm/disks/rails/<rail>/snapshots/base/`, `chmod 444` them, and
   `ln -sfn base vm/disks/rails/<rail>/snapshots/current`. Use `cp --reflink=auto` on btrfs and the
   import costs no disk.

5. **Day-to-day boots.**

   ```bash
   vm/boot-x86.sh --list-rails                  # what guest lines exist (* = default)
   vm/boot-x86.sh --rail macos-15 --list-snapshots

   # Console only (mainstream OSX-KVM-style VGA) while you debug the host stack
   vm/boot-x86.sh --testing --device vmware-svga

   # Product Reims VGPU device (needs in-tree QEMU + reims-vgpu Vulkan)
   REIMS_VGPU_BACKEND=vulkan scripts/qemu-build/qemu-build.sh --target x86_64
   vm/boot-x86.sh --testing --device reims-vgpu-pci --rail macos-15

   # Host-window screenshot (Linux/Plasma or macOS host)
   scripts/screenshot/screenshot.sh -o /tmp/screen.png
   ```

   Without `--rail` a boot follows `vm/disks/rails/current`; change it with
   `ln -sfn <rail> vm/disks/rails/current`. Neither `--rail` nor `--snapshot` repoints anything.

### arm64 guest on Apple Silicon (HVF / vmapple)

Arm bring-up is **in-tree**: Virtualization.framework via Homebrew **`macosvm`**, then QEMU’s
`vmapple` machine under HVF. There is no OSX-KVM step.

1. Install **`macosvm`**, and build the vendored QEMU:

   ```bash
   scripts/qemu-build/qemu-build.sh --target aarch64 --backend metal
   ```

2. Provision a guest from a UniversalMac IPSW with the project helpers in
   `scripts/vmapple-provision/`. The live bundle lives under `vm/guest/` (disk, aux, `vm.json` /
   ECID).

3. Configure the guest once: enable Remote Login, run `scripts/vmapple-guest-config/` for no-sleep
   settings, and optionally enable auto-login by hand in System Settings. Capture a golden under
   `vm/guest/rails/<rail>/snapshots/` with the snapshot helpers (`scripts/vmapple-snapshot/`, or
   `vm/boot-arm64.sh --rail <rail> --capture` once the disk is ready).

4. Boot:

   ```bash
   vm/boot-arm64.sh --testing --device reims-vgpu-mmio    # product
   vm/boot-arm64.sh --testing --device apple-gfx-mmio   # Apple ParavirtualizedGraphics A/B
   scripts/screenshot/screenshot.sh /tmp/screen.png
   ```

   Optional **performance ceiling** reference: the same guest under native VZ via `macosvm --gui`.

### After the first snapshot

- Prefer **`--testing`** for agent/measurement boots (time-bounded, always reverts).
- Use **`--interactive`** when you need an open-ended GUI session (still reverts unless you are in
  `--capture` mode).
- Say which rail a result came from. A number from `macos-11` and a number from `macos-26` are two
  measurements, not one — that separation is the whole reason snapshots are per-rail.
- Never commit disks, IPSWs, or OpenCore/OVMF runtime under `vm/`.
- Device/backend work lives in `crates/reims-vgpu` + the thin shims in `vendor/qemu`; rebuild QEMU after
  product changes before claiming a live boot result.

### The host window takes your keyboard shortcuts

On the `reims-vgpu-pci` / `reims-vgpu-mmio` device the guest is displayed in a window this project
owns, and while that window has keyboard focus it asks the host desktop to **stop acting on its own
shortcuts** so they reach the guest instead. Without that the desktop consumes them first: a stock
Plasma session claims 63 `Meta`/`Alt`/`Ctrl` combinations, and because a macOS guest reads host
`Meta` as `Cmd`, that covers most of what the guest expects — `Cmd+A`, `Cmd+V`, `Cmd+Q`, `Cmd+W`,
`Cmd+1`…`Cmd+9`, and `Alt+Tab`.

**Press `Ctrl+Alt+Esc` to release the grab.** While it is held your own `Alt+Tab` goes to the guest,
so this is how you get back to the host desktop. The chord is consumed rather than forwarded, and the
grab re-arms by itself the next time you focus the window — it is an escape hatch, not a mode you
have to remember you are in. The guest's own `Cmd+Option+Esc` (Force Quit) carries no `Ctrl` and is
forwarded to the guest untouched.

The window says so on stderr the first time it captures, and records it in the always-on log:

```text
window_capture_engaged mechanism=wayland_shortcuts_inhibit release=Ctrl+Alt+Esc
```

How much can be captured depends on the host, and the log names which mechanism a boot got:

| Host | Mechanism | Coverage |
|---|---|---|
| Wayland | `zwp_keyboard_shortcuts_inhibit_v1` | full, when the compositor implements it |
| X11 | `XGrabKeyboard` | full, unless another client holds the keyboard |
| macOS | `NSApplicationPresentationDisableProcessSwitching` | partial — `Cmd+Tab` and `Cmd+H` only; the window server keeps its reserved chords |

A host that cannot capture at all still runs; it emits a `window_capture_*` reason on
`/tmp/reims-vgpu-fail.log` rather than silently dropping the keys.

### Environment overrides

Set on the boot command; every one is optional and every default is "let the device decide". The
full list, with the parse, is `crates/reims-vgpu/src/env.rs`. Each accepts `1`/`on`/`true`/`yes` and
`0`/`off`/`false`/`no`, case-insensitively.

| Variable | Effect |
|---|---|
| `REIMS_VGPU_DMABUF=off` | Stop reaching guest pages through a dma-buf, even where the host can. Every guest-memory rail takes the copying path instead — which is what runs on any host without `VK_EXT_external_memory_dma_buf`, so this is how that half is exercised on a machine that has it. |
| `REIMS_VGPU_DRAW_LOG=on` | Verbose per-draw detail on top of the always-on failure log. |

An override can only **narrow** what the device does. There is no way to switch a rail *on* that the
host reported it cannot run: capability is measured from the device at startup, and asking a driver
for an extension it does not advertise fails device creation rather than degrading. `REIMS_VGPU_DMABUF`
has no on direction for that reason — on a host without the extension it is already off, and the
`vk_caps` line in `/tmp/reims-vgpu-fail.log` names which check said so.

## Repo layout

```text
AGENTS.md           - repo operating guide for agents
crates/             - Rust crates (`reims-vgpu`, `reims-vgpu-wire`, `reims-vgpu-efi`)
scripts/            - host setup, VM lifecycle, screenshot, and diagnostic helpers
vendor/             - vendored QEMU submodule and patch record
vm/                 - VM launch/configuration glue; images are private/untracked
```

`crates/reims-vgpu-wire` holds zero-copy views and parsers for the Apple
paravirtualized GPU serializer format, derived from Apple's own encoder rather
than inferred from captures. `crates/reims-vgpu`'s `runtime::decode` uses those
exports for opcodes, record framing, and field layouts on wire-covered
families (encoder blit/compute/render binds and state, and the create records
above); decode remains the mapping layer into the device's `Command` / `Kind`
model and decline naming. Gaps without a wire export (FIFO, event opcodes,
unobserved compute residency, pipeline TLV) stay local to decode.


## License

Licensed under the [GNU Lesser General Public License v3.0 or later](LICENSE)
(`LGPL-3.0-or-later`).

Metal, macOS are trademarks of Apple Inc. reims-vgpu is an independent project and is not affiliated
with, sponsored by, or endorsed by Apple Inc.
