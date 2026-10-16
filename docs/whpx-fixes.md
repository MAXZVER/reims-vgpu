# Windows WHPX Performance & Stability Architecture Guide

This document details the architecture, root causes, and resolutions for hypervisor, lifecycle, and GPU presentation bottlenecks on the **Windows 11 + WHPX + QEMU** pathway hosting macOS 26 Tahoe and earlier guests via `AppleParavirtGPU`.

---

## Table of Contents

1. [Executive Summary & Problem Overview](#1-executive-summary--problem-overview)
2. [Issue #3: Cursor Doorbell Cache Thrashing & MMIO Serialization](#2-issue-3-cursor-doorbell-cache-thrashing--mmio-serialization)
3. [Issue #4: Per-Surface WHPX Dirty Sync & Hypercall Flooding](#3-issue-4-per-surface-whpx-dirty-sync--hypercall-flooding)
4. [Issue #5: Memory Lifecycle, Late Writes (`released_write_after_release`), & v8–v14 Hardening](#4-issue-5-memory-lifecycle-late-writes-released_write_after_release--v8v14-hardening)
5. [Issue #6: Display Refresh Rate, Pacing, & Stale-Hold Protection](#5-issue-6-display-refresh-rate-pacing--stale-hold-protection)
6. [Steps A/B/C: Residency Promotion & Ordered Scanout Handoff](#6-steps-abc-residency-promotion--ordered-scanout-handoff)
7. [v15–v16: Wallpaper Synthesis & Raster Sample Count Specialization](#7-v15v16-wallpaper-synthesis--raster-sample-count-specialization)
8. [Configuration & Diagnostic Reference](#8-configuration--diagnostic-reference)

---

## 1. Executive Summary & Problem Overview

When virtualizing macOS Tahoe on Windows 11 under Windows Hypervisor Platform (WHPX) with NVIDIA RTX hardware, three severe failure modes emerged:

1. **Mouse Drag Stutter (1–3 Hz)**: Dragging windows interactively with the mouse dropped frame rates to 1–3 Hz, while scripted programmatic moves ran smoothly at 24–60 Hz.
2. **Display Freezes & Kernel Panics**: After prolonged interactive sessions or intensive app switching, the VM froze with black rectangles and `AppleParavirtPageTable.cpp:201` panics.
3. **High CPU Utilization & Frame Dropping**: The host presentation loop spun at 450–700 Hz in mailbox mode, flooding the hypervisor with 60,000+ WHPX dirty page range queries per second.

Through systematic root-cause tracing across the QEMU device model (`reims-vgpu-pci.c`, `reims-vgpu-dirty.c`) and the Rust runtime (`reims-vgpu`), these bottlenecks were isolated and resolved without compromising parity or stability.

---

## 2. Issue #3: Cursor Doorbell Cache Thrashing & MMIO Serialization

### The Problem
During interactive mouse pointer motion, frame rates collapsed to 1–3 Hz. Profiling revealed that moving the mouse generated 190–290 register writes per second to `0x1220` (`GFX_REG_EFI_DISPLAY_IRQ`).

### Root Causes
1. **Dirty Epoch Invalidation**: In `vendor/qemu/hw/display/reims-vgpu-pci.c`, *every* register write bumped the global dirty observation epoch (`reims_vgpu_dirty_epoch_bump()`) before inspecting the register offset. In the Rust runtime (`runtime/surface_currency.rs`), any change to the epoch invalidates the surface cache (`entries.clear()`). Pointer motion wiped the currency cache 250 times per second, forcing full guest memory re-reads on every draw.
2. **Drain Lock Contention**: Register `0x1220` was processed through the serialized queued MMIO path in `device/mod.rs`, queuing behind long drain tranches and GPU fence waits (up to 1.8 s delay).

### Solutions (Fix 1)
- **QEMU Epoch Bypass (`vendor/qemu`)**: Modified `reims-vgpu-pci.c` to inspect `0x1220` first. The cursor doorbell bypasses both the epoch bump and the dirty prefetch kick.
  - Telemetry slug: `dirty_epoch_skip_cursor_doorbell`.
  - Override switch: `REIMS_VGPU_DIRTY_CURSOR_EPOCH=bump`.
  - Compile-time assertion: `model/regs.rs: the_cursor_doorbell_offset_matches_the_c_define`.
- **Lock-Free Atomic Cursor Rail (Rust)**:
  - Created `model/state.rs::CursorMirror` with atomic packed coordinates `(x, y, show)`.
  - Implemented lock-free MMIO routing in `device/mod.rs` (`device_gfx_write`), servicing `0x1220` immediately using `slot.vbl_shared_gpa` and prompt actions without taking `inner` or the engine mutex.
  - Telemetry slug: `lock_free_cursor_doorbell`.

---

## 3. Issue #4: Per-Surface WHPX Dirty Sync & Hypercall Flooding

### The Problem
Host threads spent up to 650 ms out of every 1,000 ms inside WHPX hypercalls (`whpx_log_sync`). Profiling showed ~65,000 hypercalls per second executed on the drain thread, stalling rendering and pegging host CPUs.

### Root Causes
- Under WHPX, dirty page tracking is queried via `WHvQueryGpaRangeDirtyBitmap`.
- The default implementation reconstructed the entire GPA range plan for every surface and performed individual queries across multiple allocation ranges, followed by full-RAM bitmap scan, merge, and clear passes.

### Solutions
- **GPA Range Caching (`vendor/qemu`)**:
  - Implemented per-surface GPA range plan caching (`vendor/qemu/hw/display/reims-vgpu-dirty.c`).
  - Pooled query bitmap buffers to eliminate dynamic memory allocations on the hot path.
  - Metrics added to `REIMS_DIRTY_PROFILE`: `range_cache_hit`, `range_cache_build`, `hypercalls_per_sync`.
- **Target Residency Promotion (Steps B/C)**:
  - By retaining active surfaces in `DEVICE_LOCAL` memory (`runtime/target_residency.rs`), active composite passes write directly to GPU memory without forcing guest page writebacks and subsequent re-synchronization.
  - Reduced `WritesNarrowed` compute fallbacks from over 1,500 down to near zero.

---

## 4. Issue #5: Memory Lifecycle, Late Writes (`released_write_after_release`), & v8–v14 Hardening

### The Problem
During heavy window compositing, Safari and other apps experienced black rectangles and UI freezes. The fail log recorded `released_write_after_release`: the device wrote back rendered pixels to guest memory pages that the guest OS had already released and reassigned.

### Root Causes
Asynchronous GPU render passes hold writeback commitments. When the guest issued `UnmapMemory` or `DeleteIOSurfaceBacking2`, the device acknowledged the release immediately while Vulkan compute/render writebacks were still in flight. When the GPU finished, it DMA'd texels into pages that now belonged to another task or a kernel page table.

### Solutions
- **Writeback Settling at Release (`runtime/render_writeback`)**:
  - Added `settle_release` and `settle_release_with` at `GuestRelease` and `BackingRelease` checkpoints.
  - Overlapping writeback debts wait for bounded completion (`quiesce_guest_writes_bounded`) before guest page permissions are released.
  - Telemetry slugs: `release_settled_writeback`, `release_settle_raced_clear`.
- **Writeback Debt Cancellation**:
  - In `model/state.rs::unmap_surface`, unmapped surfaces immediately cancel unstarted writeback debt (`release_cancelled_writeback_debt`), preventing stale updates.
- **Classification of Late Writes (`runtime/released_pages.rs`)**:
  - Tracked and classified late writes under `released_write_after_release_owner_{gva_retained, live_mapping, condemned_mapping, none, unclassified}`.
- **Incremental Hardening (v8–v14)**:
  - **v8 (Pipeline Cache Persistence)**: Added `pipeline_cache_persist.rs` with bounded size and 10-second debounce to avoid disk write stalls while preserving warm SPIR-V pipelines.
  - **v9 (Drain Stall Watchdog)**: Added `stall_watchdog.rs` to detect hung compute/draw tranches and recover gracefully.
  - **v10 (Incarnation Guarding)**: Added generation stamps to prevent cross-generational ABA page reuse.
  - **v11 (MSAA Load Cache & sRGB Resolve)**: Dedicated resolve pairing preventing MSAA sample corruption.
  - **v12 (Unbound Buffer Zero-Page)**: Provided legal null descriptors for unbound storage buffers to eliminate GPU device loss.
  - **v13 (Load Seed Rescue)**: Enabled native seed rescue for `R16Float` and wide color targets.
  - **v14 (Stability Control Baseline)**: Capability-gated baseline ensuring zero regression across all test suites.

---

## 5. Issue #6: Display Refresh Rate, Pacing, & Stale-Hold Protection

### The Problem
The device advertised a 120 Hz display capability (`DISPLAY_REFRESH_HZ=120`), yet macOS WindowServer paced rendering at 60 Hz. Meanwhile, the host window loop spun unthrottled at 450–700 Hz, outputting stale frames and wasting CPU cycles.

### Root Causes
- macOS requires `IOFBCurrentPixelClock` and `IOFBCurrentPixelCount` in the frame buffer driver properties to dynamically configure 120 Hz timing. In their absence, `IOAcceleratorFamily2` defaults to 60 Hz.
- The host window presentation thread ran in Mailbox mode with zero wait, repeatedly presenting identical frame generations (`draws_fresh=0 draws_stale=122`).

### Solutions
- **Stale-Hold Protection (`backend/vulkan/engine/stale_hold.rs`)**:
  - Introduced `StaleHold` evaluator to gate redundant swapchain acquisitions when the surface generation has not advanced.
  - Slug: `host_window_stale_hold`.
- **Presentation Cadence Alignment (`host_window/present.rs`)**:
  - Throttled host presentation loop to match actual display refresh, eliminating runaway CPU spin-wait.

---

## 6. Steps A/B/C: Residency Promotion & Ordered Scanout Handoff

- **Step A: Parity & Topology Alignment**: Fixed primitive strip restart behavior and line-width slop tolerance in `vk_engine_parity.rs`.
- **Step B: `DEVICE_LOCAL` Residency Promotion (`runtime/target_residency.rs`)**:
  - Surfaces with `content_generation >= 2` or active scanout status are promoted to resident `DEVICE_LOCAL` allocations (`promote_device_local_confirmed`).
  - Single-use scratch surfaces take safe copy rails to prevent hypervisor page faults.
- **Step C: GPU Scanout Handoff & Ordered Copyback**:
  - Separated window presentation from CPU readback via `WindowPresentSource` and `guest_published_epoch` stamping.
  - Stamped `guest_published_after_copyback` to guarantee memory ordering between Vulkan presentation and guest OS access.

---

## 7. v15–v16: Wallpaper Synthesis & Raster Sample Count Specialization

- **v15 (Wallpaper Repair)**:
  - Added synthesized null sampled textures and samplers (`runtime/draw/synthesized_sampled.rs`, `unbound_trace.rs`) to gracefully satisfy unbound wallpaper and compositor passes without GPU hangs.
- **v16 (Raster Sample Count Specialization)**:
  - Specialized pipeline caching on `RenderSampleCount` (1, 2, 4, 8, 16, 32, 64 samples) matching Vulkan physical device limits.
  - Added verification markers: `m2v_raster_sample_count_unsupported` and `preflight_raster_sample_count_unsupported`.

---

## 8. Configuration & Diagnostic Reference

| Environment Variable | Allowed Values | Default | Purpose |
|---|---|---|---|
| `REIMS_VGPU_DIRTY_CURSOR_EPOCH` | `skip`, `bump` | `skip` | Set to `bump` to revert Issue #3 cursor bypass. |
| `REIMS_VGPU_AIR_CAPTURE` | `0`, `1` | `0` | Enable detailed shader AIR trace capture. |
| `REIMS_VGPU_GUEST_IMPORT` | `on`, `off` | `on` | Control host-pointer guest memory importation. |
| `REIMS_VGPU_STALE_HOLD` | `on`, `off` | `on` | Gate redundant host window presentations. |

### Verification Commands
```powershell
# Run the complete test suite serially with strict warning validation
cargo test -p reims-vgpu --no-default-features --features backend-vulkan,host-window -- --test-threads=1

# Run engine parity suite
cargo test -p reims-vgpu --test vk_engine_parity --no-default-features --features backend-vulkan,host-window -- --test-threads=1
```
