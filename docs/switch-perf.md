# Switch performance log

Only bare-metal numbers: the Switch itself, or a bare-metal PC. Nothing
measured in a VM belongs here.

## Setup

- Nintendo Switch (Erista, Tegra X1), Switchroot Ubuntu 24.04.4 (L4T,
  kernel 4.9.140-l4t), NVIDIA driver 32.3.1 (Vulkan 1.1.85), X11 session.
  32 GB SanDisk Ultra card. Handheld, on charger. 4 GB RAM.
- Clocks as found: CPU schedutil governor, 1785 MHz max (at max while
  running); GPU nvhost_podgov, 768 MHz max; `nvpmodel` and `jetson_clocks`
  available for Phase 5.
- Runner: `ps2run` cross-compiled with `cmake/aarch64-linux.cmake`, wgpu-native
  v29.0.1.1 on Vulkan, the Spec II build with 103 ahead-of-time VU1 programs,
  3 threads (emulation, VU1, GS). Native resolution (`PS2_SCALE` unset).
- Timing from the runner's `PS2_TIME=<first frame>,<count>`, which prints the
  mean wall time per emulated frame over that window. NTSC target: 16.67 ms.

## Measurements

| Date | Scene | Frames | ms/frame | Notes |
|---|---|---|---|---|
| 2026-10-26 | Intro and title, headless (no window) | 600-1100 | 23.81 | first run, default clocks |
| 2026-10-26 | Nürburgring race, Pontiac Vibe GT, windowed, 3 threads | 6000-6500 | 103.84 | first run, default clocks, 86.8M primitives over 9000 frames |
| 2026-10-26 | Race from saved state, windowed, 3 threads | 6050-6650 | 97.15 | baseline for the experiments below |
| 2026-10-26 | same, headless (no window, no present) | 6050-6650 | 95.00 | presentation costs ~2 ms |
| 2026-10-26 | same, `PS2_ONE_THREAD=1` | 6050-6650 | 97.34 | threading changes nothing: one serial bottleneck |
| 2026-10-26 | same, after `jetson_clocks` (CPU pinned 1785 MHz, GPU still 768 MHz) | 6050-6650 | 97.48 | no change |
| 2026-10-26 | same, power mode 3 "OC_GPU" (GPU 921.6 MHz) | 6050-6650 | 82.76 | GPU +20% clock gives -15% frame time: GPU-bound confirmed |
| 2026-10-26 | same, `PS2_GPU_PROFILE` timestamps | 6050-6650 | 97.78 | GPU busy 67.3 ms/frame: raster 57.3, decode 4.7, load_clut 3.7, upload 1.7 |

## Where the time goes (race, Nürburgring, native resolution)

- **GPU-bound.** Timestamp queries put the GPU at 67 ms of the 97 ms frame,
  57 ms of it in the compute rasterizer. The two frames profiled show the
  heaviest batches are large-triangle passes ("over 16" and "over 1000" tile
  coverage) on textured, depth-tested geometry, plus a long tail of 60 to 70
  tiny sprite batches per frame.
- **CPU mostly idle.** `top -H` during the race: the emulation thread at
  about 62% of one core, every other thread near 0%, machine 77% idle. The
  EE recompiled code and the ahead-of-time VU1 programs are not the limit.
- GPU clock during these runs: 768 MHz, which is Switchroot's power mode 0
  "Console", the same GPU clock as Nintendo's docked mode. Mode 1 "Handheld"
  caps the GPU at 460.8 MHz, so a handheld Horizon build would be ~1.6x
  slower again. Mode 3 "OC_GPU" lets the GPU reach its 921.6 MHz top step
  (an overclock past the official docked clock). `jetson_clocks` did not
  raise the GPU above the mode's 768 MHz cap; with it pinned the race
  measured 97.48 ms, unchanged.
- Target is 16.7 ms, so the race is 5.8x short at native resolution.
| 2026-10-26 | Race from saved state, headless, renderer refactored behind `Device`, wgpu | 6050-6650 | 95.12 | unchanged by the refactor |
| 2026-10-26 | same, Vulkan device (`PS2_GS=hw`), identical compute shaders via naga | 6050-6650 | 80.88 | 15% from dropping wgpu-native alone, before any graphics-pipeline work |
| 2026-10-26 | same, hardware rasterizer (`PS2_GS=hw`, graphics pipelines for 91% of batches), frames verified against the compute path | 6050-6650 | 74.34 | 142323 batches on the hardware path, 13913 through compute (snapshot 7074, alpha-test fail modes 3481, blend 2242, 16-bit frames 1116); same run day, compute path re-measured at 94.98 |
| 2026-10-27 | same, `PS2_GPU_PROFILE` on the hardware path (profiling serialises CPU and GPU, so the frame time is not comparable) | 6050-6650 | 138.89 | GPU busy 71.95 ms/frame: raster 61.3 (hardware draws including their import/export passes), decode 5.3, load_clut 3.3, upload 1.9 |

The hardware path's GPU time is no lower than the compute rasterizer's
(72 vs 67 ms); the 20 ms saved per frame so far come from dropping
wgpu-native and from the GPU no longer being the only limit. The reason is
that every batch still imports its rectangle (frame and depth) from GS
memory and exports it again: the heaviest rows above are 2 ms per batch for
a few hundred triangles, which is the cost of four full-screen passes over
swizzled memory, not of drawing. Page ownership tracking, so a target stays
resident across batches and is only written back when something reads the
memory, is the next step (docs/gs-hw-renderer.md).

## Correction (2026-10-27): the numbers above are single-threaded

`PS2_TIME` and `--frames-dir` make the runner run on one thread, and a
headless run (no `--window`) never starts the render and VU threads, so
every row above measured EE emulation, GS batching and GPU waits in series
on one core (`top` showed the process at 100% of one core while the machine
was 75% idle). The threaded path only runs windowed, and it paces to the
audio clock, so measuring it needs the new `PS2_NO_PACE=1` (run flat out)
and `PS2_SPEED_LOG=1` (print the window title's fps line to stderr each
second). On the Switch over SSH: `DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority
... --window`.

| Date | Scene | ms/frame | Notes |
|---|---|---|---|
| 2026-10-27 | Race from saved state, threaded, unpaced, windowed, compute path (`PS2_GS=gpu`) | 80-86 | CPU 75-85% of one core, GPU busy 61% |
| 2026-10-27 | same, hardware path with resident surfaces (`PS2_GS=hw`) | 45-49 | CPU 150-185% (EE and GS threads both busy), GPU busy over 100%: GPU-bound; GPU profile 48 ms/frame (raster 37, decode 5.4, load_clut 4.0, upload 2.0) |
| 2026-10-27 | same, single-threaded (`PS2_TIME`), hardware path, for comparison with the rows above | 70.5 | down from 74.3 with per-batch import/export |
| 2026-10-27 | same, threaded, hardware path with the texture cache and the setup record in the per-run uniform (a00568b plus the `load_setup` fix) | 46.8-56.2 | CPU 174-186%, GPU busy 69-81%: CPU-bound now; GPU profile 41.6 ms/frame (raster 32.9, load_clut 4.0, decode 2.4, upload 2.2) |
| 2026-10-27 | same, single-threaded (`PS2_TIME`, no profiler), hardware path with the texture cache | 90.2 | the cache costs ~20 ms of CPU per frame single-threaded; the GPU gain (48.3 -> 41.6) is hidden behind it |
| 2026-10-27 | (invalid) runs hw9-hw14 between 09:19 and 12:30 | 132-560 | `load_setup` routed through an array by value: 10x shader slowdown on Tegra, both renderers; see gs-hw-renderer.md |
| 2026-10-27 | same, threaded, hardware path after the render-thread work (texture key index and LRU list, cell-grid validity, dirty counters, arena built in mapped memory) | 41-42.5 | main thread was the limit at 98% of a core; now about 50 ms of work per frame there against 41.6 ms of GPU time |
| 2026-10-27 | same, threaded, hardware path with batches promised to the hardware path skipping the compute tile lists, decode plans and snapshots | 38-42 | main thread flush_batch+emit_all 6-8 -> 2.6 ms; 0 batches lost (descriptor pool enlarged for the race-start surface churn); CPU and GPU now both near 41 ms |
| 2026-10-27 | same, threaded, hardware path with the vertex/uniform scratch in host-cached memory | 36.4-39.5 | append_prims 8 -> 1.8% of the main thread; CPU 184-214% |
| 2026-10-27 | same, threaded, hardware path with the surface bounds table and direct-mapped CLUT tables | 35.8-38.6 | device record 12.4 ms, load_clut 2.4 ms per frame; main thread and GPU (~41 ms) now at parity |
| 2026-10-28 | same, threaded, hardware path with the copy and draw shaders dividing by the compiled-in scale and export_depth reading the depth image directly | 35.9-37.3 | GPU profile (6050-6650) total 45.5 -> 43.3 ms: copies 9.4 -> 8.1 ms (of which about 2.0 ms is the 381 dispatches per frame and 6.1 ms the 6.9 Mpx of samples, 0.88 ms per Mpx), draws 16.4 -> 15.7 ms |

Mode 0 "Console" (768 MHz GPU) throughout. Target remains 16.7 ms.
