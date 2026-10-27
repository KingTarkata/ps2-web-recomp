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
