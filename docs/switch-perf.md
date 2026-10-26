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
