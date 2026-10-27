# A hardware-rasterized GS backend

Design for a Graphics Synthesizer renderer that uses the GPU's own
rasterizer through Vulkan graphics pipelines, replacing the compute-shader
binning renderer for the Switch and for any other GPU where that renderer is
too slow. Decided on 2026-10-26 after the Phase 3 measurement: on the Switch
a Gran Turismo 4 race spends 57 of its 97 ms per frame in the compute
rasterizer, with the CPU mostly idle (`docs/switch-perf.md`).

The backend is game-independent: every title that goes through this
project's pipeline draws through it, so it is the part of the Switch work
that transfers to other PS2 ports, and it is worth sending upstream.

## 1. What the current renderer does

`runtime/src/gs/gpu/gpu.cpp` and `gs.wgsl` keep the GS's 4 MB of memory in
one storage buffer in the console's own swizzled page layouts. Drawing is a
compute dispatch per batch: triangles are binned into 16x16 tiles on the CPU
(`flush_batch`), and `raster` walks each tile's primitive list, running the
complete per-pixel GS pipeline (`plot`: scissor, scan mask, destination
alpha test, depth test, texture function, fog, alpha test with its partial
write modes, blending, color clamp, frame mask, 16- and 32-bit frame and
depth formats) directly against GS memory. Because everything stays in GS
memory, every trick that reads rendered pixels back as a texture or as a
palette works without special cases; the only hazard handling is `snapshot`
(copy the overlap when a batch reads memory it also writes) and `decode`
(unswizzle a texture once per batch into a linear cache).

That generality is what costs on Maxwell: every covered pixel of every
primitive runs a long branchy shader with scattered 32-bit loads and stores,
and the GPU's rasterizer, depth unit, blend unit and texture units all sit
unused.

## 2. Approach

Keep GS memory authoritative, but render through Vulkan render passes into
images that mirror the parts of GS memory the game draws into, and sample
textures from images decoded from GS memory. The design follows the shape
every long-lived hardware GS renderer has converged on (PCSX2's hardware
renderer is the best documented; it is GPL-licensed, so it is a reference
for the problems and their shape, not for code).

### 2.1 Memory model

- **GS memory stays in a buffer** on the GPU, exactly as now. `upload`,
  `copy_read`/`copy_write` (local copies), `load_clut`, `download` and
  `display` keep working on it unchanged. This is what makes the software
  rasterizer and the compute renderer usable as references at any point.
- **Targets.** A frame buffer or depth buffer the game draws into becomes a
  Vulkan image (R8G8B8A8 for 32-bit frames, R16 formats or a 32-bit image
  with conversion for 16-bit frames, D32 or D24S8 for depth) keyed by
  `(base pointer, width, format)`. A target is *live* while the game keeps
  drawing into it. Writes through the GS memory path (uploads, local
  copies) into a live target's pages invalidate the target or are copied
  into it, whichever is cheaper per page.
- **Write-back.** A target's pixels are written back into GS memory,
  re-swizzled by a small compute pass, when something reads that memory
  through the buffer path: a download to the EE, a CLUT load from it, a
  local copy, or the display. Write-back is per page, tracked with the same
  8 KiB page granularity `mark_rendered` uses today, so the common case of
  a frame that is only ever displayed costs one write-back per frame.
- **Textures.** A texture is decoded from GS memory into an image keyed by
  `(TBP0, TBW, PSM, size, CLUT contents hash)`, the way `decode` already
  does into a linear cache; indexed formats are expanded through the
  palette at decode time. Entries are invalidated by page when uploads or
  write-backs touch their pages.
- **Render-to-texture.** When a draw samples from pages that belong to a
  live target, the target image itself is the texture source: a copy of
  the overlapping region (what `snapshot` does today) when the draw also
  writes it, or the image bound directly when it only reads it. Palettes
  read from a rendered image go through write-back, because the CLUT path
  reads GS memory.

### 2.2 Drawing

- One graphics pipeline family: a vertex shader that takes GS vertices as
  the runtime already produces them (12.4 fixed-point XY, Z, RGBA, ST/UV/Q,
  fog) and a fragment shader that implements `plot` minus whatever the
  fixed-function units take over. Pipelines are specialised by the
  `Setup` flags that change control flow (textured, FST, IIP, fog, alpha
  test mode, blend mode class, frame format) and cached; the rest of
  `Setup` goes in a uniform block.
- **Depth test, depth write, scissor, color write mask, blending** map to
  fixed-function state where the GS semantics allow it:
  - ZTST maps directly to the depth compare op; ZMSK to depth write.
  - The GS blend `(A - B) * C + D` with selectors Cs/Cd/0 and C from
    source alpha, destination alpha or FIX maps to Vulkan blend factors
    for the common selector combinations. Combinations that do not map
    (those needing `Cd` on both sides, or the unclamped wraparound when
    COLCLAMP is off) fall back to reading the destination in the fragment
    shader: `VK_EXT_fragment_shader_interlock` or framebuffer fetch where
    available, otherwise a barrier-and-sample pass per such batch.
  - FBMSK with whole-channel masks maps to the color write mask; partial
    bit masks fall back to the destination-read path.
  - Alpha test modes that still write depth or color after a failed test
    (ZB_ONLY, FB_ONLY, RGB_ONLY) need either two draws with different
    write masks or `discard` plus a separate depth-only pass; use the
    two-draw form, as it needs no extensions.
  - DATE (destination alpha test) uses the stencil buffer: one pass writes
    stencil from the destination alpha bit, the draw tests against it.
  - Scan mask, PABE, FBA, color clamp off, and TEXA/AEM expansion stay in
    the fragment shader.
- **Texture sampling** uses the GS rules, not the sampler's: nearest or
  the GS's own bilinear, REPEAT, CLAMP, REGION_CLAMP and REGION_REPEAT via
  the mask/fix form, UV from ST/Q with the GS's divide. The texture image
  is sampled with `texelFetch`; the sampler is only used for the simple
  cases where a hardware sampler is bit-exact.
- **Upscaling.** The compute renderer's `SCALE` becomes the target image's
  size multiplier. Internal resolution options stay as they are.

### 2.3 Batching and ordering

The CPU side already groups primitives into batches with a shared setup
and detects conflicts (`batch_conflict`, `shares_frame`, `drawn_under`).
That logic stays. A batch becomes one draw call inside a render pass on the
batch's target; consecutive batches on the same target stay in one pass.
Hazards that today trigger `snapshot` trigger a pass break and a copy.

### 2.4 Interface

`runtime/src/gs/backend.hpp` declares a `Backend` with the entry points
`gpu.cpp` exports now (`init`, `draw`, `image_begin`, `image_data`,
`local_copy`, `load_clut`, `flush`, `wait_idle`, `download`, `display`,
`present`, `read_state`, `write_state`). The existing compute renderer and
the new one are two implementations; `PS2_GS=gpu|hw|soft` selects, with
`both` mirroring into the software rasterizer for comparison as it does
now. The runtime's queue and threads do not change.

### 2.5 Platform

- **Linux on the Switch (L4T):** NVIDIA's driver, Vulkan 1.1.85. No
  fragment shader interlock; framebuffer fetch is not available on this
  driver either, so the destination-read fallback is the barrier form.
- **Horizon:** Mesa NVK through the setup `refs/NetherSX2_nx` uses, or
  deko3d. NetherSX2 (MIT) ships a native deko3d GS backend and a Vulkan/NVK
  path, and is the reference for both; its launcher and build files show
  what the Horizon toolchain needs.
- **PC:** any Vulkan 1.1 GPU; this is where correctness is checked.
- Shaders are written in WGSL alongside the existing ones and compiled to
  SPIR-V offline with naga at build time (`refs/gt4-staticrecomp`, MIT,
  has a working `tools/wgsl_spirv`). GLSL is the fallback if WGSL cannot
  express something needed (stencil export, interlock).

## 3. Correctness and measurement

- `scripts/compare_runs.py` already compares frame dumps between runs.
  Every stage below is validated by running the same scripted session
  (`docs/switch-port-notes.md` section 7) with `PS2_GS=soft` and with the
  new backend, comparing frames. Exact equality is the goal for the
  non-blended, non-bilinear cases; for the rest, a per-pixel tolerance with
  the differing pixel count reported, since hardware depth interpolation
  and blend rounding will not be bit-identical to the console.
- Performance is measured only on the Switch, from the saved race state
  with `exp.sh`, into `docs/switch-perf.md`.

## 4. Stages

1. Backend interface and a second implementation that is the compute
   renderer behind it. No behaviour change. Verified by frame comparison.
2. Targets and write-back: untextured, unblended triangles and sprites
   with depth into Vulkan images; everything else still goes through the
   compute path. Verified on the title screen and car selection.
3. Textures and palettes: decode cache, GS sampling rules, texture
   functions, fog. Verified on menus.
4. Tests and blending: alpha test modes, DATE via stencil, the blend
   factor mapping, FBMSK, color clamp, and the destination-read fallback.
   Verified in a race.
5. Render-to-texture hazards and the display path. Verified by the
   reflections and post effects in a race, and by the replay and photo
   modes if reachable.
6. Switch measurement at each stage from stage 2 on; the first number that
   matters is stage 4's race.

## 5. Risks

- **Fidelity:** hardware rasterization differs from the GS in sub-pixel
  rules, depth precision and blend rounding. Games that depend on exact
  pixel results (palette tricks, depth-based effects) will show it. The
  software rasterizer stays the reference and the compute renderer stays
  available per game.
- **Hazard tracking:** the cases where a draw reads its own target are
  the usual source of bugs in this kind of renderer. The existing conflict
  detection and `snapshot` logic is reused rather than rewritten.
- **Driver limits:** the Switch driver's Vulkan 1.1 feature set rules out
  the cheap destination-read paths; batches that need them will be slower.
- **Scope:** this is the largest single piece of work in the project. The
  stages are ordered so that each one is measurable and useful on its own.

## 6. Progress log

- 2026-10-26: stages 1 and 2 done. `gpu.cpp` drives a `Device`; the wgpu
  device is verified frame- and RAM-identical; the Vulkan device runs the
  compute shaders (naga, one SPIR-V per upscale factor) frame-identical over
  the title and a 6000-frame race, and saves 15% on the Switch by itself.
- 2026-10-26: stages 3 and 4 partly done in `runtime/src/gs/vk/hw.{wgsl,cpp}`.
  Batches draw through graphics pipelines into an RGBA8 target plus a D32
  attachment, imported from and exported to GS memory per batch rectangle;
  the fragment shader shares gs.wgsl's texture sampling and per-pixel rules.
  Blending maps the GS equation onto the blend unit with a dual-source
  factor for As/128 and blend constants for FIX; depth tests map to the
  depth unit (32-bit Z through float, 24 bits exact). Title sequence: 1152
  of 1346 batches on the hardware path, seven frames identical, two
  blended frames within a channel delta of 4. Still falling back: batches
  whose setups carry different frame masks, destination-alpha factors and
  tests, PABE, colour wrap, 16-bit frames, upscaling, lines and points, and
  batches reading their own target.
- Lessons: naga flips Y for Vulkan unless `--keep-coordinate-space`;
  descriptor bindings must name the graphics stages; llvmpipe crashed on a
  fragment-depth import draw, so depth moves through buffer/image copies.
- 2026-10-26, races: the first race frames through the hardware path were
  wrong (sky and distant scenery black). Bisecting one frame batch by batch
  (`PS2_HW_ONLY=frame,count`) led to the game's full-screen clear, which
  writes Z through the "always" test: the pipeline mapped that to depth test
  off with depth write on, and Vulkan drops depth writes whenever the test
  is off, so Z memory kept the previous frame and everything behind it
  failed. With the test kept on as COMPARE_OP_ALWAYS, 500 race frames from
  a saved state match the compute renderer to within blend rounding: about
  60k of 287k pixels differ by 1 or 2, and 15 to 25 pixels per frame by
  more, all single pixels on triangle edges (fill-rule differences). The
  Tegra driver also needs the compute shaders built from a module without
  the graphics entry points (`kGsSpv` and `kHwSpv` are separate), depth is
  exported with floor rather than round (coplanar greater-or-equal draws),
  and the vertex shader clamps Z to the format's maximum as `plot` does.
- Full 6000-frame Spec II session (title, menus, Nürburgring race) on the
  hardware path versus the compute renderer: 521k batches drawn, 61k
  through compute; every menu frame within a channel delta of 7, race
  frames within blend rounding plus 4 to 22 edge pixels each, no visible
  difference and no behavioural divergence over the session. EE RAM
  differs in 305 small ranges (40 KB) that hold pixel data the game reads
  back from GS memory, by the same rounding.
- 2026-10-27, Switch: 74.3 ms per race frame on the hardware path against
  95.0 on the compute path the same day (`docs/switch-perf.md`). GPU time
  is still 72 ms per frame, almost all of it the per-batch import and
  export passes rather than drawing, which settles what comes next.
- 2026-10-27, resident surfaces (stage 5 in substance). Frame and depth
  buffers are now `Surface`s in `hw.cpp`: one image per (base, width,
  format), sized to the buffer, with a 32x32-tile map of what is valid
  (image matches memory) and dirty (image newer). Every `gpu::Command`
  carries the GS memory spans it reads and writes (`gpu.cpp` fills them
  from the ranges it already computes); the device writes dirty tiles back
  before a command reads their memory and drops tiles when one writes it,
  rectangle-precise for compute-drawn batches on the same buffers. Downloads
  and state saves flush synchronously. Lessons: a surface write-back must
  write back, not drop, aliasing tiles of other surfaces; the viewport must
  stay the full 2048 drawing space or clipping at the surface edge shifts
  texel picks; a 24-bit frame and a Z buffer that overlap in memory (the
  game's clear trick) are stored colour-then-depth right after the draw.
  Also drawn in hardware now: batches reading their own target (the
  fragment shader sets the compute path's snapshot window), alpha-test fail
  modes (two draws per primitive, exact), the blend forms Cs(1+C), Cd(1+C)
  and Cs·Ad/128+Cd (the last as two monotone steps, since DST_ALPHA is /255),
  Z formats used as frames; and `gpu.cpp` starts a new batch when a setup
  switches between hardware-drawable and not, so one odd blend no longer
  sends a 6000-primitive scene through compute. Fallbacks left in a race:
  DATE (about 20 small batches a frame), partial frame masks, lines.
- 2026-10-27, measurement correction: `PS2_TIME` (and `--frames-dir`) run
  the runner single-threaded, and headless runs never start the render
  thread, so every Switch number before today was one thread doing EE, GS
  and GPU waits in series. `PS2_NO_PACE=1 PS2_SPEED_LOG=1 --window` measures
  the threaded path flat out: the race runs at 45-49 ms per frame on the
  hardware path and 80-86 ms on the compute path (`docs/switch-perf.md`).
  The GPU profile of the hardware path is 48 ms per frame (raster 37), so
  the GPU is the limit again.
- Next: textures as images sampled with texelFetch instead of swizzled
  buffer loads per fragment (the fragment shader also reloads the setup
  record per fragment); then DATE through stencil; then the depth copies.

### 2026-10-27, afternoon: a 10x shader regression, found and fixed

The texture-cache commit (a00568b) refactored `load_setup` in gs.wgsl into a
`make_setup(d: array<u32, 25>)` helper so the hardware path could build the
same struct from its uniform copy. On the Tegra compiler an array passed by
value lands in per-fragment local memory, and every shader that reads a
setup record goes through it: the hardware fragment shaders, `decode_tex`,
and the compute rasterizer. Measured on the Switch: hardware raster 35.6 ->
372 ms per frame, decode 5.4 -> 36 ms, compute-path GPU time 57-67 -> 978 ms.

Finding it took most of the day because the first suspects were wrong: a
loop with computed indices over `hw.setup` (rewritten, no change), the
uniform read itself (replaced by an arena read, no change), the uniform
struct size (fully reverted, no change), then the device (clocks checked
with root: GPU 768 MHz, EMC 1600 MHz, GPU 99% busy; rebooted, no change).
The compute path being slow with a binary whose only shared change was
gs.wgsl pointed at `load_setup`. The fix reads the fields directly in both
`load_setup` and `setup_u` (hw.wgsl). A first version of the fix dropped the
three derived `fmt_of` fields and rendered black frames on both machines
with very fast numbers; the frame check caught it. Lessons: measure every
shader refactor on the device, never only on llvmpipe; a performance run
needs a rendered-frame check next to it; and the compute path is a useful
control because it shares gs.wgsl with the hardware path.

Results with the fix (race from the saved state, frames 6050-6650, native
frames bit-identical to the verified st-hw3 run):

| Measurement | before texture cache | hw8 (cache, setup from arena) | now (cache, setup in uniform) |
|---|---|---|---|
| GPU time per frame, profile | 48.3 ms (raster 36.7) | 44.2 ms (raster 35.6) | 41.6 ms (raster 32.9, load_clut 4.0, decode 2.4, upload 2.2) |
| threaded, unpaced, windowed | 45-49 ms | 48-56 ms | 46.8-56.2 ms, CPU 174-186%, GPU 69-81% |
| single-threaded (`PS2_TIME`, no profiler) | 70.5 ms | - | 90.2 ms |

The GPU is no longer the bottleneck in the threaded run (busy under 100%
for the first time); the texture cache costs about 20 ms of CPU per frame
in the single-threaded figure, which is the next thing to cut: block-id
validation over the sampled footprint, `blocks_of` per upload chunk, and
the linear key search in `texture_slot`.
