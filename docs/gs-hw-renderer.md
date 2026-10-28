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

### 2026-10-27, evening: the render thread, measured and trimmed

With the GPU at 41.6 ms the threaded run sat at 47-56 ms, so the CPU was
the limit. Per-thread sampling on the Switch showed the main thread, which
runs the render queue in this host, at 98% of a core while the emulation
and VU threads were at about 50% each. Section timers (`PS2_HW_TIMERS=1`
or `from,to`, printed with the device stats as `[cpu]` lines) and `perf`
(installed on the device; `/usr/lib/linux-tools/*/perf record -g -t <main
tid>`) on the race frames gave, per frame on the main thread:

| Section | before | after |
|---|---|---|
| present (device submit: record, draws, texture lookups) | 42.0 ms | 29.0 ms |
| texture_slot (key lookup: linear scan of 1536 keys, LRU scan on creation) | 11.8 ms | 1.7 ms |
| texture validity (sorted block list built and binary-searched per lookup) | 1.6 ms | 0.2 ms |
| submit memcpy of the arena into the mapped buffer | ~4 ms (10% of the thread) | 0 |
| gif::render (register writes, batching, uploads) | 21.4 ms | 21.9 ms |

Changes: a hash index over texture keys with the texture list kept in LRU
order (victim at the back) and evicted images filed by size; texture
validity as a grid of block cells checked by indexing the sampled
rectangle; a dirty-tile counter per surface instead of bitset scans; a
small recent-pipeline cache ahead of the pipeline map; the data arena built
directly in the device's mapped buffer for the coming submit (host-cached
memory, flushed at submit; the read-back buffers are host-cached too and
invalidated before the CPU reads them). Threaded race: 41-42.5 ms per
frame, GPU busy near or over 100% again, main thread about 50 ms of work
per 55 ms of wall time.

The GPU side, from a least-squares fit of the per-batch timestamps against
batch size: about 18 us fixed per batch (4 ms per frame), the rest
proportional to covered tiles, which is fragment work. The fragment shader
rebuilds the whole setup record per fragment and keeps every GS feature
live; specialising it per batch kind (texturing, alpha test, fog, blend)
is the next GPU item. Next CPU items: the GS front end (write_packed,
kick, draw: 22 ms per frame, shared code), the compute tile lists built
for batches the hardware draws (about 3 ms), the SDL presentation path
(two copies of the frame plus a GPU read-back: about 3 ms and a sync),
and the 48-byte vertex format.

### 2026-10-27, night: where the GPU time is, by elimination

Fragment shader variants (hw.wgsl `HW_VARIANT`: textured, may discard,
fogged; eight modules compiled for scale 1, chosen per setup) and the GS
scissor moved from an in-shader discard to the Vulkan scissor rectangle, so
most pipelines never discard and can test depth early. Native frames are
bit-identical. Gain: raster 32.5 -> 31.5 ms (`PS2_HW_GENERIC=1` A/B),
threaded 41-42.5 -> 38-40.5 ms. Small, which forced the measurements below
(all race frames 6050-6650, raster ms per frame):

| Experiment (`PS2_HW_DEBUG`) | raster |
|---|---|
| full shader | 31.5 |
| sampling skipped (16384) | 22.8 |
| sample one constant texel (524288): texture bandwidth removed, maths kept | 30.7 |
| trivial shader, every fragment painted (8) | 17.8 |
| single-output shader, no blend (4194304) | 17.6 |
| same without a depth attachment (12582912) | 13.5 |
| whole surface as render area (2097152) | 31.6 |

Fragments reaching the colour stage (occlusion query, 1048576): about 5 M
per frame. So: texture bandwidth is nothing; the sampling code is ~8 ms of
instructions; depth traffic ~4 ms; and a 13.5 ms floor remains with a
shader that does nothing, no blend and no depth, for a few million
fragments, i.e. the GPU is mostly draining: 230 render passes per frame,
each between full pipeline barriers. Next: keep one render pass open across
consecutive batches on the same surfaces when nothing needs the pass closed
(no imports, write-backs, texture decodes or copies), with the whole
surface as render area (the area costs nothing).

### 2026-10-27, late: one render pass across batches, and why GT4 still closes it

Mechanism (hw.cpp): consecutive hardware-drawn batches on the same colour
and depth surfaces share one render pass (whole surface as render area);
anything recorded outside a pass closes it first (`close_pass`, called by
imports, write-backs, decodes, copies, image barriers). Compute commands
the device meets while a pass is open (uploads, CLUT loads, copies,
snapshots, decodes) are held back and recorded when the pass closes, with
their `before_command` write-backs run then; a batch whose memory spans
overlap a held-back command's spans closes the pass first
(`deferred_conflict`), and so does a batch that samples memory in the
fragment shader while CLUT loads are pending (they write the CLUT buffer,
which the span check cannot see; found as a 62-pixel difference in the
race compare). Own-surface imports take every tile memory still owns, not
the batch rectangle, so the scene filling the screen costs one import, not
nineteen. Textures the cache does not take are sampled from a surface in
the same layout when there is one. CLUT loads are deduplicated by result
identity: a slot's 512 entries as 32 chunks of 16, each named by the load
that last wrote it (its parameters and the source pages' write
generation), untouched chunks inherited from the CLUT the load merges
into; a result already in a live slot is reused without a dispatch, the
slot pinned until the submit so no emitted load overwrites it. 29% of the
race's loads are reused, load_clut 4.0 -> 3.1 ms. Three bugs on the way,
each a few hundred pixels: the generation was read before a buffered batch
drawing the palette was flushed; emitted loads took the ring position after
the current slot, overwriting live slots once a reuse pointed it back; and
the early return skipped the CLUT source id the texture caches key on.
Native frames stay bit-identical; `PS2_HW_PASS_TRACE=<frame>` prints every
pass close of a frame with its reason.

Result: still 0.99 passes per batch on the race, threaded 38-42 ms, GPU
raster unchanged. The trace of frame 6100 (232 batches, 231 closes) says
why: GT4's post-process. 70 times a frame it draws a textured 64x16 block
into a 32-bit buffer (15328), then eight 8x32 sprites into the 24-bit frame
(4480), each sampling 256 bytes of that block as an 8-bit texture through a
CLUT loaded just before (a channel shuffle, as PCSX2 calls it). The
alternation of targets alone is 140 passes; the 8-bit reads of a 32-bit
surface need a write-back each time (`wb-uncached`, 70), and the pending
CLUT load forces the pass closed (`memory-read`, 69). The rest: 23
compute-path DATE batches (each also invalidates the surface), 18 copies for
batches sampling their own frame, 15 copies in a 64-wide ping-pong, a few
Z-alias overlaps.

Plan, in order of value: (1) reorder around open passes: hold back a
batch for another surface (and the compute commands between) while the
open pass continues, flush in order when a later batch's spans conflict,
hoisting held-back compute commands before held-back batches when their
spans allow; on this frame the 70 block draws would form one pass and the
70 strip draws another. (2) Read surfaces through memory addressing in the
memory-sampling shader (inverse page tables for PSMCT32), so 8-bit reads
of a 32-bit surface sample its image instead of forcing a write-back.
(3) DATE through the stencil buffer. (4) 16-entry CLUT loads keyed by the
chunks they touch rather than the whole chain, for more reuse.

### 2026-10-27, later: batches held back for other surfaces, and the real chain

Raster batches for surfaces other than the open pass's are now held back
with the compute commands (`HwRaster::plan`), drawn when the pass closes,
after the held compute commands; a batch or command whose memory spans
overlap a held batch's closes the pass first (`held_conflict`). Native
frames bit-identical. Frame 6100: 231 -> 154 closes; threaded time
unchanged (39.8-42 ms).

Why only that far: the post-process is a dependency chain, not an
interleaving. For strip i the block draw reads page i of the frame buffer
as an 8-bit REGION_REPEAT texture (uncached, so it samples memory) and
writes a 64x32 scratch at a second address; the tile draw reads that
scratch as eight 8x32 8-bit textures through a CLUT and writes page i
back. Every block draw writes the same scratch, so block i+1 must wait for
tile i to have read it: 140 passes a frame are forced by the data. The
remaining closes: that chain (70 pairs), 18 copies for batches sampling
their own frame, 14+14 copies in two 64-wide scratch ping-pongs, 23
compute-path DATE batches, a few Z-alias overlaps.

To break the chain the device would have to (a) version the scratch
surface (each block draw gets its own image of it, since it overwrites it
whole) and (b) let the tile draws sample those images through memory
addressing in the fragment shader (inverse page tables, no VRAM
round-trip), with the CLUT loads between them already reused. Then the 70
block draws form one pass and the 70 tile draws another. That is the next
GPU stage; its payoff is most of the 13.5 ms drain floor. DATE on the
stencil buffer is the smaller, independent item.

### 2026-10-27: scratch versioning tried, parked on branch `scratch-versioning`

Built: every batch for another surface, and everything that depends on
held work, is held; a list scheduler orders the held items at flush by
page-indexed memory dependencies (version-aware), CLUT loads before
textured batches, compute commands in order, batches extending the open
pass first. A small 32-bit region that held batches overwrite whole
(opaque sprites covering it) gets an image band per version in an atlas;
readers view their band or have it exported to VRAM; the latest band is
copied back into the surface at flush or when anything unversioned
touches the region. Cacheable textures that a surface holds dirty are
viewed instead of decoded.

Measured on the Switch, race frame 6100: closes 231 -> 91 (the 70 strip
pairs collapse as designed), GPU raster 31.5 -> 29.7 ms, main thread
+13 ms of bookkeeping (a std::sort on dependency lists alone is 19% of
the thread), threaded 40-42 -> 52-57 ms. Correctness did not hold: native
frames wrong and a crash in the resolve copy under llvmpipe (the flush
hook re-entering a chain, fixed, then more), car pixels wrong on the
Switch. Verdict: the pass-drain model over-predicted. 140 fewer passes
bought about 2 ms, so the 13.5 ms floor with a trivial shader is mostly
not pass drains; the candidates left are per-draw fixed costs (several
pipeline and descriptor binds per setup, 230 batches with up to a few
setups each) and the import/export dispatches around batches. The work
stays on `scratch-versioning` (WIP commit with the full description);
`switch-port` is back at the verified surface-view build.

Next: the CPU side (GS front end 22 ms, device submit 29 ms on the main
thread) and DATE on the stencil buffer; on the GPU side, measure the
per-draw fixed cost directly (one batch, many setups) before building
anything else on a model.

### 2026-10-27: batches promised to the hardware path skip the compute bookkeeping

The main thread is the limit (about 50 ms of work per frame against
41 ms of GPU time), and perf put 6-8 ms of it in `flush_batch` and
`emit_all`: per batch, span lists and per-tile primitive lists, decode
plans and the snapshot, all of which only the compute rasterizer reads.
The hardware path already skipped the decode and snapshot *dispatches*
(replaying them if it declined the batch), but the CPU work to build them
was still done for every batch.

Now the batching layer asks the device first (`Device::batch_hardware`,
answered by `HwRaster::batch_supported`, the same rules `draw()` applies
through `supported()`). A promised batch gets only its primitives and
setups appended to the arena and a raster command with no tiles, marked
`Command::hardware`. The device has nothing to fall back on for such a
batch, so a decline after the promise loses it: `note_lost` counts those
and warns (`PS2_HW_LOST_TRACE=1` prints each), and the two things that
could still decline were made rare: the vertex and uniform scratch
doubled to 32 MB per frame slot, and the surface descriptor pool, which
ran dry during the race-start frames (GT4 creates a few hundred one-page
targets with FBW 0 over frames 6284-6286, each evicting a surface whose
sets are freed only two submits later), is now sized for that churn.
Resource declines report as "no surface" / "no framebuffer" instead of
"scratch full" so this shows up in the fallback reasons.

Measured on the Switch (race frames 6050-6650, headless timers): flush_batch
plus emit_all 6-8 -> 2.6 ms per frame; 140924 batches promised, 140924
drawn, 0 lost. Threaded race 39-42 -> 38-42 ms/frame: the main thread
is now close to parity with the GPU (41 ms), so the next CPU savings
will only show once GPU time drops too. Native compare: frames 5500-6000
bit-identical to the reference; Switch frame 6100 identical to the
previous verified build.

### 2026-10-27: vertex and uniform scratch in host-cached memory

perf on the Switch main thread had `append_prims` at 8% (3 ms/frame):
the per-frame vertex and uniform scratch buffers were host-coherent,
which on Tegra is uncached write-combined memory, and vertices are built
a field at a time. They are now host-cached like the arena and flushed
in `end_frame` (after the pass close, which can still draw held batches).
append_prims 8.0 -> 1.8%, device record (`draw`) 15.4 -> 13.0 ms,
threaded race 38-42 -> 36.4-39.5 ms/frame. Native frames 5500-6000
identical, Switch frame 6100 identical.

Main-thread profile after this (threaded race): memcpy 17% (image
uploads into the arena 8.3%, SDL present copy 4%, read-back copy 2%),
gpu::draw 5.7%, load_clut 5% (1059 calls/frame), write_packed 4.9%,
gif::render 4.9%, memory_written 4.1%, gs::write 3.6%, copy_frame 3%,
kick 2.3%. GPU time is about 41 ms, so the thread is at parity now.

### 2026-10-27: surface bounds table

`memory_written`, `memory_read` and `find_surface` walk all resident
surfaces for every command and draw; each Surface carries its tile maps,
so walking them by pointer was a cache miss per surface (perf: 40% of
memory_written on the loads of the surface pointer and its width). The
surfaces' keys and memory bounds now live in a compact table
(`bounds()`, rebuilt lazily when a surface is allocated, retired or
evicted) and the span tests reject on the bounds before `span_rect`.
Switch: device record 13.1 -> 12.7 ms, threaded race 36-38.9 ms/frame;
frames identical natively and on the Switch. What is left in
memory_written is `textures_written` filling `block_ids_` over the
written span (256-byte granularity), about 1 ms per frame.

### 2026-10-27: CLUT load bookkeeping in direct-mapped tables

`load_clut` runs about 1059 times per race frame and was 3.2 ms of the
main thread; perf put half of it in the std::map walk naming the
palette source and the two hash probes of the result memo. The source
map now has a 256-entry direct-mapped cache in front and a packed
64-bit key; the memo is a 4096-entry direct-mapped table (a lost entry
only re-emits a load; reuse 187058 -> 186899 over the run). load_clut
3.2 -> 2.4 ms per frame; frames identical. A "same result already in
the current slot" shortcut was tried first and bought nothing.

### 2026-10-27: DATE on the hardware path tried, parked

Compute-drawn batches cost 4.5 ms of GPU per race frame (17 per frame,
all DATE). Tried: the colour surface's batch rectangle copied to a
snapshot image outside the pass (`vkCmdCopyImage`), bound at set 1
binding 6, and the fragment shader discarding on bit 7 of the snapshot's
alpha against DATM; DATE setups kept in batches of their own
(`Device::setup_alone`) because once they count as supported the
batching layer merges the 17 draws into one or two huge batches, which
then test one stale snapshot and drag unsupported blends to the compute
path (first attempt: 200k wrong pixels per frame, GPU 50 ms).

Isolated, the frames are visually identical (cars' paint layers; ±1-3
blend rounding over the car bodies plus scattered edge pixels, the same
category as hardware-vs-compute elsewhere), but the GPU profile got
worse: compute raster 4.5 -> 1.6 ms, hardware raster 34 -> 39 ms, total
44.6 -> 46.5 ms; threaded 36.5-38 ms (unchanged). The DATE batches
sample the frame and other surfaces, so each one now costs a pass close,
a snapshot copy and surface imports ("import-source" closes 761 ->
5335 over the run). Patch kept at
`private/notes/date-hw-experiment-2026-10-27.patch`; the stencil form
would have the same per-batch structure. Not worth it until the
per-batch fixed cost of the hardware path itself comes down.

### 2026-10-27: GS front end, first pass (shared code, with the user's go-ahead)

perf with instruction context on the Switch main thread:

- `write()` and `write_packed()`: a quarter of each was the
  initialisation guard of a function-local `static const char *seq =
  getenv(...)`, an acquire load (`ldarb`) per register write on the A57,
  plus the load of the pointer. One namespace-scope constant now
  (c01dd38): write_packed 5.9 -> 2.8%, gs::write out of the top ten.
- `gif::render` (5%): half of it is the 16-byte load of each packet
  qword from the render-queue ring, written by the EE thread (cross-core
  lines). A software prefetch eight qwords ahead changed nothing
  measurable (4.6 -> 4.3%), reverted.
- `gpu::draw` (4-6%): spread over the per-primitive float work and the
  setup cache; no single hot spot.
- `kick` (2.5%): call overhead and the vertex queue shuffles.

The remaining main-thread profile: memcpy 17-20% (texture uploads into
the arena 8%, SDL present 4%, read-back 2%), gpu::draw 6%, gif::render
4.6%, memory_written 3% (block-id fill), copy_frame 3%, kick 2.7%,
load_clut 2.5%, append_prims 2.5%. Threaded race unchanged at 36-39
ms/frame: with GPU time at about 41 ms the main thread is no longer the
limit, so the GPU side (per-draw fixed cost measured directly) is the
next lever; on the CPU side the upload copy and the present path are
what is left of any size.

### 2026-10-27: the hardware path's fixed cost, measured directly

Two measurement knobs: `PS2_HW_DEBUG` bit 27 (134217728) records every
batch as usual but issues no draw calls; bit 28 (268435456, with 27)
also begins no render passes. GPU profile, race frames 6050-6650, the
hardware-raster brackets (which include the imports, write-backs, decodes
and copies around each batch):

| run | raster-hw ms/frame | total ms/frame |
|---|---|---|
| full | 33.9 | 44.6 |
| no draws (bit 27) | 18.1 | 27.5 |
| no draws, no copies (27 + 32768) | 8.6 | 18.7 |
| no draws, never re-decode (27 + 131072) | 12.9 | 22.4 |
| no draws, no render passes (27 + 28) | 12.3 | 21.7 |

So of the 34 ms, about 16 ms is the draws themselves and 18 ms is
overhead, of which the import/export dispatches (`copy_rects`) are 9.4
ms, texture re-decodes about 5 ms and the render-pass begin/end (213
per frame) about 6 ms, overlapping. The 17.8 ms "trivial shader floor"
measured earlier was this overhead, not pass drains.

Where the copies come from (`PS2_HW_TRACE=6100,6102`, frame 6100):
6.9 Mpx moved per frame, 24 screens' worth: depth 2240/10/31 exported
988 kpx in 62 rectangles and imported 627 kpx; frame 6720/10/00 exported
586 kpx and imported 395 kpx; the frame's 24-bit aliases 4480/10/01 and
2240/10/01 imported 942 kpx; the 256-wide scratch 11200/4/00 imported
309 kpx. Write-backs by cause: 71 for CLUT loads (the post-process chain
reads the frame as a palette source), 26 for compute-drawn (DATE)
batches, 4 for uploads. With the parked hardware-DATE patch the DATE
write-backs disappear but only 0.8 Mpx with them (6.1 Mpx left): the
depth buffer at 2240 aliases the frame's 24-bit lower half 2240/10/01
that the post-process reads, so the depth surface still round-trips
(627 kpx in, 586 kpx out) every frame, and the CLUT-load write-backs of
the frame stay.

Next GPU work, in order of expected payoff: (1) the 71 CLUT-load
write-backs per frame, by reading the palette from the surface image
(a CLUT load from a resident surface, like a surface view) instead of
writing the frame back; (2) the depth 2240 / frame 2240 aliasing, by
answering the 24-bit frame reads from the depth image or keeping the
two coherent without a full export; (3) cheaper import/export shaders
(they cost about 1.4 ms per Mpx); (4) fewer passes once the copies
between batches are gone, since every copy closes the pass.

### 2026-10-28: the real-time display glitch, and CLUT loads from surface images

The windowed display showed green/magenta tile garbage on every other
frame (the user's photo), on the Switch and natively, in both thread
modes, while every headless frame dump was bit-identical: the dumps go
through the synchronous display call, the window through the
asynchronous read-back, and nobody had looked at the presented frames.
`PS2_DUMP_PRESENT=<dir>` now writes every presented frame as a PPM; on
the Switch `scrot` works over SSH (`DISPLAY=:0 XAUTHORITY=...`).

Two causes, both in the surface-view path (`PS2_HW_DEBUG=33554432`
disables views and was the bisect):

1. GT4 reads its depth buffer as a 24-bit colour texture (2240/10/01 over
   the Z buffer 2240/10/31). A view answered that read from the colour
   surface aliasing the depth memory, a stale copy until the depth
   surface is written back. The view now writes back any other surface
   holding the span dirty first (which also drops the viewed tiles for
   the import). Fixed natively; the Switch still showed residual blocks.
2. A write-back of one surface followed by an import of another aliasing
   it had no barrier between the two compute dispatches: llvmpipe
   serialises, the Tegra does not. `copy_rects` now starts with a
   compute-to-compute memory barrier on GS memory. All 155 sampled
   presented frames on the Switch are clean after both.

CLUT loads whose palette lies in a resident colour surface (GT4's
post-process, 71 per frame) now run gs.wgsl's `load_clut` from the
hardware module, whose `surface_word` reads the surface image
(`read_palette` in gs.wgsl asks the hook; the plain module's answers 0),
instead of writing the frame back first. 64934 loads over the run.
Lesson from the Tegra: a descriptor binding used by a compute entry
point must list the compute stage, or the GPU faults (MMU fault in
dmesg, "submit failed", frames frozen, "GPU 0%").

Cost: the correctness fix roughly doubled the import/export traffic
(2412 Mpx imported over the run, from 1132; written back 1710 from 856)
because every view of the 24-bit alias now writes the depth surface
back, so the GPU profile is 45.5 ms/frame (44.6 before) and the CLUT
change is neutral for now. Threaded race 36.3-38.8 ms/frame. The
next item is therefore a depth view: answer the 24-bit colour reads of
the Z buffer from the depth image itself (exact if the float depth
round-trips the 24-bit integer, which the export path already relies
on), which removes the depth export/import per frame entirely.

### 2026-10-28: depth views tried and dropped; where the copies really come from

A depth view (the 24-bit colour reads of the Z buffer answered from the
depth image, with a PSMZ32 inverse page table, depth surfaces given a
texture set, and `export_depth`'s float-to-integer conversion in
`surface_word`) was built and measured: frames 5500-5700 differed
natively (146-716 pixels in small boxes, so not rounding) even when
restricted to 24-bit readers, and the traffic barely moved (2431 Mpx
imported over the run, from 2527). Dropped; the reason it could not
help is in the attribution below: the memory-sampled (view) path is not
where the Z buffer's reads go.

Frame 6100 imports and exports attributed to the command that forced
them (`PS2_HW_TRACE`, pixels):

| what | cause | px |
|---|---|---|
| export depth 2240 | compute-drawn (DATE) batches | 655k in 60 rects |
| import depth 2240 | the next hardware batch | 614k |
| import frame 4480/10/01 (24-bit alias) | a batch's texture | 512k |
| import frame 2240/10/01 (24-bit alias) | a batch's texture | 328k |
| import depth 2272 | a batch | 328k |
| export depth 2240 | own write-back | 287k |
| export frame 2240/10/01 | a batch | 258k |
| export frame 6720 / import frame 6720 | DATE batches | 231k + 225k |
| export frame 6720 | uploads | 211k |

So the DATE batches on the compute path drive about 1.7 Mpx of the 6.9
per frame directly (the depth and frame round trips around each one),
and the 24-bit alias imports are texture decodes (cacheable textures
decoded from memory), not views. The hardware DATE patch is being
re-measured on the fixed base for the traffic it removes.

Re-measured hardware DATE on the fixed base (the saved patch applied):
traffic 6.9 -> 6.1 Mpx per frame, compute raster 4.5 -> 1.6 ms, but
hardware raster 34.4 -> 39.7 ms, total 45.5 -> 47.5 ms, threaded
unchanged at 37-40 ms/frame. Still parked: the 17 DATE batches each cost
a pass close, a snapshot copy and their own imports, about 0.3 ms apiece,
more than the round trips they save.

State at the end of 2026-10-28: switch-port is verified (native frames
bit-identical, Switch presented frames clean, frame 6100 identical) at
36-39 ms/frame threaded; GPU 45.5 ms in the profile (the alias
write-backs of the correctness fix cost about 1 ms over the 44.6
before). The remaining GPU budget is structural: 6.1-6.9 Mpx of
import/export per frame from GT4's aliasing of its frame, its halves and
its Z buffer, at about 1.4 ms per Mpx. Candidates, none measured yet:
cheaper import/export shaders; serving the 24-bit-alias texture decodes
from the frame surface image (a surface_texture through a converting
copy) instead of memory; and a DATE path that does not close the pass
(stencil pre-pass inside the pass is not possible either, since the
snapshot needs the pass closed; a per-batch read of the own target
through an input attachment would).

### 2026-10-28: cheaper import/export shaders, measured

Baseline for the day (Switch, race frames 6050-6650, GPU profile): total
45.5 ms; no draws (bit 27) 18.1 ms of hardware raster; no draws and no
copies (27 + 32768) 8.7 ms, so the copies cost 9.4 ms for 6.9 Mpx in 381
dispatches per frame (`PS2_HW_TRACE`: 239 of them 4 kpx or smaller,
together only 0.5 Mpx; 304 copy calls, 274 with a single rectangle).

Two shader changes, both bit-identical natively and on the Switch:

1. The copy entry points divided by the uniform `hw.scale` four times
   per sample. The hardware module is compiled once per scale, so `SCALE`
   is a constant there; the divisions now fold away. The draw shaders had
   the same pattern per fragment (the GS pixel and sample index) and got
   the same change.
2. `export_depth` reads the depth image through a sampled view (set 1
   binding 6, `texture_depth_2d`; depth surfaces now carry SAMPLED usage,
   colour surfaces bind a 1x1 dummy) instead of a `vkCmdCopyImageToBuffer`
   into `depth_stage` behind two barriers. About 62 depth exports per
   frame lost their copy. Imports still stage through the buffer.

| run | raster-hw ms/frame | total ms/frame | threaded ms/frame |
|---|---|---|---|
| baseline 33082db | 34.5 (no draws 18.1, no copies 8.7) | 45.5 | 38.1-39.6 |
| copy shaders (d26780c, 2e7daf8) | 32.8 (no draws 16.5, no copies 8.5) | 43.8 | 36.6-37.6 |
| plus draw shaders (818db0c) | 32.3 (no draws 16.6, no copies 8.6) | 43.3 | 35.9-37.3 |

Where the remaining 8.1 ms of copies goes: `PS2_HW_DEBUG` bit 29
(536870912) keeps every barrier, bind and dispatch but launches one
workgroup per copy; with it (and bit 27) the hardware raster is 10.4 ms
against 8.5 without copies and 16.5 with them. So the 381 dispatches
cost about 2.0 ms (5 us each: barrier, bind, launch) and the samples 6.1
ms, 0.88 ms per Mpx. At 12-16 bytes moved per sample that is within
about 2x of the Tegra's memory bandwidth, so the per-sample work is
close to its floor; packing a call's rectangles into one dispatch would
remove only 77 of the 381 launches (most calls carry one rectangle). The
copy cost is now a per-event cost: the way down is fewer copy events,
i.e. the structural candidates above (24-bit alias decodes from the
frame image; a DATE path inside the pass), not cheaper shaders.

### 2026-10-28: the exact route for the alias traffic, measured

Three steps toward serving GT4's alias reads from the images that hold
them instead of round-tripping through GS memory, all bit-identical
natively and on the Switch (`PS2_HW_DEBUG` bit 30 disables them):

1. `import_from_aliases`: invalid tiles of a surface that another
   resident surface of the same kind and pitch holds valid at a
   whole-page offset are filled by `vkCmdCopyImage` (the chain draws into
   one half of its 24-bit frame and reads it back through a texture based
   at the other half; the second Z buffer sits one page after the first).
2. Texture decodes whose texels a resident 32/24-bit surface holds
   entirely read that surface's image (`surface_word` through
   `target_load`, flags bit 12) instead of forcing its write-back. Only
   when the surface already holds every texel: decoding through a surface
   that had to import them first cost 790 kpx of imports per frame, and
   letting reads keep a surface resident kept an atlas region alive for
   another 660 kpx per frame.
3. `viewable_surface` prefers the surface whose tiles under the span are
   all valid, so the chain's views read the drawn half directly and the
   copies of step 1 mostly disappear.

| build | imports Mpx (600 frames) | write-backs | alias copies | GPU total ms | threaded ms/frame |
|---|---|---|---|---|---|
| 818db0c (morning) | 2527 | 1666 | 0 | 43.3 | 35.9-37.3 |
| steps 1+2 (08b55ac) | 2265 | 1567 | 311 | 43.5 | 36.4-37.1 |
| step 3 | 2231 | 1518 | 184 | 43.0 | 36.1-36.3 |

Lessons: an image copy on the Tegra costs about what the compute round
trip it replaces did (step 1 alone was neutral), so the win only comes
from not moving the pixels at all (step 3). The remaining traffic in
frame 6100 (6.2 Mpx, from 6.9) is the depth 2240 round trips around the
compute-drawn DATE batches (1.6 Mpx), the frame 6720 round trips for
DATE and uploads (1.0 Mpx), the target imports of the post-process frame
and its Z buffer (2240/10/01, 2272/10/31, 4480/10/01: 0.9 Mpx, forced by
`memory_written` writing the old Z back and invalidating it before the
targets import), and the scratch 11200/4/00 imports (0.36 Mpx).

### 2026-10-28: presenting through a swapchain

The Vulkan device had no window surface, so the windowed runner took the
SDL path for every frame: the display dispatch into scratch, a GPU copy
to a host-cached buffer, a host copy, `SDL_UpdateTexture` and a GL blit
by SDL's renderer, on the same GPU. `vk/present.cpp` now attaches a
`VK_KHR_surface` to the X11 (or Wayland) window the host hands to
`attach_window`, builds a swapchain and draws scratch with the shared
blit shader (`gs/gpu/blit.wgsl`, compiled to SPIR-V with the rest) at
the end of each frame's command buffer. Decisions: acquires never block
(timeout 0; a frame with no free image is skipped and counted, which
never happened in 1,100 race frames); mailbox is preferred, then
immediate (what the Switch's X server offers), then FIFO; the blit
shader keeps WebGPU's clip space (naga `--keep-coordinate-space`), so a
negative-height viewport flips it; the platform surface structures are
declared locally since the Switch sysroot has no X11 headers. The SDL
path remains the fallback (`PS2_SDL_RENDERER=1`, or no extensions) and
is still where `PS2_DUMP_PRESENT` works.

Switch, threaded race, A/B back to back twice: SDL path 35.3-36.7
ms/frame, swapchain 34.8-36.1. Main-thread timers: the present scope
outside submit fell from about 5.4 to 2.4 ms. Screenshot (scrot on the
Switch) correct and letterboxed. This also matters for Horizon, where
the SDL renderer path would not exist: the device now owns its
presentation end to end.

### 2026-10-28: the CPU side, measured on the render thread

Main-thread-only profile on the Switch (perf -t on the render thread,
threaded race, 35-36 ms/frame): 65% in ps2run, 17% libc (memcpy), 12%
the NVIDIA driver recording commands. Top symbols: memcpy 13.2% (9.4
points of it the upload copies into the arena, 2.9 MB per frame),
gpu::draw 6.7, gif::render 5.3, memory_written 4.0, write_packed 3.1,
kick 3.1, import_from_aliases 3.0, load_clut 2.8, gs::write 2.8,
append_prims 2.8, HwRaster::draw 2.6, vertex 2.3, supported 2.2,
viewable_surface 2.1. `PS2_VK_MEMBENCH` showed the arena copies run at
heap speed (2.0 GB/s, DRAM-bound on the A57); the staging (host-coherent)
memory takes 1.1 GB/s. So the upload cost is the copy itself (ring to
arena, after the EE thread's copy into the ring); removing it means
restructuring the shared upload path.

Done in our code: alias copies and views test tiles without allocating;
tile maps as one 64-bit word per row; the block-id fill skipped while no
texture recorded ids and no upload rewrote them since a covering fill.
Each bit-identical; each within noise on the frame time. Per-thread
samples explain why: the render thread runs at about 88% of a core, the
EE and VU threads at 65% each, and the GPU (43 ms in the serialised
profile, so about the 35 ms frame when overlapped) sets the frame. CPU
savings on the render thread bank for when the GPU drops; the next
CPU items, in order of size, are the upload copy (shared code, ~3 ms),
the GS front end (gif::render, write_packed, kick, vertex, gpu::draw:
about 20% of the thread, shared code) and the per-batch record work in
HwRaster::draw (supported, append_prims, texture_texels).

### 2026-10-28: where the re-decodes come from, and partial decodes

Fixed cost on the current build (Switch, no draws, hardware raster 16.4
ms): no render passes 11.4, never re-decode 11.2, no copies 9.2, and
with `PS2_HW_DEBUG` bit 21 (one-workgroup decodes) 11.3. So the ~5 ms of
decodes was texel work, not launches: 151 decodes per race frame, 4.8
Mpx of texels. `PS2_TEX_TRACE=<frame>` shows them: 70 8-bit 128x64
tiles whose memory (inside the Z buffer's range) the post-process
rewrites every frame, and 4-bit atlases up to 512x512 re-uploaded with
new content each frame (upload ids traced across frames: 131 of 193
upload ranges move, 17 change content; nothing false-stale there).
Correction to the earlier attribution: the depth exports around these
batches come from the DATE batches' compute depth test (their setups:
flags 6aaf, DATM 0, blend, Z test GEQUAL without write, alpha written
unmasked, texture 11200/4/00), not from texture reads; a depth view
built to serve Z reads as textures never triggered and was dropped.
Hardware DATE stays out: with alpha written from a texture, a stencil
mirror of the destination alpha cannot be exact.

Partial decodes: cells keep a decoded flag; a decode covers the sampled
cells that are stale or undecoded. Texels decoded 4.8 -> 2.0 Mpx per
frame; GPU total 43.0 -> 40.6 ms; threaded 34.8-35.2 ms/frame (the
30 fps budget is 33.3). Still on the list: the DATE batches (compute
raster 4.5 ms plus the depth and frame round trips around them, about
1 ms), 213 render passes (~5 ms), the copies (7 ms, 6.2 Mpx), and the
post-process target imports forced by write-back ordering (0.9 Mpx).
