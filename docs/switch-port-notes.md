# Switch port notes (Phase 0 survey)

Survey of upstream ps2-web-recomp at commit 3339995, made before any Switch
work, to find what stands between the native runner and an aarch64 / Horizon
build. Line numbers refer to that commit.

## 1. x86-specific code

**There are no x86 intrinsics and no inline assembly anywhere in `runtime/`,
`recomp/` or `host/`.** Every SIMD operation uses GCC/Clang vector extensions
(`__attribute__((vector_size(16)))`, `__builtin_convertvector`,
`__builtin_elementwise_min`, `__builtin_reduce_or`, `__builtin_shufflevector`),
which Clang lowers to NEON on aarch64 without source changes:

- `runtime/include/ps2/cpu.hpp` defines the `Reg128` union and the 8-, 16-,
  32-, 64-bit lane views used by the MMI instructions (`mmi.hpp`).
- `runtime/include/ps2/vu_ops.hpp:351-418` defines `v4u/v4i/v4f` and the VU
  vector helpers (`vfix`, `vadd`, `vmadd`, `vftoi`, `vitof`, ...).
- Generated EE code (`generated/<game>/unit_*.cpp`) and generated VU1 code
  (`vurecomp` output) only call these helpers; `recomp/vurecomp.cpp:189-191`
  emits `vmuln`/`vmaddn`/`vmsubn`, never raw float expressions.

The one x86-only file is `host/profile.cpp`, a sampling profiler that reads
`REG_RIP`/`REG_RSP` from `ucontext`. CMake already compiles it only on
x86-64 Linux (`CMakeLists.txt`, `PS2_PROFILE`). An aarch64 version would read
`uc_mcontext.pc` and `uc_mcontext.sp`; it is optional.

Alignment: `Reg128` is `alignas(16)`; guest RAM, scratchpad, VU memories and
GS memory are `alignas(64)` static arrays. All guest memory accesses go through
`memcpy`-based `load`/`store` templates in `memory.hpp`, so there are no
unaligned-pointer dereferences to worry about on ARM.

No `-march`, `-mavx`, `-ffast-math` or `-ffp-contract` flags are set anywhere.

## 2. Floating point rules

**EE FPU (`runtime/include/ps2/fpu.hpp`)**: add, sub and mul are implemented
with integer mantissa arithmetic (no host float ops at all). Division, sqrt and
rsqrt convert to `double`, use the host's IEEE division/sqrt (correctly rounded
on every platform), and pack back with PS2 overflow/underflow rules. Nothing
here depends on x86 rounding behaviour; results are bit-identical on aarch64.

**VU0 macro mode and VU1 (`vu_ops.hpp`)**: use host `float` add/sub/mul
wrapped in `vfix` (flush denormals, clamp to ±0x7F7FFFFF). IEEE single add/mul
are correctly rounded on SSE and NEON alike, so results match provided:

1. Rounding mode is round-to-nearest-even (default on both).
2. Flush-to-zero is off (default on both; `vfix` handles denormals itself
   anyway).
3. **No FMA contraction.** `vmaddn` computes `acc + vnorm(a*b)`; the `vnorm`
   bit operations between the multiply and the add make fusion impossible,
   and scalar `madd` likewise normalises in between. Clang on aarch64 defaults
   to `-ffp-contract=on`, which fuses only within one expression, so today's
   code is safe, but add `-ffp-contract=off` to the aarch64 build as insurance
   against future upstream changes.
4. `vftoi` masks out-of-range lanes before `__builtin_convertvector`, so the
   difference between x86 (`0x80000000` on overflow) and NEON (saturating)
   conversions never reaches the result.

The upstream README notes the host multiplier is already not bit-exact against
real hardware; Phase 2's determinism test compares x86 against aarch64, not
against the console.

## 3. Native runner: platform dependencies

| Area | Where | Notes |
|---|---|---|
| wgpu-native | `runtime/src/gs/gpu/gpu.cpp` only (1862 lines) | Instance/adapter/device setup at `gpu.cpp:1100-1200`; requires only `maxStorageBufferBindingSize` and `maxBufferSize` at the adapter's values, plus `TimestampQuery` when available. Compute passes, buffer copies, async map-backs for frame downloads, one render pipeline for presenting (`blit.wgsl`). |
| WGSL shaders | `runtime/src/gs/gpu/gs.wgsl` (1144 lines), `blit.wgsl` | Workgroup sizes 64, 256, 8×8, 16×16; one `override` constant (`SCALE`); 8 atomic ops; no f16, no subgroups, no texture sampling. All compute on storage buffers. Good fit for SPIR-V precompilation. |
| SDL2 | `host/window.cpp` only (374 lines) | Window, Vulkan surface via `SDL_SysWMinfo`, audio queue (`SDL_QueueAudio`), game controller with axes and rumble, `SDL_GetPerformanceCounter` frame pacing, `SDL_Delay`. Falls back to `SDL_Renderer` texture upload when wgpu cannot present. |
| Threads | `host/window.cpp:356-368`, `runtime/src/system.cpp:460-518`, `runtime/src/gs/queue.*` | Three roles: Emulation, Vu, Render. `std::thread` for emulation and VU1, GS render thread started by `gs::start_thread()`. Lock-free rings (`queue::Ring`, 32 MiB each) with atomics and futex-style `wait`/`notify`. One `std::mutex` for the texture cache in `gs.cpp`. `PS2_ONE_THREAD=1` runs everything on one thread. |
| Timers | `system.cpp:126-136`, `gs.cpp` | `std::chrono::steady_clock` only. |
| Filesystem | `disc.cpp:66-82` (`fopen`/`fseeko`/`fread` of the ISO, one reader), `system.cpp` (ELF load, patches, save states, RAM dump), `sio2.cpp` (memory card `.ps2` via fstream), `vu_micro.cpp` (microprogram recording via `std::filesystem`), `gs.cpp`/`gpu.cpp` (PPM frame dumps) | All plain stdio/fstream; nothing POSIX-specific beyond `fseeko`. |
| Guest thread switching | `runtime/src/dispatch.cpp`, `kernel.cpp` | No `ucontext`, `setjmp` or signals: a thread switch sets `c.pc |= kUnwind` and the generated code returns up the host stack. Only the IOP HLE uses `setjmp`/`longjmp` (`iop/libs.cpp`). Portable. |
| Environment variables | `PS2_GPU_LOW_POWER`, `PS2_GPU_FALLBACK`, `PS2_GPU_PROFILE`, `PS2_SCALE`, `PS2_DECODE`, `PS2_NO_DEINTERLACE`, `PS2_VU_RECORD`, `PS2_VU_INTERPRET`, `PS2_VU_CHECK`, `PS2_DUMP`, `PS2_PEEK`, `PS2_TIME`, `PS2_ONE_THREAD`, `PS2_PROFILE` | Horizon has no environment; the Switch platform layer will need a config file or defaults for these. |

## 4. Unseen VU1 microprograms natively

`runtime/src/vu_micro.cpp:838-880`: on VU1 kick the runtime looks up a
compiled program by address/hash. If none exists, it falls back to the
interpreter (`interpret(1, pc)`) and counts the call in
`g_interpreted_calls`. The runtime WebAssembly JIT (`vu_jit.cpp` + `vu_jit_web.cpp`)
is browser-only: `jit::mode()` returns 0 when `__EMSCRIPTEN__` is not defined.
So on the Switch any microprogram not recorded during `port.sh` exploration
runs interpreted, which is why Phase 5 needs a long recording run. The runner
reports interpreted calls at exit, which gives a direct way to check coverage.

## 5. Build pipeline facts

- `scripts/port.sh` needs `unzip` (via `fetch_wgpu.sh`), `curl`, `cmake`,
  `ninja`, a C++20 compiler with `<format>` (`scripts/cxx.sh` picks clang or
  GCC 13+), SDL2 dev headers for the window, and a Vulkan device for the
  headless exploring runs (llvmpipe works, slowly).
- Retail GT4 translates to 328 units (~87 MB of C++). Each exploration round
  rebuilds `ps2run`, which is the dominant cost on an 8-core VM.
- Retail GT4 ships a small loader ELF plus deflated `CORE.GT4`;
  `scripts/gt4_core.py` unpacks it. The Online Beta / Spec II disc has **no
  `CORE.GT4`** and **no `IOPRP300.IMG`**: `SCUS_974.36` is a plain 6.5 MB ELF,
  and the disc adds `DNAS310.IMG`, `IRX/LGAUD.IRX`, `IRX/PPP.IRX`,
  `IRX/PPPOE.IRX` and `NET/DNAS_NET.ERX` / `NET/DNAS_MC.ERX` / `NET/NTG_ISCK.ERX`
  over retail. Retail already loads `INET`, `NETCNF`, `DEV9`, `SMAP`, `PPP`
  and `RTINETB` at boot under upstream, so the network stack itself is not new.

## 6. Peak memory use

Retail GT4, 3000 frames headless on llvmpipe (so wgpu buffers live in host
RAM), measured with `/usr/bin/time -v`:

| Peak RSS | 268 MB |
|---|---|

Static allocations alone: 32 MiB EE RAM, 2 MiB IOP RAM, 4 MiB GS memory, two
32 MiB ring buffers. The Switch has 4 GB with roughly 3 GB usable by an
application, so memory is not a constraint; this number is for the record.

## 7. Spec II executable layout (Phase 0b)

- The Online Beta disc boots a 492 KB loader `SCUS_974.36` (base 0x1000000)
  that unpacks the real program, the way retail's `SCUS_973.28` unpacks
  `CORE.GT4`. Spec II replaces that file with the **unpacked core itself**
  (6.5 MB, base 0x100000), so `scripts/gt4_core.py` is not involved.
- Spec II's own code is injected with "PS2PLUGININJECTOR" as a third `LOAD`
  segment, section `.plugin` at 0x68A480-0x68D14C, whose address range lies
  inside the data segment's. The startup code calls into it (`jal 0x68A768`
  at 0x1001E8) before `main`. Its strings name the mod's hooks:
  `toggleBonnetCam`, `toggleGT3Cam`, `toggleTriggerSensitivity`,
  `toggleGaugeMode`, `overrideLaps`, `toggleMultiCategoryAI`, and others.
- **Blocker 1 (fixed):** `recomp/elf.cpp` resolved overlapping segments to the
  first one listed, so eerecomp saw zeros where the loader puts the plugin,
  translated them as `nop`s and the game ran off the end of the segment at
  frame 0. Segments are now walked in reverse so the later one wins, matching
  `load_elf` in the runtime. Retail translation is byte-identical after the
  change. Worth sending upstream.

- **Exploration result (3600 frames, default press script):** the game boots
  through its IOP reboot with `DNAS310.IMG`, loads the same IRX set as retail
  plus `PPP.IRX` and `PPPOE.IRX`, shows Spec II's language-select screen, then
  the "Welcome to Gran Turismo 4 Spec II" name entry. **No unhandled syscall,
  no missing IOP library and no rejected GS feature was logged.** The only
  new line is `dev9: unknown dev9 hardware.` from `DEV9.IRX`, which is the
  desired "no network adapter" outcome. Two VU1 microprograms were recorded.
- **Headless input.** Upstream's default `--press` script cannot get past the
  USA first-boot name entry, and the memory card format prompt defaults to
  Cancel with timing that drifts between runs. `scripts/blank_card.py` writes
  a formatted empty 8 MB card (PCSX2 layout, with ECC) so the prompt never
  appears. Sequences that work on llvmpipe at 60 Hz:
  - Retail: `300:start 600:cross 900:left 1200:cross 1500:right 1800:cross`
    then `cross` every 300 frames. Reaches the title, Arcade mode and car
    selection with 37 VU1 programs recorded and no misses.
  - Spec II: `300:start 600:cross 900:cross 1200:cross 1500:up 1800:left
    2100:cross 2400:right 2700:down 3000:cross` then `cross` every 300 frames
    (the language screen comes first, and its keyboard has Caps/Sym keys).
- **Blocker 2 (open):** with the blank card, Spec II accepts the name, creates
  game data, shows the "Introduction" screen, reloads its IOP modules, and at
  frame 5401 calls through a null virtual function:
  `jalr` at 0x505070 reads a vtable slot `{adjust, fn}` at `*(obj+4) + 72`
  with `fn == 0`, obj at 0x00E39B64. Retail GT4 goes through the same flow
  without trouble, so this is specific to Spec II's executable or data.
  A RAM dump at the halt (`PS2_DUMP`) shows the object is not a C++ object
  with a real vtable: `obj+4` points to 0x01FF6A40, near the top of RAM,
  into a block holding `0xCCCCCCCC` fill words and repeated
  `{1, 0x006DAF70}` pairs, and the strings next to the object read
  `__builtin__::__projects__::BootProject::CourseData::ArcadeCo...`, so this
  is the game's script object system (the "adhoc" projects) dispatching a
  method on an object whose slot 9 holds nil. A write watch
  (`PS2_WATCH=1FF6A80,10`) shows the nil was stored by the VM's own code
  (0x4E9998), not by DMA or the runtime, so the script ran a lookup that
  returned nil. The last plugin code to run before the crash is the mod's
  module init (`mMyModule`, callbacks 0x68C710 / 0x68C758, registering a
  native `GetHookSTDCVersion`), invoked from the game's module list walker
  at 0x4DD370.
- **With an existing Spec II save** (the user's `specii.ps2` from PCSX2) the
  same build goes title → Arcade → car selection (744 cars) → a fully
  rendered 3D car view, with no misses. So Spec II's engine, plugin and VU1
  paths work; only the first-boot script path fails. `port.sh --card FILE`
  (new) seeds exploring runs with such a card.
- **Spec II races.** With the save card and this script the game reaches a
  Single Race at the Nürburgring Nordschleife in a Pontiac Vibe GT, with the
  driving-options dialog, the grid, and in-car and chase cameras all drawn
  correctly, 77 VU1 programs recorded and no misses over 9000 frames:
  `300:start 600:cross 900:cross 1200:cross 1500:cross 1800:cross 2100:cross
  2400:right 2550:right 2700:right 2850:right 3000:right 3150:cross` then
  `cross` every 300 frames.
- The game data the crashed first-boot run wrote to the card is valid: a
  second run booting from that card goes straight to the title and into
  Arcade mode like the PCSX2 save does, and the directory layout and file
  sizes match the PCSX2 card exactly (garage 1344064 bytes, GAMEDATA 237952,
  icon0.ico 10704, icon.sys 964). So the memory card path round-trips and
  the first-boot crash is confined to the script that runs once after game
  data creation.
- The plugin's module init runs identically on the saved-game path and the
  new-game path (probe comparison), so the mod's injected code is not what
  breaks the first-boot script.
