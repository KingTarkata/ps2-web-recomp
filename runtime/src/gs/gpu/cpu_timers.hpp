// CPU timers for the renderer's host-side work, on when PS2_HW_TIMERS is
// set. Header-only and platform-neutral so gpu.cpp and the Vulkan device
// can share the same table; the Vulkan device prints it with its stats.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "ps2/system.hpp"

namespace cpu_timers {

enum Section : int {
    kSubmit,        // VkDevice_::submit, whole (record + queue submit; includes the sections below it)
    kWaitFrame,     // fence wait for the frame slot (GPU-bound time shows here)
    kDraw,          // HwRaster::draw, hardware-drawn batches
    kBefore,        // HwRaster::before_command, compute-run commands
    kTextureFor,    // HwRaster::texture_for, whole
    kTexels,        //   texture_texels
    kSlot,          //   texture_slot
    kBlocks,        //   texel_blocks
    kCurrent,       //   texture_current
    kDecode,        //   decode_texture (recording; the GPU work is elsewhere)
    kReadBack,      // read_back / begin_read / end_read, including flush_sync and fence waits
    kWrite,         // VkDevice_::write (queue waits)
    kUploadBlocks,  // gpu.cpp: blocks_of + blocks_written per upload chunk
    kGif,           // render thread: gif::render (register writes, vertices, batch building)
    kVblankFlush,   // render thread: gpu::flush at vblank (flush_all + submit)
    kPresent,       // render thread: on_present (display + present)
    kDownload,      // render thread: downloads
    kFlushBatch,    // gpu.cpp: flush_batch
    kEmitAll,       // gpu.cpp: emit_all
    kLoadClut,      // gpu.cpp: load_clut
    kQueueUpload,   // gpu.cpp: queue_upload
    kCount
};

inline const char *const kNames[kCount] = {"submit", "wait_frame", "draw", "before_command", "texture_for", "  texels", "  slot", "  blocks", "  current", "  decode", "read_back", "write", "upload_blocks", "gif", "vblank_flush", "present", "download", "flush_batch", "emit_all", "load_clut", "queue_upload"};

inline std::uint64_t g_ns[kCount] = {};
inline std::uint64_t g_calls[kCount] = {};

// PS2_HW_TIMERS=1 times every frame; PS2_HW_TIMERS=from,to only the
// frames in [from, to), so a heavy window is not averaged with menus.
struct Window {
    bool on = false;
    std::uint64_t from = 0, to = ~0ull;
};
inline const Window &window() {
    static const Window w = [] {
        Window r;
        const char *spec = std::getenv("PS2_HW_TIMERS");
        if (!spec) return r;
        r.on = true;
        unsigned long long a = 0, b = 0;
        if (std::sscanf(spec, "%llu,%llu", &a, &b) == 2) {
            r.from = a;
            r.to = b;
        }
        return r;
    }();
    return w;
}
inline bool enabled() { return window().on; }
inline bool active() {
    const Window &w = window();
    if (!w.on) return false;
    const std::uint64_t f = ps2::frame_count();
    return f >= w.from && f < w.to;
}

inline std::chrono::steady_clock::time_point &first_sample() {
    static std::chrono::steady_clock::time_point t;
    return t;
}

struct Scope {
    Section s;
    bool on;
    std::chrono::steady_clock::time_point t0;
    explicit Scope(Section s_) : s(s_), on(active()) {
        if (on) {
            t0 = std::chrono::steady_clock::now();
            if (first_sample() == std::chrono::steady_clock::time_point{}) first_sample() = t0;
        }
    }
    ~Scope() {
        if (on) {
            g_ns[s] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            ++g_calls[s];
        }
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
};

inline void report(std::uint64_t frames) {
    if (!enabled()) return;
    const Window &w = window();
    if (w.to != ~0ull) frames = w.to - w.from;
    const double f = frames ? static_cast<double>(frames) : 1.0;
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - first_sample()).count();
    std::fprintf(stderr, "[cpu] host time over %llu frames and %.1f s of wall time: ms per frame, %% of wall, calls per frame\n", static_cast<unsigned long long>(frames), wall_ms / 1e3);
    for (int i = 0; i < kCount; ++i)
        std::fprintf(stderr, "  %-16s %8.3f  %5.1f%%  %8.1f\n", kNames[i], static_cast<double>(g_ns[i]) / 1e6 / f, wall_ms > 0 ? static_cast<double>(g_ns[i]) / 1e6 / wall_ms * 100.0 : 0.0, static_cast<double>(g_calls[i]) / f);
}

}  // namespace cpu_timers
