// What the GS renderer needs from a graphics API. The batching, hazard
// tracking and upload logic in gpu.cpp is API-neutral and talks to one of
// these; wgpu_device.cpp implements it with wgpu-native (and Dawn in the
// browser), and runtime/src/gs/vk/ implements it with Vulkan directly.
#pragma once

#include "gpu.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps2::gs::gpu {

// The compute entry points of gs.wgsl, in the order the pipelines are built.
enum Pipeline { kUpload, kCopyRead, kCopyWrite, kLoadClut, kDisplay, kRaster, kResolve, kSnapshot, kDecode, kPipelines };
extern const char *const kEntryPoints[kPipelines];

// Each parameter record is one uniform block this long, indexed by Command::param.
constexpr std::uint32_t kParamStride = 256;

// A byte range of GS memory, `hi` exclusive.
struct Span {
    std::uint32_t lo = 0, hi = 0;
};

struct Command {
    Pipeline pipeline;
    std::uint32_t x, y;  // workgroups
    std::uint32_t param;
    // For kRaster: the batch's primitive and setup record counts, so a device
    // with a rasterizer of its own can rebuild the draw from the arena.
    std::uint32_t prims = 0, setups = 0;
    // The batching layer built no tile lists for this raster batch because the
    // device promised (batch_hardware) to draw it itself.
    bool hardware = false;
    // The GS memory the command reads and writes, for a device that keeps
    // render targets outside GS memory and has to write them back before
    // memory is read and drop them when memory is written. `ranged` false
    // means unknown: assume everything.
    bool ranged = false;
    Span reads[4] = {};
    Span writes[2] = {};
};

// The storage buffers the shaders bind, in binding order; plus the
// staging buffer for reads. Sizes are decided by the device from `Limits`.
enum class Buffer { Vram, Tables, Params, Data, Clut, Scratch, Cache, Decoded };

struct Limits {
    std::uint32_t scale;        // samples per pixel axis
    std::uint64_t vram_bytes;   // kMemorySize * scale^2
    std::uint64_t clut_bytes;
    std::uint64_t scratch_bytes;
    std::uint64_t cache_bytes;
    std::uint64_t decoded_bytes;
    std::uint64_t readback_bytes;
    std::uint32_t inverse32_words = 0;  // offset in `tables` of the PSMCT32 inverse page table (2048 words: x | y << 8 per word of a page)
};

// Parameters of the final blit to the window, written each presented frame.
struct Blit {
    std::uint32_t words[8];
};

struct Device {
    virtual ~Device() = default;

    // Creates the API objects, the buffers of `limits`, uploads the swizzle
    // tables, and builds the compute pipelines for the given scale. Prints
    // the adapter it chose. False when the API is unusable.
    virtual bool init(const Limits &limits, const std::vector<std::uint32_t> &tables) = 0;
    virtual bool timestamps() const = 0;

    // Writes bytes into a buffer through the queue, ordered before later submits.
    virtual void write(Buffer b, std::uint64_t offset, const void *data, std::uint64_t bytes) = 0;

    // Runs the commands in order with `params` and `arena` as the uniform
    // and data buffers. With `present` set, also draws the blit with the
    // last `Blit` written and presents the surface. The device keeps its
    // own GPU-time measurement for busy_ms().
    virtual void submit(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words, bool present) = 0;
    // As submit, but every command is timed; returns nanoseconds per command
    // (at most the first 2048) after waiting for the GPU. Only when timestamps().
    virtual void submit_profiled(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words,
                                 std::vector<std::uint64_t> &ns) = 0;

    // Copies from a buffer into `out`, waiting for the GPU.
    virtual void read_back(Buffer b, std::uint64_t offset, std::uint64_t bytes, void *out) = 0;
    // Queues a copy into a staging slot and returns the slot, or -1 when none
    // is free; poll_read copies the result into `out` and frees the slot once
    // the GPU is done, returning 1, or 0 while pending, or -1 on failure.
    // wait_read blocks until done. Up to kReadSlots are in flight.
    static constexpr int kReadSlots = 4;
    virtual int begin_read(Buffer b, std::uint64_t offset, std::uint64_t bytes) = 0;
    virtual int poll_read(int slot, void *out) = 0;
    virtual int wait_read(int slot, void *out) = 0;

    virtual void wait_idle() = 0;
    virtual double busy_ms() = 0;
    // Prints the device's own counters at exit, after the renderer's.
    virtual void stats() {}
    // A device with a rasterizer of its own says which setups it cannot draw,
    // so the batching layer keeps them out of batches it can (gpu.cpp emits
    // a batch boundary when this answer changes).
    virtual bool setup_unsupported(const std::uint32_t *) const { return false; }
    // Will the device draw this batch itself (not through the compute
    // rasterizer)? Then the batching layer skips the tile lists, decode plans
    // and snapshots that only the compute rasterizer reads.
    virtual bool batch_hardware(const std::uint32_t *, std::uint32_t, const std::uint32_t *, std::uint32_t) const { return false; }
    // Names a CLUT source (its GS memory span) for `id`, the value setup
    // word 22 carries above the slot bits; a texture cache keys on it.
    virtual void clut_source(std::uint32_t, Span) {}
    // An upload wrote these 256-byte blocks (addresses / 256) with content
    // identified by `id` (a hash of the bytes and their place in the image):
    // the same upload repeated gives the same ids, so a texture cache can
    // tell a texture re-streamed unchanged from one that changed.
    virtual void blocks_written(const std::uint32_t *, std::size_t, std::uint64_t) {}
    // Memory the next submit's arena may be built in directly (the device's
    // own mapped buffer for that submit), so nothing is copied at submit time.
    // False when the device has no such memory; the caller then keeps its own.
    virtual bool arena_storage(std::uint32_t *&, std::size_t &) { return false; }

    virtual bool attach_surface(const SurfaceTarget &target) = 0;
    virtual void resize_surface(std::uint32_t width, std::uint32_t height) = 0;
    virtual bool has_surface() const = 0;
    virtual void set_blit(const Blit &blit) = 0;
    virtual std::uint32_t surface_width() const = 0;
    virtual std::uint32_t surface_height() const = 0;
};

Device *make_wgpu_device();
// The Vulkan device in runtime/src/gs/vk/, when built in; null otherwise.
Device *make_vk_device();

}
