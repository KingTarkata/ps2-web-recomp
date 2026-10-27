#include "vk.hpp"

#include "context.hpp"
#include "../gs.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2::gs::vk {

namespace {

Context g_ctx;
bool g_active = false;
std::uint32_t g_scale = 1;
// GS memory in the console's swizzled layouts, scale^2 samples per pixel
// when upscaling, exactly as the compute renderer keeps it.
Buffer g_vram;
// 1024 CLUT entries per slot, 32-bit, for the palettes in flight.
Buffer g_clut;

}

bool wanted() {
    const char *backend = std::getenv("PS2_GS");
    return backend && std::strcmp(backend, "hw") == 0;
}

bool init(std::uint32_t scale) {
    if (g_active) return true;
    if (!g_ctx.init()) return false;
    g_scale = scale;
    const VkDeviceSize vram_bytes = static_cast<VkDeviceSize>(kMemorySize) * scale * scale;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!g_ctx.create_buffer(g_vram, vram_bytes, usage, false) || !g_ctx.create_buffer(g_clut, 64 * 1024 * 4, usage, false)) {
        std::fprintf(stderr, "[vk] cannot allocate GS memory on the device\n");
        shutdown();
        return false;
    }
    VkCommandBuffer cb = g_ctx.begin_once();
    vkCmdFillBuffer(cb, g_vram.buffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cb, g_clut.buffer, 0, VK_WHOLE_SIZE, 0);
    g_ctx.end_once(cb);
    g_active = true;
    std::fprintf(stderr, "[vk] hardware GS backend ready: %u MiB of GS memory at scale %u\n", static_cast<unsigned>(vram_bytes >> 20), scale);
    return true;
}

bool active() { return g_active; }

void shutdown() {
    if (g_ctx.device) {
        g_ctx.destroy_buffer(g_vram);
        g_ctx.destroy_buffer(g_clut);
    }
    g_ctx.destroy();
    g_active = false;
}

const char *device_name() { return g_ctx.name.c_str(); }

}
