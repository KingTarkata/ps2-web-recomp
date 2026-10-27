// The hardware-rasterized GS backend (docs/gs-hw-renderer.md). It owns its
// own Vulkan device, keeps GS memory in a buffer like the compute renderer
// and draws through graphics pipelines. Selected with PS2_GS=hw.
#pragma once

#include <cstdint>

namespace ps2::gs::vk {

// True when the environment asks for this backend.
bool wanted();
// Creates the device and the GS memory buffers. False leaves the compute
// renderer in charge.
bool init(std::uint32_t scale);
bool active();
void shutdown();
// Device name for logs, empty before init.
const char *device_name();

}
