// The hardware-rasterized GS backend (docs/gs-hw-renderer.md). It owns its
// own Vulkan device, keeps GS memory in a buffer like the compute renderer
// and draws through graphics pipelines. Selected with PS2_GS=hw.
#pragma once

#include <cstdint>

namespace ps2::gs::vk {

// True when the environment asks for this backend (PS2_GS=hw).
bool wanted();

}
