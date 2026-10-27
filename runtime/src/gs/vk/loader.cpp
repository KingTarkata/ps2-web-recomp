#include "loader.hpp"

#include <cstdio>
#include <dlfcn.h>
#include <initializer_list>

namespace ps2::gs::vk {

#define PS2_VK_DEFINE(name) PFN_##name name = nullptr;
PS2_VK_GLOBAL_FUNCTIONS(PS2_VK_DEFINE)
PS2_VK_INSTANCE_FUNCTIONS(PS2_VK_DEFINE)
PS2_VK_DEVICE_FUNCTIONS(PS2_VK_DEFINE)
#undef PS2_VK_DEFINE

PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;

namespace {
void *g_library = nullptr;
}

bool load_library() {
    if (g_library) return true;
    for (const char *name : {"libvulkan.so.1", "libvulkan.so"}) {
        g_library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (g_library) break;
    }
    if (!g_library) {
        std::fprintf(stderr, "[vk] no Vulkan loader library\n");
        return false;
    }
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(g_library, "vkGetInstanceProcAddr"));
    if (!vkGetInstanceProcAddr) return false;
    bool ok = true;
#define PS2_VK_LOAD(name) \
    name = reinterpret_cast<PFN_##name>(vkGetInstanceProcAddr(nullptr, #name)); \
    ok = ok && name != nullptr;
    PS2_VK_GLOBAL_FUNCTIONS(PS2_VK_LOAD)
#undef PS2_VK_LOAD
    return ok;
}

bool load_instance_functions(VkInstance instance) {
    bool ok = true;
#define PS2_VK_LOAD(name) \
    name = reinterpret_cast<PFN_##name>(vkGetInstanceProcAddr(instance, #name)); \
    if (!name) { std::fprintf(stderr, "[vk] missing %s\n", #name); ok = false; }
    PS2_VK_INSTANCE_FUNCTIONS(PS2_VK_LOAD)
#undef PS2_VK_LOAD
    return ok;
}

bool load_device_functions(VkDevice device) {
    bool ok = true;
#define PS2_VK_LOAD(name) \
    name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name)); \
    if (!name) { std::fprintf(stderr, "[vk] missing %s\n", #name); ok = false; }
    PS2_VK_DEVICE_FUNCTIONS(PS2_VK_LOAD)
#undef PS2_VK_LOAD
    return ok;
}

}
