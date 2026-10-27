#include "context.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2::gs::vk {

const char *result_name(VkResult r) {
    switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    default: return "VkResult";
    }
}

namespace {

bool prefer_low_power() { return std::getenv("PS2_GPU_LOW_POWER") != nullptr; }

int device_score(const VkPhysicalDeviceProperties &p) {
    const bool low = prefer_low_power();
    switch (p.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return low ? 2 : 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return low ? 4 : 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return std::getenv("PS2_GPU_FALLBACK") ? 5 : 1;
    default: return 0;
    }
}

}

bool Context::init() {
    if (!load_library()) return false;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ps2run";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkResult r = vkCreateInstance(&ici, nullptr, &instance);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateInstance: %s\n", result_name(r));
        return false;
    }
    if (!load_instance_functions(instance)) return false;

    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    int best = -1;
    for (VkPhysicalDevice d : devices) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        const int score = device_score(p);
        if (score > best) {
            best = score;
            physical = d;
        }
    }
    if (physical == VK_NULL_HANDLE) {
        std::fprintf(stderr, "[vk] no Vulkan device\n");
        return false;
    }
    vkGetPhysicalDeviceProperties(physical, &properties);
    vkGetPhysicalDeviceFeatures(physical, &features);
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    name = properties.deviceName;

    std::uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
    std::vector<VkQueueFamilyProperties> qf(families);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, qf.data());
    queue_family = ~0u;
    for (std::uint32_t i = 0; i < families; ++i) {
        const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if ((qf[i].queueFlags & need) == need) {
            queue_family = i;
            has_timestamps = qf[i].timestampValidBits > 0 && properties.limits.timestampComputeAndGraphics;
            break;
        }
    }
    if (queue_family == ~0u) {
        std::fprintf(stderr, "[vk] no graphics+compute queue\n");
        return false;
    }
    timestamp_period = properties.limits.timestampPeriod;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    // Only features every target has are enabled; anything optional is
    // queried per use.
    VkPhysicalDeviceFeatures enabled{};
    enabled.independentBlend = features.independentBlend;
    enabled.dualSrcBlend = features.dualSrcBlend;
    enabled.shaderInt16 = features.shaderInt16;
    enabled.fragmentStoresAndAtomics = features.fragmentStoresAndAtomics;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &enabled;
    r = vkCreateDevice(physical, &dci, nullptr, &device);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateDevice: %s\n", result_name(r));
        return false;
    }
    if (!load_device_functions(device)) return false;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family;
    r = vkCreateCommandPool(device, &pci, nullptr, &pool);
    if (r != VK_SUCCESS) return false;
    std::fprintf(stderr, "[vk] %s, Vulkan %u.%u.%u, driver %u\n", name.c_str(), VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion),
                 VK_VERSION_PATCH(properties.apiVersion), properties.driverVersion);
    return true;
}

void Context::destroy() {
    if (device) {
        vkDeviceWaitIdle(device);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (instance) vkDestroyInstance(instance, nullptr);
    pool = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
}

std::uint32_t Context::memory_type(std::uint32_t allowed, VkMemoryPropertyFlags wanted, VkMemoryPropertyFlags fallback) const {
    for (VkMemoryPropertyFlags flags : {wanted, fallback}) {
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((allowed & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    return ~0u;
}

bool Context::create_buffer(Buffer &b, VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bci, nullptr, &b.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    const VkMemoryPropertyFlags wanted = host_visible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const VkMemoryPropertyFlags fallback = host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : 0;
    const std::uint32_t type = memory_type(req.memoryTypeBits, wanted, fallback);
    if (type == ~0u) return false;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &mai, nullptr, &b.memory) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(device, b.buffer, b.memory, 0) != VK_SUCCESS) return false;
    b.size = size;
    if (host_visible && vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped) != VK_SUCCESS) return false;
    return true;
}

void Context::destroy_buffer(Buffer &b) {
    if (b.mapped) vkUnmapMemory(device, b.memory);
    if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device, b.memory, nullptr);
    b = Buffer{};
}

bool Context::create_image(Image &i, std::uint32_t width, std::uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &ici, nullptr, &i.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, i.image, &req);
    const std::uint32_t type = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (type == ~0u) return false;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &mai, nullptr, &i.memory) != VK_SUCCESS) return false;
    if (vkBindImageMemory(device, i.image, i.memory, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = i.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    if (vkCreateImageView(device, &vci, nullptr, &i.view) != VK_SUCCESS) return false;
    i.format = format;
    i.width = width;
    i.height = height;
    i.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

void Context::destroy_image(Image &i) {
    if (i.view) vkDestroyImageView(device, i.view, nullptr);
    if (i.image) vkDestroyImage(device, i.image, nullptr);
    if (i.memory) vkFreeMemory(device, i.memory, nullptr);
    i = Image{};
}

VkCommandBuffer Context::begin_once() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &ai, &cb) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

void Context::end_once(VkCommandBuffer cb) {
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device, &fci, nullptr, &fence);
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, ~0ull);
    vkDestroyFence(device, fence, nullptr);
}

}
