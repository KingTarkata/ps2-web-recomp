// The Vulkan instance, device and queue the hardware GS backend works with,
// plus small helpers for buffers, images and one-shot command buffers.
#pragma once

#include "loader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ps2::gs::vk {

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void *mapped = nullptr;  // set for host-visible buffers
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0, height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceFeatures features{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkCommandPool pool = VK_NULL_HANDLE;
    std::string name;
    bool has_timestamps = false;
    float timestamp_period = 0.0f;

    // Picks a device (an environment variable PS2_GPU_LOW_POWER prefers an
    // integrated one, otherwise a discrete one, otherwise the first) and
    // creates the device with a queue that does graphics, compute and
    // transfer. Returns false, with a message, when Vulkan is unusable.
    bool init();
    void destroy();

    std::uint32_t memory_type(std::uint32_t allowed, VkMemoryPropertyFlags wanted, VkMemoryPropertyFlags fallback) const;
    bool create_buffer(Buffer &b, VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible);
    void destroy_buffer(Buffer &b);
    bool create_image(Image &i, std::uint32_t width, std::uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect);
    void destroy_image(Image &i);

    // One command buffer recorded, submitted and waited for; for setup work
    // and tests. Rendering proper uses its own ring.
    VkCommandBuffer begin_once();
    void end_once(VkCommandBuffer cb);
};

const char *result_name(VkResult r);

}
