// Presents frames through a Vulkan swapchain on the window the runner opened:
// the display dispatch leaves the frame's pixels in the scratch buffer and
// the blit shader (gs/gpu/blit.wgsl, shared with the wgpu device) draws them
// into the acquired image, letterboxed, deinterlacing by blending fields when
// asked. Replaces the read-back, host copies and SDL renderer of the SDL
// path, which stays as the fallback (PS2_SDL_RENDERER=1, or no surface
// extension).
#pragma once
#include "../gpu/device.hpp"
#include "context.hpp"

#include <vector>

namespace ps2::gs::vk {

class Presenter {
public:
    // False when the window kind or the driver's extensions cannot be used;
    // the runner then presents through SDL.
    bool attach(Context &ctx, const gpu::SurfaceTarget &target, VkBuffer pixels, VkDeviceSize pixels_bytes, int frames);
    void destroy();
    bool active() const { return swapchain_ != VK_NULL_HANDLE; }
    std::uint32_t width() const { return extent_.width; }
    std::uint32_t height() const { return extent_.height; }
    void resize(std::uint32_t width, std::uint32_t height) {
        want_ = {width, height};
        recreate_ = true;
    }
    void set_blit(const gpu::Blit &blit) { blit_ = blit; }
    // An image for frame slot `slot`, without blocking: false skips this
    // frame's present (no image free yet, or the swapchain is being rebuilt).
    bool acquire(int slot, std::uint32_t &image, VkSemaphore &wait);
    // Records the blit pass into `cb`; the submit waits for `wait` at colour
    // output and signals `signal(slot)`, which present() then waits for.
    void record(VkCommandBuffer cb, int slot, std::uint32_t image);
    VkSemaphore signal(int slot) const { return done_[static_cast<std::size_t>(slot)]; }
    void present(std::uint32_t image, VkSemaphore wait);
    void report() const;  // presented / skipped counts (stats)

private:
    bool create_swapchain();
    void destroy_swapchain();

    Context *ctx_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{0, 0}, want_{0, 0};
    VkPresentModeKHR mode_ = VK_PRESENT_MODE_FIFO_KHR;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    std::vector<VkFramebuffer> framebuffers_;
    VkRenderPass pass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;  // per frame slot
    Buffer uniforms_;                    // per frame slot, kUniformStride apart
    std::vector<VkSemaphore> acquired_, done_;
    gpu::Blit blit_{};
    bool recreate_ = false;
    std::uint64_t presented_ = 0, skipped_ = 0;
};

}  // namespace ps2::gs::vk
