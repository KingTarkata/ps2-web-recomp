#include "present.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern const std::uint32_t kBlitSpv[];
extern const std::size_t kBlitSpv_words;

namespace ps2::gs::vk {

namespace {

constexpr VkDeviceSize kUniformStride = 256;  // >= minUniformBufferOffsetAlignment everywhere

// The platform surface structures, declared here so the build needs no X11
// or Wayland headers (the Switch sysroot has none): the layouts are those of
// vulkan_xlib.h and vulkan_wayland.h, with the native handles as void *.
struct XlibSurfaceCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    void *dpy;
    unsigned long window;
};
struct WaylandSurfaceCreateInfo {
    VkStructureType sType;
    const void *pNext;
    VkFlags flags;
    void *display;
    void *surface;
};
using CreateXlibSurface = VkResult(VKAPI_PTR *)(VkInstance, const XlibSurfaceCreateInfo *, const VkAllocationCallbacks *, VkSurfaceKHR *);
using CreateWaylandSurface = VkResult(VKAPI_PTR *)(VkInstance, const WaylandSurfaceCreateInfo *, const VkAllocationCallbacks *, VkSurfaceKHR *);

}  // namespace

bool Presenter::attach(Context &ctx, const gpu::SurfaceTarget &target, VkBuffer pixels, VkDeviceSize pixels_bytes, int frames) {
    if (!ctx.ext_surface || !ctx.ext_swapchain || !vkCreateSwapchainKHR || !vkGetPhysicalDeviceSurfaceSupportKHR) return false;
    ctx_ = &ctx;
    want_ = {target.width, target.height};
    VkResult r = VK_ERROR_EXTENSION_NOT_PRESENT;
    if (target.kind == gpu::SurfaceTarget::X11 && ctx.ext_xlib) {
        const auto create = reinterpret_cast<CreateXlibSurface>(vkGetInstanceProcAddr(ctx.instance, "vkCreateXlibSurfaceKHR"));
        XlibSurfaceCreateInfo info{VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR, nullptr, 0, target.display, static_cast<unsigned long>(target.window)};
        if (create) r = create(ctx.instance, &info, nullptr, &surface_);
    } else if (target.kind == gpu::SurfaceTarget::Wayland && ctx.ext_wayland) {
        const auto create = reinterpret_cast<CreateWaylandSurface>(vkGetInstanceProcAddr(ctx.instance, "vkCreateWaylandSurfaceKHR"));
        WaylandSurfaceCreateInfo info{VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR, nullptr, 0, target.display, target.surface};
        if (create) r = create(ctx.instance, &info, nullptr, &surface_);
    }
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] no window surface (%s); presenting through SDL\n", result_name(r));
        surface_ = VK_NULL_HANDLE;
        return false;
    }
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(ctx.physical, ctx.queue_family, surface_, &supported);
    if (!supported) {
        std::fprintf(stderr, "[vk] the queue cannot present to the window; presenting through SDL\n");
        destroy();
        return false;
    }
    // Format: the blit writes linear RGB, as the SDL path showed raw bytes.
    std::uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physical, surface_, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physical, surface_, &count, formats.data());
    format_ = count ? formats[0].format : VK_FORMAT_B8G8R8A8_UNORM;
    for (const VkSurfaceFormatKHR &f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) {
            format_ = f.format;
            break;
        }
    // Present mode: never block the render thread on vblank (the emulator
    // paces itself): mailbox, else immediate, else FIFO with non-blocking
    // acquires that skip a frame when no image is free.
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physical, surface_, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physical, surface_, &count, modes.data());
    mode_ = VK_PRESENT_MODE_FIFO_KHR;
    if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end()) mode_ = VK_PRESENT_MODE_MAILBOX_KHR;
    else if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end()) mode_ = VK_PRESENT_MODE_IMMEDIATE_KHR;
    if (std::getenv("PS2_PRESENT_FIFO")) mode_ = VK_PRESENT_MODE_FIFO_KHR;  // measurement: vsync-locked presents

    // Render pass: one colour attachment, cleared, left in present layout.
    VkAttachmentDescription at{};
    at.format = format_;
    at.samples = VK_SAMPLE_COUNT_1_BIT;
    at.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    at.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    at.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    at.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    at.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    at.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;
    // The acquire semaphore is waited for at colour output; the implicit
    // dependency orders the layout transition after it.
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1;
    rpi.pAttachments = &at;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sub;
    rpi.dependencyCount = 1;
    rpi.pDependencies = &dep;
    if (vkCreateRenderPass(ctx.device, &rpi, nullptr, &pass_) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding b[2] = {};
    b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 2;
    lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(ctx.device, &lci, nullptr, &set_layout_) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &set_layout_;
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &layout_) != VK_SUCCESS) return false;

    VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mci.codeSize = kBlitSpv_words * 4;
    mci.pCode = kBlitSpv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &mci, nullptr, &module) != VK_SUCCESS) return false;
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = module;
    stages[0].pName = "vs";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = module;
    stages[1].pName = "fs";
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &blend;
    const VkDynamicState dynamic[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dynamic;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vp;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &dy;
    gpi.layout = layout_;
    gpi.renderPass = pass_;
    const VkResult pr = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline_);
    vkDestroyShaderModule(ctx.device, module, nullptr);
    if (pr != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] blit pipeline: %s; presenting through SDL\n", result_name(pr));
        destroy();
        return false;
    }

    if (!ctx.create_buffer(uniforms_, kUniformStride * static_cast<VkDeviceSize>(frames), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true)) return false;
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<std::uint32_t>(frames)}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<std::uint32_t>(frames)}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = static_cast<std::uint32_t>(frames);
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &pool_) != VK_SUCCESS) return false;
    sets_.assign(static_cast<std::size_t>(frames), VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> layouts(static_cast<std::size_t>(frames), set_layout_);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool_;
    dai.descriptorSetCount = static_cast<std::uint32_t>(frames);
    dai.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(ctx.device, &dai, sets_.data()) != VK_SUCCESS) return false;
    for (int i = 0; i < frames; ++i) {
        VkDescriptorBufferInfo px{pixels, 0, pixels_bytes};
        VkDescriptorBufferInfo ub{uniforms_.buffer, kUniformStride * static_cast<VkDeviceSize>(i), sizeof(gpu::Blit)};
        VkWriteDescriptorSet w[2] = {};
        for (int k = 0; k < 2; ++k) {
            w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet = sets_[static_cast<std::size_t>(i)];
            w[k].dstBinding = static_cast<std::uint32_t>(k);
            w[k].descriptorCount = 1;
        }
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[0].pBufferInfo = &px;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[1].pBufferInfo = &ub;
        vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
    }
    acquired_.assign(static_cast<std::size_t>(frames), VK_NULL_HANDLE);
    done_.assign(static_cast<std::size_t>(frames), VK_NULL_HANDLE);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (int i = 0; i < frames; ++i)
        if (vkCreateSemaphore(ctx.device, &sci, nullptr, &acquired_[static_cast<std::size_t>(i)]) != VK_SUCCESS || vkCreateSemaphore(ctx.device, &sci, nullptr, &done_[static_cast<std::size_t>(i)]) != VK_SUCCESS) return false;
    if (!create_swapchain()) {
        destroy();
        return false;
    }
    std::fprintf(stderr, "[vk] presenting through a %ux%u swapchain (%s, %s)\n", extent_.width, extent_.height, format_ == VK_FORMAT_B8G8R8A8_UNORM ? "BGRA8" : format_ == VK_FORMAT_R8G8B8A8_UNORM ? "RGBA8" : "other format",
                 mode_ == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : mode_ == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo");
    return true;
}

bool Presenter::create_swapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_->physical, surface_, &caps) != VK_SUCCESS) return false;
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFFu) extent = want_;
    extent.width = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    if (!extent.width || !extent.height) return false;
    std::uint32_t count = std::max<std::uint32_t>(caps.minImageCount, mode_ == VK_PRESENT_MODE_MAILBOX_KHR ? 3u : 2u);
    if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface_;
    sci.minImageCount = count;
    sci.imageFormat = format_;
    sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;
    sci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : static_cast<VkCompositeAlphaFlagBitsKHR>(caps.supportedCompositeAlpha & -caps.supportedCompositeAlpha);
    sci.presentMode = mode_;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = swapchain_;
    VkSwapchainKHR created = VK_NULL_HANDLE;
    const VkResult r = vkCreateSwapchainKHR(ctx_->device, &sci, nullptr, &created);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] vkCreateSwapchainKHR: %s\n", result_name(r));
        return false;
    }
    destroy_swapchain();
    swapchain_ = created;
    extent_ = extent;
    count = 0;
    vkGetSwapchainImagesKHR(ctx_->device, swapchain_, &count, nullptr);
    images_.resize(count);
    vkGetSwapchainImagesKHR(ctx_->device, swapchain_, &count, images_.data());
    views_.assign(count, VK_NULL_HANDLE);
    framebuffers_.assign(count, VK_NULL_HANDLE);
    for (std::uint32_t i = 0; i < count; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = images_[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format_;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(ctx_->device, &vci, nullptr, &views_[i]) != VK_SUCCESS) return false;
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = pass_;
        fci.attachmentCount = 1;
        fci.pAttachments = &views_[i];
        fci.width = extent_.width;
        fci.height = extent_.height;
        fci.layers = 1;
        if (vkCreateFramebuffer(ctx_->device, &fci, nullptr, &framebuffers_[i]) != VK_SUCCESS) return false;
    }
    recreate_ = false;
    return true;
}

void Presenter::destroy_swapchain() {
    if (!ctx_) return;
    for (VkFramebuffer f : framebuffers_)
        if (f) vkDestroyFramebuffer(ctx_->device, f, nullptr);
    for (VkImageView v : views_)
        if (v) vkDestroyImageView(ctx_->device, v, nullptr);
    framebuffers_.clear();
    views_.clear();
    images_.clear();
    if (swapchain_) vkDestroySwapchainKHR(ctx_->device, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

void Presenter::destroy() {
    if (!ctx_) return;
    vkQueueWaitIdle(ctx_->queue);
    destroy_swapchain();
    for (VkSemaphore s : acquired_)
        if (s) vkDestroySemaphore(ctx_->device, s, nullptr);
    for (VkSemaphore s : done_)
        if (s) vkDestroySemaphore(ctx_->device, s, nullptr);
    acquired_.clear();
    done_.clear();
    if (pool_) vkDestroyDescriptorPool(ctx_->device, pool_, nullptr);
    ctx_->destroy_buffer(uniforms_);
    if (pipeline_) vkDestroyPipeline(ctx_->device, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(ctx_->device, layout_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(ctx_->device, set_layout_, nullptr);
    if (pass_) vkDestroyRenderPass(ctx_->device, pass_, nullptr);
    if (surface_ && vkDestroySurfaceKHR) vkDestroySurfaceKHR(ctx_->instance, surface_, nullptr);
    pool_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    pass_ = VK_NULL_HANDLE;
    surface_ = VK_NULL_HANDLE;
}

void Presenter::report() const {
    if (presented_ || skipped_) std::fprintf(stderr, "[vk] presented %llu frames through the swapchain, %llu skipped with no free image\n", static_cast<unsigned long long>(presented_), static_cast<unsigned long long>(skipped_));
}

bool Presenter::acquire(int slot, std::uint32_t &image, VkSemaphore &wait) {
    if (!swapchain_) return false;
    if (recreate_) {
        // The old images may still be presenting: let them finish first.
        vkQueueWaitIdle(ctx_->queue);
        if (!create_swapchain()) return false;
    }
    wait = acquired_[static_cast<std::size_t>(slot)];
    const VkResult r = vkAcquireNextImageKHR(ctx_->device, swapchain_, 0, wait, VK_NULL_HANDLE, &image);
    if (r == VK_SUBOPTIMAL_KHR) recreate_ = true;
    else if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_ = true;
        ++skipped_;
        return false;
    } else if (r != VK_SUCCESS) {
        ++skipped_;
        return false;
    }
    return true;
}

void Presenter::record(VkCommandBuffer cb, int slot, std::uint32_t image) {
    std::memcpy(static_cast<std::uint8_t *>(uniforms_.mapped) + kUniformStride * static_cast<VkDeviceSize>(slot), blit_.words, sizeof blit_.words);
    ctx_->flush(uniforms_, kUniformStride * static_cast<VkDeviceSize>(slot), sizeof blit_.words);
    // The display dispatch's pixels are read by the fragment shader.
    VkMemoryBarrier mem{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mem.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    mem.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &mem, 0, nullptr, 0, nullptr);
    VkClearValue clear{};
    clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = pass_;
    rbi.framebuffer = framebuffers_[image];
    rbi.renderArea = {{0, 0}, extent_};
    rbi.clearValueCount = 1;
    rbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &sets_[static_cast<std::size_t>(slot)], 0, nullptr);
    // Letterbox (Blit words 5, 6: the viewport the aspect allows). The blit
    // shader was written for WebGPU's clip space, whose y points up, and
    // naga keeps it (--keep-coordinate-space): a negative viewport height
    // (Vulkan 1.1) flips it back.
    const float w = static_cast<float>(extent_.width), h = static_cast<float>(extent_.height);
    const float vw = std::min(w, static_cast<float>(blit_.words[5])), vh = std::min(h, static_cast<float>(blit_.words[6]));
    const VkViewport viewport{(w - vw) * 0.5f, (h - vh) * 0.5f + vh, vw, -vh, 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, extent_};
    vkCmdSetViewport(cb, 0, 1, &viewport);
    vkCmdSetScissor(cb, 0, 1, &scissor);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRenderPass(cb);
}

void Presenter::present(std::uint32_t image, VkSemaphore wait) {
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &wait;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &image;
    const VkResult r = vkQueuePresentKHR(ctx_->queue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) recreate_ = true;
    else if (r != VK_SUCCESS) std::fprintf(stderr, "[vk] vkQueuePresentKHR: %s\n", result_name(r));
    ++presented_;
}

}  // namespace ps2::gs::vk
