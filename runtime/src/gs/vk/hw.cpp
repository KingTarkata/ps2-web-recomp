#include "hw.hpp"
#include "ps2/system.hpp"
#include "../gs.hpp"
#include "../gpu/cpu_timers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2::gs::vk {

namespace {

constexpr std::uint32_t kPrimWords = 28, kSetupWords = 25;
constexpr std::uint64_t kVertexBytes = 32u << 20, kUniformBytes = 32u << 20;  // a batch promised to the hardware path (no tile lists) must not run out of scratch
constexpr std::uint32_t kUniformStride = 256;
constexpr std::uint32_t kUniformWords = 20 + 28;  // HwParams: 20 words plus the setup record padded to 7 vec4s
constexpr std::uint32_t kMaxFailModePrims = 1024;

constexpr std::uint32_t F_TME = 2, F_ABE = 8, F_ZMSK = 32, F_ATE = 64, F_DATE = 128, F_ZTE = 512, F_PABE = 1024, F_COLCLAMP = 2048;
constexpr std::uint64_t kMemoryBytes = 4u << 20;

void page_size(std::uint32_t psm, std::uint32_t &pw, std::uint32_t &ph) {
    switch (psm) {
    case 0x02: case 0x0A: case 0x32: case 0x3A: pw = 64; ph = 64; break;  // 16-bit
    case 0x13: pw = 128; ph = 64; break;                                   // PSMT8
    case 0x14: pw = 128; ph = 128; break;                                  // PSMT4
    default: pw = 64; ph = 32; break;
    }
}

// A frame mask as a colour write mask: every byte all set or all clear.
// Returns false for partial masks, which the hardware path leaves to compute.
bool write_mask(std::uint32_t fbmsk, std::uint32_t &out) {
    out = 0;
    static const VkColorComponentFlags bits[4] = {VK_COLOR_COMPONENT_R_BIT, VK_COLOR_COMPONENT_G_BIT, VK_COLOR_COMPONENT_B_BIT, VK_COLOR_COMPONENT_A_BIT};
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t byte = (fbmsk >> (i * 8)) & 0xFFu;
        if (byte == 0) out |= bits[i];
        else if (byte != 0xFFu) return false;
    }
    return true;
}

float fat(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint32_t z_at(std::uint32_t z0, float d) {
    const float step = std::floor(d);
    if (step >= 0.0f) return z0 + static_cast<std::uint32_t>(std::min(step, static_cast<float>(0xFFFFFFFFu - z0)));
    return z0 - static_cast<std::uint32_t>(std::min(-step, static_cast<float>(z0)));
}

}

bool HwRaster::init(Context &ctx, const std::uint32_t *code, std::size_t words, const std::uint32_t *const variants[8], const std::size_t variant_words[8], VkDescriptorSetLayout set0, std::uint32_t scale) {
    ctx_ = &ctx;
    scale_ = scale;
    VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mci.codeSize = words * 4;
    mci.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &mci, nullptr, &module) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] the driver rejected the hardware raster shaders\n");
        return false;
    }
    if (std::getenv("PS2_HW_DEBUG") && (static_cast<unsigned>(std::atoi(std::getenv("PS2_HW_DEBUG"))) & 1048576u)) {
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_OCCLUSION;
        qci.queryCount = 2;
        if (vkCreateQueryPool(ctx.device, &qci, nullptr, &occlusion_) != VK_SUCCESS) occlusion_ = VK_NULL_HANDLE;
    }
    for (int v = 0; v < 8; ++v) {
        variant_modules_[v] = VK_NULL_HANDLE;
        if (!variants || !variants[v] || std::getenv("PS2_HW_GENERIC")) continue;  // PS2_HW_GENERIC: the unspecialised shader only, for A/B runs
        VkShaderModuleCreateInfo vci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        vci.codeSize = variant_words[v] * 4;
        vci.pCode = variants[v];
        if (vkCreateShaderModule(ctx.device, &vci, nullptr, &variant_modules_[v]) != VK_SUCCESS) variant_modules_[v] = VK_NULL_HANDLE;
    }
    VkDescriptorSetLayoutBinding b[6] = {};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 6;
    lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(ctx.device, &lci, nullptr, &set1_layout_) != VK_SUCCESS) return false;
    // Samplers for the cached-texture path: bilinear, REPEAT or CLAMP per axis.
    for (int i = 0; i < 4; ++i) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = (i & 2) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = (i & 1) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 0.0f;
        if (vkCreateSampler(ctx.device, &sci, nullptr, &samplers_[i]) != VK_SUCCESS) return false;
    }
    VkDescriptorSetLayoutBinding t[6] = {};
    t[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    t[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    for (int i = 0; i < 4; ++i) t[2 + i] = {static_cast<std::uint32_t>(2 + i), VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, &samplers_[i]};
    VkDescriptorSetLayoutCreateInfo tci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    tci.bindingCount = 6;
    tci.pBindings = t;
    if (vkCreateDescriptorSetLayout(ctx.device, &tci, nullptr, &set2_layout_) != VK_SUCCESS) return false;
    VkDescriptorSetLayout sets[3] = {set0, set1_layout_, set2_layout_};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 3;
    pli.pSetLayouts = sets;
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &layout_) != VK_SUCCESS) return false;
    // One set per surface and frame slot; surfaces come and go, so sets are
    // freed, but only two submits after their surface was retired. A burst of
    // short-lived targets (GT4 creates hundreds of one-page buffers over the
    // race-start frames) must not exhaust the pool meanwhile: a batch promised
    // to the hardware path has nothing to fall back on.
    const std::uint32_t max_sets = static_cast<std::uint32_t>(kMaxSurfaces) * 4 * 32 + 2;
    VkDescriptorPoolSize sizes[4] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, max_sets}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_sets * 2}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, max_sets * 2}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_sets}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = max_sets;
    dpi.poolSizeCount = 4;
    dpi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &pool_) != VK_SUCCESS) return false;
    const std::uint32_t max_tex = static_cast<std::uint32_t>(kMaxTextures + kMaxSurfaces) * 4 + 8;  // texture sets are freed late too
    VkDescriptorPoolSize sizes2[3] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, max_tex}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_tex}, {VK_DESCRIPTOR_TYPE_SAMPLER, max_tex * 4}};
    VkDescriptorPoolCreateInfo dpi2{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi2.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi2.maxSets = max_tex;
    dpi2.poolSizeCount = 3;
    dpi2.pPoolSizes = sizes2;
    if (vkCreateDescriptorPool(ctx.device, &dpi2, nullptr, &pool2_) != VK_SUCCESS) return false;
    for (FrameScratch &f : frames_) {
        // Host-cached: the CPU fills these with many small writes (a vertex
        // field at a time); uncached write-combined memory made append_prims
        // 8% of the main thread on the Switch. Flushed in end_frame.
        if (!ctx.create_buffer(f.vertices, kVertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, true)) return false;
        if (!ctx.create_buffer(f.uniforms, kUniformBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, true)) return false;
    }
    if (!ctx.create_buffer(depth_stage_, static_cast<VkDeviceSize>(kTargetSize) * kTargetSize * scale * scale * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false)) return false;
    // Depth surfaces never use the image bindings of set 1, but every
    // binding is written so no descriptor is left undefined.
    if (!ctx.create_image(dummy_, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    {
        VkCommandBuffer cb = ctx.begin_once();
        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = dummy_.image;
        bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
        ctx.end_once(cb);
    }
    if (!ctx.create_image(decoded_, 2048, 2048, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    {
        VkCommandBuffer cb = ctx.begin_once();
        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = decoded_.image;
        bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
        ctx.end_once(cb);
    }
    if (!build_pipelines(module)) return false;
    VkDescriptorSetLayout layouts[2] = {set1_layout_, set1_layout_};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool_;
    dai.descriptorSetCount = 2;
    dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(ctx.device, &dai, misc_set1_) != VK_SUCCESS) return false;
    for (int i = 0; i < 2; ++i) write_set1(misc_set1_[i], i, dummy_.view);
    frame_ = &frames_[0];
    return true;
}

bool HwRaster::write_set1(VkDescriptorSet set, int slot, VkImageView surface_view) {
    VkDescriptorBufferInfo ub{frames_[slot].uniforms.buffer, 0, kUniformWords * 4};
    VkDescriptorImageInfo image{VK_NULL_HANDLE, surface_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo stage{depth_stage_.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo dec{VK_NULL_HANDLE, decoded_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[6] = {};
    for (int k = 0; k < 6; ++k) {
        w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet = set;
        w[k].dstBinding = static_cast<std::uint32_t>(k);
        w[k].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    w[0].pBufferInfo = &ub;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[1].pImageInfo = &image;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[2].pImageInfo = &image;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[3].pBufferInfo = &stage;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[4].pImageInfo = &dec;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[5].pImageInfo = &dec;
    vkUpdateDescriptorSets(ctx_->device, 6, w, 0, nullptr);
    return true;
}

void HwRaster::dispatch_decode(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, decode_);
    bind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, set0, misc_set1_[slot_], cmd.param * gpu::kParamStride, 0);
    vkCmdDispatch(cb, cmd.x, cmd.y, 1);
}

bool HwRaster::build_pipelines(VkShaderModule module) {
    // Render passes: colour RGBA8, optionally depth D32, both kept in
    // GENERAL layout so the import/export passes can address them directly.
    for (int with_depth = 0; with_depth < 2; ++with_depth) {
        VkAttachmentDescription at[2] = {};
        at[0].format = VK_FORMAT_R8G8B8A8_UNORM;
        at[0].samples = VK_SAMPLE_COUNT_1_BIT;
        at[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        at[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        at[0].initialLayout = at[0].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        at[1].format = VK_FORMAT_D32_SFLOAT;
        at[1].samples = VK_SAMPLE_COUNT_1_BIT;
        at[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        at[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        at[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        at[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[1].initialLayout = at[1].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkAttachmentReference color{0, VK_IMAGE_LAYOUT_GENERAL}, depth{1, VK_IMAGE_LAYOUT_GENERAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &color;
        sub.pDepthStencilAttachment = with_depth ? &depth : nullptr;
        VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpi.attachmentCount = with_depth ? 2 : 1;
        rpi.pAttachments = at;
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sub;
        if (vkCreateRenderPass(ctx_->device, &rpi, nullptr, with_depth ? &pass_depth_rp_ : &pass_nodepth_) != VK_SUCCESS) return false;
    }
    // Compute pipelines for the import/export passes.
    const char *compute_names[6] = {"import_frame", "export_frame", "export_depth", "import_depth", "decode", "decode_tex"};
    VkPipeline *compute_out[6] = {&import_frame_, &export_frame_, &export_depth_, &import_depth_, &decode_, &decode_tex_};
    for (int i = 0; i < 6; ++i) {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = compute_names[i];
        ci.layout = layout_;
        if (vkCreateComputePipelines(ctx_->device, VK_NULL_HANDLE, 1, &ci, nullptr, compute_out[i]) != VK_SUCCESS) {
            std::fprintf(stderr, "[vk] cannot build %s\n", compute_names[i]);
            return false;
        }
    }
    module_ = module;
    return true;
}

// Blend key layout: bit 0 enabled, bits 1-5 src factor, bits 6-10 dst
// factor, bits 11-13 op, bit 14 uses blend constants (FIX), bit 15 the
// fragment outputs the factor C as its colour, bit 16 it outputs Cs*(1+C),
// bit 17 the destination-alpha two-step form Cs*Ad/128 + Cd (drawn twice
// per primitive; the pipeline bits describe both steps).
namespace {
constexpr std::uint32_t kFactorBits = 5;
std::uint32_t make_blend_key(VkBlendFactor src, VkBlendFactor dst, VkBlendOp op, bool constant) {
    return 1u | (static_cast<std::uint32_t>(src) << 1) | (static_cast<std::uint32_t>(dst) << 6) | (static_cast<std::uint32_t>(op) << 11) | (constant ? 1u << 14 : 0u);
}
}

std::uint32_t HwRaster::blend_key(const std::uint32_t *w, bool dual_source) {
    const std::uint32_t a = (w[7] >> 12) & 3u, b = (w[7] >> 14) & 3u, c = (w[7] >> 16) & 3u, d = (w[7] >> 18) & 3u;
    // Factor C: source alpha (dual-source output, As/128 clamped to 1),
    // destination alpha (no exact mapping: unorm alpha is /255), or FIX/128.
    VkBlendFactor fc, fnc;
    bool constant = false;
    if (c == 1) {
        // Destination alpha: only the additive form, as two steps.
        if (a == 0 && b == 2 && d == 1) return make_blend_key(VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD, false) | (1u << 17);
        return 0;
    }
    if (c == 0) {
        if (!dual_source) return 0;
        fc = VK_BLEND_FACTOR_SRC1_COLOR;
        fnc = VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    } else if (c == 2) {
        if (w[9] > 128) return 0;
        fc = VK_BLEND_FACTOR_CONSTANT_COLOR;
        fnc = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        constant = true;
    } else {
        return 0;
    }
    const auto key = [&](VkBlendFactor src, VkBlendFactor dst, VkBlendOp op) { return make_blend_key(src, dst, op, constant); };
    // (A - B) * C + D over selectors Cs (0), Cd (1), zero (2).
    if (a == b) {
        if (d == 0) return key(VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD);
        if (d == 1) return key(VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD);
        return key(VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD);
    }
    if (a == 0 && b == 1) {  // (Cs - Cd) * C + D
        if (d == 1) return key(fc, fnc, VK_BLEND_OP_ADD);
        if (d == 2) return key(fc, fc, VK_BLEND_OP_SUBTRACT);
        return 0;
    }
    if (a == 1 && b == 0) {  // (Cd - Cs) * C + D
        if (d == 0) return key(fnc, fc, VK_BLEND_OP_ADD);
        if (d == 2) return key(fc, fc, VK_BLEND_OP_REVERSE_SUBTRACT);
        return 0;
    }
    if (a == 0 && b == 2) {  // Cs * C + D
        if (d == 1) return key(fc, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD);
        if (d == 2) return key(fc, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD);
        // Cs * (1 + C): the fragment outputs the scaled colour (exact: the
        // GS clamps the same sum).
        return key(VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD) | (1u << 16);
    }
    if (a == 1 && b == 2) {  // Cd * C + D
        if (d == 0) return key(VK_BLEND_FACTOR_ONE, fc, VK_BLEND_OP_ADD);
        if (d == 2) return key(VK_BLEND_FACTOR_ZERO, fc, VK_BLEND_OP_ADD);
        // Cd * (1 + C): the fragment outputs C, scaled by the destination.
        return key(VK_BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD) | (1u << 15);
    }
    if (a == 2 && b == 0) {  // -Cs * C + D
        if (d == 1) return key(fc, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_REVERSE_SUBTRACT);
        if (d == 0) return key(fnc, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD);
        return key(VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD);
    }
    if (a == 2 && b == 1) {  // -Cd * C + D
        if (d == 0) return key(VK_BLEND_FACTOR_ONE, fc, VK_BLEND_OP_SUBTRACT);
        if (d == 1) return key(VK_BLEND_FACTOR_ZERO, fnc, VK_BLEND_OP_ADD);
        return key(VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD);
    }
    return 0;
}

VkPipeline HwRaster::pipeline_for(const PipelineKey &key) {
    for (const RecentPipeline &r : recent_)
        if (r.pipeline && r.key == key) return r.pipeline;
    const auto it = draw_.find(key);
    if (it != draw_.end()) {
        recent_[recent_next_++ % 8] = {key, it->second};
        return it->second;
    }
    VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[4] = {
        {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, x)},
        {1, 0, VK_FORMAT_R32_UINT, offsetof(Vertex, z)},
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, r)},
        {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, s)},
    };
    VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vin.vertexBindingDescriptionCount = 1;
    vin.pVertexBindingDescriptions = &binding;
    vin.vertexAttributeDescriptionCount = 4;
    vin.pVertexAttributeDescriptions = attrs;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = key.wmask;
    if (key.blend & 1u) {
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = static_cast<VkBlendFactor>((key.blend >> 1) & 31u);
        cba.dstColorBlendFactor = static_cast<VkBlendFactor>((key.blend >> 6) & 31u);
        cba.colorBlendOp = static_cast<VkBlendOp>((key.blend >> 11) & 7u);
        // The GS never blends alpha: the source alpha is written as is.
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    const VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 3;
    ds.pDynamicStates = dyn;
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = module_;
    stages[0].pName = "hw_vs";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = key.variant < 8 && variant_modules_[key.variant] ? variant_modules_[key.variant] : module_;
    stages[1].pName = key.shader == 2 ? "hw_fs_min" : key.shader ? "hw_fs_cached" : "hw_fs";
    const VkCompareOp ops[4] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_GREATER};
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    // Vulkan drops depth writes whenever the depth test is off, so a GS
    // "always" test that still writes Z keeps the test on with COMPARE_OP_ALWAYS.
    dss.depthTestEnable = key.with_depth ? VK_TRUE : VK_FALSE;
    dss.depthWriteEnable = key.with_depth && key.zwrite ? VK_TRUE : VK_FALSE;
    dss.depthCompareOp = ops[key.ztst & 3u];
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vin;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vp;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &dss;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &ds;
    gpi.layout = layout_;
    gpi.renderPass = key.with_depth ? pass_depth_rp_ : pass_nodepth_;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(ctx_->device, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] cannot build a draw pipeline (blend key %x)\n", key.blend);
        pipeline = VK_NULL_HANDLE;
    }
    draw_[key] = pipeline;
    return pipeline;
}

void HwRaster::begin_frame(int slot) {
    slot_ = slot & 1;
    frame_ = &frames_[slot_];
    query_open_ = false;
    if (occlusion_ && query_used_[slot_]) {
        // This slot's last submission has been waited on: its count is ready.
        std::uint64_t count = 0;
        if (vkGetQueryPoolResults(ctx_->device, occlusion_, static_cast<std::uint32_t>(slot_), 1, sizeof count, &count, sizeof count, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            fragments_ += count;
            ++fragment_frames_;
        }
        query_used_[slot_] = false;
    }
    frame_->vertex_used = frame_->uniform_used = 0;
    ++submits_;
    // Both frame slots have been waited on since these were retired.
    std::size_t kept = 0;
    for (Retired &r : retired_) {
        if (r.at > submits_) {
            retired_[kept++] = r;
            continue;
        }
        if (r.framebuffer) vkDestroyFramebuffer(ctx_->device, r.framebuffer, nullptr);
        if (r.sets[0]) vkFreeDescriptorSets(ctx_->device, pool_, 2, r.sets);
        if (r.set2) vkFreeDescriptorSets(ctx_->device, pool2_, 1, &r.set2);
        if (r.image.image) ctx_->destroy_image(r.image);
    }
    retired_.resize(kept);
}

std::uint32_t HwRaster::push_uniform(FrameScratch &f, const std::uint32_t words[kUniformWords]) {
    if (f.uniform_used + kUniformStride > kUniformBytes) {
        // Out of room: reuse the last slot. Whatever is recorded with it is
        // wrong, which beats writing past the buffer; the counter shows it.
        ++uniform_overflows_;
        f.uniform_used -= kUniformStride;
    }
    const std::uint32_t offset = static_cast<std::uint32_t>(f.uniform_used);
    std::memcpy(static_cast<std::uint8_t *>(f.uniforms.mapped) + offset, words, kUniformWords * 4);
    f.uniform_used += kUniformStride;
    return offset;
}

// Everything the hardware path does not implement yet sends the batch to
// the compute rasterizer: blending, destination alpha, frame masks, frame
// formats other than 32-bit, 32-bit depth, alpha-test fail modes that keep
// writing, lines, points, upscaling, and batches that read their own target.
const char *const kReasons[14] = {"scale", "empty", "snapshot", "blend", "date", "fbmsk", "fpsm", "ztst never", "zpsm", "atest fail mode", "prim kind", "scratch full", "no surface", "no framebuffer"};

void HwRaster::note_deferred(const gpu::Command &cmd) {
    ++deferred_count_;
    for (const gpu::Span &s : cmd.reads)
        if (s.hi > s.lo) deferred_reads_.push_back(s);
    for (const gpu::Span &s : cmd.writes)
        if (s.hi > s.lo) deferred_writes_.push_back(s);
}

namespace {
bool spans_overlap(const gpu::Span &a, const gpu::Span &b) { return a.hi > a.lo && b.hi > b.lo && a.lo < b.hi && b.lo < a.hi; }
}  // namespace

void HwRaster::note_held(const gpu::Command &cmd) {
    ++held_batches_;
    for (const gpu::Span &s : cmd.reads)
        if (s.hi > s.lo) held_reads_.push_back(s);
    for (const gpu::Span &s : cmd.writes)
        if (s.hi > s.lo) held_writes_.push_back(s);
}

// Would recording `cmd` now, ahead of the held batches, change what either
// sees? Its writes against their reads and writes, its reads against their
// writes.
bool HwRaster::held_conflict(const gpu::Command &cmd) const {
    if (held_reads_.empty() && held_writes_.empty()) return false;
    if (!cmd.ranged) return true;
    for (const gpu::Span &w : cmd.writes) {
        for (const gpu::Span &d : held_reads_)
            if (spans_overlap(w, d)) return true;
        for (const gpu::Span &d : held_writes_)
            if (spans_overlap(w, d)) return true;
    }
    for (const gpu::Span &r : cmd.reads)
        for (const gpu::Span &d : held_writes_)
            if (spans_overlap(r, d)) return true;
    return false;
}

HwRaster::Surface *HwRaster::viewable_surface(const gpu::Span &span, const Surface *color, const Surface *depth) {
    if (span.hi <= span.lo) return nullptr;
    for (auto &sp : surfaces_) {
        Surface &s = *sp;
        if (s.depth || !s.image.image || !s.set2 || (s.psm != 0x00u && s.psm != 0x01u) || &s == color || &s == depth) continue;
        const std::uint64_t base = static_cast<std::uint64_t>(s.bp) * 256, end = base + static_cast<std::uint64_t>(s.height / 32) * s.bw * 8192;
        if (span.lo >= base && span.hi <= end) return &s;
    }
    return nullptr;
}

HwRaster::Plan HwRaster::plan(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params) {
    const std::uint32_t *setups = nullptr;
    std::uint32_t zmode = 0;
    if (!supported(cmd, arena, params, setups, zmode)) {
        ++fallback_;
        return Plan::Declined;
    }
    if (!pass_cb_) return Plan::Draw;
    static const unsigned debug = std::getenv("PS2_HW_DEBUG") ? static_cast<unsigned>(std::atoi(std::getenv("PS2_HW_DEBUG"))) : 0u;
    if (debug & 16777216u) return Plan::Draw;  // debug: never hold a batch back
    const bool with_depth = zmode != 0 && !(debug & 8388608u);
    Surface *color = find_surface(setups[1], setups[2], setups[3], false);
    Surface *depth = with_depth ? find_surface(setups[5], setups[2], setups[6], true) : nullptr;
    if (color == pass_color_ && depth == pass_depth_ && with_depth == pass_with_depth_) return Plan::Draw;
    // Other surfaces: hold it unless something held back or drawn must come
    // first (then draw() closes the pass and switches).
    static const unsigned long long trace_frame = std::getenv("PS2_HW_PASS_TRACE") ? std::strtoull(std::getenv("PS2_HW_PASS_TRACE"), nullptr, 10) : ~0ull;
    const bool conflict = !cmd.ranged || deferred_conflict(cmd) || held_conflict(cmd);
    if (frame_count() == trace_frame)
        std::fprintf(stderr, "[pass] f%llu plan: batch for %u/%u/%02x%s while pass on %u/%u/%02x%s: %s\n", static_cast<unsigned long long>(frame_count()), setups[1], setups[2], setups[3], with_depth ? " +z" : "", pass_color_->bp, pass_color_->bw, pass_color_->psm,
                     pass_depth_ ? " +z" : "", conflict ? "draw (conflict)" : "hold");
    if (conflict) return Plan::Draw;
    return Plan::Hold;
}

bool HwRaster::deferred_conflict(const gpu::Command &cmd) const {
    if (deferred_reads_.empty() && deferred_writes_.empty()) return false;
    const auto overlaps = [](const gpu::Span &a, const gpu::Span &b) { return a.hi > a.lo && b.hi > b.lo && a.lo < b.hi && b.lo < a.hi; };
    if (!cmd.ranged) return true;
    for (const gpu::Span &w : cmd.writes) {
        for (const gpu::Span &d : deferred_reads_)
            if (overlaps(w, d)) return true;
        for (const gpu::Span &d : deferred_writes_)
            if (overlaps(w, d)) return true;
    }
    for (const gpu::Span &r : cmd.reads)
        for (const gpu::Span &d : deferred_writes_)
            if (overlaps(r, d)) return true;
    return false;
}

void HwRaster::close_pass(VkCommandBuffer cb, CloseWhy why) {
    if (pass_cb_ != cb) return;  // none open, or open in another command buffer
    ++closes_[why];
    static const unsigned long long trace_frame = std::getenv("PS2_HW_PASS_TRACE") ? std::strtoull(std::getenv("PS2_HW_PASS_TRACE"), nullptr, 10) : ~0ull;  // one frame's pass closes, with reasons
    if (frame_count() == trace_frame) {
        static const char *const names[kCloseCount] = {"import", "write-back", "decode", "copy", "texture", "memory-read", "surfaces", "failed", "end", "device", "decoded", "wb-draw", "wb-decode", "wb-before", "wb-evict", "wb-overlap", "import-own", "import-source", "wb-uncached", "depth-toggle", "color-change", "depth-change"};
        std::fprintf(stderr, "[pass] f%llu close after batch %llu: %s (surface %u/%u/%02x%s, %zu held back)\n", static_cast<unsigned long long>(frame_count()), static_cast<unsigned long long>(drawn_), names[why], pass_color_->bp, pass_color_->bw, pass_color_->psm, pass_depth_ ? " +z" : "", deferred_count_);
    }
    vkCmdEndRenderPass(cb);
    barrier_after_draw(cb, *pass_color_, pass_depth_);
    pass_cb_ = VK_NULL_HANDLE;
    pass_color_ = pass_depth_ = nullptr;
    if (pass_closed_hook_) pass_closed_hook_();
    // Drawing what was held back may have left a pass open; a caller about
    // to record outside a pass needs it closed.
    if (pass_cb_ == cb) {
        vkCmdEndRenderPass(cb);
        barrier_after_draw(cb, *pass_color_, pass_depth_);
        pass_cb_ = VK_NULL_HANDLE;
        pass_color_ = pass_depth_ = nullptr;
    }
}

void HwRaster::end_frame(VkCommandBuffer cb) {
    close_pass(cb, kCloseEndFrame);  // may draw held batches: more vertices
    ctx_->flush(frame_->vertices, 0, frame_->vertex_used);
    ctx_->flush(frame_->uniforms, 0, frame_->uniform_used);
    if (!query_open_) return;
    vkCmdEndQuery(cb, occlusion_, static_cast<std::uint32_t>(slot_));
    query_used_[slot_] = true;
    query_open_ = false;
}

void HwRaster::print_reasons() const {
    static const char *const kCloseNames[kCloseCount] = {"import", "write-back", "decode", "copy", "texture", "memory-read", "surfaces", "failed", "end", "device", "decoded", "wb-draw", "wb-decode", "wb-before", "wb-evict", "wb-overlap", "import-own", "import-source", "wb-uncached", "depth-toggle", "color-change", "depth-change"};
    std::fprintf(stderr, "[vk] render passes: %llu (%.2f per hardware-drawn batch), %llu batches held back for other surfaces; closed by:", static_cast<unsigned long long>(passes_), drawn_ ? static_cast<double>(passes_) / static_cast<double>(drawn_) : 0.0, static_cast<unsigned long long>(held_batches_));
    for (int i = 0; i < kCloseCount; ++i)
        if (closes_[i]) std::fprintf(stderr, " %s %llu", kCloseNames[i], static_cast<unsigned long long>(closes_[i]));
    std::fprintf(stderr, "\n");
    if (fragment_frames_) std::fprintf(stderr, "[vk] fragments: %.2f M per frame reached the colour stage over %llu frames (hardware-drawn batches)\n", static_cast<double>(fragments_) / static_cast<double>(fragment_frames_) / 1e6, static_cast<unsigned long long>(fragment_frames_));
    std::fprintf(stderr, "[vk] surfaces: %zu resident, %.1f Mpx imported, %.1f Mpx written back, %llu evictions, %llu aliased batches, %llu uniform overflows\n", surfaces_.size(), static_cast<double>(imported_px_) / 1e6,
                 static_cast<double>(exported_px_) / 1e6, static_cast<unsigned long long>(evictions_), static_cast<unsigned long long>(alias_batches_), static_cast<unsigned long long>(uniform_overflows_));
    std::fprintf(stderr, "[vk] textures: %zu resident, %llu decoded, %llu hits, %llu sampled from surfaces, %llu copied from own surface, %llu memory reads answered by surface views; block writes by commands %llu, write-backs %llu, own draws %llu; evictions %llu, stale at lookup %llu\n", textures_.size(),
                 static_cast<unsigned long long>(decoded_textures_), static_cast<unsigned long long>(texture_hits_), static_cast<unsigned long long>(surface_samples_), static_cast<unsigned long long>(surface_copies_), static_cast<unsigned long long>(surface_views_), static_cast<unsigned long long>(tex_drops_[0]), static_cast<unsigned long long>(tex_drops_[1]),
                 static_cast<unsigned long long>(tex_drops_[2]), static_cast<unsigned long long>(tex_drops_[3]), static_cast<unsigned long long>(tex_drops_[4]));
    std::fprintf(stderr, "[vk] fallback reasons:");
    for (int i = 0; i < 14; ++i)
        if (reasons_[i]) std::fprintf(stderr, " %s %llu", kReasons[i], static_cast<unsigned long long>(reasons_[i]));
    std::fprintf(stderr, "\n");
}

bool HwRaster::would_draw(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params) const {
    const std::uint32_t *setups = nullptr;
    std::uint32_t zmode = 0;
    const std::uint64_t saved[14] = {reasons_[0], reasons_[1], reasons_[2], reasons_[3], reasons_[4], reasons_[5], reasons_[6], reasons_[7], reasons_[8], reasons_[9], reasons_[10], reasons_[11], reasons_[12], reasons_[13]};
    const bool ok = cmd.pipeline == gpu::kRaster && supported(cmd, arena, params, setups, zmode);
    for (int i = 0; i < 14; ++i) reasons_[i] = saved[i];  // a probe, not a decision
    return ok;
}

bool HwRaster::decodes_unneeded(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params) const {
    if (!would_draw(cmd, arena, params)) return false;
    const std::uint32_t *p = params + cmd.param * (gpu::kParamStride / 4);
    const std::uint32_t *setups = arena + p[2];
    for (std::uint32_t i = 0; i < cmd.setups; ++i) {
        const std::uint32_t *w = setups + i * kSetupWords;
        if ((w[0] & F_TME) && !texture_cacheable(w)) return false;
    }
    return true;
}

gpu::Span HwRaster::texture_span(const std::uint32_t *w) const {
    std::uint32_t pw, ph;
    page_size(w[14], pw, ph);
    const std::uint32_t width = 1u << (w[15] & 0xFFu), height = 1u << (w[15] >> 8);
    const std::uint64_t per_row = std::max<std::uint32_t>(1, w[13] * 64 / pw);
    const std::uint64_t base = static_cast<std::uint64_t>(w[12]) * 256;
    return {static_cast<std::uint32_t>(std::min<std::uint64_t>(base, kMemoryBytes)), static_cast<std::uint32_t>(std::min<std::uint64_t>(base + ((width - 1) / pw + static_cast<std::uint64_t>((height - 1) / ph) * per_row + 1) * 8192, kMemoryBytes))};
}

bool HwRaster::setup_supported(const std::uint32_t *w) const {
    const std::uint32_t flags = w[0];
    if (flags & F_ABE) {
        // Blend when the factors map onto the blend unit, colours clamp, and
        // per-pixel alpha blending (PABE) is off.
        if ((flags & F_PABE) || !(flags & F_COLCLAMP) || !blend_key(w, ctx_->features.dualSrcBlend == VK_TRUE)) return false;
    }
    if (flags & F_DATE) return false;
    std::uint32_t wmask;
    if (!write_mask(w[4], wmask)) return false;
    // 32-bit frames, 24-bit frames, and Z formats used as frames (clears).
    if (w[3] != 0x00u && w[3] != 0x01u && w[3] != 0x30u && w[3] != 0x31u) return false;
    if (flags & F_ZTE) {
        const std::uint32_t ztst = (w[7] >> 8) & 3u;
        if (ztst == 0) return false;
        // 32-bit Z rides in a float depth attachment with 24 bits of mantissa; exact for Z24 and Z16.
        if (w[6] != 0x30u && w[6] != 0x31u && w[6] != 0x32u && w[6] != 0x3Au) return false;
    }
    return true;
}

const char *HwRaster::last_reason_name() const { return kReasons[last_reason_]; }

bool HwRaster::batch_supported(const std::uint32_t *setups, std::uint32_t nsetups, const std::uint32_t *prims, std::uint32_t nprims) const {
    if (scale_ != 1 || !nprims || !nsetups) return false;
    for (std::uint32_t i = 0; i < nsetups; ++i) {
        const std::uint32_t *w = setups + i * kSetupWords;
        if (!setup_supported(w)) return false;
        const std::uint32_t flags = w[0];
        if ((flags & F_ATE) && ((w[7] & 7u) != 1u) && ((w[7] >> 4) & 3u) != 0 && nprims > kMaxFailModePrims) return false;
        if ((flags & F_ABE) && (blend_key(w, ctx_->features.dualSrcBlend == VK_TRUE) & (1u << 17)) && nprims > kMaxFailModePrims) return false;
    }
    for (std::uint32_t i = 0; i < nprims; ++i)
        if (prims[i * kPrimWords + 1] > 1u) return false;
    return true;
}

bool HwRaster::supported(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params, const std::uint32_t *&setups, std::uint32_t &zmode) const {
    const auto no = [&](int reason) {
        ++reasons_[reason];
        last_reason_ = reason;
        return false;
    };
    if (scale_ != 1) return no(0);
    if (!cmd.prims || !cmd.setups) return no(1);
    const std::uint32_t *p = params + cmd.param * (gpu::kParamStride / 4);
    setups = arena + p[2];
    const std::uint32_t *prims = arena + p[1];
    zmode = 0;
    for (std::uint32_t i = 0; i < cmd.setups; ++i) {
        const std::uint32_t *w = setups + i * kSetupWords;
        const std::uint32_t flags = w[0];
        if (flags & F_ABE) {
            if ((flags & F_PABE) || !(flags & F_COLCLAMP) || !blend_key(w, ctx_->features.dualSrcBlend == VK_TRUE)) return no(3);
        }
        if (flags & F_DATE) return no(4);
        std::uint32_t wmask;
        if (!write_mask(w[4], wmask)) return no(5);
        if (w[3] != 0x00u && w[3] != 0x01u && w[3] != 0x30u && w[3] != 0x31u) return no(6);
        if (flags & F_ZTE) {
            const std::uint32_t ztst = (w[7] >> 8) & 3u;
            if (ztst == 0) return no(7);
            if (w[6] != 0x30u && w[6] != 0x31u && w[6] != 0x32u && w[6] != 0x3Au) return no(8);
            // Depth memory is only touched when the test can fail or the
            // result is written; games alias an unused Z buffer with other data.
            if (ztst != 1u || !(flags & F_ZMSK)) zmode |= 1u;
        }
        // Alpha-test fail modes that still write are two draws per
        // primitive (exact), which only pays for small batches.
        if ((flags & F_ATE) && ((w[7] & 7u) != 1u) && ((w[7] >> 4) & 3u) != 0 && cmd.prims > kMaxFailModePrims) return no(9);
        if ((flags & F_ABE) && (blend_key(w, ctx_->features.dualSrcBlend == VK_TRUE) & (1u << 17)) && cmd.prims > kMaxFailModePrims) return no(3);
    }
    for (std::uint32_t i = 0; i < cmd.prims; ++i)
        if (prims[i * kPrimWords + 1] > 1u) return no(10);
    return true;
}

void HwRaster::append_prims(FrameScratch &f, const std::uint32_t *prims, std::uint32_t count, std::vector<std::uint32_t> &first, std::vector<std::uint32_t> &counts) {
    auto *out = reinterpret_cast<Vertex *>(static_cast<std::uint8_t *>(f.vertices.mapped) + f.vertex_used);
    std::uint32_t n = 0;
    const auto color = [](std::uint32_t c, Vertex &v) {
        v.r = static_cast<float>(c & 0xFFu);
        v.g = static_cast<float>((c >> 8) & 0xFFu);
        v.b = static_cast<float>((c >> 16) & 0xFFu);
        v.a = static_cast<float>(c >> 24);
    };
    const auto tex = [](const std::uint32_t *rec, std::uint32_t k, Vertex &v) {
        v.s = fat(rec[15 + k * 3]);
        v.t = fat(rec[16 + k * 3]);
        v.q = fat(rec[17 + k * 3]);
        v.f = static_cast<float>((rec[11] >> (k * 8)) & 0xFFu);
    };
    std::uint32_t run_setup = ~0u;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t *rec = prims + i * kPrimWords;
        if (rec[0] != run_setup) {
            run_setup = rec[0];
            first.push_back(n);
            counts.push_back(0);
        }
        if (rec[1] == 0) {
            const std::uint32_t z0 = rec[24];
            const std::uint32_t z[3] = {z0, z_at(z0, fat(rec[25])), z_at(z0, fat(rec[26]))};
            for (std::uint32_t k = 0; k < 3; ++k) {
                Vertex &v = out[n++];
                v.x = fat(rec[4 + k * 2]);
                v.y = fat(rec[5 + k * 2]);
                v.z = z[k];
                color(rec[12 + k], v);
                tex(rec, k, v);
            }
            counts.back() += 3;
        } else {
            const float x0 = fat(rec[4]), y0 = fat(rec[5]), x1 = x0 + fat(rec[6]), y1 = y0 + fat(rec[7]);
            Vertex a{}, b{};
            tex(rec, 0, a);
            tex(rec, 1, b);
            const std::uint32_t c = rec[13];
            const float xs[4] = {x0, x1, x0, x1}, ys[4] = {y0, y0, y1, y1};
            const float ss[4] = {a.s, b.s, a.s, b.s}, ts[4] = {a.t, a.t, b.t, b.t};
            const int order[6] = {0, 1, 2, 2, 1, 3};
            for (int o : order) {
                Vertex &v = out[n++];
                v.x = xs[o];
                v.y = ys[o];
                v.z = rec[24];
                color(c, v);
                v.s = ss[o];
                v.t = ts[o];
                v.q = b.q;
                v.f = b.f;
            }
            counts.back() += 6;
        }
    }
    f.vertex_used += static_cast<std::uint64_t>(n) * sizeof(Vertex);
}

// ---- Surfaces --------------------------------------------------------------

HwRaster::Surface *HwRaster::find_surface(std::uint32_t bp, std::uint32_t bw, std::uint32_t psm, bool depth) {
    for (auto &s : surfaces_)
        if (s->bp == bp && s->bw == bw && s->psm == psm && s->depth == depth) return s.get();
    return nullptr;
}

bool HwRaster::allocate(Surface &s, std::uint32_t width, std::uint32_t height) {
    const VkImageUsageFlags color_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags depth_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!ctx_->create_image(s.image, width * scale_, height * scale_, s.depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM, s.depth ? depth_usage : color_usage,
                            s.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    s.width = width;
    s.height = height;
    // The image lives in GENERAL layout for its whole life.
    VkCommandBuffer cb = ctx_->begin_once();
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = s.image.image;
    bar.subresourceRange = {static_cast<VkImageAspectFlags>(s.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
    bar.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
    ctx_->end_once(cb);
    s.image.layout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorSetLayout layouts[2] = {set1_layout_, set1_layout_};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool_;
    dai.descriptorSetCount = 2;
    dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(ctx_->device, &dai, s.set1) != VK_SUCCESS) {
        s.set1[0] = s.set1[1] = VK_NULL_HANDLE;
        ctx_->destroy_image(s.image);
        return false;
    }
    for (int i = 0; i < 2; ++i) write_set1(s.set1[i], i, s.depth ? dummy_.view : s.image.view);
    if (!s.depth) {
        VkDescriptorSetAllocateInfo tai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        tai.descriptorPool = pool2_;
        tai.descriptorSetCount = 1;
        tai.pSetLayouts = &set2_layout_;
        if (vkAllocateDescriptorSets(ctx_->device, &tai, &s.set2) == VK_SUCCESS) {
            VkDescriptorImageInfo image{VK_NULL_HANDLE, s.image.view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet ws[2] = {};
            for (int i = 0; i < 2; ++i) {
                ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[i].dstSet = s.set2;
                ws[i].dstBinding = static_cast<std::uint32_t>(i);
                ws[i].descriptorCount = 1;
                ws[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                ws[i].pImageInfo = &image;
            }
            vkUpdateDescriptorSets(ctx_->device, 2, ws, 0, nullptr);
        } else {
            s.set2 = VK_NULL_HANDLE;
        }
    }
    s.plane.reset();
    return true;
}

// Drops the surface's GPU objects once no recorded command buffer can
// reference them any more (two submits from now).
void HwRaster::retire(Surface &s) {
    Retired r;
    r.image = s.image;
    r.sets[0] = s.set1[0];
    r.sets[1] = s.set1[1];
    r.set2 = s.set2;
    r.at = submits_ + 3;
    retired_.push_back(r);
    s.set2 = VK_NULL_HANDLE;
    for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
        if (it->first.first == s.image.view || it->first.second == s.image.view) {
            Retired fb;
            fb.framebuffer = it->second;
            fb.at = submits_ + 3;
            retired_.push_back(fb);
            it = framebuffers_.erase(it);
        } else {
            ++it;
        }
    }
    s.image = Image{};
    s.set1[0] = s.set1[1] = VK_NULL_HANDLE;
    s.width = s.height = 0;
    s.plane.reset();
}

// The resident surface for a buffer, created or grown so `needed` fits.
// Returns null when the image cannot be made (the batch then falls back).
HwRaster::Surface *HwRaster::surface_for(VkCommandBuffer cb, VkDescriptorSet set0, std::uint32_t bp, std::uint32_t bw, std::uint32_t psm, bool depth, const Rect &needed) {
    Surface *s = find_surface(bp, bw, psm, depth);
    if (!s) {
        if (surfaces_.size() >= kMaxSurfaces) {
            // Evict the least recently used surface, written back first.
            std::size_t victim = 0;
            for (std::size_t i = 1; i < surfaces_.size(); ++i)
                if (surfaces_[i]->last_use < surfaces_[victim]->last_use) victim = i;
            Surface &v = *surfaces_[victim];
            WhyScope why(wb_why_, kCloseWbEvict);
            write_back(cb, set0, v, Rect{0, 0, kTargetSize - 1, kTargetSize - 1});
            retire(v);
            surfaces_.erase(surfaces_.begin() + static_cast<std::ptrdiff_t>(victim));
            ++evictions_;
        }
        surfaces_.push_back(std::make_unique<Surface>());
        s = surfaces_.back().get();
        s->bp = bp;
        s->bw = bw;
        s->psm = psm;
        s->depth = depth;
    }
    s->last_use = ++use_counter_;
    // Width is the buffer's own; height starts at 512 lines and grows.
    const std::uint32_t full = std::min<std::uint32_t>(std::max<std::uint32_t>(bw, 1) * 64, kTargetSize);
    std::uint32_t width = std::max(s->width, full), height = std::max<std::uint32_t>(s->height, 512);
    while (width <= needed.x1 && width < kTargetSize) width *= 2;
    while (height <= needed.y1 && height < kTargetSize) height *= 2;
    width = std::min(width, kTargetSize);
    height = std::min(height, kTargetSize);
    if (width != s->width || height != s->height) {
        if (s->image.image) {
            write_back(cb, set0, *s, Rect{0, 0, kTargetSize - 1, kTargetSize - 1});
            retire(*s);
        }
        if (!allocate(*s, width, height)) {
            std::fprintf(stderr, "[hw] f%llu surface %u/%u/%02x%s: no %ux%u image (had %ux%u, needed to x %u y %u)\n", static_cast<unsigned long long>(frame_count()), bp, bw, psm, depth ? " depth" : "", width, height, s->width, s->height, needed.x1, needed.y1);
            retire(*s);
            return nullptr;
        }
    }
    return s;
}

VkFramebuffer HwRaster::framebuffer_for(const Surface &color, const Surface *depth) {
    const auto key = std::make_pair(color.image.view, depth ? depth->image.view : VK_NULL_HANDLE);
    const auto it = framebuffers_.find(key);
    if (it != framebuffers_.end()) return it->second;
    VkImageView views[2] = {color.image.view, depth ? depth->image.view : VK_NULL_HANDLE};
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = depth ? pass_depth_rp_ : pass_nodepth_;
    fci.attachmentCount = depth ? 2 : 1;
    fci.pAttachments = views;
    fci.width = std::min(color.width, depth ? depth->width : color.width) * scale_;
    fci.height = std::min(color.height, depth ? depth->height : color.height) * scale_;
    fci.layers = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    if (vkCreateFramebuffer(ctx_->device, &fci, nullptr, &fb) != VK_SUCCESS) return VK_NULL_HANDLE;
    framebuffers_[key] = fb;
    return fb;
}

// The rows (and, within one page row, the columns) of a surface's buffer
// that a memory span touches, conservatively.
bool HwRaster::span_rect(const Surface &s, const gpu::Span &span, Rect &out) const {
    if (span.hi <= span.lo || !s.width) return false;
    std::uint32_t pw, ph;
    page_size(s.psm, pw, ph);
    const std::uint64_t per_row = std::max<std::uint32_t>(1, s.bw * 64 / pw);
    const std::uint64_t base = static_cast<std::uint64_t>(s.bp) * 256;
    const std::uint64_t end = std::min<std::uint64_t>(base + (s.height + ph - 1) / ph * per_row * 8192, kMemoryBytes);
    if (span.hi <= base || span.lo >= end) return false;
    const std::uint64_t p0 = (std::max<std::uint64_t>(span.lo, base) - base) / 8192, p1 = (std::min<std::uint64_t>(span.hi, end) - 1 - base) / 8192;
    out.y0 = static_cast<std::uint32_t>(p0 / per_row * ph);
    out.y1 = std::min<std::uint32_t>(static_cast<std::uint32_t>((p1 / per_row + 1) * ph) - 1, s.height - 1);
    if (p0 / per_row == p1 / per_row) {
        out.x0 = std::min<std::uint32_t>(static_cast<std::uint32_t>(p0 % per_row * pw), s.width - 1);
        out.x1 = std::min<std::uint32_t>(static_cast<std::uint32_t>((p1 % per_row + 1) * pw) - 1, s.width - 1);
    } else {
        out.x0 = 0;
        out.x1 = s.width - 1;
    }
    return out.x0 <= out.x1 && out.y0 <= out.y1;
}

// The memory a rectangle of a surface occupies (first to last page).
gpu::Span HwRaster::rect_span(const Surface &s, const Rect &r) const {
    std::uint32_t pw, ph;
    page_size(s.psm, pw, ph);
    const std::uint64_t per_row = std::max<std::uint32_t>(1, s.bw * 64 / pw);
    const std::uint64_t base = static_cast<std::uint64_t>(s.bp) * 256;
    const std::uint64_t lo = base + (r.x0 / pw + static_cast<std::uint64_t>(r.y0 / ph) * per_row) * 8192;
    const std::uint64_t hi = base + (r.x1 / pw + static_cast<std::uint64_t>(r.y1 / ph) * per_row + 1) * 8192;
    return {static_cast<std::uint32_t>(std::min<std::uint64_t>(lo, kMemoryBytes)), static_cast<std::uint32_t>(std::min<std::uint64_t>(hi, kMemoryBytes))};
}

// Rectangles (tile-aligned, in pixels) covering the tiles of `within` that
// satisfy `pred`: runs along a tile row, merged with the row below when the
// run repeats, so a fully dirty screen is one rectangle.
template <class Pred> std::vector<HwRaster::Rect> HwRaster::tile_rects(const Plane &p, const Rect &within, Pred pred) const {
    std::vector<Rect> out;
    const std::uint32_t tx0 = within.x0 / kTile, tx1 = within.x1 / kTile, ty0 = within.y0 / kTile, ty1 = within.y1 / kTile;
    std::vector<std::size_t> previous;
    for (std::uint32_t ty = ty0; ty <= ty1; ++ty) {
        std::vector<std::size_t> current;
        std::uint32_t tx = tx0;
        while (tx <= tx1) {
            while (tx <= tx1 && !pred(p, ty * kTiles + tx)) ++tx;
            if (tx > tx1) break;
            const std::uint32_t start = tx;
            while (tx <= tx1 && pred(p, ty * kTiles + tx)) ++tx;
            const Rect r{start * kTile, ty * kTile, tx * kTile - 1, ty * kTile + kTile - 1};
            bool merged = false;
            for (std::size_t i : previous) {
                Rect &q = out[i];
                if (q.x0 == r.x0 && q.x1 == r.x1 && q.y1 + 1 == r.y0) {
                    q.y1 = r.y1;
                    current.push_back(i);
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                out.push_back(r);
                current.push_back(out.size() - 1);
            }
        }
        previous.swap(current);
    }
    return out;
}

void HwRaster::bind(VkCommandBuffer cb, VkPipelineBindPoint point, VkDescriptorSet set0, VkDescriptorSet set1, std::uint32_t param_offset, std::uint32_t uniform) {
    const VkDescriptorSet sets[2] = {set0, set1};
    // Set 0 carries the compute renderer's dynamic parameter block too, so
    // two dynamic offsets go with the two sets.
    const std::uint32_t offsets[2] = {param_offset, uniform};
    vkCmdBindDescriptorSets(cb, point, layout_, 0, 2, sets, 2, offsets);
}

// Moves rectangles between GS memory and a surface. Colour goes through
// the import/export shaders directly; depth through depth_stage and a
// buffer/image copy each way. The rectangles are processed one after the
// other because they share depth_stage. Rectangles are clipped to the image.
void HwRaster::copy_rects(VkCommandBuffer cb, VkDescriptorSet set0, Surface &s, bool import, const std::vector<Rect> &rects) {
    static const unsigned debug = std::getenv("PS2_HW_DEBUG") ? static_cast<unsigned>(std::atoi(std::getenv("PS2_HW_DEBUG"))) : 0u;
    if (debug & 32768u) return;  // measurement only: no copies at all (wrong output)
    FrameScratch &f = *frame_;
    VkMemoryBarrier mem{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    for (const Rect &r0 : rects) {
        if (r0.x0 >= s.width || r0.y0 >= s.height) continue;
        const Rect r{r0.x0, r0.y0, std::min(r0.x1, s.width - 1), std::min(r0.y1, s.height - 1)};
        const std::uint32_t rw = (r.x1 - r.x0 + 1) * scale_, rh = (r.y1 - r.y0 + 1) * scale_;
        std::uint32_t words[kUniformWords] = {s.bp, s.bw, s.psm, 0, r.x0, r.y0, rw, rh, scale_, 0, 0, 0};
        const std::uint32_t uniform = push_uniform(f, words);
        const std::uint32_t gx = (rw + 15) / 16, gy = (rh + 15) / 16;
        const VkBufferImageCopy region{0, rw, rh, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {static_cast<std::int32_t>(r.x0 * scale_), static_cast<std::int32_t>(r.y0 * scale_), 0}, {rw, rh, 1}};
        (import ? imported_px_ : exported_px_) += static_cast<std::uint64_t>(rw) * rh;
        static std::uint64_t trace_from = ~0ull, trace_to = 0;
        static const bool trace = [] {
            const char *spec = std::getenv("PS2_HW_TRACE");
            unsigned long long a = 0, z = ~0ull;
            if (spec && std::sscanf(spec, "%llu,%llu", &a, &z) >= 1) { trace_from = a; trace_to = z; }
            return spec != nullptr;
        }();
        if (trace && frame_count() >= trace_from && frame_count() < trace_to)
            std::fprintf(stderr, "[hw] f%llu %s %s %u/%u/%02x rect %u,%u-%u,%u (%u px)\n", static_cast<unsigned long long>(frame_count()), import ? "import" : "export", s.depth ? "depth" : "frame", s.bp, s.bw, s.psm, r.x0, r.y0, r.x1, r.y1, rw * rh);
        if (!s.depth) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, import ? import_frame_ : export_frame_);
            bind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, set0, s.set1[slot_], 0, uniform);
            vkCmdDispatch(cb, gx, gy, 1);
            continue;
        }
        if (import) {
            // depth_stage's previous user (an export's copy read or import's
            // shader write) is ordered before this write.
            mem.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            mem.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, import_depth_);
            bind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, set0, s.set1[slot_], 0, uniform);
            vkCmdDispatch(cb, gx, gy, 1);
            mem.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mem.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mem, 0, nullptr, 0, nullptr);
            vkCmdCopyBufferToImage(cb, depth_stage_.buffer, s.image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        } else {
            VkImageMemoryBarrier to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_transfer.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            to_transfer.oldLayout = to_transfer.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            to_transfer.srcQueueFamilyIndex = to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.image = s.image.image;
            to_transfer.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            mem.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            mem.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mem, 0, nullptr, 1, &to_transfer);
            vkCmdCopyImageToBuffer(cb, s.image.image, VK_IMAGE_LAYOUT_GENERAL, depth_stage_.buffer, 1, &region);
            mem.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mem.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, export_depth_);
            bind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, set0, s.set1[slot_], 0, uniform);
            vkCmdDispatch(cb, gx, gy, 1);
        }
    }
}

void HwRaster::import_tiles(VkCommandBuffer cb, VkDescriptorSet set0, Surface &s, const Rect &rect) {
    const auto rects = tile_rects(s.plane, rect, [](const Plane &q, std::uint32_t t) { return !q.valid[t]; });
    if (rects.empty()) return;
    close_pass(cb, import_why_);
    copy_rects(cb, set0, s, true, rects);
    const std::uint32_t tx0 = rect.x0 / kTile, tx1 = rect.x1 / kTile, ty0 = rect.y0 / kTile, ty1 = rect.y1 / kTile;
    for (std::uint32_t ty = ty0; ty <= ty1; ++ty)
        for (std::uint32_t tx = tx0; tx <= tx1; ++tx) s.plane.valid[ty * kTiles + tx] = true;
}

// Writes the dirty tiles of `rect` back to memory. Other surfaces that
// alias that memory lose the tiles the write covers.
void HwRaster::write_back(VkCommandBuffer cb, VkDescriptorSet set0, Surface &s, const Rect &rect, bool drop_aliases) {
    if (!s.plane.dirty_count || !s.image.image) return;
    const auto rects = tile_rects(s.plane, rect, [](const Plane &q, std::uint32_t t) { return q.dirty[t]; });
    if (rects.empty()) return;
    close_pass(cb, wb_why_);
    copy_rects(cb, set0, s, false, rects);
    // Other surfaces aliasing the memory just written: their tiles in the
    // (conservative) range no longer match memory. What they still hold
    // dirty cannot be the same words (a later draw there would have
    // written this surface back and dropped it already), so store it
    // before dropping it rather than lose it.
    for (const Rect &r : rects) {
        tex_drop_cause_ = 1;
        textures_written(rect_span(s, r));
        if (!drop_aliases) break;
        const gpu::Span span = rect_span(s, r);
        Rect o;
        for (auto &other : surfaces_)
            if (other.get() != &s && span_rect(*other, span, o)) {
                write_back(cb, set0, *other, o, false);
                invalidate(*other, o);
            }
    }
    const std::uint32_t tx0 = rect.x0 / kTile, tx1 = rect.x1 / kTile, ty0 = rect.y0 / kTile, ty1 = rect.y1 / kTile;
    for (std::uint32_t ty = ty0; ty <= ty1; ++ty)
        for (std::uint32_t tx = tx0; tx <= tx1; ++tx) s.plane.clear_dirty(ty * kTiles + tx);
}

void HwRaster::invalidate(Surface &s, const Rect &rect) {
    const std::uint32_t tx0 = rect.x0 / kTile, tx1 = rect.x1 / kTile, ty0 = rect.y0 / kTile, ty1 = rect.y1 / kTile;
    for (std::uint32_t ty = ty0; ty <= ty1; ++ty)
        for (std::uint32_t tx = tx0; tx <= tx1; ++tx) {
            s.plane.valid[ty * kTiles + tx] = false;
            s.plane.clear_dirty(ty * kTiles + tx);
        }
}

void HwRaster::mark_dirty(Surface &s, const Rect &rect) {
    const std::uint32_t tx0 = rect.x0 / kTile, tx1 = rect.x1 / kTile, ty0 = rect.y0 / kTile, ty1 = rect.y1 / kTile;
    for (std::uint32_t ty = ty0; ty <= ty1; ++ty)
        for (std::uint32_t tx = tx0; tx <= tx1; ++tx) {
            s.plane.valid[ty * kTiles + tx] = true;
            s.plane.set_dirty(ty * kTiles + tx);
        }
}

void HwRaster::memory_read(VkCommandBuffer cb, VkDescriptorSet set0, const gpu::Span &span, const Surface *except) {
    Rect r;
    for (auto &s : surfaces_)
        if (s.get() != except && span_rect(*s, span, r)) write_back(cb, set0, *s, r);
}

void HwRaster::memory_written(VkCommandBuffer cb, VkDescriptorSet set0, const gpu::Span &span, const Surface *except_a, const Surface *except_b) {
    tex_drop_cause_ = 0;
    if (!keep_textures_) textures_written(span);
    Rect r;
    for (auto &s : surfaces_)
        if (s.get() != except_a && s.get() != except_b && span_rect(*s, span, r)) {
            write_back(cb, set0, *s, r);
            invalidate(*s, r);
        }
}

// After imports (compute stores, transfer writes): make the attachments
// ready for the render pass.
void HwRaster::barrier_after_copies(VkCommandBuffer cb, const Surface &color, const Surface *depth) {
    VkImageMemoryBarrier bars[2] = {};
    for (int i = 0; i < (depth ? 2 : 1); ++i) {
        bars[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bars[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        bars[i].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        bars[i].oldLayout = bars[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bars[i].srcQueueFamilyIndex = bars[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bars[i].image = i ? depth->image.image : color.image.image;
        bars[i].subresourceRange = {static_cast<VkImageAspectFlags>(i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr,
                         depth ? 2 : 1, bars);
    // Imports into other surfaces this batch samples (texture_set) are made
    // visible to the fragment stage by the decode barrier in draw().
}

// After a render pass: its attachment writes must be visible to the next
// pass on the same images, to write-backs (compute and transfer reads) and
// to later imports (which overwrite).
void HwRaster::barrier_after_draw(VkCommandBuffer cb, const Surface &color, const Surface *depth) {
    VkImageMemoryBarrier bars[2] = {};
    for (int i = 0; i < (depth ? 2 : 1); ++i) {
        bars[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bars[i].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        bars[i].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        bars[i].oldLayout = bars[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bars[i].srcQueueFamilyIndex = bars[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bars[i].image = i ? depth->image.image : color.image.image;
        bars[i].subresourceRange = {static_cast<VkImageAspectFlags>(i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, depth ? 2 : 1, bars);
}

// After write-backs: the GS memory they stored is read by what follows.
void HwRaster::barrier_copies_visible(VkCommandBuffer cb) {
    VkMemoryBarrier mem{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mem.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mem.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mem, 0, nullptr, 0, nullptr);
}

void HwRaster::before_command(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *params, bool next_is_hw_raster) {
    cpu_timers::Scope timer(cpu_timers::kBefore);
    if (surfaces_.empty()) return;
    // The snapshot before a hardware-drawn batch copies memory the batch
    // will not read (it samples the surface image): no write-back for it.
    if (cmd.pipeline == gpu::kSnapshot && next_is_hw_raster) return;
    const std::uint64_t before = exported_px_;
    if (!cmd.ranged) {
        const Rect all{0, 0, kTargetSize - 1, kTargetSize - 1};
        for (auto &s : surfaces_) {
            write_back(cb, set0, *s, all);
            invalidate(*s, all);
        }
    } else {
        // A compute-drawn batch touches exactly its rectangle of its own
        // frame and depth buffers; the surfaces mirroring those buffers
        // give up just that rectangle. Memory spans cover the rest.
        Surface *own_color = nullptr, *own_depth = nullptr;
        if (cmd.pipeline == gpu::kRaster && cmd.prims && cmd.setups) {
            const std::uint32_t *p = params + cmd.param * (gpu::kParamStride / 4);
            const std::uint32_t *prims = arena + p[1];
            const std::uint32_t *setups = arena + p[2];
            std::uint32_t minx = ~0u, miny = ~0u, maxx = 0, maxy = 0;
            for (std::uint32_t i = 0; i < cmd.prims; ++i) {
                minx = std::min(minx, prims[i * kPrimWords + 2] & 0xFFFFu);
                miny = std::min(miny, prims[i * kPrimWords + 2] >> 16);
                maxx = std::max(maxx, prims[i * kPrimWords + 3] & 0xFFFFu);
                maxy = std::max(maxy, prims[i * kPrimWords + 3] >> 16);
            }
            bool zte = false, zwrite = false;
            for (std::uint32_t i = 0; i < cmd.setups; ++i) {
                const std::uint32_t f = setups[i * kSetupWords];
                zte = zte || (f & F_ZTE);
                zwrite = zwrite || ((f & F_ZTE) && !(f & F_ZMSK));
            }
            if (minx <= maxx && miny <= maxy && maxx < kTargetSize && maxy < kTargetSize) {
                const Rect rect{minx, miny, maxx, maxy};
                own_color = find_surface(setups[1], setups[2], setups[3], false);
                if (own_color) {
                    write_back(cb, set0, *own_color, rect);
                    invalidate(*own_color, rect);
                }
                if (zte) {
                    own_depth = find_surface(setups[5], setups[2], setups[6], true);
                    if (own_depth) {
                        write_back(cb, set0, *own_depth, rect);
                        if (zwrite) invalidate(*own_depth, rect);
                    }
                }
            }
        }
        WhyScope why(wb_why_, kCloseWbBefore);
        for (const gpu::Span &sp : cmd.reads) memory_read(cb, set0, sp);
        keep_textures_ = cmd.pipeline == gpu::kUpload;  // block ids decide for uploads
        for (const gpu::Span &sp : cmd.writes) memory_written(cb, set0, sp, own_color, own_depth);
        keep_textures_ = false;
    }
    if (exported_px_ != before) {
        static const bool trace = std::getenv("PS2_HW_TRACE") != nullptr;
        if (trace) {
            static const char *const names[] = {"upload", "copy_read", "copy_write", "load_clut", "display", "raster", "resolve", "snapshot", "decode"};
            std::fprintf(stderr, "[hw] f%llu   ^ written back for %s%s (reads %x-%x %x-%x writes %x-%x %x-%x)\n", static_cast<unsigned long long>(frame_count()), cmd.pipeline < 9 ? names[cmd.pipeline] : "?", cmd.ranged ? "" : " (unranged)",
                         cmd.reads[0].lo, cmd.reads[0].hi, cmd.reads[1].lo, cmd.reads[1].hi, cmd.writes[0].lo, cmd.writes[0].hi, cmd.writes[1].lo, cmd.writes[1].hi);
        }
        barrier_copies_visible(cb);
    }
}

void HwRaster::flush_sync(VkDescriptorSet set0, const gpu::Span &span) {
    bool hit = false;
    Rect r;
    for (auto &s : surfaces_) hit = hit || (s->plane.dirty_count && span_rect(*s, span, r));
    if (!hit) return;
    VkCommandBuffer cb = ctx_->begin_once();
    memory_read(cb, set0, span);
    VkMemoryBarrier mem{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mem.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mem.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem, 0, nullptr, 0, nullptr);
    ctx_->end_once(cb);
}

void HwRaster::drop_all() {
    for (auto &t : textures_) retire_texture(*t);
    textures_.clear();
    texture_index_.clear();
    for (auto &s : surfaces_) {
        s->plane.reset();
    }
}

// ---- Textures --------------------------------------------------------------

// Cacheable when the decode can cover the whole texture: region modes must
// stay inside it (the compute path's plan_decodes makes the same check).
bool HwRaster::texture_cacheable(const std::uint32_t *w) const {
    const std::uint32_t tw = w[15] & 0xFFu, th = w[15] >> 8;
    const std::uint32_t mu = (w[7] >> 24) & 3u, mv = (w[7] >> 26) & 3u;
    const auto inside = [](std::uint32_t mode, std::uint32_t region, std::uint32_t size) { return mode < 2 || ((region & 0x3FFu) < (1u << size) && (region >> 16) < (1u << size)); };
    // Tiny textures (palette-swapped glyphs and the like, often with a
    // palette rendered per use) cost more to decode than to sample.
    return tw <= 10 && th <= 10 && tw + th >= 12 && inside(mu, w[18], tw) && inside(mv, w[19], th);
}

void HwRaster::retire_texture(Texture &t) {
    Retired r;
    r.image = t.image;
    r.set2 = t.set2;
    r.at = submits_ + 3;
    retired_.push_back(r);
    t.image = Image{};
    t.set2 = VK_NULL_HANDLE;
}

// Memory in `span` changed by something other than an upload: give its
// blocks a fresh id. Textures decoded from them fail their stamp check at
// the next lookup and are decoded again; if an upload restores the same
// content first, their stamps match again and nothing is redone.
void HwRaster::textures_written(const gpu::Span &span) {
    if (span.hi <= span.lo) return;
    const std::uint64_t id = ++write_counter_ | (1ull << 63);
    const std::uint32_t lo = span.lo >> 8, hi = std::min<std::uint32_t>((span.hi + 255) >> 8, 16384);
    for (std::uint32_t b = lo; b < hi; ++b) block_ids_[b] = id;
    ++tex_drops_[tex_drop_cause_];
}

// The texel rectangle a batch's primitives of `setup_index` can sample:
// the UV extent over the vertices (bounded by them, perspective included),
// widened by one for bilinear, then wrapped the way the clamp modes wrap.
bool HwRaster::texture_texels(const std::uint32_t *w, const std::uint32_t *prims, std::uint32_t count, std::uint32_t setup_index, Rect &texels) const {
    const std::uint32_t tw = w[15] & 0xFFu, th = w[15] >> 8;
    const int sw = 1 << tw, sh = 1 << th;
    const bool fst = w[0] & 16u;
    float u0 = 1e30f, v0 = 1e30f, u1 = -1e30f, v1 = -1e30f;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t *rec = prims + i * kPrimWords;
        if (rec[0] != setup_index) continue;
        const std::uint32_t verts = rec[1] == 0 ? 3u : 2u;
        for (std::uint32_t k = 0; k < verts; ++k) {
            const float sc = fat(rec[15 + k * 3]), tc = fat(rec[16 + k * 3]);
            float q = fat(rec[17 + k * 3]);
            float u, v;
            if (fst) {
                u = sc / 16.0f;
                v = tc / 16.0f;
            } else {
                if (q == 0.0f) q = 1.0f;
                u = sc / q * static_cast<float>(sw);
                v = tc / q * static_cast<float>(sh);
            }
            u0 = std::min(u0, u);
            u1 = std::max(u1, u);
            v0 = std::min(v0, v);
            v1 = std::max(v1, v);
        }
    }
    if (u0 > u1 || v0 > v1) return false;
    const float pad = (w[0] & 16384u) ? 1.5f : 0.0f;
    const auto axis = [](float lo, float hi, int size, std::uint32_t mode, std::uint32_t region, int &x0, int &x1) {
        if (!(lo > -1e8f && hi < 1e8f)) {
            x0 = 0;
            x1 = size - 1;
            return;
        }
        x0 = static_cast<int>(std::floor(lo));
        x1 = static_cast<int>(std::floor(hi));
        switch (mode) {
        case 0:  // REPEAT
            if (x1 - x0 + 1 >= size || (x0 & ~(size - 1)) != (x1 & ~(size - 1))) {
                x0 = 0;
                x1 = size - 1;
            } else {
                x0 &= size - 1;
                x1 &= size - 1;
            }
            break;
        case 1:  // CLAMP
            x0 = std::clamp(x0, 0, size - 1);
            x1 = std::clamp(x1, 0, size - 1);
            break;
        case 2:  // REGION_CLAMP
            x0 = std::clamp(x0, static_cast<int>(region & 0x3FFu), std::min(static_cast<int>(region >> 16), size - 1));
            x1 = std::clamp(x1, static_cast<int>(region & 0x3FFu), std::min(static_cast<int>(region >> 16), size - 1));
            break;
        default:  // REGION_REPEAT
            x0 = 0;
            x1 = size - 1;
        }
    };
    int x0, x1, y0, y1;
    axis(u0 - pad, u1 + pad, sw, (w[7] >> 24) & 3u, w[18], x0, x1);
    axis(v0 - pad, v1 + pad, sh, (w[7] >> 26) & 3u, w[19], y0, y1);
    texels = Rect{static_cast<std::uint32_t>(std::max(x0, 0)), static_cast<std::uint32_t>(std::max(y0, 0)), static_cast<std::uint32_t>(std::max(x1, 0)), static_cast<std::uint32_t>(std::max(y1, 0))};
    return texels.x0 <= texels.x1 && texels.y0 <= texels.y1;
}

// Still matches memory where this batch samples it: every block cell under
// the sampled rectangle (and the whole CLUT) carries the id it had when the
// texture was decoded.
bool HwRaster::texture_current(const Texture &t, const Rect *texels) const {
    if (texels && t.cells_x && t.cells_y) {
        const std::uint32_t cx0 = texels->x0 / t.cw, cx1 = std::min(texels->x1 / t.cw, t.cells_x - 1);
        const std::uint32_t cy0 = texels->y0 / t.ch, cy1 = std::min(texels->y1 / t.ch, t.cells_y - 1);
        for (std::uint32_t cy = cy0; cy <= cy1; ++cy) {
            const std::uint32_t *blocks = t.cell_block.data() + static_cast<std::size_t>(cy) * t.cells_x;
            const std::uint64_t *ids = t.cell_id.data() + static_cast<std::size_t>(cy) * t.cells_x;
            for (std::uint32_t cx = cx0; cx <= cx1; ++cx)
                if (ids[cx] != block_ids_[blocks[cx] & 16383u]) return false;
        }
    }
    for (std::size_t i = 0; i < t.clut_blocks.size(); ++i)
        if (t.clut_ids[i] != block_ids_[t.clut_blocks[i] & 16383u]) return false;
    return true;
}

void HwRaster::decode_texture(VkCommandBuffer cb, VkDescriptorSet set0, Texture &t, std::uint32_t setup_offset) {
    // The decode reads memory: surfaces holding newer pixels there store them first.
    const std::uint64_t before = exported_px_;
    WhyScope why(wb_why_, kCloseWbDecode);
    memory_read(cb, set0, gpu::Span{t.lo, t.hi});
    if (t.chi > t.clo) memory_read(cb, set0, gpu::Span{t.clo, t.chi});
    if (exported_px_ != before) barrier_copies_visible(cb);
    for (std::size_t i = 0; i < t.cell_block.size(); ++i) t.cell_id[i] = block_ids_[t.cell_block[i] & 16383u];
    for (std::size_t i = 0; i < t.clut_blocks.size(); ++i) t.clut_ids[i] = block_ids_[t.clut_blocks[i] & 16383u];
    const std::uint32_t width = 1u << t.tw, height = 1u << t.th;
    std::uint32_t words[kUniformWords] = {setup_offset, 0, 0, 0, 0, 0, width, height, scale_, 0, 0, 0};
    const std::uint32_t uniform = push_uniform(*frame_, words);
    close_pass(cb, kCloseDecode);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, decode_tex_);
    bind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, set0, misc_set1_[slot_], 0, uniform);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 2, 1, &t.set2, 0, nullptr);
    vkCmdDispatch(cb, (width + 15) / 16, (height + 15) / 16, 1);
    ++decoded_textures_;
}

VkDescriptorSet HwRaster::surface_texture(VkCommandBuffer cb, VkDescriptorSet set0, const std::uint32_t *w, const std::uint32_t *prims, std::uint32_t count, std::uint32_t setup_index, const Surface *color, const Surface *depth) {
    // A surface in the texture's own layout holds the newest pixels: sample
    // it in place (after importing what memory still owns there), unless the
    // batch draws into it, which Vulkan forbids and the snapshot path serves.
    if (Surface *src = find_surface(w[12], w[13], w[14], false); src && src->image.image && src->set2) {
        Rect texels;
        if (texture_texels(w, prims, count, setup_index, texels) && texels.x1 < src->width && texels.y1 < src->height) {
            WhyScope why(import_why_, kCloseImportSource);
            import_tiles(cb, set0, *src, texels);
            src->last_use = ++use_counter_;
            if (src != color && src != depth) {
                ++surface_samples_;
                return src->set2;
            }
            // The batch draws into this surface: sample a copy of the texels
            // it reads, taken now (the snapshot semantics of the compute path).
            bool created = false;
            if (Texture *t = texture_slot(cb, w, true, created)) {
                copy_surface_texels(cb, set0, *src, *t, texels);
                ++surface_copies_;
                return t->set2;
            }
        }
        static const unsigned long long trace_frame = std::getenv("PS2_HW_PASS_TRACE") ? std::strtoull(std::getenv("PS2_HW_PASS_TRACE"), nullptr, 10) : ~0ull;
        if (frame_count() == trace_frame)
            std::fprintf(stderr, "[tex] surface %u/%u/%02x %ux%u rejected: texels %u,%u-%u,%u tw %u th %u modes %u/%u region %08x/%08x (prims %u, setup %u)\n", w[12], w[13], w[14], src->width, src->height, texels.x0, texels.y0, texels.x1, texels.y1, w[15] & 0xFFu, w[15] >> 8, (w[7] >> 24) & 3u, (w[7] >> 26) & 3u, w[18], w[19], count, setup_index);
    }
    return VK_NULL_HANDLE;
}

VkDescriptorSet HwRaster::texture_set(VkCommandBuffer cb, VkDescriptorSet set0, const std::uint32_t *w, std::uint32_t setup_offset, const std::uint32_t *prims, std::uint32_t count, std::uint32_t setup_index, const Surface *color,
                                      const Surface *depth) {
    if (VkDescriptorSet s = surface_texture(cb, set0, w, prims, count, setup_index, color, depth)) return s;
    const Texture *t = texture_for(cb, set0, w, setup_offset, prims, count, setup_index);
    return t ? t->set2 : VK_NULL_HANDLE;
}

// The cached image of a setup's texture, decoded now if it is not resident
// or no longer matches memory where the batch samples it. Records compute
// work into `cb` (call outside render passes).
// Finds or creates (image, descriptor set, block lists) the cache entry
// for a setup's texture without filling it.
HwRaster::Texture *HwRaster::texture_slot(VkCommandBuffer cb, const std::uint32_t *w, bool from_surface, bool &created) {
    created = false;
    std::array<std::uint32_t, 11> key;
    for (int i = 0; i < 9; ++i) key[i] = w[12 + i];
    key[9] = from_surface ? 0xFFFFFFFFu : w[22] >> 12;  // the CLUT source, not the slot it was loaded into
    key[10] = (w[0] & 32768u) | (from_surface ? 1u : 0u);  // AEM
    if (const auto it = texture_index_.find(key); it != texture_index_.end()) {
        it->second->last_use = ++use_counter_;
        textures_.splice(textures_.begin(), textures_, it->second->self);
        return it->second;
    }
    created = true;
    const std::uint32_t tw = w[15] & 0xFFu, th = w[15] >> 8;
    const std::uint32_t width = 1u << tw, height = 1u << th;
    if (textures_.size() >= kMaxTextures) {
        // Evict the least recently used; its image and set stay for reuse.
        std::unique_ptr<Texture> victim = std::move(textures_.back());
        textures_.pop_back();
        ++tex_drops_[3];
        std::array<std::uint32_t, 11> victim_key;
        std::memcpy(victim_key.data(), victim->key, sizeof victim_key);
        texture_index_.erase(victim_key);
        free_by_size_[(victim->tw << 8) | victim->th].push_back(std::move(victim));
    }
    std::unique_ptr<Texture> t;
    if (const auto fit = free_by_size_.find((tw << 8) | th); fit != free_by_size_.end() && !fit->second.empty()) {
        t = std::move(fit->second.back());
        fit->second.pop_back();
    }
    if (t) {
        // Whatever still samples the old contents in this command buffer
        // finishes before they are overwritten.
        VkImageMemoryBarrier war{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        war.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        war.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        war.oldLayout = war.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        war.srcQueueFamilyIndex = war.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        war.image = t->image.image;
        war.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        close_pass(cb, kCloseTexture);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &war);
        const Image image = t->image;
        const VkDescriptorSet set2 = t->set2;
        *t = Texture{};
        t->image = image;
        t->set2 = set2;
    } else {
        t = std::make_unique<Texture>();
        if (!ctx_->create_image(t->image, width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) return nullptr;
        // New image: its layout transition goes into this command buffer.
        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = t->image.image;
        bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        bar.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        close_pass(cb, kCloseTexture);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
        t->image.layout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool2_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &set2_layout_;
        if (vkAllocateDescriptorSets(ctx_->device, &dai, &t->set2) != VK_SUCCESS) {
            ctx_->destroy_image(t->image);
            return nullptr;
        }
        VkDescriptorImageInfo image{VK_NULL_HANDLE, t->image.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet ws[2] = {};
        for (int i = 0; i < 2; ++i) {
            ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet = t->set2;
            ws[i].dstBinding = static_cast<std::uint32_t>(i);
            ws[i].descriptorCount = 1;
            ws[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            ws[i].pImageInfo = &image;
        }
        vkUpdateDescriptorSets(ctx_->device, 2, ws, 0, nullptr);
    }
    std::memcpy(t->key, key.data(), sizeof t->key);
    t->from_surface = from_surface;
    t->tw = tw;
    t->th = th;
    const std::uint32_t tpsm = w[14];
    if (!from_surface) {
        std::uint32_t pw, ph;
        page_size(tpsm, pw, ph);
        const std::uint64_t per_row = std::max<std::uint32_t>(1, w[13] * 64 / pw);
        const std::uint64_t base = static_cast<std::uint64_t>(w[12]) * 256;
        t->lo = static_cast<std::uint32_t>(std::min<std::uint64_t>(base, kMemoryBytes));
        t->hi = static_cast<std::uint32_t>(std::min<std::uint64_t>(base + ((width - 1) / pw + static_cast<std::uint64_t>((height - 1) / ph) * per_row + 1) * 8192, kMemoryBytes));
        if (tpsm == 0x13u || tpsm == 0x14u || tpsm == 0x1Bu || tpsm == 0x24u || tpsm == 0x2Cu) {
            const std::uint32_t id = w[22] >> 12;
            if (id < clut_spans_.size()) {
                t->clo = clut_spans_[id].lo;
                t->chi = clut_spans_[id].hi;
            }
        }
        block_cell(tpsm, t->cw, t->ch);
        t->cells_x = (width + t->cw - 1) / t->cw;
        t->cells_y = (height + t->ch - 1) / t->ch;
        t->cell_block.resize(static_cast<std::size_t>(t->cells_x) * t->cells_y);
        t->cell_id.assign(t->cell_block.size(), 0);
        for (std::uint32_t cy = 0; cy < t->cells_y; ++cy)
            for (std::uint32_t cx = 0; cx < t->cells_x; ++cx) t->cell_block[static_cast<std::size_t>(cy) * t->cells_x + cx] = pixel_address(tpsm, w[12], w[13], cx * t->cw, cy * t->ch) >> 8;
        for (std::uint32_t b = t->clo >> 8; b < (t->chi + 255) >> 8; ++b) t->clut_blocks.push_back(b);
        t->clut_ids.resize(t->clut_blocks.size());
    }
    t->last_use = ++use_counter_;
    textures_.push_front(std::move(t));
    textures_.front()->self = textures_.begin();
    texture_index_.emplace(key, textures_.front().get());
    return textures_.front().get();
}

// Copies the texels a batch reads from the surface it also draws into.
void HwRaster::copy_surface_texels(VkCommandBuffer cb, VkDescriptorSet set0, Surface &src, Texture &t, const Rect &texels) {
    (void)set0;
    close_pass(cb, kCloseCopy);
    // Earlier sampling of the copy (fragment reads) must finish before it is
    // overwritten; the surface's attachment writes are already visible to
    // transfers (barrier_after_draw), its imports too (copies barrier).
    VkImageMemoryBarrier war{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    war.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    war.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    war.oldLayout = war.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    war.srcQueueFamilyIndex = war.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    war.image = t.image.image;
    war.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkMemoryBarrier imports{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    imports.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imports.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &imports, 0, nullptr, 1, &war);
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffset = {static_cast<std::int32_t>(texels.x0 * scale_), static_cast<std::int32_t>(texels.y0 * scale_), 0};
    region.dstOffset = region.srcOffset;
    region.extent = {(texels.x1 - texels.x0 + 1) * scale_, (texels.y1 - texels.y0 + 1) * scale_, 1};
    vkCmdCopyImage(cb, src.image.image, VK_IMAGE_LAYOUT_GENERAL, t.image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

HwRaster::Texture *HwRaster::texture_for(VkCommandBuffer cb, VkDescriptorSet set0, const std::uint32_t *w, std::uint32_t setup_offset, const std::uint32_t *prims, std::uint32_t count, std::uint32_t setup_index) {
    cpu_timers::Scope timer(cpu_timers::kTextureFor);
    Rect texels;
    bool sampled;
    {
        cpu_timers::Scope t(cpu_timers::kTexels);
        sampled = texture_texels(w, prims, count, setup_index, texels);
    }
    bool fresh = false;
    Texture *slot;
    {
        cpu_timers::Scope t(cpu_timers::kSlot);
        slot = texture_slot(cb, w, false, fresh);
    }
    if (!slot) return nullptr;
    Texture &t = *slot;
    static const unsigned debug = std::getenv("PS2_HW_DEBUG") ? static_cast<unsigned>(std::atoi(std::getenv("PS2_HW_DEBUG"))) : 0u;
    bool current;
    {
        cpu_timers::Scope tc(cpu_timers::kCurrent);
        current = !fresh && ((debug & 131072u) || texture_current(t, sampled ? &texels : nullptr));  // 131072: timing only, never re-decode
    }
    {
        static const unsigned long long trace_frame = std::getenv("PS2_HW_PASS_TRACE") ? std::strtoull(std::getenv("PS2_HW_PASS_TRACE"), nullptr, 10) : ~0ull;
        if (frame_count() == trace_frame || frame_count() + 1 == trace_frame)
            std::fprintf(stderr, "[tex] f%llu %s tbp %u tbw %u psm %02x %ux%u csa %u cpsm %02x clut slot %u source %u texels %u,%u-%u,%u\n", static_cast<unsigned long long>(frame_count()), current ? "hit" : fresh ? "new" : "stale", w[12], w[13], w[14], 1u << t.tw, 1u << t.th, w[17], w[16], w[22] & 0xFFFu, w[22] >> 12,
                         texels.x0, texels.y0, texels.x1, texels.y1);
    }
    if (current) {
        ++texture_hits_;
        return &t;
    }
    if (!fresh) ++tex_drops_[4];
    static int shown = 0;
    if (std::getenv("PS2_TEX_TRACE") && shown < 4000 && frame_count() == 5404) {
        ++shown;
        std::size_t stale = 0;
        for (std::size_t i = 0; i < t.cell_block.size(); ++i) stale += t.cell_id[i] != block_ids_[t.cell_block[i] & 16383u];
        std::size_t stale_clut = 0;
        for (std::size_t i = 0; i < t.clut_blocks.size(); ++i) stale_clut += t.clut_ids[i] != block_ids_[t.clut_blocks[i] & 16383u];
        std::fprintf(stderr, "[tex] f%llu %s tbp %u tbw %u psm %02x %ux%u clut %u: texels %u,%u-%u,%u, %zu of %zu cells stale, %zu of %zu clut blocks stale\n", static_cast<unsigned long long>(frame_count()), fresh ? "new" : "stale", w[12], w[13], w[14], 1u << t.tw, 1u << t.th, w[22] >> 12,
                     texels.x0, texels.y0, texels.x1, texels.y1, stale, t.cell_block.size(), stale_clut, t.clut_blocks.size());
    }
    {
        cpu_timers::Scope td(cpu_timers::kDecode);
        decode_texture(cb, set0, t, setup_offset);
    }
    return &t;
}

// ---- Drawing ---------------------------------------------------------------

bool HwRaster::draw(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *params, VkBuffer) {
    cpu_timers::Scope timer(cpu_timers::kDraw);
    // PS2_HW_DEBUG bits, for isolating driver trouble: 1 skip batches that
    // need depth, 8 paint every fragment magenta, 16 paint the raw texel,
    // 256 skip blended, 512 skip textured, 1024 skip triangles, 2048 print
    // the decision, 4096 skip fogged, 8192 write back and drop the target
    // after every batch (the pre-ownership behaviour, for A/B runs).
    static const unsigned debug = std::getenv("PS2_HW_DEBUG") ? static_cast<unsigned>(std::atoi(std::getenv("PS2_HW_DEBUG"))) : 0u;
    const std::uint32_t *setups = nullptr;
    std::uint32_t zmode = 0;
    bool blended = false, textured = false, fogged = false;
    const bool ok = supported(cmd, arena, params, setups, zmode);
    if (ok)
        for (std::uint32_t i = 0; i < cmd.setups; ++i) {
            blended = blended || (setups[i * kSetupWords] & F_ABE);
            textured = textured || (setups[i * kSetupWords] & F_TME);
            fogged = fogged || (setups[i * kSetupWords] & 4u);
        }
    bool triangles = false;
    if (ok)
        for (std::uint32_t i = 0; i < cmd.prims; ++i) triangles = triangles || (arena + (params + cmd.param * (gpu::kParamStride / 4))[1])[i * kPrimWords + 1] == 0;
    if (debug & 2048u) {
        std::fprintf(stderr, "[hw] decide f%llu prims %u setups %u debug %u blended %d textured %d triangles %d zmode %u ok %d flags", static_cast<unsigned long long>(frame_count()), cmd.prims, cmd.setups, debug, blended, textured,
                     triangles, zmode, ok);
        for (std::uint32_t i = 0; ok && i < cmd.setups; ++i) std::fprintf(stderr, " %04x", setups[i * kSetupWords]);
        std::fprintf(stderr, "\n");
    }
    if (!ok || ((debug & 1u) && zmode) || ((debug & 256u) && blended) || ((debug & 512u) && textured) || ((debug & 1024u) && triangles) || ((debug & 4096u) && fogged)) {
        static const bool trace_fallbacks = std::getenv("PS2_HW_TRACE") != nullptr;
        if (trace_fallbacks && !ok) {
            const std::uint32_t *pp = params + cmd.param * (gpu::kParamStride / 4);
            const std::uint32_t *st = arena + pp[2];
            const std::uint32_t *pr = arena + pp[1];
            std::uint32_t minx = ~0u, miny = ~0u, maxx = 0, maxy = 0;
            for (std::uint32_t i = 0; i < cmd.prims; ++i) {
                minx = std::min(minx, pr[i * kPrimWords + 2] & 0xFFFFu);
                miny = std::min(miny, pr[i * kPrimWords + 2] >> 16);
                maxx = std::max(maxx, pr[i * kPrimWords + 3] & 0xFFFFu);
                maxy = std::max(maxy, pr[i * kPrimWords + 3] >> 16);
            }
            std::fprintf(stderr, "[hw] f%llu fallback %s prims %u setups %u rect %u,%u-%u,%u fb %u/%u/%02x", static_cast<unsigned long long>(frame_count()), kReasons[last_reason_], cmd.prims, cmd.setups, minx, miny, maxx, maxy, st[1], st[2], st[3]);
            for (std::uint32_t i = 0; i < std::min(cmd.setups, 4u); ++i) std::fprintf(stderr, " | flags %04x test %08x fbmsk %08x fix %u", st[i * kSetupWords], st[i * kSetupWords + 7], st[i * kSetupWords + 4], st[i * kSetupWords + 9]);
            std::fprintf(stderr, "\n");
        }
        ++fallback_;
        return false;
    }
    // PS2_HW_ONLY=frame,count bisects a frame: the hardware path draws only
    // the first `count` accepted batches of that frame, nothing elsewhere.
    static unsigned long long only_frame = ~0ull, only_count = 0;
    static const bool only = [] {
        const char *spec = std::getenv("PS2_HW_ONLY");
        return spec && std::sscanf(spec, "%llu,%llu", &only_frame, &only_count) == 2;
    }();
    if (only) {
        static unsigned long long seen_frame = ~0ull, seen = 0;
        if (frame_count() != only_frame) { ++fallback_; return false; }
        if (seen_frame != only_frame) { seen_frame = only_frame; seen = 0; }
        if (seen++ >= only_count) { ++fallback_; return false; }
    }
    FrameScratch &f = *frame_;
    const std::uint64_t need = static_cast<std::uint64_t>(cmd.prims) * 6 * sizeof(Vertex);
    if (f.vertex_used + need > kVertexBytes || f.uniform_used + (2 * cmd.setups + 2 * std::min(cmd.prims, kMaxFailModePrims) + 64) * kUniformStride > kUniformBytes) {
        ++reasons_[11];
        last_reason_ = 11;
        ++fallback_;
        return false;
    }
    // Commands held back while the pass is open must land before a batch
    // that touches their memory (see note_deferred).
    if (pass_cb_ == cb && (deferred_conflict(cmd) || held_conflict(cmd))) close_pass(cb, kCloseMemoryRead);
    const std::uint32_t *p = params + cmd.param * (gpu::kParamStride / 4);
    const std::uint32_t *prims = arena + p[1];
    const std::uint32_t setup_offset = p[2];
    // The batch rectangle, from the primitive boxes.
    std::uint32_t minx = ~0u, miny = ~0u, maxx = 0, maxy = 0;
    for (std::uint32_t i = 0; i < cmd.prims; ++i) {
        const std::uint32_t lo = prims[i * kPrimWords + 2], hi = prims[i * kPrimWords + 3];
        minx = std::min(minx, lo & 0xFFFFu);
        miny = std::min(miny, lo >> 16);
        maxx = std::max(maxx, hi & 0xFFFFu);
        maxy = std::max(maxy, hi >> 16);
    }
    if (maxx >= kTargetSize || maxy >= kTargetSize || minx > maxx || miny > maxy) {
        ++fallback_;
        return false;
    }
    const Rect rect{minx, miny, maxx, maxy};
    // PS2_HW_TRACE=from,to prints the batches drawn in that frame window.
    static std::uint64_t trace_from = ~0ull, trace_to = 0;
    static const bool trace = [] {
        const char *spec = std::getenv("PS2_HW_TRACE");
        if (!spec) return false;
        unsigned long long a = 0, z = ~0ull;
        if (std::sscanf(spec, "%llu,%llu", &a, &z) >= 1) { trace_from = a; trace_to = z; }
        return true;
    }();
    if (trace && frame_count() >= trace_from && frame_count() < trace_to) {
        const std::uint32_t *w = setups + prims[0] * kSetupWords;
        std::fprintf(stderr, "[hw] f%llu batch prims %u setups %u rect %u,%u-%u,%u flags %04x fpsm %02x fbmsk %08x zpsm %02x ztst %u zmsk %u z0 %08x kind %u scissor %u,%u-%u,%u xy %.1f,%.1f %.1fx%.1f c0 %08x\n", static_cast<unsigned long long>(frame_count()), cmd.prims, cmd.setups,
                     minx, miny, maxx, maxy, w[0], w[3], w[4], w[6], (w[7] >> 8) & 3u, (w[0] & F_ZMSK) ? 1u : 0u, prims[24], prims[1],
                     w[10] & 0xFFFFu, w[11] & 0xFFFFu, w[10] >> 16, w[11] >> 16, fat(prims[4]), fat(prims[5]), fat(prims[6]), fat(prims[7]), prims[12]);
        if (w[0] & F_TME) std::fprintf(stderr, "[hw]   tex0 %.1f,%.1f q %.3f tex1 %.1f,%.1f q %.3f color %08x tw %u th %u fst %u\n", fat(prims[15]), fat(prims[16]), fat(prims[17]), fat(prims[18]), fat(prims[19]), fat(prims[20]), prims[13], w[15] & 0xFFu, w[15] >> 8, (w[0] & 16u) ? 1u : 0u);
        for (std::uint32_t i = 0; i < cmd.setups; ++i) {
            const std::uint32_t *u = setups + i * kSetupWords;
            std::fprintf(stderr, "[hw]   setup %u flags %04x fb %u/%u/%02x fbmsk %08x zb %u/%02x test %08x aref %u fix %u scissor %u,%u-%u,%u\n", i, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10] & 0xFFFFu, u[11] & 0xFFFFu, u[10] >> 16, u[11] >> 16);
        }
        for (std::uint32_t i = 0; i < std::min(cmd.prims, 16u); ++i) {
            const std::uint32_t *r = prims + i * kPrimWords;
            std::fprintf(stderr, "[hw]   prim %u setup %u kind %u box %u,%u-%u,%u xy %.1f,%.1f %.1fx%.1f z %08x c %08x %08x\n", i, r[0], r[1], r[2] & 0xFFFFu, r[2] >> 16, r[3] & 0xFFFFu, r[3] >> 16, fat(r[4]), fat(r[5]), fat(r[6]), fat(r[7]), r[24], r[12], r[13]);
        }
    }
    const bool with_depth = zmode != 0 && !(debug & 8388608u);  // debug: measure without a depth attachment
    bool zwrite_any = false, color_any = false;
    for (std::uint32_t i = 0; i < cmd.setups; ++i) {
        const std::uint32_t *w = setups + i * kSetupWords;
        zwrite_any = zwrite_any || ((w[0] & F_ZTE) && !(w[0] & F_ZMSK));
        color_any = color_any || w[4] != 0xFFFFFFFFu;
    }

    // 1. Surfaces: the colour buffer (and depth buffer) this batch draws
    // into; write back what the batch samples from any surface; drop what
    // its writes alias in other surfaces; import the tiles it touches that
    // memory still owns.
    const std::uint32_t *first = setups;
    Surface *color = surface_for(cb, set0, first[1], first[2], first[3], false, rect);
    if (!color) {
        ++reasons_[12];
        last_reason_ = 12;
        ++fallback_;
        return false;
    }
    Surface *depth = nullptr;
    if (with_depth) {
        depth = surface_for(cb, set0, first[5], first[2], first[6], true, rect);
        if (!depth) {
            ++reasons_[12];
            last_reason_ = 12;
            ++fallback_;
            return false;
        }
        color = find_surface(first[1], first[2], first[3], false);  // the vector may have shifted
    }
    // Textures sampled from memory by the fragment shader (not cached, not a
    // surface) need the memory current; cached ones write back in their
    // decode, surfaces need nothing. Depth tests read the depth surface.
    const std::uint64_t exported_before = exported_px_;
    std::vector<VkDescriptorSet> textures(cmd.setups, VK_NULL_HANDLE);
    std::vector<View> views(cmd.setups);
    bool decoded_any = false, memory_sampled = false;
    // Textures the cache does not take (tiny, or region modes outside the
    // texture): a surface in the same layout is sampled in place; otherwise
    // the fragment shader reads memory, which must be current.
    for (std::uint32_t i = 0; i < cmd.setups; ++i) {
        const std::uint32_t *w = setups + i * kSetupWords;
        if (!(w[0] & F_TME) || (!(debug & 65536u) && texture_cacheable(w))) continue;
        if (!(debug & 65536u)) {
            const std::uint64_t before_imports = imported_px_, before_copies = surface_copies_;
            textures[i] = surface_texture(cb, set0, w, prims, cmd.prims, i, color, depth);
            decoded_any = decoded_any || imported_px_ != before_imports || surface_copies_ != before_copies;
        }
        if (!textures[i]) {
            // Memory inside a resident 32/24-bit surface: the fragment shader
            // reads that surface's image through its layout (surface view),
            // so nothing is written back.
            const gpu::Span span = texture_span(w);
            Surface *v = (debug & 33554432u) ? nullptr : viewable_surface(span, color, depth);  // debug: no surface views
            Rect vr;
            if (v && span_rect(*v, span, vr)) {
                if (pass_cb_ == cb && (v == pass_color_ || v == pass_depth_)) close_pass(cb, kCloseSurfaces);  // its attachment writes land first
                const std::uint64_t before_imports = imported_px_;
                WhyScope why(import_why_, kCloseImportSource);
                import_tiles(cb, set0, *v, vr);  // what memory still owns there
                decoded_any = decoded_any || imported_px_ != before_imports;
                views[i] = View{v->bp * 64u, v->bw, static_cast<std::uint32_t>(v->height / 32) * v->bw * 2048u, 0u};
                textures[i] = v->set2;
                memory_sampled = true;  // the shader still reads the CLUT buffer
                ++surface_views_;
                continue;
            }
            memory_sampled = true;
            static const unsigned long long trace_frame = std::getenv("PS2_HW_PASS_TRACE") ? std::strtoull(std::getenv("PS2_HW_PASS_TRACE"), nullptr, 10) : ~0ull;
            if (frame_count() == trace_frame)
                std::fprintf(stderr, "[tex] memory-sampled: tbp %u tbw %u tpsm %02x tw %u th %u modes %u/%u region %08x/%08x surface %s\n", w[12], w[13], w[14], w[15] & 0xFFu, w[15] >> 8, (w[7] >> 24) & 3u, (w[7] >> 26) & 3u, w[18], w[19],
                             find_surface(w[12], w[13], w[14], false) ? "same layout" : find_surface(w[12], w[13], w[14] ^ 1u, false) ? "24/32 sibling" : "none");
            WhyScope why(wb_why_, kCloseWbUncached);
            memory_read(cb, set0, texture_span(w));
        }
    }
    // Memory sampling also reads the CLUT buffer, which held-back CLUT loads
    // have not written yet: they land first.
    if (memory_sampled && deferred_count_) close_pass(cb, kCloseMemoryRead);
    {
        WhyScope why(wb_why_, kCloseWbDraw);
        for (const gpu::Span &sp : cmd.writes) memory_written(cb, set0, sp, color, depth);
    }
    // The imports below read memory the write-backs above may have stored.
    if (exported_px_ != exported_before) {
        if (trace && frame_count() >= trace_from && frame_count() < trace_to)
            std::fprintf(stderr, "[hw] f%llu   ^ written back for this batch (reads %x-%x %x-%x writes %x-%x %x-%x)\n", static_cast<unsigned long long>(frame_count()), cmd.reads[0].lo, cmd.reads[0].hi, cmd.reads[1].lo, cmd.reads[1].hi, cmd.writes[0].lo,
                         cmd.writes[0].hi, cmd.writes[1].lo, cmd.writes[1].hi);
        barrier_copies_visible(cb);
    }
    // Every tile memory still owns is imported, not only the batch rectangle:
    // the scene fills the screen batch by batch, and an import per batch
    // closed the render pass every time (19 in a row in a race frame).
    WhyScope why_import(import_why_, kCloseImportOwn);
    if (color_any) import_tiles(cb, set0, *color, Rect{0, 0, color->width - 1, color->height - 1});
    if (depth) import_tiles(cb, set0, *depth, Rect{0, 0, depth->width - 1, depth->height - 1});
    // Textures: resolve (and decode, which reads memory the write-backs
    // above just stored) every textured setup's image before the pass.
    if (!(debug & 65536u))
        for (std::uint32_t i = 0; i < cmd.setups; ++i) {
            const std::uint32_t *w = setups + i * kSetupWords;
            if (!(w[0] & F_TME) || !texture_cacheable(w)) continue;
            const std::uint64_t before_decodes = decoded_textures_, before_imports = imported_px_, before_copies = surface_copies_;
            textures[i] = texture_set(cb, set0, w, setup_offset + i * kSetupWords, prims, cmd.prims, i, color, depth);
            decoded_any = decoded_any || decoded_textures_ != before_decodes || imported_px_ != before_imports || surface_copies_ != before_copies;
        }
    if (decoded_any) {
        // Decodes and imports (compute stores, transfer writes) feed sampling.
        close_pass(cb, kCloseDecoded);
        VkMemoryBarrier mem{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mem.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mem.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &mem, 0, nullptr, 0, nullptr);
    }
    const bool same_pass = pass_cb_ == cb && pass_color_ == color && pass_depth_ == depth && pass_with_depth_ == with_depth;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    if (!same_pass) {
        close_pass(cb, pass_color_ != color ? kCloseColorChange : pass_depth_ != depth && depth && pass_depth_ ? kCloseDepthChange : pass_with_depth_ != with_depth ? kCloseDepthToggle : kCloseSurfaces);
        barrier_after_copies(cb, *color, depth);
        framebuffer = framebuffer_for(*color, depth);
        if (!framebuffer) {
            ++reasons_[13];
            last_reason_ = 13;
            ++fallback_;
            return false;
        }
    }

    // 2. Render pass: one draw per run of primitives sharing a setup.
    if (!same_pass) {
        // The render area is the whole surface (it costs nothing on this
        // hardware, measured) so later batches of the pass fit anywhere in it.
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = with_depth ? pass_depth_rp_ : pass_nodepth_;
        rbi.framebuffer = framebuffer;
        rbi.renderArea = {{0, 0}, {color->width * scale_, color->height * scale_}};
        if (occlusion_ && !query_open_) {
            vkCmdResetQueryPool(cb, occlusion_, static_cast<std::uint32_t>(slot_), 1);
            vkCmdBeginQuery(cb, occlusion_, static_cast<std::uint32_t>(slot_), 0);
            query_open_ = true;
        }
        vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        pass_cb_ = cb;
        pass_color_ = color;
        pass_depth_ = depth;
        pass_with_depth_ = with_depth;
        pass_area_ = rbi.renderArea;
        ++passes_;
    }
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};  // only its renderArea is read below
    rbi.renderArea = pass_area_;
    // The viewport is the whole GS drawing space, not the surface: clipping
    // primitives at the surface edge would re-interpolate their attributes
    // and shift texel picks against the compute path. The render area and
    // scissor keep fragments inside the surface.
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(kTargetSize * scale_), static_cast<float>(kTargetSize * scale_), 0.0f, 1.0f};
    vkCmdSetViewport(cb, 0, 1, &viewport);
    const VkRect2D scissor = rbi.renderArea;
    vkCmdSetScissor(cb, 0, 1, &scissor);
    std::vector<std::uint32_t> firsts, counts;
    const VkDeviceSize prim_base = f.vertex_used;
    append_prims(f, prims, cmd.prims, firsts, counts);
    vkCmdBindVertexBuffers(cb, 0, 1, &f.vertices.buffer, &prim_base);
    const std::uint32_t param_offset = cmd.param * gpu::kParamStride;
    std::uint32_t run = 0;
    for (std::uint32_t i = 0; i < cmd.prims;) {
        const std::uint32_t si = prims[i * kPrimWords];
        const std::uint32_t *w = setups + si * kSetupWords;
        const bool zte = w[0] & F_ZTE;
        const std::uint32_t ztst = zte ? (w[7] >> 8) & 3u : 1u;
        const bool zwrite = zte && !(w[0] & F_ZMSK);
        const std::uint32_t blend = (w[0] & F_ABE) ? blend_key(w, ctx_->features.dualSrcBlend == VK_TRUE) : 0u;
        if (blend & (1u << 14)) {
            const float fix = static_cast<float>(w[9]) / 128.0f;
            const float constants[4] = {fix, fix, fix, fix};
            vkCmdSetBlendConstants(cb, constants);
        }
        std::uint32_t wmask = 0;
        write_mask(w[4], wmask);
        const VkDescriptorSet texture = textures[si];
        if (texture) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 2, 1, &texture, 0, nullptr);
        const std::uint32_t base_flags = (with_depth ? 1u : 0u) | ((debug & 8u) ? 2u : 0u) | ((debug & 16u) ? 4u : 0u) | ((blend >> 15) & 1u) << 4 | ((blend >> 16) & 1u) << 5 | ((debug & 16384u) ? 128u : 0u) | ((texture && views[si].words == 0) ? 256u : 0u) | ((debug & 524288u) ? 512u : 0u);  // a surface view keeps the memory-sampling shader
        // The GS scissor (inclusive, GS pixels) as the Vulkan scissor, within
        // the render area; an empty one skips the draws.
        bool skip_draw = false;
        {
            const std::int32_t sx0 = static_cast<std::int32_t>((w[10] & 0xFFFFu) * scale_), sx1 = static_cast<std::int32_t>(((w[10] >> 16) + 1) * scale_);
            const std::int32_t sy0 = static_cast<std::int32_t>((w[11] & 0xFFFFu) * scale_), sy1 = static_cast<std::int32_t>(((w[11] >> 16) + 1) * scale_);
            const std::int32_t ax0 = rbi.renderArea.offset.x, ay0 = rbi.renderArea.offset.y;
            const std::int32_t ax1 = ax0 + static_cast<std::int32_t>(rbi.renderArea.extent.width), ay1 = ay0 + static_cast<std::int32_t>(rbi.renderArea.extent.height);
            const std::int32_t x0 = std::max(sx0, ax0), y0 = std::max(sy0, ay0), x1 = std::min(sx1, ax1), y1 = std::min(sy1, ay1);
            if (x0 >= x1 || y0 >= y1) {
                skip_draw = true;
            } else {
                const VkRect2D sc{{x0, y0}, {static_cast<std::uint32_t>(x1 - x0), static_cast<std::uint32_t>(y1 - y0)}};
                vkCmdSetScissor(cb, 0, 1, &sc);
            }
        }
        // Specialised fragment shader: textured (cached image), may discard
        // (alpha test that can fail, or a scan mask), fogged. Setups that
        // sample memory keep the generic shader.
        const bool cached_fs = (base_flags & 256u) && !(debug & 262144u);
        const bool may_discard = ((w[0] & F_ATE) && ((w[7] & 7u) != 1u)) || (((w[7] >> 28) & 3u) != 0u);
        const std::uint32_t variant = (!(w[0] & F_TME) || cached_fs) ? ((cached_fs ? 1u : 0u) | (may_discard ? 2u : 0u) | ((w[0] & 4u) ? 4u : 0u)) : 0xFFu;
        const auto bind_draw = [&](std::uint32_t flags, bool write_z, std::uint32_t mask) -> bool {
            std::uint32_t words[kUniformWords] = {setup_offset + si * kSetupWords, 0, 0, flags, 0, 0, kTargetSize * scale_, kTargetSize * scale_, scale_, p[5], p[6], inverse32_words_, views[si].base, views[si].bw, views[si].words, views[si].shift};
            std::memcpy(words + 20, w, kSetupWords * 4);
            const std::uint32_t uniform = push_uniform(f, words);
            VkPipeline pipeline = (debug & 4194304u) ? pipeline_for({ztst, write_z ? 1u : 0u, with_depth ? 1u : 0u, 0u, 0xFu, 2u, 0xFFu})  // debug: minimal fragment shader, single output, no blend
                                                   : pipeline_for({ztst, write_z ? 1u : 0u, with_depth ? 1u : 0u, blend & 0x7FFFu, mask, variant < 8 ? 1u : cached_fs ? 1u : 0u, variant});
            if (!pipeline) return false;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            bind(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, set0, color->set1[slot_], param_offset, uniform);
            return true;
        };
        std::uint32_t j = i + 1;
        while (j < cmd.prims && prims[j * kPrimWords] == si) ++j;
        const std::uint32_t afail = ((w[0] & F_ATE) && ((w[7] & 7u) != 1u)) ? (w[7] >> 4) & 3u : 0u;
        bool built = true;
        if (blend & (1u << 17)) {
            // Destination-alpha blend in two additive steps per primitive:
            // the first keeps the alpha channel (the factor), the second
            // adds the remainder and writes alpha and depth.
            std::uint32_t first = firsts[run];
            for (std::uint32_t k = i; k < j && built; ++k) {
                const std::uint32_t verts = prims[k * kPrimWords + 1] == 0 ? 3u : 6u;
                built = bind_draw(base_flags, false, wmask & ~static_cast<std::uint32_t>(VK_COLOR_COMPONENT_A_BIT));
                if (built && !skip_draw) vkCmdDraw(cb, verts, 1, first, 0);
                if (built) built = bind_draw(base_flags | 64u, zwrite, wmask);
                if (built && !skip_draw) vkCmdDraw(cb, verts, 1, first, 0);
                first += verts;
            }
        } else if (afail == 0) {
            built = bind_draw(base_flags, zwrite, wmask);
            if (built && !skip_draw) vkCmdDraw(cb, counts[run], 1, firsts[run], 0);
        } else {
            // Per primitive: the passing fragments with full writes, then
            // the failing ones with what the mode keeps: 1 colour only,
            // 2 depth only, 3 RGB only (no alpha, no depth).
            const bool fail_z = afail == 2 && zwrite;
            const std::uint32_t fail_mask = afail == 2 ? 0u : afail == 3 ? (wmask & ~static_cast<std::uint32_t>(VK_COLOR_COMPONENT_A_BIT)) : wmask;
            std::uint32_t first = firsts[run];
            for (std::uint32_t k = i; k < j && built; ++k) {
                const std::uint32_t verts = prims[k * kPrimWords + 1] == 0 ? 3u : 6u;
                built = bind_draw(base_flags, zwrite, wmask);
                if (built && !skip_draw) vkCmdDraw(cb, verts, 1, first, 0);
                if (built) built = bind_draw(base_flags | 8u, fail_z, fail_mask);
                if (built && !skip_draw) vkCmdDraw(cb, verts, 1, first, 0);
                first += verts;
            }
        }
        if (!built) {
            // Pipeline creation failed: the target is now unreliable in the
            // rectangle; drop it and leave the batch to the compute path.
            close_pass(cb, kCloseFailed);
            invalidate(*color, rect);
            if (depth) invalidate(*depth, rect);
            ++fallback_;
            return false;
        }
        i = j;
        ++run;
    }
    // The pass stays open for the next batch on these surfaces.

    // 3. Bookkeeping: the rectangle is now newer than memory, and any
    // texture decoded from it no longer matches.
    if (color_any) mark_dirty(*color, rect);
    if (depth && zwrite_any) mark_dirty(*depth, rect);
    tex_drop_cause_ = 2;
    if (color_any) textures_written(rect_span(*color, rect));
    if (depth && zwrite_any) textures_written(rect_span(*depth, rect));
    if (depth && zwrite_any && color_any) {
        // Frame and depth buffers that overlap in memory (a clear trick):
        // the GS writes the colour word, then Z, so Z wins where they meet.
        // Store both now in that order and drop what each overwrote in the
        // other, since neither surface alone holds the memory's contents.
        const gpu::Span cs = rect_span(*color, rect), zs = rect_span(*depth, rect);
        if (cs.lo < zs.hi && zs.lo < cs.hi) {
            WhyScope why(wb_why_, kCloseWbOverlap);
            write_back(cb, set0, *color, rect, false);
            write_back(cb, set0, *depth, rect, false);
            Rect o;
            if (span_rect(*color, zs, o)) invalidate(*color, o);
            if (span_rect(*depth, cs, o)) invalidate(*depth, o);
            barrier_copies_visible(cb);
            ++alias_batches_;
        }
    }
    if (debug & 8192u) {
        const Rect all{0, 0, kTargetSize - 1, kTargetSize - 1};
        for (auto &s : surfaces_) {
            write_back(cb, set0, *s, all);
            invalidate(*s, all);
        }
        barrier_copies_visible(cb);
    }
    ++drawn_;
    return true;
}

void HwRaster::destroy() {
    if (!ctx_ || !ctx_->device) return;
    vkDeviceWaitIdle(ctx_->device);
    for (auto &[key, p] : draw_)
        if (p) vkDestroyPipeline(ctx_->device, p, nullptr);
    for (VkPipeline p : {import_frame_, export_frame_, export_depth_, import_depth_, decode_, decode_tex_})
        if (p) vkDestroyPipeline(ctx_->device, p, nullptr);
    ctx_->destroy_image(decoded_);
    for (auto &t : textures_) ctx_->destroy_image(t->image);
    textures_.clear();
    texture_index_.clear();
    for (auto &[size, list] : free_by_size_)
        for (auto &t : list) ctx_->destroy_image(t->image);
    free_by_size_.clear();
    for (VkSampler sm : samplers_)
        if (sm) vkDestroySampler(ctx_->device, sm, nullptr);
    if (pool2_) vkDestroyDescriptorPool(ctx_->device, pool2_, nullptr);
    if (set2_layout_) vkDestroyDescriptorSetLayout(ctx_->device, set2_layout_, nullptr);
    for (auto &[key, fb] : framebuffers_) vkDestroyFramebuffer(ctx_->device, fb, nullptr);
    for (Retired &r : retired_) {
        if (r.framebuffer) vkDestroyFramebuffer(ctx_->device, r.framebuffer, nullptr);
        if (r.image.image) ctx_->destroy_image(r.image);
    }
    for (auto &s : surfaces_) ctx_->destroy_image(s->image);
    surfaces_.clear();
    ctx_->destroy_buffer(depth_stage_);
    ctx_->destroy_image(dummy_);
    for (FrameScratch &f : frames_) {
        ctx_->destroy_buffer(f.vertices);
        ctx_->destroy_buffer(f.uniforms);
    }
    if (pass_depth_rp_) vkDestroyRenderPass(ctx_->device, pass_depth_rp_, nullptr);
    if (pass_nodepth_) vkDestroyRenderPass(ctx_->device, pass_nodepth_, nullptr);
    if (pool_) vkDestroyDescriptorPool(ctx_->device, pool_, nullptr);
    if (layout_) vkDestroyPipelineLayout(ctx_->device, layout_, nullptr);
    if (set1_layout_) vkDestroyDescriptorSetLayout(ctx_->device, set1_layout_, nullptr);
    if (module_) vkDestroyShaderModule(ctx_->device, module_, nullptr);
    if (occlusion_) vkDestroyQueryPool(ctx_->device, occlusion_, nullptr);
    for (VkShaderModule &m : variant_modules_)
        if (m) vkDestroyShaderModule(ctx_->device, m, nullptr);
}

}
