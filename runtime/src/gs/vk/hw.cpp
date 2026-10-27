#include "hw.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2::gs::vk {

namespace {

constexpr std::uint32_t kPrimWords = 28, kSetupWords = 25;
constexpr std::uint32_t kTargetSize = 2048;
constexpr std::uint64_t kVertexBytes = 16u << 20, kUniformBytes = 1u << 20;
constexpr std::uint32_t kUniformStride = 256;

constexpr std::uint32_t F_TME = 2, F_ABE = 8, F_ZMSK = 32, F_ATE = 64, F_DATE = 128, F_ZTE = 512, F_PABE = 1024, F_COLCLAMP = 2048;

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

bool HwRaster::init(Context &ctx, VkShaderModule module, VkDescriptorSetLayout set0, std::uint32_t scale) {
    ctx_ = &ctx;
    scale_ = scale;
    VkDescriptorSetLayoutBinding b[4] = {};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 4;
    lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(ctx.device, &lci, nullptr, &set1_layout_) != VK_SUCCESS) return false;
    VkDescriptorSetLayout sets[2] = {set0, set1_layout_};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 2;
    pli.pSetLayouts = sets;
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &layout_) != VK_SUCCESS) return false;
    VkDescriptorPoolSize sizes[4] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 2;
    dpi.poolSizeCount = 4;
    dpi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &pool_) != VK_SUCCESS) return false;
    for (FrameScratch &f : frames_) {
        if (!ctx.create_buffer(f.vertices, kVertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true)) return false;
        if (!ctx.create_buffer(f.uniforms, kUniformBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true)) return false;
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &set1_layout_;
        if (vkAllocateDescriptorSets(ctx.device, &dai, &f.set1) != VK_SUCCESS) return false;
    }
    if (!build_pipelines(module)) return false;
    if (!ensure_target(kTargetSize * scale, kTargetSize * scale)) return false;
    frame_ = &frames_[0];
    return true;
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
        if (vkCreateRenderPass(ctx_->device, &rpi, nullptr, with_depth ? &pass_depth_ : &pass_nodepth_) != VK_SUCCESS) return false;
    }
    // Compute pipelines for the import/export passes.
    const char *compute_names[4] = {"import_frame", "export_frame", "export_depth", "import_depth"};
    VkPipeline *compute_out[4] = {&import_frame_, &export_frame_, &export_depth_, &import_depth_};
    for (int i = 0; i < 4; ++i) {
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
// factor, bits 11-13 op, bit 14 uses blend constants (FIX).
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
        return 0;
    }
    if (a == 1 && b == 2) {  // Cd * C + D
        if (d == 0) return key(VK_BLEND_FACTOR_ONE, fc, VK_BLEND_OP_ADD);
        if (d == 2) return key(VK_BLEND_FACTOR_ZERO, fc, VK_BLEND_OP_ADD);
        return 0;
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
    const auto it = draw_.find(key);
    if (it != draw_.end()) return it->second;
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
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
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
    stages[1].module = module_;
    stages[1].pName = "hw_fs";
    const VkCompareOp ops[4] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_GREATER};
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = key.with_depth && key.ztst != 1 ? VK_TRUE : VK_FALSE;
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
    gpi.renderPass = key.with_depth ? pass_depth_ : pass_nodepth_;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(ctx_->device, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] cannot build a draw pipeline (blend key %x)\n", key.blend);
        pipeline = VK_NULL_HANDLE;
    }
    draw_[key] = pipeline;
    return pipeline;
}

bool HwRaster::ensure_target(std::uint32_t width, std::uint32_t height) {
    Target &t = target_;
    if (t.width >= width && t.height >= height) return true;
    const VkImageUsageFlags color_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags depth_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!ctx_->create_image(t.color, width, height, VK_FORMAT_R8G8B8A8_UNORM, color_usage, VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if (!ctx_->create_image(t.depth, width, height, VK_FORMAT_D32_SFLOAT, depth_usage, VK_IMAGE_ASPECT_DEPTH_BIT)) return false;
    if (!ctx_->create_buffer(t.depth_stage, static_cast<VkDeviceSize>(width) * height * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false)) return false;
    t.width = width;
    t.height = height;
    // Both images live in GENERAL layout for their whole life.
    VkCommandBuffer cb = ctx_->begin_once();
    VkImageMemoryBarrier bars[2] = {};
    for (int i = 0; i < 2; ++i) {
        bars[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bars[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bars[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bars[i].srcQueueFamilyIndex = bars[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bars[i].image = i ? t.depth.image : t.color.image;
        bars[i].subresourceRange = {static_cast<VkImageAspectFlags>(i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
        bars[i].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, bars);
    ctx_->end_once(cb);
    t.color.layout = t.depth.layout = VK_IMAGE_LAYOUT_GENERAL;
    for (int with_depth = 0; with_depth < 2; ++with_depth) {
        VkImageView views[2] = {t.color.view, t.depth.view};
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = with_depth ? pass_depth_ : pass_nodepth_;
        fci.attachmentCount = with_depth ? 2 : 1;
        fci.pAttachments = views;
        fci.width = width;
        fci.height = height;
        fci.layers = 1;
        if (vkCreateFramebuffer(ctx_->device, &fci, nullptr, with_depth ? &t.framebuffer : &t.framebuffer_nodepth) != VK_SUCCESS) return false;
    }
    for (FrameScratch &f : frames_) {
        VkDescriptorBufferInfo ub{f.uniforms.buffer, 0, 32};
        VkDescriptorImageInfo store{VK_NULL_HANDLE, t.color.view, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo load{VK_NULL_HANDLE, t.color.view, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo depth{t.depth_stage.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[4] = {};
        for (int i = 0; i < 4; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = f.set1;
            w[i].dstBinding = static_cast<std::uint32_t>(i);
            w[i].descriptorCount = 1;
        }
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        w[0].pBufferInfo = &ub;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo = &store;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w[2].pImageInfo = &load;
        w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[3].pBufferInfo = &depth;
        vkUpdateDescriptorSets(ctx_->device, 4, w, 0, nullptr);
    }
    return true;
}

void HwRaster::begin_frame(int slot) {
    frame_ = &frames_[slot & 1];
    frame_->vertex_used = frame_->uniform_used = 0;
}

std::uint32_t HwRaster::push_uniform(FrameScratch &f, const std::uint32_t words[8]) {
    const std::uint32_t offset = static_cast<std::uint32_t>(f.uniform_used);
    std::memcpy(static_cast<std::uint8_t *>(f.uniforms.mapped) + offset, words, 32);
    f.uniform_used += kUniformStride;
    return offset;
}

// Everything the hardware path does not implement yet sends the batch to
// the compute rasterizer: blending, destination alpha, frame masks, frame
// formats other than 32-bit, 32-bit depth, alpha-test fail modes that keep
// writing, lines, points, upscaling, and batches that read their own target.
const char *const kReasons[12] = {"scale", "empty", "snapshot", "blend", "date", "fbmsk", "fpsm", "ztst never", "zpsm", "atest fail mode", "prim kind", "scratch full"};

void HwRaster::print_reasons() const {
    std::fprintf(stderr, "[vk] fallback reasons:");
    for (int i = 0; i < 12; ++i)
        if (reasons_[i]) std::fprintf(stderr, " %s %llu", kReasons[i], static_cast<unsigned long long>(reasons_[i]));
    std::fprintf(stderr, "\n");
}

bool HwRaster::supported(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params, const std::uint32_t *&setups, std::uint32_t &zmode) const {
    const auto no = [&](int reason) {
        ++reasons_[reason];
        return false;
    };
    if (scale_ != 1) return no(0);
    if (!cmd.prims || !cmd.setups) return no(1);
    const std::uint32_t *p = params + cmd.param * (gpu::kParamStride / 4);
    if (p[6] > p[5]) return no(2);  // snapshot: the batch reads what it draws
    setups = arena + p[2];
    const std::uint32_t *prims = arena + p[1];
    zmode = 0;
    for (std::uint32_t i = 0; i < cmd.setups; ++i) {
        const std::uint32_t *w = setups + i * kSetupWords;
        const std::uint32_t flags = w[0];
        if (flags & F_ABE) {
            // Blend when the factors map onto the blend unit, colours clamp, and
            // per-pixel alpha blending (PABE) is off.
            if ((flags & F_PABE) || !(flags & F_COLCLAMP) || !blend_key(w, ctx_->features.dualSrcBlend == VK_TRUE)) return no(3);
        }
        if (flags & F_DATE) return no(4);
        if (w[3] != 0x00u && w[3] != 0x01u) return no(6);
        if (flags & F_ZTE) {
            const std::uint32_t ztst = (w[7] >> 8) & 3u;
            if (ztst == 0) return no(7);
            // 32-bit Z rides in a float depth attachment with 24 bits of mantissa; exact for Z24 and Z16.
            if (w[6] != 0x30u && w[6] != 0x31u && w[6] != 0x32u && w[6] != 0x3Au) return no(8);
            // Depth memory is only touched when the test can fail or the
            // result is written; games alias an unused Z buffer with other data.
            if (ztst != 1u || !(flags & F_ZMSK)) zmode |= 1u;
        }
        if ((flags & F_ATE) && ((w[7] & 7u) != 1u) && ((w[7] >> 4) & 3u) != 0) return no(9);
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

bool HwRaster::draw(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *params, VkBuffer) {
    // PS2_HW_DEBUG bits, for isolating driver trouble: 1 skip batches that
    // need depth, 2 skip the depth import draw, 4 skip the depth export.
    static const unsigned debug = std::getenv("PS2_HW_DEBUG") ? static_cast<unsigned>(std::atoi(std::getenv("PS2_HW_DEBUG"))) : 0u;
    const std::uint32_t *setups = nullptr;
    std::uint32_t zmode = 0;
    if (!supported(cmd, arena, params, setups, zmode) || ((debug & 1u) && zmode)) {
        ++fallback_;
        return false;
    }
    FrameScratch &f = *frame_;
    const std::uint64_t need = static_cast<std::uint64_t>(cmd.prims) * 6 * sizeof(Vertex);
    if (f.vertex_used + need > kVertexBytes || f.uniform_used + (cmd.setups + 4) * kUniformStride > kUniformBytes) {
        ++reasons_[11];
        ++fallback_;
        return false;
    }
    const std::uint32_t *p = params + cmd.param * (gpu::kParamStride / 4);
    const std::uint32_t *prims = arena + p[1];
    const std::uint32_t setup_offset = p[2];
    // Frame masks apply at export, so a batch whose setups mask differently is
    // drawn as one mini-batch per run of primitives sharing a setup.
    bool mixed_masks = false;
    for (std::uint32_t i = 1; i < cmd.setups; ++i) mixed_masks = mixed_masks || setups[i * kSetupWords + 4] != setups[4];
    if (mixed_masks && cmd.prims > 1) {
        std::uint32_t i = 0;
        while (i < cmd.prims) {
            const std::uint32_t si = prims[i * kPrimWords];
            std::uint32_t j = i + 1;
            while (j < cmd.prims && prims[j * kPrimWords] == si) ++j;
            gpu::Command part = cmd;
            part.prims = j - i;
            // Point the part at its primitives through a private parameter copy.
            std::uint32_t words[gpu::kParamStride / 4];
            std::memcpy(words, p, sizeof words);
            words[1] = p[1] + i * kPrimWords;
            part.param = 0;
            if (!draw_part(cb, part, set0, arena, words, setups, zmode, cmd.param)) return false;
            i = j;
        }
        return true;
    }
    return draw_part(cb, cmd, set0, arena, p, setups, zmode, cmd.param);
}

bool HwRaster::draw_part(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *p, const std::uint32_t *setups, std::uint32_t zmode,
                         std::uint32_t param_index) {
    static const unsigned debug = std::getenv("PS2_HW_DEBUG") ? static_cast<unsigned>(std::atoi(std::getenv("PS2_HW_DEBUG"))) : 0u;
    FrameScratch &f = *frame_;
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
    const std::uint32_t rw = (maxx - minx + 1) * scale_, rh = (maxy - miny + 1) * scale_;
    static const bool trace = std::getenv("PS2_HW_TRACE") != nullptr;
    if (trace) {
        const std::uint32_t *w = setups + prims[0] * kSetupWords;
        std::fprintf(stderr, "[hw] batch prims %u setups %u rect %u,%u-%u,%u flags %04x fpsm %02x fbmsk %08x zpsm %02x ztst %u zmsk %u z0 %08x kind %u scissor %u,%u-%u,%u xy %.1f,%.1f %.1fx%.1f\n", cmd.prims, cmd.setups,
                     minx, miny, maxx, maxy, w[0], w[3], w[4], w[6], (w[7] >> 8) & 3u, (w[0] & F_ZMSK) ? 1u : 0u, prims[24], prims[1],
                     w[10] & 0xFFFFu, w[11] & 0xFFFFu, w[10] >> 16, w[11] >> 16, fat(prims[4]), fat(prims[5]), fat(prims[6]), fat(prims[7]));
        if (w[0] & F_TME) std::fprintf(stderr, "[hw]   tex0 %.1f,%.1f q %.3f tex1 %.1f,%.1f q %.3f color %08x tw %u th %u fst %u\n", fat(prims[15]), fat(prims[16]), fat(prims[17]), fat(prims[18]), fat(prims[19]), fat(prims[20]), prims[13], w[15] & 0xFFu, w[15] >> 8, (w[0] & 16u) ? 1u : 0u);
    }
    const bool with_depth = zmode != 0;
    const std::uint32_t first_setup = setup_offset;  // all setups share frame and depth buffers
    const std::uint32_t rect_words[8] = {first_setup, setups[prims[0] * kSetupWords + 4], scale_, with_depth ? 1u : 0u, minx, miny, rw, rh};
    const std::uint32_t rect_uniform = push_uniform(f, rect_words);
    const std::uint32_t draw_words[8] = {first_setup, 0, scale_, with_depth ? 1u : 0u, 0, 0, target_.width, target_.height};
    const std::uint32_t draw_uniform = push_uniform(f, draw_words);
    const std::uint32_t groups_x = (rw + 15) / 16, groups_y = (rh + 15) / 16;

    const VkDescriptorSet sets[2] = {set0, f.set1};
    // Set 0 carries the compute renderer's dynamic parameter block too, so
    // two dynamic offsets go with the two sets.
    const std::uint32_t param_offset = param_index * gpu::kParamStride;
    const auto bind = [&](VkPipelineBindPoint point, std::uint32_t uniform) {
        const std::uint32_t offsets[2] = {param_offset, uniform};
        vkCmdBindDescriptorSets(cb, point, layout_, 0, 2, sets, 2, offsets);
    };
    VkMemoryBarrier mem{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    const auto image_barrier = [&](VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags from, VkPipelineStageFlags to) {
        VkImageMemoryBarrier bars[2] = {};
        for (int i = 0; i < 2; ++i) {
            bars[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            bars[i].srcAccessMask = src;
            bars[i].dstAccessMask = dst;
            bars[i].oldLayout = bars[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            bars[i].srcQueueFamilyIndex = bars[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bars[i].image = i ? target_.depth.image : target_.color.image;
            bars[i].subresourceRange = {static_cast<VkImageAspectFlags>(i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
        }
        vkCmdPipelineBarrier(cb, from, to, 0, 0, nullptr, 0, nullptr, 2, bars);
    };

    // 1. Import the rectangle's frame words into the colour target, and its
    // depth words through the staging buffer into the depth attachment.
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, import_frame_);
    bind(VK_PIPELINE_BIND_POINT_COMPUTE, rect_uniform);
    vkCmdDispatch(cb, groups_x, groups_y, 1);
    const VkBufferImageCopy depth_region{0, rw, rh, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {static_cast<std::int32_t>(minx * scale_), static_cast<std::int32_t>(miny * scale_), 0}, {rw, rh, 1}};
    if (with_depth && !(debug & 2u)) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, import_depth_);
        bind(VK_PIPELINE_BIND_POINT_COMPUTE, rect_uniform);
        vkCmdDispatch(cb, groups_x, groups_y, 1);
        mem.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mem.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mem, 0, nullptr, 0, nullptr);
        vkCmdCopyBufferToImage(cb, target_.depth_stage.buffer, target_.depth.image, VK_IMAGE_LAYOUT_GENERAL, 1, &depth_region);
    }
    image_barrier(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

    // 2. Render pass: depth import (if any), then the batch.
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = with_depth ? pass_depth_ : pass_nodepth_;
    rbi.framebuffer = with_depth ? target_.framebuffer : target_.framebuffer_nodepth;
    rbi.renderArea = {{static_cast<std::int32_t>(minx * scale_), static_cast<std::int32_t>(miny * scale_)}, {rw, rh}};
    vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(target_.width), static_cast<float>(target_.height), 0.0f, 1.0f};
    vkCmdSetViewport(cb, 0, 1, &viewport);
    const VkRect2D scissor = rbi.renderArea;
    vkCmdSetScissor(cb, 0, 1, &scissor);
    std::vector<std::uint32_t> first, counts;
    const VkDeviceSize prim_base = f.vertex_used;
    append_prims(f, prims, cmd.prims, first, counts);
    (void)draw_uniform;
    vkCmdBindVertexBuffers(cb, 0, 1, &f.vertices.buffer, &prim_base);
    std::uint32_t run = 0;
    for (std::uint32_t i = 0; i < cmd.prims;) {
        const std::uint32_t si = prims[i * kPrimWords];
        const std::uint32_t *w = setups + si * kSetupWords;
        const bool zte = w[0] & F_ZTE;
        const std::uint32_t ztst = zte ? (w[7] >> 8) & 3u : 1u;
        const bool zwrite = zte && !(w[0] & F_ZMSK);
        const std::uint32_t words[8] = {setup_offset + si * kSetupWords, 0, scale_, (with_depth ? 1u : 0u) | ((debug & 8u) ? 2u : 0u) | ((debug & 16u) ? 4u : 0u), 0, 0, target_.width, target_.height};
        const std::uint32_t uniform = push_uniform(f, words);
        const std::uint32_t blend = (w[0] & F_ABE) ? blend_key(w, ctx_->features.dualSrcBlend == VK_TRUE) : 0u;
        if (blend & (1u << 14)) {
            const float fix = static_cast<float>(w[9]) / 128.0f;
            const float constants[4] = {fix, fix, fix, fix};
            vkCmdSetBlendConstants(cb, constants);
        }
        VkPipeline pipeline = pipeline_for({ztst, zwrite ? 1u : 0u, with_depth ? 1u : 0u, blend});
        if (!pipeline) {
            // Pipeline creation failed: leave the batch to the compute path next time.
            vkCmdEndRenderPass(cb);
            ++fallback_;
            return false;
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        bind(VK_PIPELINE_BIND_POINT_GRAPHICS, uniform);
        vkCmdDraw(cb, counts[run], 1, first[run], 0);
        std::uint32_t j = i + 1;
        while (j < cmd.prims && prims[j * kPrimWords] == si) ++j;
        i = j;
        ++run;
    }
    vkCmdEndRenderPass(cb);

    // 3. Export the rectangle back to GS memory.
    image_barrier(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, export_frame_);
    bind(VK_PIPELINE_BIND_POINT_COMPUTE, rect_uniform);
    vkCmdDispatch(cb, groups_x, groups_y, 1);
    if (with_depth && !(debug & 4u)) {
        // Depth attachment -> staging buffer -> GS memory words.
        VkImageMemoryBarrier to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_transfer.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_transfer.oldLayout = to_transfer.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_transfer.srcQueueFamilyIndex = to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image = target_.depth.image;
        to_transfer.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_transfer);
        vkCmdCopyImageToBuffer(cb, target_.depth.image, VK_IMAGE_LAYOUT_GENERAL, target_.depth_stage.buffer, 1, &depth_region);
        mem.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        mem.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mem, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, export_depth_);
        bind(VK_PIPELINE_BIND_POINT_COMPUTE, rect_uniform);
        vkCmdDispatch(cb, groups_x, groups_y, 1);
    }
    ++drawn_;
    return true;
}

void HwRaster::destroy() {
    if (!ctx_ || !ctx_->device) return;
    vkDeviceWaitIdle(ctx_->device);
    for (auto &[key, p] : draw_)
        if (p) vkDestroyPipeline(ctx_->device, p, nullptr);
    for (VkPipeline p : {import_frame_, export_frame_, export_depth_, import_depth_})
        if (p) vkDestroyPipeline(ctx_->device, p, nullptr);
    if (target_.framebuffer) vkDestroyFramebuffer(ctx_->device, target_.framebuffer, nullptr);
    if (target_.framebuffer_nodepth) vkDestroyFramebuffer(ctx_->device, target_.framebuffer_nodepth, nullptr);
    ctx_->destroy_image(target_.color);
    ctx_->destroy_image(target_.depth);
    ctx_->destroy_buffer(target_.depth_stage);
    for (FrameScratch &f : frames_) {
        ctx_->destroy_buffer(f.vertices);
        ctx_->destroy_buffer(f.uniforms);
    }
    if (pass_depth_) vkDestroyRenderPass(ctx_->device, pass_depth_, nullptr);
    if (pass_nodepth_) vkDestroyRenderPass(ctx_->device, pass_nodepth_, nullptr);
    if (pool_) vkDestroyDescriptorPool(ctx_->device, pool_, nullptr);
    if (layout_) vkDestroyPipelineLayout(ctx_->device, layout_, nullptr);
    if (set1_layout_) vkDestroyDescriptorSetLayout(ctx_->device, set1_layout_, nullptr);
}

}
