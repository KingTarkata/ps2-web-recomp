// Draws raster batches with the GPU's rasterizer (hw.wgsl) when the batch
// only needs what the hardware path implements, and says no otherwise so
// the device falls back to the compute rasterizer. docs/gs-hw-renderer.md.
#pragma once

#include "../gpu/device.hpp"
#include "context.hpp"

#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace ps2::gs::vk {

class HwRaster {
public:
    // `module` holds gs.wgsl + hw.wgsl; `set0` is the compute renderer's
    // buffer set layout, bound as set 0 of every pipeline here too.
    // `code` is the SPIR-V of gs.wgsl + hw.wgsl (kHwSpvScaleN); the module is
    // created here, so a driver that rejects it only costs the hardware path.
    bool init(Context &ctx, const std::uint32_t *code, std::size_t words, VkDescriptorSetLayout set0, std::uint32_t scale);
    void destroy();

    // Called while recording a command buffer, before the device would
    // dispatch the compute raster for `cmd`. Returns true when the batch was
    // drawn here (import, draw, export all recorded into `cb`).
    bool draw(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *params, VkBuffer data_buffer);
    // Call once per submit before recording: the per-submit scratch
    // (vertex and uniform rings) belongs to the frame slot in flight.
    void begin_frame(int slot);
    std::uint64_t drawn_batches() const { return drawn_; }
    std::uint64_t fallback_batches() const { return fallback_; }
    void print_reasons() const;

private:
    struct Target {
        Image color, depth;
        Buffer depth_stage;  // D32 bits of the rectangle in flight
        VkFramebuffer framebuffer = VK_NULL_HANDLE, framebuffer_nodepth = VK_NULL_HANDLE;
        std::uint32_t width = 0, height = 0;  // samples
    };
    struct FrameScratch {
        Buffer vertices, uniforms;
        std::uint64_t vertex_used = 0, uniform_used = 0;
        VkDescriptorSet set1 = VK_NULL_HANDLE;
    };
    struct Vertex {
        float x, y;
        std::uint32_t z;
        float r, g, b, a;
        float s, t, q, f;
    };

    bool build_pipelines(VkShaderModule module);
    VkShaderModule module_ = VK_NULL_HANDLE;
    bool ensure_target(std::uint32_t width, std::uint32_t height);
    bool draw_part(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *params, const std::uint32_t *setups, std::uint32_t zmode,
                   std::uint32_t param_index);
    bool supported(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params, const std::uint32_t *&setups, std::uint32_t &zmode) const;
    std::uint32_t push_uniform(FrameScratch &f, const std::uint32_t words[8]);
    void append_prims(FrameScratch &f, const std::uint32_t *prims, std::uint32_t count, std::vector<std::uint32_t> &first, std::vector<std::uint32_t> &counts);

    Context *ctx_ = nullptr;
    std::uint32_t scale_ = 1;
    VkDescriptorSetLayout set1_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkRenderPass pass_depth_ = VK_NULL_HANDLE, pass_nodepth_ = VK_NULL_HANDLE;
    VkPipeline import_frame_ = VK_NULL_HANDLE, export_frame_ = VK_NULL_HANDLE, export_depth_ = VK_NULL_HANDLE, import_depth_ = VK_NULL_HANDLE;
    // Draw pipelines by state: depth mode and write, depth attachment, and the
    // blend equation (0 when off), built on first use.
    struct PipelineKey {
        std::uint32_t ztst, zwrite, with_depth, blend;
        bool operator<(const PipelineKey &o) const { return std::tie(ztst, zwrite, with_depth, blend) < std::tie(o.ztst, o.zwrite, o.with_depth, o.blend); }
    };
    std::map<PipelineKey, VkPipeline> draw_;
    VkPipeline pipeline_for(const PipelineKey &key);
    // Packs a GS blend (A, B, C, D selectors with FIX) into Vulkan factors;
    // 0 when the hardware cannot express it.
    static std::uint32_t blend_key(const std::uint32_t *setup, bool dual_source);
    Target target_;
    FrameScratch frames_[2];
    FrameScratch *frame_ = nullptr;
    std::uint64_t drawn_ = 0, fallback_ = 0;
    // Why batches fell back, by the first reason met (see hw.cpp kReasons).
    mutable std::uint64_t reasons_[12] = {};
};

}
