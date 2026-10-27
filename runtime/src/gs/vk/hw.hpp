// Draws raster batches with the GPU's rasterizer (hw.wgsl) when the batch
// only needs what the hardware path implements, and says no otherwise so
// the device falls back to the compute rasterizer. docs/gs-hw-renderer.md.
//
// Frame and depth buffers the game draws into become resident *surfaces*:
// images that mirror one buffer of GS memory, tile by tile. A tile is
// *valid* when the image holds the memory's contents and *dirty* when the
// image is newer than memory. Batches import the tiles they need and mark
// what they draw dirty; dirty tiles are written back only when something
// else reads that memory, and dropped when something else writes it.
#pragma once

#include "../gpu/device.hpp"
#include "context.hpp"

#include <bitset>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace ps2::gs::vk {

class HwRaster {
public:
    // `code` is the SPIR-V of gs.wgsl + hw.wgsl (kHwSpvScaleN); the module is
    // created here, so a driver that rejects it only costs the hardware path.
    // `set0` is the compute renderer's buffer set layout, bound as set 0 of
    // every pipeline here too.
    bool init(Context &ctx, const std::uint32_t *code, std::size_t words, VkDescriptorSetLayout set0, std::uint32_t scale);
    void destroy();

    // Called while recording a command buffer, before the device would
    // dispatch the compute raster for `cmd`. Returns true when the batch was
    // drawn here (imports, draw and bookkeeping all recorded into `cb`).
    bool draw(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *params, VkBuffer data_buffer);
    // Whether a setup record alone (blend, tests, formats) is drawable here.
    bool setup_supported(const std::uint32_t *setup) const;
    // Runs the texture decode of `cmd` with the hardware module's pipeline,
    // which writes the decoded cache image as well as the buffer.
    void dispatch_decode(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0);
    // Called before every command the device runs itself: writes back the
    // dirty tiles whose memory the command reads, and drops the tiles whose
    // memory it writes.
    void before_command(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *params);
    // GS memory is about to be read outside the command stream (a download
    // or a state save): write back what overlaps, synchronously.
    void flush_sync(VkDescriptorSet set0, const gpu::Span &span);
    // GS memory was written outside the command stream: every surface is stale.
    void drop_all();
    // Call once per submit before recording: the per-submit scratch
    // (vertex and uniform rings) belongs to the frame slot in flight, and
    // objects retired two submits ago can go.
    void begin_frame(int slot);
    std::uint64_t drawn_batches() const { return drawn_; }
    std::uint64_t fallback_batches() const { return fallback_; }
    void print_reasons() const;

private:
    static constexpr std::uint32_t kTargetSize = 2048, kTile = 32, kTiles = kTargetSize / kTile;
    static constexpr std::size_t kMaxSurfaces = 32;
    struct Plane {
        std::bitset<kTiles * kTiles> valid, dirty;
    };
    // GS pixels, inclusive.
    struct Rect {
        std::uint32_t x0, y0, x1, y1;
    };
    struct Surface {
        std::uint32_t bp = 0, bw = 0, psm = 0;
        bool depth = false;
        Image image;
        std::uint32_t width = 0, height = 0;  // GS pixels
        Plane plane;
        VkDescriptorSet set1[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};  // per frame slot
        std::uint64_t last_use = 0;
    };
    struct FrameScratch {
        Buffer vertices, uniforms;
        std::uint64_t vertex_used = 0, uniform_used = 0;
    };
    struct Vertex {
        float x, y;
        std::uint32_t z;
        float r, g, b, a;
        float s, t, q, f;
    };
    // Objects a recorded command buffer may still reference.
    struct Retired {
        Image image;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkDescriptorSet sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        std::uint64_t at = 0;  // submit counter when it may be destroyed
    };

    bool build_pipelines(VkShaderModule module);
    bool supported(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params, const std::uint32_t *&setups, std::uint32_t &zmode) const;
    std::uint32_t push_uniform(FrameScratch &f, const std::uint32_t words[12]);
    void append_prims(FrameScratch &f, const std::uint32_t *prims, std::uint32_t count, std::vector<std::uint32_t> &first, std::vector<std::uint32_t> &counts);
    void bind(VkCommandBuffer cb, VkPipelineBindPoint point, VkDescriptorSet set0, VkDescriptorSet set1, std::uint32_t param_offset, std::uint32_t uniform);

    // Surfaces.
    Surface *find_surface(std::uint32_t bp, std::uint32_t bw, std::uint32_t psm, bool depth);
    Surface *surface_for(VkCommandBuffer cb, VkDescriptorSet set0, std::uint32_t bp, std::uint32_t bw, std::uint32_t psm, bool depth, const Rect &needed);
    bool allocate(Surface &s, std::uint32_t width, std::uint32_t height);
    void retire(Surface &s);
    VkFramebuffer framebuffer_for(const Surface &color, const Surface *depth);
    bool span_rect(const Surface &s, const gpu::Span &span, Rect &out) const;
    gpu::Span rect_span(const Surface &s, const Rect &r) const;
    template <class Pred> std::vector<Rect> tile_rects(const Plane &p, const Rect &within, Pred pred) const;
    void copy_rects(VkCommandBuffer cb, VkDescriptorSet set0, Surface &s, bool import, const std::vector<Rect> &rects);
    void import_tiles(VkCommandBuffer cb, VkDescriptorSet set0, Surface &s, const Rect &rect);
    void write_back(VkCommandBuffer cb, VkDescriptorSet set0, Surface &s, const Rect &rect, bool drop_aliases = true);
    void invalidate(Surface &s, const Rect &rect);
    void mark_dirty(Surface &s, const Rect &rect);
    // Memory `span` is about to be read / written by something other than
    // `except`: write back (and for writes drop) what overlaps in every surface.
    void memory_read(VkCommandBuffer cb, VkDescriptorSet set0, const gpu::Span &span, const Surface *except = nullptr);
    void memory_written(VkCommandBuffer cb, VkDescriptorSet set0, const gpu::Span &span, const Surface *except_a = nullptr, const Surface *except_b = nullptr);
    void barrier_after_copies(VkCommandBuffer cb, const Surface &color, const Surface *depth);
    void barrier_after_draw(VkCommandBuffer cb, const Surface &color, const Surface *depth);
    void barrier_copies_visible(VkCommandBuffer cb);

    Context *ctx_ = nullptr;
    std::uint32_t scale_ = 1;
    VkDescriptorSetLayout set1_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkRenderPass pass_depth_ = VK_NULL_HANDLE, pass_nodepth_ = VK_NULL_HANDLE;
    VkPipeline import_frame_ = VK_NULL_HANDLE, export_frame_ = VK_NULL_HANDLE, export_depth_ = VK_NULL_HANDLE, import_depth_ = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;
    // Draw pipelines by state: depth mode and write, depth attachment, the
    // blend equation (0 when off) and the colour write mask, built on first use.
    struct PipelineKey {
        std::uint32_t ztst, zwrite, with_depth, blend, wmask;
        bool operator<(const PipelineKey &o) const { return std::tie(ztst, zwrite, with_depth, blend, wmask) < std::tie(o.ztst, o.zwrite, o.with_depth, o.blend, o.wmask); }
    };
    std::map<PipelineKey, VkPipeline> draw_;
    VkPipeline pipeline_for(const PipelineKey &key);
    // Packs a GS blend (A, B, C, D selectors with FIX) into Vulkan factors;
    // 0 when the hardware cannot express it.
    static std::uint32_t blend_key(const std::uint32_t *setup, bool dual_source);
    Buffer depth_stage_;  // D32 bits of the rectangle in flight, shared by all surfaces
    Image dummy_;         // stands in for the image bindings of depth surfaces
    Image decoded_;       // the decoded texture cache as a 2048-wide R32_UINT image
    VkDescriptorSet misc_set1_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};  // set 1 without a surface, for decodes
    VkPipeline decode_ = VK_NULL_HANDLE;
    bool write_set1(VkDescriptorSet set, int slot, VkImageView surface_view);
    std::vector<std::unique_ptr<Surface>> surfaces_;
    std::map<std::pair<VkImageView, VkImageView>, VkFramebuffer> framebuffers_;
    std::vector<Retired> retired_;
    std::uint64_t submits_ = 0, use_counter_ = 0;
    FrameScratch frames_[2];
    FrameScratch *frame_ = nullptr;
    int slot_ = 0;
    std::uint64_t drawn_ = 0, fallback_ = 0;
    std::uint64_t imported_px_ = 0, exported_px_ = 0, evictions_ = 0, uniform_overflows_ = 0, alias_batches_ = 0;
    // Why batches fell back, by the first reason met (see hw.cpp kReasons).
    mutable std::uint64_t reasons_[12] = {};
    mutable int last_reason_ = 0;
};

}
