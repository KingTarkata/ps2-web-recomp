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

#include <array>
#include <functional>
#include <list>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace ps2::gs::vk {

class HwRaster {
public:
    // `code` is the SPIR-V of gs.wgsl + hw.wgsl (kHwSpvScaleN); the module is
    // created here, so a driver that rejects it only costs the hardware path.
    // `set0` is the compute renderer's buffer set layout, bound as set 0 of
    // every pipeline here too.
    // `variants`: eight specialised modules (hw.wgsl HW_VARIANT 0-7) or nulls.
    bool init(Context &ctx, const std::uint32_t *code, std::size_t words, const std::uint32_t *const variants[8], const std::size_t variant_words[8], VkDescriptorSetLayout set0, std::uint32_t scale);
    void destroy();

    // Called while recording a command buffer, before the device would
    // dispatch the compute raster for `cmd`. Returns true when the batch was
    // drawn here (imports, draw and bookkeeping all recorded into `cb`).
    bool draw(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *params, VkBuffer data_buffer);
    // What to do with a raster batch before anything is recorded: decline it
    // (compute path), draw it now, or hold it back because it targets other
    // surfaces than the open pass and nothing drawn since needs it first. A
    // held batch is drawn when the pass closes (the device's hook), after
    // the compute commands held with it; batches on the open surfaces that
    // touch a held batch's memory close the pass first (held_conflict).
    enum class Plan { Declined, Draw, Hold };
    Plan plan(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params);
    void note_held(const gpu::Command &cmd);
    void clear_held() {
        held_reads_.clear();
        held_writes_.clear();
    }
    bool held_conflict(const gpu::Command &cmd) const;
    // Whether a setup record alone (blend, tests, formats) is drawable here.
    bool setup_supported(const std::uint32_t *setup) const;
    bool load_clut_from_surface(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0);
    static unsigned debug_bits();
    const char *last_reason_name() const;
    // The whole batch, by the same rules draw() applies (supported()).
    bool batch_supported(const std::uint32_t *setups, std::uint32_t nsetups, const std::uint32_t *prims, std::uint32_t nprims) const;
    // Runs the texture decode of `cmd` with the hardware module's pipeline,
    // which writes the decoded cache image as well as the buffer.
    void dispatch_decode(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0);
    // Called before every command the device runs itself: writes back the
    // dirty tiles whose memory the command reads, and drops the tiles whose
    // memory it writes.
    // `next_is_hw_raster`: the command after this one is a raster batch the
    // hardware path will draw; a snapshot copy before it then skips its
    // write-back (the batch samples the surface image instead).
    void before_command(VkCommandBuffer cb, const gpu::Command &cmd, VkDescriptorSet set0, const std::uint32_t *arena, const std::uint32_t *params, bool next_is_hw_raster = false);
    // before_command ran the command itself (a CLUT load reading a resident
    // surface's image): the device skips its dispatch. Cleared by the call.
    bool take_command_done() { const bool d = command_done_; command_done_ = false; return d; }
    bool would_draw(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params) const;
    // Whether the hardware path would draw `cmd` without the per-submit
    // decode the batching layer emitted for it (every texture it samples is
    // served by the texture cache or a surface).
    bool decodes_unneeded(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params) const;
    // GS memory is about to be read outside the command stream (a download
    // or a state save): write back what overlaps, synchronously.
    void flush_sync(VkDescriptorSet set0, const gpu::Span &span);
    // GS memory was written outside the command stream: every surface is stale.
    void drop_all();
    // Call once per submit before recording: the per-submit scratch
    // (vertex and uniform rings) belongs to the frame slot in flight, and
    // objects retired two submits ago can go.
    void begin_frame(int slot);
    void end_frame(VkCommandBuffer cb);  // closes the open render pass and the per-frame fragment count
    bool pass_open() const { return pass_cb_ != VK_NULL_HANDLE; }
    void set_inverse_table(std::uint32_t words) { inverse32_words_ = words; }
    // Consecutive hardware-drawn batches on the same colour (and depth)
    // surface share one render pass; anything recorded outside a pass
    // (imports, write-backs, decodes, copies, image barriers) closes it
    // first. 230 passes per frame, each between full barriers, had the GPU
    // draining more than drawing.
    enum CloseWhy { kCloseImport, kCloseWriteBack, kCloseDecode, kCloseCopy, kCloseTexture, kCloseMemoryRead, kCloseSurfaces, kCloseFailed, kCloseEndFrame, kCloseDevice, kCloseDecoded, kCloseWbDraw, kCloseWbDecode, kCloseWbBefore, kCloseWbEvict, kCloseWbOverlap, kCloseImportOwn, kCloseImportSource, kCloseWbUncached, kCloseDepthToggle, kCloseColorChange, kCloseDepthChange, kCloseCount };
    CloseWhy wb_why_ = kCloseWriteBack, import_why_ = kCloseImport;  // who asked for the write-back / import that closed the pass (statistics)
    struct WhyScope {
        CloseWhy &slot, saved;
        WhyScope(CloseWhy &s, CloseWhy v) : slot(s), saved(s) { slot = v; }
        ~WhyScope() { slot = saved; }
    };
    void close_pass(VkCommandBuffer cb, CloseWhy why = kCloseDevice);
    // Called right after a pass is closed (the device records the compute
    // commands it held back while the pass was open).
    void on_pass_closed(std::function<void()> hook) { pass_closed_hook_ = std::move(hook); }
    // A compute command the device holds back while the pass is open; its
    // before_command runs when it is recorded (at pass close), which is only
    // right if no batch drawn in between touches its memory: such a batch
    // closes the pass first (deferred_conflict).
    void note_deferred(const gpu::Command &cmd);
    void clear_deferred() {
        deferred_reads_.clear();
        deferred_writes_.clear();
        deferred_count_ = 0;
    }
    bool deferred_conflict(const gpu::Command &cmd) const;
    std::uint64_t drawn_batches() const { return drawn_; }
    std::uint64_t fallback_batches() const { return fallback_; }
    void print_reasons() const;

private:
    static constexpr std::uint32_t kTargetSize = 2048, kTile = 32, kTiles = kTargetSize / kTile;
    static constexpr std::size_t kMaxSurfaces = 32;
    // Tile validity and dirtiness, one 64-bit word per tile row (kTiles is
    // 64): rectangle operations are a few word operations per row instead of
    // a bit test per tile (the per-tile loops were 3% of the render thread).
    struct Plane {
        std::uint64_t valid[kTiles] = {}, dirty[kTiles] = {};
        std::uint32_t dirty_count = 0;  // bits set in `dirty`
        static std::uint64_t row_mask(std::uint32_t tx0, std::uint32_t tx1) { return tx1 - tx0 >= 63 ? ~0ull : (((1ull << (tx1 - tx0 + 1)) - 1) << tx0); }
        bool is_valid(std::uint32_t t) const { return (valid[t >> 6] >> (t & 63)) & 1; }
        bool is_dirty(std::uint32_t t) const { return (dirty[t >> 6] >> (t & 63)) & 1; }
        void set_valid(std::uint32_t t) { valid[t >> 6] |= 1ull << (t & 63); }
        void reset() {
            for (std::uint32_t i = 0; i < kTiles; ++i) valid[i] = dirty[i] = 0;
            dirty_count = 0;
        }
        // Tile rectangles, inclusive.
        bool all_valid(std::uint32_t tx0, std::uint32_t ty0, std::uint32_t tx1, std::uint32_t ty1) const {
            const std::uint64_t m = row_mask(tx0, tx1);
            for (std::uint32_t ty = ty0; ty <= ty1; ++ty)
                if ((valid[ty] & m) != m) return false;
            return true;
        }
        void set_valid(std::uint32_t tx0, std::uint32_t ty0, std::uint32_t tx1, std::uint32_t ty1) {
            const std::uint64_t m = row_mask(tx0, tx1);
            for (std::uint32_t ty = ty0; ty <= ty1; ++ty) valid[ty] |= m;
        }
        void invalidate(std::uint32_t tx0, std::uint32_t ty0, std::uint32_t tx1, std::uint32_t ty1) {
            const std::uint64_t m = row_mask(tx0, tx1);
            for (std::uint32_t ty = ty0; ty <= ty1; ++ty) {
                valid[ty] &= ~m;
                dirty_count -= static_cast<std::uint32_t>(__builtin_popcountll(dirty[ty] & m));
                dirty[ty] &= ~m;
            }
        }
        void mark_dirty(std::uint32_t tx0, std::uint32_t ty0, std::uint32_t tx1, std::uint32_t ty1) {
            const std::uint64_t m = row_mask(tx0, tx1);
            for (std::uint32_t ty = ty0; ty <= ty1; ++ty) {
                valid[ty] |= m;
                dirty_count += static_cast<std::uint32_t>(__builtin_popcountll(m & ~dirty[ty]));
                dirty[ty] |= m;
            }
        }
        void clear_dirty(std::uint32_t tx0, std::uint32_t ty0, std::uint32_t tx1, std::uint32_t ty1) {
            const std::uint64_t m = row_mask(tx0, tx1);
            for (std::uint32_t ty = ty0; ty <= ty1; ++ty) {
                dirty_count -= static_cast<std::uint32_t>(__builtin_popcountll(dirty[ty] & m));
                dirty[ty] &= ~m;
            }
        }
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
        // The GS memory the image covers and its page geometry, set with the
        // size: the span tests run per surface for every command.
        std::uint64_t mem_lo = 0, mem_hi = 0;
        std::uint32_t per_row = 1, pw = 64, ph = 32;
        Plane plane;
        VkDescriptorSet set1[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};  // per frame slot
        VkDescriptorSet set2 = VK_NULL_HANDLE;                        // the surface as a texture
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
    // A texture decoded from GS memory into an image, keyed by everything
    // that decides its texels: the setup's texture words, the CLUT load and
    // the alpha expansion flag. Dropped when its memory is written.
    struct Texture {
        std::uint32_t key[11] = {};
        Image image;
        std::uint32_t lo = 0, hi = 0;    // GS memory span of the texels
        std::uint32_t clo = 0, chi = 0;  // and of the CLUT source, for indexed formats
        // The texels' 256-byte blocks as a grid of block cells (cw x ch
        // texels each), with the block id each cell had at decode time: a
        // sampled rectangle is checked by indexing, no block list or sort.
        std::uint32_t cw = 1, ch = 1, cells_x = 0, cells_y = 0;
        std::vector<std::uint32_t> cell_block, clut_blocks;
        std::vector<std::uint64_t> cell_id, clut_ids;
        std::vector<std::uint8_t> cell_done;  // decoded since the last palette change: cells decode on demand, where a batch samples
        std::uint32_t tw = 0, th = 0;
        bool from_surface = false;  // filled by image copies from a surface, not decoded
        VkDescriptorSet set2 = VK_NULL_HANDLE;
        std::uint64_t last_use = 0;
        std::list<std::unique_ptr<Texture>>::iterator self;  // position in textures_ (most recent first)
    };
    static constexpr std::size_t kMaxTextures = 1536;
    // Objects a recorded command buffer may still reference.
    struct Retired {
        Image image;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkDescriptorSet sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        VkDescriptorSet set2 = VK_NULL_HANDLE;
        std::uint64_t at = 0;  // submit counter when it may be destroyed
    };

    bool build_pipelines(VkShaderModule module);
    bool supported(const gpu::Command &cmd, const std::uint32_t *arena, const std::uint32_t *params, const std::uint32_t *&setups, std::uint32_t &zmode) const;
    std::uint32_t push_uniform(FrameScratch &f, const std::uint32_t *words);
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
    bool import_from_aliases(VkCommandBuffer cb, Surface &s, const std::vector<Rect> &rects);  // rects: runs of invalid tiles
    static bool all_valid(const Plane &p, const Rect &rect);
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

    // Textures.
    bool texture_cacheable(const std::uint32_t *setup) const;
    gpu::Span texture_span(const std::uint32_t *setup) const;
    // `fragments` (GS pixels the setup's primitives cover, 0 unknown) lets a
    // decode that would cost more than sampling memory be skipped when a
    // surface view can serve the texture; `color`/`depth` are the batch's.
    Texture *texture_for(VkCommandBuffer cb, VkDescriptorSet set0, const std::uint32_t *setup, std::uint32_t setup_offset, const std::uint32_t *prims, std::uint32_t count, std::uint32_t setup_index, std::uint32_t fragments = 0,
                         const Surface *color = nullptr, const Surface *depth = nullptr);
    // The texel rectangle a batch's primitives of `setup_index` can sample
    // (false when none do), and the blocks of memory holding a texel rectangle.
    bool texture_texels(const std::uint32_t *setup, const std::uint32_t *prims, std::uint32_t count, std::uint32_t setup_index, Rect &texels) const;
    Texture *texture_slot(VkCommandBuffer cb, const std::uint32_t *setup, bool from_surface, bool &created);
    // Resolves what set 2 binds for a textured setup: a surface sampled in
    // place when one mirrors the texture's buffer in the same layout, else
    // a cached texture. Null when neither can serve (memory sampling then).
    // The texture as a resident surface in the same layout (sampled in place,
    // or copied when the batch draws into it), or null.
    VkDescriptorSet surface_texture(VkCommandBuffer cb, VkDescriptorSet set0, const std::uint32_t *setup, const std::uint32_t *prims, std::uint32_t count, std::uint32_t setup_index, const Surface *color, const Surface *depth);
    VkDescriptorSet texture_set(VkCommandBuffer cb, VkDescriptorSet set0, const std::uint32_t *setup, std::uint32_t setup_offset, const std::uint32_t *prims, std::uint32_t count, std::uint32_t setup_index, const Surface *color,
                                const Surface *depth);
    bool texture_current(const Texture &t, const Rect *texels) const;  // texels null: CLUT only
    void decode_texture(VkCommandBuffer cb, VkDescriptorSet set0, Texture &t, std::uint32_t setup_offset, const Rect &rect);  // rect: texels to decode (cell aligned)
    void copy_surface_texels(VkCommandBuffer cb, VkDescriptorSet set0, Surface &src, Texture &t, const Rect &texels);
    void textures_written(const gpu::Span &span);

public:
    void clut_source(std::uint32_t id, gpu::Span span) {
        if (id >= clut_spans_.size()) clut_spans_.resize(id + 1);
        clut_spans_[id] = span;
    }
    void blocks_written(const std::uint32_t *blocks, std::size_t n, std::uint64_t id) {
        for (std::size_t i = 0; i < n; ++i) block_ids_[blocks[i] & 16383u] = id;
        ++block_gen_;
        if (upload_trace_ && n) std::fprintf(stderr, "[hw] f%llu upload id %016llx blocks %u..%u (%zu)\n", static_cast<unsigned long long>(frame_count()), static_cast<unsigned long long>(id), blocks[0] & 16383u, blocks[n - 1] & 16383u, n);
    }

private:
    std::vector<gpu::Span> clut_spans_;
    // Content id of every 256-byte block of GS memory as last uploaded;
    // a texture remembers a hash of its blocks' ids from when it was decoded.
    std::uint64_t block_ids_[16384] = {};
    std::uint64_t write_counter_ = 0;
    // textures_written fills the ids of every block of a span; the fill is
    // redundant while no texture has recorded ids and no upload has set ids
    // since the last fill covering the span (block_gen_ counts both).
    std::uint64_t block_gen_ = 0, fill_gen_ = ~0ull, fill_skips_ = 0;
    bool upload_trace_ = false;  // PS2_HW_TRACE window: print upload ids (hw.cpp sets it per frame)
    std::uint32_t fill_lo_ = 0, fill_hi_ = 0;
    void retire_texture(Texture &t);

    Context *ctx_ = nullptr;
    std::uint32_t scale_ = 1;
    VkDescriptorSetLayout set1_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkRenderPass pass_depth_rp_ = VK_NULL_HANDLE, pass_nodepth_ = VK_NULL_HANDLE;
    VkPipeline import_frame_ = VK_NULL_HANDLE, export_frame_ = VK_NULL_HANDLE, export_depth_ = VK_NULL_HANDLE, import_depth_ = VK_NULL_HANDLE;
    VkPipeline clut_view_ = VK_NULL_HANDLE;  // gs.wgsl load_clut with this module's surface_word: palettes read from a surface image
    bool command_done_ = false;
    std::uint64_t clut_views_ = 0;  // CLUT loads answered from a surface image instead of a write-back
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkShaderModule variant_modules_[8] = {};  // specialised fragment shaders, or null
    // Draw pipelines by state: depth mode and write, depth attachment, the
    // blend equation (0 when off) and the colour write mask, built on first use.
    struct PipelineKey {
        std::uint32_t ztst, zwrite, with_depth, blend, wmask, shader, variant;  // shader 0: hw_fs, 1: hw_fs_cached; variant 0-7 specialised module, 0xFF generic
        bool operator<(const PipelineKey &o) const { return std::tie(ztst, zwrite, with_depth, blend, wmask, shader, variant) < std::tie(o.ztst, o.zwrite, o.with_depth, o.blend, o.wmask, o.shader, o.variant); }
        bool operator==(const PipelineKey &o) const { return std::tie(ztst, zwrite, with_depth, blend, wmask, shader, variant) == std::tie(o.ztst, o.zwrite, o.with_depth, o.blend, o.wmask, o.shader, o.variant); }
    };
    std::map<PipelineKey, VkPipeline> draw_;
    struct RecentPipeline {
        PipelineKey key{};
        VkPipeline pipeline = VK_NULL_HANDLE;
    };
    RecentPipeline recent_[8];  // the last few keys, ahead of the map (a few per batch)
    unsigned recent_next_ = 0;
    VkPipeline pipeline_for(const PipelineKey &key);
    // Packs a GS blend (A, B, C, D selectors with FIX) into Vulkan factors;
    // 0 when the hardware cannot express it.
    static std::uint32_t blend_key(const std::uint32_t *setup, bool dual_source);
    Buffer depth_stage_;  // D32 bits of the rectangle in flight, shared by all surfaces
    Image dummy_;         // stands in for the image bindings of depth surfaces
    Image dummy_depth_;   // stands in for the depth image binding of colour surfaces
    Image decoded_;       // the decoded texture cache as a 2048-wide R32_UINT image
    VkDescriptorSet misc_set1_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};  // set 1 without a surface, for decodes
    VkPipeline decode_ = VK_NULL_HANDLE, decode_tex_ = VK_NULL_HANDLE;
    bool write_set1(VkDescriptorSet set, int slot, VkImageView surface_view, VkImageView depth_view);
    VkDescriptorSetLayout set2_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool2_ = VK_NULL_HANDLE;
    VkSampler samplers_[4] = {};
    std::list<std::unique_ptr<Texture>> textures_;  // most recently used first; the back is the eviction victim
    // Key -> entry, for the lookup every textured batch does per setup (a
    // linear scan of 1536 keys cost 12 ms per race frame on the Switch).
    struct KeyHash {
        std::size_t operator()(const std::array<std::uint32_t, 11> &k) const {
            std::uint64_t h = 0x9E3779B97F4A7C15ull;
            for (std::uint32_t v : k) h = (h ^ v) * 0xBF58476D1CE4E5B9ull;
            return static_cast<std::size_t>(h ^ (h >> 29));
        }
    };
    std::unordered_map<std::array<std::uint32_t, 11>, Texture *, KeyHash> texture_index_;
    std::map<std::uint32_t, std::vector<std::unique_ptr<Texture>>> free_by_size_;  // evicted entries by (tw << 8 | th), images kept for reuse
    std::uint64_t decoded_textures_ = 0, texture_hits_ = 0, surface_samples_ = 0, surface_copies_ = 0;
    std::uint64_t tex_drops_[5] = {};  // by memory_written, write_back, own draw, eviction, changed blocks
    int tex_drop_cause_ = 0;
    bool keep_textures_ = false;  // uploads: textures are kept or dropped by block ids, not spans
    std::vector<std::unique_ptr<Surface>> surfaces_;
    std::map<std::pair<VkImageView, VkImageView>, VkFramebuffer> framebuffers_;
    std::vector<Retired> retired_;
    std::uint64_t submits_ = 0, use_counter_ = 0;
    FrameScratch frames_[2];
    FrameScratch *frame_ = nullptr;
    // The surfaces' keys and memory bounds side by side: the per-command span
    // tests and lookups walk all surfaces, and the Surface objects (tile maps
    // included) are too large to walk by pointer without a cache miss each.
    struct Bound {
        std::uint64_t lo, hi;
        std::uint32_t bp, bw, psm;
        bool depth;
        Surface *s;
    };
    std::vector<Bound> bounds_;
    bool bounds_dirty_ = true;
    const std::vector<Bound> &bounds();
    int slot_ = 0;
    VkCommandBuffer pass_cb_ = VK_NULL_HANDLE;  // the open render pass, if any
    std::function<void()> pass_closed_hook_;
    std::vector<gpu::Span> deferred_reads_, deferred_writes_;
    std::vector<gpu::Span> held_reads_, held_writes_;  // spans of batches held back for other surfaces
    std::uint64_t held_batches_ = 0;
    std::uint32_t inverse32_words_ = 0;
    std::uint64_t surface_views_ = 0;  // uncached texture reads answered by a surface image
    // A surface view for one setup's memory reads (see hw.wgsl HwParams).
    struct View {
        std::uint32_t base = 0, bw = 0, words = 0, shift = 0;
    };
    // The resident 32/24-bit colour surface holding `span` whole (not this
    // batch's own), preferring one whose tiles there are all valid, or null.
    Surface *viewable_surface(const gpu::Span &span, const Surface *color, const Surface *depth);
    std::size_t deferred_count_ = 0;
    Surface *pass_color_ = nullptr, *pass_depth_ = nullptr;
    bool pass_with_depth_ = false;
    VkRect2D pass_area_{};
    std::uint64_t passes_ = 0, closes_[kCloseCount] = {};
    // PS2_HW_DEBUG 1048576: an occlusion query over each frame's draws counts
    // the fragments that reach the colour stage (after depth), for fill-rate maths.
    VkQueryPool occlusion_ = VK_NULL_HANDLE;
    bool query_open_ = false, query_used_[2] = {};
    std::uint64_t fragments_ = 0, fragment_frames_ = 0;
    std::uint64_t drawn_ = 0, fallback_ = 0;
    std::uint64_t imported_px_ = 0, exported_px_ = 0, evictions_ = 0, uniform_overflows_ = 0, alias_batches_ = 0;
    std::uint64_t fill_tiles_ = 0, fill_batches_ = 0;  // tiles a batch of opaque sprites covers whole: valid without an import
    std::uint64_t alias_copied_px_ = 0, decoded_from_surfaces_ = 0, decoded_texels_ = 0, decode_skips_ = 0;  // tiles filled from another surface's image instead of memory (import_from_aliases)
    // Why batches fell back, by the first reason met (see hw.cpp kReasons).
    mutable std::uint64_t reasons_[14] = {};
    mutable int last_reason_ = 0;
};

}
