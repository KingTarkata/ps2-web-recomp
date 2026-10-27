// The Vulkan implementation of gpu::Device: runs the same compute shaders
// as the wgpu device (translated to SPIR-V by naga at build time) on a
// device of our own. This is the parity stage of docs/gs-hw-renderer.md;
// the graphics pipelines that replace the raster dispatch come next and
// share these buffers.
#include "../gpu/device.hpp"
#include "context.hpp"
#include "hw.hpp"
#include "../gpu/cpu_timers.hpp"
#include "ps2/system.hpp"

#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern const std::uint32_t kGsSpvScale1[];
extern const std::size_t kGsSpvScale1_words;
extern const std::uint32_t kGsSpvScale2[];
extern const std::size_t kGsSpvScale2_words;
extern const std::uint32_t kGsSpvScale3[];
extern const std::size_t kGsSpvScale3_words;
extern const std::uint32_t kHwSpvScale1[];
extern const std::size_t kHwSpvScale1_words;
extern const std::uint32_t kHwSpvScale2[];
extern const std::size_t kHwSpvScale2_words;
extern const std::uint32_t kHwSpvScale3[];
extern const std::size_t kHwSpvScale3_words;
#define PS2_HW_VARIANT_DECL(n) extern const std::uint32_t kHwSpvScale1V##n[]; extern const std::size_t kHwSpvScale1V##n##_words;
PS2_HW_VARIANT_DECL(0) PS2_HW_VARIANT_DECL(1) PS2_HW_VARIANT_DECL(2) PS2_HW_VARIANT_DECL(3) PS2_HW_VARIANT_DECL(4) PS2_HW_VARIANT_DECL(5) PS2_HW_VARIANT_DECL(6) PS2_HW_VARIANT_DECL(7)

namespace ps2::gs::gpu {

namespace {

using vk::Buffer;
using vk::Context;

constexpr int kFrames = 2;              // command buffers in flight
constexpr std::size_t kArenaMinWords = 8u << 20;  // 32 MB of arena per frame slot before a frame outgrows it
constexpr std::uint32_t kProfileMax = 2048;

struct Frame {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Buffer params, data;                // host-visible, rewritten per submit
    VkDescriptorSet set = VK_NULL_HANDLE;
    bool submitted = false;
};

struct ReadSlot {
    Buffer staging;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    std::uint64_t bytes = 0;
    bool busy = false;
};

class VkDevice_ final : public Device {
public:
    void stats() override {
        if (hw_on_)
            std::fprintf(stderr, "[vk] hardware raster: %llu batches drawn, %llu through the compute rasterizer\n", static_cast<unsigned long long>(hw_.drawn_batches()),
                         static_cast<unsigned long long>(hw_.fallback_batches()));
        if (hw_on_) hw_.print_reasons();
        if (hw_on_) {
            std::fprintf(stderr, "[vk] passes closed by commands:");
            for (int i = 0; i < kPipelines; ++i)
                if (closes_by_pipeline_[i]) std::fprintf(stderr, " %s %llu", kEntryPoints[i], static_cast<unsigned long long>(closes_by_pipeline_[i]));
            std::fprintf(stderr, "\n");
        }
        cpu_timers::report(ps2::frame_count());
    }
    bool init(const Limits &limits, const std::vector<std::uint32_t> &tables) override;
    bool timestamps() const override { return ctx_.has_timestamps; }
    bool setup_unsupported(const std::uint32_t *setup) const override { return hw_on_ && !hw_.setup_supported(setup); }
    void clut_source(std::uint32_t id, gpu::Span span) override { hw_.clut_source(id, span); }
    void blocks_written(const std::uint32_t *blocks, std::size_t n, std::uint64_t id) override { hw_.blocks_written(blocks, n, id); }
    bool arena_storage(std::uint32_t *&words, std::size_t &capacity) override {
        // The slot the next submit uses: its last submission must be done
        // before the CPU writes into its data buffer (the wait submit() would
        // otherwise do, moved to the start of the frame).
        Frame &f = frames_[frame_];
        wait_frame(f);
        if (!ensure_frame_capacity(f, 0, kArenaMinWords)) return false;
        words = static_cast<std::uint32_t *>(f.data.mapped);
        capacity = static_cast<std::size_t>(f.data.size / 4);
        return words != nullptr;
    }
    void write(gpu::Buffer b, std::uint64_t offset, const void *data, std::uint64_t bytes) override;
    void submit(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words, bool present) override;
    void submit_profiled(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words,
                         std::vector<std::uint64_t> &ns) override;
    void read_back(gpu::Buffer b, std::uint64_t offset, std::uint64_t bytes, void *out) override;
    int begin_read(gpu::Buffer b, std::uint64_t offset, std::uint64_t bytes) override;
    int poll_read(int slot, void *out) override;
    int wait_read(int slot, void *out) override;
    void wait_idle() override { vk::vkQueueWaitIdle(ctx_.queue); }
    double busy_ms() override;
    // Presentation comes with the graphics stage; until then the runner's
    // SDL path copies frames out through display_async.
    bool attach_surface(const SurfaceTarget &) override { return false; }
    void resize_surface(std::uint32_t, std::uint32_t) override {}
    bool has_surface() const override { return false; }
    void set_blit(const Blit &) override {}
    std::uint32_t surface_width() const override { return 0; }
    std::uint32_t surface_height() const override { return 0; }

private:
    Buffer &buffer(gpu::Buffer b, Frame *f) {
        switch (b) {
        case gpu::Buffer::Vram: return vram_;
        case gpu::Buffer::Tables: return tables_;
        case gpu::Buffer::Params: return f->params;
        case gpu::Buffer::Data: return f->data;
        case gpu::Buffer::Clut: return clut_;
        case gpu::Buffer::Scratch: return scratch_;
        case gpu::Buffer::Cache: return cache_;
        case gpu::Buffer::Decoded: return decoded_;
        }
        return vram_;
    }
    bool build_pipelines(std::uint32_t scale);
    bool ensure_frame_capacity(Frame &f, std::size_t param_bytes, std::size_t arena_words);
    void write_descriptors(Frame &f);
    void wait_frame(Frame &f);
    void record(VkCommandBuffer cb, const std::vector<Command> &commands, Frame &f, VkQueryPool queries, std::uint32_t base, bool per_command, const std::uint32_t *arena, const void *params);
    bool copy_once(VkBuffer src, std::uint64_t src_offset, VkBuffer dst, std::uint64_t dst_offset, std::uint64_t bytes, VkFence fence, VkCommandBuffer &cb);

    Context ctx_;
    vk::HwRaster hw_;
    bool hw_on_ = false;
    std::uint64_t closes_by_pipeline_[kPipelines] = {};  // passes closed by a command the device could not hold back
    Buffer vram_, tables_, clut_, scratch_, cache_, decoded_, readback_, staging_;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkPipeline pipelines_[kPipelines] = {};
    Frame frames_[kFrames];
    int frame_ = 0;
    ReadSlot reads_[kReadSlots];
    VkQueryPool queries_ = VK_NULL_HANDLE, profile_queries_ = VK_NULL_HANDLE;
    std::uint64_t measured_ns_ = 0, measured_ = 0, total_passes_ = 0;
    bool query_pending_ = false;
    int query_frame_ = -1;
};

bool VkDevice_::copy_once(VkBuffer src, std::uint64_t src_offset, VkBuffer dst, std::uint64_t dst_offset, std::uint64_t bytes, VkFence fence, VkCommandBuffer &cb) {
    if (!cb) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = ctx_.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vk::vkAllocateCommandBuffers(ctx_.device, &ai, &cb) != VK_SUCCESS) return false;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk::vkBeginCommandBuffer(cb, &bi);
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vk::vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    VkBufferCopy region{src_offset, dst_offset, bytes};
    vk::vkCmdCopyBuffer(cb, src, dst, 1, &region);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vk::vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    vk::vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    return vk::vkQueueSubmit(ctx_.queue, 1, &si, fence) == VK_SUCCESS;
}

bool VkDevice_::build_pipelines(std::uint32_t scale) {
    const std::uint32_t *code = scale == 2 ? kGsSpvScale2 : scale == 3 ? kGsSpvScale3 : kGsSpvScale1;
    const std::size_t words = scale == 2 ? kGsSpvScale2_words : scale == 3 ? kGsSpvScale3_words : kGsSpvScale1_words;
    VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mci.codeSize = words * 4;
    mci.pCode = code;
    if (vk::vkCreateShaderModule(ctx_.device, &mci, nullptr, &module_) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] cannot create the GS shader module\n");
        return false;
    }
    VkDescriptorSetLayoutBinding bindings[8] = {};
    for (std::uint32_t i = 0; i < 8; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        // The hardware raster's vertex and fragment shaders read these buffers too.
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 8;
    lci.pBindings = bindings;
    if (vk::vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &layout_) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &layout_;
    if (vk::vkCreatePipelineLayout(ctx_.device, &pli, nullptr, &pipeline_layout_) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cis[kPipelines] = {};
    for (int i = 0; i < kPipelines; ++i) {
        cis[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cis[i].stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cis[i].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cis[i].stage.module = module_;
        cis[i].stage.pName = kEntryPoints[i];
        cis[i].layout = pipeline_layout_;
    }
    const VkResult r = vk::vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, kPipelines, cis, nullptr, pipelines_);
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] compute pipelines: %s\n", vk::result_name(r));
        return false;
    }
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 * kFrames}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kFrames}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = kFrames;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    if (vk::vkCreateDescriptorPool(ctx_.device, &dpi, nullptr, &pool_) != VK_SUCCESS) return false;
    for (Frame &f : frames_) {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &layout_;
        if (vk::vkAllocateDescriptorSets(ctx_.device, &dai, &f.set) != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = ctx_.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vk::vkAllocateCommandBuffers(ctx_.device, &ai, &f.cb) != VK_SUCCESS) return false;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vk::vkCreateFence(ctx_.device, &fci, nullptr, &f.fence) != VK_SUCCESS) return false;
        if (!ensure_frame_capacity(f, 64 * 1024, 1u << 20)) return false;
    }
    for (ReadSlot &r : reads_) {
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vk::vkCreateFence(ctx_.device, &fci, nullptr, &r.fence) != VK_SUCCESS) return false;
    }
    if (ctx_.has_timestamps) {
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = 2 * kFrames;
        if (vk::vkCreateQueryPool(ctx_.device, &qci, nullptr, &queries_) != VK_SUCCESS) queries_ = VK_NULL_HANDLE;
    }
    return true;
}

bool VkDevice_::ensure_frame_capacity(Frame &f, std::size_t param_bytes, std::size_t arena_words) {
    bool changed = false;
    const std::uint64_t data_bytes = std::max<std::uint64_t>(arena_words * 4, 16);
    if (data_bytes > f.data.size) {
        wait_frame(f);
        ctx_.destroy_buffer(f.data);
        const std::uint64_t size = std::max<std::uint64_t>(f.data.size * 2, (data_bytes + (4u << 20) - 1) & ~((4ull << 20) - 1));
        if (!ctx_.create_buffer(f.data, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, true)) return false;  // cached: the CPU reads the arena back (hardware path)
        changed = true;
    }
    const std::uint64_t wanted = std::max<std::uint64_t>(param_bytes, kParamStride);
    if (wanted > f.params.size) {
        wait_frame(f);
        ctx_.destroy_buffer(f.params);
        const std::uint64_t size = std::max<std::uint64_t>(f.params.size * 2, std::max<std::uint64_t>(wanted, 64 * 1024));
        if (!ctx_.create_buffer(f.params, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true)) return false;
        changed = true;
    }
    if (changed) write_descriptors(f);
    return true;
}

void VkDevice_::write_descriptors(Frame &f) {
    VkDescriptorBufferInfo infos[8];
    const Buffer *buffers[8] = {&vram_, &tables_, &f.params, &f.data, &clut_, &scratch_, &cache_, &decoded_};
    VkWriteDescriptorSet writes[8] = {};
    for (std::uint32_t i = 0; i < 8; ++i) {
        infos[i] = {buffers[i]->buffer, 0, i == 2 ? kParamStride : VK_WHOLE_SIZE};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = f.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vk::vkUpdateDescriptorSets(ctx_.device, 8, writes, 0, nullptr);
}

void VkDevice_::wait_frame(Frame &f) {
    cpu_timers::Scope timer(cpu_timers::kWaitFrame);
    if (!f.submitted) return;
    vk::vkWaitForFences(ctx_.device, 1, &f.fence, VK_TRUE, ~0ull);
    vk::vkResetFences(ctx_.device, 1, &f.fence);
    f.submitted = false;
}

bool VkDevice_::init(const Limits &limits, const std::vector<std::uint32_t> &tables) {
    if (!ctx_.init()) return false;
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!ctx_.create_buffer(vram_, limits.vram_bytes, storage, false) || !ctx_.create_buffer(tables_, tables.size() * 4, storage, false) ||
        !ctx_.create_buffer(clut_, limits.clut_bytes, storage, false) || !ctx_.create_buffer(scratch_, limits.scratch_bytes, storage, false) ||
        !ctx_.create_buffer(cache_, limits.cache_bytes, storage, false) || !ctx_.create_buffer(decoded_, limits.decoded_bytes, storage, false) ||
        !ctx_.create_buffer(readback_, limits.readback_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, true) ||
        !ctx_.create_buffer(staging_, 4u << 20, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true)) {
        std::fprintf(stderr, "[vk] cannot allocate the GS buffers\n");
        return false;
    }
    VkCommandBuffer cb = ctx_.begin_once();
    vk::vkCmdFillBuffer(cb, vram_.buffer, 0, VK_WHOLE_SIZE, 0);
    vk::vkCmdFillBuffer(cb, clut_.buffer, 0, VK_WHOLE_SIZE, 0);
    vk::vkCmdFillBuffer(cb, scratch_.buffer, 0, VK_WHOLE_SIZE, 0);
    vk::vkCmdFillBuffer(cb, decoded_.buffer, 0, VK_WHOLE_SIZE, 0);
    vk::vkCmdFillBuffer(cb, cache_.buffer, 0, VK_WHOLE_SIZE, 0);
    ctx_.end_once(cb);
    write(gpu::Buffer::Tables, 0, tables.data(), tables.size() * 4);
    if (!build_pipelines(limits.scale)) return false;
    const std::uint32_t *hw_code = limits.scale == 2 ? kHwSpvScale2 : limits.scale == 3 ? kHwSpvScale3 : kHwSpvScale1;
    const std::size_t hw_words = limits.scale == 2 ? kHwSpvScale2_words : limits.scale == 3 ? kHwSpvScale3_words : kHwSpvScale1_words;
    const std::uint32_t *const variants[8] = {kHwSpvScale1V0, kHwSpvScale1V1, kHwSpvScale1V2, kHwSpvScale1V3, kHwSpvScale1V4, kHwSpvScale1V5, kHwSpvScale1V6, kHwSpvScale1V7};
    const std::size_t variant_words[8] = {kHwSpvScale1V0_words, kHwSpvScale1V1_words, kHwSpvScale1V2_words, kHwSpvScale1V3_words, kHwSpvScale1V4_words, kHwSpvScale1V5_words, kHwSpvScale1V6_words, kHwSpvScale1V7_words};
    hw_on_ = !std::getenv("PS2_HW_OFF") && hw_.init(ctx_, hw_code, hw_words, limits.scale == 1 ? variants : nullptr, variant_words, layout_, limits.scale);
    std::fprintf(stderr, "[gpu] %s via Vulkan (%s)\n", ctx_.name.c_str(), hw_on_ ? "hardware rasterizer with compute fallback" : "compute rasterizer only");
    return true;
}

void VkDevice_::write(gpu::Buffer b, std::uint64_t offset, const void *data, std::uint64_t bytes) {
    cpu_timers::Scope timer(cpu_timers::kWrite);
    if (b == gpu::Buffer::Vram && hw_on_) hw_.drop_all();
    Buffer &target = buffer(b, &frames_[frame_]);
    if (target.mapped) {
        vk::vkQueueWaitIdle(ctx_.queue);
        std::memcpy(static_cast<std::uint8_t *>(target.mapped) + offset, data, bytes);
        return;
    }
    const auto *src = static_cast<const std::uint8_t *>(data);
    while (bytes) {
        const std::uint64_t chunk = std::min<std::uint64_t>(bytes, staging_.size);
        vk::vkQueueWaitIdle(ctx_.queue);  // the one staging buffer is reused
        std::memcpy(staging_.mapped, src, chunk);
        VkCommandBuffer cb = ctx_.begin_once();
        VkBufferCopy region{0, offset, chunk};
        vk::vkCmdCopyBuffer(cb, staging_.buffer, target.buffer, 1, &region);
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vk::vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        ctx_.end_once(cb);
        src += chunk;
        offset += chunk;
        bytes -= chunk;
    }
}

void VkDevice_::record(VkCommandBuffer cb, const std::vector<Command> &commands, Frame &f, VkQueryPool queries, std::uint32_t base, bool per_command, const std::uint32_t *arena, const void *params) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk::vkBeginCommandBuffer(cb, &bi);
    const std::uint32_t n = static_cast<std::uint32_t>(std::min<std::size_t>(commands.size(), kProfileMax));
    if (queries) {
        vk::vkCmdResetQueryPool(cb, queries, base, per_command ? n * 2 : 2);
        if (!per_command) vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, base);
    }
    // The host wrote params and data before the submit; make them visible.
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vk::vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
    // WebGPU orders dispatches of a pass against each other's storage writes;
    // Vulkan needs a barrier between every pair for the same guarantee.
    VkMemoryBarrier between{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    between.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    const VkPipelineStageFlags shader_stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    std::size_t skipped_from = ~static_cast<std::size_t>(0);  // first decode/snapshot skipped for the coming batch
    // Compute commands met while the hardware path has a render pass open
    // are recorded when that pass closes (HwRaster::close_pass runs the
    // hook): they touch buffers, not the pass's images, and the pass's draws
    // read none of their results, since a read that would see them (a
    // texture decode, a tile import, memory sampling) closes the pass first.
    // Their order among themselves is kept.
    std::vector<std::function<void()>> deferred;
    if (hw_on_)
        hw_.on_pass_closed([&] {
            // Recorded after the pass: each command's surface write-backs
            // and invalidations (before_command) right before it.
            for (std::size_t d = 0; d < deferred.size(); ++d) deferred[d]();
            deferred.clear();
            hw_.clear_deferred();
        });
    const auto dispatch = [&](const Command &k, std::uint32_t index, bool timed, bool top_written) {
        if (timed && !top_written) vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, index * 2);
        if (k.pipeline == kDecode && hw_on_) {
            hw_.dispatch_decode(cb, k, f.set);  // fills the cache image the fragment shader reads too
        } else {
            vk::vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[k.pipeline]);
            const std::uint32_t off = k.param * kParamStride;
            vk::vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &f.set, 1, &off);
            vk::vkCmdDispatch(cb, k.x, k.y, 1);
        }
        vk::vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, shader_stages, 0, 1, &between, 0, nullptr, 0, nullptr);
        if (timed) vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, index * 2 + 1);
    };
    for (std::uint32_t i = 0; i < commands.size(); ++i) {
        const Command &c = commands[i];
        // Every command writes both of its timestamps, drawn here or not: a
        // profiled submit waits on all of them.
        const bool timed = queries && per_command && i < n;
        bool top_written = false;
        if (c.pipeline == kRaster && hw_on_) {
            if (timed) vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, i * 2);
            top_written = timed;
            if (hw_.draw(cb, c, f.set, arena, static_cast<const std::uint32_t *>(params), f.data.buffer)) {
                skipped_from = ~static_cast<std::size_t>(0);
                if (!hw_.pass_open()) vk::vkCmdPipelineBarrier(cb, shader_stages, shader_stages, 0, 1, &between, 0, nullptr, 0, nullptr);  // an open pass is closed, with its barrier, by whatever comes next
                if (timed) vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, i * 2 + 1);
                continue;
            }
        }
        if (hw_on_) {
            // The per-submit decodes and the snapshot the batching layer put
            // before a raster batch are skipped when the hardware path will
            // draw it from its own texture cache and surfaces. If it then
            // declines the batch after all, they are run here first.
            if (c.pipeline == kDecode || c.pipeline == kSnapshot) {
                std::size_t next = i + 1;
                while (next < commands.size() && (commands[next].pipeline == kDecode || commands[next].pipeline == kSnapshot)) ++next;
                if (next < commands.size() && commands[next].pipeline == kRaster && hw_.decodes_unneeded(commands[next], arena, static_cast<const std::uint32_t *>(params))) {
                    if (skipped_from > i) skipped_from = i;
                    if (timed) {
                        vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, i * 2);
                        vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, i * 2 + 1);
                    }
                    continue;
                }
            }
            if (c.pipeline == kRaster && skipped_from < i) {
                // Declined after all: replay what was skipped, in order.
                for (std::size_t k = skipped_from; k < i; ++k) {
                    hw_.before_command(cb, commands[k], f.set, arena, static_cast<const std::uint32_t *>(params), false);
                    hw_.close_pass(cb);
                    dispatch(commands[k], static_cast<std::uint32_t>(k), false, false);
                }
            }
            skipped_from = ~static_cast<std::size_t>(0);
            if (hw_.pass_open() && c.pipeline != kRaster && c.ranged) {
                hw_.note_deferred(c);
                deferred.push_back([&, c, i, timed] {
                    hw_.before_command(cb, c, f.set, arena, static_cast<const std::uint32_t *>(params), false);
                    dispatch(c, i, timed, false);
                });
                continue;
            }
            if (hw_.pass_open()) {
                ++closes_by_pipeline_[c.pipeline];
                hw_.close_pass(cb);  // a compute-drawn batch or an unranged command: everything held back lands first
            }
            // The resident surfaces may hold memory this command reads or writes.
            hw_.before_command(cb, c, f.set, arena, static_cast<const std::uint32_t *>(params), false);
        }
        dispatch(c, i, timed, top_written);
    }
    if (hw_on_) {
        hw_.end_frame(cb);  // closes the pass, which records what was held back
        hw_.on_pass_closed(nullptr);
    }
    if (queries && !per_command) vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, base + 1);
    // Make the results visible to transfers (read-backs) and the host.
    VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    done.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    done.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vk::vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &done, 0, nullptr, 0, nullptr);
    vk::vkEndCommandBuffer(cb);
}

void VkDevice_::submit(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words, bool) {
    if (commands.empty()) return;
    cpu_timers::Scope timer(cpu_timers::kSubmit);
    // Collect the GPU time of the frame measured two submits ago.
    if (query_pending_ && queries_) {
        Frame &q = frames_[query_frame_];
        if (q.submitted && vk::vkGetFenceStatus(ctx_.device, q.fence) == VK_SUCCESS) {
            std::uint64_t t[2];
            if (vk::vkGetQueryPoolResults(ctx_.device, queries_, static_cast<std::uint32_t>(query_frame_) * 2, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && t[1] > t[0]) {
                measured_ns_ += static_cast<std::uint64_t>(static_cast<double>(t[1] - t[0]) * ctx_.timestamp_period);
                ++measured_;
            }
            query_pending_ = false;
        }
    }
    Frame &f = frames_[frame_];
    wait_frame(f);
    if (!ensure_frame_capacity(f, param_bytes, arena_words)) return;
    if (arena != static_cast<const std::uint32_t *>(f.data.mapped)) std::memcpy(f.data.mapped, arena, arena_words * 4);  // built elsewhere (first frame, or outgrew the buffer)
    ctx_.flush(f.data, 0, arena_words * 4);
    std::memcpy(f.params.mapped, params, param_bytes);
    const bool measure = queries_ && !query_pending_;
    hw_.begin_frame(frame_);
    record(f.cb, commands, f, measure ? queries_ : VK_NULL_HANDLE, static_cast<std::uint32_t>(frame_) * 2, false, arena, params);
    if (measure) {
        query_pending_ = true;
        query_frame_ = frame_;
    }
    ++total_passes_;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cb;
    if (vk::vkQueueSubmit(ctx_.queue, 1, &si, f.fence) != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] submit failed\n");
        return;
    }
    f.submitted = true;
    frame_ = (frame_ + 1) % kFrames;
}

void VkDevice_::submit_profiled(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words,
                                std::vector<std::uint64_t> &ns) {
    ns.clear();
    if (commands.empty()) return;
    if (!profile_queries_) {
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = kProfileMax * 2;
        if (vk::vkCreateQueryPool(ctx_.device, &qci, nullptr, &profile_queries_) != VK_SUCCESS) return;
    }
    Frame &f = frames_[frame_];
    wait_frame(f);
    if (!ensure_frame_capacity(f, param_bytes, arena_words)) return;
    if (arena != static_cast<const std::uint32_t *>(f.data.mapped)) std::memcpy(f.data.mapped, arena, arena_words * 4);  // built elsewhere (first frame, or outgrew the buffer)
    ctx_.flush(f.data, 0, arena_words * 4);
    std::memcpy(f.params.mapped, params, param_bytes);
    hw_.begin_frame(frame_);
    record(f.cb, commands, f, profile_queries_, 0, true, arena, params);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cb;
    if (vk::vkQueueSubmit(ctx_.queue, 1, &si, f.fence) != VK_SUCCESS) return;
    f.submitted = true;
    wait_frame(f);
    const std::uint32_t n = static_cast<std::uint32_t>(std::min<std::size_t>(commands.size(), kProfileMax));
    std::vector<std::uint64_t> t(n * 2);
    if (vk::vkGetQueryPoolResults(ctx_.device, profile_queries_, 0, n * 2, t.size() * 8, t.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        ns.resize(n);
        for (std::uint32_t i = 0; i < n; ++i)
            ns[i] = t[i * 2 + 1] > t[i * 2] ? static_cast<std::uint64_t>(static_cast<double>(t[i * 2 + 1] - t[i * 2]) * ctx_.timestamp_period) : 0;
    }
    frame_ = (frame_ + 1) % kFrames;
}

void VkDevice_::read_back(gpu::Buffer b, std::uint64_t offset, std::uint64_t bytes, void *out) {
    cpu_timers::Scope timer(cpu_timers::kReadBack);
    if (b == gpu::Buffer::Vram && hw_on_) hw_.flush_sync(frames_[(frame_ + kFrames - 1) % kFrames].set, {static_cast<std::uint32_t>(offset), static_cast<std::uint32_t>(offset + bytes)});
    Buffer &src = buffer(b, &frames_[(frame_ + kFrames - 1) % kFrames]);
    auto *dst = static_cast<std::uint8_t *>(out);
    while (bytes) {
        const std::uint64_t chunk = std::min<std::uint64_t>(bytes, readback_.size);
        VkCommandBuffer cb = ctx_.begin_once();
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vk::vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        VkBufferCopy region{offset, 0, chunk};
        vk::vkCmdCopyBuffer(cb, src.buffer, readback_.buffer, 1, &region);
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vk::vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        ctx_.end_once(cb);
        ctx_.invalidate(readback_, 0, chunk);
        std::memcpy(dst, readback_.mapped, chunk);
        dst += chunk;
        offset += chunk;
        bytes -= chunk;
    }
}

int VkDevice_::begin_read(gpu::Buffer b, std::uint64_t offset, std::uint64_t bytes) {
    cpu_timers::Scope timer(cpu_timers::kReadBack);
    int slot = -1;
    for (int i = 0; i < kReadSlots; ++i)
        if (!reads_[i].busy) slot = i;
    if (slot < 0) return -1;
    ReadSlot &r = reads_[slot];
    if (r.staging.size < bytes) {
        ctx_.destroy_buffer(r.staging);
        if (!ctx_.create_buffer(r.staging, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, true)) return -1;
    }
    vk::vkResetFences(ctx_.device, 1, &r.fence);
    if (b == gpu::Buffer::Vram && hw_on_) hw_.flush_sync(frames_[(frame_ + kFrames - 1) % kFrames].set, {static_cast<std::uint32_t>(offset), static_cast<std::uint32_t>(offset + bytes)});
    Buffer &src = buffer(b, &frames_[(frame_ + kFrames - 1) % kFrames]);
    if (!copy_once(src.buffer, offset, r.staging.buffer, 0, bytes, r.fence, r.cb)) return -1;
    r.bytes = bytes;
    r.busy = true;
    return slot;
}

int VkDevice_::poll_read(int slot, void *out) {
    if (slot < 0 || slot >= kReadSlots || !reads_[slot].busy) return -1;
    ReadSlot &r = reads_[slot];
    const VkResult s = vk::vkGetFenceStatus(ctx_.device, r.fence);
    if (s == VK_NOT_READY) return 0;
    r.busy = false;
    if (s != VK_SUCCESS) return -1;
    ctx_.invalidate(r.staging, 0, r.bytes);
    std::memcpy(out, r.staging.mapped, r.bytes);
    return 1;
}

int VkDevice_::wait_read(int slot, void *out) {
    cpu_timers::Scope timer(cpu_timers::kReadBack);
    if (slot < 0 || slot >= kReadSlots || !reads_[slot].busy) return -1;
    vk::vkWaitForFences(ctx_.device, 1, &reads_[slot].fence, VK_TRUE, ~0ull);
    return poll_read(slot, out);
}

double VkDevice_::busy_ms() {
    if (!measured_) return 0.0;
    return static_cast<double>(measured_ns_) / static_cast<double>(measured_) * static_cast<double>(total_passes_) / 1e6;
}

}

Device *make_vk_device() { return new VkDevice_(); }

}
