// The Vulkan implementation of gpu::Device: runs the same compute shaders
// as the wgpu device (translated to SPIR-V by naga at build time) on a
// device of our own. This is the parity stage of docs/gs-hw-renderer.md;
// the graphics pipelines that replace the raster dispatch come next and
// share these buffers.
#include "../gpu/device.hpp"
#include "context.hpp"
#include "ps2/system.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

extern const std::uint32_t kGsSpvScale1[];
extern const std::size_t kGsSpvScale1_words;
extern const std::uint32_t kGsSpvScale2[];
extern const std::size_t kGsSpvScale2_words;
extern const std::uint32_t kGsSpvScale3[];
extern const std::size_t kGsSpvScale3_words;

namespace ps2::gs::gpu {

namespace {

using vk::Buffer;
using vk::Context;

constexpr int kFrames = 2;              // command buffers in flight
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
    bool init(const Limits &limits, const std::vector<std::uint32_t> &tables) override;
    bool timestamps() const override { return ctx_.has_timestamps; }
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
    void record(VkCommandBuffer cb, const std::vector<Command> &commands, Frame &f, VkQueryPool queries, std::uint32_t base, bool per_command);
    bool copy_once(VkBuffer src, std::uint64_t src_offset, VkBuffer dst, std::uint64_t dst_offset, std::uint64_t bytes, VkFence fence, VkCommandBuffer &cb);

    Context ctx_;
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
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
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
        if (!ctx_.create_buffer(f.data, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true)) return false;
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
        !ctx_.create_buffer(readback_, limits.readback_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true) ||
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
    std::fprintf(stderr, "[gpu] %s via Vulkan (hardware backend, compute parity stage)\n", ctx_.name.c_str());
    return true;
}

void VkDevice_::write(gpu::Buffer b, std::uint64_t offset, const void *data, std::uint64_t bytes) {
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

void VkDevice_::record(VkCommandBuffer cb, const std::vector<Command> &commands, Frame &f, VkQueryPool queries, std::uint32_t base, bool per_command) {
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
    between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    Pipeline current = kPipelines;
    for (std::uint32_t i = 0; i < commands.size(); ++i) {
        const Command &c = commands[i];
        if (c.pipeline != current) {
            vk::vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[c.pipeline]);
            current = c.pipeline;
        }
        const std::uint32_t offset = c.param * kParamStride;
        vk::vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &f.set, 1, &offset);
        if (queries && per_command && i < n) vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, i * 2);
        vk::vkCmdDispatch(cb, c.x, c.y, 1);
        vk::vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &between, 0, nullptr, 0, nullptr);
        if (queries && per_command && i < n) vk::vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, i * 2 + 1);
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
    std::memcpy(f.data.mapped, arena, arena_words * 4);
    std::memcpy(f.params.mapped, params, param_bytes);
    const bool measure = queries_ && !query_pending_;
    record(f.cb, commands, f, measure ? queries_ : VK_NULL_HANDLE, static_cast<std::uint32_t>(frame_) * 2, false);
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
    std::memcpy(f.data.mapped, arena, arena_words * 4);
    std::memcpy(f.params.mapped, params, param_bytes);
    record(f.cb, commands, f, profile_queries_, 0, true);
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
        std::memcpy(dst, readback_.mapped, chunk);
        dst += chunk;
        offset += chunk;
        bytes -= chunk;
    }
}

int VkDevice_::begin_read(gpu::Buffer b, std::uint64_t offset, std::uint64_t bytes) {
    int slot = -1;
    for (int i = 0; i < kReadSlots; ++i)
        if (!reads_[i].busy) slot = i;
    if (slot < 0) return -1;
    ReadSlot &r = reads_[slot];
    if (r.staging.size < bytes) {
        ctx_.destroy_buffer(r.staging);
        if (!ctx_.create_buffer(r.staging, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true)) return -1;
    }
    vk::vkResetFences(ctx_.device, 1, &r.fence);
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
    std::memcpy(out, r.staging.mapped, r.bytes);
    return 1;
}

int VkDevice_::wait_read(int slot, void *out) {
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
