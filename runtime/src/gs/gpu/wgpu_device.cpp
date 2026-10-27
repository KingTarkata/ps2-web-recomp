// The wgpu-native (and in the browser, Dawn) implementation of Device: the
// code that used to sit inside gpu.cpp, unchanged in behaviour.
#include "device.hpp"
#include "ps2/system.hpp"

#include <webgpu/webgpu.h>
#ifndef __EMSCRIPTEN__
#include <webgpu/wgpu.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern const char kGsShader[];
extern const char kBlitShader[];

namespace ps2::gs::gpu {

namespace {

WGPUStringView view(const char *s) { return WGPUStringView{s, WGPU_STRLEN}; }

void on_error(WGPUDevice const *, WGPUErrorType type, WGPUStringView message, void *, void *) {
    std::fprintf(stderr, "[gpu] error %d: %.*s\n", static_cast<int>(type), static_cast<int>(message.length == WGPU_STRLEN ? std::strlen(message.data) : message.length), message.data);
}

struct ReadSlot {
    WGPUBuffer buffer = nullptr;
    std::uint64_t capacity = 0, bytes = 0;
    bool busy = false, ready = false, failed = false;
};

class WgpuDevice final : public Device {
public:
    bool init(const Limits &limits, const std::vector<std::uint32_t> &tables) override;
    bool timestamps() const override { return timestamps_; }
    void write(Buffer b, std::uint64_t offset, const void *data, std::uint64_t bytes) override { wgpuQueueWriteBuffer(queue_, buffer(b), offset, data, bytes); }
    void submit(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words, bool present) override;
    void submit_profiled(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words,
                         std::vector<std::uint64_t> &ns) override;
    void read_back(Buffer b, std::uint64_t offset, std::uint64_t bytes, void *out) override;
    int begin_read(Buffer b, std::uint64_t offset, std::uint64_t bytes) override;
    int poll_read(int slot, void *out) override;
    int wait_read(int slot, void *out) override;
    void wait_idle() override;
    double busy_ms() override;
    bool attach_surface(const SurfaceTarget &target) override;
    void resize_surface(std::uint32_t width, std::uint32_t height) override;
    bool has_surface() const override { return surface_ != nullptr; }
    void set_blit(const Blit &blit) override { wgpuQueueWriteBuffer(queue_, blit_params_, 0, blit.words, sizeof blit.words); }
    std::uint32_t surface_width() const override { return surface_width_; }
    std::uint32_t surface_height() const override { return surface_height_; }

private:
    WGPUBuffer buffer(Buffer b) const {
        switch (b) {
        case Buffer::Vram: return vram_;
        case Buffer::Tables: return tables_;
        case Buffer::Params: return params_;
        case Buffer::Data: return data_;
        case Buffer::Clut: return clut_;
        case Buffer::Scratch: return scratch_;
        case Buffer::Cache: return cache_;
        case Buffer::Decoded: return decoded_;
        }
        return nullptr;
    }
    void pump();
    void wait_until(const bool &flag) {
        while (!flag) pump();
    }
    WGPUBuffer make_buffer(std::uint64_t size, WGPUBufferUsage usage) {
        WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
        d.size = size;
        d.usage = usage;
        return wgpuDeviceCreateBuffer(device_, &d);
    }
    void rebuild_group();
    void ensure_capacity(std::size_t param_bytes, std::size_t arena_words);
    void upload_inputs(const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words) {
        ensure_capacity(param_bytes, arena_words);
        wgpuQueueWriteBuffer(queue_, data_, 0, arena, arena_words * 4);
        wgpuQueueWriteBuffer(queue_, params_, 0, params, param_bytes);
    }
    void configure_surface();

    WGPUInstance instance_ = nullptr;
    WGPUAdapter adapter_ = nullptr;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    WGPUBuffer vram_ = nullptr, tables_ = nullptr, params_ = nullptr, data_ = nullptr, clut_ = nullptr, scratch_ = nullptr, cache_ = nullptr, decoded_ = nullptr, readback_ = nullptr;
    std::uint64_t params_size_ = 0, data_size_ = 0;
    WGPUBindGroupLayout layout_ = nullptr;
    WGPUPipelineLayout pipeline_layout_ = nullptr;
    WGPUBindGroup group_ = nullptr;
    WGPUComputePipeline pipelines_[kPipelines] = {};
    std::uint32_t scale_ = 1;

    bool timestamps_ = false;
    WGPUQuerySet queries_ = nullptr;
    WGPUBuffer query_resolve_ = nullptr, query_read_ = nullptr;
    bool query_pending_ = false, query_ready_ = false;
    std::uint64_t measured_ns_ = 0, measured_ = 0, total_passes_ = 0;
    WGPUQuerySet profile_queries_ = nullptr;
    WGPUBuffer profile_resolve_ = nullptr, profile_read_ = nullptr;

    ReadSlot reads_[kReadSlots];

    WGPUSurface surface_ = nullptr;
    WGPUTextureFormat surface_format_ = WGPUTextureFormat_Undefined;
    WGPUPresentMode present_mode_ = WGPUPresentMode_Fifo;
    std::uint32_t surface_width_ = 0, surface_height_ = 0;
    WGPURenderPipeline blit_ = nullptr;
    WGPUBindGroup blit_group_ = nullptr;
    WGPUBuffer blit_params_ = nullptr;
};

void WgpuDevice::pump() {
#ifdef __EMSCRIPTEN__
    browser_yield();
    wgpuInstanceProcessEvents(instance_);
#else
    if (device_) wgpuDevicePoll(device_, true, nullptr);
    else wgpuInstanceProcessEvents(instance_);
#endif
}

void WgpuDevice::rebuild_group() {
    if (group_) wgpuBindGroupRelease(group_);
    WGPUBindGroupEntry e[8] = {};
    WGPUBuffer buffers[8] = {vram_, tables_, params_, data_, clut_, scratch_, cache_, decoded_};
    for (int i = 0; i < 8; ++i) {
        e[i] = WGPU_BIND_GROUP_ENTRY_INIT;
        e[i].binding = static_cast<std::uint32_t>(i);
        e[i].buffer = buffers[i];
        e[i].size = i == 2 ? kParamStride : WGPU_WHOLE_SIZE;
    }
    WGPUBindGroupDescriptor d = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    d.layout = layout_;
    d.entryCount = 8;
    d.entries = e;
    group_ = wgpuDeviceCreateBindGroup(device_, &d);
}

void WgpuDevice::ensure_capacity(std::size_t param_bytes, std::size_t arena_words) {
    bool changed = false;
    const std::uint64_t data_bytes = std::max<std::uint64_t>(arena_words * 4, 16);
    if (data_bytes > data_size_) {
        if (data_) wgpuBufferRelease(data_);
        data_size_ = std::max<std::uint64_t>(data_size_ * 2, (data_bytes + (4u << 20) - 1) & ~((4ull << 20) - 1));
        data_ = make_buffer(data_size_, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
        changed = true;
    }
    const std::uint64_t wanted = std::max<std::uint64_t>(param_bytes, kParamStride);
    if (wanted > params_size_) {
        if (params_) wgpuBufferRelease(params_);
        params_size_ = std::max<std::uint64_t>(params_size_ * 2, std::max<std::uint64_t>(wanted, 64 * 1024));
        params_ = make_buffer(params_size_, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
        changed = true;
    }
    if (changed || !group_) rebuild_group();
}

bool WgpuDevice::init(const Limits &limits, const std::vector<std::uint32_t> &tables) {
#ifdef __EMSCRIPTEN__
    instance_ = wgpuCreateInstance(nullptr);
#else
    WGPUInstanceExtras extras = {};
    extras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
    extras.backends = WGPUInstanceBackend_Primary;
    WGPUInstanceDescriptor id = WGPU_INSTANCE_DESCRIPTOR_INIT;
    id.nextInChain = &extras.chain;
    instance_ = wgpuCreateInstance(&id);
#endif
    if (!instance_) return false;
    WGPUAdapter adapter = nullptr;
    bool adapter_done = false;
    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.powerPreference = std::getenv("PS2_GPU_LOW_POWER") ? WGPUPowerPreference_LowPower : WGPUPowerPreference_HighPerformance;
    options.forceFallbackAdapter = std::getenv("PS2_GPU_FALLBACK") != nullptr;
    WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    acb.mode = WGPUCallbackMode_AllowSpontaneous;
    acb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter a, WGPUStringView, void *out, void *done) {
        if (status == WGPURequestAdapterStatus_Success) *static_cast<WGPUAdapter *>(out) = a;
        *static_cast<bool *>(done) = true;
    };
    acb.userdata1 = &adapter;
    acb.userdata2 = &adapter_done;
    wgpuInstanceRequestAdapter(instance_, &options, acb);
    while (!adapter_done) pump();
    if (!adapter) {
        std::fprintf(stderr, "[gpu] no adapter\n");
        return false;
    }
    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success) {
        std::fprintf(stderr, "[gpu] %.*s (%.*s)\n", static_cast<int>(info.device.length), info.device.data, static_cast<int>(info.description.length), info.description.data);
        wgpuAdapterInfoFreeMembers(info);
    }
    WGPULimits supported = WGPU_LIMITS_INIT;
    wgpuAdapterGetLimits(adapter, &supported);
    WGPULimits required = WGPU_LIMITS_INIT;
    required.maxStorageBufferBindingSize = supported.maxStorageBufferBindingSize;
    required.maxBufferSize = supported.maxBufferSize;
    WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
    dd.requiredLimits = &required;
    const WGPUFeatureName timestamp_feature = WGPUFeatureName_TimestampQuery;
    timestamps_ = wgpuAdapterHasFeature(adapter, WGPUFeatureName_TimestampQuery);
    if (timestamps_) {
        dd.requiredFeatureCount = 1;
        dd.requiredFeatures = &timestamp_feature;
    }
    dd.uncapturedErrorCallbackInfo.callback = on_error;
    bool device_done = false;
    WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    dcb.mode = WGPUCallbackMode_AllowSpontaneous;
    dcb.callback = [](WGPURequestDeviceStatus status, WGPUDevice d, WGPUStringView message, void *out, void *done) {
        if (status == WGPURequestDeviceStatus_Success) *static_cast<WGPUDevice *>(out) = d;
        else std::fprintf(stderr, "[gpu] device: %.*s\n", static_cast<int>(message.length), message.data);
        *static_cast<bool *>(done) = true;
    };
    dcb.userdata1 = &device_;
    dcb.userdata2 = &device_done;
    wgpuAdapterRequestDevice(adapter, &dd, dcb);
    while (!device_done) pump();
    adapter_ = adapter;
    if (!device_) return false;
    queue_ = wgpuDeviceGetQueue(device_);

    scale_ = limits.scale;
    vram_ = make_buffer(limits.vram_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
    tables_ = make_buffer(tables.size() * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    wgpuQueueWriteBuffer(queue_, tables_, 0, tables.data(), tables.size() * 4);
    clut_ = make_buffer(limits.clut_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
    scratch_ = make_buffer(limits.scratch_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    cache_ = make_buffer(limits.cache_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    decoded_ = make_buffer(limits.decoded_bytes, WGPUBufferUsage_Storage);
    readback_ = make_buffer(limits.readback_bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    if (timestamps_) {
        WGPUQuerySetDescriptor qd = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        qd.type = WGPUQueryType_Timestamp;
        qd.count = 2;
        queries_ = wgpuDeviceCreateQuerySet(device_, &qd);
        query_resolve_ = make_buffer(16, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
        query_read_ = make_buffer(16, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    }

    WGPUBindGroupLayoutEntry le[8] = {};
    const WGPUBufferBindingType types[8] = {WGPUBufferBindingType_Storage, WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_Uniform,
                                            WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_Storage, WGPUBufferBindingType_Storage,
                                            WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_Storage};
    for (int i = 0; i < 8; ++i) {
        le[i] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        le[i].binding = static_cast<std::uint32_t>(i);
        le[i].visibility = WGPUShaderStage_Compute;
        le[i].buffer.type = types[i];
        le[i].buffer.hasDynamicOffset = i == 2;
        le[i].buffer.minBindingSize = i == 2 ? kParamStride : 0;
    }
    WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    ld.entryCount = 8;
    ld.entries = le;
    layout_ = wgpuDeviceCreateBindGroupLayout(device_, &ld);
    WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &layout_;
    pipeline_layout_ = wgpuDeviceCreatePipelineLayout(device_, &pld);

    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code = view(kGsShader);
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &source.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device_, &md);
    for (int i = 0; i < kPipelines; ++i) {
        WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        cd.layout = pipeline_layout_;
        cd.compute.module = module;
        cd.compute.entryPoint = view(kEntryPoints[i]);
        WGPUConstantEntry constant = WGPU_CONSTANT_ENTRY_INIT;
        constant.key = view("SCALE");
        constant.value = static_cast<double>(scale_);
        cd.compute.constantCount = 1;
        cd.compute.constants = &constant;
        pipelines_[i] = wgpuDeviceCreateComputePipeline(device_, &cd);
        if (!pipelines_[i]) return false;
    }
    wgpuShaderModuleRelease(module);
    ensure_capacity(0, 0);
    return true;
}

void WgpuDevice::submit_profiled(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words,
                                 std::vector<std::uint64_t> &ns) {
    constexpr std::uint32_t kMax = 2048;
    if (!profile_queries_) {
        WGPUQuerySetDescriptor qd = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        qd.type = WGPUQueryType_Timestamp;
        qd.count = kMax * 2;
        profile_queries_ = wgpuDeviceCreateQuerySet(device_, &qd);
        profile_resolve_ = make_buffer(kMax * 16, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
        profile_read_ = make_buffer(kMax * 16, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    }
    upload_inputs(params, param_bytes, arena, arena_words);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    const std::uint32_t n = static_cast<std::uint32_t>(std::min<std::size_t>(commands.size(), kMax));
    for (std::uint32_t i = 0; i < commands.size(); ++i) {
        const Command &c = commands[i];
        WGPUPassTimestampWrites writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
        writes.querySet = profile_queries_;
        writes.beginningOfPassWriteIndex = i * 2;
        writes.endOfPassWriteIndex = i * 2 + 1;
        WGPUComputePassDescriptor cd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        if (i < n) cd.timestampWrites = &writes;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &cd);
        wgpuComputePassEncoderSetPipeline(pass, pipelines_[c.pipeline]);
        const std::uint32_t offset = c.param * kParamStride;
        wgpuComputePassEncoderSetBindGroup(pass, 0, group_, 1, &offset);
        wgpuComputePassEncoderDispatchWorkgroups(pass, c.x, c.y, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    if (n) {
        wgpuCommandEncoderResolveQuerySet(enc, profile_queries_, 0, n * 2, profile_resolve_, 0);
        wgpuCommandEncoderCopyBufferToBuffer(enc, profile_resolve_, 0, profile_read_, 0, n * 16);
    }
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    ns.assign(n, 0);
    if (n) {
        bool done = false;
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowProcessEvents;
        cb.callback = [](WGPUMapAsyncStatus, WGPUStringView, void *flag, void *) { *static_cast<bool *>(flag) = true; };
        cb.userdata1 = &done;
        wgpuBufferMapAsync(profile_read_, WGPUMapMode_Read, 0, n * 16, cb);
        wait_until(done);
        const auto *t = static_cast<const std::uint64_t *>(wgpuBufferGetConstMappedRange(profile_read_, 0, n * 16));
        for (std::uint32_t i = 0; t && i < n; ++i) ns[i] = t[i * 2 + 1] > t[i * 2] ? t[i * 2 + 1] - t[i * 2] : 0;
        wgpuBufferUnmap(profile_read_);
    }
}

void WgpuDevice::submit(const std::vector<Command> &commands, const void *params, std::size_t param_bytes, const std::uint32_t *arena, std::size_t arena_words, bool present) {
    upload_inputs(params, param_bytes, arena, arena_words);
    if (query_ready_) {
        const void *mapped = wgpuBufferGetConstMappedRange(query_read_, 0, 16);
        if (mapped) {
            std::uint64_t t[2];
            std::memcpy(t, mapped, sizeof t);
            if (t[1] > t[0]) {
                measured_ns_ += t[1] - t[0];
                ++measured_;
            }
        }
        wgpuBufferUnmap(query_read_);
        query_ready_ = query_pending_ = false;
    }
    const bool measure = timestamps_ && !query_pending_;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    WGPUPassTimestampWrites writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    writes.querySet = queries_;
    writes.beginningOfPassWriteIndex = 0;
    writes.endOfPassWriteIndex = 1;
    WGPUComputePassDescriptor cd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    if (measure) cd.timestampWrites = &writes;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &cd);
    Pipeline current = kPipelines;
    ++total_passes_;
    for (const Command &c : commands) {
        if (c.pipeline != current) {
            wgpuComputePassEncoderSetPipeline(pass, pipelines_[c.pipeline]);
            current = c.pipeline;
        }
        const std::uint32_t offset = c.param * kParamStride;
        wgpuComputePassEncoderSetBindGroup(pass, 0, group_, 1, &offset);
        wgpuComputePassEncoderDispatchWorkgroups(pass, c.x, c.y, 1);
    }
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    if (measure) {
        wgpuCommandEncoderResolveQuerySet(enc, queries_, 0, 2, query_resolve_, 0);
        wgpuCommandEncoderCopyBufferToBuffer(enc, query_resolve_, 0, query_read_, 0, 16);
    }
    WGPUSurfaceTexture st = WGPU_SURFACE_TEXTURE_INIT;
    WGPUTextureView target = nullptr;
    if (present && surface_) {
        wgpuSurfaceGetCurrentTexture(surface_, &st);
        if (st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal || st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            target = wgpuTextureCreateView(st.texture, nullptr);
        } else {
            if (st.texture) wgpuTextureRelease(st.texture);
            st.texture = nullptr;
            configure_surface();
        }
    }
    if (target) {
        WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        color.view = target;
        color.loadOp = WGPULoadOp_Clear;
        color.storeOp = WGPUStoreOp_Store;
        color.clearValue = WGPUColor{0.0, 0.0, 0.0, 1.0};
        WGPURenderPassDescriptor rd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        rd.colorAttachmentCount = 1;
        rd.colorAttachments = &color;
        WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rd);
        const float w = static_cast<float>(surface_width_), h = static_cast<float>(surface_height_);
        const float aspect = g_options.widescreen ? 16.0f / 9.0f : 4.0f / 3.0f;
        const float vw = std::min(w, h * aspect), vh = vw / aspect;
        wgpuRenderPassEncoderSetViewport(rp, (w - vw) * 0.5f, (h - vh) * 0.5f, vw, vh, 0.0f, 1.0f);
        wgpuRenderPassEncoderSetPipeline(rp, blit_);
        wgpuRenderPassEncoderSetBindGroup(rp, 0, blit_group_, 0, nullptr);
        wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(rp);
        wgpuRenderPassEncoderRelease(rp);
    }
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    if (target) {
#ifndef __EMSCRIPTEN__
        wgpuSurfacePresent(surface_);
#endif
        wgpuTextureViewRelease(target);
        wgpuTextureRelease(st.texture);
    }
    if (measure) {
        query_pending_ = true;
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowProcessEvents;
        cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void *self, void *) {
            auto *d = static_cast<WgpuDevice *>(self);
            if (status == WGPUMapAsyncStatus_Success) d->query_ready_ = true;
            else d->query_pending_ = false;
        };
        cb.userdata1 = this;
        wgpuBufferMapAsync(query_read_, WGPUMapMode_Read, 0, 16, cb);
    }
#ifdef __EMSCRIPTEN__
    wgpuInstanceProcessEvents(instance_);
#else
    wgpuDevicePoll(device_, false, nullptr);
#endif
}

void WgpuDevice::read_back(Buffer b, std::uint64_t offset, std::uint64_t bytes, void *out) {
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, buffer(b), offset, readback_, 0, bytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    bool done = false;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void *flag, void *) {
        if (status != WGPUMapAsyncStatus_Success) std::fprintf(stderr, "[gpu] map failed %d\n", static_cast<int>(status));
        *static_cast<bool *>(flag) = true;
    };
    cb.userdata1 = &done;
    wgpuBufferMapAsync(readback_, WGPUMapMode_Read, 0, bytes, cb);
    wait_until(done);
    const void *mapped = wgpuBufferGetConstMappedRange(readback_, 0, bytes);
    if (mapped) std::memcpy(out, mapped, bytes);
    wgpuBufferUnmap(readback_);
}

int WgpuDevice::begin_read(Buffer b, std::uint64_t offset, std::uint64_t bytes) {
    int slot = -1;
    for (int i = 0; i < kReadSlots; ++i)
        if (!reads_[i].busy) slot = i;
    if (slot < 0) return -1;
    ReadSlot &r = reads_[slot];
    if (r.capacity < bytes) {
        if (r.buffer) wgpuBufferRelease(r.buffer);
        r.capacity = bytes;
        r.buffer = make_buffer(bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    }
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, buffer(b), offset, r.buffer, 0, bytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(queue_, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    r.bytes = bytes;
    r.busy = true;
    r.ready = r.failed = false;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void *slot, void *) {
        ReadSlot &s = *static_cast<ReadSlot *>(slot);
        if (status == WGPUMapAsyncStatus_Success) s.ready = true;
        else s.failed = true;
    };
    cb.userdata1 = &r;
    wgpuBufferMapAsync(r.buffer, WGPUMapMode_Read, 0, bytes, cb);
    return slot;
}

int WgpuDevice::poll_read(int slot, void *out) {
    if (slot < 0 || slot >= kReadSlots || !reads_[slot].busy) return -1;
#ifdef __EMSCRIPTEN__
    wgpuInstanceProcessEvents(instance_);
#else
    wgpuDevicePoll(device_, false, nullptr);
#endif
    ReadSlot &r = reads_[slot];
    if (r.failed) {
        r.busy = r.failed = false;
        return -1;
    }
    if (!r.ready) return 0;
    const void *mapped = wgpuBufferGetConstMappedRange(r.buffer, 0, r.bytes);
    const bool ok = mapped != nullptr;
    if (ok) std::memcpy(out, mapped, r.bytes);
    wgpuBufferUnmap(r.buffer);
    r.busy = r.ready = false;
    return ok ? 1 : -1;
}

int WgpuDevice::wait_read(int slot, void *out) {
    if (slot < 0 || slot >= kReadSlots || !reads_[slot].busy) return -1;
    while (reads_[slot].busy && !reads_[slot].ready && !reads_[slot].failed) pump();
    return poll_read(slot, out);
}

void WgpuDevice::wait_idle() {
    bool done = false;
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void *flag, void *) { *static_cast<bool *>(flag) = true; };
    cb.userdata1 = &done;
    wgpuQueueOnSubmittedWorkDone(queue_, cb);
    wait_until(done);
}

double WgpuDevice::busy_ms() {
    if (!measured_) return 0.0;
    return static_cast<double>(measured_ns_) / static_cast<double>(measured_) * static_cast<double>(total_passes_) / 1e6;
}

void WgpuDevice::configure_surface() {
    WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
    config.device = device_;
    config.format = surface_format_;
    config.usage = WGPUTextureUsage_RenderAttachment;
    config.width = std::max(surface_width_, 1u);
    config.height = std::max(surface_height_, 1u);
    config.presentMode = present_mode_;
    config.alphaMode = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(surface_, &config);
}

bool WgpuDevice::attach_surface(const SurfaceTarget &target) {
    if (!device_) return false;
    WGPUSurfaceDescriptor sd = WGPU_SURFACE_DESCRIPTOR_INIT;
#ifdef __EMSCRIPTEN__
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    canvas.selector = view(target.selector ? target.selector : "#canvas");
    sd.nextInChain = &canvas.chain;
#else
    WGPUSurfaceSourceXlibWindow xlib = WGPU_SURFACE_SOURCE_XLIB_WINDOW_INIT;
    WGPUSurfaceSourceWaylandSurface wayland = WGPU_SURFACE_SOURCE_WAYLAND_SURFACE_INIT;
    if (target.kind == SurfaceTarget::X11) {
        xlib.display = target.display;
        xlib.window = target.window;
        sd.nextInChain = &xlib.chain;
    } else if (target.kind == SurfaceTarget::Wayland) {
        wayland.display = target.display;
        wayland.surface = target.surface;
        sd.nextInChain = &wayland.chain;
    } else {
        return false;
    }
#endif
    surface_ = wgpuInstanceCreateSurface(instance_, &sd);
    if (!surface_) return false;
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    if (wgpuSurfaceGetCapabilities(surface_, adapter_, &caps) != WGPUStatus_Success || !caps.formatCount) {
        wgpuSurfaceRelease(surface_);
        surface_ = nullptr;
        return false;
    }
    surface_format_ = caps.formats[0];
    for (std::size_t i = 0; i < caps.formatCount; ++i)
        if (caps.formats[i] == WGPUTextureFormat_BGRA8Unorm || caps.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
            surface_format_ = caps.formats[i];
            break;
        }
    present_mode_ = WGPUPresentMode_Fifo;
    for (const WGPUPresentMode wanted : {WGPUPresentMode_Immediate, WGPUPresentMode_Mailbox})
        for (std::size_t i = 0; i < caps.presentModeCount; ++i)
            if (caps.presentModes[i] == wanted) present_mode_ = wanted;
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    surface_width_ = target.width;
    surface_height_ = target.height;
    configure_surface();
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code = view(kBlitShader);
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &source.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device_, &md);
    WGPUBindGroupLayoutEntry le[2] = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
    le[0].binding = 0;
    le[0].visibility = WGPUShaderStage_Fragment;
    le[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    le[1].binding = 1;
    le[1].visibility = WGPUShaderStage_Fragment;
    le[1].buffer.type = WGPUBufferBindingType_Uniform;
    WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    ld.entryCount = 2;
    ld.entries = le;
    WGPUBindGroupLayout layout = wgpuDeviceCreateBindGroupLayout(device_, &ld);
    WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device_, &pld);
    WGPUColorTargetState color = WGPU_COLOR_TARGET_STATE_INIT;
    color.format = surface_format_;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = view("fs");
    fragment.targetCount = 1;
    fragment.targets = &color;
    WGPURenderPipelineDescriptor rd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    rd.layout = pipeline_layout;
    rd.vertex.module = module;
    rd.vertex.entryPoint = view("vs");
    rd.fragment = &fragment;
    blit_ = wgpuDeviceCreateRenderPipeline(device_, &rd);
    wgpuShaderModuleRelease(module);
    wgpuPipelineLayoutRelease(pipeline_layout);
    blit_params_ = make_buffer(32, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBindGroupEntry be[2] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    be[0].binding = 0;
    be[0].buffer = scratch_;
    be[0].size = WGPU_WHOLE_SIZE;
    be[1].binding = 1;
    be[1].buffer = blit_params_;
    be[1].size = 32;
    WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bd.layout = layout;
    bd.entryCount = 2;
    bd.entries = be;
    blit_group_ = wgpuDeviceCreateBindGroup(device_, &bd);
    wgpuBindGroupLayoutRelease(layout);
    return blit_ != nullptr;
}

void WgpuDevice::resize_surface(std::uint32_t width, std::uint32_t height) {
    if (!surface_ || (width == surface_width_ && height == surface_height_)) return;
    surface_width_ = width;
    surface_height_ = height;
    configure_surface();
}

}

Device *make_wgpu_device() { return new WgpuDevice(); }

}
