// Appended to gs.wgsl at build time (cmake/wgsl_to_spv.cmake), so every
// helper of the compute renderer (memory layout, texture sampling, the
// Setup record) is shared. These entry points draw a batch with the GPU's
// own rasterizer into a target image that mirrors a frame buffer's words:
// RGBA8_UNORM, one texel per GS pixel sample, the frame word's four bytes
// as unorm channels so the blend unit can work on them (import and export
// convert exactly). Depth lives in a D32 attachment as z / 2^32 (exact for
// 16- and 24-bit Z).
//
// Memory model (docs/gs-hw-renderer.md): GS memory in `vram` stays the
// truth; `import_frame`/`import_depth` fill a target from it before a batch
// draws, `export_frame`/`export_depth` write the result back. The CPU side
// tracks which pages a target owns to skip the copies.

struct HwParams {
    base: u32,       // draws: setup record offset in `data`; copies: frame or depth base (block address)
    bw: u32,         // copies: buffer width (64-pixel units)
    psm: u32,        // copies: pixel format
    flags: u32,      // bit 0: depth attachment present; bits 1, 2: debug paints; bit 3: keep the fragments the alpha test fails (and drop the rest); bit 4: output the blend factor C as the colour; bit 5: output Cs * (1 + C); bit 6: output Cs * 127/128
    ox: i32,         // rectangle origin in GS pixels (copies)
    oy: i32,
    width: u32,      // rectangle size in samples (copies), target size in samples (draws)
    height: u32,
    scale: u32,
    snap_lo: u32,    // draws: the snapshot window (compute `raster` reads it from params)
    snap_hi: u32,
    // A surface view: memory words in [view_base, view_base + view_words)
    // are read from the set-2 image through the surface's PSMCT32 layout
    // (view_table: the inverse page table in `tables`), view_shift rows down.
    view_table: u32,
    view_base: u32,
    view_bw: u32,
    view_words: u32,
    view_shift: u32,
    pad3: u32,
    pad4: u32,
    pad5: u32,
    setup: array<vec4<u32>, 7>,  // draws: the setup record itself, so no fragment loads it from `data`
};

// The setup record from the uniform copy, field by field (see load_setup in
// gs.wgsl for why there is no intermediate array).
// Fragment shader specialisation. The generic module (HW_VARIANT all ones)
// reads the setup; a specialised module fixes texturing (bit 0), whether the
// shader may discard (bit 1: alpha test or scan mask), and fog (bit 2), so
// the compiler drops the rest and, without discards, tests depth early.
fn v_on(s: Setup, flag: u32, bit: u32) -> bool {
    if (HW_VARIANT == 0xFFFFFFFFu) {
        return has(s, flag);
    }
    return (HW_VARIANT & bit) != 0u;
}
fn v_may_discard() -> bool {
    return HW_VARIANT == 0xFFFFFFFFu || (HW_VARIANT & 2u) != 0u;
}

fn setup_u() -> Setup {
    var s: Setup;
    s.flags = hw.setup[0].x;
    s.fbp = hw.setup[0].y;
    s.fbw = hw.setup[0].z;
    s.fpsm = hw.setup[0].w;
    s.fbmsk = hw.setup[1].x;
    s.zbp = hw.setup[1].y;
    s.zpsm = hw.setup[1].z;
    s.modes = hw.setup[1].w;
    s.aref = hw.setup[2].x;
    s.fix = hw.setup[2].y;
    s.scax0 = i32(hw.setup[2].z & 0xFFFFu);
    s.scax1 = i32(hw.setup[2].z >> 16u);
    s.scay0 = i32(hw.setup[2].w & 0xFFFFu);
    s.scay1 = i32(hw.setup[2].w >> 16u);
    s.tbp = hw.setup[3].x;
    s.tbw = hw.setup[3].y;
    s.tpsm = hw.setup[3].z;
    s.tw = hw.setup[3].w & 0xFFu;
    s.th = hw.setup[3].w >> 8u;
    s.cpsm = hw.setup[4].x;
    s.csa = hw.setup[4].y;
    s.minu = hw.setup[4].z & 0xFFFFu;
    s.maxu = hw.setup[4].z >> 16u;
    s.minv = hw.setup[4].w & 0xFFFFu;
    s.maxv = hw.setup[4].w >> 16u;
    s.ta0 = hw.setup[5].x & 0xFFu;
    s.ta1 = (hw.setup[5].x >> 8u) & 0xFFu;
    s.fogcol = hw.setup[5].y;
    s.clut = hw.setup[5].z & 0xFFFu;  // the high bits name the CLUT load (texture caches)
    s.zmax = hw.setup[5].w;
    s.dec = hw.setup[6].x;
    s.tf = fmt_of(s.tpsm, s.tbp, s.tbw);
    s.ff = fmt_of(s.fpsm, s.fbp, s.fbw);
    s.zf = fmt_of(s.zpsm, s.zbp, s.fbw);
    return s;
}

@group(1) @binding(0) var<uniform> hw: HwParams;
@group(1) @binding(1) var target_store: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(2) var target_load: texture_2d<f32>;
// Depth travels through this buffer as D32 float bits: filled by
// import_depth and copied into the depth attachment, or copied out of it
// and converted back by export_depth. Rows are `width` samples long.
@group(1) @binding(3) var<storage, read_write> depth_stage: array<u32>;
// The decoded texture cache as a 2048-wide R32_UINT image: the same words
// as `decoded`, which is kept in step for the compute rasterizer.
@group(1) @binding(4) var decoded_write: texture_storage_2d<r32uint, write>;
@group(1) @binding(5) var decoded_read: texture_2d<u32>;

// Set 2 is one cached texture: decoded once from GS memory into an RGBA8
// image (decode_tex), then sampled with the texture unit. The samplers
// cover REPEAT and CLAMP per axis; region modes fetch texels directly.
@group(2) @binding(0) var tex_image: texture_2d<f32>;
@group(2) @binding(1) var tex_store: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(2) var samp_rr: sampler;
@group(2) @binding(3) var samp_rc: sampler;
@group(2) @binding(4) var samp_cr: sampler;
@group(2) @binding(5) var samp_cc: sampler;

fn surface_word(word: u32) -> vec2<u32> {
    if (hw.view_words == 0u || word < hw.view_base || word - hw.view_base >= hw.view_words) {
        return vec2<u32>(0u, 0u);
    }
    let rel = word - hw.view_base;
    let page = rel / 2048u;
    let e = tables[hw.view_table + (rel % 2048u)];
    let x = (page % hw.view_bw) * 64u + (e & 0xFFu);
    let y = (page / hw.view_bw) * 32u + (e >> 8u) + hw.view_shift;
    return vec2<u32>(1u, pack_word(textureLoad(tex_image, vec2<i32>(i32(x * hw.scale), i32(y * hw.scale)), 0)));
}

fn decoded_load(i: u32) -> u32 {
    return textureLoad(decoded_read, vec2<i32>(i32(i & 2047u), i32(i >> 11u)), 0).x;
}

fn decoded_store(i: u32, v: u32) {
    decoded[i] = v;
    textureStore(decoded_write, vec2<i32>(i32(i & 2047u), i32(i >> 11u)), vec4<u32>(v, 0u, 0u, 0u));
}

struct VsIn {
    @location(0) pos: vec2<f32>,    // GS pixel coordinates (already minus XYOFFSET, / 16)
    @location(1) z: u32,
    @location(2) color: vec4<f32>,  // 0..255
    @location(3) tex: vec4<f32>,    // s, t, q, fog
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) tex: vec4<f32>,
    @location(2) @interpolate(flat) z_hi: u32,
};

@vertex
fn hw_vs(v: VsIn) -> VsOut {
    var o: VsOut;
    let w = f32(hw.width) / f32(hw.scale);
    let h = f32(hw.height) / f32(hw.scale);
    // GS pixel centers are at integer coordinates; Vulkan's are at +0.5.
    let x = (v.pos.x + 0.5) / w * 2.0 - 1.0;
    let y = (v.pos.y + 0.5) / h * 2.0 - 1.0;
    // Z is clamped to the depth buffer format's maximum, as plot() does.
    let s = setup_u();
    o.clip = vec4<f32>(x, y, f32(min(v.z, s.zmax)) / 4294967296.0, 1.0);
    o.color = v.color;
    o.tex = v.tex;
    o.z_hi = v.z;
    return o;
}

struct FsOut {
    @location(0) @blend_src(0) color: vec4<f32>,
    // The GS blend coefficient C = As / 128, for SRC1 blend factors.
    @location(0) @blend_src(1) factor: vec4<f32>,
};

fn unpack_word(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(w & 0xFFu), f32((w >> 8u) & 0xFFu), f32((w >> 16u) & 0xFFu), f32(w >> 24u)) / 255.0;
}

fn pack_word(c: vec4<f32>) -> u32 {
    let v = vec4<u32>(round(clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
    return v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
}

fn debug_out(word: u32) -> FsOut {
    var o: FsOut;
    o.color = unpack_word(word);
    o.factor = vec4<f32>(1.0);
    return o;
}

// The per-pixel pipeline of `plot` for the cases the hardware path takes:
// no blending, no destination alpha test, whole-word frame mask, alpha test
// either off or discarding on failure. Everything else is drawn by the
// compute rasterizer.
@fragment
fn hw_fs(i: VsOut) -> FsOut {
    if ((hw.flags & 2u) != 0u) {
        return debug_out(0x80FF00FFu);  // debug: paint every fragment, before any test
    }
    let s = setup_u();
    // Batches that read what they draw sample the snapshot the compute
    // `snapshot` pass copied to scratch, like the compute raster does.
    snapshot_lo = hw.snap_lo;
    snapshot_hi = hw.snap_hi;
    let sx = u32(i.clip.x);
    let sy = u32(i.clip.y);
    let x = i32(sx / hw.scale);
    let y = i32(sy / hw.scale);
    // The GS scissor is the Vulkan scissor rectangle (set per setup by the host).
    if (v_may_discard()) {
        let scanmsk = mode(s, 28u, 2u);
        if ((scanmsk == 2u && (y & 1) == 0) || (scanmsk == 3u && (y & 1) != 0)) {
            discard;
        }
    }
    let k = (sy % hw.scale) * hw.scale + sx % hw.scale;
    var r = i32(i.color.x);
    var g = i32(i.color.y);
    var b = i32(i.color.z);
    var a = i32(i.color.w);
    if (has(s, F_TME) && (hw.flags & 128u) == 0u) {  // bit 7: debug, skip sampling
        var u: f32;
        var v: f32;
        if (has(s, F_FST)) {
            u = i.tex.x / 16.0;
            v = i.tex.y / 16.0;
        } else {
            var q = i.tex.z;
            if (q == 0.0) {
                q = 1.0;
            }
            u = i.tex.x / q * f32(1u << s.tw);
            v = i.tex.y / q * f32(1u << s.th);
        }
        var t: u32;
        if ((hw.flags & 256u) != 0u) {
            t = sample_cached(s, u, v);
        } else {
            t = sample(s, u, v, k);
        }
        if ((hw.flags & 4u) != 0u) {
            return debug_out(t | 0xFF000000u);  // debug: the raw texel
        }
        let tr = i32(t & 0xFFu);
        let tg = i32((t >> 8u) & 0xFFu);
        let tb = i32((t >> 16u) & 0xFFu);
        let ta = i32(t >> 24u);
        let tcc = has(s, F_TCC);
        switch mode(s, 20u, 2u) {
            case 0u {
                r = clamp255((tr * r) >> 7u);
                g = clamp255((tg * g) >> 7u);
                b = clamp255((tb * b) >> 7u);
                if (tcc) {
                    a = clamp255((ta * a) >> 7u);
                }
            }
            case 1u {
                r = tr;
                g = tg;
                b = tb;
                if (tcc) {
                    a = ta;
                }
            }
            case 2u {
                r = clamp255(((tr * r) >> 7u) + a);
                g = clamp255(((tg * g) >> 7u) + a);
                b = clamp255(((tb * b) >> 7u) + a);
                if (tcc) {
                    a = clamp255(ta + a);
                }
            }
            default {
                r = clamp255(((tr * r) >> 7u) + a);
                g = clamp255(((tg * g) >> 7u) + a);
                b = clamp255(((tb * b) >> 7u) + a);
                if (tcc) {
                    a = ta;
                }
            }
        }
    }
    if (has(s, F_FGE)) {
        let f = i32(i.tex.w);
        let fr = i32(s.fogcol & 0xFFu);
        let fg = i32((s.fogcol >> 8u) & 0xFFu);
        let fb = i32((s.fogcol >> 16u) & 0xFFu);
        r = (f * r + (255 - f) * fr) >> 8u;
        g = (f * g + (255 - f) * fg) >> 8u;
        b = (f * b + (255 - f) * fb) >> 8u;
    }
    if (has(s, F_ATE)) {
        let av = u32(a);
        var ok = false;
        switch mode(s, 0u, 3u) {
            case 0u { ok = false; }
            case 1u { ok = true; }
            case 2u { ok = av < s.aref; }
            case 3u { ok = av <= s.aref; }
            case 4u { ok = av == s.aref; }
            case 5u { ok = av >= s.aref; }
            case 6u { ok = av > s.aref; }
            default { ok = av != s.aref; }
        }
        // Fail modes that still write something are drawn twice: once
        // keeping the passing fragments, once (bit 3) keeping the failing
        // ones with the writes the mode allows.
        if (ok == ((hw.flags & 8u) != 0u)) {
            discard;
        }
    }
    if (!has(s, F_COLCLAMP)) {
        r = r & 0xFF;
        g = g & 0xFF;
        b = b & 0xFF;
    }
    var out_a = u32(a & 0xFF);
    if (has(s, F_FBA)) {
        out_a = out_a | 0x80u;
    }
    var o: FsOut;
    o.color = vec4<f32>(f32(r), f32(g), f32(b), f32(out_a)) / 255.0;
    o.factor = vec4<f32>(min(f32(a) / 128.0, 1.0));
    if ((hw.flags & 64u) != 0u) {
        // Second step of a destination-alpha blend: the unit's DST_ALPHA is
        // Ad/255, the GS wants Ad/128, so Cs*Ad/255 is added once as is and
        // once scaled by 127/128 (two monotone steps, no clamping surprises).
        o.color = vec4<f32>(vec3<f32>(f32(r), f32(g), f32(b)) * (127.0 / 128.0) / 255.0, f32(out_a) / 255.0);
    }
    if ((hw.flags & 48u) != 0u) {
        // Blend forms the unit cannot express with Cs as the source colour:
        // Cd * (1 + C) wants the factor itself as the source (bit 4), and
        // Cs * (1 + C) wants the scaled colour (bit 5). C is As/128 or FIX/128.
        var c = f32(a) / 128.0;
        if (mode(s, 16u, 2u) == 2u) {
            c = f32(s.fix) / 128.0;
        }
        if ((hw.flags & 16u) != 0u) {
            o.color = vec4<f32>(c, c, c, f32(out_a) / 255.0);
        } else {
            o.color = vec4<f32>(vec3<f32>(f32(r), f32(g), f32(b)) * (1.0 + c) / 255.0, f32(out_a) / 255.0);
        }
    }
    return o;
}

// hw_fs for runs whose texture is a cached image: the same shader without
// the memory-sampling path, so the compiler sees a much smaller program.
@fragment
fn hw_fs_cached(i: VsOut) -> FsOut {
    if ((hw.flags & 2u) != 0u) {
        return debug_out(0x80FF00FFu);  // debug: paint every fragment, before any test
    }
    let s = setup_u();
    // Batches that read what they draw sample the snapshot the compute
    // `snapshot` pass copied to scratch, like the compute raster does.
    snapshot_lo = hw.snap_lo;
    snapshot_hi = hw.snap_hi;
    let sx = u32(i.clip.x);
    let sy = u32(i.clip.y);
    let x = i32(sx / hw.scale);
    let y = i32(sy / hw.scale);
    // The GS scissor is the Vulkan scissor rectangle (set per setup by the host).
    if (v_may_discard()) {
        let scanmsk = mode(s, 28u, 2u);
        if ((scanmsk == 2u && (y & 1) == 0) || (scanmsk == 3u && (y & 1) != 0)) {
            discard;
        }
    }
    let k = (sy % hw.scale) * hw.scale + sx % hw.scale;
    var r = i32(i.color.x);
    var g = i32(i.color.y);
    var b = i32(i.color.z);
    var a = i32(i.color.w);
    if (v_on(s, F_TME, 1u) && (hw.flags & 128u) == 0u) {  // bit 7: debug, skip sampling
        var u: f32;
        var v: f32;
        if (has(s, F_FST)) {
            u = i.tex.x / 16.0;
            v = i.tex.y / 16.0;
        } else {
            var q = i.tex.z;
            if (q == 0.0) {
                q = 1.0;
            }
            u = i.tex.x / q * f32(1u << s.tw);
            v = i.tex.y / q * f32(1u << s.th);
        }
        if ((hw.flags & 512u) != 0u) {  // debug: one texel for every fragment (texture bandwidth vs sampling maths)
            u = 0.0;
            v = 0.0;
        }
        let t = sample_cached(s, u, v);  // this variant only ever samples the bound texture image
        if ((hw.flags & 4u) != 0u) {
            return debug_out(t | 0xFF000000u);  // debug: the raw texel
        }
        let tr = i32(t & 0xFFu);
        let tg = i32((t >> 8u) & 0xFFu);
        let tb = i32((t >> 16u) & 0xFFu);
        let ta = i32(t >> 24u);
        let tcc = has(s, F_TCC);
        switch mode(s, 20u, 2u) {
            case 0u {
                r = clamp255((tr * r) >> 7u);
                g = clamp255((tg * g) >> 7u);
                b = clamp255((tb * b) >> 7u);
                if (tcc) {
                    a = clamp255((ta * a) >> 7u);
                }
            }
            case 1u {
                r = tr;
                g = tg;
                b = tb;
                if (tcc) {
                    a = ta;
                }
            }
            case 2u {
                r = clamp255(((tr * r) >> 7u) + a);
                g = clamp255(((tg * g) >> 7u) + a);
                b = clamp255(((tb * b) >> 7u) + a);
                if (tcc) {
                    a = clamp255(ta + a);
                }
            }
            default {
                r = clamp255(((tr * r) >> 7u) + a);
                g = clamp255(((tg * g) >> 7u) + a);
                b = clamp255(((tb * b) >> 7u) + a);
                if (tcc) {
                    a = ta;
                }
            }
        }
    }
    if (v_on(s, F_FGE, 4u)) {
        let f = i32(i.tex.w);
        let fr = i32(s.fogcol & 0xFFu);
        let fg = i32((s.fogcol >> 8u) & 0xFFu);
        let fb = i32((s.fogcol >> 16u) & 0xFFu);
        r = (f * r + (255 - f) * fr) >> 8u;
        g = (f * g + (255 - f) * fg) >> 8u;
        b = (f * b + (255 - f) * fb) >> 8u;
    }
    if (v_may_discard() && has(s, F_ATE)) {
        let av = u32(a);
        var ok = false;
        switch mode(s, 0u, 3u) {
            case 0u { ok = false; }
            case 1u { ok = true; }
            case 2u { ok = av < s.aref; }
            case 3u { ok = av <= s.aref; }
            case 4u { ok = av == s.aref; }
            case 5u { ok = av >= s.aref; }
            case 6u { ok = av > s.aref; }
            default { ok = av != s.aref; }
        }
        // Fail modes that still write something are drawn twice: once
        // keeping the passing fragments, once (bit 3) keeping the failing
        // ones with the writes the mode allows.
        if (ok == ((hw.flags & 8u) != 0u)) {
            discard;
        }
    }
    if (!has(s, F_COLCLAMP)) {
        r = r & 0xFF;
        g = g & 0xFF;
        b = b & 0xFF;
    }
    var out_a = u32(a & 0xFF);
    if (has(s, F_FBA)) {
        out_a = out_a | 0x80u;
    }
    var o: FsOut;
    o.color = vec4<f32>(f32(r), f32(g), f32(b), f32(out_a)) / 255.0;
    o.factor = vec4<f32>(min(f32(a) / 128.0, 1.0));
    if ((hw.flags & 64u) != 0u) {
        // Second step of a destination-alpha blend: the unit's DST_ALPHA is
        // Ad/255, the GS wants Ad/128, so Cs*Ad/255 is added once as is and
        // once scaled by 127/128 (two monotone steps, no clamping surprises).
        o.color = vec4<f32>(vec3<f32>(f32(r), f32(g), f32(b)) * (127.0 / 128.0) / 255.0, f32(out_a) / 255.0);
    }
    if ((hw.flags & 48u) != 0u) {
        // Blend forms the unit cannot express with Cs as the source colour:
        // Cd * (1 + C) wants the factor itself as the source (bit 4), and
        // Cs * (1 + C) wants the scaled colour (bit 5). C is As/128 or FIX/128.
        var c = f32(a) / 128.0;
        if (mode(s, 16u, 2u) == 2u) {
            c = f32(s.fix) / 128.0;
        }
        if ((hw.flags & 16u) != 0u) {
            o.color = vec4<f32>(c, c, c, f32(out_a) / 255.0);
        } else {
            o.color = vec4<f32>(vec3<f32>(f32(r), f32(g), f32(b)) * (1.0 + c) / 255.0, f32(out_a) / 255.0);
        }
    }
    return o;
}

// Measurement only (PS2_HW_DEBUG 4194304): one output, no blend, no tests.
struct FsMin {
    @location(0) color: vec4<f32>,
};
@fragment
fn hw_fs_min(i: VsOut) -> FsMin {
    var o: FsMin;
    o.color = i.color / 255.0;
    return o;
}

fn fetch_cached(s: Setup, xi: i32, yi: i32) -> u32 {
    let x = texel_x(s, xi);
    let y = texel_y(s, yi);
    return pack_word(textureLoad(tex_image, vec2<i32>(i32(x), i32(y)), 0));
}

// `sample` over the cached image: nearest through texelFetch with the GS
// wrap rules, bilinear through the sampler when the modes are REPEAT or
// CLAMP (the unit's own bilinear, 8-bit weights), four fetches otherwise.
fn sample_cached(s: Setup, uin: f32, vin: f32) -> u32 {
    if (!has(s, F_BILINEAR)) {
        return fetch_cached(s, i32(floor(uin)), i32(floor(vin)));
    }
    let mu = mode(s, 24u, 2u);
    let mv = mode(s, 26u, 2u);
    if (mu <= 1u && mv <= 1u) {
        let uv = vec2<f32>(uin, vin) / vec2<f32>(f32(1u << s.tw), f32(1u << s.th));
        var c: vec4<f32>;
        if (mu == 0u && mv == 0u) {
            c = textureSampleLevel(tex_image, samp_rr, uv, 0.0);
        } else if (mu == 0u) {
            c = textureSampleLevel(tex_image, samp_rc, uv, 0.0);
        } else if (mv == 0u) {
            c = textureSampleLevel(tex_image, samp_cr, uv, 0.0);
        } else {
            c = textureSampleLevel(tex_image, samp_cc, uv, 0.0);
        }
        return pack_word(c);
    }
    let u = uin - 0.5;
    let v = vin - 0.5;
    let x0 = i32(floor(u));
    let y0 = i32(floor(v));
    return blend4(fetch_cached(s, x0, y0), fetch_cached(s, x0 + 1, y0), fetch_cached(s, x0, y0 + 1), fetch_cached(s, x0 + 1, y0 + 1), u - f32(x0), v - f32(y0));
}

// Decodes one texture (the setup at hw.base) into the set-2 image.
@compute @workgroup_size(16, 16)
fn decode_tex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = load_setup(hw.base);
    if (gid.x >= (1u << s.tw) || gid.y >= (1u << s.th)) {
        return;
    }
    textureStore(tex_store, vec2<i32>(i32(gid.x), i32(gid.y)), unpack_word(convert(s, load_texel(addr(s.tf, gid.x, gid.y), 0u))));
}

// The copies below address GS memory through the frame or depth buffer
// named in `hw` alone, so a target can be written back long after the
// batch that drew it (its setup record is gone by then).
fn copy_sample(gid: vec3<u32>) -> vec4<u32> {
    let tx = u32(hw.ox) * hw.scale + gid.x;
    let ty = u32(hw.oy) * hw.scale + gid.y;
    return vec4<u32>(tx, ty, (ty % hw.scale) * hw.scale + tx % hw.scale, 0u);
}

// Fills a target from GS memory: one invocation per sample.
@compute @workgroup_size(16, 16)
fn import_frame(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= hw.width || gid.y >= hw.height) {
        return;
    }
    let c = copy_sample(gid);
    let l = addr(fmt_of(hw.psm, hw.base, hw.bw), c.x / hw.scale, c.y / hw.scale);
    textureStore(target_store, vec2<i32>(i32(c.x), i32(c.y)), unpack_word(load_at(l, c.z)));
}

// Writes a target back into GS memory through the frame format's own store:
// 24-bit frames keep their top byte, 16-bit frames pack. Frame masks were
// applied as colour write masks when drawing, so nothing is masked here.
fn store_frame(psm: u32, l: Loc, k: u32, c: u32) {
    switch psm {
        case 0x02u, 0x0Au {
            store(l, k, pack16(c));
        }
        case 0x01u {
            let l32 = Loc(l.word, 0u, 0xFFFFFFFFu);
            let old = load_at(l32, k);
            store(l32, k, (c & 0x00FFFFFFu) | (old & 0xFF000000u));
        }
        default {
            store(l, k, c);
        }
    }
}

@compute @workgroup_size(16, 16)
fn export_frame(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= hw.width || gid.y >= hw.height) {
        return;
    }
    let c = copy_sample(gid);
    let l = addr(fmt_of(hw.psm, hw.base, hw.bw), c.x / hw.scale, c.y / hw.scale);
    let word = pack_word(textureLoad(target_load, vec2<i32>(i32(c.x), i32(c.y)), 0));
    store_frame(hw.psm, l, c.z, word);
}

fn depth_max(psm: u32) -> u32 {
    switch psm {
        case 0x31u { return 0xFFFFFFu; }
        case 0x32u, 0x3Au { return 0xFFFFu; }
        default { return 0xFFFFFFFFu; }
    }
}

// Depth cannot be a storage image, so it goes through depth_stage and a
// buffer/image copy on the host side. Both passes work on the rectangle.
@compute @workgroup_size(16, 16)
fn import_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= hw.width || gid.y >= hw.height) {
        return;
    }
    let c = copy_sample(gid);
    let l = addr(fmt_of(hw.psm, hw.base, hw.bw), c.x / hw.scale, c.y / hw.scale);
    depth_stage[gid.y * hw.width + gid.x] = bitcast<u32>(f32(load_at(l, c.z) & depth_max(hw.psm)) / 4294967296.0);
}

@compute @workgroup_size(16, 16)
fn export_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= hw.width || gid.y >= hw.height) {
        return;
    }
    let c = copy_sample(gid);
    let l = addr(fmt_of(hw.psm, hw.base, hw.bw), c.x / hw.scale, c.y / hw.scale);
    // The rasterizer interpolates depth between integer vertex values; the
    // GS keeps the integer part, so floor it back (round would lift half of
    // the pixels by one and break greater-or-equal tests on coplanar draws).
    let d = bitcast<f32>(depth_stage[gid.y * hw.width + gid.x]);
    var z = 0xFFFFFFFFu;
    if (d < 1.0) {
        z = u32(floor(d * 4294967296.0 + 0.001));
    }
    store(l, c.z, z & depth_max(hw.psm));
}
