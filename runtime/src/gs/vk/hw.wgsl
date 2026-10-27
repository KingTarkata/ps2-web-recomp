// Appended to gs.wgsl at build time (cmake/wgsl_to_spv.cmake), so every
// helper of the compute renderer (memory layout, texture sampling, the
// Setup record) is shared. These entry points draw a batch with the GPU's
// own rasterizer into a target image that mirrors a frame buffer's words:
// R32_UINT, one texel per GS pixel sample, holding the raw frame word.
// Depth lives in a D32 attachment as z / 2^32 (exact for 16- and 24-bit Z).
//
// Memory model (docs/gs-hw-renderer.md): GS memory in `vram` stays the
// truth; `import_frame`/`import_depth` fill a target from it before a batch
// draws, `export_frame`/`export_depth` write the result back. The CPU side
// tracks which pages a target owns to skip the copies.

struct HwParams {
    setup: u32,      // setup record offset in `data`
    fbmsk: u32,      // frame mask applied when exporting (bits kept from memory)
    scale: u32,
    flags: u32,      // bit 0: depth attachment present
    ox: i32,         // rectangle origin in GS pixels (import/export)
    oy: i32,
    width: u32,      // rectangle size in samples (import/export), target size (draw)
    height: u32,
};

@group(1) @binding(0) var<uniform> hw: HwParams;
@group(1) @binding(1) var target_store: texture_storage_2d<r32uint, write>;
@group(1) @binding(2) var target_load: texture_2d<u32>;
// Depth travels through this buffer as D32 float bits: filled by
// import_depth and copied into the depth attachment, or copied out of it
// and converted back by export_depth. Rows are `width` samples long.
@group(1) @binding(3) var<storage, read_write> depth_stage: array<u32>;

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
    o.clip = vec4<f32>(x, y, f32(v.z) / 4294967296.0, 1.0);
    o.color = v.color;
    o.tex = v.tex;
    o.z_hi = v.z;
    return o;
}

struct FsOut {
    @location(0) word: u32,
};

// The per-pixel pipeline of `plot` for the cases the hardware path takes:
// no blending, no destination alpha test, whole-word frame mask, alpha test
// either off or discarding on failure. Everything else is drawn by the
// compute rasterizer.
@fragment
fn hw_fs(i: VsOut) -> FsOut {
    if ((hw.flags & 2u) != 0u) {
        var dbg: FsOut;
        dbg.word = 0x80FF00FFu;  // debug: paint every fragment, before any test
        return dbg;
    }
    if ((hw.flags & 32u) != 0u) {
        let d = load_setup(hw.setup);  // debug: setup flags low byte, scissor x1 low byte, tpsm
        var dbg: FsOut;
        dbg.word = (d.flags & 0xFFu) | ((u32(d.scax1) & 0xFFu) << 8u) | ((d.tpsm & 0xFFu) << 16u) | 0xFF000000u;
        return dbg;
    }
    if ((hw.flags & 16u) != 0u) {
        var dbg: FsOut;  // debug: colour alpha in red, tex.x/16 in green, tex.z*64 in blue
        dbg.word = (u32(clamp(i.color.w, 0.0, 255.0)) & 0xFFu) | ((u32(clamp(i.tex.x / 16.0, 0.0, 255.0)) & 0xFFu) << 8u) | ((u32(clamp(i.tex.z * 64.0, 0.0, 255.0)) & 0xFFu) << 16u) | 0xFF000000u;
        return dbg;
    }
    let s = load_setup(hw.setup);
    let sx = u32(i.clip.x);
    let sy = u32(i.clip.y);
    let x = i32(sx / hw.scale);
    let y = i32(sy / hw.scale);
    if (x < s.scax0 || x > s.scax1 || y < s.scay0 || y > s.scay1) {
        discard;
    }
    let scanmsk = mode(s, 28u, 2u);
    if ((scanmsk == 2u && (y & 1) == 0) || (scanmsk == 3u && (y & 1) != 0)) {
        discard;
    }
    let k = (sy % hw.scale) * hw.scale + sx % hw.scale;
    var r = i32(i.color.x);
    var g = i32(i.color.y);
    var b = i32(i.color.z);
    var a = i32(i.color.w);
    if (has(s, F_TME)) {
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
        let t = sample(s, u, v, k);
        if ((hw.flags & 4u) != 0u) {
            var dbg: FsOut;
            dbg.word = t | 0xFF000000u;  // debug: the raw texel
            return dbg;
        }
        if ((hw.flags & 8u) != 0u) {
            var dbg: FsOut;
            dbg.word = (u32(clamp(u, 0.0, 255.0)) & 0xFFu) | ((u32(clamp(v, 0.0, 255.0)) & 0xFFu) << 8u) | (u32(clamp(i.tex.z * 64.0, 0.0, 255.0)) << 16u) | 0xFF000000u;  // debug: u, v, q
            return dbg;
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
        if (!ok) {
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
    o.word = u32(r) | (u32(g) << 8u) | (u32(b) << 16u) | (out_a << 24u);
    return o;
}

// Fills a target from GS memory: one invocation per sample.
@compute @workgroup_size(16, 16)
fn import_frame(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= hw.width || gid.y >= hw.height) {
        return;
    }
    let s = load_setup(hw.setup);
    let tx = u32(hw.ox) * hw.scale + gid.x;
    let ty = u32(hw.oy) * hw.scale + gid.y;
    let x = tx / hw.scale;
    let y = ty / hw.scale;
    let k = (ty % hw.scale) * hw.scale + tx % hw.scale;
    let l = addr(s.ff, x, y);
    textureStore(target_store, vec2<i32>(i32(tx), i32(ty)), vec4<u32>(load_at(l, k), 0u, 0u, 0u));
}

// Writes a target back into GS memory through the frame format's own store
// (24-bit frames keep their top byte, 16-bit frames pack), with no mask.
fn write_frame_word(s: Setup, l: Loc, k: u32, word: u32) {
    write_frame(s, l, k, word, hw.fbmsk);
}

@compute @workgroup_size(16, 16)
fn export_frame(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= hw.width || gid.y >= hw.height) {
        return;
    }
    let s = load_setup(hw.setup);
    let tx = u32(hw.ox) * hw.scale + gid.x;
    let ty = u32(hw.oy) * hw.scale + gid.y;
    let x = tx / hw.scale;
    let y = ty / hw.scale;
    let k = (ty % hw.scale) * hw.scale + tx % hw.scale;
    let l = addr(s.ff, x, y);
    let word = textureLoad(target_load, vec2<i32>(i32(tx), i32(ty)), 0).x;
    write_frame_word(s, l, k, word);
}

// Depth cannot be a storage image, so it goes through depth_stage and a
// buffer/image copy on the host side. Both passes work on the rectangle.
@compute @workgroup_size(16, 16)
fn import_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= hw.width || gid.y >= hw.height) {
        return;
    }
    let s = load_setup(hw.setup);
    let tx = u32(hw.ox) * hw.scale + gid.x;
    let ty = u32(hw.oy) * hw.scale + gid.y;
    let x = tx / hw.scale;
    let y = ty / hw.scale;
    let k = (ty % hw.scale) * hw.scale + tx % hw.scale;
    let l = addr(s.zf, x, y);
    depth_stage[gid.y * hw.width + gid.x] = bitcast<u32>(f32(load_at(l, k) & s.zmax) / 4294967296.0);
}

@compute @workgroup_size(16, 16)
fn export_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= hw.width || gid.y >= hw.height) {
        return;
    }
    let s = load_setup(hw.setup);
    let tx = u32(hw.ox) * hw.scale + gid.x;
    let ty = u32(hw.oy) * hw.scale + gid.y;
    let x = tx / hw.scale;
    let y = ty / hw.scale;
    let k = (ty % hw.scale) * hw.scale + tx % hw.scale;
    let l = addr(s.zf, x, y);
    let d = bitcast<f32>(depth_stage[gid.y * hw.width + gid.x]);
    var z = 0xFFFFFFFFu;
    if (d < 1.0) {
        z = u32(round(d * 4294967296.0));
    }
    store(l, k, z & s.zmax);
}
