// Appended to gs.wgsl for devices without a rasterizer of their own: the
// decoded texture cache lives in the `decoded` buffer alone.
fn decoded_load(i: u32) -> u32 {
    return decoded[i];
}

fn decoded_store(i: u32, v: u32) {
    decoded[i] = v;
}
