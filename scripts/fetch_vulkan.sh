#!/bin/sh
# Fetches the Vulkan headers into third_party/ for the hardware GS backend,
# the way fetch_wgpu.sh fetches wgpu-native. The Vulkan loader comes from the
# system at run time (libvulkan.so.1), and the shader compiler is the
# distribution's glslang (apt install glslang-tools).
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
tag="${1:-v1.4.335}"
mkdir -p "$here/third_party"
if [ ! -f "$here/third_party/vulkan-headers/include/vulkan/vulkan.h" ]; then
  rm -rf "$here/third_party/vulkan-headers"
  git clone -q --depth 1 --branch "$tag" https://github.com/KhronosGroup/Vulkan-Headers.git "$here/third_party/vulkan-headers"
fi
echo "Vulkan headers $tag in $here/third_party/vulkan-headers"
