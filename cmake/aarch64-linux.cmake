# Cross-compiling the native runner for 64-bit ARM Linux (Switch L4T, or
# qemu-user on the build machine). Clang is required, as on x86-64: the VU
# helpers use its elementwise builtins. Ubuntu's clang finds the headers and
# libraries of the aarch64-linux-gnu GCC packages from the target triple.
#   cmake -S . -B build/aarch64 -G Ninja -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux.cmake \
#       -DPS2_WGPU_DIR=$PWD/third_party/wgpu-native-aarch64 -DPS2_GENERATED=...
# Fetch the aarch64 wgpu-native release the way fetch_wgpu.sh does for x86-64.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
# The VU helpers depend on a multiply and an add staying separate
# operations; on AArch64 the compiler would otherwise contract them into
# fused multiply-adds and change the last bit of results that the game's
# replays and license tests rely on.
set(CMAKE_CXX_FLAGS_INIT "-ffp-contract=off")
set(CMAKE_C_FLAGS_INIT "-ffp-contract=off")
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
