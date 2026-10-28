cmake_minimum_required(VERSION 3.20)
project(ps2-web-recomp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PS2_GENERATED "" CACHE PATH "Folder with the C++ generated for a game")

if(EMSCRIPTEN)
    add_compile_options(-msimd128 -mtail-call -fwasm-exceptions -sWASM_LEGACY_EXCEPTIONS=0 -pthread)
    add_link_options(-msimd128 -mtail-call -fwasm-exceptions -sWASM_LEGACY_EXCEPTIONS=0 -pthread)
endif()

if(NOT EMSCRIPTEN)
    add_executable(eerecomp
        recomp/main.cpp
        recomp/r5900.cpp
        recomp/elf.cpp
        recomp/analysis.cpp
        recomp/codegen.cpp
        recomp/snr2.cpp)
    target_compile_options(eerecomp PRIVATE -Wall -Wextra)

    add_executable(vujit_dump recomp/vujit_dump.cpp runtime/src/vu_jit.cpp)
    target_include_directories(vujit_dump PRIVATE runtime/include runtime/src)
    add_executable(vurecomp recomp/vurecomp.cpp)
    target_include_directories(vurecomp PRIVATE runtime/include)
    target_compile_options(vurecomp PRIVATE -Wall -Wextra)
endif()

add_library(ps2runtime STATIC
    runtime/src/memory.cpp
    runtime/src/dispatch.cpp
    runtime/src/system.cpp
    runtime/src/kernel.cpp
    runtime/src/hw.cpp
    runtime/src/cop.cpp
    runtime/src/disc.cpp
    runtime/src/dmac.cpp
    runtime/src/vif.cpp
    runtime/src/vu_micro.cpp
    runtime/src/vu_jit.cpp
    runtime/src/vu_jit_web.cpp
    runtime/src/ipu/ipu.cpp
    runtime/src/gs/gs.cpp
    runtime/src/gs/gif.cpp
    runtime/src/gs/queue.cpp
    runtime/src/gs/raster.cpp
    runtime/src/iop/cpu.cpp
    runtime/src/iop/kernel.cpp
    runtime/src/iop/libs.cpp
    runtime/src/iop/hw.cpp
    runtime/src/iop/spu2.cpp
    runtime/src/iop/sio2.cpp
    runtime/src/iop/io.cpp
    runtime/src/iop/cdvd.cpp
    runtime/src/iop/sif.cpp
    runtime/src/iop/cdvdfsv.cpp)
target_include_directories(ps2runtime PUBLIC runtime/include)
target_compile_options(ps2runtime PRIVATE -Wall -Wextra)

set(PS2_WGPU_DIR "${CMAKE_SOURCE_DIR}/third_party/wgpu-native" CACHE PATH "wgpu-native release folder")
set(PS2_VULKAN_HEADERS "${CMAKE_SOURCE_DIR}/third_party/vulkan-headers/include" CACHE PATH "Vulkan headers (scripts/fetch_vulkan.sh)")
option(PS2_GS_VK "Build the hardware-rasterized GS backend (PS2_GS=hw)" ON)
find_program(PS2_NAGA naga PATHS "$ENV{HOME}/.cargo/bin" DOC "naga-cli, translates WGSL to SPIR-V (cargo install naga-cli)")
if(PS2_GS_VK AND NOT EMSCRIPTEN AND EXISTS "${PS2_VULKAN_HEADERS}/vulkan/vulkan.h" AND PS2_NAGA)
    set(PS2_SPV_DIR "${CMAKE_BINARY_DIR}/spv")
    file(MAKE_DIRECTORY "${PS2_SPV_DIR}")
    set(PS2_SPV_SOURCES)
    # Two modules per scale: the compute renderer's shaders alone, and those plus
    # the graphics entry points of hw.wgsl, so a driver that dislikes the latter
    # still runs the former.
    foreach(scale 1 2 3)
        set(out "${PS2_SPV_DIR}/gs_scale${scale}.cpp")
        add_custom_command(OUTPUT "${out}"
            COMMAND ${CMAKE_COMMAND} -DNAGA=${PS2_NAGA} -DSRC=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/gs.wgsl -DAPPEND=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/gs_hooks.wgsl -DPLAIN=1 -DOUT=${out} -DNAME=kGsSpvScale${scale} -DSCALE=${scale} -DWORK=${PS2_SPV_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/wgsl_to_spv.cmake
            DEPENDS runtime/src/gs/gpu/gs.wgsl runtime/src/gs/gpu/gs_hooks.wgsl cmake/wgsl_to_spv.cmake)
        list(APPEND PS2_SPV_SOURCES "${out}")
        set(out "${PS2_SPV_DIR}/hw_scale${scale}.cpp")
        add_custom_command(OUTPUT "${out}"
            COMMAND ${CMAKE_COMMAND} -DNAGA=${PS2_NAGA} -DSRC=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/gs.wgsl -DAPPEND=${CMAKE_SOURCE_DIR}/runtime/src/gs/vk/hw.wgsl -DOUT=${out} -DNAME=kHwSpvScale${scale} -DSCALE=${scale} -DWORK=${PS2_SPV_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/wgsl_to_spv.cmake
            DEPENDS runtime/src/gs/gpu/gs.wgsl runtime/src/gs/vk/hw.wgsl cmake/wgsl_to_spv.cmake)
        list(APPEND PS2_SPV_SOURCES "${out}")
    endforeach()
    # Specialised fragment shaders (hw.wgsl HW_VARIANT) for native resolution,
    # the only scale the hardware path draws at.
    foreach(variant 0 1 2 3 4 5 6 7)
        set(out "${PS2_SPV_DIR}/hw_scale1_v${variant}.cpp")
        add_custom_command(OUTPUT "${out}"
            COMMAND ${CMAKE_COMMAND} -DNAGA=${PS2_NAGA} -DSRC=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/gs.wgsl -DAPPEND=${CMAKE_SOURCE_DIR}/runtime/src/gs/vk/hw.wgsl -DOUT=${out} -DNAME=kHwSpvScale1V${variant} -DSCALE=1 -DVARIANT=${variant} -DWORK=${PS2_SPV_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/wgsl_to_spv.cmake
            DEPENDS runtime/src/gs/gpu/gs.wgsl runtime/src/gs/vk/hw.wgsl cmake/wgsl_to_spv.cmake)
        list(APPEND PS2_SPV_SOURCES "${out}")
    endforeach()
    set(out "${PS2_SPV_DIR}/blit.cpp")
    add_custom_command(OUTPUT "${out}"
        COMMAND ${CMAKE_COMMAND} -DNAGA=${PS2_NAGA} -DSRC=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/blit.wgsl -DOUT=${out} -DNAME=kBlitSpv -DWORK=${PS2_SPV_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/wgsl_to_spv.cmake
        DEPENDS runtime/src/gs/gpu/blit.wgsl cmake/wgsl_to_spv.cmake)
    list(APPEND PS2_SPV_SOURCES "${out}")
    target_sources(ps2runtime PRIVATE runtime/src/gs/vk/loader.cpp runtime/src/gs/vk/context.cpp runtime/src/gs/vk/vk.cpp runtime/src/gs/vk/vk_device.cpp runtime/src/gs/vk/hw.cpp runtime/src/gs/vk/present.cpp ${PS2_SPV_SOURCES})
    set_source_files_properties(runtime/src/gs/vk/context.cpp runtime/src/gs/vk/vk_device.cpp runtime/src/gs/vk/hw.cpp runtime/src/gs/vk/present.cpp PROPERTIES COMPILE_OPTIONS -Wno-missing-field-initializers)
    target_include_directories(ps2runtime PRIVATE "${PS2_VULKAN_HEADERS}")
    target_compile_definitions(ps2runtime PUBLIC PS2_GS_VK)
    target_link_libraries(ps2runtime PUBLIC dl)
endif()
if(EMSCRIPTEN OR EXISTS "${PS2_WGPU_DIR}/lib/libwgpu_native.a")
    set(PS2_GS_WGSL "${CMAKE_BINARY_DIR}/gs_wgsl.cpp")
    add_custom_command(OUTPUT "${PS2_GS_WGSL}"
        COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/gs.wgsl -DAPPEND=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/gs_hooks.wgsl -DOUT=${PS2_GS_WGSL} -DNAME=kGsShader -P ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
        DEPENDS runtime/src/gs/gpu/gs.wgsl runtime/src/gs/gpu/gs_hooks.wgsl cmake/embed.cmake)
    set(PS2_BLIT_WGSL "${CMAKE_BINARY_DIR}/blit_wgsl.cpp")
    add_custom_command(OUTPUT "${PS2_BLIT_WGSL}"
        COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/blit.wgsl -DOUT=${PS2_BLIT_WGSL} -DNAME=kBlitShader -P ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
        DEPENDS runtime/src/gs/gpu/blit.wgsl cmake/embed.cmake)
    target_sources(ps2runtime PRIVATE runtime/src/gs/gpu/gpu.cpp runtime/src/gs/gpu/wgpu_device.cpp "${PS2_GS_WGSL}" "${PS2_BLIT_WGSL}")
    target_compile_definitions(ps2runtime PUBLIC PS2_GPU)
    if(EMSCRIPTEN)
        target_compile_options(ps2runtime PUBLIC --use-port=emdawnwebgpu)
        target_link_options(ps2runtime PUBLIC --use-port=emdawnwebgpu)
    else()
        target_include_directories(ps2runtime PRIVATE "${PS2_WGPU_DIR}/include")
        target_link_libraries(ps2runtime PUBLIC "${PS2_WGPU_DIR}/lib/libwgpu_native.a" dl pthread m)
        if(APPLE)
            target_link_libraries(ps2runtime PUBLIC "-framework Metal" "-framework QuartzCore" "-framework Foundation" "-framework CoreFoundation" "-framework IOKit" "-framework IOSurface" objc)
        endif()
    endif()
endif()

if(PS2_GENERATED)
    file(GLOB PS2_UNITS CONFIGURE_DEPENDS "${PS2_GENERATED}/*.cpp")
    add_library(ps2game STATIC ${PS2_UNITS})
    target_include_directories(ps2game PRIVATE "${PS2_GENERATED}")
    target_link_libraries(ps2game PUBLIC ps2runtime)
    target_compile_options(ps2game PRIVATE -Wno-unused-label -Wno-unused-variable)

    if(EMSCRIPTEN)
        set(PS2_WEB_ELF "" CACHE FILEPATH "Game executable preloaded as /game/core.elf")
        set(PS2_WEB_BOOT_FILE "" CACHE FILEPATH "Text file holding the argv[0] the game sees, preloaded as /game/boot.txt")
        set(PS2_WEB_PATCHES "" CACHE FILEPATH "Patch list the runtime writes into memory, preloaded as /game/patches.txt")
        add_executable(ps2web host/web.cpp)
        if(PS2_WEB_BOOT_FILE)
            target_link_options(ps2web PRIVATE "SHELL:--preload-file \"${PS2_WEB_BOOT_FILE}@/game/boot.txt\"")
        endif()
        if(PS2_WEB_PATCHES)
            target_link_options(ps2web PRIVATE "SHELL:--preload-file \"${PS2_WEB_PATCHES}@/game/patches.txt\"")
        endif()
        target_link_libraries(ps2web PRIVATE ps2game ps2runtime)
        target_link_options(ps2web PRIVATE --profiling-funcs -sJSPI -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 -sPTHREAD_POOL_SIZE=2 -sALLOW_TABLE_GROWTH=1 -lidbfs.js -sSTACK_SIZE=16777216 -sENVIRONMENT=web
            -sEXPORTED_FUNCTIONS=_main,_ps2web_pad,_ps2web_pad_connected,_ps2web_resize,_ps2web_gpu_ms,_ps2web_frames,_ps2web_primitives,_ps2web_audio_ring,_ps2web_audio_queued_ms,_ps2web_idle_ms,_ps2web_vu_interpreted
            -sEXPORTED_RUNTIME_METHODS=callMain,addRunDependency,removeRunDependency,HEAPU8,ENV
            --shell-file ${CMAKE_SOURCE_DIR}/web/shell.html --preload-file ${PS2_WEB_ELF}@/game/core.elf)
        set_target_properties(ps2web PROPERTIES SUFFIX ".html" LINK_DEPENDS ${CMAKE_SOURCE_DIR}/web/shell.html)
        return()
    endif()

    add_executable(ps2run host/main.cpp)
    target_link_libraries(ps2run PRIVATE ps2game ps2runtime)
    if(PS2_GLIBC_COMPAT)
        target_sources(ps2run PRIVATE host/glibc_compat.cpp)
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_sources(ps2run PRIVATE host/profile.cpp)
        target_compile_definitions(ps2run PRIVATE PS2_PROFILE)
    endif()
    find_package(SDL2 QUIET)
    if(SDL2_FOUND AND NOT EMSCRIPTEN)
        target_sources(ps2run PRIVATE host/window.cpp)
        target_compile_definitions(ps2run PRIVATE PS2_WINDOW)
        target_link_libraries(ps2run PRIVATE SDL2::SDL2)
    endif()
endif()
