# Translates a WGSL file to SPIR-V with naga and embeds the result as a
# std::uint32_t array. The `override SCALE` of gs.wgsl is folded by naga, so
# the caller passes SCALE and the source is rewritten to a constant first.
#   cmake -DNAGA=... -DSRC=gs.wgsl -DOUT=gs_scale2.cpp -DNAME=kGsSpvScale2 -DSCALE=2 -DWORK=dir -P wgsl_to_spv.cmake
file(READ "${SRC}" wgsl)
if(APPEND)
    file(READ "${APPEND}" extra)
    # hw.wgsl writes a second blend source; the directive has to lead the module.
    if(NOT PLAIN)
        string(PREPEND wgsl "enable dual_source_blending;\n")
    endif()
    string(APPEND wgsl "\n" "${extra}")
endif()
if(SCALE)
    string(REPLACE "override SCALE: u32 = 1u;" "const SCALE: u32 = ${SCALE}u;" wgsl "${wgsl}")
endif()
file(WRITE "${WORK}/${NAME}.wgsl" "${wgsl}")
# WGSL's clip space has +y up; naga flips it for Vulkan unless told not to. The
# graphics shaders in hw.wgsl are written for Vulkan's own convention.
execute_process(COMMAND "${NAGA}" --keep-coordinate-space "${WORK}/${NAME}.wgsl" "${WORK}/${NAME}.spv" RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "naga failed on ${SRC}: ${out} ${err}")
endif()
file(READ "${WORK}/${NAME}.spv" hex HEX)
string(LENGTH "${hex}" chars)
math(EXPR words "${chars} / 8")
# SPIR-V is little-endian words; the hex dump is byte order, so swap each group.
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1u," body "${hex}")
file(WRITE "${OUT}" "#include <cstddef>\n#include <cstdint>\nextern const std::uint32_t ${NAME}[];\nextern const std::size_t ${NAME}_words;\nconst std::uint32_t ${NAME}[] = {${body}};\nconst std::size_t ${NAME}_words = ${words};\n")
