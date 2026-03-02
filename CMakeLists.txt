cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(translate-axera)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 后端选项（可组合）
# ---------------------------------------------------------------------------
option(BUILD_AX650 "Build AX650 on-chip backend" ON)
option(BUILD_AXCL  "Build AXCL PCIe backend" OFF)

message(STATUS "BUILD_AX650 = ${BUILD_AX650}")
message(STATUS "BUILD_AXCL  = ${BUILD_AXCL}")

if(NOT BUILD_AX650 AND NOT BUILD_AXCL)
    message(FATAL_ERROR "At least one of BUILD_AX650 or BUILD_AXCL must be ON")
endif()

# ---------------------------------------------------------------------------
# SDK 路径
# ---------------------------------------------------------------------------
if(BUILD_AX650)
    if(NOT BSP_MSP_DIR)
        if(EXISTS /soc/lib/libax_engine.so)
            message(STATUS "Detected board, BSP_MSP_DIR = /soc")
            set(BSP_MSP_DIR /soc)
        else()
            set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/bsp_msp_out/msp/out)
        endif()
    endif()
    message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")
endif()

if(BUILD_AXCL)
    if (NOT AXCL_DIR)
        set(AXCL_INC_DIRS /usr/include/axcl)
        set(AXCL_LIB_DIRS /usr/lib/axcl /usr/lib)
    else()
        set(AXCL_INC_DIRS
            ${AXCL_DIR}/include
            ${AXCL_DIR}/include/include
            ${AXCL_DIR}/include/axcl
            ${AXCL_DIR}/include/axcl/include
        )
        set(AXCL_LIB_DIRS
            ${AXCL_DIR}/lib
            ${AXCL_DIR}/lib/axcl
        )
        message(STATUS "AXCL_DIR = ${AXCL_DIR}")
    endif()
    include_directories(${AXCL_INC_DIRS})
    link_directories(${AXCL_LIB_DIRS})
    find_package(OpenMP QUIET)
endif()

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
set(TOKENIZER_BUILD_TESTS OFF)
add_subdirectory(third-party/tokenizer)
include_directories(third-party/tokenizer/include)

# ---------------------------------------------------------------------------
# Include dirs
# ---------------------------------------------------------------------------
include_directories(include)
include_directories(src)
include_directories(src/runner)
include_directories(src/runner/utils)

# ---------------------------------------------------------------------------
# Common sources
# ---------------------------------------------------------------------------
set(COMMON_SOURCES
    src/ax_translate.cpp
    src/runner/LLM.cpp
    src/runner/utils/memory_utils.cpp
    src/runner/utils/cqdm.cpp
)

# ---------------------------------------------------------------------------
# AX650 backend
# ---------------------------------------------------------------------------
if(BUILD_AX650)
    add_library(ax_translate_ax650 SHARED
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_ax650.cpp
    )
    target_include_directories(ax_translate_ax650 PRIVATE ${BSP_MSP_DIR}/include)
    target_link_libraries(ax_translate_ax650
        -L${BSP_MSP_DIR}/lib
        ax_engine ax_interpreter ax_sys
        pthread dl tokenizer
    )
    if(NOT BUILD_AXCL)
        set_target_properties(ax_translate_ax650 PROPERTIES OUTPUT_NAME ax_translate)
    endif()
    install(TARGETS ax_translate_ax650 DESTINATION lib)
endif()

# ---------------------------------------------------------------------------
# AXCL backend
# ---------------------------------------------------------------------------
if(BUILD_AXCL)
    set(AXCL_LINK_LIBS axcl_rt pthread dl tokenizer)
    set(AXCL_RT_CAND "")
    foreach(lib_dir ${AXCL_LIB_DIRS})
        file(GLOB _cand "${lib_dir}/libaxcl_rt.so.*")
        list(APPEND AXCL_RT_CAND ${_cand})
    endforeach()
    if(AXCL_RT_CAND)
        list(SORT AXCL_RT_CAND COMPARE NATURAL ORDER DESCENDING)
        list(GET AXCL_RT_CAND 0 AXCL_RT_REAL)
        add_library(axcl_rt_real SHARED IMPORTED)
        set_target_properties(axcl_rt_real PROPERTIES IMPORTED_LOCATION "${AXCL_RT_REAL}")
        set(AXCL_LINK_LIBS axcl_rt_real pthread dl tokenizer)
    endif()
    if(OpenMP_CXX_FOUND)
        list(APPEND AXCL_LINK_LIBS OpenMP::OpenMP_CXX)
    endif()

    add_library(ax_translate_axcl SHARED
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_axcl.cpp
        src/runner/utils/axcl_manager.cpp
    )
    target_include_directories(ax_translate_axcl PRIVATE ${AXCL_INC_DIRS})
    target_compile_definitions(ax_translate_axcl PRIVATE USE_AXCL)
    target_link_libraries(ax_translate_axcl
        ${AXCL_LINK_LIBS}
    )
    if(NOT BUILD_AX650)
        set_target_properties(ax_translate_axcl PROPERTIES OUTPUT_NAME ax_translate)
    endif()
    install(TARGETS ax_translate_axcl DESTINATION lib)
endif()

# ---------------------------------------------------------------------------
# Tests / tools
# ---------------------------------------------------------------------------
function(add_translate_test target_name src_file backend_target)
    add_executable(${target_name} ${src_file})
    target_link_libraries(${target_name} PRIVATE ${backend_target} pthread dl)
    if(${backend_target} MATCHES "axcl")
        target_compile_definitions(${target_name} PRIVATE USE_AXCL)
        if(DEFINED AXCL_INC_DIRS)
            target_include_directories(${target_name} PRIVATE ${AXCL_INC_DIRS})
        endif()
    endif()
    if(${backend_target} MATCHES "ax650")
        if(DEFINED BSP_MSP_DIR)
            target_include_directories(${target_name} PRIVATE ${BSP_MSP_DIR}/include)
        endif()
    endif()
    install(TARGETS ${target_name}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endfunction()

if(BUILD_AX650)
    if(BUILD_AXCL)
        add_translate_test(test_translate_ax650 tests/test_translate.cpp ax_translate_ax650)
        add_translate_test(translate_svr_ax650 tests/translate_svr.cpp ax_translate_ax650)
    else()
        add_translate_test(test_translate tests/test_translate.cpp ax_translate_ax650)
        add_translate_test(translate_svr tests/translate_svr.cpp ax_translate_ax650)
    endif()
endif()

if(BUILD_AXCL)
    if(BUILD_AX650)
        add_translate_test(test_translate_axcl tests/test_translate.cpp ax_translate_axcl)
        add_translate_test(translate_svr_axcl tests/translate_svr.cpp ax_translate_axcl)
    else()
        add_translate_test(test_translate tests/test_translate.cpp ax_translate_axcl)
        add_translate_test(translate_svr tests/translate_svr.cpp ax_translate_axcl)
    endif()
endif()

add_executable(translate_cli tests/translate_cli.cpp)
install(TARGETS translate_cli
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include DESTINATION include)
install(DIRECTORY pytranslate DESTINATION lib/)
