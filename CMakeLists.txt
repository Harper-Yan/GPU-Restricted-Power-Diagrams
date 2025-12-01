cmake_minimum_required(VERSION 3.18)

project(cvoro LANGUAGES CXX CUDA)

option(WITH_ASAN "toggle to asan support" OFF)
option(USE_DOUBLE "do computation in double" OFF)

find_package(OpenCL)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIR})
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ==================================================================
# Language standards
# ==================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable separable compilation so that .cu and C++ link together
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Optional: Print PTX to help debug kernels
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# ==================================================================
# OpenCL target macro
# ==================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCL_TARGET_OPENCL_VERSION=120")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cvoro_config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/cvoro_config.h)

# ==================================================================
# CUDA + C++ source separation
# ==================================================================

# CUDA-only file(s) — compiled by NVCC
set(CUDA_SOURCES
    knn_cuda.cu
    bitonic-shared.cu
    spatial.cu
    cuda_util.cu
)

set(CPP_SOURCES_LIB
    Grid.cpp
    openCL.cpp
    voronoi.cpp
)

set(SOURCES_UTILS
    Options.cpp
    Utils.cpp
)

set(SOURCES_TEST
    test_voronoi.cpp
)

# ==================================================================
# Build target
# ==================================================================
add_executable(test_voronoi
    ${CPP_SOURCES_LIB}
    ${CUDA_SOURCES}     # All CUDA code compiled via NVCC
    ${SOURCES_UTILS}
    ${SOURCES_TEST}
)

# Ensure CUDA code gets proper flags
set_target_properties(test_voronoi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")

if (OpenCL_FOUND)
    target_link_libraries(test_voronoi ${OpenCL_LIBRARY})
endif()

# AddressSanitizer
if (WITH_ASAN)
    target_compile_options(test_voronoi PRIVATE -fno-omit-frame-pointer -fsanitize=address)
    target_link_options(test_voronoi PRIVATE -fno-omit-frame-pointer -fsanitize=address)
endif()

if (APPLE)
    target_compile_options(test_voronoi PRIVATE -Wall -Wextra -Wno-deprecated-declarations)
endif()
