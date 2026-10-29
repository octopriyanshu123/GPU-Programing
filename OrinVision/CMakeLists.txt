# =====================================================================
# OrinVision - GPU image processing pipeline & benchmark for Jetson Orin
# =====================================================================
#   ./scripts/build.sh            (or the manual steps below)
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
#   ctest --test-dir build --output-on-failure      # CMake >= 3.20
#   (or: cd build && ctest --output-on-failure)
# =====================================================================
cmake_minimum_required(VERSION 3.18)

# Find nvcc even if /usr/local/cuda/bin is not in PATH (common over SSH)
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

# Jetson Orin = compute capability 8.7. Override for other GPUs, e.g.
#   cmake -DCMAKE_CUDA_ARCHITECTURES=72 ..   (Xavier)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

project(OrinVision VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for VS Code IntelliSense / clangd

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)   # optional: multi-core CPU reference

# ---------------------------------------------------------------------
# Core library: everything except main()
# ---------------------------------------------------------------------
add_library(orin_core STATIC
    src/image.cpp
    src/cpu_filters.cpp
    src/verify.cpp
    src/pipeline.cpp
    src/gpu/color.cu
    src/gpu/histogram.cu
    src/gpu/blur.cu
    src/gpu/sobel.cu
)
target_include_directories(orin_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(orin_core PUBLIC CUDA::cudart)

if(OpenMP_CXX_FOUND)
    target_link_libraries(orin_core PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_options(orin_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    # -lineinfo: lets Nsight Compute show which source line is slow
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
    # Debug builds: full GPU debug info for cuda-gdb (much slower code!)
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)

# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------
add_executable(orin_vision src/main.cpp)
target_link_libraries(orin_vision PRIVATE orin_core)
target_compile_options(orin_vision PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
enable_testing()
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE orin_core)
add_test(NAME pipeline_correctness COMMAND test_pipeline)

message(STATUS "OrinVision ${PROJECT_VERSION} | build: ${CMAKE_BUILD_TYPE} | "
               "CUDA arch: ${CMAKE_CUDA_ARCHITECTURES} | OpenMP: ${OpenMP_CXX_FOUND}")
