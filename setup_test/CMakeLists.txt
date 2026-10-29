cmake_minimum_required(VERSION 3.18)  

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

project(jetson_gpu_programing LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 87)       
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG   "-G -g -O0")

add_executable(setup_check src/setup_check.cu)
add_executable(gpu_load src/gpu_load.cu)
add_executable(dot_product src/dot_product.cu)


message(STATUS "Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architecture : sm_${CMAKE_CUDA_ARCHITECTURES}")