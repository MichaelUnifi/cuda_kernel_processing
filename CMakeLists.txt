cmake_minimum_required(VERSION 3.29)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(cuda_kernel_processing CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(sequential main.cpp)
add_executable(cuda main.cu)

set_target_properties(
        cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)