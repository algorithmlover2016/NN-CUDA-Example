cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
project(warp_perspective)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(Python REQUIRED)
MESSAGE(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(~/anaconda3/envs/torch_script/include/)
include_directories(~/anaconda3/envs/torch_script/include/python3.7m)

# Define our library target
add_library(warp_perspective SHARED op.cpp)
# Enable C++14
target_compile_features(warp_perspective PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(warp_perspective "${TORCH_LIBRARIES}")
# Link against OpenCV
target_link_libraries(warp_perspective opencv_core opencv_imgproc)

# build command
# mkdir build && cd build &&  cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..