cmake_minimum_required(VERSION 3.28)

project(An-Even-Easier-Introduction-to-CUDA-Updated
    LANGUAGES CUDA CXX)

enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(src)