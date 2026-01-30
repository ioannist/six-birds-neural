cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

project(ratchet_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(cmake/warnings.cmake)

add_library(rgpu_core STATIC
  src/core/smoke_kernels.cu
  src/core/version.cu
)

target_include_directories(rgpu_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
rgpu_set_warnings(rgpu_core)

add_executable(ratchet_gpu_cli src/cli/main.cu)

target_link_libraries(ratchet_gpu_cli PRIVATE rgpu_core)
target_compile_definitions(ratchet_gpu_cli PRIVATE RGPU_COMPILED_ARCHS=\"${CMAKE_CUDA_ARCHITECTURES}\")
rgpu_set_warnings(ratchet_gpu_cli)

enable_testing()
add_subdirectory(tests)
