#include "rgpu/version.hpp"

#include <cuda_runtime.h>

#include <cstdint>

#include "rgpu/cuda_check.hpp"

namespace rgpu {

RuntimeInfo get_runtime_info(int device) {
  RuntimeInfo info;
  cudaDeviceProp prop{};

  RGPU_CUDA_CHECK(cudaRuntimeGetVersion(&info.cudaRuntimeVersion));
  RGPU_CUDA_CHECK(cudaDriverGetVersion(&info.cudaDriverVersion));
  RGPU_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  info.gpuName = prop.name;
  info.smMajor = prop.major;
  info.smMinor = prop.minor;

  return info;
}

uint64_t fnv1a64(const void* data, size_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint64_t hash = 14695981039346656037ull;
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= static_cast<uint64_t>(p[i]);
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace rgpu
