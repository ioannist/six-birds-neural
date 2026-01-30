#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace rgpu {
namespace detail {
inline void cuda_check(cudaError_t err, const char* file, int line) {
  if (err == cudaSuccess) {
    return;
  }
  std::string msg = "CUDA error at ";
  msg += file;
  msg += ":";
  msg += std::to_string(line);
  msg += ": ";
  msg += cudaGetErrorString(err);
  throw std::runtime_error(msg);
}
}  // namespace detail
}  // namespace rgpu

#define RGPU_CUDA_CHECK(expr) ::rgpu::detail::cuda_check((expr), __FILE__, __LINE__)
#define RGPU_CUDA_CHECK_LAST() ::rgpu::detail::cuda_check(cudaGetLastError(), __FILE__, __LINE__)
