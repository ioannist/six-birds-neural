#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "rgpu/cuda_check.hpp"

namespace rgpu {

__global__ void kernel_add_iota(uint32_t* x, size_t n, uint32_t add) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] = x[idx] + add + static_cast<uint32_t>(idx);
  }
}

void smoke_step(uint32_t* d, size_t n, uint32_t add, cudaStream_t stream) {
  if (n == 0) {
    return;
  }
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  kernel_add_iota<<<grid, kBlock, 0, stream>>>(d, n, add);
  RGPU_CUDA_CHECK_LAST();
}

}  // namespace rgpu
