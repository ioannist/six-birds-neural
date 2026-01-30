#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "rgpu/cuda_check.hpp"
#include "rgpu/device_array.hpp"
#include "rgpu/version.hpp"

namespace rgpu {
void smoke_step(uint32_t* d, size_t n, uint32_t add, cudaStream_t stream = nullptr);
}

namespace {

uint32_t splitmix32(uint32_t x) {
  x += 0x9e3779b9u;
  x = (x ^ (x >> 16)) * 0x85ebca6bu;
  x = (x ^ (x >> 13)) * 0xc2b2ae35u;
  return x ^ (x >> 16);
}

bool cuda_available() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    std::printf("SKIP: no CUDA device\n");
    return false;
  }
  return true;
}

std::vector<uint32_t> run_smoke(size_t n, int steps, uint32_t seed, uint32_t add) {
  std::vector<uint32_t> host(n);
  for (size_t i = 0; i < n; ++i) {
    host[i] = splitmix32(seed + static_cast<uint32_t>(i));
  }

  rgpu::DeviceArray<uint32_t> dev(n);
  dev.copy_from_host(host.data(), n);

  for (int s = 0; s < steps; ++s) {
    rgpu::smoke_step(dev.ptr, n, add, nullptr);
  }

  RGPU_CUDA_CHECK(cudaDeviceSynchronize());
  dev.copy_to_host(host.data(), n);
  return host;
}

}  // namespace

TEST_CASE("kernel correctness small") {
  if (!cuda_available()) {
    return;
  }
  RGPU_CUDA_CHECK(cudaSetDevice(0));

  const size_t n = 1024;
  const int steps = 3;
  const uint32_t seed = 1u;
  const uint32_t add = 1u;

  std::vector<uint32_t> init(n);
  for (size_t i = 0; i < n; ++i) {
    init[i] = splitmix32(seed + static_cast<uint32_t>(i));
  }

  std::vector<uint32_t> out = run_smoke(n, steps, seed, add);

  const uint32_t steps_u = static_cast<uint32_t>(steps);
  const size_t indices[] = {0, 1, 2, 511, 1023};
  for (size_t idx : indices) {
    uint32_t expected = init[idx] + steps_u * add + steps_u * static_cast<uint32_t>(idx);
    CHECK(out[idx] == expected);
  }
}

TEST_CASE("determinism") {
  if (!cuda_available()) {
    return;
  }
  RGPU_CUDA_CHECK(cudaSetDevice(0));

  const size_t n = 4096;
  const int steps = 5;
  const uint32_t seed = 7u;
  const uint32_t add = 2u;

  std::vector<uint32_t> a = run_smoke(n, steps, seed, add);
  std::vector<uint32_t> b = run_smoke(n, steps, seed, add);

  uint64_t hash_a = rgpu::fnv1a64(a.data(), a.size() * sizeof(uint32_t));
  uint64_t hash_b = rgpu::fnv1a64(b.data(), b.size() * sizeof(uint32_t));

  CHECK(hash_a == hash_b);
}
