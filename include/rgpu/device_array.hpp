#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "rgpu/cuda_check.hpp"

namespace rgpu {

template <class T>
struct DeviceArray {
  T* ptr = nullptr;
  size_t n = 0;

  DeviceArray() = default;

  explicit DeviceArray(size_t n_) : n(n_) {
    if (n == 0) {
      return;
    }
    RGPU_CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
  }

  ~DeviceArray() {
    if (ptr) {
      cudaFree(ptr);
    }
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  DeviceArray(DeviceArray&& other) noexcept : ptr(other.ptr), n(other.n) {
    other.ptr = nullptr;
    other.n = 0;
  }

  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (ptr) {
      cudaFree(ptr);
    }
    ptr = other.ptr;
    n = other.n;
    other.ptr = nullptr;
    other.n = 0;
    return *this;
  }

  void memset0() {
    if (n == 0 || !ptr) {
      return;
    }
    RGPU_CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
  }

  void copy_from_host(const T* h, size_t n_) {
    if (n_ > n) {
      throw std::runtime_error("copy_from_host size exceeds DeviceArray capacity");
    }
    if (n_ == 0) {
      return;
    }
    RGPU_CUDA_CHECK(cudaMemcpy(ptr, h, n_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copy_to_host(T* h, size_t n_) const {
    if (n_ > n) {
      throw std::runtime_error("copy_to_host size exceeds DeviceArray capacity");
    }
    if (n_ == 0) {
      return;
    }
    RGPU_CUDA_CHECK(cudaMemcpy(h, ptr, n_ * sizeof(T), cudaMemcpyDeviceToHost));
  }
};

}  // namespace rgpu
