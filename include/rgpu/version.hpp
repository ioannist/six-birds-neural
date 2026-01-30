#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rgpu {

struct RuntimeInfo {
  int cudaRuntimeVersion = 0;
  int cudaDriverVersion = 0;
  std::string gpuName;
  int smMajor = 0;
  int smMinor = 0;
};

RuntimeInfo get_runtime_info(int device = 0);

uint64_t fnv1a64(const void* data, size_t bytes);

}  // namespace rgpu
