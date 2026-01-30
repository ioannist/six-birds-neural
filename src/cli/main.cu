#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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

bool parse_i64(const char* s, int64_t* out) {
  if (!s || !*s) {
    return false;
  }
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (!end || *end != '\0') {
    return false;
  }
  *out = static_cast<int64_t>(v);
  return true;
}

std::string hex_u64(uint64_t v) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
  return oss.str();
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\\' || c == '"') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

void usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " [--n N] [--steps STEPS] [--seed SEED] [--device DEV] [--add ADD]"
            << " [--arch ARCH]" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    size_t n = 1u << 20;
    int steps = 10;
    uint32_t seed = 1u;
    int device = 0;
    uint32_t add = 1u;
    bool arch_flag = false;
    int arch_request = 0;

    for (int i = 1; i < argc; ++i) {
      const char* arg = argv[i];
      auto next_value = [&](int64_t* out) {
        if (i + 1 >= argc) {
          usage(argv[0]);
          throw std::runtime_error("missing value for argument");
        }
        if (!parse_i64(argv[++i], out)) {
          usage(argv[0]);
          throw std::runtime_error("invalid numeric argument");
        }
      };

      if (std::strcmp(arg, "--n") == 0) {
        int64_t v = 0;
        next_value(&v);
        if (v <= 0) {
          throw std::runtime_error("--n must be positive");
        }
        n = static_cast<size_t>(v);
      } else if (std::strcmp(arg, "--steps") == 0) {
        int64_t v = 0;
        next_value(&v);
        if (v < 0) {
          throw std::runtime_error("--steps must be non-negative");
        }
        steps = static_cast<int>(v);
      } else if (std::strcmp(arg, "--seed") == 0) {
        int64_t v = 0;
        next_value(&v);
        seed = static_cast<uint32_t>(v);
      } else if (std::strcmp(arg, "--device") == 0) {
        int64_t v = 0;
        next_value(&v);
        device = static_cast<int>(v);
      } else if (std::strcmp(arg, "--add") == 0) {
        int64_t v = 0;
        next_value(&v);
        add = static_cast<uint32_t>(v);
      } else if (std::strcmp(arg, "--arch") == 0) {
        int64_t v = 0;
        next_value(&v);
        arch_flag = true;
        arch_request = static_cast<int>(v);
      } else {
        usage(argv[0]);
        return 1;
      }
    }

    int device_count = 0;
    RGPU_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
      throw std::runtime_error("no CUDA devices detected");
    }
    if (device < 0 || device >= device_count) {
      throw std::runtime_error("invalid --device index");
    }
    RGPU_CUDA_CHECK(cudaSetDevice(device));

    rgpu::RuntimeInfo info = rgpu::get_runtime_info(device);

    if (arch_flag) {
#ifdef RGPU_COMPILED_ARCHS
      std::cerr << "archRequested=" << arch_request << " compiledArchs="
                << RGPU_COMPILED_ARCHS << " runtimeSm=" << info.smMajor << "."
                << info.smMinor << std::endl;
#else
      std::cerr << "archRequested=" << arch_request << " compiledArchs=unknown"
                << " runtimeSm=" << info.smMajor << "." << info.smMinor << std::endl;
#endif
    }

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

    uint64_t sum = 0;
    for (uint32_t v : host) {
      sum += static_cast<uint64_t>(v);
    }
    uint64_t hash = rgpu::fnv1a64(host.data(), host.size() * sizeof(uint32_t));

    std::cout << "{";
    std::cout << "\"device\":" << device << ",";
    std::cout << "\"gpu\":\"" << json_escape(info.gpuName) << "\",";
    std::cout << "\"sm\":\"" << info.smMajor << "." << info.smMinor << "\",";
    std::cout << "\"cudaRuntime\":" << info.cudaRuntimeVersion << ",";
    std::cout << "\"cudaDriver\":" << info.cudaDriverVersion << ",";
    std::cout << "\"n\":" << n << ",";
    std::cout << "\"steps\":" << steps << ",";
    std::cout << "\"seed\":" << seed << ",";
    std::cout << "\"add\":" << add << ",";
    std::cout << "\"sum\":\"" << hex_u64(sum) << "\",";
    std::cout << "\"hash\":\"" << hex_u64(hash) << "\"";
    std::cout << "}" << std::endl;

    if (n > 0) {
      std::cout << "PASS" << std::endl;
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
