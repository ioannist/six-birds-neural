#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "rgpu/version.hpp"

TEST_CASE("runtime and driver versions are sane") {
  rgpu::RuntimeInfo info = rgpu::get_runtime_info();
  CHECK((info.cudaDriverVersion >= info.cudaRuntimeVersion ||
         (info.cudaDriverVersion > 0 && info.cudaRuntimeVersion > 0)));
}
