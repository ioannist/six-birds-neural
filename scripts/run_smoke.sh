#!/usr/bin/env bash
set -euxo pipefail

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

./build/bin/ratchet_gpu_cli
