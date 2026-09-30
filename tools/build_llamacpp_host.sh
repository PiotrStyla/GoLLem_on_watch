#!/usr/bin/env bash
# Build patched llama.cpp for the HOST (verification: quantize + parity) and the
# greedy_probe parity tool. Linux/macOS counterpart of build_llamacpp_host.bat.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S vendor/llama.cpp -B vendor/llama.cpp/build-host -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON

cmake --build vendor/llama.cpp/build-host --target llama-quantize llama-perplexity llama-tokenize

c++ -std=c++17 -O2 \
  -I vendor/llama.cpp/include -I vendor/llama.cpp/ggml/include \
  tools/greedy_probe.cpp -o vendor/llama.cpp/build-host/bin/greedy_probe \
  -L vendor/llama.cpp/build-host/src -L vendor/llama.cpp/build-host/bin -lllama -Wl,-rpath,'$ORIGIN'

echo
echo "Host tools ready in vendor/llama.cpp/build-host/bin/"
