#!/usr/bin/env bash
# Clone upstream llama.cpp at a pinned commit into vendor/llama.cpp and apply
# the Pollock SwiGLU patch. Run once before the first build.
set -euo pipefail
cd "$(dirname "$0")/.."

PINNED=2a53ace3be21b1231d42019a278f14aebc65d49c

if [ -d vendor/llama.cpp/.git ]; then
  echo "vendor/llama.cpp already present."
else
  mkdir -p vendor
  git init -q vendor/llama.cpp
  git -C vendor/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
  git -C vendor/llama.cpp fetch -q --depth 1 origin "$PINNED"
  git -C vendor/llama.cpp checkout -q FETCH_HEAD
fi

if git -C vendor/llama.cpp apply --reverse --check ../../tools/pollock-swiglu.patch 2>/dev/null; then
  echo "patch already applied."
else
  git -C vendor/llama.cpp apply ../../tools/pollock-swiglu.patch
  echo "patch applied."
fi
git -C vendor/llama.cpp diff --stat
