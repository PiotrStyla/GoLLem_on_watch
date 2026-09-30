# Clone upstream llama.cpp at a pinned commit into vendor/llama.cpp and apply
# the Pollock SwiGLU patch. Run once before the first build.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Pinned = "2a53ace3be21b1231d42019a278f14aebc65d49c"

if (Test-Path "vendor/llama.cpp/.git") {
    Write-Host "vendor/llama.cpp already present."
} else {
    New-Item -ItemType Directory -Force -Path vendor | Out-Null
    git init -q vendor/llama.cpp
    git -C vendor/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
    git -C vendor/llama.cpp fetch -q --depth 1 origin $Pinned
    git -C vendor/llama.cpp checkout -q FETCH_HEAD
}

$already = git -C vendor/llama.cpp apply --reverse --check ../../tools/pollock-swiglu.patch 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "patch already applied."
} else {
    git -C vendor/llama.cpp apply ../../tools/pollock-swiglu.patch
    Write-Host "patch applied."
}
git -C vendor/llama.cpp diff --stat
