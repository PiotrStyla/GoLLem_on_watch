# Download the Pollock Mini LM 125M files from Hugging Face, convert them to
# GGUF and quantize the two variants the app bundles (Windows counterpart of
# tools/fetch_model.sh).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Hf = "SlayerLab/pollock-mini-lm-125m"
$Sha = @{
    "model.safetensors" = "1c1002909119b4913841ee29efd44bd86d62d81f2ae27de01602b64d4af0dbec"
    "tokenizer.json"    = "3733307577230bb4802d2c774d8e2323f7f64e4139c13712736a57daf91bdda1"
}
New-Item -ItemType Directory -Force -Path models, models/bundled | Out-Null

$Files = @("model.safetensors", "tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json")
foreach ($File in $Files) {
    if (Test-Path "models/$File") { Write-Host "$File already present."; continue }
    Write-Host "Downloading $File ..."
    curl.exe -fL --progress-bar -o "models/$File" "https://huggingface.co/$Hf/resolve/main/$File"
}

foreach ($File in $Sha.Keys) {
    Write-Host -NoNewline "Verifying $File... "
    $Actual = (Get-FileHash "models/$File" -Algorithm SHA256).Hash.ToLower()
    if ($Actual -ne $Sha[$File]) {
        Write-Host "CHECKSUM MISMATCH"
        Write-Host "  expected $($Sha[$File])"
        Write-Host "  actual   $Actual"
        exit 1
    }
    Write-Host "ok"
}

if (-not (Test-Path "models/pollock-mini-lm-125m-f16.gguf")) {
    python tools/convert_pollock_to_gguf.py models `
        --outfile models/pollock-mini-lm-125m-f16.gguf --outtype f16
}

$Quant = "vendor/llama.cpp/build-host/bin/llama-quantize.exe"
if (-not (Test-Path $Quant)) {
    Write-Host "llama-quantize not found - run tools/build_llamacpp_host.bat first."
    exit 1
}

& $Quant models/pollock-mini-lm-125m-f16.gguf models/bundled/pollock-mini-lm-125m-Q4_K_M.gguf Q4_K_M
& $Quant models/pollock-mini-lm-125m-f16.gguf models/bundled/pollock-mini-lm-125m-Q8_0.gguf  Q8_0

Write-Host ""
Write-Host "Models ready in models/bundled/. Next: build the app (see README)."
