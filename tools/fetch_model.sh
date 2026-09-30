#!/usr/bin/env bash
# Download the Pollock Mini LM 125M files from Hugging Face, convert them to
# GGUF and quantize the two variants the app bundles. Run once before the first
# build - the GGUFs land in models/bundled/, which is an app asset source dir.
set -euo pipefail
cd "$(dirname "$0")/.."

HF=SlayerLab/pollock-mini-lm-125m
# model.safetensors|tokenizer.json checksums from the model's release_manifest.json
declare -A SHA=(
  [model.safetensors]=1c1002909119b4913841ee29efd44bd86d62d81f2ae27de01602b64d4af0dbec
  [tokenizer.json]=3733307577230bb4802d2c774d8e2323f7f64e4139c13712736a57daf91bdda1
)
mkdir -p models models/bundled

for FILE in model.safetensors tokenizer.json config.json tokenizer_config.json special_tokens_map.json; do
  if [ -f "models/$FILE" ]; then
    echo "$FILE already present."
    continue
  fi
  echo "Downloading $FILE ..."
  curl -fL --progress-bar -o "models/$FILE" "https://huggingface.co/$HF/resolve/main/$FILE"
done

for FILE in "${!SHA[@]}"; do
  printf "Verifying %s... " "$FILE"
  if shasum -a 256 "models/$FILE" | grep -q "${SHA[$FILE]}"; then
    echo ok
  else
    echo "CHECKSUM MISMATCH (expected ${SHA[$FILE]})"
    shasum -a 256 "models/$FILE"
    exit 1
  fi
done

if [ ! -f models/pollock-mini-lm-125m-f16.gguf ]; then
  python tools/convert_pollock_to_gguf.py models \
    --outfile models/pollock-mini-lm-125m-f16.gguf --outtype f16
fi

QUANT=vendor/llama.cpp/build-host/bin/llama-quantize
[ -x "$QUANT" ] || QUANT=vendor/llama.cpp/build-host/bin/llama-quantize.exe
if [ ! -x "$QUANT" ]; then
  echo "llama-quantize not found - run tools/build_llamacpp_host.sh first."
  exit 1
fi

"$QUANT" models/pollock-mini-lm-125m-f16.gguf models/bundled/pollock-mini-lm-125m-Q4_K_M.gguf Q4_K_M
"$QUANT" models/pollock-mini-lm-125m-f16.gguf models/bundled/pollock-mini-lm-125m-Q8_0.gguf  Q8_0

# GoLLeM-110M-PL-SFT (Polish, instruction-tuned; plain GPT-2 - no patch needed)
GOLLEM_DIR=models/gollem-sft
GOLLEM_HF=SlayerLab/goLLeM-110M-PL-SFT-merged
mkdir -p "$GOLLEM_DIR"
for FILE in model.safetensors tokenizer.json config.json tokenizer_config.json; do
  [ -f "$GOLLEM_DIR/$FILE" ] && continue
  echo "Downloading GoLLeM $FILE ..."
  curl -fL --progress-bar -o "$GOLLEM_DIR/$FILE" "https://huggingface.co/$GOLLEM_HF/resolve/main/$FILE"
done
if [ ! -f models/goLLeM-110M-PL-SFT-f16.gguf ]; then
  python tools/convert_gollem_to_gguf.py "$GOLLEM_DIR" \
    --outfile models/goLLeM-110M-PL-SFT-f16.gguf --outtype f16
fi
"$QUANT" models/goLLeM-110M-PL-SFT-f16.gguf models/bundled/goLLeM-110M-PL-SFT-Q4_K_M.gguf Q4_K_M

echo
echo "Models ready in models/bundled/. Next: build the app (see README)."
