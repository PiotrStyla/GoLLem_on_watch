#!/usr/bin/env python3
"""Convert SlayerLab/goLLeM-110M-PL-SFT-merged (HF safetensors) to GGUF.

GoLLeM-110M is a textbook GPT-2 (12 layers / 12 heads / 768, GELU MLP,
tied embeddings, ctx 512) with a Polish byte-level BPE (dynaword-32k) whose
pretokenizer uses the GPT-2 regex. Everything is stock llama.cpp `gpt2` arch -
no patch needed. The only override vs upstream `convert_hf_to_gguf.py` is
pinning `tokenizer.ggml.pre = "gpt-2"` (the hash-based detector only knows
released upstream tokenizers).

Usage:
    python tools/convert_gollem_to_gguf.py models/gollem-sft \\
        --outfile models/goLLeM-110M-PL-SFT-f16.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VENDOR_LLAMACPP = Path(__file__).resolve().parent.parent / "vendor" / "llama.cpp"
for candidate in (VENDOR_LLAMACPP, Path(__file__).resolve().parent.parent / "llama.cpp"):
    if (candidate / "conversion").is_dir():
        sys.path.insert(1, str(candidate / "gguf-py"))
        sys.path.insert(1, str(candidate))
        break
else:
    sys.exit(f"llama.cpp checkout not found; run tools/setup_llamacpp first (looked in {VENDOR_LLAMACPP})")

import gguf  # noqa: E402
from conversion.gpt2 import GPT2Model  # noqa: E402


class GollemModel(GPT2Model):
    """Plain GPT-2; only the BPE pretokenizer id is pinned."""

    model_arch = gguf.MODEL_ARCH.GPT2

    def get_vocab_base_pre(self, tokenizer) -> str:
        del tokenizer  # byte-level BPE with the GPT-2 pretokenization regex
        return "gpt-2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16"])
    args = parser.parse_args()

    ftype = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    }[args.outtype]

    model = GollemModel(args.model_dir, ftype, args.outfile)
    model.write()
    print(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
