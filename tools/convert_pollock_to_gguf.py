#!/usr/bin/env python3
"""Convert SlayerLab/pollock-mini-lm-125m (HF safetensors) to a llama.cpp GGUF file.

Pollock is a GPT-2-style decoder with a SwiGLU MLP (`PollockSwiGLU`): one fused
`mlp.c_fc` projection producing [gate | value], then `silu(gate) * value` and a
`mlp.c_proj` down-projection. Upstream llama.cpp's `gpt2` arch expects a plain
GELU MLP, so this converter splits `c_fc` into `ffn_gate` + `ffn_up` and the
patched `src/models/gpt2.cpp` builds a SwiGLU FFN whenever `ffn_gate` is present.

Everything else (tokenizer, special tokens, GPT-2 tensor naming) is delegated to
llama.cpp's own `conversion` package, so the output is byte-compatible with
standard `convert_hf_to_gguf.py` GPT-2 conversions.

Usage:
    python tools/convert_pollock_to_gguf.py models/hf --outfile models/pollock-mini-lm-125m-f16.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# llama.cpp checkout produced by tools/setup_llamacpp.* (vendor/llama.cpp),
# with the sibling project layout: <project>/tools/this.py, <project>/vendor/llama.cpp.
VENDOR_LLAMACPP = Path(__file__).resolve().parent.parent / "vendor" / "llama.cpp"
for candidate in (VENDOR_LLAMACPP, Path(__file__).resolve().parent.parent / "llama.cpp"):
    if (candidate / "conversion").is_dir():
        sys.path.insert(1, str(candidate / "gguf-py"))  # the gguf package itself
        sys.path.insert(1, str(candidate))
        break
else:
    sys.exit(f"llama.cpp checkout not found; run tools/setup_llamacpp first (looked in {VENDOR_LLAMACPP})")

import gguf  # noqa: E402  (gguf-py from the llama.cpp checkout)
from conversion import ModelBase  # noqa: E402
from conversion.gpt2 import GPT2Model  # noqa: E402

MODEL_ID = "PollockForCausalLM"


@ModelBase.register(MODEL_ID)
class PollockModel(GPT2Model):
    """GPT-2 layout + fused SwiGLU `c_fc` split into gate/up pairs."""

    model_arch = gguf.MODEL_ARCH.GPT2

    def set_gguf_parameters(self):
        # GPT2Model hardcodes n_ff = 4 * n_embd; Pollock matches SwiGLU to a
        # smaller parameter budget (n_inner = 2144 vs 3072).
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.hparams["n_ctx"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(self.hparams["n_inner"])
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def get_vocab_base_pre(self, tokenizer) -> str:
        # Byte-level BPE with GPT-2 pretokenization; pin instead of hash-detecting
        # (the hash table only covers released upstream tokenizers).
        del tokenizer
        return "gpt-2"

    def modify_tensors(self, data_torch, name, bid):
        # mlp.c_fc: Conv1D stores [in, 2*n_ff]; transpose to [2*n_ff, in] and
        # split gate/value halves (PollockSwiGLU chunks the fused projection).
        if name.endswith((".mlp.c_fc.weight", ".mlp.c_fc.bias")):
            assert bid is not None, name
            if name.endswith(".weight"):
                data_torch = data_torch.transpose(1, 0)
            suffix = ".weight" if name.endswith(".weight") else ".bias"
            n_ff = data_torch.shape[0] // 2
            gate, up = data_torch[:n_ff], data_torch[n_ff:]
            yield f"blk.{bid}.ffn_gate{suffix}", gate.contiguous()
            yield f"blk.{bid}.ffn_up{suffix}", up.contiguous()
            return

        # tied lm_head: keep embeddings as the single output projection
        if name.endswith("lm_head.weight"):
            return

        # attn.c_attn / attn.c_proj / mlp.c_proj: GPT2Model does the Conv1D
        # transpose and the GGUF name mapping.
        yield from super().modify_tensors(data_torch, name, bid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="directory with model.safetensors + tokenizer.json")
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16"])
    args = parser.parse_args()

    ftype = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    }[args.outtype]

    model = PollockModel(args.model_dir, ftype, args.outfile)
    model.write()
    print(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
