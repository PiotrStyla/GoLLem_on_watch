#!/usr/bin/env python3
"""Greedy-decoding parity check: HuggingFace Pollock vs the converted GGUF.

Both sides tokenize the same prompt and greedily decode N tokens. If the SwiGLU
patch, the c_fc->gate/up split and the BPE vocab are right, the id sequences
match exactly. The one exception class: argmax flips on *near-ties* (torch BLAS
and ggml accumulate matmuls in different orders, so logits equal to ~1e-3 can
swap ranks). The check measures the HF logit margin at the first divergence and
passes it as a tie-flip only when the margin is tiny.

    python tools/parity_check.py --gguf models/pollock-mini-lm-125m-f32.gguf

Requires `transformers`/`torch` on the HF side and a host build of llama.cpp
with tools/greedy_probe (tools/build_llamacpp_host.bat / .sh).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):",
    "1 2 3 4 5",
]

# logit margin below which an argmax flip counts as float noise, not a bug
TIE_MARGIN = 0.05


def load_hf(model_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()
    return tok, model


def hf_generate(tok, model, prompt: str, n: int) -> tuple[list[int], list[int]]:
    import torch

    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]),
            do_sample=False,
            max_new_tokens=n,
            pad_token_id=tok.eos_token_id,
        )
    return ids, out[0].tolist()[len(ids):]


def hf_margin(model, seq: list[int], expected: int, got: int) -> float:
    """Logit gap between the HF-preferred and the GGUF-preferred token at the
    last position of `seq`."""
    import torch

    with torch.no_grad():
        logits = model(torch.tensor([seq])).logits[0, -1]
    return float(logits[expected] - logits[got])


def gguf_side(probe: Path, gguf: Path, prompt: str, n: int) -> tuple[list[int], list[int]]:
    res = subprocess.run(
        [str(probe), str(gguf), prompt, str(n)],
        capture_output=True, text=True, check=True,
    )
    prompt_ids: list[int] = []
    gen_ids: list[int] = []
    for line in res.stdout.splitlines():
        if line.startswith("PROMPT_TOKENS"):
            prompt_ids = [int(x) for x in line.split()[1:]]
        elif line.startswith("GEN_TOKENS"):
            gen_ids = [int(x) for x in line.split()[1:]]
    return prompt_ids, gen_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--gguf", type=Path, default=ROOT / "models/pollock-mini-lm-125m-f16.gguf")
    ap.add_argument("--probe", type=Path,
                    default=ROOT / "vendor/llama.cpp/build-host/bin/greedy_probe.exe")
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()

    for p in (args.gguf, args.probe):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    tok, model = load_hf(args.model_dir)
    failures = 0
    tie_flips = 0

    for prompt in PROMPTS:
        hf_prompt, hf_gen = hf_generate(tok, model, prompt, args.tokens)
        gg_prompt, gg_gen = gguf_side(args.probe, args.gguf, prompt, args.tokens)

        tok_ok = hf_prompt == gg_prompt
        note = ""
        if hf_gen == gg_gen:
            gen_ok = True
        else:
            k = next((i for i, (a, b) in enumerate(zip(hf_gen, gg_gen)) if a != b),
                     min(len(hf_gen), len(gg_gen)))
            gap = hf_margin(model, hf_prompt + hf_gen[:k], hf_gen[k], gg_gen[k])
            if abs(gap) < TIE_MARGIN:
                gen_ok = True
                tie_flips += 1
                note = f" (tie-flip at token {k}: HF margin {gap:.4f} < {TIE_MARGIN})"
            else:
                gen_ok = False

        status = "OK  " if (tok_ok and gen_ok) else "FAIL"
        print(f"[{status}] {prompt!r}")
        print(f"    tokenizer: {'match' if tok_ok else 'MISMATCH'} ({len(hf_prompt)} vs {len(gg_prompt)} ids)")
        if not tok_ok:
            print(f"      hf  : {hf_prompt}")
            print(f"      gguf: {gg_prompt}")
        if hf_gen == gg_gen:
            print("    greedy:   match")
        elif note:
            print(f"    greedy:   near-tie flip{note}")
        else:
            n = min(len(hf_gen), len(gg_gen))
            same = sum(1 for a, b in zip(hf_gen, gg_gen) if a == b)
            print(f"    greedy:   MISMATCH {same}/{n} identical")
            print(f"      hf  : {hf_gen}")
            print(f"      gguf: {gg_gen}")
        failures += 0 if (tok_ok and gen_ok) else 1

    print()
    if failures == 0:
        extra = f" ({tie_flips} near-tie flip(s) within {TIE_MARGIN})" if tie_flips else ""
        print(f"PARITY OK{extra}")
    else:
        print(f"{failures}/{len(PROMPTS)} prompts diverged")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
