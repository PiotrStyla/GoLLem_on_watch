# Patch llama.cpp: SwiGLU w architekturze `gpt2`

Duchowy odpowiednik `docs/LLAMACPP_ON_WATCHOS.md` z oryginału: jeden mały patch
do upstream llama.cpp, bez którego model po prostu nie działa poprawnie.

## Problem

Pollock to GPT-2 z jednym wyjątkiem: MLP jest typu **SwiGLU**

```
gate, value = chunk(x @ c_fc)      # c_fc: 768 -> 2*2144
h = silu(gate) * value
out = h @ c_proj                    # c_proj: 2144 -> 768
```

podczas gdy `llama_model_gpt2` buduje klasyczny MLP GELU (`LLM_FFN_GELU,
LLM_FFN_SEQ`). Sam GGUF nie wystarczy — graf nie ma ścieżki dla bramkowanego
aktywowania.

## Rozwiązanie (`tools/pollock-swiglu.patch`)

Dwa miejsca w `src/models/gpt2.cpp` (commit `2a53ace3`):

1. **`load_arch_tensors`** — dwa opcjonalne tensory na warstwę
   (`TENSOR_NOT_REQUIRED`):

   ```cpp
   layer.ffn_gate   = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);
   layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i),   {n_ff},        TENSOR_NOT_REQUIRED);
   ```

2. **`graph::graph`** — jeśli `ffn_gate` istnieje, FFN budowane jest jako
   `LLM_FFN_SILU, LLM_FFN_PAR`, czyli `ggml_swiglu_split(gate, up)`:

   ```cpp
   const bool swiglu = model.layers[il].ffn_gate != NULL;
   cur = build_ffn(cur,
           ffn_up, ffn_up_b, NULL,
           ffn_gate, ffn_gate_b, NULL,   // <- było NULL
           ffn_down, ffn_down_b, NULL,
           NULL,
           swiglu ? LLM_FFN_SILU : LLM_FFN_GELU,
           swiglu ? LLM_FFN_PAR : LLM_FFN_SEQ, il);
   ```

Wykrywanie po obecności tensora, nie po nowym kluczu GGUF: zwykłe konwersje
GPT-2 (bez `ffn_gate`) zachowują się **bit w bit** jak dotychczas — patch jest
z natury bezpieczny dla istniejących plików.

Konwerter (`tools/convert_pollock_to_gguf.py`) dokłada split
`mlp.c_fc -> blk.N.ffn_gate + blk.N.ffn_up` (oraz biasów) i przypina
`tokenizer.ggml.pre = "gpt-2"` (dokładny regex pretokenizacji GPT-2 +
byte-level BPE — Pollock używa tej samej).

## Weryfikacja

`tools/parity_check.py` porównuje greedy decoding Transformers vs GGUF na
czterech promptach:

- **tokenizacja: 4/4 identyczna** (BPE + regex GPT-2, pin `tokenizer.ggml.pre = "gpt-2"`);
- **greedy: identyczny** — z wyjątkiem argmax-flipów na near-tie (torch BLAS i
  ggml akumulują macierze w innej kolejności). Zmierzony przypadek z produkcji:
  w punkcie rozbieżności HF top-2 = 9.4301 vs 9.421, odstęp **0.009** — harness
  samodzielnie mierzy ten odstęp i klasyfikuje flip < 0.05 jako szum float, nie
  błąd. Wszystko powyżej progu = twardy FAIL.

## Upstreaming

Patch kwalifikuje się do upstreamu jako „gated MLP variant for GPT-2-style
models" (obejmuje też przyszłe warianty GELU-GLU). Do tego czasu patch jest
pinowany razem z commitem llama.cpp w `tools/setup_llamacpp.*`.
