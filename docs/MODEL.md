# Dlaczego Pollock Mini LM 125M

Cel: mały LLM **z fabryka.ai**, który realnie zmieści się na zegarku i działa
całkowicie offline. Ograniczenia sprzętowe zegarka (1–1,5 GB RAM, CPU
2–4 małe rdzenie, ~250–400 MB realnego budżetu na proces) odsiewają wszystko
powyżej ~200M parametrów.

## Kandydaci z ekosystemu fabryka.ai

| model | parametry | dostępne wagi | decyzja |
|---|---|---|---|
| **Pollock Mini LM 125M** (SlayerLab) | 127,8M | tak, HF (safetensors + tokenizer) | **wybrany** |
| GoLLeM 250M (gollem-v4-250m-pl) | 250M | nie (tylko API platformy) | odpada — brak wag do konwersji |
| SlayerLab sub150-32M | 32M | nie (tylko API) | odpada — brak wag |
| Bielik 11B / SLAYER 27B | 11–27B | — | odpada — klasy serwerowej |

Pollock jest jedynym publicznym małym modelem fabryka.ai z wagami, a jego
rozmiar (sub-128M) to dokładnie klasa, którą oryginalny projekt udowodnił na
Apple Watch (Falcon-H1-Tiny 90M — 57 MB, SmolLM2 135M — 104 MB).

## Budżet na zegarku (Q4_K_M, ctx 1024)

| pozycja | rozmiar |
|---|---|
| wagi Q4_K_M | ~88 MB |
| KV cache (f16): 1024 tok × 16 warstw × 2 × 12 głów × 64 × 2 B | ~50 MB |
| bufury obliczeniowe (n_batch=32) + runtime | ~20–40 MB |
| **razem w szczycie** | **~170 MB** |

Q8_0 (135 MB wag) daje ~220 MB w szczycie — stąd przełącznik w apce.

## Architektura i konwersja

- GPT-2-style decoder: 16 warstw / 12 głów / szerokość 768, pozycje bezwzględne
  uczące się, kontekst 2048, słownik 12 288 (byte-level BPE, pretokenizacja
  GPT-2).
- Jedyne odstępstwo od klasycznego GPT-2 to **MLP typu SwiGLU** (szerokość
  2144, `silu(gate) * value`, bez biasów w oryginalnym treningu) — stąd patch w
  llama.cpp ([LLAMACPP_PATCH.md](LLAMACPP_PATCH.md)) i split `c_fc -> ffn_gate +
  ffn_up` w konwerterze.
- `tie_word_embeddings`: brak osobnego `lm_head` — głowica wyjściowa to macierz
  osadzeń.
- Tokeny specjalne: `|endoftext|` (id 12285, bos/eos/pad), `|im_start|`,
  `|im_end|` — format czatu w apce używa tych markerów (jak w oryginale).

Jakość wg karty modelu (zero-shot, lm-eval 0.4.12): BLiMP 78.0%, LAMBADA 31.3%
(ppl 42.1), HellaSwag 30.4%, PIQA 61.7%, SciQ 64.3%, ARC-Easy 42.6%.
Intelligence Index (Open SLM FP32): 13.4.

## Ograniczenia i odpowiedzialne użycie

Skopiowane z karty modelu, bo obowiązują też tu:

- surowy model bazowy — bez instruction tuningu, RLHF i alignmentu;
- może halucynować i generować treści toksyczne/stronnicze;
- mały rozmiar limituje wiedzę, rozumowanie i spójność długiego tekstu;
- trenowany i oceniany głównie po angielsku;
- **nie** używać do porad medycznych, prawnych, finansowych ani decyzji
  o ludziach.

Licencja: `mixed-upstream-dataset-terms` (patrz LICENSE.md w repo modelu) —
warunki źródeł danych ocenia użytkownik dla swojego zastosowania.
