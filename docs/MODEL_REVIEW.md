# Uwagi do modeli SlayerLab/fabryka.ai — z pracy przy WatchLLM

Notatka robocza z konwersji modeli do GGUF, testów parity (HF vs llama.cpp) i
uruchomienia na Galaxy Watch4. Każda uwaga ma pokrycie w konkretnej rzeczy, na
którą trafiliśmy — nie w ogólnych życzeniach.

## Metoda (co zostało wykonane na wagach)

- konwersja HF → GGUF (split `mlp.c_fc`, kontrola kształtów, słownika BPE),
- **parity greedy**: Transformers vs llama.cpp, 4 prompty × 32 tokeny —
  tokenizacja i ciągi tokenów muszą być identyczne (near-tie flipy mierzone
  odstępem logitów),
- generowanie na żywym zegarku (Watch4 Classic, 32-bit): TTFT, tok/s, peak PSS.

Wyniki parity: Pollock 4/4 (EN), GoLLeM-110M-PL-SFT 4/4 (PL, w tym diakrytyki).
Modele są **bit-exact po konwersji** — uwagi poniżej dotyczą wydań, nie jakości
konwersji.

---

## Pollock Mini LM 125M (`pollock-mini-lm-125m`)

### Co blokuje użytkowników (trafiliśmy na to)

1. **Architektura nie jest „GPT-2 zamiennym" w narzędziach.** `PollockSwiGLU`
   skleja gate+value w jedną projekcję `c_fc: 768→2×2144` i wymaga
   `trust_remote_code`. W efekcie do llama.cpp potrzebny był patch
   (`docs/LLAMACPP_PATCH.md`), a do konwersji ręczny split tensora. Gdyby
   `mlp.c_fc`/`mlp.c_proj` miały standardowe nazwy i rozdzielenie gate/up
   (albo w repo był gotowy GGUF + skrypt konwersji), bariera wejścia spada do zera.
   Rekomendacja: **opublikować GGUF** (f16 + Q4_K_M) albo wysłać patch SwiGLU
   do upstream llama.cpp (kwalifikuje się — dotyczy każdego GPT-2 z gated MLP).
2. **Model jest angielski.** Laboratorium jest warszawskie, odbiorcy polscy.
   Z naszego użycia: do zegarka PL potrzebowaliśmy osobnego modelu (GoLLeM).
3. **Brak wariantu SFT.** Surowy base na zegarku odpowiada kontynuacją tekstu —
   użytkownik oczekuje odpowiedzi na pytanie.
4. **Karta nie podaje presetu samplingu.** Dla modeli tej klasy to nie detal:
   bez `repetition_penalty` dekodowanie zachłanne wpada w pętle (potwierdzone na
   GoLLeM-45M; ich karty wprost to opisują).

### Sugestie treningu

- **Polski rodzeńsny checkpoint przez continued pretraining**: architektura i
  pipeline zostają, 2–4 mld tokenów PL (SlayerLab ma gotowy `gollem-corpus-2b-pl`
  / `polish-dynaword`) przy LR ~1/10 oryginalnego. Przy 125M i tak trenowano
  ~123 tok/param — dalszy wzrost jakości przyjdzie z **danych, nie z liczby
  tokenów**: więcej deduplikacji, mniej web-only, domieszka rejestrów
  encyklopedycznych/literackich (wniosek z GoLLeM-v4: mix 88% web).
- **Tokenizer**: 12 288 pozycji przy angielskim jest OK; do PL użyć
  dynaword-32k — własna ablcja GoLLeM-45M pokazuje 1,96× gęstość i tańszy
  trening przy tym samym budżecie.
- **SFT grounded-QA** jak GoLLeM-110M-PL-SFT (26k par wystarczyło na formę
  odpowiedzi przy 110M) + publikacja szablonu promptu (patrz niżej, wspólny
  problem).
- Skoro MLP jest gated: rozważyć **RoPE zamiast pozycji uczonych** — odblokowuje
  wydłużanie kontekstu bez retreningu embeddingów (blokada GoLLeM 512/1024).

---

## GoLLeM-110M-PL (v2/v3) i GoLLeM-110M-PL-SFT-merged

### Co blokuje użytkowników

1. **Szablon promptu SFT nie jest publiczny.** Najpoważniejsza uwaga: 26 194 par
   grounded-QA, „gate treningowy: forma odpowiedzi 15/15" — ale w repo ani karcie
   nie ma formatu. Przyrządy (i nasza apka) **nie wiedzą, jak zadać pytanie**.
   Musieliśmy użyć surowej kontynuacji (PLAIN), żeby nie zmyślać formatu, przez
   co SFT prawdopodobnie nie wykorzystuje swojego formatowania. Rekomendacja:
   opublikować `chat_template` w `tokenizer_config.json` (standard HF — zadziała
   w transformers, llama.cpp i każdej apce) + 100 przykładowych par.
2. **`tokenizer_config.json` minimalny, brak `special_tokens_map.json`** —
   narzędzia muszą zgadywać bos/eos (trafiliśmy: skrypt pobierania pobrał 404
   jako plik i padł). Drobne, ale kosztuje godziny debugowania po stronie
   użytkowników.
3. **Kontekst 512** przy v4-250m z 1024 — przy pozycjach uczonych (GPT-2) to
   twardy limit.
4. **Dekodowanie zachłanne wpada w pętle** (wlasna uwaga dla 45M; dla 110M
   podobnie — nasz runner potrzebował temp+top-k). W karcie brak rekomendowanego
   presetu dla 110M (jest dla 45M: `repetition_penalty 1.15`; dla SFT w przykładach
   1.3).
5. **SFT „zjadło" jakość LM**: LAMBADA ppl 55 056, 8Tags ~losowo — karta to
   uczciwie raportuje (szacunek za to), ale to nie musi być cena SFT.

### Sugestie treningu

- **SFT z replay**: do 26k par QA domieszać 10–20% korpusu pretreningowego
  (albo 1 epoka zamiast 3, LR 1e–5). Cel: zachować perplexity bazowego przy
  formie odpowiedzi. Alternatywa: LoRA → merge (mniejsze przesunięcie wag).
- **Metryka jakości SFT**: oprócz formy odpowiedzi mierzyć perplexity
  kontrolne na held-out PL (np. BPB z SpeakLeash jak w 45M) — regresja z 55k ppl
  byłaby widoczna od razu.
- **Kontekst**: przy okazji kolejnej iteracji zamienić pozycje uczone na RoPE i
  dociągnąć do 2048 continued pretrainingiem (~0,5–1 mld tokenów, LR 1/10).
- **v3 vs CBD (−2,0)**: karta sugeruje v2 do zastosowań wrażliwych na hate —
  przy v4 kolejna ekspozycja niech obejmuje kontrolowaną domieszkę danych
  moderacyjnych zamiast czystej drugiej epoki.
- Dane: v3 to „druga epoka na tych samych danych" — kolejny krok lepiej zrobić
  na **nowym materiale** (gollem-corpus-16b-pl ma 16,6 mld tokenów; 110M z 2 mld
  jest mocno niedotrenowany wobec własnego reżimu „overtraining dla inferencji").

---

## GoLLeM-45M-PL (ablacja tokenizera)

Bardzo dobra robota metodologiczna (BPB zamiast ppl między tokenizerami,
niezależna reprodukcja 1:1). Uwagi:

1. **199 nieużywanych tokenów + `bad_words_ids`** — zamiast blokować na poziomie
   generowania, lepiej je usunąć przy eksporcie (mniejszy embedding, zero pułapek
   dla integratorów — nasza ścieżka GGUF musiałaby je specjalnie traktować).
2. **Pętle przy dekodowaniu zachłannym** — dla modeli <100M warto rozważyć
   trening z karą za powtórzenia (unlikelihood) albo choć standardowy preset
   `repetition_penalty` wprost w README (jest częściowo).
3. **Kontynuacja wynalazku**: wniosek „polski tokenizer 2,5× taniej przy tym
   samym BPB" jest wart skalowania — 110M/250M przy wyrównanym budżecie FLOPs
   (jak R2b) rozstrzygnie, czy efekt trzyma się powyżej 45M (Bielik v3 sugeruje,
   że tak — warto to domknąć w jednym protokole).

---

## gollem-v4-250m-pl (trajektoria treningowa)

1. **„Nie kończy generowania" (EOS rate ≈ 0)** — to nie tylko cecha; da się to
   naprawić danymi (proporcja przykładów zakończonych EOS) albo głowicą
   bias-ową dla EOS. Przy 250M base to częste, ale dla użytkowników blokujące.
2. **Custom loader zamiast `AutoModel`** — świetne do badań trajektorii, ale
   blokuje cały ekosystem (llama.cpp, TGI, apki). Rekomendacja: do finalnego
   `ckpt_48828` dorobić eksport w formacie czytanym przez standardowe narzędzia
   (GPT2LMHeadModel albo wprost GGUF) — badawcza trajektoria może zostać po
   staremu.
3. **Licencja „research-only"** dla wag — rozumiem przy trajektorii, ale finalny
   checkpoint w badawczej licencji np. Apache/MIT (jak 45M) rozszerzy oddziaływanie
   bez ryzyka produkcyjnego (karta i tak wyklucza production use opisem).
4. Sugestia pomiarowa: dodać do karty **BPB po polsku dla GoLLeM-110M-v3** w tym
   samym protokole (jest dla 45M i v4) — teraz trudno porównać 110M z 250M.

## gollem-v5-ckpts (angielski, tiny)

Kampania pomiarowa z wynikami negatywnymi — bardzo wartościowa, ale repo to
szereg checkpointów. Sugestia: finalny skonsolidowany checkpoint + tabela
„co nie zadziałało i przy jakim szumie" wprost na górze karty (artykuł
`gollem-v5` to ma, warto przenieść wnioski do model-card).

---

## Wspólne dla wszystkich wydań (najwyższy stosunek wartości do wysiłku)

1. **Publikujcie GGUF + szablon promptu przy wydaniu.** To jest dokładnie ta
   praca, którą każdy powtórzy po Was (my: patch llama.cpp + split tensora +
   dochodzenie formatu). Jeden plik `chat_template` w `tokenizer_config.json`
   kasuje problem formatu dla całego ekosystemu.
2. **Preset samplingu w karcie każdego modelu** (temp/top-k/repetition_penalty) —
   przy tej skali to różnica między „działa" a „pętla".
3. **Licencje**: `mixed-upstream-dataset-terms` (Pollock) i CC-BY-SA (GoLLeM 110M)
   utrudniają użycie w produktach (share-alike!). Rozważyć wyraźne FAQ: co wolno
   przy fine-tuningu i dystrybucji pochodnych.
4. Trzymajcie obecny rygor pomiarów (reprodukcje, BPB zamiast ppl między
   tokenizery, jawne wyniki negatywne) — to wyróżnia te wydania na plus i warto
   go rozciągnąć na powyższe braki.
