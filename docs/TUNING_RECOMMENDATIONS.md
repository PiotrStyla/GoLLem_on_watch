# Zalecenia tuningowe dla budowniczego modelu (SlayerLab/fabryka.ai)

Adresat: osoba prowadząca trening. Cel nadrzędny wypływa z naszego wdrożenia:
**mały model PL, który realnie działa na zegarku** — odpowiada na pytania, kończy
generowanie, trzyma jakość LM i mieści się w budżecie czasu odpowiedzi.

Kontekst pomiarowy (Galaxy Watch4 Classic, 2× Cortex-A55 1.2 GHz, 32-bit):
GoLLeM-110M-PL-SFT Q4 = **7,3 s TTFT, 0,49 tok/s, peak PSS 204 MB**. Każda
rekomendacja poniżej ma poprawić któryś z tych parametrów albo usunąć blokadę
integracyjną, na którą wpadliśmy (pełne uzasadnienie: `MODEL_REVIEW.md`).

## 1. Dane i pretrening (priorytet: PL-110M v4 / 250M)

1. **Kolejna iteracja 110M: nowe dane, nie druga epoka.** Regresja CBD (−2,0)
   po 2. epoce v3 to sygnał przetrenowania tych samych sekwencji. Repozytorium
   ma 16,6 mld tokenów (`gollem-corpus-16b-pl`), a 110M widział 2 mld.
   Receptura: 8–10 mld tokenów, mix jak v4 ale z korektą rejestrów:
   - web (HPLT PL) **70%** (zamiast 88%),
   - encyklopedyczny + literacki **20%** (Wikipedia/Wikisource/Wolne Lektury/eltec),
   - nauka/news/prawo **10%**, przy czym **prawo ≤ 0,5%** (wniosek v3: zero
     legalese poprawiło rejestr),
   - kontrolowana domieszka **0,5–1% treści moderacyjnych/hate-speech**
     (naprawa regresji CBD bez osobnego przebiegu).
2. **Optymalizator**: trzymać Muon+AdamW (v4) — jest w repo i działa; dla 110M
   sam AdamW też daje radę, ale nie ma powodu zmieniać sprawdzonego stacku.
   Harmonogram: cosine z 200-krokowym warmupem, anneal od 80% tokenu budżetu
   (v4: od kroku 30 000 — działało), na końcu **10× obniżka LR** i kontynuacja do
   plateau (wasz własny wynalazek z R008: „obniżyć i restartować zamiast ciągnąć
   zepsuty anneal" — stosować wprost jako standard).
3. **Budżet tokenów/parametr**: celować 60–100 tok/param (v4: 97 — dobry punkt;
   Pollock: 123 i dalej zyskiwały z jakości danych). Dla 250M = 15–25 mld tok.
4. **Miara sukcesu pretreningu**: BPB na prywatnym held-out (jak 45M) **oraz**
   loss walidacyjny co 2% budżetu — trzymać publiczną trajektorię (v4: bardzo
   dobry wzorzec).

## 2. Tokenizer

5. **dynaword-32k wszędzie.** Wasza ablcja (45M) dowiodła 1,96× gęstości i
   2,5× tańszego treningu przy tym samym BPB — rozstrzygnąć wniosek na 110M/250M
   w jednym protokole z wyrównanym FLOPs (wzór: R2b).
6. **Uciąć nieużywane tokeny** (45M: 199 pozycji + `bad_words_ids` w runtime) —
   eksporować słownik bez paddingu; integratorzy nie powinni blokować tokenów
   po stronie aplikacji.
7. Przy nowym tokenizerze od razu ustawić `tokenizer.ggml.pre`-kompatybilną
   pretokenizację (regex GPT-2 — sprawdzone w naszej konwersji) i zapisać ją w
   `tokenizer_config.json`.

## 3. SFT (priorytet: UX na urządzeniu)

8. **Szablon promptu opublikować przy wydaniu** (`chat_template` w
   `tokenizer_config.json` + 100 par w README). To warunek, by SFT w ogóle
   zadziałało poza Waszym skryptem — my musieliśmy wysyłać surowy prompt.
9. **SFT z replay zamiast czystego SFT.** Obecne 3 epoki na 26k par dały:
   formę odpowiedzi TAK, ale LAMBADA ppl 55 056 i 8Tags ~losowo (catastrophic
   forgetting). Receptura:
   - mix: **1 część SFT : 3–7 części korpusu pretreningowego** (cały czas),
   - LR **1e–5**, 1–2 epoki (nie 3), AdamW, wd 0,1 (jak teraz),
   - albo LoRA r=16–32 → merge (mniejsze przesunięcie wag bazowych).
10. **Bramki wydania SFT** (przejść wszystkie, inaczej nie puszczać):
    - forma odpowiedzi 15/15 (jak teraz),
    - **BPB/perplexity na held-out PL nie gorsze niż 1,5× bazowego** (cel:
      LAMBADA ppl rzędu 10², nie 10⁴),
    - EOS rate ≥ 90% na 50 pytaniach z odpowiedziami jednozdaniowymi,
    - brak pętli powtórzeń przy dekodowaniu z Waszym presetem.
11. **Dodać ~200 par zakończonych EOS wprost** („krótka odpowiedź → EOS") —
    w parze z punktem 12 to najtańszy sposób na poprawę UX.

## 4. Generowanie (naprawia się też po stronie treningu)

12. **EOS rate ≈ 0 w v4-250m to defekt do usunięcia, nie cecha.** Po stronie
    treningu: (a) domieszka przykładów zakończonych EOS (jak w pkt 11),
    (b) bias dla EOS w głowicy wyjściowej na ostatnich krokach, (c) ewentualnie
    kara za brak EOS w DPO/prostej regularyzacji.
13. **Preset samplingu w karcie każdego wydania** (dla 110M proponuję punkt
    startowy do strojenia): `temperature 0.8, top_k 40, top_p 0.95,
    repetition_penalty 1.3` (jak w Waszych przykładach SFT) oraz wariant
    deterministyczny do testów: greedy + `repetition_penalty 1.15`.
    Wskazówka z naszego portu: dla narzędzi (fakty liczbowe) greedy jest
    konieczny — warto mieć przetestowany tryb „factual".

## 5. Architektura i kontekst

14. **RoPE zamiast pozycji uczonych** przy kolejnej skali — blokuje to dziś
    wydłużanie kontekstu (110M: 512, v4: 1024). Plan minimalny: RoPE od startu
    przy nowym treningu; dla istniejących ckpt — 0,5–1 mld tok continued
    pretraining z interpolacją pozycji do 2048.
15. **Kontekst 2048** jako cel produktowy (zegarek: pytanie + fakt z narzędzia +
    odpowiedź zmieszczą się z zapasem; KV cache i tak jest mały przy tej skali).
16. **Standardowy format wag przy wydaniu.** `PollockSwiGLU` (sklejone
    gate+value w jednym `c_fc`) i custom loader v4 kosztowały nas patch w
    llama.cpp i ręczny split tensora. Rekomendacja dla gated-MLP: rozdzielone
    `gate_proj/up_proj/down_proj` w nazwach kompatybilnych z GPT-2/Llama
    toolingiem; dla v4: eksport finalnego ckpt do `GPT2LMHeadModel`/GGUF obok
    oryginalnego loadera.

## 6. Wydanie (checklista — u nas każdy brak = godziny pracy integratora)

- [ ] GGUF gotowe w repo (f16 + Q4_K_M) z sumami kontrolnymi,
- [ ] `chat_template` w `tokenizer_config.json` + `special_tokens_map.json`,
- [ ] preset samplingu w karcie,
- [ ] 10–20 zdań testowych z oczekiwanymi wyjściami (my używamy greedy parity
      HF↔llama.cpp — Wasz `inference-samples/r008/` jest wzorcem, przenieść to
      do każdego wydania),
- [ ] jasne FAQ licencyjne (CC-BY-SA i `mixed-upstream-dataset-terms` blokują
      produkcyjne użycie — rozważyć wariant Apache dla wag, jak w GoLLeM-45M),
- [ ] pomiar on-device (TTFT/tok/s/peak RAM) na referencyjnym sprzęcie — nasz
      punkt odniesienia: Watch4, 7,3 s TTFT / 0,49 tok/s / 204 MB dla 110M Q4;
      **każda nowa wersja powinna to przebić albo wyjaśnić dlaczego nie**.

## 7. Priorytety (gdyby robić tylko trzy rzeczy)

1. **SFT z replay + opublikowany `chat_template`** — natychmiastowy skok
   użyteczności na urządzeniu, niski koszt.
2. **Nowe dane dla 110M z korektą mixu (pkt 1) + domieszka EOS** — jakość i
   kończenie odpowiedzi w jednym przebiegu.
3. **RoPE + ctx 2048 + standardowy eksport (GGUF)** — odblokowuje ekosystem i
   przyszłe wydłużanie kontekstu.
