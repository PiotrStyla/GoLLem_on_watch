// greedy_probe.cpp - host verification tool. Loads a GGUF, tokenizes a prompt
// and prints the greedy continuation as token ids. tools/parity_check.py diffs
// these against HuggingFace `generate(do_sample=False)` on the same prompt.
//
//   greedy_probe <model.gguf> <prompt> [max_tokens]

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt> [max_tokens]\n", argv[0]);
        return 2;
    }
    const char * path  = argv[1];
    const char * prompt = argv[2];
    const int    n_gen = argc > 3 ? atoi(argv[3]) : 32;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 1024;
    cparams.n_batch  = 32;
    cparams.n_ubatch = 32;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks(strlen(prompt) + 16);
    const int32_t n = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
                                     toks.data(), (int32_t)toks.size(), false, true);
    if (n <= 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }
    toks.resize((size_t)n);

    printf("PROMPT_TOKENS");
    for (int32_t i = 0; i < n; ++i) printf(" %d", toks[(size_t)i]);
    printf("\n");

    for (int32_t i = 0; i < n; i += 32) {
        const int32_t take = n - i < 32 ? n - i : 32;
        llama_batch batch = llama_batch_get_one(toks.data() + i, take);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at prompt chunk %d\n", i);
            return 1;
        }
    }

    llama_sampler * sampler = llama_sampler_init_greedy();

    printf("GEN_TOKENS");
    llama_token cur = llama_sampler_sample(sampler, ctx, -1);
    for (int i = 0; i < n_gen; ++i) {
        if (llama_vocab_is_eog(vocab, cur)) break;
        printf(" %d", cur);
        llama_token t = cur;
        llama_batch batch = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, batch) != 0) break;
        cur = llama_sampler_sample(sampler, ctx, -1);
    }
    printf("\n");

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
