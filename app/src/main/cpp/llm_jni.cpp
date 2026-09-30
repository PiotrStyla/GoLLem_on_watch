// JNI bridge over upstream llama.cpp - Android counterpart of
// WatchLLM/LlamaCppEngine.swift. One model/context/sampler at a time; all calls
// come from a single inference dispatcher in LlmRunner.
//
// Special-token literals are assembled from fragments on purpose: the exact
// marker strings are also special to the tool harness this project is authored
// in. The values are the standard pipe-style GPT-2/Qwen markers:
//   ENDOFTEXT = "<" + "|endoftext|" + ">"   (renders: pipe-endoftext)
//   IM_START  = "<" + "|im_start|"  + ">"
//   IM_END    = "<" + "|im_end|"    + ">"

#include <jni.h>
#include <sys/stat.h>

#include <algorithm>
#include <ctime>
#include <string>
#include <vector>

#include "llama.h"

namespace {

llama_model *       g_model  = nullptr;
llama_context *     g_ctx    = nullptr;
llama_sampler *     g_sampler = nullptr;
const llama_vocab * g_vocab  = nullptr;

llama_context_params g_cparams;

double g_weights_mb = 0.0;
int    g_threads    = 0;
int    g_im_end_tok = -1;

std::string tok_marker(const char * inner) {
    // "<" + inner + ">" with inner like "|im_end|"
    std::string s = "<";
    s += inner;
    s += ">";
    return s;
}

std::string jstr(JNIEnv * env, jstring s) {
    if (!s) return {};
    const char * c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

void free_sampler() {
    if (g_sampler) llama_sampler_free(g_sampler);
    g_sampler = nullptr;
}

void unload_locked() {
    free_sampler();
    if (g_ctx)   llama_free(g_ctx);
    if (g_model) llama_model_free(g_model);
    g_ctx   = nullptr;
    g_model = nullptr;
    g_vocab = nullptr;
}

int32_t token_for_text(const std::string & s) {
    if (!g_vocab) return -1;
    llama_token buf[8];
    const int32_t n = llama_tokenize(g_vocab, s.c_str(), (int32_t)s.size(), buf, 8, false, true);
    return n == 1 ? buf[0] : -1;
}

// Feed tokens to the model in n_batch chunks and sample the first output token.
int32_t eval_tokens(llama_token * toks, int32_t n) {
    if (!g_ctx || !g_sampler || n <= 0) return -1;
    const int32_t n_batch = (int32_t)g_cparams.n_batch;
    for (int32_t i = 0; i < n; i += n_batch) {
        const int32_t take = std::min(n_batch, n - i);
        llama_batch batch = llama_batch_get_one(toks + i, take);
        if (llama_decode(g_ctx, batch) != 0) return -1;
    }
    return llama_sampler_sample(g_sampler, g_ctx, -1);
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_dev_watchllm_LlmEngine_nativeUnload(JNIEnv *, jobject) {
    unload_locked();
}

JNIEXPORT jstring JNICALL
Java_dev_watchllm_LlmEngine_nativeInit(JNIEnv * env, jobject, jstring jpath,
                                       jint n_ctx, jint n_batch, jint n_threads) {
    unload_locked();

    static bool backend_ready = false;
    if (!backend_ready) {
        llama_backend_init();
        backend_ready = true;
    }

    const std::string path = jstr(env, jpath);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_model) {
        return env->NewStringUTF(("llama.cpp could not load " + path).c_str());
    }

    g_cparams = llama_context_default_params();
    g_cparams.n_ctx           = (uint32_t)n_ctx;
    // One token is decoded at a time, so a large batch only inflates the
    // compute buffers; the prompt still arrives in n_batch chunks.
    g_cparams.n_batch         = (uint32_t)n_batch;
    g_cparams.n_ubatch        = (uint32_t)n_batch;
    g_cparams.n_threads       = (int32_t)n_threads;
    g_cparams.n_threads_batch = (int32_t)n_threads;
    g_cparams.no_perf         = true;

    g_ctx = llama_init_from_model(g_model, g_cparams);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        return env->NewStringUTF("could not allocate the inference context");
    }

    g_vocab      = llama_model_get_vocab(g_model);
    g_im_end_tok = token_for_text(tok_marker("|im_end|"));

    struct stat st {};
    g_weights_mb = 0.0;
    if (stat(path.c_str(), &st) == 0) g_weights_mb = (double)st.st_size / 1e6;
    g_threads = n_threads;

    return nullptr;  // success
}

JNIEXPORT void JNICALL
Java_dev_watchllm_LlmEngine_nativeReset(JNIEnv *, jobject) {
    if (g_ctx) llama_memory_clear(llama_get_memory(g_ctx), true);
}

JNIEXPORT void JNICALL
Java_dev_watchllm_LlmEngine_nativeSetSampling(JNIEnv *, jobject, jboolean factual) {
    free_sampler();
    g_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (factual) {
        // Tool answers must be relayed verbatim; sampling only adds drift.
        llama_sampler_chain_add(g_sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(g_sampler, llama_sampler_init_dist((uint32_t)time(nullptr)));
    }
}

JNIEXPORT jintArray JNICALL
Java_dev_watchllm_LlmEngine_nativeTokenizeChat(JNIEnv * env, jobject,
                                               jstring jprompt, jstring jcontext,
                                               jstring jsystem) {
    jintArray empty = env->NewIntArray(0);
    if (!g_vocab) return empty;

    const std::string marker_sys  = tok_marker("|im_start|");
    const std::string marker_end  = tok_marker("|im_end|");

    std::string text;
    const std::string context = jstr(env, jcontext);
    const std::string system  = jstr(env, jsystem);
    if (!context.empty() || !system.empty()) {
        text += marker_sys + "system\n";
        if (!context.empty()) text += context + " ";
        text += system;
        text += marker_end + "\n";
    }
    text += marker_sys + "user\n" + jstr(env, jprompt) + marker_end + "\n";
    text += marker_sys + "assistant\n";

    std::vector<llama_token> toks(text.size() + 16);
    const int32_t n = llama_tokenize(g_vocab, text.c_str(), (int32_t)text.size(),
                                     toks.data(), (int32_t)toks.size(),
                                     false /* add_special */, true /* parse_special */);
    if (n <= 0) return empty;

    jintArray out = env->NewIntArray(n);
    std::vector<jint> tmp((size_t)n);
    for (int32_t i = 0; i < n; ++i) tmp[(size_t)i] = toks[(size_t)i];
    env->SetIntArrayRegion(out, 0, n, tmp.data());
    return out;
}

JNIEXPORT jint JNICALL
Java_dev_watchllm_LlmEngine_nativePrompt(JNIEnv * env, jobject, jintArray jtoks) {
    if (!g_ctx || !jtoks) return -1;
    const jsize n = env->GetArrayLength(jtoks);
    if (n <= 0) return -1;
    std::vector<llama_token> toks((size_t)n);
    env->GetIntArrayRegion(jtoks, 0, n, reinterpret_cast<jint *>(toks.data()));
    return eval_tokens(toks.data(), (int32_t)n);
}

JNIEXPORT jint JNICALL
Java_dev_watchllm_LlmEngine_nativeStep(JNIEnv *, jobject, jint token) {
    if (!g_ctx || !g_sampler) return -1;
    llama_token t = (llama_token)token;
    llama_batch batch = llama_batch_get_one(&t, 1);
    if (llama_decode(g_ctx, batch) != 0) return -1;
    return (jint)llama_sampler_sample(g_sampler, g_ctx, -1);
}

JNIEXPORT jbyteArray JNICALL
Java_dev_watchllm_LlmEngine_nativePiece(JNIEnv * env, jobject, jint token) {
    jbyteArray empty = env->NewByteArray(0);
    if (!g_vocab) return empty;
    char buf[256];
    const int32_t n = llama_token_to_piece(g_vocab, (llama_token)token, buf, (int32_t)sizeof(buf), 0, false);
    if (n <= 0) return empty;
    jbyteArray out = env->NewByteArray(n);
    env->SetByteArrayRegion(out, 0, n, reinterpret_cast<const jbyte *>(buf));
    return out;
}

JNIEXPORT jboolean JNICALL
Java_dev_watchllm_LlmEngine_nativeIsStop(JNIEnv *, jobject, jint token) {
    if (!g_vocab) return JNI_FALSE;
    const llama_token t = (llama_token)token;
    if (llama_vocab_is_eog(g_vocab, t)) return JNI_TRUE;
    return t == g_im_end_tok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jdouble JNICALL
Java_dev_watchllm_LlmEngine_nativeWeightsMb(JNIEnv *, jobject) {
    return g_weights_mb;
}

JNIEXPORT jdouble JNICALL
Java_dev_watchllm_LlmEngine_nativeContextMb(JNIEnv *, jobject) {
    if (!g_ctx || !g_model) return 0.0;
    // No byte-size API for the memory object; compute the f16 KV cache like
    // LlamaCppEngine.swift does: n_ctx x n_layer x (K+V) x n_head_kv x head_dim x 2 B.
    const int32_t n_layer   = llama_model_n_layer(g_model);
    const int32_t n_head_kv = llama_model_n_head_kv(g_model);
    const int32_t head_dim  = llama_model_n_embd(g_model) / llama_model_n_head(g_model);
    const double bytes = (double)g_cparams.n_ctx * n_layer * 2.0 * n_head_kv * head_dim * 2.0;
    return bytes / 1e6;
}

JNIEXPORT jint JNICALL
Java_dev_watchllm_LlmEngine_nativeThreads(JNIEnv *, jobject) {
    return g_threads;
}

}  // extern "C"
