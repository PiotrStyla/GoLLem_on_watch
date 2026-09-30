// LlmEngine.kt - Kotlin half of the JNI bridge (llm_jni.cpp). Thin, stateless
// wrapper; concurrency is handled by LlmRunner.

package dev.watchllm

class LlmEngine {
    external fun nativeInit(modelPath: String, nCtx: Int, nBatch: Int, nThreads: Int): String?
    external fun nativeUnload()
    external fun nativeReset()
    external fun nativeSetSampling(factual: Boolean)
    external fun nativeTokenizeChat(prompt: String, context: String?, system: String?): IntArray
    /** Batched prompt eval; returns the first sampled token or -1. */
    external fun nativePrompt(tokens: IntArray): Int
    external fun nativeStep(token: Int): Int
    external fun nativePiece(token: Int): ByteArray?
    external fun nativeIsStop(token: Int): Boolean
    external fun nativeWeightsMb(): Double
    external fun nativeContextMb(): Double
    external fun nativeThreads(): Int

    companion object {
        init {
            System.loadLibrary("watchllm")
        }
    }
}
