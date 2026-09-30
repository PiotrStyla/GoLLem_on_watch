// LlmRunner.kt - drives llama.cpp on a background dispatcher and publishes UI
// state. Port of WatchLLM/LLMRunner.swift: same states, same tool contract
// (a web tool runs before decoding and its fact is folded into the prompt),
// same park/resume behaviour for backgrounding.

package dev.watchllm

import android.content.Context
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class LlmRunner(private val context: Context, private val autorun: Boolean = false) {

    enum class State { IDLE, LOADING, READY, GENERATING, PARKED, FAILED }

    data class Stats(
        val promptTokens: Int = 0,
        val generatedTokens: Int = 0,
        val timeToFirstTokenS: Double = 0.0,
        val tokensPerSecond: Double = 0.0,
        val peakFootprintMB: Double = 0.0,
        val weightsMB: Double = 0.0,
        val contextMB: Double = 0.0,
        val threads: Int = 0,
    )

    data class UiState(
        val state: State = State.IDLE,
        val output: String = "",
        val stats: Stats = Stats(),
        val model: ModelSpec = AppConfig.defaultModel,
        val toolNote: String? = null,
        val error: String? = null,
    )

    private val _ui = MutableStateFlow(UiState())
    val ui: StateFlow<UiState> = _ui

    private val engine = LlmEngine()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val cancelFlag = AtomicBoolean(false)
    private val parkFlag = AtomicBoolean(false)

    /** Snapshot of an interrupted run; only touched on the inference dispatcher. */
    private class Run(
        var stopAtSentence: Boolean = false,
        var answer: String = "",
        var next: Int = -1,
        var remaining: Int = 0,
        var generated: Int = 0,
        val decoder: Utf8Stream = Utf8Stream(),
        val startedAt: Double = nowS(),
        var firstTokenAt: Double? = null,
    )

    private var parked: Run? = null

    // MARK: lifecycle

    fun load() = load(AppConfig.defaultModel)

    fun select(spec: ModelSpec) {
        val cur = _ui.value
        if (spec == cur.model || cur.state == State.GENERATING) return
        _ui.value = UiState(model = spec)
        load(spec)
    }

    private fun load(spec: ModelSpec) {
        if (_ui.value.state != State.IDLE && _ui.value.state != State.FAILED) return
        _ui.value = _ui.value.copy(state = State.LOADING, error = null, output = "", toolNote = null)
        scope.launch {
            try {
                val file = resolveModelFile(context, spec)
                // Watch SoCs are dual/quad-core little parts; more threads than
                // cores just adds contention.
                val threads = max(1, minOf(2, Runtime.getRuntime().availableProcessors()))
                val err = engine.nativeInit(
                    file.absolutePath, AppConfig.contextTokens, AppConfig.batchTokens, threads
                )
                if (err != null) throw EngineError.Load(err)
                val w = engine.nativeWeightsMb()
                val c = engine.nativeContextMb()
                val t = engine.nativeThreads()
                _ui.value = _ui.value.copy(
                    state = State.READY,
                    stats = _ui.value.stats.copy(weightsMB = w, contextMB = c, threads = t),
                )
                Log.i(TAG, "loaded: weights=${"%.1f".format(w)} MB ctx=${"%.1f".format(c)} MB threads=$t")
                // FH1_AUTORUN equivalent: generate the bench prompt on launch
                // for repeatable measurement runs (`adb shell am start ... --ez bench true`).
                if (autorun) generate(AppConfig.benchPrompt)
            } catch (e: Exception) {
                _ui.value = _ui.value.copy(state = State.FAILED, error = e.message ?: "load failed")
            }
        }
    }

    // MARK: generation

    fun generate(prompt: String) {
        if (_ui.value.state != State.READY) return
        cancelFlag.set(false)
        parkFlag.set(false)
        _ui.value = _ui.value.copy(
            state = State.GENERATING, output = "", toolNote = null, error = null,
            stats = _ui.value.stats.copy(
                generatedTokens = 0, tokensPerSecond = 0.0,
                timeToFirstTokenS = 0.0, peakFootprintMB = 0.0,
            ),
        )

        // A matching web tool runs here, before any decoding starts, and its
        // result is folded into the prompt. Inference itself is always offline.
        val spec = ToolBox.match(prompt)
        if (spec != null) {
            scope.launch {
                try {
                    val fact = ToolBox.run(spec, prompt)
                    Log.i(TAG, "TOOL ${spec.name} -> $fact")
                    _ui.value = _ui.value.copy(toolNote = spec.name)
                    startDecoding(prompt, fact)
                } catch (e: Exception) {
                    // Fail closed: a question that needs live data, answered with
                    // no live data, produces confident invented numbers.
                    Log.e(TAG, "TOOL ${spec.name} failed: ${e.message}")
                    _ui.value = _ui.value.copy(
                        state = State.READY,
                        toolNote = "${spec.name} unavailable",
                        output = "Couldn't fetch ${spec.name}, so I didn't ask the model - " +
                            "it would have made the numbers up.\n\n${e.message}",
                    )
                }
            }
        } else {
            scope.launch { startDecoding(prompt, null) }
        }
    }

    private suspend fun startDecoding(prompt: String, context: String?) {
        parked = null
        engine.nativeReset()
        engine.nativeSetSampling(context != null)

        val toks = engine.nativeTokenizeChat(buildPrompt(prompt, context))
        if (toks.isEmpty()) {
            _ui.value = _ui.value.copy(state = State.FAILED, error = "prompt produced no tokens")
            return
        }
        val started = nowS()
        val first = engine.nativePrompt(toks)
        if (first < 0) {
            _ui.value = _ui.value.copy(
                state = State.FAILED, error = "prompt is longer than the context window")
            return
        }
        _ui.value = _ui.value.copy(stats = _ui.value.stats.copy(promptTokens = toks.size))
        runLoop(Run(
            stopAtSentence = context != null, next = first,
            remaining = AppConfig.maxNewTokens, startedAt = started,
        ))
    }

    fun stop() {
        cancelFlag.set(true)
        if (_ui.value.state == State.PARKED) _ui.value = _ui.value.copy(state = State.READY)
    }

    /** Android can freeze a backgrounded app; stop at a token boundary instead
     *  of racing it. The KV cache stays resident so resuming is exact. */
    fun park() {
        if (_ui.value.state != State.GENERATING) return
        parkFlag.set(true)
        _ui.value = _ui.value.copy(state = State.PARKED)
    }

    fun resumeIfParked() {
        if (_ui.value.state != State.PARKED) return
        parkFlag.set(false)
        _ui.value = _ui.value.copy(state = State.GENERATING)
        val run = parked
        scope.launch {
            if (run == null) {
                _ui.value = _ui.value.copy(state = State.READY)
            } else {
                parked = null
                runLoop(run)
            }
        }
    }

    /** Formats the user question (and optional tool fact) for the loaded model.
     *  Pollock: Qwen-style markers it was tokenized with; GoLLeM: plain
     *  continuation (its SFT template is not public - do not invent one). */
    private fun buildPrompt(prompt: String, context: String?): String {
        val spec = _ui.value.model
        return when (spec.format) {
            dev.watchllm.PromptFormat.IM_CHAT -> {
                val start = "<" + "|im_start|" + ">"
                val end = "<" + "|im_end|" + ">"
                buildString {
                    if (context != null || spec.systemPrompt != null) {
                        append(start).append("system\n")
                        if (context != null) append(context).append(' ')
                        append(spec.systemPrompt.orEmpty()).append(end).append('\n')
                    }
                    append(start).append("user\n").append(prompt).append(end).append('\n')
                    append(start).append("assistant\n")
                }
            }
            dev.watchllm.PromptFormat.PLAIN -> buildString {
                if (context != null) append(context).append("\n\n")
                append(prompt)
            }
        }
    }

    /** True once the text ends on a sentence terminator. The digit guard keeps
     *  "25.7" from looking like the end of a sentence. */
    private fun endsSentence(text: String): Boolean {
        if (text.length < 20) return false
        val t = text.trim()
        if (t.isEmpty() || ".!?".indexOf(t.last()) < 0) return false
        return !(t.length >= 2 && t[t.length - 2].isDigit())
    }

    private suspend fun runLoop(start: Run) {
        var r = start
        val pending = StringBuilder()
        var lastFlush = 0.0
        var peak = 0.0
        var emitted = 0

        suspend fun flush(force: Boolean) {
            val t = nowS()
            // Re-layout is expensive on a watch; back off as the transcript grows.
            val interval =
                if (emitted < 1200) 0.08 else if (emitted < 4000) 0.20 else 0.40
            if (!force && t - lastFlush <= interval) return
            lastFlush = t
            peak = max(peak, footprintMb())

            val chunk = pending.toString()
            pending.clear()
            emitted += chunk.length
            val generated = r.generated
            val ttft = (r.firstTokenAt ?: t) - r.startedAt
            val elapsed = t - (r.firstTokenAt ?: t)
            val rate = if (elapsed > 0) max(0, generated - 1).toDouble() / elapsed else 0.0

            _ui.value = _ui.value.copy(
                output = _ui.value.output + chunk,
                stats = _ui.value.stats.copy(
                    generatedTokens = generated,
                    tokensPerSecond = rate,
                    timeToFirstTokenS = ttft,
                    peakFootprintMB = max(_ui.value.stats.peakFootprintMB, peak),
                ),
            )
        }

        while (r.remaining > 0) {
            if (cancelFlag.get()) {
                pending.append(r.decoder.drain()); flush(true)
                _ui.value = _ui.value.copy(state = State.READY)
                return
            }
            if (parkFlag.get()) {
                flush(true)
                parked = r
                return
            }

            val tok = r.next
            if (engine.nativeIsStop(tok)) break
            if (r.firstTokenAt == null) r.firstTokenAt = nowS()

            val bytes = engine.nativePiece(tok)
            if (bytes != null && bytes.isNotEmpty()) {
                val piece = r.decoder.push(bytes)
                pending.append(piece)
                r.answer += piece
            }
            r.generated++
            r.remaining--

            if (r.stopAtSentence && endsSentence(r.answer)) break

            val next = engine.nativeStep(tok)
            if (next < 0) break
            r.next = next
            flush(false)
        }

        pending.append(r.decoder.drain())
        flush(true)
        if (_ui.value.state == State.GENERATING) _ui.value = _ui.value.copy(state = State.READY)
        val s = _ui.value.stats
        Log.i(TAG, "BENCH tok/s=${"%.3f".format(s.tokensPerSecond)} " +
            "ttft=${"%.3f".format(s.timeToFirstTokenS)}s gen=${s.generatedTokens} " +
            "prompt=${s.promptTokens} peak=${"%.1f".format(s.peakFootprintMB)}MB " +
            "threads=${s.threads}")
    }

    fun close() {
        engine.nativeUnload()
    }

    companion object {
        private const val TAG = "WatchLLM"

        private fun nowS(): Double = System.nanoTime() / 1e9
    }
}
