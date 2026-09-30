// Support.kt - model specs, config and small helpers (port of Support.swift).

package dev.watchllm

import android.os.Debug
import java.io.File

data class ModelSpec(
    val id: String,
    val name: String,
    val file: String,
    val detail: String,
    /** Pollock is a raw base model; a system turn only appears when a tool or the
     *  user config supplies one. */
    val systemPrompt: String? = null,
)

object AppConfig {
    val models = listOf(
        ModelSpec(
            id = "pollock-q4",
            name = "Pollock 125M Q4",
            file = "pollock-mini-lm-125m-Q4_K_M.gguf",
            detail = "GPT-2 + SwiGLU - 88 MB",
        ),
        ModelSpec(
            id = "pollock-q8",
            name = "Pollock 125M Q8",
            file = "pollock-mini-lm-125m-Q8_0.gguf",
            detail = "GPT-2 + SwiGLU - 135 MB",
        ),
    )
    val defaultModel = models.first()

    /** KV cache length in tokens. The model itself is trained for 2048; 1024
     *  keeps the KV cache around 50 MB, which fits every watch we target. */
    const val contextTokens = 1024
    /** Prompt eval arrives in chunks of this many tokens. */
    const val batchTokens = 32
    const val maxNewTokens = 512
    const val benchPrompt = "What is the sun?"
}

sealed class EngineError(message: String) : Exception(message) {
    class NotAvailable(file: String) :
        EngineError("$file.gguf is missing - run tools/fetch_model first")
    class Load(msg: String) : EngineError(msg)
}

/** Resident set size of this process in MB - the closest Android analogue of
 *  watchOS `phys_footprint`. */
fun footprintMb(): Double {
    val mi = Debug.MemoryInfo()
    Debug.getMemoryInfo(mi)
    return mi.totalPss / 1024.0
}

/**
 * A token can end mid-UTF-8. Buffer partial scalars so the UI never shows a
 * replacement character. (Port of UTF8Stream from Support.swift.)
 */
class Utf8Stream {
    private val buf = ArrayList<Byte>()

    fun push(bytes: ByteArray): String {
        bytes.forEach { buf.add(it) }
        val split = completePrefixLength(buf)
        if (split == 0) return ""
        val head = buf.subList(0, split).toByteArray()
        repeat(split) { buf.removeAt(0) }
        return String(head, Charsets.UTF_8)
    }

    fun drain(): String {
        if (buf.isEmpty()) return ""
        val out = buf.toByteArray()
        buf.clear()
        return String(out, Charsets.UTF_8)
    }

    private fun completePrefixLength(b: List<Byte>): Int {
        var i = b.size
        var scanned = 0
        while (i > 0 && scanned < 4) {
            val c = b[i - 1].toInt() and 0xFF
            if (c and 0x80 == 0) return i
            if (c and 0xC0 == 0xC0) {
                val need = if (c >= 0xF0) 4 else if (c >= 0xE0) 3 else 2
                return if (b.size - (i - 1) >= need) b.size else i - 1
            }
            i--
            scanned++
        }
        return b.size
    }
}

/** Resolves a model file: external dir first (adb push), then internal, then a
 *  copy of the bundled asset. */
fun resolveModelFile(context: android.content.Context, spec: ModelSpec): File {
    val candidates = listOf(
        File(context.getExternalFilesDir(null), "models/${spec.file}"),
        File(context.filesDir, "models/${spec.file}"),
    )
    candidates.firstOrNull { it.exists() }?.let { return it }

    // models/bundled/ is an assets root dir, so files sit at the asset root.
    val assetPath = spec.file
    val out = candidates[1]
    out.parentFile?.mkdirs()
    try {
        context.assets.open(assetPath).use { input ->
            out.outputStream().use { output -> input.copyTo(output) }
        }
    } catch (e: Exception) {
        throw EngineError.NotAvailable(spec.file.removeSuffix(".gguf"))
    }
    return out
}
