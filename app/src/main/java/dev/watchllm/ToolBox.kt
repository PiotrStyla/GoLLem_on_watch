// ToolBox.kt - declarative web lookups (port of WatchLLM/Tools/Tools.swift).
//
// The *code* here is generic and ships in the repo; the actual tool definitions
// live in an optional, git-ignored assets/Tools.json. With no such file present
// `specs` is empty, no network code ever runs and the app is fully offline.
//
// A tool runs BEFORE generation; its result becomes one sentence of "fact" that
// is folded into the prompt. The model itself never touches the network.

package dev.watchllm

import android.content.Context
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import org.json.JSONArray
import org.json.JSONObject

data class DerivedField(val name: String, val expr: String)

data class OAuthRefresh(
    val tokenURL: String,
    val clientId: String,
    val clientSecret: String?,
    val refreshToken: String,
)

data class WebToolSpec(
    val name: String,
    val triggers: List<String>,
    val url: String,
    val windowDays: Int?,
    val maxChars: Int?,
    val headers: Map<String, String>?,
    val fields: Map<String, List<String>>?,
    val jsonPath: List<String>?,
    val derived: List<DerivedField>?,
    val auth: OAuthRefresh?,
    val factTemplate: String,
)

class ToolException(message: String) : Exception(message)

object ToolBox {
    @Volatile
    private var loaded: List<WebToolSpec>? = null

    fun init(context: Context) {
        if (loaded != null) return
        loaded = try {
            val text = context.assets.open("Tools.json").bufferedReader().use { it.readText() }
            parse(text)
        } catch (e: Exception) {
            emptyList()
        }
    }

    val specs: List<WebToolSpec> get() = loaded ?: emptyList()
    val isEnabled: Boolean get() = specs.isNotEmpty()

    fun match(query: String): WebToolSpec? {
        val q = query.lowercase()
        return specs.firstOrNull { s -> s.triggers.any { q.contains(it.lowercase()) } }
    }

    // -- JSON -> spec -------------------------------------------------------

    private fun parse(text: String): List<WebToolSpec> {
        val arr = JSONArray(text)
        val out = ArrayList<WebToolSpec>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            fun strs(key: String): List<String>? =
                if (o.has(key)) o.getJSONArray(key).let { a -> (0 until a.length()).map { a.getString(it) } } else null
            val fields = mutableMapOf<String, List<String>>()
            o.optJSONObject("fields")?.let { f ->
                f.keys().forEach { k ->
                    val a = f.getJSONArray(k)
                    fields[k] = (0 until a.length()).map { a.getString(it) }
                }
            }
            val headers = mutableMapOf<String, String>()
            o.optJSONObject("headers")?.let { h -> h.keys().forEach { headers[it] = h.getString(it) } }
            val derived = mutableListOf<DerivedField>()
            o.optJSONArray("derived")?.let { d ->
                for (j in 0 until d.length()) {
                    val e = d.getJSONObject(j)
                    derived.add(DerivedField(e.getString("name"), e.getString("expr")))
                }
            }
            val auth = o.optJSONObject("auth")?.let {
                OAuthRefresh(
                    it.getString("tokenURL"), it.getString("clientId"),
                    it.optString("clientSecret").ifEmpty { null }, it.getString("refreshToken"),
                )
            }
            out.add(
                WebToolSpec(
                    name = o.getString("name"),
                    triggers = strs("triggers") ?: emptyList(),
                    url = o.getString("url"),
                    windowDays = if (o.has("windowDays")) o.getInt("windowDays") else null,
                    maxChars = if (o.has("maxChars")) o.getInt("maxChars") else null,
                    headers = headers.ifEmpty { null },
                    fields = fields.ifEmpty { null },
                    jsonPath = strs("jsonPath"),
                    derived = derived.ifEmpty { null },
                    auth = auth,
                    factTemplate = o.getString("factTemplate"),
                )
            )
        }
        return out
    }

    // -- execution ----------------------------------------------------------

    /** Walks `path` through a decoded JSON tree. */
    private fun resolve(root: Any?, path: List<String>): Any? {
        var node: Any? = root
        for (key in path) {
            node = when {
                key == "*" && node is JSONObject -> node.keys().asSequence().firstOrNull()?.let { node.get(it) }
                key == "*" && node is JSONArray -> if (node.length() > 0) node.get(0) else null
                key.toIntOrNull() != null && node is JSONArray ->
                    key.toInt().let { if (it in 0 until node.length()) node.get(it) else null }
                node is JSONObject -> node.opt(key)
                else -> return null
            }
            if (node == null) return null
        }
        return node
    }

    /** Strictly left-to-right expression over named numbers and literals,
     *  e.g. "subMin / total * 100". No operator precedence - small models cannot
     *  do arithmetic, so every figure must be computed here. */
    private fun evaluate(expr: String, nums: Map<String, Double>): Double? {
        val tokens = ArrayList<String>()
        val cur = StringBuilder()
        for (ch in expr) {
            if ("+-*/".indexOf(ch) >= 0) {
                tokens.add(cur.toString().trim()); cur.setLength(0)
                tokens.add(ch.toString())
            } else {
                cur.append(ch)
            }
        }
        tokens.add(cur.toString().trim())
        tokens.removeAll { it.isEmpty() }
        if (tokens.isEmpty()) return null
        var acc = nums[tokens[0]] ?: tokens[0].toDoubleOrNull() ?: return null
        var i = 1
        while (i + 1 < tokens.size) {
            val v = nums[tokens[i + 1]] ?: tokens[i + 1].toDoubleOrNull() ?: return null
            acc = when (tokens[i]) {
                "/" -> if (v == 0.0) return null else acc / v
                "*" -> acc * v
                "+" -> acc + v
                "-" -> acc - v
                else -> return null
            }
            i += 2
        }
        return acc
    }

    private fun httpGet(url: String, headers: Map<String, String>?): String {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 8000
        conn.readTimeout = 8000
        headers?.forEach { (k, v) -> conn.setRequestProperty(k, v) }
        try {
            val code = conn.responseCode
            if (code !in 200..299) throw ToolException("tool request failed (HTTP $code)")
            val out = ByteArrayOutputStream()
            conn.inputStream.use { it.copyTo(out) }
            return out.toString("UTF-8")
        } finally {
            conn.disconnect()
        }
    }

    private fun accessToken(auth: OAuthRefresh): String {
        val conn = URL(auth.tokenURL).openConnection() as HttpURLConnection
        conn.connectTimeout = 8000
        conn.readTimeout = 8000
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded")
        val body = buildString {
            append("client_id=").append(URLEncoder.encode(auth.clientId, "UTF-8"))
            append("&refresh_token=").append(URLEncoder.encode(auth.refreshToken, "UTF-8"))
            append("&grant_type=refresh_token")
            if (!auth.clientSecret.isNullOrEmpty()) {
                append("&client_secret=").append(URLEncoder.encode(auth.clientSecret, "UTF-8"))
            }
        }
        try {
            conn.outputStream.use { it.write(body.toByteArray()) }
            val code = conn.responseCode
            if (code !in 200..299) throw ToolException("could not refresh the OAuth token")
            val text = conn.inputStream.bufferedReader().use { it.readText() }
            return JSONObject(text).getString("access_token")
        } finally {
            conn.disconnect()
        }
    }

    /** Performs the lookup and returns a single sentence of fact to prepend to
     *  the prompt. */
    suspend fun run(spec: WebToolSpec, query: String = ""): String {
        var urlString = spec.url
        if (spec.windowDays != null) {
            val f = SimpleDateFormat("yyyy-MM-dd", Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }
            val end = Date()
            val start = Date(end.time - spec.windowDays * 86_400L)
            urlString = urlString
                .replace("{startDate}", f.format(start))
                .replace("{endDate}", f.format(end))
        }
        if (urlString.contains("{query}")) {
            var topic = query.lowercase()
            for (t in spec.triggers) if (topic.startsWith(t.lowercase())) topic = topic.substring(t.length)
            topic = topic.trim(' ', '?', '.', '!', ',')
            urlString = urlString.replace("{query}", URLEncoder.encode(topic, "UTF-8"))
        }
        val token = spec.auth?.let { accessToken(it) }
        val headers = spec.headers.orEmpty().toMutableMap()
        if (token != null) headers["Authorization"] = "Bearer $token"
        val json = JSONObject(httpGet(urlString, headers))

        val paths = spec.fields.orEmpty().toMutableMap()
        spec.jsonPath?.let { paths["value"] = it }
        if (paths.isEmpty()) throw ToolException("tool response did not contain the expected field")

        val text = mutableMapOf<String, String>()
        val nums = mutableMapOf<String, Double>()
        for ((name, path) in paths) {
            val node = resolve(json, path)
                ?: throw ToolException("tool response did not contain the expected field")
            var raw = node.toString()
            val limit = spec.maxChars
            if (limit != null && raw.length > limit) {
                var cut = raw.substring(0, limit)
                cut = cut.substringBeforeLast('.', cut).let { if (it.length < cut.length) "$it." else it }
                raw = cut
            }
            raw = formatted(raw)
            text[name] = raw
            raw.replace(",", "").toDoubleOrNull()?.let { nums[name] = it }
        }
        for (field in spec.derived ?: emptyList()) {
            val v = evaluate(field.expr, nums) ?: continue
            nums[field.name] = v
            // Ratios read better with one decimal; counts read better grouped.
            text[field.name] =
                if (v < 100) "%.1f".format(Locale.US, v)
                else formatted(Math.round(v).toString())
        }

        var out = spec.factTemplate
        for ((name, value) in text) out = out.replace("{$name}", value)
        return out
    }

    /** Groups digits so a subscriber count reads naturally when spoken back. */
    private fun formatted(raw: String): String {
        val n = raw.toLongOrNull() ?: return raw
        return String.format(Locale.US, "%,d", n)
    }
}
