// WatchLLMApp.kt - Compose UI (port of ContentView.swift): model picker, ask
// control with system voice/keyboard input, streaming output with auto-scroll,
// tool note and live benchmark stats.

package dev.watchllm.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.watchllm.AppConfig
import dev.watchllm.LlmRunner
import dev.watchllm.ModelSpec

@Composable
fun WatchLLMApp(runner: LlmRunner) {
    val ui by runner.ui.collectAsState()
    var query by remember { mutableStateOf("") }
    var picking by remember { mutableStateOf(false) }
    val scroll = rememberScrollState()

    MaterialTheme {
        Surface(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scroll)
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                // model row
                TextButton(
                    onClick = { picking = true },
                    enabled = ui.state != LlmRunner.State.GENERATING,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("[CPU] ${ui.model.name}", fontSize = 13.sp)
                    Spacer(Modifier.weight(1f))
                    Text(">", fontSize = 13.sp)
                }

                // ask control
                when (ui.state) {
                    LlmRunner.State.IDLE, LlmRunner.State.LOADING ->
                        Text("Loading model...", style = MaterialTheme.typography.bodySmall)
                    LlmRunner.State.FAILED ->
                        Text(ui.error ?: "failed", color = MaterialTheme.colorScheme.error,
                             style = MaterialTheme.typography.bodySmall)
                    LlmRunner.State.GENERATING ->
                        Button(onClick = { runner.stop() }) { Text("Stop") }
                    LlmRunner.State.PARKED ->
                        Text("Paused - app in background", style = MaterialTheme.typography.bodySmall)
                    LlmRunner.State.READY -> {
                        OutlinedTextField(
                            value = query,
                            onValueChange = { query = it },
                            label = { Text("Ask (keyboard or voice)") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                        )
                        Button(
                            onClick = {
                                val t = query.trim()
                                if (t.isNotEmpty()) runner.generate(t)
                            },
                            enabled = query.isNotBlank(),
                        ) { Text("Ask") }
                    }
                }

                if (query.isNotEmpty() && ui.state == LlmRunner.State.READY) {
                    Text(query, fontWeight = FontWeight.SemiBold,
                         style = MaterialTheme.typography.bodySmall)
                }

                ui.toolNote?.let {
                    Text("[tool] $it", color = MaterialTheme.colorScheme.primary,
                         style = MaterialTheme.typography.labelSmall)
                }

                if (ui.output.isNotEmpty()) {
                    Text(ui.output, fontSize = 15.sp)
                }

                Spacer(Modifier.height(1.dp))
                if (ui.stats.generatedTokens > 0) {
                    StatsView(ui)
                }
            }
        }
    }

    LaunchedEffect(ui.output) {
        if (ui.output.isNotEmpty()) scroll.animateScrollTo(scroll.maxValue)
    }

    if (picking) {
        AlertDialog(
            onDismissRequest = { picking = false },
            title = { Text("Model") },
            text = {
                Column {
                    AppConfig.models.forEach { spec: ModelSpec ->
                        TextButton(onClick = {
                            runner.select(spec)
                            picking = false
                        }) {
                            Column(Modifier.fillMaxWidth()) {
                                Text(spec.name, fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
                                Text(spec.detail, style = MaterialTheme.typography.labelSmall)
                            }
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { picking = false }) { Text("Close") }
            },
        )
    }
}

@Composable
private fun StatsView(ui: dev.watchllm.LlmRunner.UiState) {
    val s = ui.stats
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        HorizontalDivider()
        StatRow("tokens/sec", if (s.tokensPerSecond > 0) "%.2f".format(s.tokensPerSecond) else "-")
        StatRow("first token", if (s.timeToFirstTokenS > 0) "%.2f s".format(s.timeToFirstTokenS) else "-")
        StatRow("generated", "${s.generatedTokens} tok")
        StatRow("prompt", "${s.promptTokens} tok")
        StatRow("peak memory", if (s.peakFootprintMB > 0) "%.0f MB".format(s.peakFootprintMB) else "-")
        StatRow("weights", "%.0f MB".format(s.weightsMB))
        StatRow("kv cache", "%.0f MB".format(s.contextMB))
        StatRow("threads", "${s.threads}")
    }
}

@Composable
private fun StatRow(k: String, v: String) {
    Row {
        Text(k, fontSize = 12.sp, fontFamily = FontFamily.Monospace,
             color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.weight(1f))
        Text(v, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
    }
}
