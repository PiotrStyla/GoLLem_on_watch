// MainActivity.kt - hosts the Compose UI and forwards lifecycle to the runner:
// park at a token boundary when backgrounded, resume exactly on foreground.

package dev.watchllm

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import dev.watchllm.ui.WatchLLMApp

class MainActivity : ComponentActivity() {

    private lateinit var runner: LlmRunner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ToolBox.init(applicationContext)
        runner = LlmRunner(applicationContext, intent.getBooleanExtra("bench", false))
        setContent { WatchLLMApp(runner) }
        runner.load()
    }

    override fun onStart() {
        super.onStart()
        if (::runner.isInitialized) runner.resumeIfParked()
    }

    override fun onStop() {
        if (::runner.isInitialized) runner.park()
        super.onStop()
    }

    override fun onDestroy() {
        if (::runner.isInitialized && isFinishing) runner.close()
        super.onDestroy()
    }
}
