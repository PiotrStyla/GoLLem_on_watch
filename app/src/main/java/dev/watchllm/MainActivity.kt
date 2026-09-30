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
        // Streaming text is read off the screen; also avoids One UI freezing the
        // process (D-state) the moment the display sleeps (verified on SM-R865F).
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        ToolBox.init(applicationContext)
        val prompt = intent.getStringExtra("prompt")
            ?: if (intent.getBooleanExtra("bench", false)) dev.watchllm.AppConfig.benchPrompt else null
        runner = LlmRunner(applicationContext, prompt)
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
