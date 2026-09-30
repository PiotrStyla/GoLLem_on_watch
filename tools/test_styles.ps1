# Run both prompt styles against the watch app and print what the model
# answered. Run this whenever the watch is awake (ideally on the charger with
# the screen on - Wear OS sleeps the adb listener when the display dies).
#
#   .\tools\test_styles.ps1
#
# Prints the OUTPUT log lines for:
#   1) question style:     "Jaka jest stolica Polski?"
#   2) completion style:   "Stolica Polski to"
# and pulls ui_q1.png / ui_q2.png screenshots into the project root.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Adb = ".toolchain\android-sdk\platform-tools\adb.exe"
if (-not (Test-Path $Adb)) { $Adb = "adb" }

# The watch is already paired; wait up to 3 min for it to surface via mDNS.
$Serial = $null
for ($i = 0; $i -lt 36; $i++) {
    $line = (& $Adb devices | Select-String "`tdevice" | Select-Object -First 1)
    if ($line) { $Serial = ($line -split "`t")[0]; break }
    Start-Sleep -Seconds 5
}
if (-not $Serial) { throw "watch not visible - wake it up / put it on the charger and retry" }
Write-Host "watch: $Serial"

function Invoke-Style([string]$Prompt, [string]$Shot) {
    & $Adb -s $Serial shell input keyevent KEYCODE_WAKEUP | Out-Null
    & $Adb -s $Serial shell am force-stop dev.watchllm | Out-Null
    & $Adb -s $Serial shell am start -n dev.watchllm/.MainActivity --es prompt "$Prompt" | Out-Null
    # ~60 s: model load (~10 s) + generation; keep the display awake meanwhile
    for ($i = 0; $i -lt 12; $i++) {
        Start-Sleep -Seconds 5
        & $Adb -s $Serial shell input keyevent KEYCODE_WAKEUP | Out-Null
    }
    & $Adb -s $Serial exec-out screencap -p | Set-Content -Encoding Byte $Shot
    Write-Host ""
    Write-Host "== $Prompt"
    (& $Adb -s $Serial logcat -d -s WatchLLM:V | Select-String "BENCH|OUTPUT" | Select-Object -Last 2)
}

Invoke-Style "Jaka jest stolica Polski?" "ui_q1.png"
Invoke-Style "Stolica Polski to"        "ui_q2.png"
