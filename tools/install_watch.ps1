# Install WatchLLM on a Galaxy Watch (Wear OS 3+) over Wi-Fi adb.
#
# 1) On the watch: Settings -> About watch -> tap "Software version"/"Build" 5x,
#    then Settings -> Developer options -> ON: "ADB debugging" + "Debug over Wi-Fi".
#    Two addresses appear: pairing (under "Pair device with pairing code")
#    and connect (next to "Debug over Wi-Fi").
#
# Usage:
#   .\tools\install_watch.ps1 pair    192.168.0.23:37123   # once per watch; type the 6-digit code shown on the watch
#   .\tools\install_watch.ps1 install 192.168.0.23:37123   # connect + install APK + start bench
#
param(
    [Parameter(Mandatory = $true)][ValidateSet("pair", "install")][string]$Mode,
    [Parameter(Mandatory = $true)][string]$Address
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Adb = ".toolchain\android-sdk\platform-tools\adb.exe"
if (-not (Test-Path $Adb)) { $Adb = "adb" }
$Apk = "app\build\outputs\apk\debug\app-debug.apk"

switch ($Mode) {
    "pair" {
        Write-Host "Enter the 6-digit pairing code shown on the watch:"
        & $Adb pair $Address
    }
    "install" {
        & $Adb connect $Address
        & $Adb install -r $Apk
        & $Adb shell am start -n dev.watchllm/.MainActivity --ez bench true
        Write-Host ""
        Write-Host "Bench started (GoLLeM PL, default model). Live stats:"
        & $Adb logcat -s WatchLLM:*
    }
}
