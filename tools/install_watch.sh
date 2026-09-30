#!/usr/bin/env bash
# Install WatchLLM on a Galaxy Watch (Wear OS 3+) over Wi-Fi adb.
# See tools/install_watch.ps1 for the on-watch setup steps.
#
# Usage:
#   ./tools/install_watch.sh pair    192.168.0.23:37123   # type the 6-digit code shown on the watch
#   ./tools/install_watch.sh install 192.168.0.23:37123   # connect + install APK + start bench
set -euo pipefail
cd "$(dirname "$0")/.."

ADB=.toolchain/android-sdk/platform-tools/adb.exe
command -v "$ADB" >/dev/null 2>&1 || ADB=adb
APK=app/build/outputs/apk/debug/app-debug.apk

case "${1:-}" in
  pair)
    echo "Enter the 6-digit pairing code shown on the watch:"
    "$ADB" pair "$2"
    ;;
  install)
    "$ADB" connect "$2"
    "$ADB" install -r "$APK"
    "$ADB" shell am start -n dev.watchllm/.MainActivity --ez bench true
    echo
    echo "Bench started (GoLLeM PL, default model). Live stats:"
    "$ADB" logcat -s WatchLLM:*
    ;;
  *)
    echo "usage: $0 pair|install <IP:PORT>"
    exit 2
    ;;
esac
