#!/usr/bin/env bash
# One-shot Android toolchain for this project: JDK 17, Android SDK
# (platform 35, build-tools, NDK 26.3, CMake 3.22.1) and Gradle - all inside
# .toolchain/ so nothing touches the system. Then generates the Gradle wrapper
# and writes local.properties.
#
# Skips anything already downloaded. Needs ~4 GB of disk and internet.
set -euo pipefail
cd "$(dirname "$0")/.."

TC=.toolchain
SDK=$TC/android-sdk
mkdir -p "$TC" "$SDK/licenses"

# pre-accepted SDK licenses (standard Google license hashes) so sdkmanager
# never goes interactive
printf '\n24333f8a63b6825ea9c5514f83c2829b004d1fee\n8933bad161af4178b1185d1a37fbf41ea5269c55\n' \
  > "$SDK/licenses/android-sdk-license"
printf '\n84831b9409646a918e30573bab4c9c91346d8abd\n' \
  > "$SDK/licenses/android-sdk-preview-license"

fetch() { # url out
  [ -f "$2" ] || curl -fL --progress-bar -o "$2" "$1"
}

fetch "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse" "$TC/jdk17.zip"
fetch "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip"      "$TC/cmdline-tools.zip"
fetch "https://services.gradle.org/distributions/gradle-8.10.2-bin.zip"                          "$TC/gradle.zip"

extract() { # zip dest-marker-dir
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"
}

[ -d "$TC/jdk" ] || {
  extract "$TC/jdk17.zip" "$TC"
  mv "$TC"/jdk-17* "$TC/jdk"
}
[ -d "$SDK/cmdline-tools/latest" ] || {
  extract "$TC/cmdline-tools.zip" "$TC"
  mkdir -p "$SDK/cmdline-tools"
  mv "$TC/cmdline-tools" "$SDK/cmdline-tools/latest"
}
[ -d "$TC/gradle" ] || {
  extract "$TC/gradle.zip" "$TC"
  mv "$TC"/gradle-8* "$TC/gradle"
}

export JAVA_HOME="$PWD/$TC/jdk"
"$SDK/cmdline-tools/latest/bin/sdkmanager.bat" --sdk_root="$PWD/$SDK" \
  "platform-tools" "platforms;android-35" "build-tools;35.0.0" \
  "ndk;26.3.11579264" "cmake;3.22.1"

# Gradle wants a Windows path in local.properties
SDK_WIN=$(cd "$SDK" && pwd -W 2>/dev/null || pwd)
printf 'sdk.dir=%s\n' "$SDK_WIN" > local.properties

"$TC/gradle/bin/gradle" wrapper --gradle-version 8.10.2
echo
echo "Toolchain ready: $TC (JAVA_HOME=$TC/jdk, SDK=$SDK)"
