@echo off
rem Build patched llama.cpp for the HOST (verification: quantize + parity) and
rem the greedy_probe parity tool. For the Android build Gradle/CMake compiles
rem vendor/llama.cpp itself; this script is verification-only.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0\.."

cmake -S vendor\llama.cpp -B vendor\llama.cpp\build-host -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON
if errorlevel 1 exit /b 1

cmake --build vendor\llama.cpp\build-host --target llama-quantize llama-perplexity llama-tokenize
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /EHsc /O2 /I vendor\llama.cpp\include /I vendor\llama.cpp\ggml\include ^
   tools\greedy_probe.cpp /Fe:vendor\llama.cpp\build-host\bin\greedy_probe.exe ^
   /link /LIBPATH:vendor\llama.cpp\build-host\src /LIBPATH:vendor\llama.cpp\build-host\bin llama.lib
if errorlevel 1 exit /b 1

echo.
echo Host tools ready in vendor\llama.cpp\build-host\bin\
