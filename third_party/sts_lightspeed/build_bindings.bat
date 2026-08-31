@echo off
rem Build sts_lightspeed python bindings with MSVC + Ninja
rem Manually set up MSVC env (VsDevCmd needs reg.exe which is blocked by sandbox)
set MSVC=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.42.34433
set SDK=C:\Program Files (x86)\Windows Kits\10
set SDKV=10.0.22621.0
set PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKV%\x64;%PATH%
set INCLUDE=%MSVC%\include;%SDK%\Include\%SDKV%\ucrt;%SDK%\Include\%SDKV%\um;%SDK%\Include\%SDKV%\shared;%SDK%\Include\%SDKV%\winrt
set LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKV%\ucrt\x64;%SDK%\Lib\%SDKV%\um\x64
set CMAKE="C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set NINJA=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
%CMAKE% -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="C:\Youqing-Vault\Embodied-AI\07-projects\sts-agent\.venv\Scripts\python.exe" -DPYTHON_EXECUTABLE="C:\Youqing-Vault\Embodied-AI\07-projects\sts-agent\.venv\Scripts\python.exe"
if errorlevel 1 exit /b 1
%CMAKE% --build build --target slaythespire
