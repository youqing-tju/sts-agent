@echo off
setlocal enabledelayedexpansion
rem ============================================================================
rem  Build sts_lightspeed Python bindings -> slaythespire.cp313-win_amd64.pyd
rem  Full guide: docs\build-simulator.md
rem
rem  Usage (run from anywhere):
rem      third_party\sts_lightspeed\build_bindings.bat
rem
rem  NOTE: the previous version hard-coded the VS / Windows SDK / Python paths of
rem  one specific machine, so it broke on any other machine. This version
rem  auto-detects common install locations; if detection fails, fill in the
rem  USER CONFIG section below.
rem ============================================================================

rem ------------------------------ USER CONFIG --------------------------------
rem Leave blank to auto-detect.
rem Visual Studio install dir, e.g. C:\Program Files\Microsoft Visual Studio\2022\Professional
set "CFG_VSPATH="
rem Windows SDK version, e.g. 10.0.22621.0
set "CFG_SDKV="
rem Python used for building (blank = <repo root>\.venv\Scripts\python.exe)
set "CFG_PYEXE="
rem ---------------------------------------------------------------------------

rem --- Locate repo root (this script lives in <root>\third_party\sts_lightspeed\) ---
set "ROOT=%~dp0..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

rem --- Python ---
if not defined CFG_PYEXE set "CFG_PYEXE=%ROOT%\.venv\Scripts\python.exe"
if not exist "%CFG_PYEXE%" (
  echo [ERROR] Python not found: %CFG_PYEXE%
  echo         Create it first from the repo root:
  echo             python -m venv .venv
  echo             .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)

rem --- Visual Studio: try vswhere first, then fall back to common paths ---
if not defined CFG_VSPATH (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if exist "!VSWHERE!" (
    for /f "usebackq delims=" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "CFG_VSPATH=%%i"
  )
)
if not defined CFG_VSPATH (
  for %%E in (Enterprise Professional Community BuildTools) do (
    if not defined CFG_VSPATH if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Tools\MSVC" set "CFG_VSPATH=%ProgramFiles%\Microsoft Visual Studio\2022\%%E"
    if not defined CFG_VSPATH if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\%%E\VC\Tools\MSVC" set "CFG_VSPATH=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\%%E"
  )
)
if not defined CFG_VSPATH (
  echo [ERROR] Visual Studio 2022 with the C++ toolchain was not found.
  echo         Install VS 2022 and check the "Desktop development with C++" workload,
  echo         or set CFG_VSPATH manually in the USER CONFIG section.
  echo         See docs\build-simulator.md
  exit /b 1
)

rem --- MSVC toolset: pick the highest version ---
set "CFG_MSVC="
for /f "delims=" %%i in ('dir /b /ad /o-n "%CFG_VSPATH%\VC\Tools\MSVC" 2^>nul') do (
  if not defined CFG_MSVC set "CFG_MSVC=%CFG_VSPATH%\VC\Tools\MSVC\%%i"
)
if not defined CFG_MSVC (
  echo [ERROR] No toolset under: %CFG_VSPATH%\VC\Tools\MSVC
  echo         The "Desktop development with C++" workload is probably missing.
  exit /b 1
)

rem --- Windows SDK: pick the highest version ---
set "CFG_SDK=%ProgramFiles(x86)%\Windows Kits\10"
if not defined CFG_SDKV (
  for /f "delims=" %%i in ('dir /b /ad /o-n "%CFG_SDK%\Include" 2^>nul') do (
    if not defined CFG_SDKV set "CFG_SDKV=%%i"
  )
)
if not defined CFG_SDKV (
  echo [ERROR] Windows SDK not found under %CFG_SDK%\Include
  exit /b 1
)

rem --- CMake / Ninja shipped with Visual Studio ---
set "CFG_CMAKE=%CFG_VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "CFG_NINJA=%CFG_VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not exist "%CFG_CMAKE%" (
  echo [ERROR] cmake.exe not found: %CFG_CMAKE%
  echo         Install the "C++ CMake tools for Windows" component in VS Installer.
  exit /b 1
)
if not exist "%CFG_NINJA%" (
  echo [ERROR] ninja.exe not found: %CFG_NINJA%
  exit /b 1
)

echo ============================================================
echo  Visual Studio : %CFG_VSPATH%
echo  MSVC toolset  : %CFG_MSVC%
echo  Windows SDK   : %CFG_SDKV%
echo  Python        : %CFG_PYEXE%
echo ============================================================

rem --- Assemble a clean build env. VsDevCmd is NOT used on purpose:
rem     it depends on reg.exe, which is unavailable in some sandboxes.
set "PATH=%CFG_MSVC%\bin\Hostx64\x64;%CFG_SDK%\bin\%CFG_SDKV%\x64;%PATH%"
set "INCLUDE=%CFG_MSVC%\include;%CFG_SDK%\Include\%CFG_SDKV%\ucrt;%CFG_SDK%\Include\%CFG_SDKV%\um;%CFG_SDK%\Include\%CFG_SDKV%\shared;%CFG_SDK%\Include\%CFG_SDKV%\winrt"
set "LIB=%CFG_MSVC%\lib\x64;%CFG_SDK%\Lib\%CFG_SDKV%\ucrt\x64;%CFG_SDK%\Lib\%CFG_SDKV%\um\x64"

pushd "%~dp0"
"%CFG_CMAKE%" -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM="%CFG_NINJA%" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="%CFG_PYEXE%" -DPYTHON_EXECUTABLE="%CFG_PYEXE%"
if errorlevel 1 (
  popd
  echo [FAILED] CMake configure step failed.
  exit /b 1
)
"%CFG_CMAKE%" --build build --target slaythespire
if errorlevel 1 (
  popd
  echo [FAILED] Build failed.
  exit /b 1
)

rem --- Copy the artifact to the repo root (Python imports from there) ---
for /f "delims=" %%f in ('dir /b /s "build\slaythespire*.pyd" 2^>nul') do (
  echo [COPY] %%~nxf  -^>  %ROOT%\
  copy /y "%%f" "%ROOT%\" >nul
)
popd

echo.
echo [DONE] Build finished. Verify with:
echo        cd /d "%ROOT%"
echo        .venv\Scripts\python.exe verify_env.py 12345
endlocal
