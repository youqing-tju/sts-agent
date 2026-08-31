@echo off
set MSVC=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.42.34433
set PATH=%MSVC%\bin\Hostx64\x64;%PATH%
dumpbin /symbols "C:\Youqing-Vault\Embodied-AI\07-projects\sts-agent\third_party\sts_lightspeed\build\CMakeFiles\slaythespire.dir\src\combat\Actions.cpp.obj" > actions_sym.txt
findstr "MakeTempCardInDrawPile" actions_sym.txt
echo ===
findstr /c:"File Type" actions_sym.txt
