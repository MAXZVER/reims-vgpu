@echo off
rem Builds vsrbench.exe (GPU time of NGX VSR per size and quality) next to rvsr.dll. No Vulkan SDK needed: an import
rem library for the system's vulkan-1.dll is generated from its export table.
setlocal
if "%VULKAN_INCLUDE%"=="" set VULKAN_INCLUDE=E:\sdk\vulkan-headers\include
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
lib /nologo /def:vulkan-1-bench.def /out:vulkan-1.lib /machine:x64 >nul || exit /b 1
cl /nologo /O2 /MT /EHsc /std:c++17 /I"%VULKAN_INCLUDE%" vsrbench.cpp /link vulkan-1.lib /OUT:vsrbench.exe || exit /b 1
del /q vsrbench.obj vulkan-1.exp 2>nul
echo built %~dp0vsrbench.exe
