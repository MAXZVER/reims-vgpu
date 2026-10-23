@echo off
rem Builds rslg.dll (C ABI over NVIDIA Streamline: DLSS Frame Generation + Reflex + PCL, Vulkan) with MSVC.
rem Needs the Streamline SDK (not in this repository; github.com/NVIDIA-RTX/Streamline releases) at %STREAMLINE_SDK%,
rem e.g. E:\sdk\streamline-2.14.16, and Vulkan headers at %VULKAN_INCLUDE% (default E:\sdk\vulkan-headers\include).
rem Output: rslg.dll next to this file; ship it with the SDK's bin\x64 DLLs (sl.*.dll, nvngx_dlssg.dll, NvLowLatencyVk.dll).
setlocal
if "%STREAMLINE_SDK%"=="" set STREAMLINE_SDK=E:\sdk\streamline-2.14.16
if "%VULKAN_INCLUDE%"=="" set VULKAN_INCLUDE=E:\sdk\vulkan-headers\include
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /MT /EHsc /std:c++17 /LD /I"%STREAMLINE_SDK%\include" /I"%VULKAN_INCLUDE%" rslg.cpp ^
   /link /LIBPATH:"%STREAMLINE_SDK%\lib\x64" sl.interposer.lib /OUT:rslg.dll || exit /b 1
del /q rslg.obj rslg.exp rslg.lib 2>nul
echo built %~dp0rslg.dll
