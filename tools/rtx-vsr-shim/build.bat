@echo off
rem Builds rvsr.dll (C ABI over NVIDIA RTX Video Super Resolution, Vulkan) with MSVC.
rem Needs the RTX Video SDK (not in this repository; NVIDIA developer download) at %RTX_VIDEO_SDK%,
rem e.g. E:\sdk\rtx-video-sdk-1.1.0, and Vulkan headers at %VULKAN_INCLUDE% (default: E:\sdk\vulkan-headers\include, a copy of only vulkan/ and vk_video/ - the whole MSYS2 include dir breaks MSVC).
rem Output: rvsr.dll next to this file; ship it with nvngx_vsr.dll from the SDK beside qemu-system-x86_64.exe.
setlocal
if "%RTX_VIDEO_SDK%"=="" set RTX_VIDEO_SDK=E:\sdk\rtx-video-sdk-1.1.0
if "%VULKAN_INCLUDE%"=="" set VULKAN_INCLUDE=E:\sdk\vulkan-headers\include
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /MT /EHsc /std:c++17 /LD /I"%RTX_VIDEO_SDK%\include" /I"%VULKAN_INCLUDE%" rvsr.cpp ^
   /link /LIBPATH:"%RTX_VIDEO_SDK%\lib\Windows\x64" nvsdk_ngx_s.lib advapi32.lib user32.lib /OUT:rvsr.dll || exit /b 1
del /q rvsr.obj rvsr.exp rvsr.lib 2>nul
echo built %~dp0rvsr.dll
