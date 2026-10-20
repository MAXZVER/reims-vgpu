@echo off
rem Builds dlssgwin.exe (DLSS-G generation check in a focused window; see dlssgwin.cpp) with MSVC.
rem Needs Vulkan headers at %VULKAN_INCLUDE% (default E:\sdk\vulkan-headers\include); run it with rslg.dll and
rem the Streamline DLLs in one directory (its first argument).
setlocal
if "%VULKAN_INCLUDE%"=="" set VULKAN_INCLUDE=E:\sdk\vulkan-headers\include
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /MT /EHsc /std:c++17 /I"%VULKAN_INCLUDE%" dlssgwin.cpp /link user32.lib /OUT:dlssgwin.exe || exit /b 1
del /q dlssgwin.obj 2>nul
echo built %~dp0dlssgwin.exe
