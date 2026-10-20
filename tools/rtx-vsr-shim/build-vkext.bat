@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
lib /nologo /def:vulkan-1-bench.def /out:vulkan-1.lib /machine:x64 >nul || exit /b 1
cl /nologo /O2 /MT /EHsc /I"E:\sdk\vulkan-headers\include" vkext.cpp /link vulkan-1.lib /OUT:vkext.exe || exit /b 1
del /q vkext.obj vulkan-1.exp 2>nul
