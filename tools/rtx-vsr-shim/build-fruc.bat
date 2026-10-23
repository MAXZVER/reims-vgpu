@echo off
rem Builds frucbench.exe. Needs the Optical Flow SDK (not in this repository) at %OPTICAL_FLOW_SDK%.
setlocal
if "%OPTICAL_FLOW_SDK%"=="" set OPTICAL_FLOW_SDK=E:\sdk\optical-flow-sdk-5.0.7\Optical_Flow_SDK_5.0.7
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /MT /EHsc /std:c++17 /I"%OPTICAL_FLOW_SDK%\NvOFFRUC\Interface" frucbench.cpp /link /OUT:frucbench.exe || exit /b 1
del /q frucbench.obj 2>nul
copy /y "%OPTICAL_FLOW_SDK%\NvOFFRUC\NvOFFRUCSample\bin\win64\NvOFFRUC.dll" . >nul
copy /y "%OPTICAL_FLOW_SDK%\NvOFFRUC\NvOFFRUCSample\bin\win64\cudart64_110.dll" . >nul
echo built %~dp0frucbench.exe
