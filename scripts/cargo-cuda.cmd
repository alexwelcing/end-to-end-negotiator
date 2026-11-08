@echo off
rem Runs cargo inside an MSVC x64 dev environment with CUDA 12.2, which nvcc
rem needs to compile candle's CUDA kernels. MSVC 14.37 (VS 17.7) is a
rem supported host compiler for CUDA 12.2; newer toolsets are rejected by nvcc.
rem Usage: scripts\cargo-cuda.cmd build --release
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
set "PATH=%CUDA_PATH%\bin;%PATH%"
if not defined CUDA_COMPUTE_CAP set CUDA_COMPUTE_CAP=86
cargo %* --features cuda
