@echo off
echo ================================================
echo   Configuring nnUNet for CPU-Only Training
echo ================================================
echo.
echo This will force nnUNet to use CPU instead of GPU.
echo WARNING: Training will be MUCH slower on CPU!
echo.
pause

echo.
echo Setting environment variable for CPU-only mode...
setx nnUNet_USE_CUDA "False"

echo.
echo ================================================
echo   Configuration Complete!
echo ================================================
echo.
echo nnUNet will now use CPU for training.
echo.
echo IMPORTANT: You need to RESTART your terminal/command prompt
echo for the environment variable to take effect!
echo.
pause
