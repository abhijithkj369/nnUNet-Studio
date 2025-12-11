@echo off
echo ================================================
echo   Installing PyTorch with CUDA Support
echo ================================================
echo.
echo This will install PyTorch with CUDA 11.8 support for GPU training.
echo.
pause

call venv\Scripts\activate

echo.
echo Uninstalling old PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo ================================================
echo   Installation Complete!
echo ================================================
echo.
echo Verifying CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
pause
