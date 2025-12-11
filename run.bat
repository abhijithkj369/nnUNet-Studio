@echo off
REM Quick launch script for nnUNetv2 Training UI

echo ================================================
echo   nnUNetv2 Training Interface Launcher
echo ================================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting Gradio application...
echo Access the UI at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

python app.py

pause
