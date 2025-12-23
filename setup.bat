@echo off
echo ========================================
echo Domain Expert Chatbot Setup Script
echo ========================================
echo.

echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo ✓ Python found

echo.
echo [2/7] Updating pip...
python -m pip install --upgrade pip
echo ✓ Pip updated

echo.
echo [3/7] Removing old virtual environment...
if exist venv (
    rmdir /s /q venv
    echo ✓ Old environment removed
)

echo.
echo [4/7] Creating virtual environment...
python -m venv venv
echo ✓ Virtual environment created

echo.
echo [5/7] Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated

echo.
echo [6/7] Installing dependencies...
echo This may take 5-10 minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed

echo.
echo [7/7] Testing installation...
python -c "import streamlit; print('✓ Setup complete!')"

echo.
echo ========================================
echo Setup completed successfully! 🎉
echo ========================================
echo.
echo To run: run_app.bat
echo.
pause