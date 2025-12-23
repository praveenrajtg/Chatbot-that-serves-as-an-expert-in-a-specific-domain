@echo off
echo Starting Domain Expert Chatbot...
echo.

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found. Running setup first...
    call setup.bat
    if errorlevel 1 (
        echo Setup failed. Please check the error messages above.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 🤖 Domain Expert Chatbot Starting...
echo ========================================
echo.
echo The application will open in your default browser.
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo.

REM Run the Streamlit app
streamlit run app.py

echo.
echo Application stopped.
pause