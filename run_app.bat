@echo off
setlocal

set "PYTHON_EXE=D:\Download\Anaconda_envs\envs\CV\python.exe"
set "PROJECT_DIR=%~dp0"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python interpreter not found:
    echo %PYTHON_EXE%
    echo Please edit run_app.bat and update PYTHON_EXE.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
echo Starting Streamlit app...
echo Project: %PROJECT_DIR%
echo Python : %PYTHON_EXE%
echo.
"%PYTHON_EXE%" -m streamlit run app.py

endlocal
