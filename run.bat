@echo off
setlocal enabledelayedexpansion

REM ======================================================
REM  🦥 Sl0thfier Launcher
REM  - Ensures correct working directory
REM  - Activates uv virtual environment
REM  - Runs main GUI (preprocess.py)
REM ======================================================

REM Move to project root (same folder as this script)
cd /d "%~dp0"

REM Logging setup
if not exist logs mkdir logs
set LOG_FILE=logs\run_%DATE:~0,4%-%DATE:~5,2%-%DATE:~8,2%_%TIME:~0,2%-%TIME:~3,2%.log
set LOG_FILE=%LOG_FILE: =0%

echo ===================================================== >> "%LOG_FILE%"
echo   Starting Sl0thfier... [%DATE% %TIME%] >> "%LOG_FILE%"
echo ===================================================== >> "%LOG_FILE%"

REM Display Python & environment info
echo [INFO] Checking Python environment...
uv python --version >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [ERROR] uv not found or environment broken.
    pause
    exit /b 1
)

REM Run the app
echo [INFO] Launching Sl0thfier GUI...
uv run python -m sl0thfier.preprocess >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [ERROR] Application crashed. Check logs\ folder for details.
    echo.
    type "%LOG_FILE%"
    pause
    exit /b 1
)

echo [INFO] Sl0thfier finished successfully.
pause
exit /b 0
