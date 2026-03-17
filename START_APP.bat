@echo off
title EduSense - Student Feedback Analyzer
color 0A
setlocal

echo.
echo  =======================================
echo    EduSense - Sentiment Analyzer
echo  =======================================
echo.
echo  Starting server... please wait
echo.

:: Go to the folder where this file is saved
cd /d "%~dp0"

:: Prefer the project virtual environment if available
set "PYTHON_EXE=python"
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" if not exist "venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

:: Start Flask in background
start /B "" "%PYTHON_EXE%" app.py > server.log 2>&1

:: Wait 4 seconds for server to start
echo  Loading model...
timeout /t 4 /nobreak > nul

:: Open browser automatically
echo  Opening browser...
start http://127.0.0.1:5000

echo.
echo  =======================================
echo   Server is running!
echo   Website: http://127.0.0.1:5000
echo   Dashboard: http://127.0.0.1:5000/dashboard
echo  =======================================
echo.
echo  Keep this window open while using the site.
echo  Press Ctrl+C or close this window to stop.
echo.

pause > nul
