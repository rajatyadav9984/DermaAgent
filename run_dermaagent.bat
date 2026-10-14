@echo off
title DermaAgent - AI Skin Screening System
echo ============================================================
echo Starting DermaAgent Server & Opening Web App...
echo ============================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found!
    echo Please make sure you run this script from C:\DermaAgent
    pause
    exit /b 1
)

:: Start Flask Backend Server (Opens browser automatically when server is ready)
".venv\Scripts\python.exe" backend\app.py

pause




