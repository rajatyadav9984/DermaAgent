@echo off
title DermaAgent - AI Skin Screening System
echo ============================================================
echo Starting DermaAgent Server & Opening Web App...
echo ============================================================

cd /d C:\DermaAgent

:: Open browser automatically after 2 seconds delay
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5000/

:: Start Flask Backend Server
.\.venv\Scripts\python.exe backend\app.py

pause
