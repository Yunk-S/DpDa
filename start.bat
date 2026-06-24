@echo off
cd /d "%~dp0"
title DpDa - Disease Prediction System

echo ========================================
echo     DpDa Disease Prediction System
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit
)

echo Checking dependencies...
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Train inference models on first launch (if missing)
if not exist "output\inference_models\stroke_model.pkl" (
    echo Training inference models (one-time)...
    python train_inference_models.py
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to train inference models
        pause
        exit
    )
)

echo.
echo ========================================
echo     Starting server...
echo     http://localhost:5000
echo ========================================
echo.

python app.py
