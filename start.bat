@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title DpDa - Disease Prediction System

echo.
echo ========================================
echo     DpDa Disease Prediction System
echo ========================================
echo.

REM ----- Step 1: Check Python -----
where python >nul 2>&1
if errorlevel 1 goto no_python
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [1/4] Python %PYVER% detected.
goto step2

:no_python
echo [ERROR] Python not found. Please install Python 3.8+ from python.org
echo         Make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:step2
REM ----- Step 2: Check dependencies -----
echo [2/4] Checking dependencies...
python -c "import flask, pandas, numpy, sklearn, joblib, matplotlib" >nul 2>&1
if errorlevel 1 goto install_deps
echo        All dependencies ready.
goto step3

:install_deps
echo        Installing missing dependencies, please wait...
python -m pip install -r requirements.txt >nul 2>&1
if errorlevel 1 goto install_failed
echo        Dependencies installed.
goto step3

:install_failed
echo [ERROR] Failed to install dependencies.
echo         Run manually:  pip install -r requirements.txt
pause
exit /b 1

:step3
REM ----- Step 3: Check inference models -----
echo [3/4] Checking inference models...
if exist "output\inference_models\stroke_model.pkl" goto models_ok
echo        First launch detected. Training inference models...
echo        This may take 1-2 minutes.
python train_inference_models.py
if errorlevel 1 goto models_warn
echo        Inference models ready.
goto step4

:models_warn
echo [WARNING] Failed to train inference models.
echo           The app will use simulated data for predictions.
goto step4

:models_ok
echo        Inference models ready.

:step4
REM ----- Step 4: Check checkpoints -----
echo [4/4] Checking analysis checkpoints...
if exist "output\checkpoints\awelm_heart.json" goto ckpt_ok
echo        Generating checkpoints (one-time)...
python generate_checkpoints.py >nul 2>&1
if errorlevel 1 goto ckpt_warn
echo        Checkpoints ready.
goto start_server

:ckpt_warn
echo [WARNING] Checkpoint generation had errors. Continuing anyway.
goto start_server

:ckpt_ok
echo        Checkpoints ready.

:start_server
echo.
echo ========================================
echo     Server starting at http://localhost:5000
echo     Press Ctrl+C to stop the server
echo ========================================
echo.

REM Browser is auto-opened by app.py after server is ready (avoids opening duplicate tabs/windows)

REM Start the Flask app
python app.py
set APPERR=%errorlevel%

echo.
if %APPERR% neq 0 (
    echo [ERROR] Server stopped unexpectedly (exit code %APPERR%).
    echo         Check app.log for details.
) else (
    echo Server stopped normally.
)
echo.
echo Press any key to close this window...
pause >nul
