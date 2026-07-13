@echo off
setlocal enableextensions

REM Project runner for Classroom Behavior Monitoring
REM Usage:
REM   run_project.bat [app|analyze|train] [args]
REM Defaults to 'app' (Streamlit UI)

set WORKDIR=%~dp0
set VENV_DIR=%WORKDIR%classroom_monitor_env
set VENV_PY=%VENV_DIR%\Scripts\python.exe
set VENV_ACT=%VENV_DIR%\Scripts\activate.bat

if "%1"=="" set MODE=app
if NOT "%1"=="" set MODE=%1%
shift

if not exist "%VENV_PY%" (
  echo [ERROR] Python venv not found at %VENV_PY%
  echo Create it or update the path in this script.
  exit /b 1
)

call "%VENV_ACT%"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)

if /I "%MODE%"=="app" (
  echo [INFO] Launching Streamlit app...
  streamlit run "%WORKDIR%streamlit_app_enhanced.py" %*
  goto :eof
)

if /I "%MODE%"=="app-basic" (
  echo [INFO] Launching Basic Streamlit app...
  streamlit run "%WORKDIR%streamlit_app.py" %*
  goto :eof
)

if /I "%MODE%"=="analyze" (
  echo [INFO] Running video analyzer...
  "%VENV_PY%" "%WORKDIR%video_analyzer.py" %*
  goto :eof
)

if /I "%MODE%"=="train" (
  echo [INFO] Training model...
  "%VENV_PY%" "%WORKDIR%model_trainer.py" %*
  goto :eof
)

echo [ERROR] Unknown mode '%MODE%'. Use one of: app | analyze | train
exit /b 1
