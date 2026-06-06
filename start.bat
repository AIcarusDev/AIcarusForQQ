@echo off
setlocal EnableDelayedExpansion
chcp 65001 > nul
cd /d "%~dp0"

set "ENV_CONFIG_FILE=.launcher_env"
set "REQUIREMENTS_FILE=requirements.txt"
set "MENU_SCRIPT=scripts\launch_menu.py"

rem Check for reset argument
if "%1"=="--reset" (
    if exist "%ENV_CONFIG_FILE%" del "%ENV_CONFIG_FILE%"
    echo [INFO] Environment configuration reset.
    shift
)

rem Configuration templates check
set "TPL_ENV=templates\.env.template"
set "TPL_CFG=templates\config.yaml.template"
set "DEST_ENV=.env"
set "DEST_CFG=config\config_user.yaml"
set "NEED_CONFIG="

if not exist "%DEST_ENV%" (
    if exist "%TPL_ENV%" (
        echo [INIT] Creating .env from template...
        copy "%TPL_ENV%" "%DEST_ENV%" >nul
        echo [INFO] Created .env
        set "NEED_CONFIG=1"
    )
)

if not exist "%DEST_CFG%" (
    if exist "%TPL_CFG%" (
        echo [INIT] Creating config\config_user.yaml from template...
        copy "%TPL_CFG%" "%DEST_CFG%" >nul
        echo [INFO] Created config\config_user.yaml
        set "NEED_CONFIG=1"
    )
)

if not defined NEED_CONFIG goto config_done
echo.
echo ==========================================================
echo [IMPORTANT] Configuration files have been created.
echo Please edit them before continuing:
echo   1. Edit .env (Add your API KEY)
echo   2. Edit config\config_user.yaml (Optional settings)
echo ==========================================================
echo.
pause
:config_done
rem End configuration templates check

rem Check if configuration exists
if exist "%ENV_CONFIG_FILE%" (
    set /p PYTHON_CMD=<"%ENV_CONFIG_FILE%"
    echo [INFO] Using saved environment: !PYTHON_CMD!
    goto launch
)

:setup_env
cls
echo ==================================================
echo          AIcarusForQQ Environment Setup
echo ==================================================
echo.
echo Please select Python environment:
echo.
echo [1] Create new virtual environment (Recommended)
echo [2] Use existing virtual environment
echo [3] Use Conda environment
echo [4] Use system Python
echo.
set /p CHOICE="Enter your choice (1-4): "

if "%CHOICE%"=="1" goto create_venv
if "%CHOICE%"=="2" goto use_existing_venv
if "%CHOICE%"=="3" goto use_conda
if "%CHOICE%"=="4" goto use_system
goto setup_env

:create_venv
echo.
echo [INFO] Creating virtual environment in 'venv'...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create venv. Make sure 'python' is in your PATH and is version 3.x.
    echo If you have 'python3' command but not 'python', try option 4 with 'python3'.
    pause
    exit /b 1
)

echo [INFO] Upgrading pip...
venv\Scripts\python -m pip install --upgrade pip

echo [INFO] Installing requirements from %REQUIREMENTS_FILE%...
venv\Scripts\python -m pip install -r %REQUIREMENTS_FILE%
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

set "PYTHON_CMD=venv\Scripts\python"
echo !PYTHON_CMD!> "%ENV_CONFIG_FILE%"
echo [INFO] Environment setup complete.
goto launch

:use_existing_venv
echo.
echo Enter path to venv folder (relative or absolute).
echo Example: .venv
set /p VENV_PATH="Path: "
if exist "%VENV_PATH%\Scripts\python.exe" (
    set "PYTHON_CMD=%VENV_PATH%\Scripts\python"
) else (
    echo [ERROR] Python executable not found in %VENV_PATH%\Scripts\
    pause
    goto setup_env
)
echo !PYTHON_CMD!> "%ENV_CONFIG_FILE%"
goto launch

:use_conda
echo.
set /p CONDA_ENV="Enter Conda environment name: "
rem Use 'conda run' to execute in the environment without full activation script complexity
set "PYTHON_CMD=conda run -n %CONDA_ENV% --no-capture-output python"
echo !PYTHON_CMD!> "%ENV_CONFIG_FILE%"
goto launch

:use_system
echo.
set /p SYSTEM_PYTHON="Enter python command (default: python): "
if "%SYSTEM_PYTHON%"=="" set "SYSTEM_PYTHON=python"
set "PYTHON_CMD=%SYSTEM_PYTHON%"
echo !PYTHON_CMD!> "%ENV_CONFIG_FILE%"
goto launch

:launch
echo.
echo [INFO] Opening AIcarusForQQ startup menu...
echo [CMD] !PYTHON_CMD! %MENU_SCRIPT% %*
echo.

!PYTHON_CMD! %MENU_SCRIPT% %*

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error.
    pause
)
