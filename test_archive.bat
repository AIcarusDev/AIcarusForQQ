@echo off
chcp 65001 > nul
cd /d "%~dp0"

:: 优先用 .launcher_env 里记录的 Python（与 start.bat 共用）
set "PYTHON_CMD="
if exist ".launcher_env" (
    set /p PYTHON_CMD=<".launcher_env"
)

:: 没有记录时回退到 .venv
if not defined PYTHON_CMD (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_CMD=.venv\Scripts\python"
    ) else (
        set "PYTHON_CMD=python"
    )
)

echo [INFO] Python: %PYTHON_CMD%
echo [INFO] 启动记忆归档提取测试...
echo.

%PYTHON_CMD% scripts\test_archive_prompt.py

echo.
pause
