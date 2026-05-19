@echo off
setlocal enabledelayedexpansion

:: Local AI Generator - Optimized Setup & Run Script
:: =====================================================

echo.
echo ========================================================
echo    Local AI Generator for Aseprite - Auto Setup
echo ========================================================
echo.

:: 🔒 ป้องกันปัญหา Path มีช่องว่าง
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Current directory: %SCRIPT_DIR%

:: [1/4] Check required files
echo [1/4] Checking required files...
if not exist "startup_script.py" (
    echo [ERROR] startup_script.py not found.
    pause
    exit /b 1
)

if not exist "sd_server.py" (
    if exist "paste.txt" (
        echo Found paste.txt - renaming to sd_server.py...
        move /y "paste.txt" "sd_server.py" >nul 2>&1
        echo [OK] Created sd_server.py
    ) else (
        echo [ERROR] sd_server.py not found.
        pause
        exit /b 1
    )
) else (
    echo [OK] sd_server.py found
)

:: [2/4] Check Python installation & version (AI libs require 3.10+)
echo [2/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.10+ from https://python.org
    echo IMPORTANT: Check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: ตรวจสอบเวอร์ชัน Python อย่างปลอดภัยผ่าน Python เอง
python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Python 3.10+ is required for PyTorch/Diffusers.
    echo Found: 
    python --version
    echo Installation may fail. Please upgrade Python.
    pause
    exit /b 1
)
echo [OK] Python version compatible

:: [3/4] Virtual Environment
echo [3/4] Setting up virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo [OK] Virtual environment exists
) else (
    echo Creating venv...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create venv. Try running as Administrator.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

:: [4/4] Directories
echo [4/4] Creating directories...
if not exist "loras" mkdir loras
if not exist "models" mkdir models
echo [OK] Directories ready

echo.
echo ========================================================
echo              Setup Complete! Starting Server...
echo ========================================================
echo.

call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate venv.
    pause
    exit /b 1
)

echo [INFO] Running startup_script.py...
echo [INFO] First run will install dependencies. This may take a few minutes.
echo.

:: 💡 ใช้ python เพื่อแสดง progress ตอนติดตั้ง dependencies
:: หลังติดตั้งเสร็จแล้ว หากต้องการรันแบบไม่มี Console ให้เปลี่ยนเป็น: pythonw startup_script.py
python startup_script.py

echo.
echo ========================================================
echo                   Server Stopped
echo ========================================================
pause
exit /b 0