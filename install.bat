@echo off
REM Point Cloud Detection Scanner - Easy Setup Script
REM This script automates the installation process for new users

echo ========================================
echo Point Cloud Detection Scanner - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/5] Python detected:
python --version
echo.

REM Check Python version (must be 3.8+)
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.

echo [2/5] Creating virtual environment (recommended)...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [WARNING] Could not create virtual environment
        echo Continuing with system Python...
    ) else (
        echo Virtual environment created successfully
    )
)
echo.

echo [3/5] Installing BASIC dependencies (OpenCV + NumPy)...
echo This allows laser scanning without AI features
echo.
python -m pip install --upgrade pip
python -m pip install opencv-python numpy psutil
if errorlevel 1 (
    echo [ERROR] Failed to install basic dependencies
    pause
    exit /b 1
)
echo Basic installation complete!
echo.

echo ========================================
echo Installation Options
echo ========================================
echo.
echo [A] Quick Start - Basic laser scanning only (already installed)
echo [B] Full Install - Add AI depth estimation (PyTorch, 2GB+ download)
echo [C] Skip for now - I'll install extras later
echo.
choice /C ABC /N /M "Select option (A/B/C): "

if errorlevel 3 goto :skip_extras
if errorlevel 2 goto :full_install
if errorlevel 1 goto :quick_start

:full_install
echo.
echo [4/5] Installing FULL dependencies (AI depth + 3D viewer)...
echo This may take 5-10 minutes depending on your internet speed...
echo.
python -m pip install torch torchvision timm open3d
if errorlevel 1 (
    echo [WARNING] Some packages failed to install
    echo You can still use basic laser scanning
    echo To retry AI features later, run: pip install torch torchvision timm open3d
)
goto :post_install

:quick_start
echo.
echo [4/5] Quick start mode - basic dependencies only
echo.
goto :post_install

:skip_extras
echo.
echo [4/5] Skipping optional dependencies
echo.
goto :post_install

:post_install
echo.
echo [5/5] Checking for camera calibration...
if exist "dual_checkerboard_3d\calibration\camera_calibration_default.npz" (
    echo Default calibration found
) else (
    echo No calibration found - will use generic defaults
    echo For best accuracy, run: python dual_checkerboard_3d\checkerboard.py
)
echo.

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To start scanning:
echo   1. cd scanning
echo   2. python laser_3d_scanner_advanced.py
echo.
echo First-time tips:
echo   - Scanner will auto-detect missing calibration
echo   - Press 'B' in scanner to see keyboard controls
echo   - Start with Mode 1 (Red Laser) for best results
echo.
echo Optional: Run calibration for better accuracy
echo   python dual_checkerboard_3d\checkerboard.py
echo.
pause
