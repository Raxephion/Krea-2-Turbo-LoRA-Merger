@echo off
setlocal

echo ============================================
echo  Krea 2 Turbo LoRA Merger - Installer
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo and make sure "Add python.exe to PATH" is checked during install.
    pause
    exit /b 1
)

echo Creating virtual environment in .venv ...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo Upgrading pip ...
python -m pip install --upgrade pip

echo.
echo Installing requirements (this can take a few minutes) ...
pip install -r requirements.txt

echo.
echo ------------------------------------------------------------
echo NOTE: The above installs a CPU version of torch by default.
echo If you have an NVIDIA GPU and want CUDA-accelerated merging,
echo run this AFTER install.bat finishes (adjust cu121 as needed
echo for your CUDA version, see https://pytorch.org/get-started/locally/):
echo.
echo     .venv\Scripts\activate
echo     pip install torch --index-url https://download.pytorch.org/whl/cu121
echo ------------------------------------------------------------
echo.

if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Installation complete!
echo  Run the app using run.bat
echo ============================================
pause
