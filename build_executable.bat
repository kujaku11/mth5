@echo off
REM Build MTH5 Validator Standalone Executable (Windows)
REM Quick build script using PyInstaller

echo ===============================================================================
echo MTH5 Validator - Standalone Executable Builder (Windows)
echo ===============================================================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    python -m pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

echo Building executable...
echo.

pyinstaller ^
    --onefile ^
    --name mth5-validator ^
    --console ^
    --clean ^
    --hidden-import=h5py.defs ^
    --hidden-import=h5py.utils ^
    --hidden-import=h5py._proxy ^
    --hidden-import=h5py.h5ac ^
    --collect-all=mt_metadata ^
    --collect-all=mth5 ^
    --copy-metadata=mth5 ^
    --copy-metadata=mt_metadata ^
    --copy-metadata=h5py ^
    --exclude-module=matplotlib ^
    --exclude-module=tkinter ^
    --exclude-module=PyQt5 ^
    --exclude-module=PyQt6 ^
    --exclude-module=jupyter ^
    --exclude-module=notebook ^
    --log-level=WARN ^
    mth5\utils\cli.py

if errorlevel 1 (
    echo.
    echo ===============================================================================
    echo BUILD FAILED
    echo ===============================================================================
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo BUILD SUCCESSFUL
echo ===============================================================================
echo.
echo Executable location: dist\mth5-validator.exe
echo.
echo Test it:
echo   dist\mth5-validator.exe --help
echo   dist\mth5-validator.exe validate your_file.mth5
echo.
echo ===============================================================================

pause
