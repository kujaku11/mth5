#!/usr/bin/env python
"""
Build standalone MTH5 Validator executable using PyInstaller

This script creates a standalone executable for the MTH5 validator that can
be distributed without requiring Python installation.

Usage:
    python build_executable.py

Output:
    dist/mth5-validator.exe (Windows)
    dist/mth5-validator (Linux/Mac)

Author: MTH5 Development Team
Date: February 7, 2026
"""

import os
import platform
import shutil
import sys
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller

        print(f"✓ PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("✗ PyInstaller not found")
        print("\nInstalling PyInstaller...")
        os.system(f"{sys.executable} -m pip install pyinstaller")
        try:
            import PyInstaller

            print(f"✓ PyInstaller {PyInstaller.__version__} installed")
            return True
        except ImportError:
            print("✗ Failed to install PyInstaller")
            return False


def clean_build_dirs():
    """Clean previous build artifacts."""
    dirs_to_clean = ["build", "dist", "__pycache__"]
    files_to_clean = ["*.spec"]

    print("\nCleaning previous build artifacts...")
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")

    for pattern in files_to_clean:
        for file in Path(".").glob(pattern):
            file.unlink()
            print(f"  Removed {file}")


def build_executable():
    """Build the executable using PyInstaller."""
    import PyInstaller.__main__

    # Determine platform-specific settings
    system = platform.system()
    exe_name = "mth5-validator"
    if system == "Windows":
        exe_name += ".exe"

    print(f"\nBuilding executable for {system}...")
    print("=" * 80)

    # PyInstaller arguments
    args = [
        "mth5/utils/cli.py",  # Entry point
        "--onefile",  # Single executable
        f'--name={exe_name.replace(".exe", "")}',  # Output name
        "--console",  # Console application
        "--clean",  # Clean cache
        # Hidden imports (h5py needs special handling)
        "--hidden-import=h5py.defs",
        "--hidden-import=h5py.utils",
        "--hidden-import=h5py._proxy",
        "--hidden-import=h5py.h5ac",
        # Collect all submodules
        "--collect-all=mt_metadata",
        "--collect-all=mth5",
        "--copy-metadata=mth5",
        "--copy-metadata=mt_metadata",
        "--copy-metadata=h5py",
        # Exclude unnecessary modules to reduce size
        "--exclude-module=matplotlib",
        "--exclude-module=tkinter",
        "--exclude-module=PyQt5",
        "--exclude-module=PyQt6",
        "--exclude-module=PySide2",
        "--exclude-module=PySide6",
        "--exclude-module=jupyter",
        "--exclude-module=notebook",
        "--exclude-module=IPython",
        "--exclude-module=PIL",
        "--exclude-module=pytest",
        # Logging
        "--log-level=WARN",
    ]

    # Add platform-specific options
    if system == "Windows":
        args.extend(
            [
                "--version-file=version_info.txt",  # Optional: version info
            ]
        )

    print("\nPyInstaller configuration:")
    print(f"  Entry point: mth5/utils/cli.py")
    print(f"  Output name: {exe_name}")
    print(f"  Mode: Single file")
    print(f"  Console: Yes")
    print(f"  Platform: {system}")

    try:
        PyInstaller.__main__.run(args)
        print("\n" + "=" * 80)
        print("✓ Build completed successfully!")
        return True
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Build failed: {e}")
        return False


def test_executable():
    """Test the built executable."""
    system = platform.system()
    exe_path = Path("dist") / (
        "mth5-validator.exe" if system == "Windows" else "mth5-validator"
    )

    if not exe_path.exists():
        print(f"\n✗ Executable not found: {exe_path}")
        return False

    print(f"\nTesting executable: {exe_path}")
    print("=" * 80)

    # Get file size
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

    # Test help command
    print("\n  Running: mth5-validator --help")
    print("-" * 80)
    result = os.system(f'"{exe_path}" --help')

    if result == 0:
        print("-" * 80)
        print("✓ Executable test passed!")
        return True
    else:
        print("✗ Executable test failed")
        return False


def print_summary():
    """Print build summary and usage instructions."""
    system = platform.system()
    exe_name = "mth5-validator.exe" if system == "Windows" else "mth5-validator"
    exe_path = Path("dist") / exe_name

    print("\n" + "=" * 80)
    print("BUILD SUMMARY")
    print("=" * 80)

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Executable created: {exe_path.absolute()}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Platform: {system}")

        print("\n" + "-" * 80)
        print("USAGE")
        print("-" * 80)
        print(f"\n  Basic validation:")
        print(f"    {exe_path} validate data.mth5")
        print(f"\n  Verbose output:")
        print(f"    {exe_path} validate data.mth5 --verbose")
        print(f"\n  Check data integrity:")
        print(f"    {exe_path} validate data.mth5 --check-data")
        print(f"\n  JSON output:")
        print(f"    {exe_path} validate data.mth5 --json")

        print("\n" + "-" * 80)
        print("DISTRIBUTION")
        print("-" * 80)
        print(f"\n  The executable in dist/ can be distributed standalone.")
        print(f"  Users do NOT need Python or any dependencies installed.")
        print(f"  Copy {exe_name} to any system and run it directly.")

        if system == "Windows":
            print(f"\n  Windows Note:")
            print(f"    First run may trigger Windows Defender.")
            print(f"    Consider code signing for production distribution.")
        elif system == "Darwin":
            print(f"\n  macOS Note:")
            print(f"    Users may need to allow in System Preferences > Security.")
            print(f"    Consider signing and notarizing for distribution.")
    else:
        print(f"\n✗ Build failed - executable not found")

    print("\n" + "=" * 80)


def main():
    """Main build process."""
    print("=" * 80)
    print("MTH5 Validator - Standalone Executable Builder")
    print("=" * 80)
    print(f"\nPlatform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")

    # Check PyInstaller
    if not check_pyinstaller():
        print("\n✗ Cannot proceed without PyInstaller")
        sys.exit(1)

    # Clean previous builds
    clean_build_dirs()

    # Build executable
    if not build_executable():
        print("\n✗ Build failed")
        sys.exit(1)

    # Test executable
    test_executable()

    # Print summary
    print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
