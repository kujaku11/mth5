#!/usr/bin/env python
"""
Build Standalone MTH5 Validator Executable
===========================================

Builds a lightweight executable (~20-30 MB) that only depends on h5py.

Much smaller and simpler than the full mth5-based validator because:
- No scipy/numpy/obspy dependencies
- No matplotlib/pandas dependencies
- No mth5 I/O stack
- Just h5py + standard library

Usage:
    python build_standalone_validator.py

Output:
    dist/mth5-validator.exe (Windows) - ~20-30 MB
    dist/mth5-validator (Linux/Mac)

Author: MTH5 Development Team
Date: February 9, 2026
"""

import os
import platform
import shutil
import sys
from pathlib import Path


def check_dependencies():
    """Check that required dependencies are installed."""
    print("Checking dependencies...")

    # Check h5py
    try:
        import h5py

        print(f"  ✓ h5py {h5py.__version__}")
    except ImportError:
        print("  ✗ h5py not found - installing...")
        os.system(f"{sys.executable} -m pip install h5py")

    # Check PyInstaller
    try:
        import PyInstaller

        print(f"  ✓ PyInstaller {PyInstaller.__version__}")
        return True
    except ImportError:
        print("  ✗ PyInstaller not found - installing...")
        os.system(f"{sys.executable} -m pip install pyinstaller")
        try:
            import PyInstaller

            print(f"  ✓ PyInstaller {PyInstaller.__version__} installed")
            return True
        except ImportError:
            print("  ✗ Failed to install PyInstaller")
            return False


def clean_build_dirs():
    """Clean previous build artifacts."""
    import stat
    import time

    def handle_remove_readonly(func, path, exc):
        """Error handler for Windows read-only files."""
        if func in (os.rmdir, os.remove, os.unlink):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

    dirs_to_clean = ["build", "dist"]
    files_to_clean = ["*.spec"]

    print("\nCleaning previous build artifacts...")

    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            try:
                time.sleep(0.1)
                shutil.rmtree(dir_name, onerror=handle_remove_readonly)
                print(f"  Removed {dir_name}/")
            except Exception as e:
                print(f"  Warning: Could not fully remove {dir_name}/ ({e})")

    for pattern in files_to_clean:
        for file in Path(".").glob(pattern):
            try:
                file.unlink()
                print(f"  Removed {file}")
            except Exception as e:
                print(f"  Warning: Could not remove {file} ({e})")


def build_executable():
    """Build the standalone validator executable."""
    import PyInstaller.__main__

    system = platform.system()
    exe_name = "mth5-validator"
    if system == "Windows":
        exe_name += ".exe"

    print(f"\nBuilding Standalone MTH5 Validator for {system}...")
    print("=" * 80)

    # Simple PyInstaller arguments - h5py + minimal numpy
    args = [
        "mth5_validator_standalone.py",  # Standalone entry point
        "--onefile",  # Single executable
        f'--name={exe_name.replace(".exe", "")}',  # Output name
        "--console",  # Console application
        "--clean",  # Clean cache
        # Collect h5py completely (includes numpy automatically)
        "--collect-all=h5py",
        "--hidden-import=h5py.defs",
        "--hidden-import=h5py.utils",
        "--hidden-import=h5py._proxy",
        "--hidden-import=h5py.h5ac",
        "--copy-metadata=h5py",
        "--copy-metadata=numpy",
        # Exclude heavy packages we don't need
        "--exclude-module=matplotlib",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "--exclude-module=tkinter",
        "--exclude-module=PyQt5",
        "--exclude-module=PyQt6",
        "--exclude-module=jupyter",
        "--exclude-module=IPython",
        "--exclude-module=pytest",
        "--exclude-module=obspy",
        "--exclude-module=mt_metadata",
        "--exclude-module=mth5",
        # Logging
        "--log-level=WARN",
    ]

    print("\nPyInstaller configuration:")
    print(f"  Entry point: mth5_validator_standalone.py")
    print(f"  Output name: {exe_name}")
    print(f"  Mode: Single file")
    print(f"  Dependencies: h5py only")
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
        print(f"\n✓ Standalone executable created: {exe_path.absolute()}")
        print(f"  Size: {size_mb:.1f} MB (lightweight - h5py only!)")
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
        print("ADVANTAGES")
        print("-" * 80)
        print(f"\n  ✓ Small size: ~20-30 MB (vs 150+ MB for full mth5)")
        print(f"  ✓ Simple dependencies: h5py only")
        print(f"  ✓ No scipy/obspy/matplotlib bloat")
        print(f"  ✓ Fast startup and execution")
        print(f"  ✓ Standalone - no Python install needed")

        print("\n" + "-" * 80)
        print("DISTRIBUTION")
        print("-" * 80)
        print(f"\n  The executable in dist/ can be distributed standalone.")
        print(f"  Users do NOT need Python or any dependencies installed.")
        print(f"  Copy {exe_name} to any system and run it directly.")
    else:
        print(f"\n✗ Build failed - executable not found")

    print("\n" + "=" * 80)


def main():
    """Main build process."""
    print("=" * 80)
    print("MTH5 Standalone Validator - Executable Builder")
    print("=" * 80)
    print(f"\nPlatform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")

    # Check standalone script exists
    if not Path("mth5_validator_standalone.py").exists():
        print("\n✗ Error: mth5_validator_standalone.py not found!")
        print("  Make sure you're running this from the mth5 repository root.")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        print("\n✗ Cannot proceed without required dependencies")
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
