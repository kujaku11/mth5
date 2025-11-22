# -*- coding: utf-8 -*-
"""
Global pytest configuration and fixtures for MTH5 tests.

This conftest.py provides session-scoped fixtures for creating and caching
synthetic MTH5 test files that are used across multiple test modules.
The caching system ensures that these expensive file creation operations
happen only once per test session, dramatically improving test performance.

FEATURES:
- Session-scoped MTH5 file caching with global persistence
- Multi-tier caching (global cache + local cache)
- Pytest-xdist compatible (unique files per worker)
- Automatic cleanup and file management
- Support for multiple MTH5 file types (test1, test2, test3, test12rr)

USAGE:
Test modules can simply import and use the fixtures:
    def test_something(global_test1_mth5, global_test12rr_mth5):
        # Use the cached MTH5 files directly
        pass

PERFORMANCE:
- First test session: Creates global cache (~30-60s one-time cost)
- Subsequent sessions: Uses cache (< 1s setup per file type)
- Global cache persists across pytest sessions for maximum reuse
"""

import atexit
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from loguru import logger

from mth5.data.make_mth5_from_asc import (
    create_test1_h5,
    create_test2_h5,
    create_test3_h5,
    create_test12rr_h5,
)
from mth5.helpers import close_open_files
from mth5.mth5 import MTH5
from mth5.timeseries.spectre.helpers import add_fcs_to_mth5


# =============================================================================
# Global cache management
# =============================================================================

# Global cache directory - persists across test sessions
GLOBAL_CACHE_DIR = Path(tempfile.gettempdir()) / "mth5_global_test_cache"

# Track cached files for cleanup
_CACHED_FILES = {}

# File type mappings
MTH5_TYPES = {
    "test1": {
        "cache_name": "test1_global.h5",
        "create_func": lambda folder: create_test1_h5(
            target_folder=folder, file_version="0.1.0", force_make_mth5=True
        ),
    },
    "test1_v2": {
        "cache_name": "test1_v2_global.h5",
        "create_func": lambda folder: create_test1_h5(
            target_folder=folder, file_version="0.2.0", force_make_mth5=True
        ),
    },
    "test2": {
        "cache_name": "test2_global.h5",
        "create_func": lambda folder: create_test2_h5(
            target_folder=folder, file_version="0.1.0", force_make_mth5=True
        ),
    },
    "test3": {
        "cache_name": "test3_global.h5",
        "create_func": lambda folder: create_test3_h5(
            target_folder=folder, file_version="0.1.0", force_make_mth5=True
        ),
    },
    "test12rr": {
        "cache_name": "test12rr_global.h5",
        "create_func": lambda folder: create_test12rr_h5(
            target_folder=folder, file_version="0.1.0", force_make_mth5=True
        ),
    },
}


def ensure_global_cache_exists(mth5_type):
    """Ensure global cache exists for the specified MTH5 type."""
    if mth5_type not in MTH5_TYPES:
        raise ValueError(f"Unknown mth5_type: {mth5_type}")

    cache_info = MTH5_TYPES[mth5_type]
    global_cache_path = GLOBAL_CACHE_DIR / cache_info["cache_name"]

    if global_cache_path.exists():
        logger.debug(f"Global cache exists for {mth5_type}: {global_cache_path}")
        return global_cache_path

    # Create cache directory if needed
    GLOBAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating global cache for {mth5_type}...")

    # Create in temporary directory first
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = cache_info["create_func"](temp_dir)

        # Copy to global cache
        try:
            shutil.copy2(temp_path, global_cache_path)
            logger.info(f"Global cache created: {global_cache_path}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to create global cache: {e}")
            # Return temp path as fallback, though it will be deleted
            return temp_path

    return global_cache_path


def create_session_mth5_file(mth5_type, with_fcs=False):
    """
    Create a session-scoped MTH5 file with optional Fourier coefficients.

    Returns a unique copy from the global cache for use during the test session.
    """
    # Ensure global cache exists
    global_cache_path = ensure_global_cache_exists(mth5_type)

    # Create unique session file
    session_dir = tempfile.mkdtemp(prefix=f"mth5_session_{mth5_type}_")
    unique_id = str(uuid.uuid4())[:8]
    session_file = Path(session_dir) / f"{mth5_type}_session_{unique_id}.h5"

    # Copy from global cache
    shutil.copy2(global_cache_path, session_file)

    # Add Fourier coefficients if requested
    if with_fcs:
        logger.debug(f"Adding Fourier coefficients to {session_file}")
        with MTH5() as m:
            m.open_mth5(session_file, mode="a")
            add_fcs_to_mth5(m, fc_decimations=None)

    # Track for cleanup
    _CACHED_FILES[mth5_type] = session_file

    logger.debug(f"Created session MTH5 file: {session_file}")
    return session_file


# Register cleanup function
def cleanup_session_files():
    """Clean up session files and temporary directories."""
    close_open_files()
    for mth5_type, file_path in _CACHED_FILES.items():
        try:
            if file_path.exists():
                file_path.unlink()
                # Try to remove parent directory if empty
                try:
                    file_path.parent.rmdir()
                except OSError:
                    pass  # Directory not empty or other error
        except (OSError, PermissionError):
            logger.warning(f"Failed to cleanup {mth5_type} file: {file_path}")


atexit.register(cleanup_session_files)


# =============================================================================
# Session-scoped fixtures - created once per pytest session
# =============================================================================


@pytest.fixture(scope="session")
def global_test1_mth5():
    """
    Session-scoped fixture providing a cached test1 MTH5 file.

    This file is created once per test session and reused by all tests.
    Safe for read-only operations across multiple tests.
    """
    file_path = create_session_mth5_file("test1", with_fcs=False)
    yield file_path
    # Cleanup handled by atexit


@pytest.fixture(scope="session")
def global_test1_v2_mth5():
    """
    Session-scoped fixture providing a cached test1 MTH5 file (version 0.2.0).

    This file is created once per test session and reused by all tests.
    Safe for read-only operations across multiple tests.
    """
    file_path = create_session_mth5_file("test1_v2", with_fcs=False)
    yield file_path
    # Cleanup handled by atexit


@pytest.fixture(scope="session")
def global_test1_mth5_with_fcs():
    """
    Session-scoped fixture providing a cached test1 MTH5 file with Fourier coefficients.

    This includes pre-computed Fourier coefficients for spectrogram tests.
    Safe for read-only operations across multiple tests.
    """
    file_path = create_session_mth5_file("test1", with_fcs=True)
    yield file_path
    # Cleanup handled by atexit


@pytest.fixture(scope="session")
def global_test2_mth5():
    """Session-scoped fixture providing a cached test2 MTH5 file."""
    file_path = create_session_mth5_file("test2", with_fcs=False)
    yield file_path
    # Cleanup handled by atexit


@pytest.fixture(scope="session")
def global_test3_mth5():
    """Session-scoped fixture providing a cached test3 MTH5 file."""
    file_path = create_session_mth5_file("test3", with_fcs=False)
    yield file_path
    # Cleanup handled by atexit


@pytest.fixture(scope="session")
def global_test12rr_mth5():
    """
    Session-scoped fixture providing a cached test12rr MTH5 file.

    This file contains multiple stations (test1 and test2) for multivariate tests.
    """
    file_path = create_session_mth5_file("test12rr", with_fcs=False)
    yield file_path
    # Cleanup handled by atexit


@pytest.fixture(scope="session")
def global_test12rr_mth5_with_fcs():
    """
    Session-scoped fixture providing a cached test12rr MTH5 file with Fourier coefficients.

    This file contains multiple stations with pre-computed Fourier coefficients
    for multivariate spectrogram tests.
    """
    file_path = create_session_mth5_file("test12rr", with_fcs=True)
    yield file_path
    # Cleanup handled by atexit


# =============================================================================
# Function-scoped fixtures - fresh copy for each test that modifies data
# =============================================================================


@pytest.fixture
def fresh_test1_mth5(global_test1_mth5):
    """
    Function-scoped fixture providing a fresh copy of test1 MTH5 file.

    Use this fixture when your test needs to modify the MTH5 file.
    Each test gets its own isolated copy.
    """
    temp_dir = tempfile.mkdtemp()
    unique_id = str(uuid.uuid4())[:8]
    fresh_file = Path(temp_dir) / f"test1_fresh_{unique_id}.h5"

    shutil.copy2(global_test1_mth5, fresh_file)

    yield fresh_file

    # Cleanup
    close_open_files()
    try:
        fresh_file.unlink()
        fresh_file.parent.rmdir()
    except (OSError, PermissionError):
        pass


@pytest.fixture
def fresh_test12rr_mth5(global_test12rr_mth5):
    """
    Function-scoped fixture providing a fresh copy of test12rr MTH5 file.

    Use this fixture when your test needs to modify the MTH5 file.
    Each test gets its own isolated copy.
    """
    temp_dir = tempfile.mkdtemp()
    unique_id = str(uuid.uuid4())[:8]
    fresh_file = Path(temp_dir) / f"test12rr_fresh_{unique_id}.h5"

    shutil.copy2(global_test12rr_mth5, fresh_file)

    yield fresh_file

    # Cleanup
    close_open_files()
    try:
        fresh_file.unlink()
        fresh_file.parent.rmdir()
    except (OSError, PermissionError):
        pass


# =============================================================================
# Utility functions for backward compatibility
# =============================================================================


def create_fast_mth5_copy(mth5_type, target_folder):
    """
    Create a fast copy of an MTH5 file using the global cache.

    This is a backward-compatible function for existing test code.
    """
    global_cache_path = ensure_global_cache_exists(mth5_type)

    unique_id = str(uuid.uuid4())[:8]
    target_file = Path(target_folder) / f"{mth5_type}_{unique_id}.h5"

    shutil.copy2(global_cache_path, target_file)
    return target_file


# =============================================================================
# Pytest configuration
# =============================================================================


def pytest_sessionstart(session):
    """
    Called before the test session starts.

    Pre-populate the global cache to ensure best performance.
    """
    logger.info("MTH5 Test Session Starting - Pre-populating global cache...")

    # Pre-create the most commonly used cache files
    # This ensures they're available immediately for any test
    try:
        ensure_global_cache_exists("test1")
        ensure_global_cache_exists("test12rr")
        logger.info("Global cache pre-population completed successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-populate global cache: {e}")
        logger.warning("Tests will still work but may be slower on first run")


def pytest_sessionfinish(session, exitstatus):
    """
    Called after the test session finishes.

    Perform final cleanup.
    """
    logger.info("MTH5 Test Session Ending - Cleaning up...")
    cleanup_session_files()
    close_open_files()
    logger.info("MTH5 Test Session cleanup completed")


# =============================================================================
# Pytest plugin configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with MTH5-specific settings."""
    # Add custom markers if needed
    config.addinivalue_line("markers", "mth5_slow: mark test as slow MTH5 operation")
    config.addinivalue_line(
        "markers", "mth5_cache: mark test as using MTH5 global cache"
    )
