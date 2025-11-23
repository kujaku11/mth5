# -*- coding: utf-8 -*-
"""
Pytest configuration for version_1 tests.

This conftest provides pytest-xdist compatible fixtures that ensure
unique file paths for each worker process, preventing file locking issues.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def worker_id(request):
    """
    Get the current pytest-xdist worker ID.

    Returns:
        str: Worker ID (e.g., 'gw0', 'gw1', 'master') or 'master' if not using xdist
    """
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"


@pytest.fixture(scope="session")
def test_dir_path():
    """Get the test directory path."""
    return Path(__file__).parent


def get_worker_safe_filename(base_filename: str, worker_id: str) -> str:
    """
    Generate a worker-safe filename by inserting the worker ID before the extension.

    Args:
        base_filename: Original filename (e.g., "test.h5")
        worker_id: Worker ID from pytest-xdist (e.g., "gw0", "master")

    Returns:
        str: Worker-safe filename (e.g., "test_gw0.h5", "test_master.h5")
    """
    path = Path(base_filename)
    stem = path.stem
    suffix = path.suffix
    return f"{stem}_{worker_id}{suffix}"


@pytest.fixture(scope="session")
def make_worker_safe_path(test_dir_path, worker_id):
    """
    Factory fixture to create worker-safe file paths.

    Returns a function that takes a base filename and returns a Path object
    with the worker ID inserted to avoid conflicts in parallel testing.

    Usage:
        def test_something(make_worker_safe_path):
            test_file = make_worker_safe_path("test.h5")
            # test_file will be Path("test_gw0.h5") for worker gw0
    """

    def _make_path(base_filename: str) -> Path:
        """Create worker-safe path for the given filename."""
        safe_filename = get_worker_safe_filename(base_filename, worker_id)
        return test_dir_path / safe_filename

    return _make_path


@pytest.fixture(scope="session")
def cleanup_test_files(request, test_dir_path):
    """
    Register test files for cleanup after session ends.

    Returns a function to register files for cleanup.
    """
    files_to_cleanup = []

    def _register_file(filepath: Path):
        """Register a file for cleanup."""
        if filepath not in files_to_cleanup:
            files_to_cleanup.append(filepath)

    def _cleanup():
        """Clean up registered files."""
        for filepath in files_to_cleanup:
            if filepath.exists():
                try:
                    filepath.unlink()
                except (PermissionError, OSError) as e:
                    # Log but don't fail if cleanup fails
                    print(f"Warning: Could not delete {filepath}: {e}")

    # Register cleanup function
    request.addfinalizer(_cleanup)

    return _register_file
