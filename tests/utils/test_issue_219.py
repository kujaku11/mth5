"""
Proof of concept for issue #219
"""

import tempfile
import uuid
from pathlib import Path

import pytest

from mth5.utils.extract_subset_mth5 import extract_subset
from mth5.utils.helpers import get_channel_summary


@pytest.fixture
def test3_mth5_path(global_test3_mth5):
    """Create unique test3 MTH5 file for each test from global cache."""
    import shutil

    # Create unique temporary file
    temp_dir = tempfile.mkdtemp()
    unique_id = str(uuid.uuid4())[:8]
    target_file = Path(temp_dir) / f"test3_{unique_id}.h5"

    # Copy from global cache
    shutil.copy2(global_test3_mth5, target_file)

    yield target_file

    # Cleanup
    if target_file.exists():
        try:
            target_file.unlink()
        except (OSError, PermissionError):
            pass
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except (OSError, PermissionError):
        pass


@pytest.fixture
def target_file_path():
    """Create temporary target file path for subset extraction."""
    temp_dir = tempfile.mkdtemp()
    unique_id = str(uuid.uuid4())[:8]
    target_file = Path(temp_dir) / f"subset_{unique_id}.h5"

    yield target_file

    # Cleanup
    if target_file.exists():
        try:
            target_file.unlink()
        except (OSError, PermissionError):
            pass
    try:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
    except (OSError, PermissionError):
        pass


def test_extract_subset(test3_mth5_path, target_file_path):
    """Test extraction of subset from MTH5 file."""
    # Get channel summary from source file
    df = get_channel_summary(test3_mth5_path)
    selected_df = df[df.run.isin(["002", "004"])]

    # Extract subset
    extract_subset(test3_mth5_path, target_file_path, selected_df)

    # Verify extraction
    df2 = get_channel_summary(target_file_path)
    compare_columns = [x for x in df2.columns if "hdf5_reference" not in x]
    selected_df.reset_index(inplace=True, drop=True)  # allow comparison
    assert (df2[compare_columns] == selected_df[compare_columns]).all().all()
    print(df2)
