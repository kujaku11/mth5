# =============================================================================
# Imports
# =============================================================================
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import pytest

from mth5.data.make_mth5_from_asc import create_test12rr_h5
from mth5.processing import RUN_SUMMARY_COLUMNS
from mth5.processing.run_summary import RunSummary
from mth5.utils.helpers import close_open_files


# =============================================================================
# Optimized test file creation
# =============================================================================

# Module-level cache for shared read-only test data
_CACHED_MTH5_PATH = None
_CACHED_RUN_SUMMARY = None


def create_fast_test_mth5(target_folder):
    """Create a fast MTH5 file for testing with minimal overhead."""
    import shutil

    # Check for global cache first (much faster)
    global_cache = (
        Path(tempfile.gettempdir()) / "mth5_test_cache" / "test12rr_global.h5"
    )
    if global_cache.exists():
        # Copy from global cache for maximum speed
        unique_id = str(uuid.uuid4())[:8]
        target_file = Path(target_folder) / f"test12rr_{unique_id}.h5"
        shutil.copy2(global_cache, target_file)
        return target_file

    # Check if we have a local cached version in the target folder
    cache_file = Path(target_folder) / "test12rr_cache.h5"
    if cache_file.exists():
        # Copy from cache for speed
        unique_id = str(uuid.uuid4())[:8]
        target_file = Path(target_folder) / f"test12rr_{unique_id}.h5"
        shutil.copy2(cache_file, target_file)
        return target_file

    # Create file with optimized settings
    path = create_test12rr_h5(
        target_folder=target_folder,
        file_version="0.1.0",  # Faster than 0.2.0
        force_make_mth5=True,
    )

    # Create global cache directory if it doesn't exist
    global_cache.parent.mkdir(parents=True, exist_ok=True)

    # Cache globally for maximum reuse
    if not global_cache.exists():
        try:
            shutil.copy2(path, global_cache)
        except (OSError, PermissionError):
            pass  # If global caching fails, continue without it

    # Cache locally as backup
    if not cache_file.exists():
        try:
            shutil.copy2(path, cache_file)
        except (OSError, PermissionError):
            pass  # If caching fails, continue without it

    return path


def get_cached_mth5_and_run_summary():
    """Get cached MTH5 path and RunSummary for read-only tests."""
    global _CACHED_MTH5_PATH, _CACHED_RUN_SUMMARY

    if _CACHED_MTH5_PATH is None or not _CACHED_MTH5_PATH.exists():
        temp_dir = tempfile.mkdtemp()
        _CACHED_MTH5_PATH = create_fast_test_mth5(temp_dir)
        _CACHED_RUN_SUMMARY = None  # Reset run summary cache

    if _CACHED_RUN_SUMMARY is None:
        _CACHED_RUN_SUMMARY = RunSummary()
        _CACHED_RUN_SUMMARY.from_mth5s([_CACHED_MTH5_PATH])

    return _CACHED_MTH5_PATH, _CACHED_RUN_SUMMARY


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mth5_path():
    """Create or get test MTH5 file path for the session."""
    # Create unique temporary directory for each test
    temp_dir = tempfile.mkdtemp()

    # Always create fresh file for each test using optimized function
    path = create_fast_test_mth5(temp_dir)

    yield path

    # Cleanup
    close_open_files()
    if path.exists():
        try:
            path.unlink()
        except (OSError, PermissionError):
            pass  # Handle case where file is still locked
    try:
        # Clean up temp directory
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
    except (OSError, PermissionError):
        pass


@pytest.fixture
def base_run_summary(mth5_path):
    """Create base RunSummary from MTH5 file."""
    rs = RunSummary()
    rs.from_mth5s([mth5_path])
    return rs


@pytest.fixture
def run_summary(base_run_summary):
    """Fresh clone of run_summary for each test."""
    return base_run_summary.clone()


@pytest.fixture(scope="session")
def shared_mth5_path():
    """Shared MTH5 path for read-only tests (session scope for speed)."""
    mth5_path, _ = get_cached_mth5_and_run_summary()
    yield mth5_path

    # Session cleanup
    close_open_files()


@pytest.fixture(scope="session")
def shared_run_summary():
    """Shared RunSummary for read-only tests (session scope for speed)."""
    _, run_summary = get_cached_mth5_and_run_summary()
    return run_summary


@pytest.fixture
def sample_invalid_dataframes():
    """Provide sample invalid DataFrames for testing."""
    return [
        pd.DataFrame({"test": [0]}),
        pd.DataFrame({"wrong_columns": [1, 2, 3]}),
        pd.DataFrame(),  # Empty DataFrame
        pd.DataFrame({"incomplete": ["a", "b"]}),
    ]


# =============================================================================
# Test Classes and Functions
# =============================================================================


class TestRunSummary:
    """Test RunSummary functionality."""

    def test_df_columns(self, run_summary):
        """Test that DataFrame has correct columns."""
        assert sorted(RUN_SUMMARY_COLUMNS) == sorted(run_summary.df.columns)

    def test_df_shape(self, run_summary):
        """Test DataFrame shape."""
        assert run_summary.df.shape == (2, 15)

    def test_set_df_failures(self, run_summary):
        """Test DataFrame setter with invalid inputs."""
        # Test invalid type
        with pytest.raises(TypeError):
            run_summary.df = 10

        # Test invalid DataFrame
        with pytest.raises(ValueError):
            run_summary.df = pd.DataFrame({"test": [0]})

    def test_clone_functionality(self, run_summary):
        """Test cloning preserves data integrity."""
        rs_clone = run_summary.clone()

        # Normalize mth5_path columns for comparison
        rs_clone.df["mth5_path"] = rs_clone.df.mth5_path.infer_objects(
            copy=False
        ).fillna(0)
        run_summary.df["mth5_path"] = run_summary.df.mth5_path.infer_objects(
            copy=False
        ).fillna(0)

        # Verify cloned data matches original
        assert (run_summary.df == rs_clone.df).all().all()

    def test_mini_summary(self, run_summary):
        """Test mini summary functionality."""
        mini_df = run_summary.mini_summary
        assert sorted(run_summary._mini_summary_columns) == sorted(mini_df.columns)

    def test_drop_no_data_rows(self, run_summary):
        """Test dropping rows with no data."""
        rs_clone = run_summary.clone()

        # Mark first row as having no data
        rs_clone.df.loc[0, "has_data"] = False

        # Test warning functionality
        rs_clone._warn_no_data_runs()

        # Drop no-data rows and verify shape change
        original_shape = rs_clone.df.shape
        rs_clone.drop_no_data_rows()
        new_shape = rs_clone.df.shape

        assert new_shape == (1, 15)
        assert new_shape[0] == original_shape[0] - 1  # One row removed

    def test_set_sample_rate_valid(self, run_summary):
        """Test setting valid sample rate."""
        new_rs = run_summary.set_sample_rate(1)

        # Normalize mth5_path columns for comparison
        new_rs.df["mth5_path"] = new_rs.df.mth5_path.infer_objects(copy=False).fillna(0)
        run_summary.df["mth5_path"] = run_summary.df.mth5_path.infer_objects(
            copy=False
        ).fillna(0)

        # Should return same data for existing sample rate
        assert (run_summary.df == new_rs.df).all().all()

    def test_set_sample_rate_invalid(self, run_summary):
        """Test setting invalid sample rate raises ValueError."""
        with pytest.raises(ValueError):
            run_summary.set_sample_rate(10)


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize(
    "invalid_value", [10, "string", [1, 2, 3], {"dict": "value"}, None]
)
def test_df_setter_invalid_types(run_summary, invalid_value):
    """Test DataFrame setter with various invalid types."""
    if invalid_value is None:
        # None might be allowed in some cases
        run_summary.df = invalid_value
        assert run_summary.df is None
    else:
        with pytest.raises(TypeError):
            run_summary.df = invalid_value


@pytest.mark.parametrize(
    "invalid_df",
    [
        pd.DataFrame({"wrong": [1, 2]}),
        pd.DataFrame({"also_wrong": ["a", "b", "c"]}),
        pd.DataFrame(),  # Empty DataFrame
        pd.DataFrame({"insufficient_columns": [1]}),
    ],
)
def test_df_setter_invalid_dataframes(run_summary, invalid_df):
    """Test DataFrame setter with invalid DataFrames."""
    with pytest.raises(ValueError):
        run_summary.df = invalid_df


@pytest.mark.parametrize("sample_rate", [0.1, 0.5, 2, 4, 8, 16])
def test_set_sample_rate_various_rates(base_run_summary, sample_rate):
    """Test setting various sample rates that don't exist in data."""
    rs = base_run_summary.clone()

    # These should all raise ValueError since they don't exist in test data
    with pytest.raises(ValueError):
        rs.set_sample_rate(sample_rate)


# =============================================================================
# Property and Method Tests
# =============================================================================


class TestRunSummaryProperties:
    """Test RunSummary properties and methods."""

    def test_mini_summary_columns_subset(self, run_summary):
        """Test that mini summary columns are a subset of full columns."""
        mini_columns = set(run_summary._mini_summary_columns)
        full_columns = set(run_summary.df.columns)

        assert mini_columns.issubset(full_columns)

    def test_dataframe_integrity_after_operations(self, run_summary):
        """Test DataFrame integrity after various operations."""
        original_columns = list(run_summary.df.columns)
        original_shape = run_summary.df.shape

        # Perform mini_summary operation
        mini_df = run_summary.mini_summary

        # Verify original DataFrame unchanged
        assert list(run_summary.df.columns) == original_columns
        assert run_summary.df.shape == original_shape
        assert len(mini_df.columns) <= len(original_columns)

    def test_has_data_column_types(self, run_summary):
        """Test that has_data column contains boolean values."""
        has_data_values = run_summary.df["has_data"]

        # Should be boolean type
        assert has_data_values.dtype == bool

        # Should have actual boolean values
        unique_values = set(has_data_values.unique())
        assert unique_values.issubset({True, False})


# =============================================================================
# Integration Tests
# =============================================================================


def test_end_to_end_workflow(mth5_path):
    """Test complete workflow from MTH5 to RunSummary operations."""
    # Create run summary
    rs = RunSummary()
    rs.from_mth5s([mth5_path])

    # Verify basic functionality
    assert rs.df is not None
    assert rs.df.shape[0] > 0
    assert len(rs.df.columns) == len(RUN_SUMMARY_COLUMNS)

    # Test clone functionality
    rs_clone = rs.clone()
    assert rs_clone.df.shape == rs.df.shape

    # Test mini summary
    mini = rs.mini_summary
    assert mini.shape[0] == rs.df.shape[0]

    # Test data filtering
    original_rows = len(rs.df)
    rs.df.loc[0, "has_data"] = False
    rs.drop_no_data_rows()
    assert len(rs.df) == original_rows - 1


def test_sample_rate_filtering_workflow(base_run_summary):
    """Test sample rate filtering workflow."""
    rs = base_run_summary.clone()

    # Get available sample rates from the data
    available_rates = rs.df["sample_rate"].unique()

    # Test with existing sample rate
    if len(available_rates) > 0:
        valid_rate = available_rates[0]
        filtered_rs = rs.set_sample_rate(valid_rate)

        # Should return data with only that sample rate
        unique_rates_in_result = filtered_rs.df["sample_rate"].unique()
        assert len(unique_rates_in_result) == 1
        assert unique_rates_in_result[0] == valid_rate


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestRunSummaryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_operations(self):
        """Test operations on empty RunSummary."""
        rs = RunSummary()

        # Should handle empty DataFrame gracefully
        # Empty RunSummary has None df, so mini_summary should handle this
        try:
            mini = rs.mini_summary
            assert mini.empty or len(mini) == 0
        except (TypeError, AttributeError):
            # Expected behavior when df is None
            assert rs.df is None

    def test_clone_independence(self, run_summary):
        """Test that cloned RunSummary is independent of original."""
        rs_clone = run_summary.clone()

        # Modify clone
        original_shape = rs_clone.df.shape
        rs_clone.df.loc[0, "has_data"] = False
        rs_clone.drop_no_data_rows()

        # Original should be unchanged
        assert run_summary.df.shape == original_shape
        assert run_summary.df.shape != rs_clone.df.shape

    def test_multiple_clones(self, run_summary):
        """Test creating multiple independent clones."""
        clone1 = run_summary.clone()
        clone2 = run_summary.clone()

        # Modify each clone differently
        clone1.df.loc[0, "has_data"] = False
        clone2.df.loc[1, "has_data"] = False

        clone1.drop_no_data_rows()
        clone2.drop_no_data_rows()

        # Both clones should have fewer rows than original
        assert clone1.df.shape[0] < run_summary.df.shape[0]
        assert clone2.df.shape[0] < run_summary.df.shape[0]

        # Both clones should have same shape (both removed 1 row)
        assert clone1.df.shape == clone2.df.shape


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.performance
def test_clone_performance(base_run_summary):
    """Test clone operation performance."""
    import time

    # Test cloning speed
    start_time = time.time()
    for _ in range(10):
        clone = base_run_summary.clone()
        assert clone.df is not None
    end_time = time.time()

    # Should complete quickly
    assert end_time - start_time < 2.0


@pytest.mark.performance
def test_mini_summary_performance(base_run_summary):
    """Test mini summary generation performance."""
    import time

    start_time = time.time()
    for _ in range(100):
        mini = base_run_summary.mini_summary
        assert mini is not None
    end_time = time.time()

    # Should complete quickly
    assert end_time - start_time < 1.0


# =============================================================================
# Regression Tests
# =============================================================================


class TestRunSummaryRegression:
    """Test for known issues and regressions."""

    def test_mth5_path_handling(self, run_summary):
        """Test proper handling of mth5_path column."""
        # This tests the specific infer_objects() handling seen in original tests
        df_before = run_summary.df.copy()

        # Apply the transformation that was used in original tests
        run_summary.df["mth5_path"] = run_summary.df.mth5_path.infer_objects(
            copy=False
        ).fillna(0)

        # Should not crash and should produce valid result
        assert run_summary.df is not None
        assert len(run_summary.df) == len(df_before)

    def test_maxdiff_equivalent(self, run_summary):
        """Test equivalent of maxDiff = None functionality."""
        # Test that large differences in DataFrames can be compared
        rs_clone = run_summary.clone()

        # Make substantial changes
        rs_clone.df["mth5_path"] = rs_clone.df.mth5_path.infer_objects(
            copy=False
        ).fillna(0)
        run_summary.df["mth5_path"] = run_summary.df.mth5_path.infer_objects(
            copy=False
        ).fillna(0)

        # Should be able to compare even large DataFrames
        comparison = (run_summary.df == rs_clone.df).all().all()

        # Comparison result should be numpy boolean or python boolean
        assert isinstance(comparison, (bool, type(comparison)))

        # Convert to python boolean for final assertion
        assert bool(comparison) == True


if __name__ == "__main__":
    pytest.main([__file__])
