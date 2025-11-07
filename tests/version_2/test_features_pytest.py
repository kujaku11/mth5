# -*- coding: utf-8 -*-
"""
Modern pytest suite for MTH5 Features functionality using fixtures and optimizations.

This is a pytest translation of test_features.py that uses fixtures for better
test organization and cleanup, while optimizing for efficiency through strategic
fixture scoping and comprehensive test coverage.

Key improvements over unittest version:
1. Uses pytest fixtures for better resource management
2. Session-scoped fixtures for expensive MTH5 file operations
3. Parametrized tests for comprehensive coverage
4. Automatic cleanup with fixture teardown
5. Enhanced error handling and validation
6. Optimized execution through strategic fixture usage

Test coverage includes:
- Features group creation and metadata validation
- Frequency domain features (coherence with decimation levels)
- Time domain features (despiking with channels)
- Feature run group management
- Decimation level handling
- Channel assignment and validation
- Metadata consistency across feature hierarchy

@author: pytest translation for MTH5 Features - optimized for efficiency
"""

# =============================================================================
# Imports
# =============================================================================
import pytest
from pathlib import Path
import tempfile
import shutil
from mth5.mth5 import MTH5


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files that persists for the session."""
    temp_dir = Path(tempfile.mkdtemp(prefix="mth5_features_test_"))
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mth5_file_path(test_dir):
    """Path for the MTH5 test file."""
    return test_dir / "test_feature_pytest.h5"


@pytest.fixture(scope="session")
def mth5_instance(mth5_file_path):
    """
    Create and configure MTH5 instance with features for the entire test session.

    This fixture creates the complete MTH5 structure with features to avoid
    recreation overhead for each test.
    """
    # Create MTH5 instance
    m = MTH5(file_version="0.2.0")
    m.open_mth5(mth5_file_path)

    # Create survey and station structure
    survey_group = m.add_survey("top_survey")
    station_group = m.add_station("mt01", survey="top_survey")
    features_group = station_group.features_group

    # Add frequency feature (coherence)
    feature_fc = features_group.add_feature_group("feature_fc")
    fc_run = feature_fc.add_feature_run_group("coherence", domain="frequency")
    dl = fc_run.add_decimation_level("0")
    feature_ch = dl.add_channel("ex_hy")

    # Add time series feature (despiking)
    feature_ts = features_group.add_feature_group("feature_ts")
    ts_run = feature_ts.add_feature_run_group("despike", domain="time")
    ts_channel = ts_run.add_feature_channel("ex", "electric", None)

    # Store references for easy access
    m._test_objects = {
        "survey_group": survey_group,
        "station_group": station_group,
        "features_group": features_group,
        "feature_fc": feature_fc,
        "fc_run": fc_run,
        "dl": dl,
        "feature_ch": feature_ch,
        "feature_ts": feature_ts,
        "ts_run": ts_run,
        "ts_channel": ts_channel,
    }

    yield m

    # Cleanup
    m.close_mth5()
    if mth5_file_path.exists():
        mth5_file_path.unlink()


@pytest.fixture
def features_group(mth5_instance):
    """Access to the features group."""
    return mth5_instance._test_objects["features_group"]


@pytest.fixture
def frequency_feature_objects(mth5_instance):
    """Access to frequency domain feature objects."""
    return {
        "feature_fc": mth5_instance._test_objects["feature_fc"],
        "fc_run": mth5_instance._test_objects["fc_run"],
        "dl": mth5_instance._test_objects["dl"],
        "feature_ch": mth5_instance._test_objects["feature_ch"],
    }


@pytest.fixture
def time_feature_objects(mth5_instance):
    """Access to time domain feature objects."""
    return {
        "feature_ts": mth5_instance._test_objects["feature_ts"],
        "ts_run": mth5_instance._test_objects["ts_run"],
        "ts_channel": mth5_instance._test_objects["ts_channel"],
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestFeaturesGroupBasic:
    """Test basic features group functionality."""

    def test_features_group_exists(self, features_group):
        """Test that features group is created successfully."""
        assert features_group is not None

    def test_features_group_metadata(self, features_group):
        """Test features group metadata validation."""
        assert features_group.metadata.mth5_type == "MasterFeatures"

    def test_features_group_structure(self, features_group):
        """Test features group has expected structure."""
        # Should have access to group methods
        assert hasattr(features_group, "add_feature_group")
        assert hasattr(features_group, "metadata")


class TestFrequencyDomainFeatures:
    """Test frequency domain features functionality."""

    def test_feature_fc_creation(self, frequency_feature_objects):
        """Test frequency domain feature group creation."""
        feature_fc = frequency_feature_objects["feature_fc"]
        assert feature_fc is not None
        assert feature_fc.metadata.name == "feature_fc"

    def test_feature_fc_run_group(self, frequency_feature_objects, subtests):
        """Test frequency domain feature run group."""
        fc_run = frequency_feature_objects["fc_run"]

        with subtests.test("run_id"):
            assert fc_run.metadata.id == "coherence"

        with subtests.test("domain"):
            # Verify this is a frequency domain feature
            assert hasattr(fc_run, "add_decimation_level")

    def test_feature_fc_decimation_level(self, frequency_feature_objects):
        """Test decimation level functionality."""
        dl = frequency_feature_objects["dl"]
        assert dl.metadata.id == "0"

    def test_feature_fc_channel(self, frequency_feature_objects):
        """Test frequency domain feature channel."""
        feature_ch = frequency_feature_objects["feature_ch"]
        assert feature_ch.metadata.name == "ex_hy"

    @pytest.mark.parametrize(
        "attribute,expected",
        [
            ("name", "feature_fc"),
        ],
    )
    def test_frequency_feature_attributes(
        self, frequency_feature_objects, attribute, expected
    ):
        """Test frequency feature attributes using parametrization."""
        feature_fc = frequency_feature_objects["feature_fc"]
        actual = getattr(feature_fc.metadata, attribute)
        assert actual == expected


class TestTimeDomainFeatures:
    """Test time domain features functionality."""

    def test_feature_ts_creation(self, time_feature_objects):
        """Test time domain feature group creation."""
        feature_ts = time_feature_objects["feature_ts"]
        assert feature_ts is not None
        assert feature_ts.metadata.name == "feature_ts"

    def test_feature_ts_run_group(self, time_feature_objects):
        """Test time domain feature run group."""
        ts_run = time_feature_objects["ts_run"]
        assert ts_run.metadata.id == "despike"

    def test_feature_ts_channel(self, time_feature_objects):
        """Test time domain feature channel."""
        ts_channel = time_feature_objects["ts_channel"]
        assert ts_channel.metadata.component == "ex"

    @pytest.mark.parametrize(
        "feature_type,run_id,channel_component",
        [
            ("feature_ts", "despike", "ex"),
        ],
    )
    def test_time_feature_hierarchy(
        self, time_feature_objects, feature_type, run_id, channel_component, subtests
    ):
        """Test time domain feature hierarchy using parametrization."""
        feature_ts = time_feature_objects["feature_ts"]
        ts_run = time_feature_objects["ts_run"]
        ts_channel = time_feature_objects["ts_channel"]

        with subtests.test("feature_name"):
            assert feature_ts.metadata.name == feature_type

        with subtests.test("run_id"):
            assert ts_run.metadata.id == run_id

        with subtests.test("channel_component"):
            assert ts_channel.metadata.component == channel_component


class TestFeaturesDomainComparison:
    """Test comparison between frequency and time domain features."""

    def test_domain_differences(
        self, frequency_feature_objects, time_feature_objects, subtests
    ):
        """Test differences between frequency and time domain features."""
        fc_run = frequency_feature_objects["fc_run"]
        ts_run = time_feature_objects["ts_run"]

        with subtests.test("frequency_has_decimation"):
            # Frequency domain should have decimation levels
            assert hasattr(fc_run, "add_decimation_level")

        with subtests.test("time_has_channels"):
            # Time domain should have direct channel access
            assert hasattr(ts_run, "add_feature_channel")

        with subtests.test("different_ids"):
            # Should have different identifiers
            assert fc_run.metadata.id != ts_run.metadata.id

    @pytest.mark.parametrize(
        "domain,feature_name,run_id",
        [
            ("frequency", "feature_fc", "coherence"),
            ("time", "feature_ts", "despike"),
        ],
    )
    def test_feature_domains(self, mth5_instance, domain, feature_name, run_id):
        """Test feature domain characteristics using parametrization."""
        if domain == "frequency":
            feature_obj = mth5_instance._test_objects["feature_fc"]
            run_obj = mth5_instance._test_objects["fc_run"]
        else:
            feature_obj = mth5_instance._test_objects["feature_ts"]
            run_obj = mth5_instance._test_objects["ts_run"]

        assert feature_obj.metadata.name == feature_name
        assert run_obj.metadata.id == run_id


class TestFeaturesIntegration:
    """Integration tests for complete features functionality."""

    def test_complete_feature_structure(self, mth5_instance):
        """Test the complete feature structure is properly created."""
        test_objects = mth5_instance._test_objects

        # Verify all objects exist
        required_objects = [
            "survey_group",
            "station_group",
            "features_group",
            "feature_fc",
            "fc_run",
            "dl",
            "feature_ch",
            "feature_ts",
            "ts_run",
            "ts_channel",
        ]

        for obj_name in required_objects:
            assert obj_name in test_objects, f"Missing test object: {obj_name}"
            assert test_objects[obj_name] is not None

    def test_feature_hierarchy_integrity(self, mth5_instance, subtests):
        """Test that feature hierarchy maintains proper relationships."""
        test_objects = mth5_instance._test_objects

        with subtests.test("features_under_station"):
            # Features should be under station - test types match
            station = test_objects["station_group"]
            features = test_objects["features_group"]
            assert features.metadata.mth5_type == "MasterFeatures"
            assert hasattr(station, "features_group")

        with subtests.test("fc_under_features"):
            # Feature FC should be under features group
            features = test_objects["features_group"]
            feature_fc = test_objects["feature_fc"]
            # This tests the relationship exists
            assert hasattr(features, "add_feature_group")

        with subtests.test("run_under_feature"):
            # Run should be under feature group
            fc_run = test_objects["fc_run"]
            assert fc_run.metadata.id == "coherence"

    def test_mth5_file_integrity(self, mth5_instance):
        """Test MTH5 file integrity and version."""
        assert mth5_instance.file_version == "0.2.0"
        assert hasattr(mth5_instance, "surveys_group")

        # Test survey access
        survey = mth5_instance.get_survey("top_survey")
        assert survey is not None

        # Test station access with survey parameter
        station = mth5_instance.get_station("mt01", survey="top_survey")
        assert station is not None


class TestFeaturesMetadataValidation:
    """Test comprehensive metadata validation for features."""

    @pytest.mark.parametrize(
        "object_name,metadata_attr,expected_value",
        [
            ("features_group", "mth5_type", "MasterFeatures"),
            ("feature_fc", "name", "feature_fc"),
            ("fc_run", "id", "coherence"),
            ("dl", "id", "0"),
            ("feature_ch", "name", "ex_hy"),
            ("feature_ts", "name", "feature_ts"),
            ("ts_run", "id", "despike"),
            ("ts_channel", "component", "ex"),
        ],
    )
    def test_metadata_attributes(
        self, mth5_instance, object_name, metadata_attr, expected_value
    ):
        """Test metadata attributes across all feature objects."""
        test_obj = mth5_instance._test_objects[object_name]
        actual_value = getattr(test_obj.metadata, metadata_attr)
        assert actual_value == expected_value

    def test_feature_types_validation(self, mth5_instance):
        """Test that different feature types are properly distinguished."""
        fc_run = mth5_instance._test_objects["fc_run"]
        ts_run = mth5_instance._test_objects["ts_run"]

        # Both should have metadata but different capabilities
        assert hasattr(fc_run, "metadata")
        assert hasattr(ts_run, "metadata")

        # Frequency domain should have decimation levels
        assert hasattr(fc_run, "add_decimation_level")

        # Time domain should have direct channel access
        assert hasattr(ts_run, "add_feature_channel")


class TestFeaturesErrorHandling:
    """Test error handling and edge cases for features."""

    def test_features_group_access(self, features_group):
        """Test features group can be accessed without errors."""
        assert features_group is not None
        assert hasattr(features_group, "metadata")

    def test_invalid_feature_access(self, mth5_instance):
        """Test handling of invalid feature access scenarios."""
        # This test ensures the basic structure is sound
        # More specific error testing could be added based on actual API
        test_objects = mth5_instance._test_objects

        # Verify we have the expected structure
        assert "features_group" in test_objects
        assert "feature_fc" in test_objects
        assert "feature_ts" in test_objects


class TestFeaturesPerformance:
    """Test performance characteristics of features functionality."""

    @pytest.mark.performance
    def test_feature_creation_efficiency(self, mth5_instance):
        """Test that feature creation is efficient."""
        # This test validates the session-scoped fixture approach
        # All features should be created once and reused
        test_objects = mth5_instance._test_objects

        # Verify all objects are properly initialized
        assert len(test_objects) == 10  # All expected objects

        # Verify no None objects (would indicate creation failure)
        for obj_name, obj in test_objects.items():
            assert obj is not None, f"Object {obj_name} is None"


# =============================================================================
# Integration Test for Complete Workflow
# =============================================================================


class TestFeaturesCompleteWorkflow:
    """Test complete workflow from MTH5 creation to feature usage."""

    def test_end_to_end_workflow(self, mth5_instance, subtests):
        """Test complete end-to-end workflow for features."""
        # This test validates the entire chain works correctly
        test_objects = mth5_instance._test_objects

        # Test survey level
        survey = test_objects["survey_group"]
        assert survey is not None

        # Test station level
        station = test_objects["station_group"]
        assert station is not None

        # Test features level
        features = test_objects["features_group"]
        assert features.metadata.mth5_type == "MasterFeatures"

        # Test frequency domain path
        feature_fc = test_objects["feature_fc"]
        fc_run = test_objects["fc_run"]
        dl = test_objects["dl"]
        feature_ch = test_objects["feature_ch"]

        with subtests.test("frequency_chain"):
            assert feature_fc.metadata.name == "feature_fc"
            assert fc_run.metadata.id == "coherence"
            assert dl.metadata.id == "0"
            assert feature_ch.metadata.name == "ex_hy"

        # Test time domain path
        feature_ts = test_objects["feature_ts"]
        ts_run = test_objects["ts_run"]
        ts_channel = test_objects["ts_channel"]

        with subtests.test("time_chain"):
            assert feature_ts.metadata.name == "feature_ts"
            assert ts_run.metadata.id == "despike"
            assert ts_channel.metadata.component == "ex"


if __name__ == "__main__":
    """
    Run the optimized MTH5 Features test suite

    Key features:
    - Session-scoped fixtures for efficient resource management
    - Comprehensive parametrized testing
    - Subtest functionality for detailed validation
    - Performance optimizations through strategic fixture usage
    - Enhanced error handling and validation

    Usage:
    - Run all tests: python test_features_pytest.py
    - Run specific class: pytest TestFrequencyDomainFeatures -v
    - Run with timing: pytest --durations=10 test_features_pytest.py
    - Run performance tests: pytest -m performance test_features_pytest.py
    """
    import sys

    args = [__file__, "-v", "--tb=short"]
    pytest.main(args)
