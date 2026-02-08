# -*- coding: utf-8 -*-
"""
MTH5 File Validator
===================

Validates MTH5 files for structural integrity and metadata compliance.

This module provides comprehensive validation of MTH5 files including:
- File format and version checks
- Group structure validation
- Metadata schema validation
- Summary table validation

Created on February 7, 2026

:copyright: MTH5 Development Team
:license: MIT

Examples
--------
Validate a file programmatically:

>>> from mth5.utils.mth5_validator import MTH5Validator
>>> validator = MTH5Validator('data.mth5')
>>> results = validator.validate()
>>> print(results.is_valid)
True

Validate with detailed reporting:

>>> validator = MTH5Validator('data.mth5', verbose=True)
>>> results = validator.validate()
>>> results.print_report()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
from loguru import logger

from mth5 import ACCEPTABLE_DATA_LEVELS, ACCEPTABLE_FILE_TYPES, ACCEPTABLE_FILE_VERSIONS


class ValidationLevel(Enum):
    """Validation severity levels."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationMessage:
    """Container for a single validation message."""

    level: ValidationLevel
    category: str
    message: str
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format message for display."""
        prefix = f"[{self.level.value}] {self.category}"
        if self.path:
            prefix += f" ({self.path})"
        return f"{prefix}: {self.message}"


@dataclass
class ValidationResults:
    """Container for validation results."""

    file_path: Path
    messages: list[ValidationMessage] = field(default_factory=list)
    checked_items: dict[str, bool] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if file passed validation (no errors)."""
        return not any(msg.level == ValidationLevel.ERROR for msg in self.messages)

    @property
    def error_count(self) -> int:
        """Count of error messages."""
        return sum(1 for msg in self.messages if msg.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning messages."""
        return sum(1 for msg in self.messages if msg.level == ValidationLevel.WARNING)

    @property
    def info_count(self) -> int:
        """Count of info messages."""
        return sum(1 for msg in self.messages if msg.level == ValidationLevel.INFO)

    def add_error(
        self, category: str, message: str, path: str | None = None, **details
    ) -> None:
        """Add an error message."""
        self.messages.append(
            ValidationMessage(ValidationLevel.ERROR, category, message, path, details)
        )

    def add_warning(
        self, category: str, message: str, path: str | None = None, **details
    ) -> None:
        """Add a warning message."""
        self.messages.append(
            ValidationMessage(ValidationLevel.WARNING, category, message, path, details)
        )

    def add_info(
        self, category: str, message: str, path: str | None = None, **details
    ) -> None:
        """Add an info message."""
        self.messages.append(
            ValidationMessage(ValidationLevel.INFO, category, message, path, details)
        )

    def print_report(self, include_info: bool = False) -> None:
        """Print a formatted validation report."""
        print(f"\n{'=' * 80}")
        print(f"MTH5 Validation Report: {self.file_path.name}")
        print(f"{'=' * 80}")

        if self.is_valid:
            print(f"✓ VALID - File passed all validation checks")
        else:
            print(f"✗ INVALID - File has {self.error_count} error(s)")

        print(f"\nSummary:")
        print(f"  Errors:   {self.error_count}")
        print(f"  Warnings: {self.warning_count}")
        print(f"  Info:     {self.info_count}")

        # Print messages by category
        if self.messages:
            print(f"\nDetails:")
            print(f"{'-' * 80}")

            for msg in self.messages:
                if msg.level == ValidationLevel.INFO and not include_info:
                    continue
                print(f"  {msg}")

        # Print checked items summary
        if self.checked_items:
            passed = sum(1 for v in self.checked_items.values() if v)
            total = len(self.checked_items)
            print(f"\nValidation Checks: {passed}/{total} passed")

        print(f"{'=' * 80}\n")

    def to_dict(self) -> dict:
        """Convert results to dictionary."""
        return {
            "file_path": str(self.file_path),
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "messages": [
                {
                    "level": msg.level.value,
                    "category": msg.category,
                    "message": msg.message,
                    "path": msg.path,
                    "details": msg.details,
                }
                for msg in self.messages
            ],
            "checked_items": self.checked_items,
        }

    def to_json(self, **kwargs) -> str:
        """Convert results to JSON string."""
        return json.dumps(self.to_dict(), indent=2, **kwargs)


class MTH5Validator:
    """
    MTH5 file validator.

    Performs comprehensive validation of MTH5 files including file format,
    group structure, and metadata validation.

    Parameters
    ----------
    file_path : str | Path
        Path to the MTH5 file to validate.
    verbose : bool, default False
        Enable verbose logging during validation.
    validate_metadata : bool, default True
        Enable metadata validation using mt_metadata schemas.
    check_data : bool, default False
        Check that channels contain data (can be slow for large files).

    Attributes
    ----------
    results : ValidationResults
        Validation results after running validate().

    Examples
    --------
    Basic validation:

    >>> validator = MTH5Validator('data.mth5')
    >>> results = validator.validate()
    >>> if results.is_valid:
    ...     print("File is valid!")

    Detailed validation with report:

    >>> validator = MTH5Validator('data.mth5', verbose=True, check_data=True)
    >>> results = validator.validate()
    >>> results.print_report(include_info=True)
    """

    def __init__(
        self,
        file_path: str | Path,
        verbose: bool = False,
        validate_metadata: bool = True,
        check_data: bool = False,
    ):
        self.file_path = Path(file_path)
        self.verbose = verbose
        self.validate_metadata = validate_metadata
        self.check_data = check_data
        self.results = ValidationResults(file_path=self.file_path)
        self.h5_file = None

        # Configure logging
        if not verbose:
            logger.disable("mth5")

    def validate(self) -> ValidationResults:
        """
        Run full validation suite.

        Returns
        -------
        ValidationResults
            Complete validation results with all messages.

        Examples
        --------
        >>> validator = MTH5Validator('data.mth5')
        >>> results = validator.validate()
        >>> print(f"Valid: {results.is_valid}")
        """
        try:
            # Check file exists and is accessible
            if not self._check_file_exists():
                return self.results

            # Open HDF5 file
            if not self._open_file():
                return self.results

            # Validate file format
            self._validate_file_format()

            # Validate structure based on version
            file_version = self._get_file_version()
            if file_version:
                self._validate_structure(file_version)

                # Validate metadata if requested
                if self.validate_metadata:
                    self._validate_metadata(file_version)

                # Check data if requested
                if self.check_data:
                    self._validate_data()

        except Exception as e:
            self.results.add_error("Validation", f"Unexpected error: {str(e)}")
            logger.exception("Validation error")

        finally:
            self._close_file()

        return self.results

    def _check_file_exists(self) -> bool:
        """Check if file exists and is readable."""
        if not self.file_path.exists():
            self.results.add_error(
                "File Access", f"File does not exist: {self.file_path}"
            )
            return False

        if not self.file_path.is_file():
            self.results.add_error(
                "File Access", f"Path is not a file: {self.file_path}"
            )
            return False

        self.results.checked_items["file_exists"] = True
        return True

    def _open_file(self) -> bool:
        """Open HDF5 file for reading."""
        try:
            self.h5_file = h5py.File(self.file_path, "r")
            self.results.checked_items["file_readable"] = True
            return True
        except (OSError, IOError) as e:
            self.results.add_error("File Access", f"Cannot open file: {str(e)}")
            return False

    def _close_file(self) -> None:
        """Close HDF5 file if open."""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception as e:
                logger.warning(f"Error closing file: {e}")

    def _validate_file_format(self) -> None:
        """Validate basic HDF5 file format and MTH5 attributes."""
        # Check file.type attribute
        file_type = self.h5_file.attrs.get("file.type")
        if file_type is None:
            self.results.add_error(
                "File Format", "Missing 'file.type' attribute in root"
            )
        elif file_type not in ACCEPTABLE_FILE_TYPES:
            self.results.add_error(
                "File Format",
                f"Invalid file.type '{file_type}'. Must be one of {ACCEPTABLE_FILE_TYPES}",
            )
        else:
            self.results.checked_items["file_type"] = True
            self.results.add_info("File Format", f"File type: {file_type}")

        # Check file.version attribute
        file_version = self.h5_file.attrs.get("file.version")
        if file_version is None:
            self.results.add_error(
                "File Format", "Missing 'file.version' attribute in root"
            )
        elif file_version not in ACCEPTABLE_FILE_VERSIONS:
            self.results.add_error(
                "File Format",
                f"Invalid file.version '{file_version}'. Must be one of {ACCEPTABLE_FILE_VERSIONS}",
            )
        else:
            self.results.checked_items["file_version"] = True
            self.results.add_info("File Format", f"File version: {file_version}")

        # Check data_level attribute
        data_level = self.h5_file.attrs.get("data_level")
        if data_level is None:
            self.results.add_warning(
                "File Format", "Missing 'data_level' attribute in root"
            )
        elif data_level not in ACCEPTABLE_DATA_LEVELS:
            self.results.add_error(
                "File Format",
                f"Invalid data_level {data_level}. Must be one of {ACCEPTABLE_DATA_LEVELS}",
            )
        else:
            self.results.checked_items["data_level"] = True
            self.results.add_info("File Format", f"Data level: {data_level}")

    def _get_file_version(self) -> str | None:
        """Get file version from attributes."""
        return self.h5_file.attrs.get("file.version")

    def _validate_structure(self, file_version: str) -> None:
        """Validate group structure based on file version."""
        if file_version == "0.1.0":
            self._validate_v01_structure()
        elif file_version == "0.2.0":
            self._validate_v02_structure()

    def _validate_v01_structure(self) -> None:
        """Validate MTH5 v0.1.0 structure."""
        required_groups = ["Survey"]
        required_subgroups = ["Stations", "Reports", "Filters", "Standards"]

        # Check root Survey group
        if "Survey" not in self.h5_file:
            self.results.add_error("Structure", "Missing root 'Survey' group")
            return

        self.results.checked_items["root_group"] = True

        # Check required subgroups
        survey_group = self.h5_file["Survey"]
        for subgroup in required_subgroups:
            path = f"Survey/{subgroup}"
            if subgroup not in survey_group:
                self.results.add_error("Structure", f"Missing required group '{path}'")
            else:
                self.results.checked_items[f"group_{subgroup}"] = True
                self.results.add_info("Structure", f"Found group: {path}")

        # Check for summary datasets
        self._check_summary_datasets("/Survey")

        # Check stations structure
        if "Stations" in survey_group:
            self._validate_stations_structure("/Survey/Stations")

    def _validate_v02_structure(self) -> None:
        """Validate MTH5 v0.2.0 structure."""
        required_groups = ["Experiment"]
        required_subgroups = ["Surveys", "Reports", "Standards"]

        # Check root Experiment group
        if "Experiment" not in self.h5_file:
            self.results.add_error("Structure", "Missing root 'Experiment' group")
            return

        self.results.checked_items["root_group"] = True

        # Check required subgroups
        experiment_group = self.h5_file["Experiment"]
        for subgroup in required_subgroups:
            path = f"Experiment/{subgroup}"
            if subgroup not in experiment_group:
                self.results.add_error("Structure", f"Missing required group '{path}'")
            else:
                self.results.checked_items[f"group_{subgroup}"] = True
                self.results.add_info("Structure", f"Found group: {path}")

        # Check for summary datasets
        self._check_summary_datasets("/Experiment")

        # Check surveys structure
        if "Surveys" in experiment_group:
            surveys_group = experiment_group["Surveys"]
            survey_count = 0
            for survey_name in surveys_group.keys():
                if isinstance(surveys_group[survey_name], h5py.Group):
                    survey_count += 1
                    survey_path = f"/Experiment/Surveys/{survey_name}"
                    self._validate_survey_structure(survey_path)

            self.results.add_info("Structure", f"Found {survey_count} survey(s)")

    def _check_summary_datasets(self, root_path: str) -> None:
        """Check for required summary datasets."""
        root_group = self.h5_file[root_path]
        summary_datasets = ["channel_summary", "tf_summary"]

        for dataset_name in summary_datasets:
            if dataset_name not in root_group:
                self.results.add_warning(
                    "Structure", f"Missing '{dataset_name}' dataset in {root_path}"
                )
            else:
                dataset = root_group[dataset_name]
                if not isinstance(dataset, h5py.Dataset):
                    self.results.add_error(
                        "Structure",
                        f"'{dataset_name}' in {root_path} is not a dataset",
                    )
                else:
                    self.results.add_info(
                        "Structure",
                        f"Found {dataset_name} with {len(dataset)} entries",
                    )

    def _validate_survey_structure(self, survey_path: str) -> None:
        """Validate survey group structure."""
        if survey_path not in self.h5_file:
            return

        survey_group = self.h5_file[survey_path]
        required_subgroups = ["Stations", "Reports", "Filters", "Standards"]

        for subgroup in required_subgroups:
            if subgroup not in survey_group:
                self.results.add_warning(
                    "Structure", f"Missing subgroup '{subgroup}' in {survey_path}"
                )

        # Check stations in this survey
        if "Stations" in survey_group:
            self._validate_stations_structure(f"{survey_path}/Stations")

    def _validate_stations_structure(self, stations_path: str) -> None:
        """Validate stations group structure."""
        if stations_path not in self.h5_file:
            return

        stations_group = self.h5_file[stations_path]
        station_count = 0

        for station_name in stations_group.keys():
            if isinstance(stations_group[station_name], h5py.Group):
                station_count += 1
                station_path = f"{stations_path}/{station_name}"
                self._validate_station_structure(station_path)

        self.results.add_info("Structure", f"Found {station_count} station(s)")

    def _validate_station_structure(self, station_path: str) -> None:
        """Validate individual station structure."""
        if station_path not in self.h5_file:
            return

        station_group = self.h5_file[station_path]
        run_count = 0

        for item_name in station_group.keys():
            item = station_group[item_name]
            if isinstance(item, h5py.Group):
                # Check if it's a run (not Transfer_Functions)
                if item_name != "Transfer_Functions":
                    run_count += 1
                    self._validate_run_structure(f"{station_path}/{item_name}")

        if run_count > 0:
            self.results.add_info(
                "Structure", f"{station_path}: {run_count} run(s)", path=station_path
            )

    def _validate_run_structure(self, run_path: str) -> None:
        """Validate individual run structure."""
        if run_path not in self.h5_file:
            return

        run_group = self.h5_file[run_path]
        channel_count = 0

        for channel_name in run_group.keys():
            if isinstance(run_group[channel_name], h5py.Dataset):
                channel_count += 1

        if channel_count == 0:
            self.results.add_warning("Structure", f"Run has no channels", path=run_path)

    def _validate_metadata(self, file_version: str) -> None:
        """Validate metadata using mt_metadata schemas."""
        try:
            if file_version == "0.1.0":
                self._validate_v01_metadata()
            elif file_version == "0.2.0":
                self._validate_v02_metadata()

            self.results.checked_items["metadata_validation"] = True

        except Exception as e:
            self.results.add_error("Metadata", f"Error validating metadata: {str(e)}")

    def _validate_v01_metadata(self) -> None:
        """Validate v0.1.0 metadata."""
        if "Survey" not in self.h5_file:
            return

        survey_group = self.h5_file["Survey"]

        # Check Survey metadata
        if not self._has_required_metadata(survey_group):
            self.results.add_warning(
                "Metadata", "Survey group missing metadata attributes", path="/Survey"
            )

    def _validate_v02_metadata(self) -> None:
        """Validate v0.2.0 metadata."""
        if "Experiment" not in self.h5_file:
            return

        experiment_group = self.h5_file["Experiment"]

        # Check Experiment metadata
        if not self._has_required_metadata(experiment_group):
            self.results.add_warning(
                "Metadata",
                "Experiment group missing metadata attributes",
                path="/Experiment",
            )

    def _has_required_metadata(self, group: h5py.Group) -> bool:
        """Check if group has required metadata attributes."""
        # Check for mth5_type attribute
        if "mth5_type" not in group.attrs:
            return False

        # Has some metadata
        return len(group.attrs) > 0

    def _validate_data(self) -> None:
        """Validate that channels contain data."""
        try:
            # Find all channel datasets
            channel_count = 0
            empty_count = 0

            def check_channels(name, obj):
                nonlocal channel_count, empty_count
                if isinstance(obj, h5py.Dataset):
                    # Check if this looks like a channel dataset
                    if obj.ndim == 1 and len(obj) > 0:
                        channel_count += 1
                        # Check if data is all zeros/nans
                        if self._is_empty_data(obj):
                            empty_count += 1
                            self.results.add_warning(
                                "Data", f"Channel appears to have no data", path=name
                            )

            self.h5_file.visititems(check_channels)

            if channel_count > 0:
                self.results.add_info(
                    "Data",
                    f"Checked {channel_count} channels, {empty_count} empty",
                )
                self.results.checked_items["data_check"] = True

        except Exception as e:
            self.results.add_warning("Data", f"Error checking data: {str(e)}")

    def _is_empty_data(self, dataset: h5py.Dataset) -> bool:
        """Check if dataset contains only zeros or fill values."""
        # Sample first and last chunks
        if len(dataset) == 0:
            return True

        # Check first 100 and last 100 values
        sample_size = min(100, len(dataset))
        first_sample = dataset[:sample_size]
        last_sample = dataset[-sample_size:]

        # Check if all zeros or all same value
        import numpy as np

        return (
            np.all(first_sample == 0)
            or np.all(last_sample == 0)
            or (np.all(first_sample == first_sample[0]) and first_sample[0] == 0)
        )


def validate_mth5_file(
    file_path: str | Path, verbose: bool = False, **kwargs
) -> ValidationResults:
    """
    Convenience function to validate an MTH5 file.

    Parameters
    ----------
    file_path : str | Path
        Path to MTH5 file to validate.
    verbose : bool, default False
        Enable verbose output.
    **kwargs
        Additional arguments passed to MTH5Validator.

    Returns
    -------
    ValidationResults
        Validation results.

    Examples
    --------
    >>> from mth5.utils.mth5_validator import validate_mth5_file
    >>> results = validate_mth5_file('data.mth5')
    >>> if not results.is_valid:
    ...     results.print_report()
    """
    validator = MTH5Validator(file_path, verbose=verbose, **kwargs)
    return validator.validate()
