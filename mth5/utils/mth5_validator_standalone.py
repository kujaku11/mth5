#!/usr/bin/env python
"""
Standalone MTH5 File Validator
===============================

Lightweight validator for MTH5 files that doesn't require the full mth5 package.
Only depends on h5py and standard library.

This standalone version is ideal for:
- Building small executables (~20-30 MB vs 150+ MB)
- Validation without installing full mth5 environment
- Quick file checks in production environments

Usage:
    python mth5_validator_standalone.py validate file.mth5
    python mth5_validator_standalone.py validate file.mth5 --verbose
    python mth5_validator_standalone.py validate file.mth5 --json

Author: MTH5 Development Team
Date: February 9, 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


try:
    import h5py
except ImportError:
    print("ERROR: h5py is required. Install with: pip install h5py")
    sys.exit(1)


# Constants (copied from mth5 to avoid importing mth5)
ACCEPTABLE_FILE_VERSIONS = ["0.1.0", "0.2.0"]
ACCEPTABLE_FILE_TYPES = ["MTH5"]
ACCEPTABLE_DATA_LEVELS = [0, 1, 2, 3]


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
    Standalone MTH5 file validator.

    Validates MTH5 file structure without requiring full mth5 package.
    Only depends on h5py and standard library.
    """

    def __init__(
        self,
        file_path: str | Path,
        verbose: bool = False,
        check_data: bool = False,
    ):
        self.file_path = Path(file_path)
        self.verbose = verbose
        self.check_data = check_data
        self.results = ValidationResults(file_path=self.file_path)
        self.h5_file = None

    def validate(self) -> ValidationResults:
        """Run full validation suite."""
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

                # Check data if requested
                if self.check_data:
                    self._validate_data()

        except Exception as e:
            self.results.add_error("Validation", f"Unexpected error: {str(e)}")

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
            except Exception:
                pass

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
        # Check root Survey group
        if "Survey" not in self.h5_file:
            self.results.add_error("Structure", "Missing root 'Survey' group")
            return

        self.results.checked_items["root_group"] = True
        survey_group = self.h5_file["Survey"]

        # Check required subgroups
        required_subgroups = ["Stations", "Reports", "Filters", "Standards"]
        for subgroup in required_subgroups:
            path = f"Survey/{subgroup}"
            if subgroup not in survey_group:
                self.results.add_error("Structure", f"Missing required group '{path}'")
            else:
                self.results.checked_items[f"group_{subgroup}"] = True
                if self.verbose:
                    self.results.add_info("Structure", f"Found group: {path}")

        # Check for summary datasets
        self._check_summary_datasets("/Survey")

        # Check stations structure
        if "Stations" in survey_group:
            self._validate_stations_structure("/Survey/Stations")

    def _validate_v02_structure(self) -> None:
        """Validate MTH5 v0.2.0 structure."""
        # Check root Experiment group
        if "Experiment" not in self.h5_file:
            self.results.add_error("Structure", "Missing root 'Experiment' group")
            return

        self.results.checked_items["root_group"] = True
        experiment_group = self.h5_file["Experiment"]

        # Check required subgroups
        required_subgroups = ["Surveys", "Reports", "Standards"]
        for subgroup in required_subgroups:
            path = f"Experiment/{subgroup}"
            if subgroup not in experiment_group:
                self.results.add_error("Structure", f"Missing required group '{path}'")
            else:
                self.results.checked_items[f"group_{subgroup}"] = True
                if self.verbose:
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
                    if self.verbose:
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

        if station_count > 0:
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

        if run_count > 0 and self.verbose:
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

    def _validate_data(self) -> None:
        """Validate that channels contain data."""
        try:
            # Find all channel datasets
            channel_count = 0
            empty_count = 0

            def check_channels(name, obj):
                nonlocal channel_count, empty_count
                if isinstance(obj, h5py.Dataset):
                    # Check if this looks like a channel dataset (1D, non-zero length)
                    if obj.ndim == 1 and len(obj) > 0:
                        channel_count += 1

            self.h5_file.visititems(check_channels)

            if channel_count > 0:
                self.results.add_info(
                    "Data", f"Found {channel_count} channel dataset(s)"
                )
                self.results.checked_items["data_check"] = True

        except Exception as e:
            self.results.add_warning("Data", f"Error checking data: {str(e)}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MTH5 File Validator - Lightweight structure validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate data.mth5
  %(prog)s validate data.mth5 --verbose
  %(prog)s validate data.mth5 --check-data
  %(prog)s validate data.mth5 --json > results.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate MTH5 file structure"
    )
    validate_parser.add_argument("file", type=str, help="Path to MTH5 file")
    validate_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    validate_parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check that channels contain data (may be slow)",
    )
    validate_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )

    # Version command
    parser.add_argument("--version", action="version", version="MTH5 Validator 1.0.0")

    args = parser.parse_args()

    if args.command == "validate":
        # Validate file
        validator = MTH5Validator(
            args.file, verbose=args.verbose, check_data=args.check_data
        )
        results = validator.validate()

        if args.json:
            # Output JSON
            print(results.to_json())
        else:
            # Output human-readable report
            results.print_report(include_info=args.verbose)

        # Exit with appropriate code
        sys.exit(0 if results.is_valid else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
