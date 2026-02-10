# -*- coding: utf-8 -*-
"""
MTH5 CLI Tools
==============

Command-line interface tools for MTH5 file operations.

Created on February 7, 2026

:copyright: MTH5 Development Team
:license: MIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mth5.utils.mth5_validator import MTH5Validator


def validate_command(args: argparse.Namespace) -> int:
    """
    Execute validation command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    # Create validator
    validator = MTH5Validator(
        file_path=args.file,
        verbose=args.verbose,
        validate_metadata=not args.skip_metadata,
        check_data=args.check_data,
    )

    # Run validation
    print(f"Validating MTH5 file: {args.file}")
    results = validator.validate()

    # Output results
    if args.json:
        print(results.to_json())
    else:
        results.print_report(include_info=args.verbose)

    # Return exit code
    return 0 if results.is_valid else 1


def main() -> int:
    """
    Main entry point for mth5-cli tool.

    Returns
    -------
    int
        Exit code.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        prog="mth5-cli",
        description="MTH5 Command-Line Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate an MTH5 file
  mth5-cli validate data.mth5

  # Validate with verbose output
  mth5-cli validate data.mth5 --verbose

  # Validate and check data
  mth5-cli validate data.mth5 --check-data

  # Output results as JSON
  mth5-cli validate data.mth5 --json

For more information, visit: https://mth5.readthedocs.io/
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (mth5 package)",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        help="Command to execute",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an MTH5 file",
        description="Validate MTH5 file format, structure, and metadata",
    )

    validate_parser.add_argument("file", type=Path, help="Path to MTH5 file")

    validate_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed information",
    )

    validate_parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip metadata validation",
    )

    validate_parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check that channels contain data (slower)",
    )

    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    validate_parser.set_defaults(func=validate_command)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
