"""
Example: Using the MTH5 File Validator
=======================================

This script demonstrates how to use the MTH5 validator both programmatically
and from the command line.

Author: MTH5 Development Team
Date: February 7, 2026
"""

from pathlib import Path

from mth5.mth5 import MTH5
from mth5.utils.mth5_validator import MTH5Validator, validate_mth5_file


def example_basic_validation():
    """Example 1: Basic validation"""
    print("=" * 80)
    print("Example 1: Basic Validation")
    print("=" * 80)

    # Validate a file
    mth5_file = "example_data.mth5"
    results = validate_mth5_file(mth5_file)

    # Check results
    if results.is_valid:
        print(f"✓ {mth5_file} is valid!")
    else:
        print(f"✗ {mth5_file} has errors")
        results.print_report()


def example_detailed_validation():
    """Example 2: Detailed validation with all checks"""
    print("\n" + "=" * 80)
    print("Example 2: Detailed Validation")
    print("=" * 80)

    # Create validator with all options
    validator = MTH5Validator(
        file_path="example_data.mth5",
        verbose=True,
        validate_metadata=True,
        check_data=True,  # This checks channel data (slower)
    )

    # Run validation
    results = validator.validate()

    # Print detailed report
    results.print_report(include_info=True)

    # Access individual components
    print(f"\nDetailed Statistics:")
    print(f"  Errors:   {results.error_count}")
    print(f"  Warnings: {results.warning_count}")
    print(f"  Info:     {results.info_count}")


def example_json_output():
    """Example 3: JSON output for integration"""
    print("\n" + "=" * 80)
    print("Example 3: JSON Output")
    print("=" * 80)

    # Validate and get JSON
    results = validate_mth5_file("example_data.mth5")
    json_output = results.to_json()

    print(json_output)


def example_batch_validation():
    """Example 4: Validate multiple files"""
    print("\n" + "=" * 80)
    print("Example 4: Batch Validation")
    print("=" * 80)

    # List of files to validate
    data_dir = Path("data")
    mth5_files = list(data_dir.glob("*.mth5"))

    results_summary = []

    for mth5_file in mth5_files:
        print(f"\nValidating: {mth5_file.name}")
        results = validate_mth5_file(mth5_file)

        results_summary.append(
            {
                "file": mth5_file.name,
                "valid": results.is_valid,
                "errors": results.error_count,
                "warnings": results.warning_count,
            }
        )

    # Print summary
    print("\n" + "-" * 80)
    print("Batch Validation Summary:")
    print("-" * 80)
    for summary in results_summary:
        status = "✓ PASS" if summary["valid"] else "✗ FAIL"
        print(
            f"{status} | {summary['file']:<30} | "
            f"Errors: {summary['errors']}, Warnings: {summary['warnings']}"
        )


def example_create_and_validate():
    """Example 5: Create a file and validate it"""
    print("\n" + "=" * 80)
    print("Example 5: Create and Validate")
    print("=" * 80)

    # Create a new MTH5 file
    test_file = Path("test_validation.mth5")

    # Create file
    with MTH5(file_version="0.2.0") as m:
        m.open_mth5(test_file, "w")

        # Add a survey
        survey = m.add_survey("test_survey")

        # Add a station
        station = m.add_station("TEST001", survey="test_survey")

    # Validate the newly created file
    print(f"\nValidating newly created file: {test_file}")
    results = validate_mth5_file(test_file)

    if results.is_valid:
        print("✓ File structure is valid!")
    else:
        print("✗ File has issues:")
        results.print_report()

    # Cleanup
    if test_file.exists():
        test_file.unlink()
        print(f"\nCleaned up: {test_file}")


def example_custom_validation():
    """Example 6: Custom validation logic"""
    print("\n" + "=" * 80)
    print("Example 6: Custom Validation Logic")
    print("=" * 80)

    # Run validation
    validator = MTH5Validator("example_data.mth5")
    results = validator.validate()

    # Custom checks based on results
    if results.is_valid:
        print("✓ All validation checks passed!")

        # Additional custom logic
        if results.warning_count > 0:
            print(f"\nNote: {results.warning_count} warning(s) found.")
            print("These don't prevent file usage but should be reviewed.")

    else:
        print("✗ Validation failed!")

        # Analyze errors
        error_categories = {}
        for msg in results.messages:
            if msg.level.value == "ERROR":
                category = msg.category
                error_categories[category] = error_categories.get(category, 0) + 1

        print("\nError breakdown by category:")
        for category, count in error_categories.items():
            print(f"  {category}: {count} error(s)")


if __name__ == "__main__":
    """
    Run examples.

    Note: These examples assume you have MTH5 files to validate.
    If running without actual files, some examples will fail.
    """

    print("\n" + "=" * 80)
    print("MTH5 Validator Examples")
    print("=" * 80)
    print("\nNote: Most examples require existing MTH5 files.")
    print("Example 5 creates its own test file.\n")

    # Run the example that creates its own file
    try:
        example_create_and_validate()
    except Exception as e:
        print(f"Example failed: {e}")

    # Uncomment to run other examples if you have MTH5 files:
    # example_basic_validation()
    # example_detailed_validation()
    # example_json_output()
    # example_batch_validation()
    # example_custom_validation()

    print("\n" + "=" * 80)
    print("Command-Line Usage Examples:")
    print("=" * 80)
    print(
        """
# Basic validation
mth5-cli validate data.mth5

# Verbose output with details
mth5-cli validate data.mth5 --verbose

# Check data integrity
mth5-cli validate data.mth5 --check-data

# Skip metadata validation (structure only)
mth5-cli validate data.mth5 --skip-metadata

# JSON output for scripting
mth5-cli validate data.mth5 --json > validation_report.json

# Batch validation with shell
for file in data/*.mth5; do
    echo "Validating: $file"
    mth5-cli validate "$file"
done
"""
    )
