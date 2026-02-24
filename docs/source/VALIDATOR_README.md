# MTH5 File Validator

A comprehensive validation tool for MTH5 (Magnetotelluric HDF5) files that checks file format, structure, and metadata compliance.

## Features

- **File Format Validation**: Verify HDF5 file attributes (type, version, data level)
- **Structure Validation**: Check group hierarchy based on file version (v0.1.0 or v0.2.0)
- **Metadata Validation**: Validate metadata using mt_metadata schemas
- **Data Validation**: Optional check for channel data integrity
- **Multiple Interfaces**: Use programmatically or via command-line
- **Flexible Output**: Human-readable reports or JSON for integration

## Installation

The validator is included with mth5. After installing mth5, the validator is automatically available:

```bash
pip install mth5
```

For development installation:

```bash
cd mth5
pip install -e .
```

## Quick Start

### Command-Line Usage

```bash
# Basic validation
mth5-cli validate myfile.mth5

# Verbose output
mth5-cli validate myfile.mth5 --verbose

# Check data integrity (slower)
mth5-cli validate myfile.mth5 --check-data

# JSON output
mth5-cli validate myfile.mth5 --json
```

### Python API Usage

```python
from mth5.utils.mth5_validator import validate_mth5_file

# Quick validation
results = validate_mth5_file('myfile.mth5')

if results.is_valid:
    print("File is valid!")
else:
    results.print_report()
```

## Validation Checks

### File Format Checks

- **file.type**: Must be "MTH5"
- **file.version**: Must be "0.1.0" or "0.2.0"
- **data_level**: Must be 0, 1, or 2

### Structure Checks (v0.1.0)

```
/Survey
  ├── Stations/
  ├── Reports/
  ├── Filters/
  ├── Standards/
  ├── channel_summary (dataset)
  └── tf_summary (dataset)
```

### Structure Checks (v0.2.0)

```
/Experiment
  ├── Surveys/
  │   └── {survey_id}/
  │       ├── Stations/
  │       ├── Reports/
  │       ├── Filters/
  │       └── Standards/
  ├── Reports/
  ├── Standards/
  ├── channel_summary (dataset)
  └── tf_summary (dataset)
```

### Station/Run Structure

Each station should contain:
- One or more run groups
- Each run should contain channel datasets

### Metadata Checks

- Validates metadata attributes exist
- Checks for required mth5_type attributes
- Uses mt_metadata schemas for validation

### Data Checks (Optional)

- Verifies channels contain data
- Detects empty or all-zero channels
- Samples data without loading full arrays

## Command-Line Interface

### mth5-cli validate

Validate an MTH5 file.

**Usage:**
```bash
mth5-cli validate FILE [OPTIONS]
```

**Arguments:**
- `FILE`: Path to MTH5 file to validate

**Options:**
- `-v, --verbose`: Enable verbose output with detailed information
- `--skip-metadata`: Skip metadata validation (structure only)
- `--check-data`: Check that channels contain data (slower)
- `--json`: Output results as JSON

**Exit Codes:**
- `0`: File is valid
- `1`: File has errors or validation failed

**Examples:**

```bash
# Basic validation
mth5-cli validate data.mth5

# Detailed validation report
mth5-cli validate data.mth5 --verbose

# Full validation including data
mth5-cli validate data.mth5 --check-data --verbose

# JSON output for scripting
mth5-cli validate data.mth5 --json > report.json

# Batch validation
for file in data/*.mth5; do
    mth5-cli validate "$file" || echo "Failed: $file"
done
```

## Python API

### validate_mth5_file()

Convenience function for quick validation.

```python
from mth5.utils import validate_mth5_file

results = validate_mth5_file(
    file_path='data.mth5',
    verbose=False,
    validate_metadata=True,
    check_data=False
)

print(f"Valid: {results.is_valid}")
print(f"Errors: {results.error_count}")
```

### MTH5Validator Class

Full-featured validator class for advanced usage.

```python
from mth5.utils.mth5_validator import MTH5Validator

# Create validator
validator = MTH5Validator(
    file_path='data.mth5',
    verbose=True,
    validate_metadata=True,
    check_data=False
)

# Run validation
results = validator.validate()

# Access results
print(f"Valid: {results.is_valid}")
print(f"Errors: {results.error_count}")
print(f"Warnings: {results.warning_count}")

# Print report
results.print_report(include_info=True)

# Get JSON
json_str = results.to_json()

# Get dictionary
data_dict = results.to_dict()
```

### ValidationResults Object

Results object returned by validation.

**Properties:**
- `is_valid` (bool): True if no errors
- `error_count` (int): Number of errors
- `warning_count` (int): Number of warnings
- `info_count` (int): Number of info messages
- `messages` (list): All validation messages
- `checked_items` (dict): Dictionary of validation checks performed

**Methods:**
- `print_report(include_info=False)`: Print formatted report
- `to_dict()`: Convert to dictionary
- `to_json(**kwargs)`: Convert to JSON string
- `add_error(category, message, path=None, **details)`: Add error message
- `add_warning(category, message, path=None, **details)`: Add warning message
- `add_info(category, message, path=None, **details)`: Add info message

## Use Cases

### Pre-Processing Validation

Validate files before processing:

```python
from mth5.utils.mth5_validator import validate_mth5_file

def process_mth5(filepath):
    # Validate first
    results = validate_mth5_file(filepath)
    
    if not results.is_valid:
        print(f"Cannot process {filepath}:")
        results.print_report()
        return False
    
    # Process file
    # ...
    return True
```

### Archive Quality Control

Check files meet archive standards:

```python
from pathlib import Path
from mth5.utils.mth5_validator import MTH5Validator

def qa_check_archive(archive_dir):
    """Quality check all files in archive."""
    failed = []
    
    for mth5_file in Path(archive_dir).glob("**/*.mth5"):
        validator = MTH5Validator(
            mth5_file,
            validate_metadata=True,
            check_data=True
        )
        results = validator.validate()
        
        if not results.is_valid:
            failed.append(mth5_file)
    
    return failed
```

### Automated Testing

Use in test suites:

```python
import pytest
from mth5.utils.mth5_validator import validate_mth5_file

def test_mth5_file_valid(mth5_file):
    """Test that generated MTH5 file is valid."""
    results = validate_mth5_file(mth5_file)
    
    assert results.is_valid, f"Validation failed:\n{results.messages}"
    assert results.error_count == 0
```

### CI/CD Integration

GitHub Actions example:

```yaml
name: Validate MTH5 Files

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install mth5
      - name: Validate MTH5 files
        run: |
          for file in data/*.mth5; do
            mth5-cli validate "$file" --json > "${file%.mth5}_validation.json"
            mth5-cli validate "$file" || exit 1
          done
```

## Validation Levels

### ERROR
Critical issues that indicate an invalid file:
- Missing required file attributes
- Invalid file version or type
- Missing required groups
- Corrupted file structure

### WARNING
Issues that should be reviewed but don't prevent usage:
- Missing optional metadata
- Empty summary tables
- Runs with no channels
- Missing recommended attributes

### INFO
Informational messages:
- File version and type
- Number of surveys/stations/runs
- Summary of validation checks
- Data statistics

## Performance

### Speed Considerations

- **Basic validation** (format + structure): Very fast, <1 second
- **With metadata validation**: Fast, 1-5 seconds
- **With data checking**: Slower, depends on file size (samples data efficiently)

### Large Files

For large files (>1GB), consider:

```python
# Skip data checking for speed
validator = MTH5Validator(
    file_path='large_file.mth5',
    check_data=False  # Much faster
)
```

## Creating a Standalone Executable

For users without Python, create a standalone executable using PyInstaller:

```bash
# Install PyInstaller
pip install pyinstaller

# Create standalone executable
pyinstaller --onefile \
    --name mth5-validator \
    --add-data "mth5:mth5" \
    mth5/utils/cli.py

# Executable will be in dist/
./dist/mth5-validator validate data.mth5
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
pip install --upgrade mth5 h5py mt_metadata
```

### File Access Errors

Ensure file is not open in another program:
```python
# Close any open references
import h5py
h5py.File.close_open_files()
```

### Validation Too Strict

Skip certain checks:
```python
validator = MTH5Validator(
    file_path='data.mth5',
    validate_metadata=False  # Skip metadata validation
)
```

## Contributing

To add new validation checks:

1. Add check method to `MTH5Validator` class in `mth5_validator.py`
2. Call from `validate()` method
3. Add results to `ValidationResults`
4. Update tests
5. Document in this README

## License

MIT License - See LICENSE file in mth5 repository.

## Support

- **Documentation**: https://mth5.readthedocs.io/
- **Issues**: https://github.com/kujaku11/mth5/issues
- **Discussions**: https://github.com/kujaku11/mth5/discussions

## See Also

- [MTH5 Documentation](https://mth5.readthedocs.io/)
- [MT Metadata](https://github.com/kujaku11/mt_metadata)
- [HDF5 Documentation](https://www.hdfgroup.org/)
