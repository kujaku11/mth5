# MTH5 Validator Quick Start

## Installation

The validator is included with mth5 v0.6.1+:

```bash
pip install mth5
```

## Quick Validation

### Command Line

```bash
# Basic validation
mth5-cli validate myfile.mth5

# Verbose output (shows all checks)
mth5-cli validate myfile.mth5 --verbose

# Check data integrity (slower)
mth5-cli validate myfile.mth5 --check-data

# JSON output (for scripts/CI)
mth5-cli validate myfile.mth5 --json
```

### Python

```python
from mth5.utils.mth5_validator import validate_mth5_file

# Validate a file
results = validate_mth5_file('myfile.mth5')

# Check results
if results.is_valid:
    print("✓ File is valid!")
else:
    print(f"✗ Found {results.error_count} errors")
    results.print_report()
```

## What Gets Validated?

✓ File format (HDF5 structure)  
✓ MTH5 version (0.1.0 or 0.2.0)  
✓ Required groups (Survey/Experiment, Stations, etc.)  
✓ Group hierarchy  
✓ Metadata attributes  
✓ Summary tables  
✓ Optional: Channel data integrity  

## Common Use Cases

### Pre-Processing Check
```python
from mth5.utils.mth5_validator import validate_mth5_file

def safe_process(filepath):
    results = validate_mth5_file(filepath)
    if not results.is_valid:
        raise ValueError(f"Invalid file: {filepath}")
    # Continue processing...
```

### Batch Validation
```bash
# Validate all files in directory
for file in data/*.mth5; do
    mth5-cli validate "$file" || echo "FAILED: $file"
done
```

### CI/CD Integration
```bash
# In your CI pipeline
mth5-cli validate data.mth5 --json > validation_report.json
mth5-cli validate data.mth5  # Exit code 0=valid, 1=invalid
```

## Output Examples

### Valid File
```
================================================================================
MTH5 Validation Report: data.mth5
================================================================================
✓ VALID - File passed all validation checks

Summary:
  Errors:   0
  Warnings: 0
  Info:     8
```

### Invalid File
```
================================================================================
MTH5 Validation Report: data.mth5
================================================================================
✗ INVALID - File has 2 error(s)

Summary:
  Errors:   2
  Warnings: 1
  Info:     5

Details:
--------------------------------------------------------------------------------
  [ERROR] File Format: Missing 'file.version' attribute in root
  [ERROR] Structure: Missing required group 'Experiment'
  [WARNING] Structure: Missing 'channel_summary' dataset
```

## JSON Output

```bash
mth5-cli validate data.mth5 --json
```

```json
{
  "file_path": "data.mth5",
  "is_valid": true,
  "error_count": 0,
  "warning_count": 0,
  "info_count": 8,
  "messages": [
    {
      "level": "INFO",
      "category": "File Format",
      "message": "File type: MTH5",
      "path": null
    }
  ]
}
```

## Advanced Usage

### Full Validation (with data checks)
```python
from mth5.utils.mth5_validator import MTH5Validator

validator = MTH5Validator(
    'data.mth5',
    verbose=True,
    validate_metadata=True,
    check_data=True  # Checks channel data
)

results = validator.validate()
results.print_report(include_info=True)
```

### Structure Only (fast)
```python
validator = MTH5Validator(
    'data.mth5',
    validate_metadata=False,  # Skip metadata checks
    check_data=False          # Skip data checks
)
results = validator.validate()
```

## Help

```bash
# CLI help
mth5-cli --help
mth5-cli validate --help

# Python help
python -c "from mth5.utils.mth5_validator import MTH5Validator; help(MTH5Validator)"
```

## Documentation

- **Full Documentation**: [docs/VALIDATOR_README.md](docs/VALIDATOR_README.md)
- **Examples**: [examples/validator_examples.py](examples/validator_examples.py)
- **Implementation**: [VALIDATOR_IMPLEMENTATION.md](VALIDATOR_IMPLEMENTATION.md)

## Troubleshooting

### Import Error
```python
# Correct import (avoid circular import)
from mth5.utils.mth5_validator import MTH5Validator, validate_mth5_file
```

### File Access Error
Ensure file is not open elsewhere:
```python
import h5py
h5py.File.close_open_files()
```

### Validation Too Strict
Skip optional checks:
```python
validator = MTH5Validator('data.mth5', validate_metadata=False)
```

## Support

- **Issues**: https://github.com/kujaku11/mth5/issues
- **Docs**: https://mth5.readthedocs.io/
- **Examples**: See [examples/validator_examples.py](examples/validator_examples.py)
