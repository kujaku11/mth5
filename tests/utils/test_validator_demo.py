"""
Quick test of the MTH5 validator
"""

import tempfile
from pathlib import Path

import numpy as np

from mth5.mth5 import MTH5
from mth5.utils.mth5_validator import validate_mth5_file


# Create a temporary test file
temp_file = Path(tempfile.gettempdir()) / "test_validator_demo.mth5"

print("=" * 80)
print("MTH5 Validator Demo")
print("=" * 80)

# Create a new MTH5 file
print(f"\n1. Creating test MTH5 file: {temp_file}")
with MTH5(file_version="0.2.0") as m:
    m.open_mth5(temp_file, "w")

    # Add survey
    survey = m.add_survey("demo_survey")
    print("   ✓ Added survey")

    # Add station
    station = m.add_station("DEMO001", survey="demo_survey")
    print("   ✓ Added station")

    # Add run
    run = station.add_run("DEMO001a")
    print("   ✓ Added run")

    # Add channel with data
    data = np.random.random(1000)
    run.add_channel("Ex", "electric", data)
    print("   ✓ Added channel with data")

# Validate the file
print("\n2. Validating the file...")
print("-" * 80)

results = validate_mth5_file(temp_file, verbose=True)

# Print results
results.print_report(include_info=True)

# Test CLI equivalent
print("\n3. Command-line equivalent:")
print("-" * 80)
print(f"   mth5-cli validate {temp_file}")
print(f"   mth5-cli validate {temp_file} --verbose")
print(f"   mth5-cli validate {temp_file} --json")

# JSON output example
print("\n4. JSON output example:")
print("-" * 80)
json_output = results.to_json()
print(json_output[:500] + "..." if len(json_output) > 500 else json_output)

# Cleanup
if temp_file.exists():
    temp_file.unlink()
    print(f"\n✓ Cleaned up test file")

print("\n" + "=" * 80)
print("Demo completed successfully!")
print("=" * 80)
