# SWMR Basic Examples

This guide provides basic examples for testing and using SWMR (Single Writer Multiple Reader) functionality in MTH5.

## Overview

These examples demonstrate:
1. File creation without SWMR
2. SWMR writer mode activation
3. Adding data in SWMR mode
4. Flushing data to readers
5. Error handling for invalid modes

## Example 1: File Creation Error Handling

SWMR mode requires an existing file - you cannot create a new file directly in SWMR mode.

```python
import tempfile
from pathlib import Path
from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error

def test_swmr_file_creation_error():
    """Test that creating new file with SWMR raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_swmr_new.mth5"
        
        mth5 = MTH5(file_version="0.2.0")
        
        try:
            # This should raise an error - can't create new file with SWMR
            mth5.open_mth5(
                str(test_file), 
                mode="w", 
                single_writer_multiple_reader=True
            )
            print("ERROR: Should have raised MTH5Error")
        except MTH5Error as e:
            print(f"✅ Correctly raised error: {e}")
        finally:
            mth5.close_mth5()
```

**Expected Output:**
```
✅ Correctly raised error: Cannot use SWMR mode on non-existent file...
```

## Example 2: SWMR Writer Mode

The correct workflow for SWMR writer mode:

```python
import tempfile
from pathlib import Path
import numpy as np
from mth5.mth5 import MTH5

def test_swmr_writer_mode():
    """Test SWMR writer mode activation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_swmr_writer.mth5"
        
        # Step 1: Create file without SWMR
        print("Step 1: Creating initial file...")
        mth5 = MTH5(file_version="0.2.0")
        mth5.open_mth5(str(test_file), mode="w")
        survey = mth5.add_survey("test_survey")
        station = mth5.add_station("STA001", survey="test_survey")
        mth5.close_mth5()
        print("✓ File created")
        
        # Step 2: Reopen in SWMR writer mode
        print("Step 2: Reopening in SWMR writer mode...")
        mth5 = MTH5(file_version="0.2.0")
        mth5.open_mth5(
            str(test_file), 
            mode="a", 
            single_writer_multiple_reader=True
        )
        print("✓ SWMR writer mode activated")
        
        # Step 3: Verify SWMR status
        if mth5.is_swmr_mode():
            print("✓ SWMR mode verified active")
        
        # Step 4: Add data in SWMR mode
        print("Step 3: Adding data in SWMR mode...")
        station = mth5.get_station("STA001", survey="test_survey")
        run = station.add_run("run_001")
        
        # Add channel data
        data = np.random.randn(1000)
        ch = run.add_channel("Ex", "electric", data, channel_dtype="float32")
        print(f"✓ Added channel with {len(data)} samples")
        
        # Step 5: Flush data (critical for readers to see updates)
        print("Step 4: Flushing data...")
        mth5.flush()
        print("✓ Data flushed")
        
        mth5.close_mth5()
        print("✅ SWMR writer mode test complete")
```

**Expected Output:**
```
Step 1: Creating initial file...
✓ File created
Step 2: Reopening in SWMR writer mode...
✓ SWMR writer mode activated
✓ SWMR mode verified active
Step 3: Adding data in SWMR mode...
✓ Added channel with 1000 samples
Step 4: Flushing data...
✓ Data flushed
✅ SWMR writer mode test complete
```

## Example 3: SWMR Reader Mode

Readers can access the file while a writer is active:

```python
import tempfile
from pathlib import Path
import numpy as np
from mth5.mth5 import MTH5

def test_swmr_reader_mode():
    """Test SWMR reader mode"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_swmr_reader.mth5"
        
        # Create file with data
        print("Step 1: Creating file with data...")
        mth5_w = MTH5(file_version="0.2.0")
        mth5_w.open_mth5(str(test_file), mode="w")
        survey = mth5_w.add_survey("test_survey")
        station = mth5_w.add_station("STA001", survey="test_survey")
        run = station.add_run("run_001")
        data = np.random.randn(500)
        run.add_channel("Hy", "magnetic", data)
        mth5_w.close_mth5()
        print("✓ File created with 500 samples")
        
        # Open in SWMR reader mode
        print("Step 2: Opening in SWMR reader mode...")
        mth5_r = MTH5()
        mth5_r.open_mth5(
            str(test_file), 
            mode="r", 
            single_writer_multiple_reader=True
        )
        print("✓ SWMR reader mode activated")
        
        # Verify SWMR status
        if mth5_r.is_swmr_mode():
            print("✓ SWMR mode verified active")
        
        # Read data
        print("Step 3: Reading data...")
        channel_df = mth5_r.channel_summary.to_dataframe()
        print(f"✓ Found {len(channel_df)} channels")
        
        if len(channel_df) > 0:
            print(f"  - Channel: {channel_df.iloc[0].component}")
            print(f"  - Samples: {channel_df.iloc[0].n_samples}")
        
        mth5_r.close_mth5()
        print("✅ SWMR reader mode test complete")
```

**Expected Output:**
```
Step 1: Creating file with data...
✓ File created with 500 samples
Step 2: Opening in SWMR reader mode...
✓ SWMR reader mode activated
✓ SWMR mode verified active
Step 3: Reading data...
✓ Found 1 channels
  - Channel: Hy
  - Samples: 500
✅ SWMR reader mode test complete
```

## Example 4: Invalid Mode Handling

SWMR mode is not compatible with file creation modes:

```python
import tempfile
from pathlib import Path
from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error

def test_swmr_invalid_modes():
    """Test that invalid modes are rejected"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_swmr_modes.mth5"
        
        # Create file first
        mth5 = MTH5(file_version="0.2.0")
        mth5.open_mth5(str(test_file), mode="w")
        mth5.close_mth5()
        
        # Test mode='w' with SWMR (should fail)
        print("Testing mode='w' with SWMR...")
        mth5 = MTH5()
        try:
            mth5.open_mth5(
                str(test_file), 
                mode="w", 
                single_writer_multiple_reader=True
            )
            print("❌ ERROR: Should have raised error")
        except MTH5Error:
            print("✅ Correctly rejected mode='w'")
        
        # Test mode='x' with SWMR (should fail)
        print("Testing mode='x' with SWMR...")
        mth5 = MTH5()
        try:
            mth5.open_mth5(
                str(test_file), 
                mode="x", 
                single_writer_multiple_reader=True
            )
            print("❌ ERROR: Should have raised error")
        except MTH5Error:
            print("✅ Correctly rejected mode='x'")
```

**Expected Output:**
```
Testing mode='w' with SWMR...
✅ Correctly rejected mode='w'
Testing mode='x' with SWMR...
✅ Correctly rejected mode='x'
```

## Example 5: Non-existent File Handling

SWMR requires the file to exist:

```python
import tempfile
from pathlib import Path
from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error

def test_swmr_nonexistent_file():
    """Test SWMR with non-existent file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "nonexistent.mth5"
        
        mth5 = MTH5()
        try:
            mth5.open_mth5(
                str(test_file), 
                mode="a", 
                single_writer_multiple_reader=True
            )
            print("❌ ERROR: Should have raised error")
        except MTH5Error as e:
            print(f"✅ Correctly raised error: {e}")
```

**Expected Output:**
```
✅ Correctly raised error: Cannot use SWMR mode on non-existent file...
```

## Complete Test Script

A complete test script combining all examples is available at:
[examples/test_swmr_basic.py](../../examples/test_swmr_basic.py)

Run it with:
```bash
python examples/test_swmr_basic.py
```

## Key Takeaways

1. **Create file first**: Always create the file in normal mode before enabling SWMR
2. **Use mode='a' or 'r+'**: Writers must use append or read-write mode
3. **Use mode='r'**: Readers must use read-only mode
4. **Flush regularly**: Writers should call `mth5.flush()` to make data visible to readers
5. **Check status**: Use `mth5.is_swmr_mode()` to verify SWMR is active
6. **Handle errors**: Always use try/except blocks for robust code

## Next Steps

- See [SWMR Concurrent Examples](SWMR_EXAMPLES_CONCURRENT.md) for multi-process examples
- Read the [SWMR User Guide](SWMR_GUIDE.md) for detailed documentation
- Check the [SWMR Migration Guide](SWMR_MIGRATION.md) for converting existing files
