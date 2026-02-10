# MTH5 SWMR Migration Guide

## Overview

This guide helps you migrate existing MTH5 files to support SWMR (Single Writer Multiple Reader) mode, or shows you how to create new SWMR-compatible files.

## Background

SWMR mode requires HDF5 files to be created with `libver='latest'` to use HDF5 file format version 3 or higher. Files created without this setting cannot use SWMR mode.

## Quick Reference

### ❌ Old Way (No SWMR Support)
```python
from mth5.mth5 import MTH5

mth5 = MTH5(file_version='0.2.0')
mth5.open_mth5('data.mth5', mode='w')  # Missing libver='latest'
# This file CANNOT be opened in SWMR mode later
```

### ✅ New Way (SWMR Compatible)
```python
from mth5.mth5 import MTH5

mth5 = MTH5(file_version='0.2.0')
mth5.open_mth5('data.mth5', mode='w', libver='latest')  # SWMR compatible!
# This file CAN be opened in SWMR mode later
```

## Migration Strategies

### Strategy 1: Copy to New SWMR-Compatible File (Recommended)

This creates a new file with the correct format and copies all data.

```python
from mth5.mth5 import MTH5
from pathlib import Path

def migrate_to_swmr_compatible(old_file, new_file):
    """
    Migrate an existing MTH5 file to SWMR-compatible format.
    
    Parameters
    ----------
    old_file : str or Path
        Path to existing MTH5 file (any format).
    new_file : str or Path
        Path for new SWMR-compatible file.
    
    Returns
    -------
    Path
        Path to the new SWMR-compatible file.
    """
    print(f"Migrating {old_file} -> {new_file}")
    
    # Open old file for reading
    old_mth5 = MTH5()
    old_mth5.open_mth5(old_file, mode='r')
    
    # Get experiment metadata from old file
    experiment = old_mth5.to_experiment(has_data=True)
    
    # Create new SWMR-compatible file
    new_mth5 = MTH5(file_version=old_mth5.file_version)
    new_mth5.open_mth5(new_file, mode='w', libver='latest')
    
    # Copy metadata and structure
    new_mth5.from_experiment(experiment)
    
    # Close both files
    old_mth5.close_mth5()
    new_mth5.close_mth5()
    
    print(f"Migration complete: {new_file}")
    print(f"New file is SWMR-compatible: {Path(new_file).exists()}")
    
    return Path(new_file)


# Example usage
old_file = 'legacy_data.mth5'
new_file = 'legacy_data_swmr.mth5'
migrate_to_swmr_compatible(old_file, new_file)

# Now you can use SWMR mode!
mth5 = MTH5()
mth5.open_mth5(new_file, mode='a', single_writer_multiple_reader=True)
```

### Strategy 2: Check and Warn

Add compatibility checks to your code.

```python
import h5py
from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error

def check_swmr_compatible(filename):
    """
    Check if an MTH5 file supports SWMR mode.
    
    Parameters
    ----------
    filename : str
        Path to MTH5 file.
    
    Returns
    -------
    bool
        True if file is SWMR-compatible, False otherwise.
    """
    try:
        with h5py.File(filename, 'r') as f:
            # Check file format version
            # SWMR requires HDF5 format version 3 (libver='latest')
            return f.id.get_access_plist().get_libver_bounds()[0] >= h5py.h5f.LIBVER_LATEST
    except Exception:
        return False


def open_with_swmr_check(filename, mode='r', use_swmr=False):
    """
    Open MTH5 file with SWMR compatibility check.
    
    Parameters
    ----------
    filename : str
        Path to MTH5 file.
    mode : str
        File opening mode.
    use_swmr : bool
        Whether to use SWMR mode.
    
    Returns
    -------
    MTH5
        Opened MTH5 object.
    """
    # Check compatibility if SWMR requested
    if use_swmr and not check_swmr_compatible(filename):
        print(f"⚠️  WARNING: File {filename} is not SWMR-compatible!")
        print(f"   File was created without libver='latest'")
        print(f"   Please migrate using Strategy 1 (see migration guide)")
        raise MTH5Error(
            f"File {filename} does not support SWMR mode. "
            f"Create a new file with libver='latest' or migrate existing file."
        )
    
    # Open file
    mth5 = MTH5()
    mth5.open_mth5(filename, mode=mode, single_writer_multiple_reader=use_swmr)
    return mth5


# Example usage
try:
    mth5 = open_with_swmr_check('old_file.mth5', mode='a', use_swmr=True)
    # Successfully opened in SWMR mode
except MTH5Error as e:
    print(f"Error: {e}")
    print("Please migrate the file first")
```

### Strategy 3: Batch Migration Script

For migrating multiple files:

```python
#!/usr/bin/env python
"""
Batch migrate MTH5 files to SWMR-compatible format.

Usage:
    python migrate_batch.py /path/to/files/*.mth5
"""
import sys
from pathlib import Path
from mth5.mth5 import MTH5


def migrate_file(input_file, output_dir=None, suffix='_swmr'):
    """
    Migrate a single MTH5 file.
    
    Parameters
    ----------
    input_file : Path
        Path to input file.
    output_dir : Path, optional
        Output directory. If None, uses same directory as input.
    suffix : str
        Suffix to add to output filename.
    
    Returns
    -------
    Path or None
        Path to migrated file, or None if migration failed.
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return None
    
    # Determine output path
    if output_dir:
        output_path = Path(output_dir) / f"{input_path.stem}{suffix}.h5"
    else:
        output_path = input_path.parent / f"{input_path.stem}{suffix}.h5"
    
    # Check if output already exists
    if output_path.exists():
        response = input(f"⚠️  Output file exists: {output_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print(f"Skipping {input_path}")
            return None
    
    try:
        print(f"🔄 Migrating: {input_path}")
        
        # Open and read old file
        old_mth5 = MTH5()
        old_mth5.open_mth5(str(input_path), mode='r')
        experiment = old_mth5.to_experiment(has_data=True)
        file_version = old_mth5.file_version
        old_mth5.close_mth5()
        
        # Create new SWMR-compatible file
        new_mth5 = MTH5(file_version=file_version)
        new_mth5.open_mth5(str(output_path), mode='w', libver='latest')
        new_mth5.from_experiment(experiment)
        new_mth5.close_mth5()
        
        print(f"✅ Migrated: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to migrate {input_path}: {e}")
        return None


def main(file_patterns):
    """
    Batch migrate MTH5 files.
    
    Parameters
    ----------
    file_patterns : list
        List of file paths or glob patterns.
    """
    from glob import glob
    
    # Collect all files
    all_files = []
    for pattern in file_patterns:
        if '*' in pattern:
            all_files.extend(glob(pattern))
        else:
            all_files.append(pattern)
    
    if not all_files:
        print("No files found to migrate")
        return 1
    
    print(f"\nFound {len(all_files)} file(s) to migrate\n")
    
    # Migrate each file
    successful = 0
    failed = 0
    
    for file_path in all_files:
        result = migrate_file(file_path)
        if result:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"✅ Successful: {successful}")
    if failed > 0:
        print(f"❌ Failed:     {failed}")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python migrate_batch.py <file_pattern> [<file_pattern> ...]")
        print("\nExamples:")
        print("  python migrate_batch.py data.mth5")
        print("  python migrate_batch.py /path/to/*.mth5")
        print("  python migrate_batch.py file1.mth5 file2.mth5 file3.mth5")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1:]))
```

## Best Practices Going Forward

### 1. Always Use libver='latest' for New Files

```python
# For all new files
mth5 = MTH5(file_version='0.2.0')
mth5.open_mth5('new_data.mth5', mode='w', libver='latest')
```

### 2. Document SWMR Compatibility

Add to your file metadata or README:

```python
# In your code comments or documentation
"""
This MTH5 file was created with libver='latest' and supports SWMR mode
for concurrent read/write access.
"""
```

### 3. Provide Fallback for Legacy Files

```python
from mth5.utils.exceptions import MTH5Error

def open_mth5_auto(filename, use_swmr=False):
    """
    Open MTH5 file with automatic SWMR fallback.
    
    If SWMR fails, falls back to normal mode.
    """
    mth5 = MTH5()
    
    if use_swmr:
        try:
            mth5.open_mth5(filename, mode='a', single_writer_multiple_reader=True)
            print(f"✅ Opened {filename} in SWMR mode")
        except MTH5Error:
            print(f"⚠️  SWMR not supported, opening in normal mode")
            mth5.open_mth5(filename, mode='a')
    else:
        mth5.open_mth5(filename, mode='a')
    
    return mth5
```

## FAQ

### Q: Can I enable SWMR on an existing file?

**A:** No. HDF5 file format is determined at creation time. You must migrate to a new file created with `libver='latest'`.

### Q: Will migrated files be compatible with older MTH5 versions?

**A:** Yes! Files created with `libver='latest'` can be read by any MTH5 version. The only difference is they **also** support SWMR mode, which older versions won't use.

### Q: Does migration preserve all data and metadata?

**A:** Yes! The `to_experiment()` and `from_experiment()` methods preserve:
- All surveys, stations, runs, and channels
- All metadata
- All data arrays
- Transfer functions (if present)
- Filters

### Q: How much disk space does migration require?

**A:** Temporarily, twice the file size (original + copy). After migration, you can delete the original if desired.

### Q: Can I use SWMR mode without migrating?

**A:** No. Files must be created with `libver='latest'` to support SWMR. There is no way to enable it after creation.

### Q: What about files created by third-party tools?

**A:** If they were created without `libver='latest'`, they need migration. Check with:

```python
import h5py

with h5py.File('file.mth5', 'r') as f:
    libver = f.id.get_access_plist().get_libver_bounds()[0]
    print(f"libver >= LATEST: {libver >= h5py.h5f.LIBVER_LATEST}")
```

## Performance Considerations

### Migration Performance

- **Small files** (< 100 MB): ~1-5 seconds
- **Medium files** (100 MB - 1 GB): ~10-60 seconds
- **Large files** (> 1 GB): ~1-5 minutes

### Migration Tips

1. **Disable compression temporarily** for faster migration
2. **Use SSD** if possible
3. **Migrate during low-usage periods**
4. **Test on a small file first**

### Parallel Migration

For many files, use parallel processing:

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def migrate_parallel(file_list, max_workers=4):
    """Migrate multiple files in parallel."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(migrate_file, file_list))
    
    successful = sum(1 for r in results if r is not None)
    print(f"Migrated {successful}/{len(file_list)} files")
    return results
```

## Testing SWMR Compatibility

After migration, verify SWMR support:

```python
from mth5.mth5 import MTH5

def verify_swmr_support(filename):
    """
    Verify file supports SWMR mode.
    
    Returns
    -------
    bool
        True if SWMR works, False otherwise.
    """
    try:
        # Try opening in SWMR mode
        mth5 = MTH5()
        mth5.open_mth5(filename, mode='a', single_writer_multiple_reader=True)
        
        # Check SWMR is active
        is_swmr = mth5.is_swmr_mode()
        
        mth5.close_mth5()
        
        if is_swmr:
            print(f"✅ {filename} supports SWMR mode")
            return True
        else:
            print(f"❌ {filename} opened but SWMR not active")
            return False
            
    except Exception as e:
        print(f"❌ {filename} does not support SWMR: {e}")
        return False


# Test your migrated file
verify_swmr_support('data_swmr.mth5')
```

## Summary

1. **New files**: Always use `libver='latest'`
2. **Existing files**: Migrate using Strategy 1 (copy method)
3. **Check compatibility**: Use verification functions
4. **Batch operations**: Use provided migration scripts
5. **Document**: Note SWMR compatibility in your workflows

## Additional Resources

- [SWMR User Guide](SWMR_GUIDE.md) - Complete SWMR documentation
- [MTH5 Documentation](https://mth5.readthedocs.io/)
- [HDF5 SWMR Specification](https://docs.hdfgroup.org/hdf5/develop/_s_w_m_r.html)
- [h5py SWMR Guide](https://docs.h5py.org/en/stable/swmr.html)

## Support

If you encounter issues during migration:

1. Check the file is readable: `mth5.open_mth5(file, 'r')`
2. Verify disk space: Migration needs 2x file size temporarily
3. Check permissions: Ensure write access to output directory
4. Review error messages: Most issues are clearly reported
5. Open an issue: [MTH5 GitHub Issues](https://github.com/kujaku11/mth5/issues)
