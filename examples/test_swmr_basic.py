#!/usr/bin/env python
"""
Basic test script for SWMR (Single Writer Multiple Reader) functionality in MTH5.

This script tests:
1. File creation without SWMR
2. SWMR writer mode activation
3. Adding data in SWMR mode
4. Flushing data to readers
5. Error handling for invalid modes

Usage:
    python test_swmr_basic.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error


def test_swmr_file_creation_error():
    """Test that creating new file with SWMR raises error"""
    print("\n" + "=" * 70)
    print("TEST 1: File creation with SWMR (should fail)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_swmr_new.mth5"

        mth5 = MTH5(file_version="0.2.0")

        try:
            # This should raise an error
            mth5.open_mth5(str(test_file), mode="w", single_writer_multiple_reader=True)
            print("❌ FAILED: Should have raised MTH5Error")
            return False
        except MTH5Error as e:
            print(f"✅ PASSED: Correctly raised error: {e}")
            return True
        finally:
            if mth5._has_open_file:
                mth5.close_mth5()


def test_swmr_writer_mode():
    """Test SWMR writer mode activation"""
    print("\n" + "=" * 70)
    print("TEST 2: SWMR writer mode activation")
    print("=" * 70)

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
        try:
            mth5.open_mth5(str(test_file), mode="a", single_writer_multiple_reader=True)
            print("✓ SWMR writer mode activated")

            # Step 3: Verify SWMR status
            if mth5.is_swmr_mode():
                print("✓ SWMR mode verified active")
            else:
                print("❌ SWMR mode not active")
                return False

            # Step 4: Add data in SWMR mode
            print("Step 3: Adding data in SWMR mode...")
            station = mth5.get_station("STA001", survey="test_survey")
            run = station.add_run("run_001")

            # Add channel data
            data = np.random.randn(1000)
            ch = run.add_channel("Ex", "electric", data, channel_dtype="float32")
            print(f"✓ Added channel with {len(data)} samples")

            # Step 5: Flush data
            print("Step 4: Flushing data...")
            mth5.flush()
            print("✓ Data flushed")

            print("✅ PASSED: SWMR writer mode test")
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}")
            return False
        finally:
            if mth5._has_open_file:
                mth5.close_mth5()


def test_swmr_reader_mode():
    """Test SWMR reader mode"""
    print("\n" + "=" * 70)
    print("TEST 3: SWMR reader mode")
    print("=" * 70)

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
        try:
            mth5_r.open_mth5(
                str(test_file), mode="r", single_writer_multiple_reader=True
            )
            print("✓ SWMR reader mode activated")

            # Verify SWMR status
            if mth5_r.is_swmr_mode():
                print("✓ SWMR mode verified active")
            else:
                print("❌ SWMR mode not active")
                return False

            # Read data
            print("Step 3: Reading data...")
            channel_df = mth5_r.channel_summary.to_dataframe()
            print(f"✓ Found {len(channel_df)} channels")

            if len(channel_df) > 0:
                print(f"  - Channel: {channel_df.iloc[0].component}")
                print(f"  - Samples: {channel_df.iloc[0].n_samples}")

            print("✅ PASSED: SWMR reader mode test")
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}")
            return False
        finally:
            if mth5_r._has_open_file:
                mth5_r.close_mth5()


def test_swmr_invalid_modes():
    """Test that invalid modes are rejected"""
    print("\n" + "=" * 70)
    print("TEST 4: Invalid mode handling")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_swmr_modes.mth5"

        # Create file
        mth5 = MTH5(file_version="0.2.0")
        mth5.open_mth5(str(test_file), mode="w")
        mth5.close_mth5()

        # Test mode='w' with SWMR (should fail)
        print("Test 4a: mode='w' with SWMR (should fail)...")
        mth5 = MTH5()
        try:
            mth5.open_mth5(str(test_file), mode="w", single_writer_multiple_reader=True)
            print("❌ FAILED: Should have raised error for mode='w'")
            return False
        except MTH5Error:
            print("✓ Correctly rejected mode='w'")
        finally:
            if mth5._has_open_file:
                mth5.close_mth5()

        # Test mode='x' with SWMR (should fail)
        print("Test 4b: mode='x' with SWMR (should fail)...")
        mth5 = MTH5()
        try:
            mth5.open_mth5(str(test_file), mode="x", single_writer_multiple_reader=True)
            print("❌ FAILED: Should have raised error for mode='x'")
            return False
        except MTH5Error:
            print("✓ Correctly rejected mode='x'")
        finally:
            if mth5._has_open_file:
                mth5.close_mth5()

        print("✅ PASSED: Invalid mode handling test")
        return True


def test_swmr_nonexistent_file():
    """Test SWMR with non-existent file"""
    print("\n" + "=" * 70)
    print("TEST 5: Non-existent file handling")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "nonexistent.mth5"

        mth5 = MTH5()
        try:
            mth5.open_mth5(str(test_file), mode="a", single_writer_multiple_reader=True)
            print("❌ FAILED: Should have raised error for non-existent file")
            return False
        except MTH5Error as e:
            print(f"✅ PASSED: Correctly raised error: {e}")
            return True
        finally:
            if mth5._has_open_file:
                mth5.close_mth5()


def main():
    """Run all SWMR tests"""
    print("\n" + "=" * 70)
    print("MTH5 SWMR MODE TEST SUITE")
    print("=" * 70)

    tests = [
        ("File Creation Error", test_swmr_file_creation_error),
        ("Writer Mode", test_swmr_writer_mode),
        ("Reader Mode", test_swmr_reader_mode),
        ("Invalid Modes", test_swmr_invalid_modes),
        ("Non-existent File", test_swmr_nonexistent_file),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
