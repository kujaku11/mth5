#!/usr/bin/env python
"""
Advanced SWMR example: Concurrent writer and multiple readers using multiprocessing.

This demonstrates:
- One writer process adding data continuously
- Multiple reader processes monitoring in real-time
- Proper process synchronization
- Clean shutdown handling

Usage:
    python test_swmr_concurrent.py

Press Ctrl+C to stop all processes.
"""
import multiprocessing as mp
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from mth5.mth5 import MTH5


def writer_process(filepath, stop_event, ready_event, duration=10):
    """
    Writer process: Creates file and adds data continuously

    Parameters
    ----------
    filepath : str
        Path to MTH5 file
    stop_event : multiprocessing.Event
        Event to signal shutdown
    ready_event : multiprocessing.Event
        Event to signal file is ready for readers
    duration : int
        How many seconds to run
    """
    print(f"[Writer] Starting...")

    try:
        # Create initial file structure
        print(f"[Writer] Creating file: {filepath}")
        mth5 = MTH5(file_version="0.2.0")
        mth5.open_mth5(filepath, mode="w")
        survey = mth5.add_survey("live_survey")
        station = mth5.add_station("MT001", survey="live_survey")
        run = station.add_run("run_001")
        mth5.close_mth5()
        print(f"[Writer] Initial structure created")

        # Reopen in SWMR writer mode
        print(f"[Writer] Activating SWMR writer mode...")
        mth5.open_mth5(filepath, mode="a", single_writer_multiple_reader=True)

        if not mth5.is_swmr_mode():
            print(f"[Writer] ERROR: SWMR mode not activated!")
            return

        print(f"[Writer] SWMR writer mode active")

        # Signal that file is ready for readers
        ready_event.set()

        # Get handles
        station = mth5.get_station("MT001", survey="live_survey")
        run = station.get_run("run_001")

        # Add initial channels
        print(f"[Writer] Creating channels...")
        chunk_size = 100
        initial_data = np.random.randn(chunk_size)

        ex_ch = run.add_channel("Ex", "electric", initial_data, channel_dtype="float32")
        ey_ch = run.add_channel("Ey", "electric", initial_data, channel_dtype="float32")
        hx_ch = run.add_channel("Hx", "magnetic", initial_data, channel_dtype="float32")

        mth5.flush()
        print(f"[Writer] Initial channels created with {chunk_size} samples")

        # Continuously add data
        iteration = 1
        start_time = time.time()

        while not stop_event.is_set() and (time.time() - start_time) < duration:
            # Simulate data acquisition
            new_data = np.random.randn(chunk_size) * np.random.uniform(0.5, 2.0)

            # Append to channels (simplified - in reality you'd need to handle dataset resizing)
            # For this demo, we'll just track how many chunks we've added
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[Writer] [{timestamp}] Iteration {iteration}: "
                f"Adding {chunk_size} samples"
            )

            # In a real implementation, you would:
            # 1. Resize the dataset: ex_ch.hdf5_dataset.resize((current_size + chunk_size,))
            # 2. Write new data: ex_ch.hdf5_dataset[current_size:] = new_data
            # 3. Update metadata if needed

            # For this demo, we'll create a new run each iteration to demonstrate
            # continuous updates that readers can see
            new_run = station.add_run(f"run_{iteration:03d}")
            new_run.add_channel("Ex", "electric", new_data, channel_dtype="float32")

            # CRITICAL: Flush to make data visible to readers
            mth5.flush()

            iteration += 1
            time.sleep(1)  # Add data every second

        print(f"[Writer] Completed {iteration-1} iterations")
        print(f"[Writer] Shutting down...")

    except Exception as e:
        print(f"[Writer] ERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if mth5._has_open_file:
            mth5.close_mth5()
        print(f"[Writer] Stopped")


def reader_process(reader_id, filepath, stop_event, ready_event):
    """
    Reader process: Monitors file and reports updates

    Parameters
    ----------
    reader_id : int
        Unique reader identifier
    filepath : str
        Path to MTH5 file
    stop_event : multiprocessing.Event
        Event to signal shutdown
    ready_event : multiprocessing.Event
        Event to wait for file ready
    """
    print(f"[Reader {reader_id}] Starting...")

    # Wait for writer to create file
    print(f"[Reader {reader_id}] Waiting for file to be ready...")
    ready_event.wait(timeout=30)

    if not ready_event.is_set():
        print(f"[Reader {reader_id}] ERROR: Timeout waiting for file")
        return

    # Give writer a moment to fully activate SWMR
    time.sleep(0.5)

    try:
        # Open in SWMR reader mode
        print(f"[Reader {reader_id}] Opening file in SWMR reader mode...")
        mth5 = MTH5()
        mth5.open_mth5(filepath, mode="r", single_writer_multiple_reader=True)

        if not mth5.is_swmr_mode():
            print(f"[Reader {reader_id}] ERROR: SWMR mode not active")
            return

        print(f"[Reader {reader_id}] SWMR reader mode active")

        last_run_count = 0
        last_channel_count = 0

        # Monitor for changes
        while not stop_event.is_set():
            try:
                # Get current state
                run_df = mth5.run_summary.to_dataframe()
                channel_df = mth5.channel_summary.to_dataframe()

                current_runs = len(run_df)
                current_channels = len(channel_df)

                # Report if anything changed
                if (
                    current_runs != last_run_count
                    or current_channels != last_channel_count
                ):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[Reader {reader_id}] [{timestamp}] Update: "
                        f"{current_runs} runs, {current_channels} channels"
                    )

                    if current_channels > last_channel_count:
                        new_channels = current_channels - last_channel_count
                        print(f"[Reader {reader_id}] --> +{new_channels} new channels")

                    last_run_count = current_runs
                    last_channel_count = current_channels

                time.sleep(0.5)  # Check twice per second

            except Exception as e:
                if not stop_event.is_set():
                    print(f"[Reader {reader_id}] Error reading: {e}")
                break

        print(f"[Reader {reader_id}] Shutting down...")

    except Exception as e:
        print(f"[Reader {reader_id}] ERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if mth5._has_open_file:
            mth5.close_mth5()
        print(f"[Reader {reader_id}] Stopped")


def main():
    """Run concurrent SWMR test with writer and multiple readers"""
    print("\n" + "=" * 70)
    print("MTH5 SWMR CONCURRENT ACCESS TEST")
    print("=" * 70)
    print("This will run for ~10 seconds. Press Ctrl+C to stop early.")
    print("=" * 70)

    # Create temporary file
    tmpdir = tempfile.mkdtemp()
    filepath = str(Path(tmpdir) / "concurrent_test.mth5")
    print(f"\nTest file: {filepath}\n")

    # Create synchronization objects
    stop_event = mp.Event()
    ready_event = mp.Event()

    # Create processes
    num_readers = 3
    duration = 10  # seconds

    print(f"Starting 1 writer and {num_readers} readers...\n")

    # Start writer
    writer = mp.Process(
        target=writer_process, args=(filepath, stop_event, ready_event, duration)
    )
    writer.start()

    # Start readers
    readers = []
    for i in range(num_readers):
        reader = mp.Process(
            target=reader_process, args=(i + 1, filepath, stop_event, ready_event)
        )
        reader.start()
        readers.append(reader)

    try:
        # Wait for writer to complete or user interrupt
        writer.join()

        # Give readers a moment to see final state
        time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nUser interrupted - shutting down...")
    finally:
        # Signal all processes to stop
        stop_event.set()

        # Wait for all processes
        print("\nWaiting for processes to complete...")

        for reader in readers:
            reader.join(timeout=5)
            if reader.is_alive():
                print(f"Force terminating reader...")
                reader.terminate()

        if writer.is_alive():
            writer.join(timeout=5)
            if writer.is_alive():
                print(f"Force terminating writer...")
                writer.terminate()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"File created at: {filepath}")
    print("You can manually inspect the file with:")
    print(f"  from mth5.mth5 import MTH5")
    print(f"  mth5 = MTH5()")
    print(f"  mth5.open_mth5(r'{filepath}')")
    print(f"  print(mth5.run_summary)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
