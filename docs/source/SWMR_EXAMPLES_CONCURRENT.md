# SWMR Concurrent Access Examples

This guide demonstrates advanced SWMR usage with concurrent writer and multiple readers using Python's multiprocessing module.

## Overview

This example shows:
- One writer process adding data continuously in real-time
- Multiple reader processes monitoring file updates concurrently
- Proper process synchronization using events
- Clean shutdown handling with signal management
- Real-world data streaming scenarios

## Architecture

```
┌─────────────────┐
│  Writer Process │  Creates file structure
│   (main thread) │  Activates SWMR writer mode
└────────┬────────┘  Adds data continuously
         │           Flushes regularly
         │
         ├──────────────────────────────────┐
         │                                   │
┌────────▼────────┐              ┌──────────▼─────────┐
│  Reader 1       │              │  Reader N          │
│  (child proc)   │     ...      │  (child proc)      │
│                 │              │                    │
│  Opens SWMR     │              │  Opens SWMR        │
│  reader mode    │              │  reader mode       │
│  Monitors       │              │  Monitors          │
│  updates        │              │  updates           │
└─────────────────┘              └────────────────────┘
```

## Complete Example

### Writer Process

The writer creates the file structure and continuously adds data:

```python
import multiprocessing as mp
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
        # Step 1: Create initial file structure
        print(f"[Writer] Creating file: {filepath}")
        mth5 = MTH5(file_version="0.2.0")
        mth5.open_mth5(filepath, mode="w")
        survey = mth5.add_survey("live_survey")
        station = mth5.add_station("MT001", survey="live_survey")
        run = station.add_run("run_001")
        mth5.close_mth5()
        print(f"[Writer] Initial structure created")
        
        # Step 2: Reopen in SWMR writer mode
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
        
        # Step 3: Add initial channels
        print(f"[Writer] Creating channels...")
        chunk_size = 100
        initial_data = np.random.randn(chunk_size)
        
        ex_ch = run.add_channel("Ex", "electric", initial_data, 
                                channel_dtype="float32")
        ey_ch = run.add_channel("Ey", "electric", initial_data, 
                                channel_dtype="float32")
        hx_ch = run.add_channel("Hx", "magnetic", initial_data, 
                                channel_dtype="float32")
        
        mth5.flush()
        print(f"[Writer] Initial channels created with {chunk_size} samples")
        
        # Step 4: Continuously add data
        iteration = 1
        start_time = time.time()
        
        while not stop_event.is_set() and (time.time() - start_time) < duration:
            # Simulate data acquisition
            new_data = np.random.randn(chunk_size) * np.random.uniform(0.5, 2.0)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[Writer] [{timestamp}] Iteration {iteration}: "
                  f"Adding {chunk_size} samples")
            
            # Create new run to demonstrate continuous updates
            # (In production, you would resize and append to existing datasets)
            new_run = station.add_run(f"run_{iteration:03d}")
            new_run.add_channel("Ex", "electric", new_data, 
                               channel_dtype="float32")
            
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
        mth5.close_mth5()
        print(f"[Writer] Stopped")
```

### Reader Process

Readers monitor the file and report updates in real-time:

```python
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
                run_df = mth5.run_summary
                channel_df = mth5.channel_summary.to_dataframe()
                
                current_runs = len(run_df)
                current_channels = len(channel_df)
                
                # Report if anything changed
                if (current_runs != last_run_count or 
                    current_channels != last_channel_count):
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[Reader {reader_id}] [{timestamp}] Update: "
                          f"{current_runs} runs, {current_channels} channels")
                    
                    if current_channels > last_channel_count:
                        new_channels = current_channels - last_channel_count
                        print(f"[Reader {reader_id}] --> +{new_channels} "
                              f"new channels")
                    
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
        mth5.close_mth5()
        print(f"[Reader {reader_id}] Stopped")
```

### Main Orchestration

The main function coordinates all processes:

```python
import tempfile

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
    
    # Configuration
    num_readers = 3
    duration = 10  # seconds
    
    print(f"Starting 1 writer and {num_readers} readers...\n")
    
    # Start writer
    writer = mp.Process(
        target=writer_process,
        args=(filepath, stop_event, ready_event, duration)
    )
    writer.start()
    
    # Start readers
    readers = []
    for i in range(num_readers):
        reader = mp.Process(
            target=reader_process,
            args=(i + 1, filepath, stop_event, ready_event)
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
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
```

## Expected Output

```
======================================================================
MTH5 SWMR CONCURRENT ACCESS TEST
======================================================================
This will run for ~10 seconds. Press Ctrl+C to stop early.
======================================================================

Test file: /tmp/tmp_abc123/concurrent_test.mth5

Starting 1 writer and 3 readers...

[Writer] Starting...
[Reader 1] Starting...
[Reader 2] Starting...
[Reader 3] Starting...
[Writer] Creating file: /tmp/tmp_abc123/concurrent_test.mth5
[Reader 1] Waiting for file to be ready...
[Reader 2] Waiting for file to be ready...
[Reader 3] Waiting for file to be ready...
[Writer] Initial structure created
[Writer] Activating SWMR writer mode...
[Writer] SWMR writer mode active
[Writer] Creating channels...
[Writer] Initial channels created with 100 samples
[Reader 1] Opening file in SWMR reader mode...
[Reader 2] Opening file in SWMR reader mode...
[Reader 3] Opening file in SWMR reader mode...
[Reader 1] SWMR reader mode active
[Reader 2] SWMR reader mode active
[Reader 3] SWMR reader mode active
[Writer] [10:30:15] Iteration 1: Adding 100 samples
[Reader 1] [10:30:15] Update: 2 runs, 4 channels
[Reader 1] --> +1 new channels
[Reader 2] [10:30:15] Update: 2 runs, 4 channels
[Reader 3] [10:30:15] Update: 2 runs, 4 channels
[Writer] [10:30:16] Iteration 2: Adding 100 samples
[Reader 2] [10:30:16] Update: 3 runs, 5 channels
[Reader 2] --> +1 new channels
...
[Writer] Completed 10 iterations
[Writer] Shutting down...
[Writer] Stopped
[Reader 1] Shutting down...
[Reader 1] Stopped
[Reader 2] Shutting down...
[Reader 2] Stopped
[Reader 3] Shutting down...
[Reader 3] Stopped

======================================================================
TEST COMPLETE
======================================================================
```

## Complete Script

The complete working script is available at:
[examples/test_swmr_concurrent.py](../../examples/test_swmr_concurrent.py)

Run it with:
```bash
python examples/test_swmr_concurrent.py
```

## Key Implementation Details

### 1. Process Synchronization

Use `multiprocessing.Event` objects for coordination:

```python
stop_event = mp.Event()   # Signals shutdown to all processes
ready_event = mp.Event()  # Signals file is ready for readers
```

### 2. Writer Must Flush

Readers won't see updates until the writer flushes:

```python
# Add data
station.add_run(f"run_{i:03d}")

# Make it visible to readers
mth5.flush()  # CRITICAL!
```

### 3. Reader Polling

Readers should poll periodically for updates:

```python
while not stop_event.is_set():
    # Check for updates
    channel_df = mth5.channel_summary.to_dataframe()
    # Process updates...
    time.sleep(0.5)  # Poll interval
```

### 4. Graceful Shutdown

Handle cleanup properly:

```python
try:
    writer.join()  # Wait for completion
except KeyboardInterrupt:
    stop_event.set()  # Signal all processes
finally:
    # Force termination if needed
    for proc in [writer] + readers:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
```

### 5. Windows Compatibility

Windows requires explicit spawn mode:

```python
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
```

## Real-World Use Cases

### 1. Real-Time Data Acquisition

```python
# Writer: Data acquisition system
while acquiring:
    new_samples = instrument.read()
    run.append_data(new_samples)
    mth5.flush()
    time.sleep(sample_interval)

# Readers: Real-time monitoring dashboards
while monitoring:
    latest_data = mth5.get_latest_samples()
    dashboard.update(latest_data)
```

### 2. Live Processing Pipeline

```python
# Writer: Field data collector
# Reader 1: Quality control monitor
# Reader 2: Processing pipeline
# Reader 3: Backup/archival system
```

### 3. Distributed Monitoring

```python
# Writer: Central data hub
# Readers: Remote monitoring stations (network file sharing)
```

## Performance Considerations

### Flush Frequency

Balance between data visibility and performance:

```python
# High frequency: More overhead, faster visibility
mth5.flush()  # After every update

# Medium frequency: Balanced
if iteration % 10 == 0:
    mth5.flush()  # Every 10 updates

# Low frequency: Better performance, delayed visibility
if time.time() - last_flush > 5.0:
    mth5.flush()  # Every 5 seconds
```

### Reader Polling

Adjust polling rate based on update frequency:

```python
# Fast updates (sub-second)
time.sleep(0.1)  # Poll 10 times per second

# Normal updates (seconds)
time.sleep(1.0)  # Poll once per second

# Slow updates (minutes)
time.sleep(10.0)  # Poll every 10 seconds
```

## Troubleshooting

### Issue: Readers Don't See Updates

**Solution:** Ensure writer calls `flush()` regularly

### Issue: "File is Locked" Errors

**Solution:** Verify only one writer is active

### Issue: Slow Performance

**Solution:** Reduce flush frequency or reader polling rate

### Issue: Process Hangs on Shutdown

**Solution:** Implement proper timeout and termination logic

## Next Steps

- Review [SWMR User Guide](SWMR_GUIDE.md) for detailed concepts
- Check [SWMR Basic Examples](SWMR_EXAMPLES_BASIC.md) for simpler examples
- Read [SWMR Migration Guide](SWMR_MIGRATION.md) for existing file conversion

## Additional Resources

- [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html)
- [HDF5 SWMR specification](https://docs.hdfgroup.org/hdf5/develop/_s_w_m_r.html)
- [MTH5 API documentation](../modules.html)
