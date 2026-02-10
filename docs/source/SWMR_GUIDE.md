# Single Writer Multiple Reader (SWMR) Mode in MTH5

## Overview

SWMR (Single Writer Multiple Reader) mode allows one process to write to an MTH5 file while multiple other processes read from it **simultaneously**. This is particularly useful for:

- Real-time data collection and monitoring
- Live data processing pipelines
- Concurrent analysis during data acquisition
- Distributed processing systems

## Quick Start

### Writer (Data Acquisition)

```python
from mth5.mth5 import MTH5
import numpy as np

# Open file as SWMR writer
mth5_writer = MTH5()
mth5_writer.open_mth5('realtime_data.mth5', mode='a', single_writer_multiple_reader=True)

# Add data incrementally
survey = mth5_writer.add_survey('live_survey')
station = mth5_writer.add_station('STA001', survey='live_survey')
run = station.add_run('run_001')

# Add channel data
data = np.random.random(1000)
run.add_channel('Ex', 'electric', data)

# IMPORTANT: Flush to make data visible to readers
mth5_writer.flush()

# Continue adding more data...
mth5_writer.close_mth5()
```

### Reader (Concurrent Processing)

```python
from mth5.mth5 import MTH5

# Open same file as SWMR reader (while writer is active)
mth5_reader = MTH5()
mth5_reader.open_mth5('realtime_data.mth5', mode='r', single_writer_multiple_reader=True)

# Read data
run_df = mth5_reader.run_summary
print(f"Current runs: {len(run_df)}")

# Refresh to see new data (reopen or re-read summary)
channel_df = mth5_reader.channel_summary.to_dataframe()

mth5_reader.close_mth5()
```

## Important Gotchas & Requirements

### ✅ DO's

1. **Use Existing Files Only**
   ```python
   # Create file first without SWMR
   mth5 = MTH5()
   mth5.open_mth5('data.mth5', 'w')
   mth5.close_mth5()
   
   # Then open with SWMR
   mth5.open_mth5('data.mth5', 'a', single_writer_multiple_reader=True)
   ```

2. **Correct Modes**
   - **Writer**: Use `mode='a'` or `mode='r+'`
   - **Reader**: Use `mode='r'`
   
3. **Regular Flushing (Writers)**
   ```python
   # Flush after significant data additions
   station = mth5_writer.add_station('STA001', survey='survey')
   mth5_writer.flush()  # Readers can now see this station
   ```

4. **Close All Handles Before Activation**
   ```python
   # Wrong - will fail
   station = mth5.add_station('STA001')
   # station object still has open handle
   mth5.__hdf5_obj.swmr_mode = True  # FAILS!
   
   # Right - handled automatically by open_mth5()
   mth5.open_mth5('data.mth5', 'a', single_writer_multiple_reader=True)
   ```

5. **Use libver='latest'**
   - Automatically set by `single_writer_multiple_reader=True`
   - Or explicitly: `open_mth5('data.mth5', 'a', libver='latest')`

### ❌ DON'Ts

1. **Don't Use with New Files**
   ```python
   # WRONG - will raise MTH5Error
   mth5.open_mth5('new_file.mth5', 'w', single_writer_multiple_reader=True)
   ```

2. **Don't Use Write Mode ('w')**
   ```python
   # WRONG - incompatible with SWMR
   mth5.open_mth5('existing.mth5', 'w', single_writer_multiple_reader=True)
   ```

3. **Don't Delete/Restructure in SWMR Writer**
   ```python
   # WRONG - cannot delete in SWMR mode
   mth5_writer.remove_station('STA001')  # Will likely fail
   
   # RIGHT - append only
   mth5_writer.add_station('STA002')  # OK
   ```

4. **Don't Forget to Flush (Writers)**
   ```python
   # Readers won't see new data until flush!
   mth5_writer.add_station('STA001')
   mth5_writer.flush()  # Now readers can see it
   ```

5. **Don't Open Multiple SWMR Writers**
   ```python
   # WRONG - only ONE writer allowed
   writer1 = MTH5()
   writer1.open_mth5('data.mth5', 'a', single_writer_multiple_reader=True)
   
   writer2 = MTH5()
   writer2.open_mth5('data.mth5', 'a', single_writer_multiple_reader=True)  # FAILS!
   ```

## Complete Working Example

### Real-Time Data Acquisition System

**Writer (data_collector.py)**:
```python
#!/usr/bin/env python
"""
Real-time MT data collector using SWMR mode
"""
import time
import numpy as np
from mth5.mth5 import MTH5

def collect_data():
    # Create initial file structure
    print("Setting up MTH5 file...")
    mth5 = MTH5(file_version='0.2.0')
    mth5.open_mth5('realtime_mt.mth5', 'w')
    survey = mth5.add_survey('live_survey')
    station = mth5.add_station('MT001', survey='live_survey')
    run = station.add_run('run_001a')
    mth5.close_mth5()
    
    # Reopen in SWMR writer mode
    print("Activating SWMR writer mode...")
    mth5.open_mth5('realtime_mt.mth5', 'a', single_writer_multiple_reader=True)
    
    # Get handles
    station = mth5.get_station('MT001', survey='live_survey')
    run = station.get_run('run_001a')
    
    # Simulate real-time data collection
    print("Starting data collection (Ctrl+C to stop)...")
    try:
        sample_rate = 100  # Hz
        chunk_size = 1000  # samples per chunk
        chunk_count = 0
        
        while True:
            # Simulate data acquisition
            ex_data = np.random.randn(chunk_size) * 0.01
            ey_data = np.random.randn(chunk_size) * 0.01
            hx_data = np.random.randn(chunk_size) * 10
            hy_data = np.random.randn(chunk_size) * 10
            hz_data = np.random.randn(chunk_size) * 5
            
            # Add or append to channels
            if chunk_count == 0:
                # First chunk - create channels
                run.add_channel('Ex', 'electric', ex_data, 
                               channel_dtype='float32', max_shape=(None,))
                run.add_channel('Ey', 'electric', ey_data,
                               channel_dtype='float32', max_shape=(None,))
                run.add_channel('Hx', 'magnetic', hx_data,
                               channel_dtype='float32', max_shape=(None,))
                run.add_channel('Hy', 'magnetic', hy_data,
                               channel_dtype='float32', max_shape=(None,))
                run.add_channel('Hz', 'magnetic', hz_data,
                               channel_dtype='float32', max_shape=(None,))
                print("Created channels")
            else:
                # Subsequent chunks - append data
                # Note: In SWMR, you can only append to datasets with unlimited dimensions
                ex_channel = run.get_channel('Ex')
                # Resize and add data (simplified - actual implementation may vary)
                current_size = len(ex_channel.hdf5_dataset)
                ex_channel.hdf5_dataset.resize((current_size + chunk_size,))
                ex_channel.hdf5_dataset[current_size:] = ex_data
                # Repeat for other channels...
            
            chunk_count += 1
            
            # Flush to make data visible to readers
            mth5.flush()
            print(f"Chunk {chunk_count}: Added {chunk_size} samples, flushed to disk")
            
            # Wait before next chunk (simulate real-time acquisition)
            time.sleep(chunk_size / sample_rate)  # Real-time pace
            
    except KeyboardInterrupt:
        print("\nStopping data collection...")
    finally:
        mth5.close_mth5()
        print("Data collection complete")

if __name__ == '__main__':
    collect_data()
```

**Reader (data_monitor.py)**:
```python
#!/usr/bin/env python
"""
Real-time MT data monitor using SWMR mode
"""
import time
from mth5.mth5 import MTH5

def monitor_data():
    print("Opening MTH5 file in SWMR reader mode...")
    
    mth5 = MTH5()
    mth5.open_mth5('realtime_mt.mth5', 'r', single_writer_multiple_reader=True)
    
    print("Monitoring data (Ctrl+C to stop)...")
    last_sample_count = 0
    
    try:
        while True:
            # Get current state
            channel_df = mth5.channel_summary.to_dataframe()
            
            if not channel_df.empty:
                # Check Ex channel
                ex_row = channel_df[channel_df.component == 'Ex']
                if not ex_row.empty:
                    current_samples = ex_row.iloc[0].n_samples
                    
                    if current_samples != last_sample_count:
                        new_samples = current_samples - last_sample_count
                        print(f"Data update: {current_samples} total samples "
                              f"(+{new_samples} new)")
                        last_sample_count = current_samples
                        
                        # Process new data if needed
                        # station = mth5.get_station('MT001', survey='live_survey')
                        # run = station.get_run('run_001a')
                        # ex = run.get_channel('Ex')
                        # data = ex.hdf5_dataset[:]  # Get all data
                        # ... process data ...
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        mth5.close_mth5()
        print("Monitor stopped")

if __name__ == '__main__':
    monitor_data()
```

**Usage**:
```bash
# Terminal 1: Start data collection
python data_collector.py

# Terminal 2: Monitor data in real-time
python data_monitor.py

# Terminal 3: Run additional analysis
python analyze_data.py  # Another SWMR reader
```

## Advanced Usage

### Multiple Readers

```python
# Reader 1: Live plotting
mth5_plotter = MTH5()
mth5_plotter.open_mth5('data.mth5', 'r', single_writer_multiple_reader=True)

# Reader 2: Real-time processing
mth5_processor = MTH5()
mth5_processor.open_mth5('data.mth5', 'r', single_writer_multiple_reader=True)

# Reader 3: Quality monitoring
mth5_qc = MTH5()
mth5_qc.open_mth5('data.mth5', 'r', single_writer_multiple_reader=True)

# All can read simultaneously!
```

### Checking SWMR Status

```python
mth5 = MTH5()
mth5.open_mth5('data.mth5', 'a', single_writer_multiple_reader=True)

# Check if SWMR is active
if mth5.is_swmr_mode():
    print("SWMR mode is active")
    print(f"File is {'writable' if mth5.h5_is_write() else 'read-only'}")
```

### Error Handling

```python
from mth5.mth5 import MTH5
from mth5.utils.exceptions import MTH5Error

try:
    mth5 = MTH5()
    mth5.open_mth5('data.mth5', 'w', single_writer_multiple_reader=True)
except MTH5Error as e:
    print(f"Cannot use SWMR with mode='w': {e}")
    # Use correct mode
    mth5.open_mth5('data.mth5', 'a', single_writer_multiple_reader=True)
```

## Performance Considerations

### Writer Best Practices

1. **Batch Writes**: Accumulate data before writing
   ```python
   # Good - batch writes
   buffer = []
   for i in range(100):
       buffer.append(acquire_sample())
   channel.append_data(np.array(buffer))
   mth5.flush()
   ```

2. **Flush Frequency**: Balance visibility vs performance
   ```python
   # Flush every N samples or M seconds
   samples_since_flush = 0
   for sample in data_stream:
       add_sample(sample)
       samples_since_flush += 1
       
       if samples_since_flush >= 1000:  # Flush every 1000 samples
           mth5.flush()
           samples_since_flush = 0
   ```

3. **Use Chunking**: Optimize HDF5 chunk size
   ```python
   # Set appropriate chunk size for dataset
   run.add_channel('Ex', 'electric', data,
                  chunks=(10000,),  # 10k samples per chunk
                  max_shape=(None,))
   ```

### Reader Best Practices

1. **Minimize Reopens**: Keep file open, refresh metadata
   ```python
   # Don't reopen frequently
   while monitoring:
       # Read updated summary
       df = mth5.channel_summary.to_dataframe()
       time.sleep(1)
   ```

2. **Cache Static Data**: Don't re-read unchanged data
   ```python
   # Cache metadata
   station_metadata = station.metadata
   ```

## Troubleshooting

### Common Errors

#### "Unable to set SWMR mode"
- **Cause**: Open dataset handles exist
- **Solution**: Handled automatically by `open_mth5()`, ensures clean state

#### "SWMR mode cannot be used with mode='w'"
- **Cause**: Trying to create new file in SWMR mode
- **Solution**: Create file first, then reopen with SWMR

#### "File does not exist"
- **Cause**: Trying SWMR on non-existent file
- **Solution**: Create file first without SWMR

#### Reader Not Seeing New Data
- **Cause**: Writer hasn't flushed
- **Solution**: Writer must call `mth5.flush()` regularly

### Platform-Specific Issues

**Windows**:
- File locking may be more strict
- Ensure no other programs have file open

**Network File Systems**:
- SWMR may not work reliably on some network drives
- Test on local disk first

## Comparison: SWMR vs Normal Mode

| Feature | Normal Mode | SWMR Mode |
|---------|------------|-----------|
| **Concurrent Access** | No | Yes (1 writer, N readers) |
| **Writer Operations** | All | Append only, no delete |
| **File Creation** | Yes | No |
| **Performance** | Slightly faster | Small overhead |
| **Complexity** | Simple | Moderate |
| **Use Case** | Batch processing | Real-time systems |

## Summary

SWMR mode in MTH5 enables powerful real-time data collection and processing workflows:

✅ **Use SWMR when**:
- Collecting data in real-time
- Need concurrent monitoring/processing
- Running live data pipelines

❌ **Don't use SWMR when**:
- Batch processing completed data
- Need to delete/restructure data
- Single-process access is sufficient

**Key Points**:
1. File must exist before SWMR activation
2. Writer uses `mode='a'`, readers use `mode='r'`
3. Writer must flush regularly for readers to see updates
4. Only one writer allowed, unlimited readers
5. Writer can only append, not delete/restructure

## References

- [HDF5 SWMR Documentation](https://docs.hdfgroup.org/hdf5/develop/_s_w_m_r.html)
- [h5py SWMR Guide](https://docs.h5py.org/en/stable/swmr.html)
- MTH5 Documentation: [https://mth5.readthedocs.io/](https://mth5.readthedocs.io/)
