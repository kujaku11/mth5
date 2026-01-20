# Filter Duplication Analysis

## Key Finding: The filters are being added 3 times during MTH5 creation!

### Evidence from debug_filter_tracking.py output:

1. **First Round** (calls #478-477): All 6 unique filters added correctly
   - hy: magnetic field 3 pole butterworth low-pass, v to counts (magnetic), hy time offset
   - hz: magnetic field 3 pole butterworth low-pass, v to counts (magnetic), hz time offset  
   - ex: electric field 5 pole butterworth low-pass, electric field 1 pole butterworth high-pass, mv per km to v per m, v per m to v, v to counts (electric), electric time offset
   - ey: electric field 5 pole butterworth low-pass, electric field 1 pole butterworth high-pass, mv per km to v per m, v per m to v, v to counts (electric), electric time offset
   - hx: magnetic field 3 pole butterworth low-pass, v to counts (magnetic), hx time offset

2. **Second Round** (calls #496-516): EXACT same filters added again!
3. **Third Round** (calls #517-525): EXACT same filters added yet again!

### Source Location:
All calls originate from the same call stack:
```
mth5.py:956 in from_experiment
→ station.py:797 in update_metadata  
→ base.py:252 in write_metadata
→ station.py:550/551 in metadata (property getter)
→ run.py:243 in metadata (property getter)
→ channel.py:750 in from_dict
```

### Root Cause Analysis:
The duplication occurs because **`channel.py:750 in from_dict`** is being called **3 times** for each channel during MTH5 creation.

This suggests the issue is either:
1. The MTH5 metadata writing process calls `from_dict` multiple times
2. The channel metadata is being processed/serialized multiple times during MTH5 creation
3. There's a loop or recursive call in the MTH5 metadata handling that processes each channel 3 times

### Pattern:
- Each filter is added exactly 3 times
- The pattern is consistent across all channels (ex, ey, hx, hy, hz)
- The call stack is identical for all additions
- Total calls: 525 (which is 6 original filters × 5 channels × 3 repetitions + some overhead)

### Next Steps:
Need to investigate why `channel.py:750 in from_dict` is called 3 times during MTH5 creation for each channel.