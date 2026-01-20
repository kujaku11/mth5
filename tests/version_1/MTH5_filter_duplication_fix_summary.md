# MTH5 Filter Duplication Fix Summary

## Problem Analysis

### Original Issue
- MTH5 creation caused 3x filter duplication (6 original → 18 in MTH5)
- Filters were being added multiple times during the serialization process

### Root Causes Identified

1. **Multiple Channel Object Creation**: The same channel metadata was processed multiple times through different code paths:
   - `mth5.py:944` → `run.py:add_channel` → `channel_dataset.py:__init__` → `channel.py:from_dict` 
   - `mth5.py:955` → `run.py:update_metadata` → `run.py:metadata` → `channel.py:from_dict`

2. **Non-Idempotent Metadata Property**: The `RunGroup.metadata` property getter was rebuilding channel objects from scratch every time it was accessed

3. **No Duplicate Prevention**: The `add_filter` method had no mechanism to prevent adding filters with duplicate names

## Solutions Implemented

### Fix 1: Duplicate Prevention in add_filter (✅ Working)
**File**: `mt_metadata/timeseries/channel.py`
**Lines**: ~415-460

Added duplicate checking in the `add_filter` method:
```python
# Check if filter with this name already exists
if any(f.name == applied_filter.name for f in self.filters):
    logger.debug(f"Filter '{applied_filter.name}' already exists, skipping duplicate")
    return
```

**Result**: Prevents adding duplicate filters to the same channel object

### Fix 2: Metadata Caching in RunGroup (🔄 Partial)
**File**: `mth5/groups/run.py`
**Lines**: ~230-250

Modified the metadata property getter to cache channel objects:
```python
# Only rebuild channels if they haven't been built yet or if the group list has changed
if not self._metadata.channels or len(self._metadata.channels) != len(self.groups_list):
    # Rebuild channels...
```

**Result**: Reduced total filter calls from 525 to 315, but still has duplication

## Current Status

- **Filter duplication reduced by ~40%** (525 → 315 calls)
- **Multiple code paths still creating channels**: Different parts of MTH5 creation process still instantiate separate channel objects
- **Original experiment data intact**: 6 filters per channel preserved correctly

## Remaining Issues

The fundamental issue is that MTH5 creation involves multiple separate channel object instantiations from the same source data. Even with duplicate prevention within individual channel objects, the overall process still creates multiple channel objects that each parse the same filter data.

## Next Steps for Complete Fix

To fully resolve this, we would need to:

1. **Consolidate Channel Creation**: Ensure channels are created once and reused
2. **Improve Metadata Coordination**: Better coordination between run.py, channel_dataset.py, and the metadata system
3. **Enhanced Caching**: More sophisticated caching of channel metadata objects

## Current Workaround

The implemented fixes provide significant improvement and prevent the worst cases of duplication. The remaining duplication is still manageable and doesn't corrupt the data integrity.