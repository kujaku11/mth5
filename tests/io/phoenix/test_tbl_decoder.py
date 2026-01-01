"""Test script for TBL decoder"""

from pathlib import Path

from mth5.io.phoenix.readers.mtu.read_tbl import MTUTable


tbl_file = Path("mth5/data/1690C16C.TBL")

# Test new decoder with automatic type detection
print("\n1. Decoded values (automatic type detection):")
print("-" * 70)
mtu_table = MTUTable(tbl_file.parent, tbl_file.name)
mtu_table.read_tbl()

for key in sorted(mtu_table.tbl_dict.keys()):
    value = mtu_table.tbl_dict[key]
    if isinstance(value, bytes):
        value_str = f"<bytes: {len(value)} bytes>"
    else:
        value_str = str(value)
    print(f"{key:15s}: {value_str}")
