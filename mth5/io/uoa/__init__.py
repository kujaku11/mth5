"""
University of Adelaide (UoA) Magnetotelluric Instrument Readers

Created on Fri Nov 7 12:22:00 2025

@author: bkay

Instruments Supported
---------------------

1. **Earth Data Logger PR6-24** (read_uoa)
   - 6-channel broadband/long-period MT system
   - ASCII format: one file per channel containing microvolts
   - Hardware: Bz voltage divider (fluxgate only; 15kΩ/10kΩ), E-field terminal box (×10 gain)
   - Sensors: LEMI-120 induction coils or Bartington Mag-03 fluxgates

2. **Orange Box** (read_orange)
   - Legacy 8-channel long-period MT system (UoA/Flinders)
   - Binary format: 18 bytes per sample, 8 channels
   - Sensors: Bartington fluxgates, non-polarizing electrodes
   - Used in: ~2000 to ~2014


Hardware Calibration Notes
--------------------------

**PR6-24 (EDL)**:
- Bz channel: 15kΩ/10kΩ voltage divider (Bartington mode only)
- Ex, Ey channels: ×10 gain from terminal box (both modes)
- ADC: 24-bit, ±8.388V range, ~1 μV per count

**Orange Box**:
- Magnetic: 24-bit ADC, ±70,000 nT full-scale
- Electric: 24-bit ADC, ±100,000 μV full-scale (dipole-normalized)
- By channel: Inverted by hardware convention
- Ex, Ey channels: Inverted by hardware convention


"""

from .pr624 import read_uoa, UoAReader
from .orange import read_orange, OrangeReader

__all__ = ["read_uoa", "UoAReader", "read_orange", "OrangeReader"]
