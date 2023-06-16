# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:21:35 2023

@author: jpeacock

Calibrations can come in json files.  the JSON file includes filters
for all lowpass filters, so you need to match the lowpass filter used in the 
setup with the lowpass filter.  Then you need to add the dipole length and
sensor calibrations.
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import json

from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

# =============================================================================

cal_fn = Path(
    r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\calibrations\10621_647A2F41.rxcal.json"
)

with open(cal_fn, "r") as fid:
    jd = json.load(fid)
