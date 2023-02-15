# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:30:12 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import h5py
import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================

fcs = [
    {
        "time": "2021-01-01T00:00:00",
        "frequency": 1,
        "fc": 1 + 2j,
        "decimation_level": 0,
    }
]


h = h5py.File(r"c:\Users\jpeacock\fc_test.h5", mode="a")

# Think about atomic unit being either a single window or a collection of
# windows under a decimation level.
