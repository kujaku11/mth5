# -*- coding: utf-8 -*-
"""

Created on Wed Sep 30 10:20:12 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mth5.timeseries import RunTS
from obspy import read

# =============================================================================
# read seismic file
# =============================================================================
def read_miniseed(fn):
    obs_stream = read(fn)
    
    return obs_stream