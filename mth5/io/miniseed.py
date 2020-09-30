# -*- coding: utf-8 -*-
"""

Created on Wed Sep 30 10:20:12 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from mth5.timeseries import RunTS
from obspy import read as obspy_read

# =============================================================================
# read seismic file
# =============================================================================
def read_miniseed(fn):
        
    obs_stream = obspy_read(fn)
    run_obj = RunTS()
    run_obj.from_obspy_stream(obs_stream)
    
    return run_obj