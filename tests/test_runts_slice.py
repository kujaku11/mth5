# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 12:45:52 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from mth5.mth5 import MTH5
from mth5.timeseries import ChannelTS, RunTS


# =============================================================================

m = MTH5()
m.open_mth5(r"c:\Users\jpeacock\Documents\rslice_test.h5", "w")

sg = m.add_survey("test")
st = sg.stations_group.add_station("mt01")
r = st.add_run("001")

ch = ChannelTS("electric", None)
ch.channel_metadata.component = "ex"
ch.start = "1980-01-01T00:00:00+00:00"
ch.sample_rate = 8
ch.ts = np.arange(4096)

rts = RunTS(array_list=[ch])
r.from_runts(rts)

m.close_mth5()

m = MTH5()
m.open_mth5(r"c:\Users\jpeacock\Documents\rslice_test.h5", "r")

r = m.get_run("mt01", "001", "test")
r_slice = r.to_runts("1980-01-01T00:00:00+00:00", n_samples=256)






