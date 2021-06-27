# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:14:08 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

from pathlib import Path
from mth5 import mth5

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mt_metadata.utils import STATIONXML_FIR, STATIONXML_ELECTRIC

# from mt_metadata.utils import MT_EXPERIMENT_SINGLE_STATION
translator = XMLInventoryMTExperiment()
experiment = translator.xml_to_mt(stationxml_fn=STATIONXML_ELECTRIC)

fn = Path(r"c:\Users\jpeacock\from_electric_station_stationxml.h5")
if fn.exists():
    fn.unlink()

m = mth5.MTH5()
m.open_mth5(fn)
m.from_experiment(experiment, 0)
