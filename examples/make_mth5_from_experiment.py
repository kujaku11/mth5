# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:07:12 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.mth5 import MTH5
from mt_metadata.utils import STATIONXML_02

translator = XMLInventoryMTExperiment()
mt_experiment = translator.xml_to_mt(stationxml_fn=STATIONXML_02)

mth5_obj = MTH5()
mth5_obj.open_mth5(r"test.h5", "w")

mth5_obj.from_experiment(mt_experiment)

run_01 = mth5_obj.get_run("REW09", "a")

runts_object = run_01.to_runts()  

 