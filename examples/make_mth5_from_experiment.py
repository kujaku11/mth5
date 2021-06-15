# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:07:12 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

from mt_metadata.timeseries import Experiment
from mth5.mth5 import MTH5
from mt_metadata.utils import MT_EXPERIMENT_SINGLE_STATION

mt_experiment = Experiment()
mt_experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)

mth5_obj = MTH5()
mth5_obj.open_mth5(MT_EXPERIMENT_SINGLE_STATION.parent.joinpath(r"test.h5"), mode="w")

mth5_obj.from_experiment(mt_experiment)

# run_01 = mth5_obj.get_run("REW09", "a")

# runts_object = run_01.to_runts()
