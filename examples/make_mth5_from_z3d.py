# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:56:40 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mth5 import read_file
from mth5 import mth5
from mth5 import metadata
from mth5.utils.helpers import structure_dict

# =============================================================================
#
# =============================================================================
z3d_dir = Path(r"c:\Users\jpeacock\Documents\example_data")
h5_fn = Path(r"c:\Users\jpeacock\Documents\from_z3d.h5")

z3d_list = list(z3d_dir.glob("*.z3d"))

# write some simple metadata for the survey
survey = metadata.Survey()
survey.acquired_by.author = "MT Master"
survey.archive_id = 'TST01'
survey.archive_network = "MT"
survey.name = "test"

# open mth5 file
m = mth5.MTH5(h5_fn)
m.open_mth5()

# add survey metadata
m.survey_group.metadata.from_dict(survey.to_dict())

# initialize a station
station_group = m.add_station(nims_station.archive_id, station_metadata=nims_station)

# make a run group
run_group = station_group.add_run(run_ts.metadata.id, run_metadata=run_ts.metadata)

# add data to the run group
channels = run_group.from_runts(run_ts)


# add station metadata from z3d files
for fn in z3d_list:
    mtts_obj, extra = read_file(fn)
    





