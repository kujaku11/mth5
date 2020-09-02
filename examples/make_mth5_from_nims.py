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
nims_fn = Path(r"c:\Users\jpeacock\Documents\example_data\data_rgr006a.bnn")
h5_fn = Path(r"c:\Users\jpeacock\Documents\from_nims.h5")

run_ts, extra = read_file(nims_fn)

# write some simple metadata for the survey
survey = metadata.Survey()
survey.acquired_by.author = "MT Master"
survey.archive_id = "TST01"
survey.archive_network = "MT"
survey.name = "test"

# make station metadata using extra metadata from nims file
nims_station = metadata.Station()
nims_station.from_dict(structure_dict(extra))
nims_station.archive_id = "rgr006"
nims_station.channels_recorded = run_ts.metadata.channels_recorded_all
nims_station.time_period.start = run_ts.start.iso_str
nims_station.time_period.end = run_ts.end.iso_str

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
