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
from mth5.utils.mttime import MTime

# =============================================================================
#
# =============================================================================
# nims_fn = Path(r"c:\Users\jpeacock\Documents\example_data\data_rgr006a.bnn")
nims_dir = Path(r"c:\Users\jpeacock\Documents\example_data\mnp")
h5_fn = Path(r"c:\Users\jpeacock\Documents\from_nims.h5")

processing_start = MTime()
processing_start.now()

# write some simple metadata for the survey
survey = metadata.Survey()
survey.acquired_by.author = "MT Master"
survey.archive_id = "TST01"
survey.archive_network = "MT"
survey.name = "test"

m = mth5.MTH5(h5_fn)
m.open_mth5()

# add survey metadata
survey_group = m.survey_group
survey_group.metadata.from_dict(survey.to_dict())
survey_group.write_metadata()

for nims_fn in nims_dir.iterdir():

    run_ts = read_file(nims_fn)

    # initialize a station
    station_group = m.add_station(
        run_ts.station_metadata.fdsn.id, station_metadata=run_ts.station_metadata
    )

    # make a run group
    run_group = station_group.add_run(run_ts.metadata.id, run_metadata=run_ts.metadata)

    # add data to the run group
    channels = run_group.from_runts(run_ts)

    # validate run metadata
    run_group.validate_run_metadata()

    # need to update the station summary table entry
    station_group.summary_table.add_row(
        run_group.table_entry,
        station_group.summary_table.locate("id", run_group.metadata.id),
    )

    # update station metadata to ensure consistency
    station_group.validate_station_metadata()

survey_group.update_survey_metadata()

processing_end = MTime()
processing_end.now()

print(
    f"Making MTH5 file took {(processing_end - processing_start) // 60:02.0f}:"
    f"{(processing_end - processing_start) % 60:02.0f} minutes"
)
