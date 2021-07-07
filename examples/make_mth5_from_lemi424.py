# -*- coding: utf-8 -*-
"""
This is an example script to read in LEMI424 files for a single station.

The LEMI424 outputs files in hour blocks, so as long as the files are 
continuous all the files can be considered a single run.

The user will have to input some metadata like station name and run id.

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

from mt_metadata import timeseries as metadata
from mt_metadata.utils.mttime import MTime

# =============================================================================
# set to true if you want to interact with the mth5 object in the console
interact = False
lemi424_dir = Path(r"path/to/lemi424/files")
h5_fn = "from_lemi424.mth5"

if h5_fn.exists():
    h5_fn.unlink()
    print(f"INFO: Removed existing file {h5_fn}")

processing_start = MTime()
processing_start.now()

# write some simple metadata for the survey
survey = metadata.Survey()
survey.acquired_by.author = "MT Master"
survey.archive_id = "TST01"
survey.archive_network = "MT"
survey.name = "test"

m = mth5.MTH5()
m.open_mth5(h5_fn, "w")

# add survey metadata
survey_group = m.survey_group
survey_group.metadata.update(survey)
survey_group.write_metadata()

# initialize a station
station_group = m.add_station("mt001")

# loop over all files in a directory, should be a single station
fn_list = sorted(lemi424_dir.glob("*.TXT"))

run_ts = read_file(fn_list[0])
# make a run group from first file
run_group = station_group.add_run(
    run_ts.run_metadata.id, run_metadata=run_ts.run_metadata
)

for lemi424_fn in fn_list[1:]:

    new_run_ts = read_file(lemi424_fn)

    # add data to the run group
    channels = run_group.from_runts(run_ts)

    # validate run metadata
    run_group.validate_run_metadata()

    # update station metadata to ensure consistency
    station_group.validate_station_metadata()

survey_group.update_survey_metadata()

processing_end = MTime()
processing_end.now()

print(
    f"Making MTH5 file took {(processing_end - processing_start) // 60:02.0f}:"
    f"{(processing_end - processing_start) % 60:02.0f} minutes"
)

if not interact:
    m.close_mth5()
