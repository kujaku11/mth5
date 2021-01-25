# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:56:40 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
import zipfile
from mth5 import read_file
from mth5 import mth5
from mth5.utils.pathing import DATA_DIR

from mt_metadata import timeseries as metadata
from mt_metadata.utils.mttime import MTime

# =============================================================================
#
# =============================================================================


def test_make_mth5_from_nims():
    print(f"Data Directory: {DATA_DIR}")
    # set to true if you want to interact with the mth5 object in the console
    interact = False
    nims_dir = DATA_DIR.joinpath("nims")
    h5_fn = DATA_DIR.joinpath("from_nims.mth5")

    if h5_fn.exists():
        h5_fn.unlink()
        print(f"INFO: Removed existing file {h5_fn}")

    # need to unzip the data
    with zipfile.ZipFile(nims_dir.joinpath("nims.zip"), "r") as zip_ref:
        zip_ref.extractall(nims_dir)

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
    survey_group.metadata.from_dict(survey.to_dict())
    survey_group.write_metadata()

    for nims_fn in zip_ref.filelist:

        run_ts = read_file(nims_dir.joinpath(nims_fn.filename))

        # initialize a station
        station_group = m.add_station(
            run_ts.station_metadata.id, station_metadata=run_ts.station_metadata
        )

        # make a run group
        run_group = station_group.add_run(
            run_ts.run_metadata.id, run_metadata=run_ts.run_metadata
        )

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

    if not interact:
        m.close_mth5()


if __name__ == "__main__":
    test_make_mth5_from_nims()
