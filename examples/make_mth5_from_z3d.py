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
from mth5.utils.mttime import MTime

start = MTime()
start.now()
# =============================================================================
#
# =============================================================================
z3d_dir = Path(r"c:\Users\jpeacock\Documents\example_data")
h5_fn = Path(r"c:\Users\jpeacock\Documents\from_z3d.h5")

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

# add station metadata from z3d files
ch_list = []
for fn in list(z3d_dir.glob("*.z3d")):
    mtts_obj, extra = read_file(fn)
    extra = structure_dict(extra)
    station_metadata = metadata.Station()
    station_metadata.from_dict({'station': extra['station']})
    run_metadata = metadata.Run()
    run_metadata.from_dict({'run': extra['run']})

    station_group = m.add_station(station_metadata.id,
                                  station_metadata=station_metadata)
    
    run_id = station_group.locate_run(mtts_obj.sample_rate, mtts_obj.start)
    if run_id is None:
        run_id = station_group.make_run_name()
        
    run_group = station_group.add_run(run_id, run_metadata)
    
    ch_list.append(run_group.from_mtts(mtts_obj))
    
    # need to update the station summary table entry
    station_group.summary_table.add_row(run_group.table_entry, 
                                        station_group.summary_table.locate("id", 
                                                                           run_id))
    
end = MTime()
end.now()

print(f"Conversion to MTH5 took {end-start:.2f} seconds")
    
    
    





