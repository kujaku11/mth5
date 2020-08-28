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

from mth5.io import zen
from mth5 import mth5
from mth5 import metadata

# =============================================================================
#
# =============================================================================
z3d_dir = Path(r"c:\Users\jpeacock\Documents\example_data")

z3d_list = list(z3d_dir.glob("*.z3d"))

# write some simple metadata for the survey
survey = metadata.Survey()
survey.acquired_by.author = "MT Master"
survey.archive_id = 'TST01'
survey.archive_network = "MT"
survey.name = "test"

# add station metadata from z3d files
z_obj = zen.Z3D(z3d_list[0])
z_obj.read_all_info()





