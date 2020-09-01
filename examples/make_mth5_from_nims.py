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

# =============================================================================
#
# =============================================================================
nims_fn = r"c:\Users\jpeacock\Documents\example_data\data_rgr006a.bnn"

run_ts, extra = read_file(nims_fn)

# write some simple metadata for the survey
survey = metadata.Survey()
survey.acquired_by.author = "MT Master"
survey.archive_id = 'TST01'
survey.archive_network = "MT"
survey.name = "test"






