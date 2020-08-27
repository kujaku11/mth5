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

# =============================================================================
# 
# =============================================================================
z3d_dir = Path(r"c:\Users\jpeacock\Documents\example_data")

z3d_list = list(z3d_dir.glob('*.z3d'))

