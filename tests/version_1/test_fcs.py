# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:59:26 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
from mth5.mth5 import MTH5

# =============================================================================

df = pd.read_csv(
    r"c:\Users\jpeacock\OneDrive - DOI\mt\fcs\test1_dec_level_0.csv"
)
df["time"] = pd.to_datetime(df.time)
