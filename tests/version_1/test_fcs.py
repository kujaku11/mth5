# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:59:26 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import numpy as np
from mth5.mth5 import MTH5

from mth5.helpers import to_numpy_type

# =============================================================================

df = pd.read_csv(
    r"c:\Users\jpeacock\OneDrive - DOI\mt\fcs\test1_dec_level_0.csv"
)
df["time"] = pd.to_datetime(df.time).astype(str)


def cast_df_to_numpy(df):
    b = np.zeros(
        df.shape[0],
        dtype=[
            ("time", "S32"),
            ("frequency", float),
            ("coefficient", complex),
        ],
    )
    b["frequency"] = df["frequency"]
    b["time"] = df["time"]
    b["coefficient"] = df["ex"]

    return b


n = cast_df_to_numpy(df[["time", "frequency", "ex"]])

with MTH5() as m:
    m.file_version = "0.1.0"
    m.open_mth5(r"c:\Users\jpeacock\OneDrive - DOI\mt\fcs\fc_test.h5")
    sg = m.add_station("mt01")
    fcg = sg.fourier_coefficients_group.add_fc_group("default")
    dl = fcg.add_decimation_level("1")
    ch = dl.add_channel("ex", fc_data=n)
