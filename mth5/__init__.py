"""Top-level package for MTH5."""

# =============================================================================
# Imports
# =============================================================================
import sys
import numpy as np
import xarray as xr
import h5py
from loguru import logger

from mth5.io.reader import read_file
import mth5.timeseries.scipy_filters

# =============================================================================
# Package Variables
# =============================================================================

__author__ = """Jared Peacock"""
__email__ = "jpeacock@usgs.gov"
__version__ = "0.5.0"


# =============================================================================
# Initialize Loggers
# =============================================================================
config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "level": "INFO",
            "colorize": True,
            "format": "<level>{time} | {level: <3} | {name} | {function} | {message}</level>",
        },
    ],
    "extra": {"user": "someone"},
}
logger.configure(**config)
# logger.disable("mth5")

# need to set this to make sure attributes of data arrays and data sets
# are kept when doing xarray computations like merge.
xr.set_options(keep_attrs=True)

# =============================================================================
# Defualt Parameters
# =============================================================================
CHUNK_SIZE = 8196
ACCEPTABLE_FILE_TYPES = ["mth5", "MTH5", "h5", "H5"]
ACCEPTABLE_FILE_SUFFIXES = [f".{x}" for x in ACCEPTABLE_FILE_TYPES]
ACCEPTABLE_FILE_VERSIONS = ["0.1.0", "0.2.0"]
ACCEPTABLE_DATA_LEVELS = [0, 1, 2, 3]

### transfer function summary table dtype
TF_DTYPE_LIST = [
    ("station", "S30"),
    ("survey", "S50"),
    ("latitude", float),
    ("longitude", float),
    ("elevation", float),
    ("tf_id", "S30"),
    ("units", "S60"),
    ("has_impedance", bool),
    ("has_tipper", bool),
    ("has_covariance", bool),
    ("period_min", float),
    ("period_max", float),
    ("hdf5_reference", h5py.ref_dtype),
    ("station_hdf5_reference", h5py.ref_dtype),
]

TF_DTYPE = np.dtype(TF_DTYPE_LIST)

### Channel summary table dtype
CHANNEL_DTYPE_LIST = [
    ("survey", "S30"),
    ("station", "S30"),
    ("run", "S20"),
    ("latitude", float),
    ("longitude", float),
    ("elevation", float),
    ("component", "S20"),
    ("start", "S36"),
    ("end", "S36"),
    ("n_samples", np.int64),
    ("sample_rate", float),
    ("measurement_type", "S30"),
    ("azimuth", float),
    ("tilt", float),
    ("units", "S60"),
    ("has_data", bool),
    ("hdf5_reference", h5py.ref_dtype),
    ("run_hdf5_reference", h5py.ref_dtype),
    ("station_hdf5_reference", h5py.ref_dtype),
]

CHANNEL_DTYPE = np.dtype(CHANNEL_DTYPE_LIST)

### Fourier coefficient summary table dtype
FC_DTYPE_LIST = [
    ("survey", "S30"),
    ("station", "S30"),
    ("run", "S20"),
    ("decimation_level", "S44"),
    ("latitude", float),
    ("longitude", float),
    ("elevation", float),
    ("component", "S20"),
    ("start", "S36"),
    ("end", "S36"),
    ("n_samples", np.int64),
    ("sample_rate", float),
    ("measurement_type", "S30"),
    ("units", "S60"),
    ("hdf5_reference", h5py.ref_dtype),
    ("decimation_level_reference", h5py.ref_dtype),
    ("run_hdf5_reference", h5py.ref_dtype),
    ("station_hdf5_reference", h5py.ref_dtype),
]

FC_DTYPE = np.dtype(FC_DTYPE_LIST)
### Run summary table dtype

RUN_SUMMARY_DTYPE = [
    ("channel_scale_factors", float),
    ("duration", float),
    ("end", str),
    ("has_data", bool),
    ("input_channels", list),
    ("mth5_path", str),
    ("n_samples", int),
    ("output_channels", list),
    ("run", str),
    ("sample_rate", float),
    ("start", str),
    ("station", str),
    ("survey", str),
    ("run_hdf5_reference", object),
    ("station_hdf5_reference", object),
]

RUN_SUMMARY_COLUMNS = [entry[0] for entry in RUN_SUMMARY_DTYPE]

### Standards dtype
STANDARDS_DTYPE_LIST = [
    ("attribute", "S72"),
    ("type", "S15"),
    ("required", bool),
    ("style", "S72"),
    ("units", "S32"),
    ("description", "S300"),
    ("options", "S150"),
    ("alias", "S72"),
    ("example", "S72"),
    ("default", "S72"),
]
STANDARDS_DTYPE = np.dtype(STANDARDS_DTYPE_LIST)
