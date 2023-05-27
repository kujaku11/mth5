"""Top-level package for MTH5."""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import h5py

from mth5.io.reader import read_file
from mth5.utils.mth5_logger import setup_logger, load_logging_config
import mth5.timeseries.scipy_filters

# =============================================================================
# Package Variables
# =============================================================================

__author__ = """Jared Peacock"""
__email__ = "jpeacock@usgs.gov"
__version__ = "0.3.1"


# =============================================================================
# Initialize Loggers
# =============================================================================


load_logging_config()
debug_logger = setup_logger(__name__, fn="mth5_debug", level="info")
debug_logger.debug("Starting MTH5 Debug Log File")

# need to set this to make sure attributes of data arrays and data sets
# are kept when doing xarray computations like merge.
xr.set_options(keep_attrs=True)

# =============================================================================
# Defualt Parameters
# =============================================================================
CHUNK_SIZE = 8196
ACCEPTABLE_FILE_TYPES = ["mth5", "MTH5", "h5", "H5"]
ACCEPTABLE_FILE_VERSIONS = ["0.1.0", "0.2.0"]
ACCEPTABLE_DATA_LEVELS = [0, 1, 2, 3]

TF_DTYPE = np.dtype(
    [
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
)

CHANNEL_DTYPE = dtype = np.dtype(
    [
        ("survey", "S30"),
        ("station", "S30"),
        ("run", "S20"),
        ("latitude", float),
        ("longitude", float),
        ("elevation", float),
        ("component", "S20"),
        ("start", "S36"),
        ("end", "S36"),
        ("n_samples", int),
        ("sample_rate", float),
        ("measurement_type", "S30"),
        ("azimuth", float),
        ("tilt", float),
        ("units", "S60"),
        ("hdf5_reference", h5py.ref_dtype),
        ("run_hdf5_reference", h5py.ref_dtype),
        ("station_hdf5_reference", h5py.ref_dtype),
    ]
)
