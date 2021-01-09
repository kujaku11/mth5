"""Top-level package for MTH5."""
# =============================================================================
# Imports
# =============================================================================
import logging
import yaml

from mth5.io.reader import read_file
from mth5.utils.mth5_logger import get_logger, load_logging_config

# =============================================================================
# Package Variables
# =============================================================================

__author__ = """Jared Peacock"""
__email__ = "jpeacock@usgs.gov"
__version__ = "0.1.0"


# =============================================================================
# Initialize Loggers
# =============================================================================


load_logging_config()
debug_logger = get_logger(__name__, fn="mth5_debug", level="debug")
debug_logger.debug("Starting MT Metadata Debug Log File")

error_logger = get_logger("error", fn="mth5_error", level="error")