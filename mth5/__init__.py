"""Top-level package for MTH5."""
# =============================================================================
# Imports
# =============================================================================
from mth5.io.reader import read_file
from mth5.utils.mth5_logger import setup_logger, load_logging_config
from mth5.mth5 import MTH5

# =============================================================================
# Package Variables
# =============================================================================

__author__ = """Jared Peacock"""
__email__ = "jpeacock@usgs.gov"
__version__ = "0.2.4"


# =============================================================================
# Initialize Loggers
# =============================================================================


load_logging_config()
debug_logger = setup_logger(__name__, fn="mth5_debug", level="info")
debug_logger.debug("Starting MTH5 Debug Log File")

# =============================================================================
# Defualt Parameters
# =============================================================================
CHUNK_SIZE = 8196
