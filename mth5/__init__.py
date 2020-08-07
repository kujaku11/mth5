"""Top-level package for MTH5."""

import logging
import yaml
from mth5.utils.mth5logger import CONF_FILE

__author__ = """Jared Peacock"""
__email__ = "jpeacock@usgs.gov"
__version__ = "0.1.0"

# configure log file
config_file = CONF_FILE
if config_file is not None:
    with open(config_file, "r") as fid:
        config_dict = yaml.safe_load(fid)
    logging.config.dictConfig(config_dict)

    # open root logger
    logger = logging.getLogger(__name__)

    # make sure everything is working
    logger.info("Started MTH5")
    logger.debug("Beginning debug mode for MTH5")
    debug_fn = logger.root.handlers[1].baseFilename
    error_fn = logger.root.handlers[2].baseFilename

    logger.info("Debug Log file can be found at {0}".format(debug_fn))
    logger.info("Error Log file can be found at {0}".format(error_fn))

else:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.basicConfig(filename="mth5_debug.log", format=formatter, level=logging.INFO)
    st = logging.StreamHandler()
    st.setFormatter(formatter)
    st.setLevel(logging.INFO)

    # open root logger
    logger = logging.getLogger(__name__)
    logger.addHandler(st)

    # make sure everything is working
    logger.info("Started MTH5")
    logger.debug("Beginning debug mode for MTH5")
    handler = logger.root.handlers[1]
    logger.info("Log file can be found at {0}".format(handler.baseFilename))
