# -*- coding: utf-8 -*-
"""
Root logging

Created on Mon May 18 15:34:05 2020

@author: jpeacock
"""

import logging
import logging.config
import os
from pathlib import Path
import yaml

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
CONF_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
CONF_FILE = Path.joinpath(CONF_PATH, "logging_config.yaml")

if not CONF_FILE.exists():
    CONF_FILE = None


class MTH5Logger:
    @staticmethod
    def load_config():
        config_file = Path(CONF_FILE)
        with open(config_file, "r") as fid:
            config_dict = yaml.safe_load(fid)
        logging.config.dictConfig(config_dict)

    @staticmethod
    def get_logger(logger_name, fn=None):
        logger = logging.getLogger(logger_name)
        logger.addHandler(logging.NullHandler())
        if fn is not None:
            fn_handler = logging.FileHandler(fn, mode="a")
            fn_handler.setFormatter(FORMATTER)
            fn_handler.setLevel(logging.DEBUG)
            logger.addHandler(fn_handler)

        return logger
