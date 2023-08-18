# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:51:20 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from types import SimpleNamespace
import json

# =============================================================================


def read_json_to_object(fn):
    """
    read a json file directly into an object

    :param fn: DESCRIPTION
    :type fn: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    with open(fn, "r") as fid:
        obj = json.load(fid, object_hook=lambda d: SimpleNamespace(**d))
    return obj
