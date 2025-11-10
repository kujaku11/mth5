# -*- coding: utf-8 -*-
"""
Helper utilities for Phoenix Geophysics reader module.

Created on Tue Jun 20 15:51:20 2023

@author: jpeacock
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================


def read_json_to_object(fn: str | Path) -> SimpleNamespace:
    """
    Read a JSON file directly into a SimpleNamespace object.

    Parameters
    ----------
    fn : str or Path
        Path to the JSON file to read.

    Returns
    -------
    SimpleNamespace
        Object containing the JSON data as attributes.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    IOError
        If there's an error reading the file.

    Examples
    --------
    >>> obj = read_json_to_object("config.json")
    >>> print(obj.some_attribute)

    Notes
    -----
    This function uses json.load with an object_hook to convert
    all dictionaries to SimpleNamespace objects, allowing dot
    notation access to nested JSON properties.
    """
    with open(fn, "r") as fid:
        obj = json.load(fid, object_hook=lambda d: SimpleNamespace(**d))
    return obj
