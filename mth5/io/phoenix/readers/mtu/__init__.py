"""
MTU_utils - Utility to read the Canadian Phoenix MTU-5A instrument
time series binary files in Matlab (and Python)

A bunch of simple scripts to read the legacy Phoenix MTU-5A binary format
files ... including the time series (.TSN) and table (.TBL) formats.

Original files by:

DONG Hao
donghao@cugb.edu.cn
China University of Geosciences, Beijing

Updated and adapted to Python by:
Peacock, J.R. (2025-12-31)
"""

from .mtu_table import MTUTable

__all__ = ["MTUTable"]
# __version__ = "1.0.0"
