# -*- coding: utf-8 -*-
"""
Map an MT object to seismic StationXML as close as we can.

Add extra fields where needed.


Created on Tue May 12 14:09:28 2020

@author: jpeacock
"""

from obspy.core.util import AttribDict
from obspy.core import inventory
from mth5 import metadata


def to_inventory_channel(mt_channel):
    """
    convert an MT channel to a Obspy inventory.Channel
    
    :param mt_channel: An mt metadata.Channel object
    """

    pass
