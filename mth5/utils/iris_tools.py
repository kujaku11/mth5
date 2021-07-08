# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:08:57 2021

@author: jpeacock
"""
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

def get_station(network, station, start=None, end=None)

c = Client("IRIS")
t = UTCDateTime("2015-09-02T22:17:15.000000Z")
st = c.get_waveforms("EM", "FLC54", None, None, t, t+3600)

