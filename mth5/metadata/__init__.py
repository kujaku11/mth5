# -*- coding: utf-8 -*-
"""
==================
metadata
==================

This module deals with metadata as defined by the MT metadata standards.
`metadata documentation 
<https://github.com/kujaku11/MTarchive/blob/tables/docs/mt_metadata_guide.pdf>`_.

There are multiple containers for each type of metadata, named appropriately.

Each container will be able to read and write:
    * dictionary
    * json
    * xml
    * csv?
    * pandas.Series
    * anything else?

Because a lot of the name words in the metadata are split by '.' there are some
issues we need to deal with.  I wrote in get and set attribute functions
to handle these types of names so the user shouldn't have to work about
splitting the names themselves.

These containers will be the building blocks for the metadata and how they are
interchanged between the HDF5 file and the user.  A lot of the metadata you
can get directly from the raw time series files, but the user will need to
input a decent amount on their own.  Dictionaries are the most fundamental
type we should be dealing with.

Each container has an attribute called _attr_dict which dictates if the
attribute is included in output objects, the data type, whether it is a
required parameter, and the style of output.  This should help down the road
with validation and keeping the data types consistent.  And if things change
you should only have to changes these dictionaries.

self._attr_dict = {'nameword':{'type': str, 'required': True, 'style': 'name'}}

Created on Sun Apr 24 20:50:41 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)
    
:license: 
    MIT


"""
### !!! DO NOT CHANGE THE ORDER !!!
from .standards.schema import Standards
from .base import Base
from .declination import Declination
from .location import Location
from .instrument import Instrument
from .fdsn import Fdsn 
from .rating import Rating
from .data_quality import DataQuality
from .citation import Citation
from .copyright import Copyright
from .person import Person
from .software import Software
from .provenance import Provenance
from .diagnostic import Diagnostic
from .battery import Battery
from .electrode import Electrode
from .timing_system import TimingSystem
from .time_period import TimePeriod
from .orientation import Orientation
from .filtered import Filtered
from .filter import Filter
from .data_logger import DataLogger
from .survey import Survey
from .station import Station
from .run import Run
from .channel import Channel
from .auxiliary import Auxiliary
from .electric import Electric
from .magnetic import Magnetic


__all__ = ["Base", "Standards", "Declination", "Location", "Instrument", "Fdsn",
           "Rating", "DataQuality", "Citation", "Copyright", "Provenance", "Person",
           "Diagnostic", "Battery", "Electrode", "TimingSystem", "TimePeriod",
           "Orientation", "Software", "Filtered", "Filter", "DataLogger", "Survey",
           "Station", "Run", "Channel", "Auxiliary", "Electric", "Magnetic" ]
