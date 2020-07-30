Stations
------------------------

.. contents:: :local:

Stations are the top level for an MT sounding.  There are 2 station containers :class:`mth5.groups.MasterStationsGroup` and :class:`mth5.groups.StationGroup`.  

Master Stations Group
^^^^^^^^^^^^^^^^^^^^^^^

:class:`mth5.groups.MasterStationsGroup` is an umbrella container that holds a collection of :class:`mth5.groups.StationGroup` objects and contains a summary table that summarizes all stations within the survey.   Use :class:`mth5.groups.MasterStationsGroup` to add/get/remove stations. 

No metadata currently accompanies :class:`mth5.groups.MasterStationsGroup`. There will soon be a list of :class:`mth5.groups.StationGroup` objects for all stations.   


There are 2 ways to add/remove/get stations.  Add/get will return a :class:`mth5.groups.StationGroup`.  If adding a station that has the same name as an existing station the :class:`mth5.groups.StationGroup` returned will be of the existing station and no station will be added.  Change the name or update the existing staiton.  If getting a station that does not exist a :class:`mth5.utils.exceptions.MTH5Error` will be raised. 

1) Using `stations_group`
""""""""""""""""""""""""""

The first way to add/get/remove stations is from the :attribute:`mth5.MTH5.stations_group` which is a :class:`mth5.groups.MasterStationsGroup` object.

	>>> stations = mth5_obj.stations_group
	>>> type(stations)
	mth5.groups.MasterStationsGroup
	>>> stations
	/Survey/Stations:
	====================
		|- Group: MT001
		---------------
			|- Group: MT001a
			----------------
				--> Dataset: Ex
				.................
				--> Dataset: Ey
				.................
				--> Dataset: Hx
				.................
				--> Dataset: Hy
				.................
				--> Dataset: Hz
				.................
				--> Dataset: Summary
				......................

From the *stations_group* you can add/remove/get a station.

To add a station::
	
	>>> new_station = stations.add_station('MT002')
	>>> print(type(new_station))
	mth5.groups.StationGroup
	>>> new_station
	/Survey/Stations/MT002:
	====================
	--> Dataset: Summary
	......................
	
To get an existing station::

	>>> existing_station = stations.get_station('MT001')
	
To remove an existing station::
	
	>>> stations.remove_station('MT002')
	>>> stations.group_list
	['Summary', 'MT001']

2) Using Covnenience methods
"""""""""""""""""""""""""""""""

The second way to add/remove/get stations is from the convenience functions in :class:`mth5.MTH5`.  These use the same methods as the :class:`mth5.groups.MasterStationsGroup` but can be accessed directly.

To add a station::

	>>> new_station = mth5_obj.add_station('MT002')
	>>> mth5_obj
	/:
	====================
		|- Group: Survey
		----------------
			|- Group: Filters
			-----------------
				--> Dataset: Summary
				......................
			|- Group: Reports
			-----------------
				--> Dataset: Summary
				......................
			|- Group: Standards
			-------------------
				--> Dataset: Summary
				......................
			|- Group: Stations
			------------------
				|- Group: MT001
				---------------
					--> Dataset: Summary
					......................
				|- Group: MT002
				---------------
					--> Dataset: Summary
					......................
				--> Dataset: Summary
				......................

To get an existing station::

	>>> existing_station = mth5_obj.get_station('MT002')
	
To remove an existing station::

	>>> mth5_obj.remove_station('MT002')
	
Summary Table
""""""""""""""""""

==================== ==================================================
Column               Description
==================== ==================================================
archive_id           Station archive name
start                Start time of the station (ISO format)
end                  End time of the station (ISO format)
components           All components measured by the station
measurement_type     All measurement types collected by the station 
location.latitude    Station latitude (decimal degrees)
location.longitude   Station longitude (decimal degrees) 
location.elevation   Station elevation (meters)
hdf5_reference       Internal HDF5 reference
==================== ==================================================
	
Station Group
^^^^^^^^^^^^^^^^^

A single station is contained within a :class:`mth5.groups.StationGroup` object, which has the appropriate metadata for a single station.  :class:`mth5.groups.StationGroup` contains all the runs for that station.    
	
Summary Table
""""""""""""""""""

The summary table in :class:`mth5.groups.StationGroup` summarizes all runs for that station.

==================== ==================================================
Column               Description
==================== ==================================================
id                   Run ID 
start                Start time of the run (ISO format)
end                  End time of the run (ISO format) 
components           All components measured for that run
measurement_type     Type of measurement for that run
sample_rate          Sample rate of the run (samples/second)
hdf5_reference       Internal HDF5 reference
==================== ==================================================

Metadata
"""""""""""""""

Metadata is accessed through the `metadata` property, which is a :class:`mth5.metadata.Station` object. 

.. code-block:: python

	>>> type(new_station.metadata)
	mth5.metadata.Station
	>>> new_station.metadata
	{
		"station": {
			"acquired_by.author": null,
			"acquired_by.comments": null,
			"archive_id": "FL001",
			"channel_layout": "X",
			"channels_recorded": [
				"Hx",
				"Hy",
				"Hz",
				"Ex",
				"Ey"
			],
			"comments": null,
			"data_type": "BB, LP",
			"geographic_name": "Beachy Keen, FL, USA",
			"hdf5_reference": "<HDF5 object reference>",
			"id": "FL001",
			"location.declination.comments": "Declination obtained from the instrument GNSS NMEA sequence",
			"location.declination.model": "Unknown",
			"location.declination.value": -4.1,
			"location.elevation": 0.0,
			"location.latitude": 29.7203555,
			"location.longitude": -83.4854715,
			"mth5_type": "Station",
			"orientation.method": "compass",
			"orientation.reference_frame": "geographic",
			"provenance.comments": null,
			"provenance.creation_time": "2020-05-29T21:08:40+00:00",
			"provenance.log": null,
			"provenance.software.author": "Anna Kelbert, USGS",
			"provenance.software.name": "mth5_metadata.m",
			"provenance.software.version": "2020-05-29",
			"provenance.submitter.author": "Anna Kelbert, USGS",
			"provenance.submitter.email": "akelbert@usgs.gov",
			"provenance.submitter.organization": "USGS Geomagnetism Program",
			"time_period.end": "2015-01-29T16:18:14+00:00",
			"time_period.start": "2015-01-08T19:49:15+00:00"
		}
	}

.. seealso:: :class:`mth5.groups.StationGroup`
