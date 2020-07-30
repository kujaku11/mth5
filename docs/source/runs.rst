Runs
--------------

.. contents:: :local:

A run is a collection of channels that recorded at similar start and end times at the same sample rate for a given station.  A run is contained within a :class:`mth5.groups.RunGroup` object.  A run is the next level down from a station.  

The main way to add/remove/get a run object is through a :class:`mth5.groups.StationGroup` object

Accessing through StationGroup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can get a :class:`mth5.groups.StationGroup` using either method in the previous section.

>>> new_station = mth5_obj.add_station('MT003')

or 

>>> new_station = mth5_obj.stations_group.add_station('MT003')

Add Run
"""""""""""

>>> # if you don't already have a run name one can be assigned based on existing runs
>>> new_run_name = new_station.make_run_name()
>>> new_run = new_station.add_run(new_run_name)

Or 

>>> new_run = mth5_obj.add_run('MT003', 'MT003a')

Get Run
"""""""""""

Similar methods for get/remove a run

>>> existing_run = new_station.get_run('MT003a')

or

>>> existing_run = mth5_obj.get_run('MT003', 'MT003a')

Remove Run
"""""""""""""""

>>> new_station.remove_run('MT003a')

or 

>>> mth5_obj.remove_run('MT003', 'MT003a')

Summary Table
^^^^^^^^^^^^^^^^^^^^

The summary table summarizes all channels for that run.

==================== ==================================================
Column               Description
==================== ==================================================
component            Component name
start                Start time of the channel (ISO format) 
end                  End time of the channel (ISO format0
n_samples            Number of samples for the channel
measurement_type     Measuremnt type of the channel
units                Units of the channel data 
hdf5_reference       HDF5 internal reference
==================== ==================================================

Metadata
^^^^^^^^^^^^^^^

Metadata is accessed through the `metadata` property, which is a :class:`mth5.metadata.Run` object.

.. code-block:: python

	>>> type(new_run)
	mth5.metadata.Run
	>>> new_run.metadata
	{
		"run": {
			"acquired_by.author": "BB",
			"acquired_by.comments": "it's cold in florida",
			"channels_recorded_auxiliary": null,
			"channels_recorded_electric": null,
			"channels_recorded_magnetic": null,
			"comments": null,
			"data_logger.firmware.author": "Barry Narod",
			"data_logger.firmware.name": null,
			"data_logger.firmware.version": null,
			"data_logger.id": "1305-1",
			"data_logger.manufacturer": "Barry Narod",
			"data_logger.model": "NIMS",
			"data_logger.power_source.comments": "voltage measurements not recorded",
			"data_logger.power_source.id": null,
			"data_logger.power_source.type": "battery",
			"data_logger.power_source.voltage.end": null,
			"data_logger.power_source.voltage.start": null,
			"data_logger.timing_system.comments": null,
			"data_logger.timing_system.drift": 0.0,
			"data_logger.timing_system.type": "GPS",
			"data_logger.timing_system.uncertainty": 1.0,
			"data_logger.type": null,
			"data_type": "BB, LP",
			"hdf5_reference": "<HDF5 object reference>",
			"id": "MT003a",
			"metadata_by.author": "Anna Kelbert; Paul Bedrosian",
			"metadata_by.comments": "Paul Bedrosian: Ey, electrode dug up",
			"mth5_type": "Run",
			"provenance.comments": null,
			"provenance.log": null,
			"sample_rate": 8.0,
			"time_period.end": "2015-01-19T14:54:54+00:00",
			"time_period.start": "2015-01-08T19:49:15+00:00"
		}
	}

.. seealso:: :class:`mth5.groups.RunGroup` and :class:`mth5.metadata.Run` for more information.
