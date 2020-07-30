

Basics
----------

.. contents::  :local:

**MTH5** is written to make read/writing an *.mth5* file easier.

.. hint:: MTH5 is comprehensively logged, therefore if any problems arise you can always check the mth5_debug.log and the mth5_error.log, which will be written to your current working directory.

Each MTH5 file has default groups. A 'group' is basically like a folder that can contain other groups or datasets.  These are:

	* **Survey**    --> The master or root group of the HDF5 file
	* **Filters**   --> Holds all filters and filter information
	* **Reports**   --> Holds any reports relevant to the survey
	* **Standards** --> A summary of metadata standards used  
	* **Stations**  --> Holds all the stations an subsequent data
	
Each group also has a summary table to make it easier to search and access different parts of the file. Each entry in the table will have an HDF5 reference that can be directly used to get the appropriate group or dataset without using the path. 


Opening and Closing Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To open a new *.mth5* file::

>>> from mth5 import mth5
>>> mth5_obj = mth5.MTH5()
>>> mth5_obj.open(r"path/to/file.mth5", mode="w")
	
To open an exiting *.mth5* file::


>>> from mth5 import mth5
>>> mth5_obj = mth5.MTH5()
>>> mth5_obj.open(r"path/to/file.mth5", mode="a")
	
.. note:: If 'w' is used for the mode, it will overwrite any file of the same name, so be careful you don't overwrite any files.  Using 'a' for the mode is safer as this will open  an existing file of the same name and will give you write privilages.

To close a file::

	>>> mth5_obj.close_mth5()
	2020-06-26T15:01:05 - mth5.mth5.MTH5.close_mth5 - INFO - Flushed and 
	closed example_02.mth5
	
.. note:: Once a MTH5 file is closed any data contained within cannot be accessed.  All groups are weakly referenced, therefore once the file closes the group can no longer access the HDF5 group and you will get a similar message as below.  This is to remove any lingering references to the HDF5 file which will be important for parallel computing.

>>> 2020-06-26T15:21:47 - mth5.groups.Station.__str__ - WARNING - MTH5 file is closed and cannot be accessed. MTH5 file is closed and cannot be accessed.

A MTH5 object is represented by the file structure and
can be displayed at anytime from the command line.

	
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
			--> Dataset: Summary
			......................
				
This file does not contain a lot of stations, but this can get verbose if there are a lot of stations and filters. If you want to check what stations are in the current file.

>>> mth5_obj.station_list
['Summary', 'MT001']
	
	
Each group has a property attribute with an appropriate container including convenience methods.  Each group has a property attribute called `group_list` that lists all groups the next level down.

.. seealso:: :mod:`mth5.groups` and :mod:`mth5.metadata` for more information.  

Metadata
^^^^^^^^^^^^^^^^^

Each group object has a container called `metadata` that holds the appropriate metadata (:mod:`mth5.metadata`) data according to the standards defined at `MT Metadata Standards <https://github.com/kujaku11/MTarchive/blob/tables/docs/mt_metadata_guide.pdf>`__. The exceptions are the HDF5 file object which has metadata that describes the file type and is not part of the standards, and the stations_group, which is just a container to hold a collection of stations. 

Input metadata will be validated against the standards and if it does not conform will throw an error. 

The basic Python type used to store metadata is a dictionary, but there are three ways to input/output the metadata, dictionary, JSON, and XML.  Many people have their own way of storing metadata so this should accommodate most everyone.  If you store your metadata as JSON or XML you will need to read in the file first and input the appropriate element to the metadata. 

Setting Attributes
"""""""""""""""""""

Metadata can be input either manually by setting the appropriate attribute::

>>> existing_station = mth5_obj.get_station('MT001')
>>> existing_station.metadata.archive_id = 'MT010'

.. hint:: Currently, if you change any `metadata` attribute you will need to mannually update the attribute in the HDF5 group: :: 

	>>> existing_station.write_metadata() 
	
Metadata Help
"""""""""""""""""

To get help with any metadata attribute you can use::


>>> existing_station.metadata.attribute_information('archive_id')
archive_id:
	alias: []
	description: station name that is archived {a-z;A-Z;0-9}
	example: MT201
	options: []
	required: True
	style: alpha numeric
	type: string
	units: None
	
If no argument is given information for all metadata attributes will be printed.

Creating New Attributes
"""""""""""""""""""""""""

If you want to add new standard attributes to the metadata you can do this through :function:`mth5.metadata.Base.add_base_attribute method`

>>> extra = {'type': str,
...          'style': 'controlled vocabulary',
...          'required': False,
...          'units': 'celsius',
...          'description': 'local temperature',
...          'alias': ['temp'],
...          'options': [ 'ambient', 'air', 'other'],
...          'example': 'ambient'}
>>> existing_station.metadata.add_base_attribute('temperature', 'ambient', extra)

Dictionary Input/Output
"""""""""""""""""""""""""

You can input a dictionary of attributes

.. note:: The dictionary must be of the form {'level': {'key': 'value'}}, where 'level' is either [ 'survey' | 'station' | 'run' | 'channel' | 'filter' ]

.. code-block:: python

	>>> meta_dict = {'station': {'archive_id': 'MT010'}}
	>>> existing_station.metadata.from_dict(meta_dict)
	>>> exiting_station.metadata.to_dict()
	{'station': OrderedDict([('acquired_by.author', None),
              ('acquired_by.comments', None),
              ('archive_id', 'MT010'),
              ('channel_layout', 'X'),
              ('channels_recorded', ['Hx', 'Hy', 'Hz', 'Ex', 'Ey']),
              ('comments', None),
              ('data_type', 'BB, LP'),
              ('geographic_name', 'Beachy Keen, FL, USA'),
              ('hdf5_reference', '<HDF5 object reference>'),
              ('id', 'FL001'),
              ('location.declination.comments',
               'Declination obtained from the instrument GNSS NMEA sequence'),
              ('location.declination.model', 'Unknown'),
              ('location.declination.value', -4.1),
              ('location.elevation', 0.0),
              ('location.latitude', 29.7203555),
              ('location.longitude', -83.4854715),
              ('mth5_type', 'Station'),
              ('orientation.method', 'compass'),
              ('orientation.reference_frame', 'geographic'),
              ('provenance.comments', None),
              ('provenance.creation_time', '2020-05-29T21:08:40+00:00'),
              ('provenance.log', None),
              ('provenance.software.author', 'Anna Kelbert, USGS'),
              ('provenance.software.name', 'mth5_metadata.m'),
              ('provenance.software.version', '2020-05-29'),
              ('provenance.submitter.author', 'Anna Kelbert, USGS'),
              ('provenance.submitter.email', 'akelbert@usgs.gov'),
              ('provenance.submitter.organization',
               'USGS Geomagnetism Program'),
              ('time_period.end', '2015-01-29T16:18:14+00:00'),
              ('time_period.start', '2015-01-08T19:49:15+00:00')])}


JSON Input/Output
"""""""""""""""""""""""""""

JSON input is as a string, therefore you will need to read the file first.

.. code-block:: python

	>>> json_string = '{"station": {"archive_id": "MT010"}}
	>>> existing_station.metadata.from_json(json_string)
	>>> print(existing_station.metadata.to_json(nested=True))	
	{
		"station": {
			"acquired_by": {
				"author": null,
				"comments": null
			},
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
			"id": "MT010",
			"location": {
				"latitude": 29.7203555,
				"longitude": -83.4854715,
				"elevation": 0.0,
				"declination": {
					"comments": "Declination obtained from the instrument GNSS NMEA sequence",
					"model": "Unknown",
					"value": -4.1
				}
			},
			"mth5_type": "Station",
			"orientation": {
				"method": "compass",
				"reference_frame": "geographic"
			},
			"provenance": {
				"creation_time": "2020-05-29T21:08:40+00:00",
				"comments": null,
				"log": null,
				"software": {
					"author": "Anna Kelbert, USGS",
					"version": "2020-05-29",
					"name": "mth5_metadata.m"
				},
				"submitter": {
					"author": "Anna Kelbert, USGS",
					"organization": "USGS Geomagnetism Program",
					"email": "akelbert@usgs.gov"
				}
			},
			"time_period": {
				"end": "2015-01-29T16:18:14+00:00",
				"start": "2015-01-08T19:49:15+00:00"
			}
		}
	}

XML Input/Output
"""""""""""""""""""""""""""

You can input as a XML element following the form previously mentioned.  If you store your metadata in XML files you will need to read the and input the appropriate element into the metadata.

.. code-block:: python

	>>> from xml.etree import cElementTree as et
	>>> root = et.Element('station')
	>>> et.SubElement(root, 'archive_id', {'text': 'MT010'})
	>>> existing_station.from_xml(root)
	>>> print(existing_station.to_xml(string=True)
	<?xml version="1.0" ?>
	<station>
		<acquired_by>
			<author>None</author>
			<comments>None</comments>
		</acquired_by>
		<archive_id>MT010</archive_id>
		<channel_layout>X</channel_layout>
		<channels_recorded>
			<item>Hx</item>
			<item>Hy</item>
			<item>Hz</item>
			<item>Ex</item>
			<item>Ey</item>
		</channels_recorded>
		<comments>None</comments>
		<data_type>BB, LP</data_type>
		<geographic_name>Beachy Keen, FL, USA</geographic_name>
		<hdf5_reference type="h5py_reference">&lt;HDF5 object reference&gt;</hdf5_reference>
		<id>FL001</id>
		<location>
			<latitude type="float" units="degrees">29.7203555</latitude>
			<longitude type="float" units="degrees">-83.4854715</longitude>
			<elevation type="float" units="degrees">0.0</elevation>
			<declination>
				<comments>Declination obtained from the instrument GNSS NMEA sequence</comments>
				<model>Unknown</model>
				<value type="float" units="degrees">-4.1</value>
			</declination>
		</location>
		<mth5_type>Station</mth5_type>
		<orientation>
			<method>compass</method>
			<reference_frame>geographic</reference_frame>
		</orientation>
		<provenance>
			<creation_time>2020-05-29T21:08:40+00:00</creation_time>
			<comments>None</comments>
			<log>None</log>
			<software>
				<author>Anna Kelbert, USGS</author>
				<version>2020-05-29</version>
				<name>mth5_metadata.m</name>
			</software>
			<submitter>
				<author>Anna Kelbert, USGS</author>
				<organization>USGS Geomagnetism Program</organization>
				<email>akelbert@usgs.gov</email>
			</submitter>
		</provenance>
		<time_period>
			<end>2015-01-29T16:18:14+00:00</end>
			<start>2015-01-08T19:49:15+00:00</start>
		</time_period>
	</station>
	
.. seealso:: :mod:`mth5.metadata` for more information.
	
