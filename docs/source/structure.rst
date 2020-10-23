===============
MTH5 Structure
===============

.. contents::  :local:

Metadata Validation
---------------------

Metadata validation can be done in many different ways.  After writing most of the code I learned about JSON and XML validation, but since I was naive to those I developed a structure that seems to work.  That structure is this, all metadata key words have certain attributes that describe how the metadata value should be represented and validated.  Those are:

	* **type** - How the value should be represented based on very basic types
	
		- *string*
		- *number* (float or integer)
		- *boolean*
		
	* **required** -  A boolean (True or False) denoting whether the metadata key word required to represent the data.
	* **style** - How the value should be represented within the type.  For instance is the value a controlled string where there are only a few options, or is the value a controlled naming convention where only a 5 character alpha-numeric string is allowed.  The styles are
	
		- *Alpha Numeric* a string with alphabetic and numberic characters
		- *Free Form* a free form string
		- *Controlled Vocabulary* only certain values are allowed according to **options**
		- *Date* a date and/or time string in ISO format
		- *Number* a float or integer
		- *Boolean* the value can only be True or False 
		
	* **units** - Units of the value
	* **description** - Full description of what the metadata key is meant to convey.
	* **options** - Any options of a **Controlled Vocabulary** style.
	* **alias** - Any aliases that may represent the same metadata key.
	* **example** - An example value to inform the user.
	
Each metadata key word has these attributes and are stored in CSV files under */mth5/standards/* in files for each basic and more complex levels.  Therefore if the standards change or are updated these are the files that will need to be adjusted.  

When MTH5 is initiated these CSV files are read in to :class:`mth5.standards.Schema.Standards` as the basis to validate new values against.  :class:`mth5.standards.Schema.Standards` has property values that are :class:`mth5.standards.Schema.BaseDict` objects for each of the basic and more complex metadata structures.  The basic structures are metadata that describe a single entity, like *location* or *person* or *instrument*.  These are then collected by more complex structures like **citation** or **survey** or **station**.

Each of the aforementioned metadata attributes are used in a validation function.  For example **type** has is validated and if the input is not the specified type, the function will try to put it into standards type.  If the **type** is *integer* and the value 10.9 is given, the output of the validator will be 10.  Or if the **type** is a string and 10.9 is given the output will be "10.9".

Metadata Structure
--------------------

Each metadata class is built on :class:`mth5.metadata.Base` where the main chage is into the '_attr_dict' attribute which is a :class:`mth5.standards.Schema.BaseDict`.  The attributes of the metadata class are given, if the attribute is a basic metadata object that is specified.  The purpose of this structure is that so a user can access attributes of the metadata in a Pythonic way.  For example:

	>>> from mth5 import metadata
	>>> station = metadata.Station()
	>>> station.location.latitude = 10.9
	
The basic representation of :class:`mth5.metadata.Base` is a dictionary:

.. code-block:: python
	
	>>> from mth5 import metadata
	>>> station = metadata.Station()
	>>> station.location.latitude = 10.9
	>>> station
	{
		"station": {
			"acquired_by.author": null,
			"channels_recorded": [],
			"data_type": null,
			"geographic_name": null,
			"id": null,
			"location.declination.model": null,
			"location.declination.value": null,
			"location.elevation": 0.0,
			"location.latitude": 10.9,
			"location.longitude": 0.0,
			"orientation.method": null,
			"orientation.reference_frame": "geographic",
			"provenance.creation_time": "2020-10-23T00:32:04.780657+00:00",
			"provenance.software.author": null,
			"provenance.software.name": null,
			"provenance.software.version": null,
			"provenance.submitter.author": null,
			"provenance.submitter.email": null,
			"provenance.submitter.organization": null,
			"time_period.end": "1980-01-01T00:00:00+00:00",
			"time_period.start": "1980-01-01T00:00:00+00:00"
		}
	}
	
The metadata can be output as XML or JSON as well with functions `to_json` and `to_xml`.  JSON can be structured such that each level is parsed out:

.. code-block:: python

	>>> print(s.to_json(nested=True))
	{
		"station": {
			"acquired_by": {
				"author": null
			},
			"channels_recorded": [],
			"data_type": null,
			"geographic_name": null,
			"id": null,
			"location": {
				"latitude": 10.9,
				"longitude": 0.0,
				"elevation": 0.0,
				"declination": {
					"model": null,
					"value": null
				}
			},
			"orientation": {
				"method": null,
				"reference_frame": "geographic"
			},
			"provenance": {
				"creation_time": "2020-10-23T00:32:04.780657+00:00",
				"software": {
					"author": null,
					"version": null,
					"name": null
				},
				"submitter": {
					"author": null,
					"organization": null,
					"email": null
				}
			},
			"time_period": {
				"end": "1980-01-01T00:00:00+00:00",
				"start": "1980-01-01T00:00:00+00:00"
			}
		}
	}

Similarly, the :class:`mth5.metadata.Base` can read in an XML or JSON string.  The benefit of this is that it is flexible to what people are more familiar with.  

MTH5 Group Objects
--------------------

Each group within an MTH5 file has a corresponding group object that provides convenience functions to interact with the HDF5 file and add, get, and remove groups or datasets to that group.  For example a station has a corresponding :class:`mth5.groups.StationGroup` object that provides functions to add, get, and remove a run.  These are meant to make it easier for the user to interact with the HDF5 file without too much trouble. 

.. note:: Each group contains a weak reference to the HDF5 group, this way when a file is closed there are no lingering references to a closed HDF5 file.  

Time Series Objects
--------------------

All datasets at the *Channel* level are represented by :class:`mth5.timeseries.ChannelTS` objects.  The :class:`mth5.timeseries.ChannelTS` are based on :class:`xarray.DataArray` objects.  This way memory usage is minimal because xarray is lazy and only uses what is called for.  Another benefit is that metadata can directly accompany the data.  Currently the model is that all metadata are input into a :class:`mth5.metadata.Base` object to be validated first and then the :class:`xarray.DataArray` can be updated.  This is not automated at this point so the user just needs to use the function update_xarray_metadata when metadata values are changed.  Another advantage of using xarray is that the time series data are indexed by time making it easier to align, trim, extract, sort, etc.  

All run datasets are represented by :class:`mth5.timeseries.RunTS` objects, which are based on :class:`xarray.DataSet` which is a collection of :class:`xarray.DataArray` objects.  The benefits of using xarray are that many of the methods such as aligning, indexing, sorting are already developed and are robust.  Therefore the useability is easier without more coding. 

Another reason why xarray was picked as the basis for representing the data is that it works seamlessly with other programs like Dask for parallel computing, and plotting tools like hvplot.
   




 