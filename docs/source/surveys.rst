There are 2 survey containers :class:`mth5.groups.MasterSurveysGroup` and :class:`mth5.groups.SurveyGroup`.  

Master Surveys Group
^^^^^^^^^^^^^^^^^^^^^^^

:class:`mth5.groups.MasterSurveysGroup` is an umbrella container that holds a collection of :class:`mth5.groups.SurveyGroup` objects and contains a summary table that summarizes all surveys within the survey.   Use :class:`mth5.groups.MasterSurveysGroup` to add/get/remove surveys. 

No metadata currently accompanies :class:`mth5.groups.MasterSurveysGroup`.   


Add/get will return a :class:`mth5.groups.SurveyGroup`.  If adding a survey that has the same name as an existing survey the :class:`mth5.groups.SurveyGroup` returned will be of the existing survey and no survey will be added.  Change the name or update the existing staiton.  If getting a survey that does not exist a :class:`mth5.utils.exceptions.MTH5Error` will be raised. 

Add/Get/Remove Survey
""""""""""""""""""""""""""

Add/get/remove surveys can be done from the :py:attr:`mth5.MTH5.surveys_group` which is a :class:`mth5.groups.MasterSurveysGroup` object.

.. code-block::

    >>> # v0.1.0
    >>> surveys = mth5_obj.surveys_group
    >>> # v0.2.0
    >>> surveys = mth5_obj.get_survey("example").surveys_group
    >>> type(surveys)
    mth5.groups.MasterSurveysGroup
    >>> surveys
    /Survey/Surveys:
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

From the *surveys_group* you can add/remove/get a survey.

To add a survey::
	
	>>> new_survey = surveys.add_survey('MT002')
	>>> print(type(new_survey))
	mth5.groups.SurveyGroup
	>>> new_survey
	/Survey/Surveys/MT002:
	====================

	
To get an existing survey::

	>>> existing_survey = surveys.get_survey('MT001')
	
To remove an existing survey::
	
	>>> surveys.remove_survey('MT002')
	>>> surveys.group_list
	['MT001']

	
Summary Table
""""""""""""""""""

==================== ==================================================
Column               Description
==================== ==================================================
archive_id           Survey archive name
start                Start time of the survey (ISO format)
end                  End time of the survey (ISO format)
components           All components measured by the survey
measurement_type     All measurement types collected by the survey 
location.latitude    Survey latitude (decimal degrees)
location.longitude   Survey longitude (decimal degrees) 
location.elevation   Survey elevation (meters)
hdf5_reference       Internal HDF5 reference
==================== ==================================================
	
Survey Group
^^^^^^^^^^^^^^^^^

A single survey is contained within a :class:`mth5.groups.SurveyGroup` object, which has the appropriate metadata for a single survey.  :class:`mth5.groups.SurveyGroup` contains all the runs for that survey.    
	
Summary Table
""""""""""""""""""

The summary table in :class:`mth5.groups.SurveyGroup` summarizes all runs for that survey.

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

Survey Metadata
"""""""""""""""""

Metadata is accessed through the `metadata` property, which is a :class:`mt_metadata.timeseries.Survey` object. 

.. code-block:: python

	>>> type(new_survey.metadata)
	mt_metadata.timeseries.Survey
	>>> new_survey.metadata
	{
		"survey": {
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
			"mth5_type": "Survey",
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

.. seealso:: :class:`mth5.groups.SurveyGroup` and :class:`mt_metadata.timeseries.Survey`
