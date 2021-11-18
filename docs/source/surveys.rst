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

    >>> # v0.1.0 does not have a surveys_group
    >>> # v0.2.0
    >>> surveys = mth5_obj.surveys_group
    >>> type(surveys)
    mth5.groups.MasterSurveysGroup

From the *surveys_group* you can add/remove/get a survey.

To add a survey::
	
	>>> new_survey = surveys.add_survey('survey_01')
	>>> print(type(new_survey))
	mth5.groups.SurveyGroup

	
To get an existing survey::

	>>> existing_survey = surveys.get_survey('MT001')
	
To remove an existing survey::
	
	>>> surveys.remove_survey('MT002')
	>>> surveys.group_list
	['MT001']

	
Summary Table 
""""""""""""""""""
This table includes all channels within the Survey.  This can be quite large and can take a long time to build.  

==================== ==================================================
Column               Description
==================== ==================================================
survey               Survey ID for channel 
station              Station ID for channel  
run                  Run ID for channel
latitude             Survey latitude (decimal degrees)
longitude            Survey longitude (decimal degrees) 
elevation            Survey elevation (meters)
component            Channel component
start                Start time for the channel (ISO format) 
end                  End time for the channel (ISO format) 
n_samples            Number of samples in the channel
sample_rate          Channel sample rate
measurement_type     Channel measurement type
azimuth              Channel azimuth (degrees from N)
tilt                 Channel tilt (degrees from horizontal)
units                Channel units 
hdf5_reference       HDF5 internal reference
==================== ==================================================
	
Survey Group
^^^^^^^^^^^^^^^^^

A single survey is contained within a :class:`mth5.groups.SurveyGroup` object, which has the appropriate metadata for a single survey.  :class:`mth5.groups.SurveyGroup` contains all the runs for that survey.    
	

Survey Metadata
"""""""""""""""""

Metadata is accessed through the `metadata` property, which is a :class:`mt_metadata.timeseries.Survey` object. 

.. code-block:: python

	>>> type(new_survey.metadata)
	mt_metadata.timeseries.Survey
	>>> new_survey.metadata
	{
		"survey": {
			"citation_dataset.doi": null,
			"citation_journal.doi": null,
			"country": null,
			"datum": null,
			"geographic_name": null,
			"id": null,
			"name": null,
			"northwest_corner.latitude": 0.0,
			"northwest_corner.longitude": 0.0,
			"project": null,
			"project_lead.email": null,
			"project_lead.organization": null,
			"release_license": "CC-0",
			"southeast_corner.latitude": 0.0,
			"southeast_corner.longitude": 0.0,
			"summary": null,
			"time_period.end_date": "1980-01-01",
			"time_period.start_date": "1980-01-01"
		}
	}

.. seealso:: :class:`mth5.groups.SurveyGroup` and :class:`mt_metadata.timeseries.Survey`
