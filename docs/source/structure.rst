===============
MTH5 Groups
===============

Each group within an MTH5 file has a corresponding group object that provides convenience functions to interact with the HDF5 file and add, get, and remove groups or datasets to that group.  For example a station has a corresponding :class:`mth5.groups.StationGroup` object that provides functions to add, get, and remove a run.  These are meant to make it easier for the user to interact with the HDF5 file without too much trouble.

v0.1.0 Groups
---------------
* **SurveyGroup**: The master or root group of the HDF5 file
* **FiltersGroup**: Holds all filters and filter information
	
    * **ZPKGroup**: Holds pole zero filters
    * **FAPGroup**: Holds frequency look up table filters
    * **FIRGroup**: Holds finite impulse response filters
    * **CoefficientGroup**: Holds coefficient filters
    * **TimeDelayGroup**: Holds time delay filters
		
* **Reports**: Holds any reports relevant to the survey
* **StandardsGroup**: A summary of metadata standards used  
* **StationsGroup**: Holds all the stations an subsequent data
  
    * **StationGroup**: Holds a single station
	   
        * **RunGroup**: Holds a single run
   
	        * **ChannelDataset**: Holds a single channel


v0.2.0 Groups
-----------------

* **ExperimentGroup**: The master or root group of the HDF5 file
* **ReportsGroup**: Holds any reports relevant to the experiment
* **StandardsGroup**: A summary of metadata standards used  
* **SurveysGroup**: Holds the surveys in an experiment 

    * **SurveyGroup**: The master or root group of the HDF5 file
    * **FiltersGroup**: Holds all filters and filter information
	
        * **ZPKGroup**: Holds pole zero filters
        * **FAPGroup**: Holds frequency look up table filters
        * **FIRGroup**: Holds finite impulse response filters
        * **CoefficientGroup**: Holds coefficient filters
        * **TimeDelayGroup**: Holds time delay filters
		
    * **Reports**: Holds any reports relevant to the survey 
    * **StationsGroup**: Holds all the stations an subsequent data
	  
        * **StationGroup**: Holds a single station
	   
            * **RunGroup**: Holds a single run
	   
	            * **ChannelDataset**: Holds a single channel
				
.. note:: Each group contains a weak reference to the HDF5 group, this way when a file is closed there are no lingering references to a closed HDF5 file. 

Base Groups
-------------

Each group inherits :mod:`mth5.groups.base.BaseGroup` which provides basic methods for the group. This includes setting up the proper metadata and group structure.  The `__str__` and `__repr__` methods are overwritten to show the structure of the group. 

A `groups_list` property is provided that lists all the subgroups in the group.  

Group Descriptions
--------------------

.. toctree::
    :maxdepth: 1
	
    surveys	
    stations
    runs




   




 