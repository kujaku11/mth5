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

A `groups_list` is provided that lists all the subgroups in the group.    




=====================
Time Series Objects
=====================

All datasets at the *Channel* level are represented by :class:`mth5.timeseries.ChannelTS` objects.  The :class:`mth5.timeseries.ChannelTS` are based on :class:`xarray.DataArray` objects.  This way memory usage is minimal because xarray is lazy and only uses what is called for.  Another benefit is that metadata can directly accompany the data.  Currently the model is that all metadata are input into a :class:`mth5.metadata.Base` object to be validated first and then the :class:`xarray.DataArray` can be updated.  This is not automated at this point so the user just needs to use the function update_xarray_metadata when metadata values are changed.  Another advantage of using xarray is that the time series data are indexed by time making it easier to align, trim, extract, sort, etc.  

All run datasets are represented by :class:`mth5.timeseries.RunTS` objects, which are based on :class:`xarray.DataSet` which is a collection of :class:`xarray.DataArray` objects.  The benefits of using xarray are that many of the methods such as aligning, indexing, sorting are already developed and are robust.  Therefore the useability is easier without more coding. 

Another reason why xarray was picked as the basis for representing the data is that it works seamlessly with other programs like Dask for parallel computing, and plotting tools like hvplot.
   




 