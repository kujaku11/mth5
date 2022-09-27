=============
Usage v0.1.0
=============


.. important:: This usage describes using MTH5 version 0.1.0 where `Survey` is the top level of the MTH5 file structure.  The current version is 0.2.0.  Some version 0.1.0 files have been archived and this is here for reference.


Basic Groups
^^^^^^^^^^^^^

Each MTH5 file has default groups. A 'group' is basically like a folder that can contain other groups or datasets.  These are:

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
	
Each group also has a summary table to make it easier to search and access different parts of the file. Each entry in the table will have an HDF5 reference that can be directly used to get the appropriate group or dataset without using the path. 

.. seealso:: :mod:`mth5.groups`

Example
^^^^^^^^^

.. toctree::
    :maxdepth: 1
	
    ../examples/notebooks/v0.1.0_example.ipynb




