=============
File Readers
=============

.. contents::  :local:

Basics
--------

The file readers have been setup to be like plugins, well hacked so it is setup like plugins.  Further work needs to be done to fully set it up as Python plugins, but for now it works.  

There is a generic reader that is loaded when MTH5 is imported called `read_file`. It will pick the correct reader based on the file extension or if the extension is ambiguous the user can input the specific file type.

>>> import mth5
>>> run_obj, extra_metadata = mth5.read_file(r"/home/mt_data/mt001.bin")
>>> run_obj, extra_metadata = mth5.read_file(r"/home/mt_data/mt001.bin", file_type='nims')  

This will currently read in 3 types of files:

	=============== ========== ============= =============
	File Structure  MTH5 Key   File Types    Returns
	=============== ========== ============= =============
	NIMS            nims       [.bin, .bnn]  RunTS
	Zonge Z3D       zen        [.z3d]        MTTS
	USGS ASCII      usgs_ascii [.asc, .gzip] RunTS
	=============== ========== ============= =============

NIMS and USGS ASCII will return a :class:`mth5.timeseries.RunTS` object and Zonge Z3D returns a :class:`mth5.timeseries.MTTS` object.  The return type depends on the structure of the file.  NIMS records each channel in a single block of data, so all channels are in a single file.  Whereas, Z3D files are for a single channel.  It might make sense in to return the same data type, but for now this is the way it is.  Also returned are any extra metadata that might not belong to a channel or run.  Specifically, most files have information about location and some other metadata about the station that could be helpful in filling out metadata for the station. 

Adding Plugins
-----------------

Everyone has their own file structure and therefore there will need to be various readers for the different data formats.  If you have a data format that isn't supported adding a reader would be a welcomed contribution.  To keep things somewhat uniform here are some guidelines to add a reader.

Reader Structure
^^^^^^^^^^^^^^^^^^^

The reader should be setup with having a class that contains the metadata which is inherited to a class that holds the data.  This makes things a little easier to separate and read.  It helps if the metadata has similar names as the standards but don't have to be it just means you have to do some translation.  

It helps if you have property values of the main information, if attributes are not appropriate, that is passed onto :class:`mth5.timeseries.MTTS` or :class:`mth5.timeseries.RunTS`.

.. code-block:: python

	from mth5 import timeseries
	
    class MyFileMetadata:
		""" Read in metadata into appropriate objects """
		def __init__(self, fn):
			self.fn = fn
			self.start = None
			self.end = None
			self.sample_rate = None
			
		def read_metadata():
			""" function to read in metadata and fill attribute values """
			pass
			
	class MyFile(MyFileMetadata):
		""" inheret metadata and read data """
		def __init__(self, fn):
			self.fn = fn
			self.data = None
			
			super().__init__()
			
		@property
		def ex(self):
			""" ex convenience property """
			# if a pandas dataframe or numpy structured array
			return timeseries.MTTS('electric', 
								   data=self.data['ex'],
								   metadata={'electric': {
								    'time_period.start': self.start,
									'time_period.end': self.end,
									'sample_rate': self.sample_rate}})
									
		@property
		def extra_metadata(self):
			""" any extra metadata that does not fit in a run or channel """
			return {'station.location.latitude': self.latitude,
				    'station.location.longitude': self.longitude',
					'station.location.elevation': self.elevation}
			
		def read_my_file(self):
			""" read in data """
			# suggest reading into a data type like numpy, pandas, xarray
			# xarray is the main object used for time series data in mth5
			return RunTS([self.ex, self.ey, self.hx, self.hy, self.hx]), extra_metadata
			

	def read_my_file(fn):
		""" the helper function to read the file """
		new_obj = MyFile(fn)
		return new_obj.read_my_file()
		
Once you have come up a reader you can add it to the reader module.  You just need to add a file name and associated file types.

In the dictionary in mth5.reader 'readers' add a line like:

	"my_file": {"file_types": ["dat", "data"], "reader": my_file.read_my_file},
		
Then you can see if your reader works

>>> import mth5
>>> run, extra = mth5.read_file(r"/home/mt_data/test.dat", file_type='my_file')


 	