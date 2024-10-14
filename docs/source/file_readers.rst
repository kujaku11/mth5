=============
File Readers
=============

Basics
--------

The file readers have been setup to be like plugins, well hacked so it is setup like plugins.  Further work needs to be done to fully set it up as Python plugins, but for now it works.  


There is a generic reader that is loaded when MTH5 is imported called `read_file`. It will pick the correct reader based on the file extension or if the extension is ambiguous the user can input the specific file type.

>>> import mth5
>>> run_obj = mth5.read_file(r"/home/mt_data/mt001.bin")
>>> run_obj = mth5.read_file(r"/home/mt_data/mt001.bin", file_type='nims')  

Supported Data Types
---------------------

The following file types are supported:

	=============== ========== ============= =============
	File Structure  MTH5 Key   File Types    Returns
	=============== ========== ============= =============
	NIMS            nims       [.bin, .bnn]  RunTS
	LEMI424         lemi       [.txt]        RunTS 
	USGS ASCII      usgs_ascii [.asc, .gzip] RunTS
	Zonge Z3D       zen        [.z3d]        ChannelTS
	Phoenix         phoenix    [.td_*, .bin] ChannelTS
	=============== ========== ============= =============

As you can see, NIMS, USGS ASCII, LEMI424 will return a :class:`mth5.timeseries.RunTS` object and Zonge Z3D and Phoenix returns a :class:`mth5.timeseries.ChannelTS` object.  The return type depends on the structure of the file.  Long period instruments usually record each channel in a single block of data, so all channels are in a single file.  Whereas, broadband instruments record each channel as a single file.  It might make sense in to return the same data type, but for now this is the way it is.  Also returned are any extra metadata that might not belong to a channel or run.  Specifically, most files have information about location and some other metadata about the station that could be helpful in filling out metadata for the station. 

Collection
------------

To further make intake easier, collections classes have been developed for each data type.  For example LEMI424 files are typically 1 day long, but that does not necessarily mean that is the defines a single run.  A run may incorporate multiple days, thus multiple files.  The role of the `LEMICollection` object is to organize the files, in a logical way and assign a run ID to runs defined as a continuous block of data that may spand multiple files.  See the next section for further examples. 

Examples of Reading in Data
---------------------------------------

.. note:: In these examples the data are stored locally and will not run if you don't have the data.  We are working on setting up a repository to store data.  But if you have data you can easily swap out the directory path.

.. toctree::
    :maxdepth: 1

    ../examples/notebooks/make_mth5_from_lemi.ipynb
    ../examples/notebooks/make_mth5_from_nims.ipynb
    ../examples/notebooks/make_mth5_from_phoenix_real.ipynb
    ../examples/notebooks/make_mth5_from_z3d.ipynb


Adding Plugins
---------------

Everyone has their own file structure and therefore there will need to be various readers for the different data formats.  If you have a data format that isn't supported adding a reader would be a welcomed contribution.  To keep things somewhat uniform here are some guidelines to add a reader.

Reader Structure
^^^^^^^^^^^^^^^^^^^

The reader should be setup with having a class that contains the metadata which is inherited to a class that holds the data.  This makes things a little easier to separate and read.  It helps if the metadata has similar names as the standards but don't have to be it just means you have to do some translation.  

It helps if you have properties, if attributes are not appropriate, for important information that is passed onto :class:`mth5.timeseries.ChannelTS` or :class:`mth5.timeseries.RunTS`.

.. code-block:: python

	from mth5.timeseries import ChannelTS, RunTS

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
		def station_metadata(self):
			""" Any station metadata within the file """
			
			station_meta_dict = {}
			station_meta_dict['location.latitude'] = self.latitude
			
			return {'Station': station_meta_dict}
			
		@property
		def run_metadata(self):
			""" Any run metadata within the file """
			
			run_meta_dict = {}
			run_meta_dict['id'] = f"{self.station}a"
			
			return {'Run': run_meta_dict}
			
		@property
		def channel_metadata(self):
			""" channel metadata filled from information in the file """
			channel_meta_dict = {}
			channel_meta_dict['time_period.start'] = self.start
			channel_meta_dict['time_period.end'] = self.end
			channel_meta_dict['sample_rate'] = self.sample_rate
			
			return {'Electric': channel_meta_dict}
			
			
		@property
		def ex(self):
			""" ex convenience property """
			# if a pandas dataframe or numpy structured array
			return timeseries.ChannelTS('electric', 
								   data=self.data['ex'],
								   channel_metadata=self.channel_metadata,
								   station_metadata=self.station_metadata,
								   run_metadata=self.run_metadata)
			
		def read_my_file(self):
			""" read in data """
			# suggest reading into a data type like numpy, pandas, xarray
			# xarray is the main object used for time series data in mth5
			return RunTS([self.ex, self.ey, self.hx, self.hy, self.hx])
			

	def read_my_file(fn):
		""" the helper function to read the file """
		new_obj = MyFile(fn)
		return new_obj.read_my_file()

.. seealso:: :class:`mth5.io.zen` and :class:`mth5.io.nims` for working examples. 
			
Once you have come up a reader you can add it to the reader module.  You just need to add a file name and associated file types.

In the dictionary in mth5.reader 'readers' add a line like:

.. code-block:: python

	"my_file": {"file_types": ["dat", "data"], "reader": my_file.read_my_file},
		
Then you can see if your reader works

>>> import mth5
>>> run = mth5.read_file(r"/home/mt_data/test.dat", file_type='my_file')

Collections Structure
^^^^^^^^^^^^^^^^^^^^^^^

If you add a reader, you should also add a collection class that can sort your given file structure into runs.  This should inherit :class:`mth5.io.collection.Collection`.

.. code-block:: python

	import pandas as pd
    from mth5.io import Collection
	from mth5.io.my_file import MyFileReader
	
	class MyCollection(Collection):
		
		def __init__(self, **kwargs):
			super()__init__(self, **kwargs)
			self.file_ext = "my_file_extension"
			
		def to_dataframe(self, sample_rates, run_name_zeros=4, calibration_path=None):
			""" 
			Create a :class:`pandas.DataFrame` from my_file_type files.  This should
			be specific enough to your file structure and generic enough to plug in.
			
			This method should only read the metadata from the files, and not open
			the entire file.
			
			:param sample_rates: sample rate to get, will always be 1 for LEMI data
			defaults to [1]
			:type sample_rates: int or list, optional
			:param run_name_zeros: number of zeros to assing to the run name,
			defaults to 4
			:type run_name_zeros: int, optional
			:param calibration_path: path to calibration files, defaults to None
			:type calibration_path: string or Path, optional
			:return: Dataframe with information of each TXT file in the given
			directory.
			:rtype: :class:`pandas.DataFrame`
			
			"""
			
			entries = []
			for fn in self.get_files(self.file_ext):
				my_file_obj = MyFileReader(fn)
				n_samples = int(my_file_obj.n_samples)
				my_file_obj.read_metadata()

				entry = {}
				entry["survey"] = self.survey_metadata.id
				entry["station"] = self.station_metadata.id
				entry["run"] = None
				entry["start"] = my_file_obj.start.isoformat()
				entry["end"] = my_file_obj.end.isoformat()
				entry["channel_id"] = my_file_obj.channel_metadata.id
				entry["component"] = my_file_obj.channel_metadata.component
				entry["fn"] = fn
				entry["sample_rate"] = my_file_obj.sample_rate
				entry["file_size"] = my_file_obj.file_size
				entry["n_samples"] = my_file_obj.n_samples
				entry["sequence_number"] = 0
				entry["instrument_id"] = "MyInstrument"
				entry["calibration_fn"] = None

				entries.append(entry)

			# make pandas dataframe and set data types
			df = self._sort_df(
				self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros
			)

			return df
			
		def assign_run_names(self, df, zeros=4):
			"""
			Assign run names based on the file structure. Remember
			a run is defined as a continuously recorded block of
			data.  So if your file structure splits up files during
			a run make sure there is logic to assign the same run ID
			to those files that are in the same run.
			
			Below is an example of testing the start time of one file
			against the end time of the next file.

			Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}.

			:param df: Dataframe with the appropriate columns
			:type df: :class:`pandas.DataFrame`
			:param zeros: number of zeros in run name, defaults to 4
			:type zeros: int, optional
			:return: Dataframe with run names
			:rtype: :class:`pandas.DataFrame`

			"""
			
			count = 1
			for row in df.itertuples():
				if row.Index == 0:
					df.loc[row.Index, "run"] = f"sr1_{count:0{zeros}}"
					previous_end = row.end
				else:
					if (
						row.start - previous_end
					).total_seconds() / row.sample_rate == row.sample_rate:
						df.loc[row.Index, "run"] = f"sr1_{count:0{zeros}}"
					else:
						count += 1
						df.loc[row.Index, "run"] = f"sr1_{count:0{zeros}}"
					previous_end = row.end

			return df


Adding Client
^^^^^^^^^^^^^^^^^^^^^^^

If you add a reader, you should also add a `Client`.  This should inherit :class:`mth5.clients.base.BaseClient`.

.. code-block:: python

	class NewFileTypeClient(ClientBase):
    def __init__(
        self,
        data_path,
        save_path=None,
        mth5_filename="from_new_file_type.h5",
        **kwargs
    ):
        super().__init__(
            data_path,
            save_path=save_path,
            sample_rates=[1],
            mth5_filename=mth5_filename,
            **kwargs
        )

        self.collection = NewFileCollection(self.data_path)

    def make_mth5_from_new_file_type(self, **kwargs):
        """
        Create an MTH5 from the new file type using the newly create Collections
		object for the new file type.  

        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        runs = self.get_run_dict()

        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(self.save_path, "w")
			# here is where you put the code specific to your
			# new file type and how to get the data into an 
			# mth5.  It should be organized as runs with
			# logical names and use as much metadata as possible
			# from the data.  
            survey_group = m.add_survey(self.collection.survey_id)

            for station_id in runs.keys():
                station_group = survey_group.stations_group.add_station(
                    station_id
                )
                for run_id, run_df in runs[station_id].items():
                    run_group = station_group.add_run(run_id)
                    run_ts = read_file(run_df.fn.to_list())
                    run_ts.run_metadata.id = run_id
                    run_group.from_runts(run_ts)
                station_group.metadata.update(run_ts.station_metadata)
                station_group.write_metadata()

            # update survey metadata from input station
            survey_group.update_metadata()

        return self.save_path


Adding to MakeMTH5
^^^^^^^^^^^^^^^^^^^^^^^

If you add a reader, you should also add the client to `MakeMTH5` for convenience.  

.. code-block:: python

	@classmethod
    def from_new_filetype(
        cls,
        data_path,
        mth5_filename=None,
        save_path=None,
        **kwargs,
    ):
        """
        Doc string

        Any H5 file parameters like compression, shuffle, etc need to have a
        prefix of 'h5'. For example h5_compression='gzip'.

        >>> MakeMTH5.from_lemi424(
            data_path, 'test', 'mt01', **{'h5_compression_opts': 1}
            )


        :param data_path: Directory where data files are, could be a single
         station or a full directory of stations.
        :type data_path: str or Path
        :param mth5_filename: filename for the H5, defaults to 'from_lemi424.h5'
        :type mth5_filename: str, optional
        :param save_path: path to save H5 file to, defaults to None which will
         place the file in `data_path`
        :type save_path: str or Path, optional
        :return: Path to MTH5 file
        :rtype: Path
        """
        maker = cls(**kwargs)
        kw_dict = maker.get_h5_kwargs()

        lemi_client = NewFileClient(
            data_path,
            save_path=save_path,
            mth5_filename=mth5_filename,
            **kw_dict,
        )

        return lemi_client.make_mth5_from_new_file_type(**kwargs**)


			

