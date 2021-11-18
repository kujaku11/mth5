Make an MTH5 from FDSN Data Management Center
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data Availability
~~~~~~~~~~~~~~~~~~~

As IRIS will be one of the key data management centers for MT data, we provide an example of how to pull data from an FDSN client as a StationXML and miniseed and create an MTH5.  First, you need to know what data you want then you can build a request for those data.  There are mulitple tools provided by IRIS to search for data availability. Check out `IRIS Data Availability <https://ds.iris.edu/ds/newsletter/vol19/no2/478/introducing-the-iris-data-availability-webservice-quickly-find-what-data-is-in-the-archive/>`_ for more information.  

Once you have your found your data availability you can then create a query request by either making a :class:`pandas.DataFrame` or a comma-separated-value (CSV) file.  

The columns need to be:

================= ============================================
Column Name       Description
================= ============================================	
network           `FDSN network code <http://docs.fdsn.org/projects/source-identifiers/en/v1.0/network-codes.html>`_ 
station           FDSN station code 'location'        FDSN location code, usually blank
channel           `FDSN channel code <http://docs.fdsn.org/projects/source-identifiers/en/v1.0/channel-codes.html>`_ 
start             ISO date time (YYYY-MM-DDThh:mm:ss)
end               ISO date time (YYYY-MM-DDThh:mm:ss) 
================= ============================================

.. note:: Since we are currently using Obspy to request data and metadata, network, station, and channel all need to be input, and there should be a row for each channel requested.

And example CSV file (example.csv) would look like:

.. code-block:: python

    network,station,location,channel,start,end
    ZU,CAS04,,LQE,2020-06-02T19:00:00,2020-07-13T19:00:00
    ZU,CAS04,,LQN,2020-06-02T19:00:00,2020-07-13T19:00:00
    ZU,CAS04,,LFE,2020-06-02T19:00:00,2020-07-13T19:00:00
    ZU,CAS04,,LFN,2020-06-02T19:00:00,2020-07-13T19:00:00
    ZU,CAS04,,LFZ,2020-06-02T19:00:00,2020-07-13T19:00:00
    ZU,NVR08,,LQE,2020-06-02T19:00:00,2020-07-13T19:00:00
    ZU,NVR08,,LQN,2020-06-02T19:00:00,2020-07-13T19:00:00
    ZU,NVR08,,LFE,2020-06-02T19:00:00,2020-07-13T19:00:00
    ZU,NVR08,,LFN,2020-06-02T19:00:00,2020-07-13T19:00:00
    ZU,NVR08,,LFZ,2020-06-02T19:00:00,2020-07-13T19:00:00

or if you want to do this in a script or notebook:

.. code-block:: python
    
	import pandas as pd
	
	ZUCAS04LQ1 = [
            "ZU",
            "CAS04",
            "",
            "LQE",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        ZUCAS04LQ2 = [
            "ZU",
            "CAS04",
            "",
            "LQN",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        ZUCAS04BF1 = [
            "ZU",
            "CAS04",
            "",
            "LFE",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        ZUCAS04BF2 = [
            "ZU",
            "CAS04",
            "",
            "LFN",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        ZUCAS04BF3 = [
            "ZU",
            "CAS04",
            "",
            "LFZ",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        ZUNRV08LQ1 = [
            "ZU",
            "NVR08",
            "",
            "LQE",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        ZUNRV08LQ2 = [
            "ZU",
            "NVR08",
            "",
            "LQN",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        ZUNRV08BF1 = [
            "ZU",
            "NVR08",
            "",
            "LFE",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        ZUNRV08BF2 = [
            "ZU",
            "NVR08",
            "",
            "LFN",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        ZUNRV08BF3 = [
            "ZU",
            "NVR08",
            "",
            "LFZ",
            "2020-06-02T19:00:00",
            "2020-07-13T19:00:00",
        ]
        metadata_list = [
            ZUCAS04LQ1,
            ZUCAS04LQ2,
            ZUCAS04BF1,
            ZUCAS04BF2,
            ZUCAS04BF3,
            ZUNRV08LQ1,
            ZUNRV08LQ2,
            ZUNRV08BF1,
            ZUNRV08BF2,
            ZUNRV08BF3,
        ]
		
        df = pd.DataFrame(
            metadata_list, 
            columns=[
                "network",
                "station",
                "location",
				"channel",
				"start",
				"end"
				]
			)

Get Metadata Only
~~~~~~~~~~~~~~~~~~
	
Once you have your query request you might want to make sure that it is what you are looking for before you download data.  Here is an example on how to do that. 

.. code-block:: python

    from mth5.clients.make_mth5 import MakeMTH5

    st = MTime(get_now_utc())
    make_mth5_object = MakeMTH5()

    # get only metadata from the FDSN IRIS DMC
    inventory, streams = make_mth5.get_inventory_from_df("example.csv", "iris", data=False)
	
From here you can interogate the inventory.  This is an :class:`obspy.Inventory` object.

.. code-block:: python
   
    print(inventory)
	
	Inventory created at 2021-10-26T20:12:26.284257Z
	Created by: ObsPy 1.2.2
		    https://www.obspy.org
	Sending institution: MTH5
	Contains:
		Networks (1):
			ZU
		Stations (2):
			ZU.CAS04 (Corral Hollow, CA, USA)
			ZU.NVR08 (Rhodes Salt Marsh, NV, USA)
		Channels (10):
			ZU.CAS04..LFZ, ZU.CAS04..LFN, ZU.CAS04..LFE, ZU.CAS04..LQN, 
			ZU.CAS04..LQE, ZU.NVR08..LFZ, ZU.NVR08..LFN, ZU.NVR08..LFE, 
			ZU.NVR08..LQN, ZU.NVR08..LQE
			
This looks correct from what was requested.  If it does not look correct have a look at your inputs. 

Build MTH5 File v0.1.0
~~~~~~~~~~~~~~~~~~~~~~~~

Now we can request data and build an MTH5 file.  

.. code-block:: python

    st = MTime(get_now_utc())
    make_mth5 = MakeMTH5()
	make_mth5.mth5_version = '0.1.0'
 
	# This will build and MTH5 and allow the user to interact with
	# the MTH5 from Python.  If you just want to write a file
	# set 'interact' to False, or don't input 'interact' by default is False
    mth5_object = make_mth5.make_mth5_from_fdsnclient("example.csv", interact=True)

    et = MTime(get_now_utc())

    print(f"Took {(int(et - st) // 60)}:{(et - st) % 60:05.2f} minutes")

Now we can interogate the created MTH5 file

Check the file name

>>> mth5_object.filename
WindowsPath('ZU_CAS04_NVR08.h5')

Have a look at the contents

>>> mth5_object
/:
====================
    |- Group: Survey
    ----------------
        |- Group: Filters
        -----------------
            |- Group: coefficient
            ---------------------
                |- Group: v to counts (electric)
                --------------------------------
                |- Group: v to counts (magnetic)
                --------------------------------
            |- Group: fap
            -------------
            |- Group: fir
            -------------
            |- Group: time_delay
            --------------------
                |- Group: electric time offset
                ------------------------------
                |- Group: hx time offset
                ------------------------
                |- Group: hy time offset
                ------------------------
                |- Group: hz time offset
                ------------------------
            |- Group: zpk
            -------------
                |- Group: electric field 1 pole butterworth high-pass
                -----------------------------------------------------
                    --> Dataset: poles
                    ....................
                    --> Dataset: zeros
                    ....................
                |- Group: electric field 5 pole butterworth low-pass
                ----------------------------------------------------
                    --> Dataset: poles
                    ....................
                    --> Dataset: zeros
                    ....................
                |- Group: magnetic field 3 pole butterworth low-pass
                ----------------------------------------------------
                    --> Dataset: poles
                    ....................
                    --> Dataset: zeros
                    ....................
                |- Group: mv per km to v per m
                ------------------------------
                    --> Dataset: poles
                    ....................
                    --> Dataset: zeros
                    ....................
                |- Group: v per m to v
                ----------------------
                    --> Dataset: poles
                    ....................
                    --> Dataset: zeros
                    ....................
        |- Group: Reports
        -----------------
        |- Group: Standards
        -------------------
            --> Dataset: summary
            ......................
        |- Group: Stations
        ------------------
            |- Group: CAS04
            ---------------
                |- Group: a
                -----------
                    --> Dataset: ex
                    .................
                    --> Dataset: ey
                    .................
                    --> Dataset: hx
                    .................
                    --> Dataset: hy
                    .................
                    --> Dataset: hz
                    .................
                |- Group: b
                -----------
                    --> Dataset: ex
                    .................
                    --> Dataset: ey
                    .................
                    --> Dataset: hx
                    .................
                    --> Dataset: hy
                    .................
                    --> Dataset: hz
                    .................
                |- Group: c
                -----------
                    --> Dataset: ex
                    .................
                    --> Dataset: ey
                    .................
                    --> Dataset: hx
                    .................
                    --> Dataset: hy
                    .................
                    --> Dataset: hz
                    .................
                |- Group: d
                -----------
                    --> Dataset: ex
                    .................
                    --> Dataset: ey
                    .................
                    --> Dataset: hx
                    .................
                    --> Dataset: hy
                    .................
                    --> Dataset: hz
                    .................
            |- Group: NVR08
            ---------------
                |- Group: a
                -----------
                    --> Dataset: ex
                    .................
                    --> Dataset: ey
                    .................
                    --> Dataset: hx
                    .................
                    --> Dataset: hy
                    .................
                    --> Dataset: hz
                    .................
                |- Group: b
                -----------
                    --> Dataset: ex
                    .................
                    --> Dataset: ey
                    .................
                    --> Dataset: hx
                    .................
                    --> Dataset: hy
                    .................
                    --> Dataset: hz
                    .................
                |- Group: c
                -----------
                    --> Dataset: ex
                    .................
                    --> Dataset: ey
                    .................
                    --> Dataset: hx
                    .................
                    --> Dataset: hy
                    .................
                    --> Dataset: hz
                    .................	

Have a look at a single channel

>>> ex = mth5_object.get_channel("CAS04", "a", "ex")
>>> ex
Channel Electric:
-------------------
	component:        ex
	data type:        electric
	data format:      int32
	data shape:       (11267,)
	start:            2020-06-02T19:00:00+00:00
	end:              2020-06-02T22:07:46+00:00
	sample rate:      1.0
	
>>> ex.metadata
{
    "electric": {
        "channel_number": null,
        "comments": "run_ids: [d,c,b,a]",
        "component": "ex",
        "data_quality.rating.value": 0,
        "dipole_length": 92.0,
        "filter.applied": [
            false,
            false,
            false,
            false,
            false,
            false
        ],
        "filter.name": [
            "electric field 5 pole butterworth low-pass",
            "electric field 1 pole butterworth high-pass",
            "mv per km to v per m",
            "v per m to v",
            "v to counts (electric)",
            "electric time offset"
        ],
        "hdf5_reference": "<HDF5 object reference>",
        "measurement_azimuth": 13.2,
        "measurement_tilt": 0.0,
        "mth5_type": "Electric",
        "negative.elevation": 329.4,
        "negative.id": "200406D",
        "negative.latitude": 37.633351,
        "negative.longitude": -121.468382,
        "negative.manufacturer": "Oregon State University",
        "negative.model": "Pb-PbCl2 kaolin gel Petiau 2 chamber type",
        "negative.type": "electrode",
        "positive.elevation": 329.4,
        "positive.id": "200406B",
        "positive.latitude": 37.633351,
        "positive.longitude": -121.468382,
        "positive.manufacturer": "Oregon State University",
        "positive.model": "Pb-PbCl2 kaolin gel Petiau 2 chamber type",
        "positive.type": "electrode",
        "sample_rate": 1.0,
        "time_period.end": "2020-06-02T22:07:46+00:00",
        "time_period.start": "2020-06-02T19:00:00+00:00",
        "type": "electric",
        "units": "counts"
    }
}

Make sure there is data

>>> ex.hdf5_dataset[0:20]
array([144812, 144378, 146214, 144470, 145275, 138457, 140016, 139721,
       134734, 133218, 134701, 135473, 131894, 130241, 132098, 132902,
       129276, 128781, 132266, 133412])
	   
.. important:: Make sure to close the MTH5 file

>>> mth5_object.close_mth5()
2021-10-26 16:06:34,097 [line 569] mth5.mth5.MTH5.close_mth5 - INFO: Flushing and closing ZU_CAS04_NVR08.h5

Build MTH5 File v0.2.0
~~~~~~~~~~~~~~~~~~~~~~~~

Now we can request data and build an MTH5 file.  

.. code-block:: python

    st = MTime(get_now_utc())
    make_mth5 = MakeMTH5()
	make_mth5.mth5_version = '0.2.0'
 
	# This will build and MTH5 and allow the user to interact with
	# the MTH5 from Python.  If you just want to write a file
	# set 'interact' to False, or don't input 'interact' by default is False
    mth5_object = make_mth5.make_mth5_from_fdsnclient("example.csv", interact=True)

    et = MTime(get_now_utc())

    print(f"Took {(int(et - st) // 60)}:{(et - st) % 60:05.2f} minutes")

Now we can interogate the created MTH5 file

Check the file name

>>> mth5_object.filename
WindowsPath('ZU_CAS04_NVR08.h5')

Have a look at the contents

>>> mth5_object
/:
====================
    |- Group: Experiment
    --------------------
        |- Group: Reports
        -----------------
        |- Group: Standards
        -------------------
            --> Dataset: summary
            ......................
        |- Group: Surveys
        -----------------
            |- Group: ZU
            ------------
                |- Group: Filters
                -----------------
                    |- Group: coefficient
                    ---------------------
                        |- Group: v to counts (electric)
                        --------------------------------
                        |- Group: v to counts (magnetic)
                        --------------------------------
                    |- Group: fap
                    -------------
                    |- Group: fir
                    -------------
                    |- Group: time_delay
                    --------------------
                        |- Group: electric time offset
                        ------------------------------
                        |- Group: hx time offset
                        ------------------------
                        |- Group: hy time offset
                        ------------------------
                        |- Group: hz time offset
                        ------------------------
                    |- Group: zpk
                    -------------
                        |- Group: electric field 1 pole butterworth high-pass
                        -----------------------------------------------------
                            --> Dataset: poles
                            ....................
                            --> Dataset: zeros
                            ....................
                        |- Group: electric field 5 pole butterworth low-pass
                        ----------------------------------------------------
                            --> Dataset: poles
                            ....................
                            --> Dataset: zeros
                            ....................
                        |- Group: magnetic field 3 pole butterworth low-pass
                        ----------------------------------------------------
                            --> Dataset: poles
                            ....................
                            --> Dataset: zeros
                            ....................
                        |- Group: mv per km to v per m
                        ------------------------------
                            --> Dataset: poles
                            ....................
                            --> Dataset: zeros
                            ....................
                        |- Group: v per m to v
                        ----------------------
                            --> Dataset: poles
                            ....................
                            --> Dataset: zeros
                            ....................
                |- Group: Reports
                -----------------
                |- Group: Standards
                -------------------
                    --> Dataset: summary
                    ......................
                |- Group: Stations
                ------------------
                    |- Group: CAS04
                    ---------------
                        |- Group: a
                        -----------
                            --> Dataset: ex
                            .................
                            --> Dataset: ey
                            .................
                            --> Dataset: hx
                            .................
                            --> Dataset: hy
                            .................
                            --> Dataset: hz
                            .................
                        |- Group: b
                        -----------
                            --> Dataset: ex
                            .................
                            --> Dataset: ey
                            .................
                            --> Dataset: hx
                            .................
                            --> Dataset: hy
                            .................
                            --> Dataset: hz
                            .................
                        |- Group: c
                        -----------
                            --> Dataset: ex
                            .................
                            --> Dataset: ey
                            .................
                            --> Dataset: hx
                            .................
                            --> Dataset: hy
                            .................
                            --> Dataset: hz
                            .................
                        |- Group: d
                        -----------
                            --> Dataset: ex
                            .................
                            --> Dataset: ey
                            .................
                            --> Dataset: hx
                            .................
                            --> Dataset: hy
                            .................
                            --> Dataset: hz
                            .................
                    |- Group: NVR08
                    ---------------
                        |- Group: a
                        -----------
                            --> Dataset: ex
                            .................
                            --> Dataset: ey
                            .................
                            --> Dataset: hx
                            .................
                            --> Dataset: hy
                            .................
                            --> Dataset: hz
                            .................
                        |- Group: b
                        -----------
                            --> Dataset: ex
                            .................
                            --> Dataset: ey
                            .................
                            --> Dataset: hx
                            .................
                            --> Dataset: hy
                            .................
                            --> Dataset: hz
                            .................
                        |- Group: c
                        -----------
                            --> Dataset: ex
                            .................
                            --> Dataset: ey
                            .................
                            --> Dataset: hx
                            .................
                            --> Dataset: hy
                            .................
                            --> Dataset: hz
                            .................

Have a look at a single channel

>>> ex = mth5_object.get_channel("CAS04", "a", "ex", survey="ZU")
>>> ex
Channel Electric:
-------------------
	component:        ex
	data type:        electric
	data format:      int32
	data shape:       (11267,)
	start:            2020-06-02T19:00:00+00:00
	end:              2020-06-02T22:07:46+00:00
	sample rate:      1.0
	
Make sure there is data

>>> ex.hdf5_dataset[0:20]
array([144812, 144378, 146214, 144470, 145275, 138457, 140016, 139721,
       134734, 133218, 134701, 135473, 131894, 130241, 132098, 132902,
       129276, 128781, 132266, 133412])
	   
.. important:: Make sure to close the MTH5 file

>>> mth5_object.close_mth5()
2021-10-31 16:06:34,097 [line 569] mth5.mth5.MTH5.close_mth5 - INFO: Flushing and closing ZU_CAS04_NVR08.h5