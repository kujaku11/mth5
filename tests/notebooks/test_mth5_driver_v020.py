#!/usr/bin/env python
# coding: utf-8

# # Make MTH5 from IRIS Data Managment Center v0.2.0
#
# This example demonstrates how to build an MTH5 from data archived at IRIS, it could work with any MT data stored at an FDSN data center (probably).
#
# We will use the `mth5.clients.FDSN` class to build the file.  There is also second way using the more generic `mth5.clients.MakeMTH5` class, which will be highlighted below.
#
# **Note:** this example assumes that data availability (Network, Station, Channel, Start, End) are all previously known.  If you do not know the data that you want to download use [IRIS tools](https://ds.iris.edu/ds/nodes/dmc/tools/##) to get data availability.

# In[ ]:


from pathlib import Path

import numpy as np
import pandas as pd
from mth5.mth5 import MTH5
from mth5.clients.make_mth5 import FDSN

from matplotlib import pyplot as plt

from mt_metadata.transfer_functions.core import TF


# get_ipython().run_line_magic("matplotlib", "widget")


# ## Set the path to save files to as the current working directory

# In[ ]:


default_path = Path().cwd()


# ## Initialize a MakeMTH5 object
#
# Here, we are setting the MTH5 file version to 0.2.0 so that we can have multiple surveys in a single file.  Also, setting the client to "IRIS".  Here, we are using `obspy.clients` tools for the request.  Here are the available [FDSN clients](https://docs.obspy.org/packages/obspy.clients.fdsn.html).
#
# **Note:** Only the "IRIS" client has been tested.

# In[ ]:


fdsn_object = FDSN(mth5_version="0.2.0")
fdsn_object.client = "IRIS"


# ## Make the data inquiry as a DataFrame
#
# There are a few ways to make the inquiry to request data.
#
# 1. Make a DataFrame by hand.  Here we will make a list of entries and then create a DataFrame with the proper column names
# 2. You can create a CSV file with a row for each entry. There are some formatting that you need to be aware of.  That is the column names and making sure that date-times are YYYY-MM-DDThh:mm:ss
#
#
# | Column Name         |   Description                                                                                                 |
# | ------------------- | --------------------------------------------------------------------------------------------------------------|
# | **network**         | [FDSN Network code (2 letters)](http://www.fdsn.org/networks/)                                                |
# | **station**         | [FDSN Station code (usually 5 characters)](https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/)|
# | **location**        | [FDSN Location code (typically not used for MT)](http://docs.fdsn.org/projects/source-identifiers/en/v1.0/location-codes.html) |
# | **channel**         | [FDSN Channel code (3 characters)](http://docs.fdsn.org/projects/source-identifiers/en/v1.0/channel-codes.html)|
# | **start**           | Start time (YYYY-MM-DDThh:mm:ss) UTC |
# | **end**             | End time (YYYY-MM-DDThh:mm:ss) UTC  |

# In[ ]:


channels = ["LFE", "LFN", "LFZ", "LQE", "LQN"]
CAS04 = ["8P", "CAS04", "2020-06-02T19:00:00", "2020-07-13T19:00:00"]
NVR08 = ["8P", "NVR08", "2020-06-02T19:00:00", "2020-07-13T19:00:00"]

request_list = []
for entry in [CAS04, NVR08]:
    for channel in channels:
        request_list.append([entry[0], entry[1], "", channel, entry[2], entry[3]])

# Turn list into dataframe
request_df = pd.DataFrame(request_list, columns=fdsn_object.request_columns)
request_df


# ## Save the request as a CSV
#
# Its helpful to be able to save the request as a CSV and modify it and use it later.  A CSV can be input as a request to `MakeMTH5`

# In[ ]:


request_df.to_csv(default_path.joinpath("fdsn_request.csv"))


# ## Get only the metadata from IRIS
#
# It can be helpful to make sure that your request is what you would expect.  For that you can request only the metadata from IRIS.  The request is quick and light so shouldn't need to worry about the speed.  This returns a StationXML file and is loaded into an `obspy.Inventory` object.

# In[ ]:


inventory, data = fdsn_object.get_inventory_from_df(request_df, data=False)


# Have a look at the Inventory to make sure it contains what is requested.

# In[ ]:


inventory


# ## Make an MTH5 from a request
#
# Now that we've created a request, and made sure that its what we expect, we can make an MTH5 file.  The input can be either the DataFrame or the CSV file.
#
# We are going to time it just to get an indication how long it might take.  Should take about 4 minutes.
#
# **Note:** we are setting `interact=False`.  If you want to just to keep the file open to interogat it set `interact=True`.
#
# ### Make an MTH5 using MakeMTH5
#
# Another way to make a file is using the `mth5.clients.MakeMTH5` class, which is more generic than FDSN, but doesn't have as many methods.  The `MakeMTH5` class is meant to be a convienence method for the various clients.
#
# ```
# from mth5.clients import MakeMTH5
#
# make_mth5_object = MakeMTH5(mth5_version='0.2.0', interact=False)
# mth5_filename = make_mth5_object.from_fdsn_client(request_df, client="IRIS")
# ```

# In[ ]:


# get_ipython().run_cell_magic(
#    "time",
#    "",
#    '\n'
mth5_object = fdsn_object.make_mth5_from_fdsn_client(request_df, interact=False)
print(f"Created {mth5_object}")
# )


# In[ ]:


# open file already created
mth5_object = MTH5()
mth5_object.open_mth5("8P_CAS04_NVR08.h5")


# ## Have a look at the contents of the created file

# In[ ]:


mth5_object


# ## Channel Summary
#
# A convenience table is supplied with an MTH5 file.  This table provides some information about each channel that is present in the file.  It also provides columns `hdf5_reference`, `run_hdf5_reference`, and `station_hdf5_reference`, these are internal references within an HDF5 file and can be used to directly access a group or dataset by using `mth5_object.from_reference` method.
#
# **Note:** When a MTH5 file is close the table is resummarized so when you open the file next the `channel_summary` will be up to date. Same with the `tf_summary`.

# In[ ]:


mth5_object.channel_summary.clear_table()
mth5_object.channel_summary.summarize()

ch_df = mth5_object.channel_summary.to_dataframe()
ch_df


# ## Have a look at a station
#
# Lets grab one station `CAS04` and have a look at its metadata and contents.
# Here we will grab it from the `mth5_object`.

# In[ ]:


cas04 = mth5_object.get_station("CAS04", survey="CONUS_South")
cas04.metadata


# ### Changing Metadata
# If you want to change the metadata of any group, be sure to use the `write_metadata` method.  Here's an example:

# In[ ]:


cas04.metadata.location.declination.value = -13.5
cas04.write_metadata()
print(cas04.metadata.location.declination)


# ## Have a look at a single channel
#
# Let's pick out a channel and interogate it. There are a couple ways
# 1. Get a channel the first will be from the `hdf5_reference` [*demonstrated here*]
# 2. Get a channel from `mth5_object`
# 3. Get a station first then get a channel
#

# In[ ]:


ex = mth5_object.from_reference(ch_df.iloc[0].hdf5_reference).to_channel_ts()
print(ex)


# In[ ]:


ex.channel_metadata


# ## Calibrate time series data
# Most data loggers output data in digital counts.  Then a series of filters that represent the various instrument responses are applied to get the data into physical units.  The data can then be analyzed and processed. Commonly this is done during the processing step, but it is important to be able to look at time series data in physical units.  Here we provide a `remove_instrument_response` method in the `ChananelTS` object.  Here's an example:

# In[ ]:


print(ex.channel_response_filter)
ex.channel_response_filter.plot_response(np.logspace(-4, 1, 50))


# In[ ]:


ex.remove_instrument_response(plot=True)


# ## Have a look at a run
#
# Let's pick out a run, take a slice of it, and interogate it. There are a couple ways
# 1. Get a run the first will be from the `run_hdf5_reference` [*demonstrated here*]
# 2. Get a run from `mth5_object`
# 3. Get a station first then get a run

# In[ ]:


run_from_reference = mth5_object.from_reference(
    ch_df.iloc[0].run_hdf5_reference
).to_runts(start=ch_df.iloc[0].start.isoformat(), n_samples=360)
print(run_from_reference)


# In[ ]:


run_from_reference.plot()


# ### Calibrate Run

# In[ ]:


calibrated_run = run_from_reference.calibrate()
calibrated_run.plot()


# ## Load Transfer Functions
#
# You can download the transfer functions for **CAS04** and **NVR08** from [IRIS SPUD EMTF](http://ds.iris.edu/spud/emtf).  This has already been done as EMTF XML format and will be loaded here.

# In[ ]:


cas04_tf = r"USMTArray.CAS04.2020.xml"
nvr08_tf = r"USMTArray.NVR08.2020.xml"


# In[ ]:


# from mt_metadata.transfer_functions.core import TF


# In[ ]:


for tf_fn in [cas04_tf, nvr08_tf]:
    tf_obj = TF(tf_fn)
    tf_obj.read()
    mth5_object.add_transfer_function(tf_obj)


# ### Have a look at the transfer function summary

# In[ ]:


mth5_object.tf_summary.summarize()
tf_df = mth5_object.tf_summary.to_dataframe()
tf_df


# ### Plot the transfer functions using MTpy
#
# **Note:** This currently works on branch `mtpy/v2_plots`

# In[ ]:

## COMMENT OUT MTPY STUFF -- kk 20230604
# from mtpy import MTCollection
#
#
# # In[ ]:
#
#
# mc = MTCollection()
# mc.open_collection(r"8P_CAS04_NVR08")
#
#
# # In[ ]:
#
#
# pmr = mc.plot_mt_response(["CAS04", "NVR08"], plot_style="1")
#
#
# # ## Plot Station locations
# #
# # Here we can plot station locations for all stations in the file, or we can give it a bounding box.  If you have internet access a basemap will be plotted using [Contextily](https://contextily.readthedocs.io/en/latest/).
#
# # In[ ]:
#
#
# st = mc.plot_stations(pad=0.9, fig_num=5, fig_size=[6, 4])
#
#
# # In[ ]:
#
#
# st.fig.get_axes()[0].set_xlim((-121.9, -117.75))
# st.fig.get_axes()[0].set_ylim((37.35, 38.5))
# st.update_plot()


# In[ ]:


mth5_object.close_mth5()


# In[ ]:
