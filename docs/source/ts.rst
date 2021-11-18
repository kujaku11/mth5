=====================
Time Series Objects
=====================

All datasets at the *Channel* level are represented by :class:`mth5.timeseries.ChannelTS` objects.  The :class:`mth5.timeseries.ChannelTS` are based on :class:`xarray.DataArray` objects.  This way memory usage is minimal because xarray is lazy and only uses what is called for.  Another benefit is that metadata can directly accompany the data.  Currently the model is that all metadata are input into a :class:`mth5.metadata.Base` object to be validated first and then the :class:`xarray.DataArray` can be updated.  This is not automated at this point so the user just needs to use the function update_xarray_metadata when metadata values are changed.  Another advantage of using xarray is that the time series data are indexed by time making it easier to align, trim, extract, sort, etc.  

All run datasets are represented by :class:`mth5.timeseries.RunTS` objects, which are based on :class:`xarray.DataSet` which is a collection of :class:`xarray.DataArray` objects.  The benefits of using xarray are that many of the methods such as aligning, indexing, sorting are already developed and are robust.  Therefore the useability is easier without more coding. 

Another reason why xarray was picked as the basis for representing the data is that it works seamlessly with other programs like Dask for parallel computing, and plotting tools like hvplot.

Examples
-------------

.. toctree::
    :maxdepth:2
    
    notebooks/ts_example.ipynb