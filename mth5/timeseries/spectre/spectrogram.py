"""
Module contains a class that represents a spectrogram,
i.e. A 2D time series of Fourier coefficients with axes time and frequency.
"""
# Standard library imports
from typing import List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import xarray as xr

# Local imports
from mt_metadata.transfer_functions.processing.aurora.band import Band
from mt_metadata.transfer_functions.processing.aurora.frequency_bands import FrequencyBands
from mth5.timeseries.xarray_helpers import covariance_xr, initialize_xrda_2d


class Spectrogram(object):
    """
    Class to contain methods for STFT objects.
    TODO: Add support for cross powers
    TODO: Add OLS Z-estimates
    TODO: Add Sims/Vozoff Z-estimates

    """

    def __init__(
        self,
        dataset: Optional[xr.Dataset] = None
    ):
        """Constructor"""
        self._dataset = dataset
        self._frequency_increment = None
        self._frequency_band = None

    def _lowest_frequency(self):  # -> float:
        pass  # return self.dataset.frequency.min

    def _highest_frequency(self):  # -> float:
        pass  # return self.dataset.frequency.max

    def __str__(self) -> str:
        """Returns a Description of frequency coverage"""
        intro = "Spectrogram:"
        frequency_coverage = (
            f"{self.dataset.dims['frequency']} harmonics, {self.frequency_increment}Hz spaced \n"
            f" from {self.dataset.frequency.data[0]} to {self.dataset.frequency.data[-1]} Hz."
        )
        time_coverage = f"\n{self.dataset.dims['time']} Time observations"
        time_coverage = f"{time_coverage} \nStart: {self.dataset.time.data[0]}"
        time_coverage = f"{time_coverage} \nEnd:   {self.dataset.time.data[-1]}"

        channel_coverage = list(self.dataset.data_vars.keys())
        channel_coverage = "\n".join(channel_coverage)
        channel_coverage = f"\nChannels present: \n{channel_coverage}"
        return (
            intro
            + "\n"
            + frequency_coverage
            + "\n"
            + time_coverage
            + "\n"
            + channel_coverage
        )

    def __repr__(self):
        return self.__str__()

    @property
    def dataset(self):
        """returns the underlying xarray data"""
        return self._dataset

    @property
    def dataarray(self):
        """returns the underlying xarray data"""
        return self._dataset.to_array()

    @property
    def time_axis(self):
        """returns the time axis of the underlying xarray"""
        return self.dataset.time

    @property
    def frequency_axis(self):
        """returns the frequency axis of the underlying xarray"""
        return self.dataset.frequency

    @property
    def frequency_band(self) -> Band:
        """ returns a frequency band object representing the spectrograms band (assumes continuous)"""
        if self._frequency_band is None:
            band = Band(
            frequency_min=self.frequency_axis.min().item(),
            frequency_max=self.frequency_axis.max().item()
            )
            self._frequency_band = band
        return self._frequency_band

    @property
    def frequency_increment(self):
        """
        returns the "delta f" of the frequency axis
        - assumes uniformly sampled in frequency domain
        """
        if self._frequency_increment is None:
            frequency_axis = self.dataset.frequency
            self._frequency_increment = frequency_axis.data[1] - frequency_axis.data[0]
        return self._frequency_increment

    def num_harmonics_in_band(self, frequency_band, epsilon=1e-7):
        """

        Returns the number of harmonics within the frequency band in the underlying dataset

        Parameters
        ----------
        band
        stft_obj

        Returns
        -------

        """
        cond1 = self._dataset.frequency >= frequency_band.lower_bound - epsilon
        cond2 = self._dataset.frequency <= frequency_band.upper_bound + epsilon
        num_harmonics = (cond1 & cond2).data.sum()
        return num_harmonics

    def extract_band(self, frequency_band, channels=[]):
        """
        Returns another instance of Spectrogram, with the frequency axis reduced to the input band.

        Parameters
        ----------
        frequency_band
        channels

        Returns
        -------
        spectrogram: aurora.time_series.spectrogram.Spectrogram
            Returns a Spectrogram object with only the extracted band for a dataset

        """
        extracted_band_dataset = extract_band(
            frequency_band,
            self.dataset,
            channels=channels,
            epsilon=self.frequency_increment / 2.0,
        )
        # Drop NaN values along the frequency dimension
        # extracted_band_dataset = extracted_band_dataset.dropna(dim='frequency', how='any')
        spectrogram = Spectrogram(dataset=extracted_band_dataset)
        return spectrogram

    def cross_power_label(self, ch1: str, ch2: str, join_char: str = "_"):
        """ joins channel names with join_char"""
        return f"{ch1}{join_char}{ch2}"

    def _validate_frequency_bands(
        self,
        frequency_bands: FrequencyBands,
        strict: bool = True,
    ):
        """
        Make sure that the frequency bands passed are relevant.  If not, drop and warn.

        :param frequency_bands: A collection of bands
        :type frequency_bands: FrequencyBands
        :param strict: If true, band must be contained to be valid, if false, any overlapping band is valid.
        :type strict: bool
        :return:
        """
        if strict:
            valid_bands = [x for x in frequency_bands.bands() if self.frequency_band.contains(x)]
        else:
            valid_bands = [x for x in frequency_bands.bands() if self.frequency_band.overlaps(x)]
        lower_bounds = [x.lower_bound for x in valid_bands]
        upper_bounds = [x.upper_bound for x in valid_bands]
        valid_frequency_bands = FrequencyBands(
            pd.DataFrame(data={
                "lower_bound": lower_bounds,
                "upper_bound": upper_bounds,
            })
        )

        # TODO: If strict, only take bands that are contained
        return valid_frequency_bands

    def cross_powers(self, frequency_bands, channel_pairs=None):
        """
        Compute cross powers between channel pairs for given frequency bands.

        TODO: Add handling for case when band in frequency_bands is not contained
        in self.frequencies.

        Parameters
        ----------
        frequency_bands : FrequencyBands
            The frequency bands to compute cross powers for.
        channel_pairs : list of tuples, optional
            List of channel pairs to compute cross powers for.
            If None, all possible pairs will be used.

        Returns
        -------
        xr.Dataset
            Dataset containing cross powers for all channel pairs.
            Each variable is named by the channel pair (e.g. 'ex_hy')
            and contains a 2D array with dimensions (frequency, time).
            All variables share common frequency and time coordinates.
        """
        from itertools import combinations_with_replacement

        valid_frequency_bands = self._validate_frequency_bands(frequency_bands)
        # If no channel pairs specified, use all possible pairs
        if channel_pairs is None:
            channels = list(self.dataset.data_vars.keys())
            channel_pairs = list(combinations_with_replacement(channels, 2))

        # Create variable names from channel pairs
        var_names = [self.cross_power_label(ch1, ch2) for ch1, ch2 in channel_pairs]

        # Initialize a single multi-channel 2D xarray
        xpower_array = initialize_xrda_2d(
            var_names,
            coords={'frequency': frequency_bands.band_centers(),
                   'time': self.dataset.time.values},
            dtype=complex
        )

        # Compute cross powers for each band and channel pair
        for band in valid_frequency_bands.bands():
            # Extract band data
            band_data = self.extract_band(band).dataset

            # Compute cross powers for each channel pair
            for ch1, ch2 in channel_pairs:
                label = self.cross_power_label(ch1, ch2)
                # Always compute as ch1 * conj(ch2)
                xpower = (band_data[ch1] * band_data[ch2].conj()).mean(dim='frequency')

                # Store the cross power
                xpower_array.loc[dict(frequency=band.center_frequency, variable=label, time=slice(None))] = xpower

        return xpower_array

    def covariance_matrix(
        self,
        band_data: Optional['Spectrogram'] = None,
        method: str = "numpy_cov"
    ) -> xr.DataArray:
        """
        Compute full covariance matrix for spectrogram data.

        For complex-valued data, the result is a Hermitian matrix where:
        - diagonal elements are real-valued variances
        - off-diagonal element [i,j] is E[ch_i * conj(ch_j)]
        - off-diagonal element [j,i] is the complex conjugate of [i,j]

        Parameters
        ----------
        band_data : Spectrogram, optional
            If provided, compute covariance for this data
            If None, use the full spectrogram
        method : str
            Computation method. Currently only supports 'numpy_cov'

        Returns
        -------
        xr.DataArray
            Hermitian covariance matrix with proper channel labeling
            For channels i,j: matrix[i,j] = E[ch_i * conj(ch_j)]
        """
        data = band_data or self
        flat_data = data.flatten(chunk_by="time")

        if method == "numpy_cov":
            # Convert to DataArray for covariance_xr
            stacked = flat_data.to_array(dim="variable")
            return covariance_xr(stacked)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _get_all_channel_pairs(self) -> List[Tuple[str, str]]:
        """Get all unique channel pairs (upper triangle)"""
        channels = list(self.dataset.data_vars.keys())
        pairs = []
        for i, ch1 in enumerate(channels[:-1]):
            for ch2 in channels[i+1:]:
                pairs.append((ch1, ch2))
        return pairs

    def flatten(self, chunk_by: Optional[str] = "time") -> xr.Dataset:
        """

            Reshape the 2D spectrogram into a 1D flattened xarray (time-chunked by default).

        Parameters
        ----------
        chunk_by: str
            Controlled vocabulary ["time", "frequency"]. Reshaping the 2D spectrogram can be done two ways
            (basically "row-major", or column-major). In xarray, but we either keep frequency constant and
            iterate over time, or keep time constant and iterate over frequency (in the inner loop).


        Returns
        -------
        xarray.Dataset : The dataset from the band spectrogram, stacked.

        Development Notes:
        The flattening used in tf calculation by default is opposite to here
        dataset.stack(observation=("frequency", "time"))
        However, for feature extraction, it may make sense to swap the order:
        xrds = band_spectrogram.dataset.stack(observation=("time", "frequency"))
        This is like chunking into time windows and allows individual features to be computed on each time window -- if desired.
        Still need to split the time series though--Splitting to time would be a reshape by (last_freq_index-first_freq_index).
        Using pure xarray this may not matter but if we drop down into numpy it could be useful.


        """
        if chunk_by == "time":
            observation = ("time", "frequency")
        elif chunk_by == "frequency":
            observation = ("frequency", "time")
        return self.dataset.stack(observation=observation)


def extract_band(
    frequency_band: Band,
    fft_obj: Union[xr.Dataset, xr.DataArray],
    channels: Optional[list] = None,
    epsilon: float = 1e-7
) -> Union[xr.Dataset, xr.DataArray]:
    """
        Extracts a frequency band from xr.DataArray representing a spectrogram.

        TODO: Update variable names.

        Development Notes:
            Base dataset object should be a xr.DataArray (not xr.Dataset)
            - drop=True does not play nice with h5py and Dataset, results in a type error.
            File "stringsource", line 2, in h5py.h5r.Reference.__reduce_cython__
            TypeError: no default __reduce__ due to non-trivial __cinit__
            However, it works OK with DataArray.

        Parameters
        ----------
        frequency_band: mt_metadata.transfer_functions.processing.aurora.band.Band
            Specifies interval corresponding to a frequency band
        fft_obj: xarray.core.dataset.Dataset
            Short-time-Fourier-transformed datat.  Can be multichannel.
        channels: list
            Channel names to extract.
        epsilon: float
            Use this when you are worried about missing a frequency due to
            round off error.  This is in general not needed if we use a df/2 pad
            around true harmonics.

        Returns
        -------
        extracted_band: xr.DataArray
            The frequencies within the band passed into this function
    """
    cond1 = fft_obj.frequency >= frequency_band.lower_bound - epsilon
    cond2 = fft_obj.frequency <= frequency_band.upper_bound + epsilon
    try:
        extracted_band = fft_obj.where(cond1 & cond2, drop=True)
    except TypeError:  # see Note #1
        tmp = fft_obj.to_array()
        extracted_band = tmp.where(cond1 & cond2, drop=True)
        extracted_band = extracted_band.to_dataset("variable")
    if channels:
        extracted_band = extracted_band[channels]
    return extracted_band
