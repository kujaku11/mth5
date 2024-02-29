# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:18:08 2021

@author: jpeacock

 Code from ts_filters testing butter bandpass
def tst_butter_bandpass():
    np.random.seed(0)
    sample_rate = 1000.0
    dt = 1./sample_rate
    n_samples = int(1e5)
    N2 = int(n_samples/2)
    t = np.arange(n_samples) * dt
    data = np.random.random(n_samples)
    fig, axs = plt.subplots(nrows=5)
    axs[0].plot(t, data)
    axs[0].legend(["original time series",])
    df = 1./(n_samples*dt)
    frqs = np.arange(N2)*df
    fft = np.fft.fft(data)
    ampl_spec = np.abs(fft[0:N2])
    axs[1].loglog(frqs[1:], ampl_spec[1:])
    axs[1].legend(["original spectrum",])


    # First arg should be a high pass filter (but its a LPF)
    i_plot = 2
    filter_args = [1.0, None]
    data2 = butter_bandpass_filter(data, filter_args[0], filter_args[1], sample_rate)
    fft2 = np.fft.fft(data2)
    ampl_spec2 = np.abs(fft2[0:N2])
    axs[i_plot].loglog(frqs[1:], ampl_spec2[1:])
    axs[i_plot].legend([f"{filter_args[0]},{filter_args[1]}",])

    i_plot = 3
    filter_args = [None, 10.0]
    lpf_data = butter_bandpass_filter(data, filter_args[0], filter_args[1], sample_rate)
    lpf_fft = np.fft.fft(lpf_data)
    lpf_ampl_spec = np.abs(lpf_fft[0:N2])
    axs[i_plot].loglog(frqs[1:], lpf_ampl_spec[1:])
    axs[i_plot].legend([f"{filter_args[0]},{filter_args[1]}",])


    i_plot = 4
    filter_args = [1.0, 10.0]
    lpf_data = butter_bandpass_filter(data, filter_args[0], filter_args[1], sample_rate)
    lpf_fft = np.fft.fft(lpf_data)
    lpf_ampl_spec = np.abs(lpf_fft[0:N2])
    axs[i_plot].loglog(frqs[1:], lpf_ampl_spec[1:])
    axs[i_plot].legend([f"{filter_args[0]},{filter_args[1]}", ])

    # i_plot = 5
    # filter_args = [10.0, 1.0]
    # lpf_data = butter_bandpass_filter(data, filter_args[0], filter_args[1], sample_rate)
    # lpf_fft = np.fft.fft(lpf_data)
    # lpf_ampl_spec = np.abs(lpf_fft[0:N2])
    # axs[i_plot].loglog(frqs[1:], lpf_ampl_spec[1:])
    # axs[i_plot].legend([f"{filter_args[0]},{filter_args[1]}", ])


    plt.show()
    print("OK")


def main():
    tst_butter_bandpass()

if __name__ == "__main__":
    main()
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import scipy.signal as ssig
import unittest

from mth5.timeseries.ts_filters import butter_bandpass_filter
from mth5.timeseries.ts_filters import butter_bandpass

# =============================================================================
#
# =============================================================================


class TestTSFilters(unittest.TestCase):
    """
    Test remove response, make a fake signal add some trends,
    """

    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        self.sample_rate = 1000.0
        self.n_samples = int(1e5)
        self.data = np.random.random(self.n_samples)
        self.n_spectra = 4

        self.frqs = np.fft.rfftfreq(self.n_samples, d=1. / self.sample_rate)[1:]
        self.filter_args = self.n_spectra * [None]
        self.time_series = self.n_spectra * [None]
        self.spectra = self.n_spectra * [None]
        self.amplitude_spectra = self.n_spectra * [None]

        self.filter_args[0] = [None, None]  # leaves data unchanged
        self.filter_args[1] = [1.0, None]  # cuts frequecies below 1.0 (1st arg)
        self.filter_args[2] = [None, 10.0]  # cuts frequecies above 10.0 (2nd arg)
        self.filter_args[3] = [1.0, 10.0]  # cuts frequecies out of band [1.0,10.0]

        for i_spectrum in range(self.n_spectra):
            corners = self.filter_args[i_spectrum]
            #print(f"corners {corners}")
            self.time_series[i_spectrum] = butter_bandpass_filter(self.data,
                                                                  corners[0],
                                                                  corners[1],
                                                                  self.sample_rate)
            self.spectra[i_spectrum] = np.fft.rfft(self.time_series[i_spectrum])
            self.amplitude_spectra[i_spectrum] = np.abs(self.spectra[i_spectrum][1:])

        #### Uncomment plotter for debugging
        # import matplotlib.pyplot as plt
        # t = np.arange(self.n_samples) /self.sample_rate
        # num_plots = self.n_spectra + 1
        # fig, axs = plt.subplots(nrows=num_plots, sharex=True)
        # for i_plot in range(num_plots):
        #     if i_plot == 0:
        #         axs[i_plot].plot(t, self.data)
        #         axs[i_plot].legend(["original time series", ])
        #     else:
        #         i_spectrum = i_plot - 1
        #         corners = self.filter_args[i_spectrum]
        #         axs[i_plot].loglog(self.frqs, self.amplitude_spectra[i_spectrum])
        #         axs[i_plot].legend([f"corners {i_spectrum} {corners[0]},{corners[1]}", ])
        #
        # plt.show()

    def test_butter_bandpass_filter(self):
        # check that [None, None] returns unchanged time series
        assert np.sum(self.data - self.time_series[0]) == 0

        # check that corners [1.0, None] act as a HPF
        # take ratio of HPF to Original data
        ratio = self.amplitude_spectra[1]/self.amplitude_spectra[0]
        # get indices of array where frequencies are in the cut band
        low_cut_freq_ndxs =  np.where(self.frqs<self.filter_args[1][0])[0]
        # assert the spectral amplitude is diminished
        assert np.median(ratio[low_cut_freq_ndxs]) < 0.125
        # get indices of array where frequencies are in the pass band
        high_pass_freq_ndxs = np.where(self.frqs > self.filter_args[1][0])[0]
        # assert the spectral amplitude is simiar
        assert np.isclose(np.median(ratio[high_pass_freq_ndxs]), 1, atol=1e-2)

        # check that corners [None, 10.0] act as a LPF
        # take ratio of LPF to Original data
        ratio = self.amplitude_spectra[2] / self.amplitude_spectra[0]
        # get indices of array where frequencies are in the pass band
        low_pass_freq_ndxs = np.where(self.frqs < self.filter_args[2][1])[0]
        # assert the spectral amplitude is simiar
        assert np.isclose(np.median(ratio[low_pass_freq_ndxs]), 1, atol=1e-2)

        # get indices of array where frequencies are in the cut band
        high_cut_freq_ndxs = np.where(self.frqs > self.filter_args[2][1])[0]
        # assert the spectral amplitude is diminished
        assert np.median(ratio[high_cut_freq_ndxs]) < 1e-3

        # check that corners [1.0, 10.0] act as a Band pass
        ratio = self.amplitude_spectra[3] / self.amplitude_spectra[0]
        cond1 = self.frqs < self.filter_args[3][0]
        cond2 = self.frqs > self.filter_args[3][1]
        low_cut_freq_ndxs = np.where(cond1)[0]
        high_cut_freq_ndxs = np.where(cond2)[0]
        bandpass_freq_ndxs = np.where(~cond1 & ~cond2)[0]
        assert np.median(ratio[low_cut_freq_ndxs]) < 0.05
        assert np.median(ratio[high_cut_freq_ndxs]) < 1e-3
        assert np.isclose(np.median(ratio[bandpass_freq_ndxs]), 1, atol=2e-2)


    def test_butter_bandpass(self):
        with self.assertRaises(ValueError):
            butter_bandpass(None, None, self.sample_rate)



# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
