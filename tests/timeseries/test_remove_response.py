# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:18:08 2021

@author: jpeacock
"""

import numpy as np
from scipy import signal as sps

from mt_metadata.timeseries.filters import PoleZeroFilter
from mth5.timeseries import ChannelTS
from mth5.timeseries import ts_filters
from mth5.utils.exceptions import MTTSError

from matplotlib import pyplot as plt

c = ChannelTS()
c.sample_rate = 10

n_samples = 4000
t = np.arange(n_samples) * c.sample_interval
c.ts = np.sum([np.cos(2*np.pi*w*t + phi) for w, phi in zip(np.logspace(-4, 1, 20), np.random.rand(20))], axis=0)

pz = PoleZeroFilter(units_in="volts", units_out="nanotesla", name="instrument_response")
pz.poles = [(-6.283185+10.882477j), (-6.283185-10.882477j), (-12.566371+0j)]
pz.zeros = []
pz.normalization_factor = 2002.269 

window = sps.windows.hann(n_samples)

c.channel_response_filter.filters_list.append(pz)

ts_npow = ts_filters.zero_pad(c.ts)

f = np.fft.rfftfreq(ts_npow.size, c.sample_interval)

cr = c.channel_response_filter.complex_response(f)

ts_fft = np.fft.rfft(ts_npow) 

ts_calibrated = np.fft.irfft(ts_fft / cr)

ts_bp = ts_filters.butter_bandpass_filter(ts_calibrated, .001, 4.9, c.sample_rate)

# bp_ts = window * calibrated_ts

fig = plt.figure(1)

ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(t, c.ts)

ax2 = fig.add_subplot(3, 2, 2)
ax2.loglog(f, np.abs(ts_fft)) 

ax3 = fig.add_subplot(3, 2, 3, sharex=ax1)
ax3.plot(t, ts_calibrated[0:n_samples])

ax4 = fig.add_subplot(3, 2, 4)
ax4.loglog(f, np.abs(np.fft.rfft(ts_calibrated)))

ax5 = fig.add_subplot(3, 2, 5, sharex=ax1)
ax5.plot(t, ts_bp[0:n_samples])

ax6 = fig.add_subplot(3, 2, 6)
ax6.loglog(f, np.abs(np.fft.rfft(ts_bp)))

plt.show()


 