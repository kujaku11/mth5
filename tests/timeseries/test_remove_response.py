# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:18:08 2021

@author: jpeacock
"""

import numpy as np

from mt_metadata.timeseries.filters import PoleZeroFilter
from mth5.timeseries import ChannelTS
from mth5.timeseries import ts_filters
from mth5.utils.exceptions import MTTSError

from matplotlib import pyplot as plt

c = ChannelTS()
c.sample_rate = 10

n_samples = 4096
t = np.arange(n_samples) * c.sample_interval
c.ts = np.sum([np.cos(2*np.pi*w*t + phi) for w, phi in zip(np.logspace(-3, 3, 20), np.random.rand(20))], axis=0)

pz = PoleZeroFilter(units_in="volts", units_out="nanotesla", name="instrument_response")
pz.poles = [(-6.283185+10.882477j), (-6.283185-10.882477j), (-12.566371+0j)]
pz.zeros = []
pz.normalization_factor = 2002.269 

c.channel_response_filter.filters_list.append(pz)

npow_ts = ts_filters.zero_pad(c.ts)

f = np.fft.rfftfreq(npow_ts.size, c.sample_interval)

cr = c.channel_response_filter.complex_response(f)

calibrated_ts = np.fft.irfft(np.fft.rfft(npow_ts) / cr)

fig = plt.figure(1)

ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(t, c.ts)

ax2 = fig.add_subplot(3, 2, 2)
 

ax3 = fig.add_subplot(3, 2, 3, sharex=ax1)
ax3.plot(t.calibrated_ts)

ax3



 