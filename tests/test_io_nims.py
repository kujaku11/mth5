# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:37:05 2020

:author: Jared Peacock

:license: MIT

"""

import numpy as np
import re

test_data = np.array(
    [
        13,
        1,
        131,
        129,
        200,
        224,
        77,
        23,
        1,
        104,
        33,
        239,
        196,
        0,
        7,
        51,
        71,
        55,
        130,
        33,
        239,
        183,
        0,
        7,
        47,
        71,
        55,
        130,
        33,
        239,
        184,
        0,
        7,
        37,
        71,
        55,
        131,
        33,
        239,
        183,
        0,
        7,
        43,
        71,
        55,
        133,
        33,
        239,
        186,
        0,
        7,
        42,
        71,
        55,
        130,
        33,
        239,
        186,
        0,
        7,
        42,
        71,
        55,
        133,
        33,
        239,
        186,
        0,
        7,
        45,
        71,
        55,
        128,
        33,
        239,
        190,
        0,
        7,
        47,
        71,
        55,
        134,
        119,
        255,
        60,
        6,
        255,
        251,
        149,
        255,
        55,
        198,
        255,
        252,
        206,
        255,
        55,
        3,
        255,
        251,
        123,
        255,
        55,
        23,
        255,
        253,
        6,
        255,
        57,
        145,
        255,
        254,
        134,
        255,
        63,
    ],
    dtype=np.uint8,
)


test_sequence = [1, 131]

# want to find the index there the test data is equal to the test sequence
def to_string(data):
    return "".join(map(chr, data))


# index_arr = np.array([m.start() for m in re.finditer(to_string(test_sequence),
#                                                      to_string(test_data))])

# np_where = np.where((test_data[:-1]==test_sequence[0]) & (test_data[1:]==test_sequence[1]))

t = np.vstack([np.roll(test_data, shift) for shift in -np.arange(len(test_sequence))]).T
find_index = np.where(np.all(t == test_sequence, axis=1))[0]
