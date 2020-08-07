# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:18:50 2020

:author: Jared Peacock

:license: MIT

"""
import datetime


leap_second_dict = {
    0: {"min": datetime.date(1980, 1, 1), "max": datetime.date(1981, 7, 1)},
    1: {"min": datetime.date(1981, 7, 1), "max": datetime.date(1982, 7, 1)},
    2: {"min": datetime.date(1982, 7, 1), "max": datetime.date(1983, 7, 1)},
    3: {"min": datetime.date(1983, 7, 1), "max": datetime.date(1985, 7, 1)},
    4: {"min": datetime.date(1985, 7, 1), "max": datetime.date(1988, 1, 1)},
    5: {"min": datetime.date(1988, 1, 1), "max": datetime.date(1990, 1, 1)},
    6: {"min": datetime.date(1990, 1, 1), "max": datetime.date(1991, 1, 1)},
    7: {"min": datetime.date(1991, 1, 1), "max": datetime.date(1992, 7, 1)},
    8: {"min": datetime.date(1992, 7, 1), "max": datetime.date(1993, 7, 1)},
    9: {"min": datetime.date(1993, 7, 1), "max": datetime.date(1994, 7, 1)},
    10: {"min": datetime.date(1994, 7, 1), "max": datetime.date(1996, 1, 1)},
    11: {"min": datetime.date(1996, 1, 1), "max": datetime.date(1997, 7, 1)},
    12: {"min": datetime.date(1997, 7, 1), "max": datetime.date(1999, 1, 1)},
    13: {"min": datetime.date(1999, 1, 1), "max": datetime.date(2006, 1, 1)},
    14: {"min": datetime.date(2006, 1, 1), "max": datetime.date(2009, 1, 1)},
    15: {"min": datetime.date(2009, 1, 1), "max": datetime.date(2012, 6, 30)},
    16: {"min": datetime.date(2012, 6, 30), "max": datetime.date(2015, 6, 30)},
    17: {"min": datetime.date(2015, 6, 30), "max": datetime.date(2016, 12, 31)},
    18: {"min": datetime.date(2016, 12, 31), "max": datetime.date(2020, 12, 1)},
}
