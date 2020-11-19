# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:32:41 2020

@author: iant

FIND FIRST DETECTOR ROWS
"""


import re
import numpy as np
import matplotlib.pyplot as plt
import datetime

from tools.file.hdf5_functions import make_filelist


regex = re.compile("20(18|19|20).*_SO_.*_136")
file_level = "hdf5_level_1p0a"


detector_start_rows = []
detector_datetimes = []


hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)
for hdf5_file, hdf5_filename in zip(hdf5_files, hdf5_filenames):
    
    detector_top = hdf5_file["Channel/WindowTop"][...]
    detector_start_rows.append(detector_top[0])
    
    year = hdf5_filename[0:4]
    month = hdf5_filename[4:6]
    day = hdf5_filename[6:8]
    
    
    detector_datetime = datetime.datetime(int(year), int(month), int(day))
    detector_datetimes.append(detector_datetime)
    
detector_start_rows = np.asfarray(detector_start_rows)

plt.scatter(detector_datetimes, detector_start_rows)
plt.xlabel("Time")
plt.ylabel("Detector top row used")

plt.text(datetime.datetime(2018, 4, 1), 121, "Not pointing to sun centre at start of mission")
plt.text(datetime.datetime(2018, 8, 11), 120.5, "Nominal pointing starts 11th August 2018")