# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:53:22 2025

@author: iant

COMPARE LEVEL 1.OA UVIS OCCULTATION SPECTRA PROCESSED IN THE MATLAB AND PYTHON PIPELINES
PYTHON LOOKS BETTER EXCEPT FOR 20250101_175230
SUFFICIENT TO MOVE TO NEXT STAGE

MUST BE AT BIRA TO ACCESS DATA
"""

import re
# import numpy as np
# from datetime import datetime
from tools.file.hdf5_functions import open_hdf5_file
from tools.file.hdf5_functions import make_filelist2
# # from tools.file.download_h5 import download_h5
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
# from tools.sql.heaters_temp import get_temperature_range
# from numpy.polynomial import Polynomial
from matplotlib.backends.backend_pdf import PdfPages

data_path_uvispy = r"X:\projects\work\NOMAD\test\iant\hdf5"

data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
# data_path = r"C:\Users\iant\Documents\DATA\hdf5"

# # download_h5(h5, path=r"C:\Users\iant\Documents\DATA\hdf5")
file_level = "hdf5_level_1p0a"


regex = re.compile("20......_.*_UVIS_.*")

h5fs_py, h5s_py, _ = make_filelist2(regex, file_level, path=data_path_uvispy, silent=False)

with PdfPages("uvis_occ_py_matlab_comparison.pdf") as pdf:
    for h5_ix, (h5_py, h5f_py) in enumerate(zip(h5s_py, h5fs_py)):

        h5f_m = open_hdf5_file(h5_py, path=data_path)

        y_py = h5f_py["Science/Y"][...]
        y_m = h5f_m["Science/Y"][...]

        plt.figure(figsize=(25, 10))
        plt.title("%s Python" % h5_py[0:15])
        plt.plot(y_py.T)
        plt.ylim(-0.05, 1.1)
        plt.xlim(-1, y_py.shape[1]+1)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(25, 10))
        plt.title("%s Matlab" % h5_py[0:15])
        plt.plot(y_m.T)
        plt.ylim(-0.05, 1.1)
        plt.xlim(-1, y_py.shape[1]+1)
        pdf.savefig()
        plt.close()
