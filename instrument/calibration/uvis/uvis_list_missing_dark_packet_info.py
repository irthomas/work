# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 09:29:19 2026

@author: iant

CHECK WHICH UVIS FILES HAVE MISSING LAST PACKETS

"""

import re
import numpy as np
# from datetime import datetime
from tools.file.hdf5_functions import open_hdf5_file
from tools.file.hdf5_functions import make_filelist2
# # from tools.file.download_h5 import download_h5
# import matplotlib.pyplot as plt
from tools.general.progress_bar import progress


data_path = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
# data_path = r"C:\Users\iant\Documents\DATA\hdf5"

file_level = "hdf5_level_0p3b"

regex = re.compile("20......_.*_UVIS_D")

h5fs, h5s, _ = make_filelist2(regex, file_level, path=data_path, silent=False, open_files=False)


# for h5_ix, (h5, h5f) in enumerate(zip(h5s, h5fs)):
for h5_ix, h5 in enumerate(progress(h5s)):

    h5f = open_hdf5_file(h5, path=data_path, silent=True)

    hstart = h5f["Channel/HStart"][0]
    hend = h5f["Channel/HEnd"][0]
    binning = h5f["Channel/HorizontalAndCombinedBinningSize"][0] + 1
    ncols = int((hend - hstart + 1 - 24) / binning)

    y = h5f["Science/Y"][3, :, :]
    n_nans = np.sum(np.isnan(y))
    n_bad_rows = int(n_nans / ncols)

    # if n_nans > 0:

    vstart = h5f["Channel/VStart"][0]
    vend = h5f["Channel/VEnd"][0]
    nrows = vend - vstart + 1
    npackets = nrows // 15
    nrows_last_packet = nrows - npackets * 15

    tc20_params = [
        h5f["Telecommand20/UVISCopRow"][...].flatten()[0],
        h5f["Telecommand20/UVISDurationTime"][...].flatten()[0],
        h5f["Telecommand20/UVISStartTime"][...].flatten()[0],
        h5f["Telecommand20/LNODurationTime"][...].flatten()[0],
    ]

    n_good_rows = nrows - n_bad_rows

    if n_bad_rows > 0:
        packets_missing = ((n_bad_rows - nrows_last_packet) / 15) + 1
    else:
        packets_missing = 0.0

    if n_nans > 0:
        with open("uvis_missing_packets.txt", "a") as f:
            f.write("%s\t%i\t%i\t%0.1f\t%i\t%i\t%i\t%i\n" % (h5, n_good_rows, n_bad_rows, packets_missing, *tc20_params))
    else:
        with open("uvis_not_missing_packets.txt", "a") as f:
            f.write("%s\t%i\t%i\t%0.1f\t%i\t%i\t%i\t%i\n" % (h5, n_good_rows, n_bad_rows, packets_missing, *tc20_params))
