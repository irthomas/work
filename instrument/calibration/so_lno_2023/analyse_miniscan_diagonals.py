# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:13:08 2023

@author: iant


DEFINE THE SOLAR LINES IN THE CORRECTED MINISCAN ARRAYS
ANALYSE THE AOTF AND BLAZE FUNCTIONS AND SAVE THEM TO A FILE

"""

import os
import h5py
import numpy as np
# from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from lmfit import Model
from scipy import integrate

from instrument.nomad_so_instrument_v03 import aotf_peak_nu
from instrument.nomad_lno_instrument_v02 import nu0_aotf


from instrument.calibration.so_lno_2023.fit_absorption_miniscan_array import trap_absorption


MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

channel = "so"
# channel = "lno"

AOTF_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\%s_aotfs.txt" %channel)
BLAZE_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\%s_blazes.txt" %channel)


# define_edges = True #for defining the coefficients for new miniscans
define_edges = False #for running through the data

plot = []


sinc_apriori = {"a":150000.0, "b":30000.0, "c":0.0, "d":0.5, "e":-8.0}


#solar line dict
solar_line_dict = {
    # "LNO-20200201-001633-194-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[1500, 2500], "abs_region_rows":[300, -1], "abs_region_cols":[1859, 1944], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"blaze_rows":[]}
    # ],
    # "LNO-20181021-064022-125-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[500, 1000], "abs_region_rows":[0, -1], "abs_region_cols":[700, 810], "cutoffs":[0.93, 0.93], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2900, 3199], "abs_region_rows":[0, -1], "abs_region_cols":[2950, 3090], "cutoffs":[0.93, 0.93], "smfac":25},
    #     {"blaze_rows":[580, 970, 1360]}
    # ],
    # "LNO-20181021-064022-140-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[400, 700], "abs_region_rows":[0, -1], "abs_region_cols":[450, 580], "cutoffs":[0.97, 0.97], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[700, 1000], "abs_region_rows":[0, -1], "abs_region_cols":[800, 900], "cutoffs":[0.97, 0.97], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2100, 2500], "abs_region_rows":[0, -1], "abs_region_cols":[2230, 2360], "cutoffs":[0.97, 0.97], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2600], "abs_region_rows":[0, -1], "abs_region_cols":[2390, 2490], "cutoffs":[0.97, 0.97], "smfac":25},
    #     {"blaze_rows":[950, 1350, 1710]}
    #     ],


    # "LNO-20181021-071529-167-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[850, 1300], "abs_region_rows":[0, 1700], "abs_region_cols":[990, 1250], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[1400, 1700], "abs_region_rows":[0, 1700], "abs_region_cols":[1520, 1600], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2700], "abs_region_rows":[0, -1], "abs_region_cols":[2420, 2540], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"blaze_rows":[850]}
    #     ],

    # "LNO-20181021-071529-194-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[1800, 2100], "abs_region_rows":[0, -1], "abs_region_cols":[1910, 2000], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"blaze_rows":[]}
    #     ],
    
    # "LNO-20181106-192332-146-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[150, 350], "abs_region_rows":[0, -1], "abs_region_cols":[220, 280], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[1050, 1350], "abs_region_rows":[0, -1], "abs_region_cols":[1110, 1200], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2600], "abs_region_rows":[0, -1], "abs_region_cols":[2410, 2500], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2800, 3100], "abs_region_rows":[0, -1], "abs_region_cols":[2910, 2980], "cutoffs":[0.95, 0.95], "smfac":25},
    #     {"blaze_rows":[150, 1310]}
    #     ],
    
    # "LNO-20181106-192332-161-4":[
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[100, 400], "abs_region_rows":[0, 1600], "abs_region_cols":[210, 280], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[300, 550], "abs_region_rows":[0, -1], "abs_region_cols":[380, 470], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[700, 1000], "abs_region_rows":[0, -1], "abs_region_cols":[840, 930], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2000, 2200], "abs_region_rows":[0, -1], "abs_region_cols":[2040, 2140], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2200, 2400], "abs_region_rows":[0, -1], "abs_region_cols":[2260, 2340], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2500], "abs_region_rows":[0, -1], "abs_region_cols":[2340, 2420], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"arr_region_rows":[0, -1], "arr_region_cols":[2520, 2850], "abs_region_rows":[0, 1300], "abs_region_cols":[2530, 2610], "cutoffs":[0.98, 0.98], "smfac":25},
    #     {"blaze_rows":[1250]}
    #     ],
    
    
    # "LNO-20181106-195839-170-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181106-195839-197-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181209-172841-122-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181209-172841-137-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181209-180348-158-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20181209-180348-191-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190212-142357-122-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190212-142357-140-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190212-145904-158-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190212-145904-194-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190408-040458-194-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190609-015021-140-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190621-234127-122-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190622-001634-140-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190728-012702-128-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190728-012702-134-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190809-101441-146-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20190809-101441-152-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20191002-000902-176-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20191002-000902-182-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],

    # "LNO-20200201-001633-200-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200207-212317-113-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200207-212317-116-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200624-113644-146-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200624-113644-152-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200628-135310-158-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200628-135310-164-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200812-135659-170-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200812-135659-176-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200827-133646-182-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20200827-133646-194-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20201018-141050-128-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20201018-141050-188-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210208-022300-134-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210208-022300-188-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210306-015642-185-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210306-015642-188-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210402-123335-143-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210402-123335-173-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210606-021551-176-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20210606-021551-188-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220323-231521-116-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220323-231521-122-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220325-123611-128-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220325-123611-134-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220417-151406-152-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220417-151406-164-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220418-030236-158-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220418-030236-170-8":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220619-140101-164-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220619-140101-176-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220624-020349-179-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20220624-020349-182-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20221030-012837-128-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20221030-012837-149-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20221031-125107-152-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],
    # "LNO-20221031-125107-155-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         ],




    # "SO-20180716-000706-178-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[500, 900], "abs_region_rows":[0, -1], "abs_region_cols":[680, 750], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[1000, 1400], "abs_region_rows":[0, -1], "abs_region_cols":[1220, 1290], "cutoffs":[0.97, 0.97], "smfac":25},
    #         {"blaze_rows":[1100, 1500]}
    #         ],
    
    "SO-20181010-084333-184-4":[
            {"arr_region_rows":[0, -1], "arr_region_cols":[1200, 1600], "abs_region_rows":[0, -1], "abs_region_cols":[1310, 1420], "cutoffs":[0.97, 0.97], "smfac":25},
            {"arr_region_rows":[0, -1], "arr_region_cols":[1300, 1800], "abs_region_rows":[0, -1], "abs_region_cols":[1440, 1520], "cutoffs":[0.97, 0.97], "smfac":25},
            {"blaze_rows":[1500]}
            ],
            
    # "SO-20181114-084251-186-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[800, 1200], "abs_region_rows":[0, -1], "abs_region_cols":[940, 1020], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[1200, 1600], "abs_region_rows":[0, -1], "abs_region_cols":[1330, 1410], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[1300, 1700], "abs_region_rows":[0, -1], "abs_region_cols":[1440, 1515], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[2700, 3200], "abs_region_rows":[0, -1], "abs_region_cols":[2930, 3030], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"blaze_rows":[]}
    #         ],
    

    # "SO-20181206-171850-181-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, 400], "abs_region_rows":[0, -1], "abs_region_cols":[170, 260], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[2000, 2500], "abs_region_rows":[0, -1], "abs_region_cols":[2220, 2350], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"blaze_rows":[390]}
    #         ],

    
    # "SO-20190416-024455-184-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[1100, 1500], "abs_region_rows":[0, -1], "abs_region_cols":[1240, 1340], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[1200, 1600], "abs_region_rows":[0, -1], "abs_region_cols":[1360, 1430], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[1700, 2200], "abs_region_rows":[0, -1], "abs_region_cols":[1910, 2020], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"blaze_rows":[1500]}
    #         ],

    # "SO-20210226-085144-175-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"blaze_rows":[]}
    #         ],

    # "SO-20210226-085144-178-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"blaze_rows":[]}
    #         ],

    # "SO-20211105-155547-178-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"blaze_rows":[]}
    #         ],

    # "SO-20211105-155547-181-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[1900, 2400], "abs_region_rows":[0, -1], "abs_region_cols":[2090, 2300], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"blaze_rows":[390]}
    #         ],
    
    # "SO-20230112-084925-178-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"blaze_rows":[]}
    #         ],
    
    # "SO-20230112-084925-181-4":[
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[1200, 1500], "abs_region_rows":[0, -1], "abs_region_cols":[1320, 1400], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"arr_region_rows":[0, -1], "arr_region_cols":[1900, 2400], "abs_region_rows":[0, -1], "abs_region_cols":[2070, 2230], "cutoffs":[0.98, 0.98], "smfac":25},
    #         {"blaze_rows":[360]}
    #         ],    

}



# # code to make the dictionary above
# channel = "so"
# # channel = "lno"
# for f in [f for f in os.listdir(os.path.join(MINISCAN_PATH, channel)) if "-4.h5" in f]:
#     s = '''"%s":[
#         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
#         {"blaze_rows":[]}
#         ],''' %(os.path.splitext(f)[0])
    
#     print(s)
# stop()



def sinefunction(x, a, b, c, d, e):
    """modified sine function for fitting to corrected miniscan columns"""
    return a + (b * np.sin(x*np.pi/180.0 + c*x + d)) + e*x

def index(list_, value):
    """get index of value in the list, or -1 if not in list"""
    try:
        ix = list_.index(value)
    except ValueError:
        return -1
    return ix        


if define_edges:
    plot = ["fit coeffs", "uncorrected array", "residual array", "corrected array", "corrected array 2"]
    naxes = [2,2]

else:
    plot = ["absorptions"]
    naxes = [1,1]

#clear files
with open(AOTF_OUTPUT_PATH, "w") as f:
    f.write("")
with open(BLAZE_OUTPUT_PATH, "w") as f:
    f.write("")
    
loop = 0
for h5_prefix, solar_line_data_all in solar_line_dict.items(): #loop through files
    channel = h5_prefix.split("-")[0].lower()

    with h5py.File(os.path.join(MINISCAN_PATH, channel, "%s.h5" %h5_prefix), "r") as f:
        
        
        aotf = f["aotf"][...]
        keys = list(f.keys())
        
        arrs = []
        for i in range(len(keys)-1):
            arrs.append(f["array%02i" %i][...])
            
    aotf_solar_line_data = [solar_line_data for solar_line_data in solar_line_data_all if "arr_region_rows" in solar_line_data.keys()]
    blaze_solar_line_data = [solar_line_data for solar_line_data in solar_line_data_all if "blaze_rows" in solar_line_data.keys()][0]
    
    
    if define_edges: #just run through once to define edges
        aotf_solar_line_data = aotf_solar_line_data[0:1]
    
    for solar_line_data in aotf_solar_line_data: #loop through list of dictionaries, one per absorption line
    
        arr_region_rows = solar_line_data["arr_region_rows"][:]
        arr_region_cols = solar_line_data["arr_region_cols"][:]
        abs_region_rows = solar_line_data["abs_region_rows"][:]
        abs_region_cols = solar_line_data["abs_region_cols"][:]
        cutoffs = solar_line_data["cutoffs"][:]
        smfac = solar_line_data["smfac"] #smoothing factor
    
    
    
                
        #change to arr size if -1s
        if arr_region_rows[1] == -1 or define_edges:
            arr_region_rows[1] = np.min([a.shape[0] for a in arrs])
        if arr_region_cols[1] == -1 or define_edges:
            arr_region_cols[1] = np.min([a.shape[1] for a in arrs])
        if abs_region_rows[1] == -1 or define_edges:
            abs_region_rows[1] = np.min([a.shape[0] for a in arrs])
        if abs_region_cols[1] == -1 or define_edges:
            abs_region_cols[1] = np.min([a.shape[1] for a in arrs])
        if define_edges:
            arr_region_rows[0] = 0
            arr_region_cols[0] = 0
            abs_region_rows[0] = 0
            abs_region_cols[0] = 0
        
    
        #cut the unwanted edges from the array to speed up sine fitting
        aotf_cut = aotf[:, arr_region_cols[0]:arr_region_cols[1]]
        
        if len(plot) > 0:
            fig1, ax1 = plt.subplots(figsize=(14, 10), ncols=naxes[0], nrows=naxes[1], squeeze=0, constrained_layout=True)
            if len(ax1) != 1:
                ax1 = ax1.flatten()
        
        
        if define_edges:
            #just do first array
            arrs = arrs[0:1]
        
        
        for rep, arr in enumerate(arrs):
        
            print(loop, h5_prefix, rep)
            loop += 1
                
            # HR_SCALER = int(arr.shape[1]/320)
            
            #cut arrays 
            arr = arr[:, arr_region_cols[0]:arr_region_cols[1]]
            
            ix = index(plot, "uncorrected array")
            if ix > -1:
                im = ax1[ix].imshow(arr, aspect="auto")
                plt.colorbar(im, ax=ax1[ix]) 
                ax1[ix].set_title("Uncorrected miniscan array")
            
            
            
        
            
            if define_edges:
                fits = np.zeros((5, arr.shape[1]))
                residual = np.copy(arr)
                
                for i in range(arr.shape[1]):
                    y = arr[:, i]
                    x = np.arange(len(y))
                    
                    
                    
                    smodel = Model(sinefunction)
                    result = smodel.fit(y, x=x, a=sinc_apriori["a"], b=sinc_apriori["b"], c=sinc_apriori["c"], d=sinc_apriori["d"], e=sinc_apriori["e"])
                    # print(result.best_values)
                    
                    yfit = result.best_fit
                    # print(result.fit_report())
                    fits[:, i] = np.array([v for v in result.best_values.values()])
                    residual[:, i] /= yfit
                
                
                if "fit coeffs" in plot:
                    fig1, axes1 = plt.subplots(figsize=(25, 5), ncols=5)
                    for j in range(5):
                        axes1[j].set_title("Fit 1 coefficient %i" %j)
                        axes1[j].plot(fits[j, :])
                    
                ix = index(plot, "residual array")
                if ix > -1:
                    vmax = np.min((np.nanmax(residual), 1.2))
                    im = ax1[ix].imshow(residual, aspect="auto", vmax=vmax)
                    plt.colorbar(im, ax=ax1[ix]) 
                    ax1[ix].set_title("Corrected 1 miniscan residuals")
                    
                cutoff = cutoffs[0]
                
                arr2 = np.copy(arr)
                arr2[residual < cutoff] = np.nan
                
                
                ix = index(plot, "corrected array")
                if ix > -1:
                    im = ax1[ix].imshow(arr2, aspect="auto")
                    plt.colorbar(im, ax=ax1[ix]) 
                    ax1[ix].set_title("Corrected 1 miniscan array cutoff=%0.3f" %cutoff)
                
                
                fits2 = np.zeros((5, arr.shape[1]))
                residual2 = np.copy(arr)
                
                
                for i in range(arr2.shape[1]):
                    y = arr2[:, i]
                    x = np.arange(len(y))
                    
                    
                    smodel = Model(sinefunction)
                    result = smodel.fit(y, x=x, nan_policy="omit", a=sinc_apriori["a"], b=sinc_apriori["b"], c=sinc_apriori["c"], d=sinc_apriori["d"], e=sinc_apriori["e"])
                    
                    yfit = result.best_fit
                
                
                
                    # print(result.fit_report())
                    coeffs = np.array([v for v in result.best_values.values()])
                    fits2[:, i] = coeffs
                    
                    yfit_tmp = np.zeros_like(y) + np.nan
                    yfit_tmp[~np.isnan(y)] = yfit
                    
                    fit_sim = sinefunction(x, *coeffs)
                    
                    residual2[:, i] /= fit_sim
                
                
                if "fit coeffs" in plot:
                    for j in range(5):
                        axes1[j].set_title("Fit 2 coefficient %i" %j)
                        axes1[j].plot(fits2[j, :])
                        
                
                """temp code to plot first coeff and smooth"""
                plt.figure()
                plt.plot(fits2[0, :])
                smooth = savgol_filter(fits2[0, :], 99, 1)
                plt.plot(smooth)
                print([fits2[0, i] for i in solar_line_data["abs_region_cols"]])
                print([smooth[i] for i in solar_line_data["abs_region_cols"]])
                
                #save blaze to file
                np.savetxt("%s_fit_coeff0.txt" %h5_prefix, smooth)

                plt.figure()
                plt.plot(fits2[4, :])
                smooth = savgol_filter(fits2[4, :], 899, 1)
                plt.plot(smooth)
                print([fits2[4, i] for i in solar_line_data["abs_region_cols"]])
                print([smooth[i] for i in solar_line_data["abs_region_cols"]])









                
                ix = index(plot, "residual array 2")
                if ix > -1:
                    im = ax1[ix].imshow(residual2, aspect="auto")
                    plt.colorbar(im, ax=ax1[ix]) 
                    ax1[ix].set_title("Corrected 2 miniscan residuals")
                
                
                cutoff2 = cutoffs[1]
                
                arr3 = np.copy(arr)
                arr3[residual2 < cutoff2] = np.nan
                
                ix = index(plot, "corrected array 2")
                if ix > -1:
                    im = ax1[ix].imshow(arr3, aspect="auto")
                    plt.colorbar(im, ax=ax1[ix]) 
                    ax1[ix].set_title("Corrected 2 miniscan array cutoff=%0.3f" %cutoff)
                    
                # #find rows with nans and plot some of them - replaced by normalised flattened spectra
                # nan_row_ixs = np.where(np.isnan(np.mean(arr3, axis=1)))[0]
                # plt.figure()
                # for i in np.linspace(0, len(nan_row_ixs)-1, num=10):
                #     plt.plot(np.arange(arr_region_cols[0], arr_region_cols[1]), arr3[nan_row_ixs[int(i)], :], label=nan_row_ixs[int(i)])
                # plt.legend()
                
                #make normalised rows
                nanmax = np.nanmax(arr3, axis=1)
                arr3_norm = arr3 / np.repeat(nanmax, arr3.shape[1]).reshape((-1, arr3.shape[1]))
                arr3_norm_mean = np.nanmean(arr3_norm, axis=0)
                
                arr4 = arr3 / arr3_norm_mean
                # plot horizontally normalised spectra i.e. blaze function removed
                # plt.figure()
                # plt.imshow(arr4)

                #find rows with nans and plot some of them - plot flattened (blaze removed) normalised spectra
                nan_row_ixs = np.where(np.isnan(np.mean(arr4, axis=1)))[0]
                plt.figure()
                for i in np.linspace(0, len(nan_row_ixs)-1, num=10):
                    plt.plot(np.arange(arr_region_cols[0], arr_region_cols[1]), arr4[nan_row_ixs[int(i)], :]/np.nanmean(arr4[nan_row_ixs[int(i)], :]), label=nan_row_ixs[int(i)])
                plt.legend()
                plt.ylim((0.8, 1.2))
            
    
            else:       
                # #get subset of arr, make an emtpy copy
                # arr4 = np.zeros_like(arr)
                
                # #smooth each column with sav gol filter
                # for i in range(arr.shape[1]):
                #     arr4[:, i] = savgol_filter(arr[:, i], smfac, 1)
                
                # #get indices of edges within the subarray
                # col_edges = [abs_region_cols[0]-arr_region_cols[0], abs_region_cols[1]-arr_region_cols[0]]
                # row_edges = [abs_region_rows[0]-arr_region_rows[0], abs_region_rows[1]-arr_region_rows[0]]
                # col_abs = range(col_edges[0], col_edges[1])
                # row_abs = range(row_edges[0], row_edges[1])
                
                # #N x 2 array containing the first and last columns of the subarray
                # #need to use loop to convert range to indices for 2d indexing
                # col_side_values = np.array([arr[i, (col_edges[0], col_edges[1])] for i in row_abs])
                
                # #arr of continuum rows
                # arr_cont = np.array([np.interp(col_abs, col_edges, col_side_values[i, :]) for i in range(col_side_values.shape[0])])
                # #arr of rows containing the absorption
                # arr_abs = np.array(arr4[row_edges[0]:(row_edges[1]), col_edges[0]:(col_edges[1])])
                
                # #subtract to leave the continuum-corrected absorption
                # arr_sub = arr_cont - arr_abs
                
                # #integrate under the curve
                # traps = integrate.trapezoid(arr_sub, axis=1)
                
                # #get aotf frequency and wavenumbers
                # max_abs_col_ix = np.argmax(np.mean(arr_sub, axis=0)) #column ix with deepest absorption
                # aotf_peak_col = aotf_cut[:, max_abs_col_ix + col_edges[0]]
                
                # if channel == "lno":
                #     aotf_nu = [nu0_aotf(A) for A in aotf_peak_col[row_abs]]
                # if channel == "so":
                #     aotf_nu = [aotf_peak_nu(A, 0.0) for A in aotf_peak_col[row_abs]] #needs temperature
                    
                # #test: fit sine to edges, try to fit first onto second
                # #two steps: multiply by blaze from coefficient 0, then correct for small slope difference (coeff 4)
                # plt.figure()
                # xs = []
                # ys = []
                # for i in range(2):
                #     y = col_side_values[:, i]
                #     x = np.arange(len(y))
                    
                    
                #     if i == 0:
                #         coeff4 = [-0.725064, 413.378]#-5.429171805988111
                #         scaler = 89954.07674925645 / 77807.4163744896
                #     if i == 1:
                #         coeff4 = [0.0]#-5.098763716017966
                #         scaler = 1.0
                #     slope = np.polyval(coeff4, x)
                #     y *= scaler #from smoothed coeff 0
                #     y -= slope
                    
                #     # smodel = Model(sinefunction)
                #     # result = smodel.fit(y, x=x, a=sinc_apriori["a"], b=sinc_apriori["b"], c=sinc_apriori["c"], d=sinc_apriori["d"], e=sinc_apriori["e"])
                #     # yfit = result.best_fit

                #     plt.plot(x, y)
                #     # plt.plot(x, yfit)
                #     plt.title("Fit vs raw columns %i and %i" %(solar_line_data["abs_region_cols"][0], solar_line_data["abs_region_cols"][1]))
                    
                #     xs.append(x)
                #     ys.append(y)
                    
                # xs = np.asarray(xs)
                # ys = np.asarray(ys)
                # plt.figure()
                # plt.plot(xs[0, :], ys[0, :]-ys[1, :])

                # coeffs = np.polyfit(xs[0, :], ys[0, :]-ys[1, :], 1)
                # polyval = np.polyval(coeffs, xs[0, :])
                # plt.plot(xs[0, :], polyval)

                # #assume slope difference between columns is linear
                # # find slope difference from linear polyfit to plt.plot(col_side_values[:, 1]-col_side_values[:, 0])
                
    

                # savgol = savgol_filter(traps, 125, 2)
                
                aotf_nu, traps, savgol = trap_absorption(arr, aotf_cut, channel, h5_prefix, abs_region_cols, abs_region_rows, arr_region_cols, arr_region_rows)
                
                if "absorptions" in plot:
                    ax1[0][0].plot(aotf_nu, traps)
                    ax1[0][0].plot(aotf_nu, savgol)
                    
                # stop()
                
                aotf_max = np.max(savgol)
                line = "\t".join(["%0.4f" %i for i in aotf_nu]) + "\t" + "\t".join(["%0.6f" %(i/aotf_max) for i in savgol])
                with open(AOTF_OUTPUT_PATH, "a") as f:
                    f.write(line+"\n")

    #find blaze functions
    if not define_edges:
        blaze_row_indices = blaze_solar_line_data["blaze_rows"]
        for blaze_row_index in blaze_row_indices:
            for rep, arr in enumerate(arrs):
                blaze_hr = arr[blaze_row_index, :]
        
                savgol = savgol_filter(blaze_hr, 125, 2)
                blaze_max = np.max(savgol)
                
                
                aotf_central_col = aotf[blaze_row_index, int(aotf.shape[1]/2)]
                
                line = "%0.4f\t" %aotf_central_col + "\t".join(["%0.6f" %(i/blaze_max) for i in savgol])
                with open(BLAZE_OUTPUT_PATH, "a") as f:
                    f.write(line+"\n")

    
                        
# from tools.plotting.colours import get_colours


# arr_sec = arr[:, 650:800]
# max_row = np.max(arr_sec, axis=1)
# min_row = np.min(arr_sec, axis=1)

# max_row_rep = np.repeat(max_row, arr_sec.shape[1]).reshape((-1, arr_sec.shape[1]))
# min_row_rep = np.repeat(min_row, arr_sec.shape[1]).reshape((-1, arr_sec.shape[1]))

# arr_norm = (arr_sec - min_row_rep) / (max_row_rep - min_row_rep)
# arr_norm_mean = np.mean(arr_norm, axis=0)

# start_ixs = np.arange(0, 30)
# end_ixs = np.arange(120, 150)
# cont_ixs = np.concatenate((start_ixs, end_ixs))

# polyfit = np.polyfit(cont_ixs, arr_norm_mean[cont_ixs], 2)
# polyval = np.polyval(polyfit, np.arange(150))


# colours = get_colours(arr_sec.shape[0])
# for i, row in enumerate(arr_norm):
#     plt.plot(row, alpha=0.1, color=colours[i])
# plt.plot(polyval, "k")



# plt.scatter(arr_norm.T, alpha=0.1, color=np.asarray(colours))


# for arr_s in arr_sec:
#     plt.plot(arr_s / max(arr_s), alpha=0.1)