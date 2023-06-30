# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:13:08 2023

@author: iant
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


MINISCAN_PATH = os.path.normcase(r"C:\Users\iant\Documents\DATA\miniscans")

AOTF_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\aotfs.txt")
BLAZE_OUTPUT_PATH = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\blazes.txt")


# define_edges = True #for defining the coefficients for new miniscans
define_edges = False #for running through the data

plot = []

#solar line dict
solar_line_dict = {
    "LNO-20200201-001633-194-4":[
        {"arr_region_rows":[0, -1], "arr_region_cols":[1500, 2500], "abs_region_rows":[300, -1], "abs_region_cols":[1859, 1944], "cutoffs":[0.98, 0.98], "smfac":25},
        {"blaze_rows":[]}
    ],
    "LNO-20181021-064022-125-4":[
        {"arr_region_rows":[0, -1], "arr_region_cols":[500, 1000], "abs_region_rows":[0, -1], "abs_region_cols":[700, 810], "cutoffs":[0.93, 0.93], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2900, 3199], "abs_region_rows":[0, -1], "abs_region_cols":[2950, 3090], "cutoffs":[0.93, 0.93], "smfac":25},
        {"blaze_rows":[580, 970, 1360]}
    ],
    "LNO-20181021-064022-140-4":[
        {"arr_region_rows":[0, -1], "arr_region_cols":[400, 700], "abs_region_rows":[0, -1], "abs_region_cols":[450, 580], "cutoffs":[0.97, 0.97], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[700, 1000], "abs_region_rows":[0, -1], "abs_region_cols":[800, 900], "cutoffs":[0.97, 0.97], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2100, 2500], "abs_region_rows":[0, -1], "abs_region_cols":[2230, 2360], "cutoffs":[0.97, 0.97], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2600], "abs_region_rows":[0, -1], "abs_region_cols":[2390, 2490], "cutoffs":[0.97, 0.97], "smfac":25},
        {"blaze_rows":[950, 1350, 1710]}
        ],


    "LNO-20181021-071529-167-4":[
        {"arr_region_rows":[0, -1], "arr_region_cols":[850, 1300], "abs_region_rows":[0, 1700], "abs_region_cols":[990, 1250], "cutoffs":[0.95, 0.95], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[1400, 1700], "abs_region_rows":[0, 1700], "abs_region_cols":[1520, 1600], "cutoffs":[0.95, 0.95], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2700], "abs_region_rows":[0, -1], "abs_region_cols":[2420, 2540], "cutoffs":[0.95, 0.95], "smfac":25},
        {"blaze_rows":[850]}
        ],

    "LNO-20181021-071529-194-4":[
        {"arr_region_rows":[0, -1], "arr_region_cols":[1800, 2100], "abs_region_rows":[0, -1], "abs_region_cols":[1910, 2000], "cutoffs":[0.95, 0.95], "smfac":25},
        {"blaze_rows":[]}
        ],
    
    "LNO-20181106-192332-146-4":[
        {"arr_region_rows":[0, -1], "arr_region_cols":[150, 350], "abs_region_rows":[0, -1], "abs_region_cols":[220, 280], "cutoffs":[0.95, 0.95], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[1050, 1350], "abs_region_rows":[0, -1], "abs_region_cols":[1110, 1200], "cutoffs":[0.95, 0.95], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2600], "abs_region_rows":[0, -1], "abs_region_cols":[2410, 2500], "cutoffs":[0.95, 0.95], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2800, 3100], "abs_region_rows":[0, -1], "abs_region_cols":[2910, 2980], "cutoffs":[0.95, 0.95], "smfac":25},
        {"blaze_rows":[150, 1310]}
        ],
    
    "LNO-20181106-192332-161-4":[
        {"arr_region_rows":[0, -1], "arr_region_cols":[100, 400], "abs_region_rows":[0, 1600], "abs_region_cols":[210, 280], "cutoffs":[0.98, 0.98], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[300, 550], "abs_region_rows":[0, -1], "abs_region_cols":[380, 470], "cutoffs":[0.98, 0.98], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[700, 1000], "abs_region_rows":[0, -1], "abs_region_cols":[840, 930], "cutoffs":[0.98, 0.98], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2000, 2200], "abs_region_rows":[0, -1], "abs_region_cols":[2040, 2140], "cutoffs":[0.98, 0.98], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2200, 2400], "abs_region_rows":[0, -1], "abs_region_cols":[2260, 2340], "cutoffs":[0.98, 0.98], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2300, 2500], "abs_region_rows":[0, -1], "abs_region_cols":[2340, 2420], "cutoffs":[0.98, 0.98], "smfac":25},
        {"arr_region_rows":[0, -1], "arr_region_cols":[2520, 2850], "abs_region_rows":[0, 1300], "abs_region_cols":[2530, 2610], "cutoffs":[0.98, 0.98], "smfac":25},
        {"blaze_rows":[1250]}
        ],
    
    
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

}



# code to make the dictionary above
# for f in [f for f in os.listdir(os.path.join(MINISCAN_PATH, "lno")) if ".h5" in f]:
#     s = '''"%s":[
#         {"arr_region_rows":[0, -1], "arr_region_cols":[0, -1], "abs_region_rows":[0, -1], "abs_region_cols":[0, -1], "cutoffs":[0.98, 0.98], "smfac":25},
#         ],''' %(os.path.splitext(f)[0])
    
#     print(s)
# stop()



def sinefunction(x, a, b, c, d, e):
    return a + (b * np.sin(x*np.pi/180.0 + c + e*x)) + d*x

def index(list_, value):
    try:
        ix = list_.index(value)
    except ValueError:
        return -1
    return ix        


if define_edges:
    plot = ["uncorrected array", "residual array", "corrected array", "corrected array 2"]
    naxes = [2,2]

# else:
#     plot = ["absorptions"]
#     naxes = [1,1]

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
    
    for solar_line_data in aotf_solar_line_data: #loop through list of dictionaries
    
        arr_region_rows = solar_line_data["arr_region_rows"]
        arr_region_cols = solar_line_data["arr_region_cols"]
        abs_region_rows = solar_line_data["abs_region_rows"]
        abs_region_cols = solar_line_data["abs_region_cols"]
        cutoffs = solar_line_data["cutoffs"]
        smfac = solar_line_data["smfac"]
    
    
    
                
        #change to arr size if -1s
        if arr_region_rows[1] == -1 or define_edges:
            arr_region_rows[1] = np.min([a.shape[0] for a in arrs])
        if arr_region_cols[1] == -1 or define_edges:
            arr_region_cols[1] = np.min([a.shape[1] for a in arrs])
        if abs_region_rows[1] == -1 or define_edges:
            abs_region_rows[1] = np.min([a.shape[0] for a in arrs])
        if abs_region_cols[1] == -1 or define_edges:
            abs_region_cols[1] = np.min([a.shape[1] for a in arrs])
        
    
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
                    result = smodel.fit(y, x=x, a=0, b=30000, c=0, d=0, e=0)
                    
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
                    im = ax1[ix].imshow(residual, aspect="auto")
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
                    result = smodel.fit(y, x=x, nan_policy="omit", a=0, b=30000, c=0, d=0, e=0)
                    
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
                    
                #find rows with nans and plot some of them
                nan_row_ixs = np.where(np.isnan(np.mean(arr3, axis=1)))[0]
                plt.figure()
                for i in np.linspace(0, len(nan_row_ixs)-1, num=10):
                    plt.plot(np.arange(arr_region_cols[0], arr_region_cols[1]), arr3[nan_row_ixs[int(i)], :], label=nan_row_ixs[int(i)])
                plt.legend()
                
            
    
            else:        
                arr4 = np.zeros_like(arr)
                for i in range(arr.shape[1]):
                    arr4[:, i] = savgol_filter(arr[:, i], smfac, 1)
                
                
                col_edges = [abs_region_cols[0]-arr_region_cols[0], abs_region_cols[1]-arr_region_cols[0]]
                row_edges = [abs_region_rows[0]-arr_region_rows[0], abs_region_rows[1]-arr_region_rows[0]]
                col_abs = range(col_edges[0], col_edges[1])
                row_abs = range(row_edges[0], row_edges[1])
                
                col_side_values = np.array([arr[i, (col_edges[0], col_edges[1])] for i in row_abs])
                
                arr_cont = np.array([np.interp(col_abs, col_edges, col_side_values[i, :]) for i in range(col_side_values.shape[0])])
                arr_abs = np.array(arr4[row_edges[0]:(row_edges[1]), col_edges[0]:(col_edges[1])])
                
                arr_sub = arr_cont - arr_abs
                
                traps = integrate.trapezoid(arr_sub, axis=1)
                
                #get aotf frequency and wavenumbers
                max_abs_col_ix = np.argmax(np.mean(arr_sub, axis=0)) #column ix with deepest absorption
                aotf_peak_col = aotf_cut[:, max_abs_col_ix + col_edges[0]]
                
                if channel == "lno":
                    aotf_nu = [nu0_aotf(A) for A in aotf_peak_col[row_abs]]
                if channel == "so":
                    aotf_nu = [aotf_peak_nu(A) for A in aotf_peak_col[row_abs]] #needs temperature
                    
                    
                    


                savgol = savgol_filter(traps, 125, 2)
                
                if "absorptions" in plot:
                    ax1[0][0].plot(aotf_nu, traps)
                    ax1[0][0].plot(aotf_nu, savgol)
                
                aotf_max = np.max(savgol)
                line = "\t".join(["%0.4f" %i for i in aotf_nu]) + "\t" + "\t".join(["%0.6f" %(i/aotf_max) for i in savgol])
                with open(AOTF_OUTPUT_PATH, "a") as f:
                    f.write(line+"\n")

    #find blaze functions
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

    
                        


#read in AOTFs
aotfs = {"nu":[], "aotf":[]}
with open(AOTF_OUTPUT_PATH, "r") as f:
    lines = f.readlines()
    
for line in lines:
    line_split = line.split("\t")
    n_nus = int(len(line_split)/2)
    aotfs["nu"].append(np.asfarray(line_split[0:n_nus]))
    aotfs["aotf"].append(np.asfarray(line_split[n_nus:]))
    
plt.figure()
plt.title("AOTF functions")
plt.xlabel("Wavenumber")
plt.ylabel("AOTF transmittance")
plt.grid()
for nu, aotf in zip(aotfs["nu"], aotfs["aotf"]):
    plt.plot(nu, aotf)

#read in blazes
blazes = {"aotf":[], "blaze":[]}
with open(BLAZE_OUTPUT_PATH, "r") as f:
    lines = f.readlines()
    
for line in lines:
    line_split = line.split("\t")
    blazes["aotf"].append(line_split[0])
    blazes["blaze"].append(np.asfarray(line_split[1:]))
    
plt.figure()
plt.title("Blaze functions")
plt.xlabel("Pixel number")
plt.ylabel("Blaze function")
plt.grid()
for aotf_freq, blaze in zip(blazes["aotf"], blazes["blaze"]):
    plt.plot(np.linspace(0., 320., num=len(blaze)), blaze, label=aotf_freq)

plt.legend()