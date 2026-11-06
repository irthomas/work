# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:32:52 2023

@author: iant

GET ONE FILENAME FOR EACH OCCULTATION
INCLUDE INGRESS EGRESS LETTER, AS MERGED HAVE THE SAME DATETIME
OUTPUT TO FILE FOR READING IN WITH EXCEL TO SEARCH FOR DESIRED ORDER COMBINATIONS

TO BE DONE: PRIORITY GIVEN AS TO WHICH ORDER IS FIRST IN THE LIST
"""


import re
# import matplotlib.pyplot as plt
# import numpy as np
import os
# import glob
# import h5py

from tools.file.paths import paths
from tools.general.progress_bar import progress
from tools.file.hdf5_functions import make_filelist2, open_hdf5_file

# ORDERS_ORDER = [134, 136, 132, 129]


def one_filename_per_occ_dict(regex, file_level, path):
    # make filelists
    _, h5s, titles = make_filelist2(regex, file_level, path=path, open_files=False)

    h5_basenames = [os.path.splitext(os.path.basename(h5))[0] for h5 in h5s]

    h5_prefixes_all = [h5[:15] for h5 in h5_basenames]

    unique_prefixes = sorted(list(set(h5_prefixes_all)))

    d = {}

    for unique_prefix in progress(unique_prefixes):
        # get orders for the prefix
        # get indices where
        ixs = [i for i, v in enumerate(h5_basenames) if v[:15] == unique_prefix]

        try:
            orders = [int(h5_basenames[ix].split("_")[-1]) for ix in ixs]
            h5_names = [h5_basenames[ix] for ix in ixs]
            # ingress or egress
            ing_egs = [h5_basenames[ix].split("_")[-2] for ix in ixs]

            for ing_eg in sorted(set(ing_egs)):
                ixs_ing_eg = [i for i, v in enumerate(ing_egs) if v == ing_eg]
                orders_ing_eg = [orders[ix] for ix in ixs_ing_eg]
                h5s_ing_eg = [h5_names[ix] for ix in ixs_ing_eg]

                d[(unique_prefix, ing_eg)] = {"type": "normal", "n_orders": len(orders_ing_eg), "orders": sorted(orders_ing_eg), "filenames": h5s_ing_eg}
            continue
        except ValueError:
            pass

        if len(ixs) == 1 and h5_basenames[ixs[0]][-2:] == "_S":
            # if fullscan, open and get orders

            ix = ixs[0]

            ing_eg = h5_basenames[ix].split("_")[-2]

            h5f = open_hdf5_file(h5s[ix], path=path, silent=True)
            orders_all = h5f["Channel/DiffractionOrder"][...]
            h5f.close()
            orders = sorted(list(set(orders_all)))
            d[(unique_prefix, ing_eg)] = {"type": "fullscan", "n_orders": len(orders), "orders": sorted(orders), "filenames": [h5s[ix]]}

        elif len(ixs) == 2 and h5_basenames[ixs[0]][-2:] == "_S":
            # if split merged fullscan, open and get orders

            for ix in ixs:

                ing_eg = h5_basenames[ix].split("_")[-2]

                h5f = open_hdf5_file(h5s[ix], path=path, silent=True)
                orders_all = h5f["Channel/DiffractionOrder"][...]
                h5f.close()
                orders = sorted(list(set(orders_all)))
                d[(unique_prefix, ing_eg)] = {"type": "fullscan", "n_orders": len(orders), "orders": sorted(orders), "filenames": [h5s[ix]]}

        elif h5_basenames[ixs[0]][-2:] in ["CM", "CF", "CL"]:
            continue

        else:
            print([h5_basenames[i] for i in ixs])

    return d


# file_level = "hdf5_level_1p0a"
# regex = "........_......_1p0a_SO_*"
# path = paths["DATA_DIRECTORY"]

# # don't load again if already loaded (takes a few mins to run on HDD)
# if "d" not in globals():
#     d = one_filename_per_occ_dict(regex, file_level, path)

# # save to csv file for opening in excel
# with open("order_combinations.csv", "w") as f:

#     for k, v in d.items():
#         h5_prefix = k[0]
#         ing_eg = k[1]
#         obstype = v["type"]
#         n_orders = v["n_orders"]
#         orders = v["orders"]
#         orders_txt = ",".join(["%i" % i for i in orders])
#         f.write(f"{h5_prefix},{ing_eg},{obstype},{n_orders},{orders_txt}\n")
