# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:32:52 2023

@author: iant

GET ONE FILENAME FOR EACH OCCULTATION
INCLUDE INGRESS EGRESS LETTER, AS MERGED HAVE THE SAME DATETIME
OUTPUT TO FILE FOR READING IN WITH EXCEL TO SEARCH FOR DESIRED ORDER COMBINATIONS

OPTIONAL: PRIORITY GIVEN AS TO WHICH ORDER IS FIRST IN THE LIST (NOT YET READY)
"""


import re
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import h5py

from tools.file.paths import paths
from tools.general.progress_bar import progress

# ORDERS_ORDER = [134, 136, 132, 129]


file_level = "hdf5_level_1p0a"

h5_dir = os.path.join(paths["DATA_DIRECTORY"], file_level)


def make_obs_dict(h5_dir):
    #make filelists
    h5_filepaths = sorted(glob.glob(h5_dir+os.sep+"**"+os.sep+"*SO*.h5", recursive=True))
    
    h5_basenames = [os.path.splitext(os.path.basename(h5))[0] for h5 in h5_filepaths]
    
    h5_prefixes_all = [h5[:15] for h5 in h5_basenames]
    
    unique_prefixes = sorted(list(set(h5_prefixes_all)))
    
    d = {}
    
    for unique_prefix in progress(unique_prefixes):
        #get orders for the prefix
        #get indices where 
        ixs = [i for i, v in enumerate(h5_basenames) if v[:15]==unique_prefix]
        
        try:
            orders = [int(h5_basenames[ix].split("_")[-1]) for ix in ixs]
            #ingress or egress
            ing_eg = h5_basenames[ixs[0]].split("_")[-2]
            
            d[(unique_prefix, ing_eg)] = {"type":"normal", "n_orders":len(orders), "orders":sorted(orders)}
            continue
        except ValueError:
            pass
        
        if len(ixs) == 1 and h5_basenames[ixs[0]][-2:] == "_S":
            #if fullscan, open and get orders
            
            ix = ixs[0]
            
            ing_eg = h5_basenames[ix].split("_")[-2]
            
            with h5py.File(h5_filepaths[ix], "r") as h5f:
                orders_all = h5f["Channel/DiffractionOrder"][...]
            orders = sorted(list(set(orders_all)))
            d[(unique_prefix, ing_eg)] = {"type":"fullscan", "n_orders":len(orders), "orders":sorted(orders)}                        
    
        elif len(ixs) == 2 and h5_basenames[ixs[0]][-2:] == "_S":
            #if split merged fullscan, open and get orders
            
            
            for ix in ixs:
    
                ing_eg = h5_basenames[ix].split("_")[-2]
    
                with h5py.File(h5_filepaths[ix], "r") as h5f:
                    orders_all = h5f["Channel/DiffractionOrder"][...]
                orders = sorted(list(set(orders_all)))
                d[(unique_prefix, ing_eg)] = {"type":"fullscan", "n_orders":len(orders), "orders":sorted(orders)}                        
        
        else:
            print([h5_basenames[i] for i in ixs])
            
    return d
        

#don't load again if already loaded (takes a few mins to run on HDD)
if "d" not in globals():
    d = make_obs_dict(h5_dir)

# save to csv file for opening in excel
with open("order_combinations.csv", "w") as f:
    
    for k,v in d.items():
        h5_prefix = k[0]
        ing_eg = k[1]
        obstype = v["type"]
        n_orders = v["n_orders"]
        orders = v["orders"]
        orders_txt = ",".join(["%i" %i for i in orders])
        f.write(f"{h5_prefix},{ing_eg},{obstype},{n_orders},{orders_txt}\n")