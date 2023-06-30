# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:15:40 2023

@author: iant

GET ALL ORDER COMBINATIONS FROM ACTUAL MEASUREMENTS


"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages



HDF5_DIRECTORY = r"E:\DATA\hdf5\hdf5_level_1p0a"
h5s = sorted(glob.glob(HDF5_DIRECTORY+r"\**\*.h5", recursive=True))


d = {}

for h5 in h5s:
    h5_basename = os.path.basename(h5)
    h5_prefix = h5_basename[0:27]
    
    if "SO_A" not in h5_basename:
        continue
    
    order = h5_basename.replace(".h5","").split("_")[-1]
    try:
        order = int(order)
    except Exception:
        continue
    
    if h5_prefix not in d.keys():
        d[h5_prefix] = [order]
    
    else:
        d[h5_prefix].append(order)
        

#reverse dictionary
d2 = {}

for h5_prefix, orders in d.items():
    
    orders = tuple(orders)
    if orders not in d2.keys():
        d2[orders] = [datetime.strptime(h5_prefix[0:15], "%Y%m%d_%H%M%S")]
    else:
        d2[orders].append(datetime.strptime(h5_prefix[0:15], "%Y%m%d_%H%M%S"))
        
        
#plot and save to pdf
with PdfPages("order_combinations_for_zachary.pdf") as pdf:
    
    for orders, dts in d2.items():
        
        n_obs = len(dts)
        if n_obs < 100:
            continue
        
        plt.figure(figsize=(15, 6), constrained_layout=True)
        plt.grid()
        plt.xlabel("Observation date")
        plt.ylabel("Diffraction order")
        
        order_str = ", ".join(["%i" %i for i in orders])
        plt.title("SO order combination %s measured %i times" %(order_str, n_obs))
            
        for order in orders:
            plt.scatter(dts, np.zeros(len(dts))+order, s=2, c="k", alpha=1.0, rasterized=True)
            
        plt.xlim([datetime(2018, 1, 1), datetime(2024, 1, 1)])
        plt.ylim([110, 210])

        pdf.savefig()
        plt.close()
    
