# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:34:48 2023

@author: iant

PLOT CO2 AND H2O ICE PROFILES
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from tools.file.paths import paths
from tools.plotting.colours import get_colours

ref_dir = paths["REFERENCE_DIRECTORY"]


h2o_filename = "hri_waterice_warren_ice.txt" #"hri_waterice_afcrl_ice.txt"
co2_filename = "gsfc_co2ice_hansen.txt"

h2o_filepath = os.path.join(ref_dir, h2o_filename)
co2_filepath = os.path.join(ref_dir, co2_filename)



def get_snicar_output(size_um, ice, slab_cm=10, um_range=[2., 3.8]):
    


    filename = "snicar_%ium_%icm_%s.txt" %(size_um, slab_cm, ice)
    filepath = os.path.join(ref_dir, filename)
    
    data = np.loadtxt(filepath, delimiter="\t", usecols=[0,1])
    
    ixs = np.where((data[:, 0] > um_range[0]) & (data[:, 0] < um_range[1]))[0]
    # print(data[:, 0])
    
    d = {"snicar_um":data[ixs, 0], size_um:data[ixs, 1]}

    return d


def get_data_from_file(filepath, ice, out="ssa", um_range=[2., 3.8], slab_cm=10):

    reffs, column_numbers = {"ssa":[
                      [0.01, 0.05, 0.10, 0.50, 1.00, 5.00, 10.00, 50.00, 100.00],
                      [2, 5, 8, 11, 14, 17, 20, 23, 26]
                      ]}[out]    

    with open(filepath, "r") as f:
        d = {reff:[] for reff in reffs}
        d["um"] = []
        for line in f.readlines():
            if line[0] != "#":
                line_split = line.split()
                floats_split = [float(f) for f in line_split]
                if floats_split[0] > um_range[0] and floats_split[0] < um_range[1]:
                    d["um"].append(floats_split[0])
                    for reff, column_number in zip(reffs, column_numbers):
                        d[reff].append(floats_split[column_number])
                    
                    # data.append([float(line_split[0]), float(line_split[column_number])])

    for key in d.keys():
        d[key] = np.asfarray(d[key])

    for size_um in [250, 500, 750, 1000, 1500]:
        snicar_d = get_snicar_output(size_um, ice)
        d["snicar_um"] = snicar_d["snicar_um"]
        d[size_um] = snicar_d[size_um]
    
    
    
    return d




h2o_d = get_data_from_file(h2o_filepath, "h2o")
co2_d = get_data_from_file(co2_filepath, "co2")

# ice = "h2o"
# for size_um in [250, 500, 750, 1000, 1500]:
#     d = get_snicar_output(size_um, ice)
#     h2o_d["snicar_um"] = d["snicar_um"]
#     h2o_d[size_um] = d["snicar_%i" %size_um]

# ice = "co2"
# for size_um in [250, 500, 750, 1000, 1500]:
#     d = get_snicar_output(size_um, ice)
#     co2_d["snicar_um"] = d["snicar_um"]
#     co2_d[size_um] = d["snicar_%i" %size_um]


colours = get_colours(len(h2o_d.keys())-2)

title = os.path.splitext(h2o_filename)[0]
plt.figure(figsize=(12, 7), constrained_layout=True)
plt.title(title)
plt.xlabel("Wavenumber cm-1")
plt.ylabel("SSA / Reflectance")
loop = 0
for key in h2o_d.keys():
    if "um" not in str(key):
        if key > 200:
            plt.plot(10000./h2o_d["snicar_um"], h2o_d[key]/0.075787, color=colours[loop], label="H2O %sum" %key)
            loop += 1
        elif key > 0.005:
            plt.plot(10000./h2o_d["um"], h2o_d[key], color=colours[loop], label="H2O %sum" %key)
            loop += 1

plt.fill_between([min(10000./h2o_d["um"]-100), 2696], y1=[0., 0.], y2=[1., 1.], color="gray", alpha=0.3, label="Low SNR (orders <120, 140-166)")
plt.fill_between([3147, 3753], y1=[0., 0.], y2=[1., 1.], color="gray", alpha=0.3)
plt.fill_between([3753, 4667], y1=[0., 0.], y2=[1., 1.], color="blue", alpha=0.3, label="Best SNR (orders 167-206)")

plt.grid()
plt.legend()
plt.savefig(title+".png")




title = os.path.splitext(co2_filename)[0]
plt.figure(figsize=(12, 7), constrained_layout=True)
plt.title(title)

plt.xlabel("Wavenumber cm-1")
plt.ylabel("SSA / Reflectance")
loop = 0
for key in co2_d.keys():
    if "um" not in str(key):
        if key > 200:
            plt.plot(10000./co2_d["snicar_um"], co2_d[key], color=colours[loop], label="CO2 %sum" %key)
            loop += 1
        elif key > 0.005:
            plt.plot(10000./co2_d["um"], co2_d[key], color=colours[loop], label="CO2 %sum" %key)
            loop += 1

plt.fill_between([min(10000./co2_d["um"]-100), 2696], y1=[0., 0.], y2=[1., 1.], color="gray", alpha=0.3, label="Low SNR (orders <120, 140-166)")
plt.fill_between([3147, 3753], y1=[0., 0.], y2=[1., 1.], color="gray", alpha=0.3)
plt.fill_between([3753, 4667], y1=[0., 0.], y2=[1., 1.], color="blue", alpha=0.3, label="Best SNR (orders 167-206)")

plt.grid()
plt.legend()
plt.savefig(title+".png")
