# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:49:18 2023

@author: iant

READ IN PHOBOS JSON OUTPUT AND ANALYSE
"""

import numpy as np
import matplotlib.pyplot as plt
import json

from tools.datasets.get_phobos_crism_data import get_phobos_crism_data


crism_d = get_phobos_crism_data()



with open("lno_phobos_output.json", "r") as f:
    phobos_d = json.load(f)

for h5 in phobos_d.keys():
    for key in phobos_d[h5].keys():
        phobos_d[h5][key] = np.asarray(phobos_d[h5][key])

fig1, ax1a = plt.subplots(figsize=(12, 5))
fig1.suptitle("Phobos Radiometric Scaling")

fig2, ax2a = plt.subplots(figsize=(12, 5))
fig2.suptitle("Phobos Radiometric Scaling")
ax2a.scatter(crism_d["x"], crism_d["phobos_red"], color="tab:red", marker="x", alpha=0.7, label="CRISM Phobos red")
ax2a.scatter(crism_d["x"], crism_d["phobos_blue"], color="tab:blue", marker="x", alpha=0.7, label="CRISM Phobos blue")


nominal_orders = np.asarray([160, 162, 164, 166, 168, 170])
nominal_um = np.asarray([
    2.769762434224507519,
    2.735567836271118658,
    2.702207252901958512,
    2.669650539011573454,
    2.637868984975721531,
    2.606835232211300646,
])


interps = []

for h5 in phobos_d.keys():
    orders = phobos_d[h5]["orders"]
    norm_vals = phobos_d[h5]["norm_vals"]
    norm_vals_norm = norm_vals/np.max(np.mean(norm_vals[-3:]))
    norm_vals_interp = np.interp(nominal_orders, orders, norm_vals_norm)

    ax1a.plot(nominal_um, norm_vals_interp)
    
    interps.append(norm_vals_interp)

interps = np.asarray(interps)

interp_mean = np.mean(interps, axis=0)
interp_std = np.std(interps, axis=0)

ax1a.errorbar(nominal_um, y=interp_mean, yerr=interp_std, color="k")


order_ums = [
    2.769762434224507519,
    2.669650539011573454,
    2.637868984975721531,
    2.606835232211300646,
]
order_ixs = [0,3,4,5]

#calculate scaling factor to calibrate to crism red
matching_crism_indices = np.where((crism_d["x"] > np.min(order_ums)) & (crism_d["x"] < np.max(order_ums)))[0]
crism_red_mean = np.mean(crism_d["phobos_red"][matching_crism_indices])
crism_blue_mean = np.mean(crism_d["phobos_blue"][matching_crism_indices])
lno_mean = np.mean(interp_mean[order_ixs])

red_scalar = crism_red_mean / lno_mean
blue_scalar = crism_blue_mean / lno_mean

y_column_mean_norm_red = interp_mean*red_scalar
y_column_mean_norm_blue = interp_mean*blue_scalar
y_column_std_norm_red = interp_std*red_scalar
y_column_std_norm_blue = interp_std*blue_scalar


x_plt = nominal_um

y_plt1 = interp_mean*red_scalar
y_err1 = interp_std*red_scalar


# ax2a.fill_between(x_plt, y1=y_plt1 - y_err1, y2=y_plt1 + y_err1, color="darkred", alpha=0.5)
ax2a.errorbar(x_plt, y=y_plt1, yerr=y_err1, color="darkred", capsize=2, label="LNO scaled to Phobos red")
ax2a.scatter(x_plt, y_plt1, color="darkred")

y_plt2 = interp_mean*blue_scalar
y_err2 = interp_std*blue_scalar


# ax2a.fill_between(x_plt, y1=y_plt2 - y_err2, y2=y_plt2 + y_err2, color="darkblue", alpha=0.5)
ax2a.errorbar(x_plt, y=y_plt2, yerr=y_err2, color="darkblue", capsize=2, label="LNO scaled to Phobos blue")
ax2a.scatter(x_plt, y_plt2, color="darkblue")

ax2a.grid()
ax2a.legend()
ax2a.set_ylim((0.0, 0.1))
ax2a.set_xlabel("Wavelength (microns)")
ax2a.set_ylabel("CRISM Phobos I/F (Fraeman 2014)")
fig2.subplots_adjust(bottom=0.15)





    
for h5 in phobos_d.keys():
    
    snr = phobos_d[h5]["norm_vals"] / phobos_d[h5]["norm_stds"]
    um = phobos_d[h5]["um"]
    
    orders = phobos_d[h5]["orders"]
    norm_vals = phobos_d[h5]["norm_vals"]
    norm_vals_norm = norm_vals/np.max(np.mean(norm_vals[-3:]))
    norm_vals_interp = np.interp(nominal_orders, orders, norm_vals_norm)

    ax1a.plot(nominal_um, norm_vals_interp)


    # red_scaled = phobos_d[h5]["red_scaled"]
    # red_err_scaled = red_scaled / snr
    
    # blue_scaled = phobos_d[h5]["blue_scaled"]
    # blue_err_scaled = blue_scaled / snr


    
    # ax2a.errorbar(um, y=red_scaled, yerr=red_err_scaled, color="darkred", capsize=2, label=h5, alpha=0.5)
    # ax2a.scatter(um, red_scaled, color="darkred", alpha=0.5)

    # ax2a.errorbar(um, y=blue_scaled, yerr=blue_err_scaled, color="darkblue", capsize=2, label=h5, alpha=0.5)
    # ax2a.scatter(um, blue_scaled, color="darkblue", alpha=0.5)

