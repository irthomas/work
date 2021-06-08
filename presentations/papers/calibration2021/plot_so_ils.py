# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:22:26 2020

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
import re

from tools.file.hdf5_functions import make_filelist
from tools.spectra.baseline_als import baseline_als
from tools.general.get_minima_maxima import get_local_minima
from tools.plotting.colours import get_colours
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.file.json import write_json, read_json
from instrument.nomad_so_instrument import nu_mp

# regex = re.compile("20(18|19|20|21)[0-9][0-9]0[1-4]_.*_.*_SO_._[IE]_140")  #order 129 CH4 T=-15
regex = re.compile("20(18|19|20|21)[0-9][0-9][0-9][0-9]_.*_.*_SO_._[IE]_140")  #order 129 CH4 T=-15
file_level = "hdf5_level_1p0a"

READ_FROM_FILE = True


if not READ_FROM_FILE:
    hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)

pixels = np.arange(320.)
m = 140.0

MIN_ALT = 20.0
MAX_ALT = 60.0
MIN_TRANS = 0.3
BIN_TOP = 132
MIN_BAND_DEPTH = 0.99

#for order 140
a1s = 0.0152272 + pixels*(0.000110348)
a2s = 0.0655881 + pixels*(5.24952E-05)
a3s = 0.808981 + pixels*(0.000305451)
a4s = -0.0508321 + pixels*(-0.000346319)
a5s = 0.0447671 + pixels*(0.000421223)
a6s = 0.302995 + pixels*(0.000393483)

colours = get_colours(len(pixels), cmap="viridis")

if not READ_FROM_FILE:
    total_spectra = 0
    
    # fig1, ax1 = plt.subplots(figsize=(8,4))
    
    d = {"x":[], "y":[], "c":[]}
    
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):
        print(file_index, hdf5_filename)
        
        channel = hdf5_filename.split("_")[3].lower()
        
    
        detector_data_all = hdf5_file["Science/Y"][...]
        window_top_all = hdf5_file["Channel/WindowTop"][...]
        binning = hdf5_file["Channel/Binning"][0] + 1
        temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_2_TEMPERATURE_%s" %channel.upper()][1:10])
        alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
        bins = hdf5_file["Science/Bins"][:, 0]
        
        nu = nu_mp(m, pixels, temperature)
        
        y_mean = np.mean(detector_data_all[:, 50:], axis=1)
        
        alt_indices = np.where((alts > MIN_ALT) & (alts < MAX_ALT) & (bins == BIN_TOP) & (y_mean > MIN_TRANS))[0]  #bin3 only
        
        for alt_index in alt_indices:
            y = detector_data_all[alt_index, :]
            
            y_baseline = baseline_als(y)
            y_norm = y / y_baseline
            
            if np.min(y_norm[50:]) < MIN_BAND_DEPTH:
        
                # ax1.plot(y_norm, color="b")
                # ax1.text(0, 0.98, "Alt = %0.1f, T = %0.2f" %(alts[alt_index], y_mean[alt_index]))
                
                minima_indices = get_local_minima(y_norm[60:290]) + 60
                
                minima_indices_deep = np.where(y_norm[minima_indices] < MIN_BAND_DEPTH)[0]
                
                for minimum_index in minima_indices[minima_indices_deep]:
                    # baseline_indices_before = np.where(y_norm[minimum_index-5:minimum_index+1] > 0.999)[0] + minimum_index-5
                    # baseline_indices_after = np.where(y_norm[minimum_index:minimum_index+5] > 0.999)[0] + minimum_index
                    
                    # if len(baseline_indices_before) == 0:
                    baseline_indices_before = [minimum_index - 5]
                    # if len(baseline_indices_after) == 0:
                    baseline_indices_after = [minimum_index + 5]
                    
                    baseline_indices = np.arange(np.max(baseline_indices_before)-4, np.min(baseline_indices_after)+5)
                    baseline_indices_fit = np.arange(np.max(baseline_indices_before)-2, np.min(baseline_indices_after)+3)
                
                    if np.max(baseline_indices) < 320:
    
                        # ax1.scatter(minima_indices[minima_indices_deep], y_norm[minima_indices[minima_indices_deep]], color="k")
                        # ax1.scatter(baseline_indices, y_norm[baseline_indices], color="r")
                        
                        peak = y_norm[baseline_indices]*-1.0 + 1.0
                        
                        x_hr, y_hr, nu_centre = fit_gaussian_absorption(nu[baseline_indices_fit], y_norm[baseline_indices_fit], hr_num=1500)
                        if len(x_hr) > 0:
                            peak_norm = peak / np.max(y_hr*-1.0 + 1.0) * 0.95
                            
                            d["x"].append(nu[baseline_indices] - nu_centre)
                            d["y"].append(peak_norm)
                            d["c"].append(colours[minimum_index])
                            
                            # ax2.scatter(nu[baseline_indices] - nu_centre - 0.0175, peak_norm, color=colours[minimum_index], s=2, alpha=0.1)
                            total_spectra += 1
    
    d["x"] = np.array(d["x"])
    d["y"] = np.array(d["y"])
    d["c"] = np.array(d["c"])
    write_json("so_ils_140.json", d)
    print(total_spectra, "plotted")

else:
    d = read_json("so_ils_140.json")
    d["x"] = np.array(d["x"])
    d["y"] = np.array(d["y"])
    d["c"] = np.array(d["c"])



fig2, ax2 = plt.subplots(figsize=(9,5))
# for i, (x,y,c) in enumerate(zip(d["x"][:1000, :], d["y"][:1000, :], d["c"][:1000, :])):
for i, (x,y,c) in enumerate(zip(d["x"][:, :], d["y"][:, :], d["c"][:, :])):
    if np.mod(i, 100) == 0:
        print("%i/%i" %(i, len(d["x"])))
    ax2.scatter(x - 0.02, y, color=c, s=2, alpha=0.05)


sm = ScalarMappable(norm=plt.Normalize(0., 319.), cmap=plt.get_cmap("viridis"))
sm.set_array([])       
cbar = fig2.colorbar(sm)
cbar.ax.set_ylabel("Pixel number at absorption centre", rotation=270, labelpad=20)

ax2.set_xlabel("Wavenumber relative to absorption centre")
ax2.set_ylabel("Normalised absorption")
ax2.set_ylim([-0.2, 1.2])
ax2.set_xlim([-0.75, 0.75])

#order 140 nominal:
t = 0.0
# p0 = t_p0(t)
nu = np.arange(-1.0, 1.0, 0.001)


# plt.figure(figsize=(8,4))
for i in [80, 120, 160, 200, 240]: 
    a1 = a1s[i]
    a2 = a2s[i]
    a3 = a3s[i]
    a4 = a4s[i]
    a5 = a5s[i]
    a6 = a6s[i]
    
    ils0 = a3 * np.exp(-0.5 * ((nu - a1) / a2) ** 2)
    
    ils1 = a6 * np.exp(-0.5 * ((nu - a4) / a5) ** 2)
    
    ils = ils0 + ils1 
    
    ils_norm = ils / np.max(ils)
    
    px_shift = int(np.argmax(ils_norm) - len(nu)/2)
    nu_shift = nu[px_shift] - nu[0]

    plt.plot(nu - nu_shift, ils_norm, label="Pixel %i" %i, linewidth=3, color=colours[i], path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
    
plt.legend(loc="upper right")
plt.grid()
plt.tight_layout()
plt.savefig("so_ils_order%i.png" %m, dpi=150)
