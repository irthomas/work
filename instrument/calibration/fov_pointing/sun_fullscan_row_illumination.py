# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:59:49 2021

@author: iant

CHECK FULLSCAN SUN POINTING OVER TIME
"""

import re
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
import h5py
from datetime import datetime, timedelta

from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours
from tools.file.hdf5_filename_to_datetime import hdf5_filename_to_datetime
from tools.spectra.non_uniform_savgol import non_uniform_savgol

import spiceypy as sp
from tools.spice.load_spice_kernels import load_spice_kernels



load_spice_kernels()


#SAVE_FIGS = False
SAVE_FIGS = True


regex = re.compile("........_......_.*_SO_1_S")
file_level = "hdf5_level_0p2a"


hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level, silent=True, open_files=False)

if "d" not in locals():
    d = {"filenames":[], "rows":[], "y_rows":[]}
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
        
        # if np.mod(file_index, 100) == 0:
        #     print("%i/%i" %(file_index, len(hdf5_filenames)))
    
        with h5py.File(hdf5_file, "r") as h5:
            bins = h5["Science/Bins"][:, 0]
            alts = h5["Geometry/Point0/TangentAltAreoid"][:, 0]
            y_all = h5["Science/Y"][...]
            aotfs = h5["Channel/AOTFFrequency"][:]
            
            
    
        #check if slow fullscan
        unique_bins = sorted(list(set(bins)))
        if len(unique_bins) != 24:
            # print("Skipping")
            continue
        
        
        #check diffraction orders
        
        
        #check if ingress or egress        
        if alts[0] > alts[-1]:
            #if ingress
            start_index = 48
        else:
            start_index = len(alts) - 48
            
        #check bin start
        if bins[start_index] != 116:
            continue
        
        #check altitude
        if alts[start_index] < 250:
            print("Error: altitude too low")
            continue
        
        print("%s," %hdf5_filename)
        # print()
        # print()
            
        indices = np.arange(start_index, start_index+24, 1)
        
            
        y_mean = np.mean(y_all[indices, 190:210], axis=1)
    
        d["filenames"].append(hdf5_filename)
        d["y_rows"].append(y_mean/max(y_mean))
        d["rows"].append(bins[indices])



        
colours = get_colours(len(d["rows"]))

fwhms = []
obs_datetimes = []
fw_centres = []

plt.figure()
for i, (filename, rows, y_rows) in enumerate(zip(d["filenames"], d["rows"], d["y_rows"])):
    # if i == 50:
        
        if y_rows[0] > 0.5 or y_rows[-1] > 0.5:
            continue
        plt.plot(y_rows, rows, color=colours[i], label=filename)
        x1a = np.searchsorted(y_rows[0:12], 0.5, side="left") - 1
        x1b = x1a+2
        x2a = np.searchsorted(-y_rows[12:24], -0.5, side="left") - 1 + 12
        x2b = x2a+2
        
        x1 = np.polyval(np.polyfit(y_rows[x1a:x1b], rows[x1a:x1b], 1), y_rows[x1a:x1b])
        x2 = np.polyval(np.polyfit(y_rows[x2a:x2b], rows[x2a:x2b], 1), y_rows[x2a:x2b])
        
        fwhm1 = np.polyval(np.polyfit(y_rows[x1a:x1b], rows[x1a:x1b], 1), 0.5)
        fwhm2 = np.polyval(np.polyfit(y_rows[x2a:x2b], rows[x2a:x2b], 1), 0.5)
        
        plt.plot(y_rows[x1a:x1b], x1, "k--")
        plt.plot(y_rows[x2a:x2b], x2, "k--")
        plt.scatter([0.5, 0.5], [fwhm1, fwhm2], c="k")
        
        fwhms.append(fwhm2-fwhm1)
        fw_centres.append(np.mean([fwhm1, fwhm2]))
        
        dt = hdf5_filename_to_datetime(filename)
        obs_datetimes.append(dt)

plt.xlabel("Normalised illumination pattern")
plt.ylabel("Detector row")
plt.title("SO detector row illumination")
plt.grid()
if SAVE_FIGS:
    plt.savefig("so_detector_rows_illumination.png")


#set up subplots
fig = plt.figure(figsize=(15, 7), constrained_layout=True)
gs = fig.add_gridspec(3, 1)
ax1a = fig.add_subplot(gs[0, 0])
ax1b = fig.add_subplot(gs[1, 0], sharex=ax1a)
ax1c = fig.add_subplot(gs[2, 0], sharex=ax1a)






abcorr="None"
ref="J2000"
observer="-143" #observer
target = "SUN"


datetime_start = datetime(year=2018, month=4, day=21)
datetimes = [datetime_start + timedelta(days=x) for x in range((obs_datetimes[-1]-obs_datetimes[0]).days)]

date_strs = [datetime.strftime(x, "%Y-%m-%d") for x in datetimes]
date_ets = [sp.str2et(x) for x in date_strs]
tgo_pos = np.asfarray([sp.spkpos(target, time, ref, abcorr, observer)[0] for time in list(date_ets)])



tgo_dist = la.norm(tgo_pos,axis=1)
code = sp.bodn2c(target)
pradii = sp.bodvcd(code, 'RADII', 3) # 10 = Sun
sun_radius = pradii[1][0]
sun_diameter_arcmins = np.arctan(sun_radius/tgo_dist) * sp.dpr() * 60.0 * 2.0




ax1a.plot_date(datetimes, sun_diameter_arcmins, linestyle="-", ms=0)
ax1b.scatter(obs_datetimes, fwhms)
ax1c.scatter(obs_datetimes, fw_centres)

dt_start = obs_datetimes[0]
seconds = [(i - dt_start).total_seconds() for i in obs_datetimes]

savgol_centre = non_uniform_savgol(seconds, fw_centres, 49, 2)
ax1c.plot(obs_datetimes, savgol_centre)



ax1a.set_ylabel("Solar diameter as seen\nfrom TGO (arcminutes)")
ax1b.set_ylabel("FWHM illuminated rows")
ax1c.set_ylabel("Detector row FWHM centre")
ax1c.set_xlabel("Observation Date")

ax1a.set_title("Apparent diameter of Sun")
ax1b.set_title("Number of illuminated rows (FWHM)")
ax1c.set_title("Centre of illuminated rows")

ax1a.xaxis.set_major_locator(MonthLocator(bymonth=None, interval=3, tz=None))    
ax1a.grid()
ax1b.grid()
ax1c.grid()


if SAVE_FIGS:
    fig.savefig("so_detector_illumination.png", dpi=300)
   
