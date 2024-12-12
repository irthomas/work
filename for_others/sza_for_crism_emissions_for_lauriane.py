# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:45:33 2024

@author: iant

CALCULATE CRISM LIMB SZAS FOR LAURIANE

"""


# from tools.spice.get_mars_year_ls import get_mars_year_ls
# from tools.spice.load_spice_kernels import load_spice_kernels
# from tools.general.progress_bar import progress
from tools.general.get_minima_maxima import get_local_maxima

import spiceypy as sp
import numpy as np

from datetime import datetime, timedelta

import matplotlib.pyplot as plt

# load only needs SPICE kernels
sp_paths = [
    "C:/Users/iant/Documents/DATA/local_spice_kernels/de421.bsp",
    "C:/Users/iant/Documents/DATA/local_spice_kernels/kernels/pck/pck00010.tpc",
    "C:/Users/iant/Documents/DATA/local_spice_kernels/kernels/lsk/naif0012.tls",
]
for sp_path in sp_paths:
    sp.furnsh(sp_path)

# load the CRISM file
FILEPATH = r"C:/Users/iant/Downloads/Clancy2017_volume_emission_rates_o2_singlet_delta.csv"
# columns are: MY, Ls, Lat, Lon (0-360), LST, ?, data, data, altitude
# 29 ,  301.332 ,   -86.98 ,   222.00 ,    18.69 ,     0.60 ,   0.3583E+07 ,   0.7681E+06 ,    12.87
with open(FILEPATH, "r") as f:
    lines_str = f.readlines()

# strip whitespace and conver to float array
lines = [[float(s.strip()) for s in line.split(",")] for line in lines_str]
lines = np.asarray(lines)

# get start and end MY/Ls dates
start_my = lines[0, 0:2]
end_my = lines[-1, 0:2]

print("Clancy et al. time range", start_my, "to", end_my)

# create a grid of evenly spaced datetimes and ephemeris times encompassing the whole observation time range
# instead of looping through each row in the file to calculate the time, do all at once using interpolation
# note that these should be changed if a different dataset is loaded!
start_dt = datetime(2009, 7, 1)  # start 1st July 2009
end_dt = datetime(2019, 1, 1)
start_my = 29  # must give Martian year corresponding to start_dt


delta = 600.0
n_times = (end_dt - start_dt).total_seconds() / delta

dts = [start_dt + timedelta(seconds=delta*i) for i in np.arange(n_times)]
utcs = [datetime.strftime(dt, "%Y-%m-%dT%H:%M:%S") for dt in dts]
ets = [sp.utc2et(utc) for utc in utcs]  # grid of ephemeris times


print("Calculating MY Ls for reference grid")
"""either 1) use SPICE function lspcn"""
ls_only = [sp.lspcn("MARS", et, "NONE") * sp.dpr() for et in ets]

# function only gives Ls - find discontinuities to get the MY transitions
maxima = get_local_maxima(ls_only)  # get year transition indices

# find correct MY for each Ls in the file
mys_only = np.searchsorted(maxima, np.arange(len(ls_only))) + start_my
mys = np.stack((mys_only, ls_only)).T  # output MY and Ls together

"""or 2) use 2015 paper function - gives MY and Ls together"""
# mys = [get_mars_year_ls(dt) for dt in dts]


mys = np.asarray(mys)
print("Calculation grid time range", mys[0, :], "to", mys[-1, :])  # check it's good

# combine MY and Ls to get a float for each grid point
lss = mys[:, 0] * 360.0 + mys[:, 1]

# combine MY and Ls to get a float for each row in the file
lss_clancy = lines[:, 0] * 360.0 + lines[:, 1]


print("Interpolating times in the file onto the grid to get approximate ephemeris times for each row in the file")
ets_clancy = np.interp(lss_clancy, lss, ets)

print("Checking that the local solar times approximately match")
lons_clancy = lines[:, 3]  # longitudes from the file
# use spice to calculate the local solar time
lsts_raw = [sp.et2lst(et, 499, lon/sp.dpr(), "PLANETOGRAPHIC") for et, lon in zip(ets_clancy, lons_clancy)]
# convert output to hours
lsts = [lst[0] + lst[1]/60 + lst[2]/3600 for lst in lsts_raw]

# now compare calculated local solar times to those in the file
lsts_clancy = lines[:, 4]  # LSTs from the file

# if the difference is greater than 0.1 hours then there is a problem with that row
bad_ixs = np.where(np.abs(lsts - lsts_clancy) > 0.1)[0]

plt.figure()
plt.title("Error in Local Solar Time from MY/Ls calculation of the UTC")
plt.ylabel("LST Error")
plt.plot(lsts - lsts_clancy, label="Before LST correction")


# correct for the ephemeris time uncertainty by using the lst
delta_lsts = lsts - lsts_clancy  # difference between file and calculation (hours)
seconds_per_hour = (24. * 3600. + 39. * 60. + 35.) / 24.  # seconds per hour on Mars
delta_ts = seconds_per_hour * delta_lsts  # delta time required to make the LSTs match exactly (seconds)

# apply the delta time to correct the ephemeris times
ets_corr = ets_clancy - delta_ts

# repeat the spice local time calculation and convert to hours
lsts_raw_corr = [sp.et2lst(et, 499, lon/sp.dpr(), "PLANETOGRAPHIC") for et, lon in zip(ets_corr, lons_clancy)]
lsts_corr = [lst[0] + lst[1]/60 + lst[2]/3600 for lst in lsts_raw_corr]

# plot the time-corrected LSTs to check that the local times match the file local solar times
plt.plot([lsts_corr[i] - lsts_clancy[i] for i in range(len(lsts_corr)) if i not in bad_ixs], label="After LST correction")
plt.legend()

# if all good, calculate the illumination angles and sza using the corrected ephemeris times
lats_clancy = lines[:, 2]  # get latitudes from file

# calculate tangent point cartesians (radius can be anything)
tanpnts = [sp.latrec(4000.0, lon / sp.dpr(), lat / sp.dpr()) for lon, lat in zip(lons_clancy, lats_clancy)]

# calculate the illumination geometry parameters. The observer can be anything as we only need solar incidence angle
illumins = [sp.ilumin("ELLIPSOID", "MARS", et, "IAU_MARS", "NONE", "SUN", tanpnt) for et, tanpnt in zip(ets_corr, tanpnts)]
# output is trgepc, srfvec, phase, incdnc, emissn. We just need the incidence in degrees converted to degrees
szas = [illumin[3] * sp.dpr() for illumin in illumins]

# now we have the SZAs we need, write out the file, adding an extra two colums: SZA and error
lines_new = []
for i, (line_str, sza) in enumerate(zip(lines_str, szas)):
    if i in bad_ixs:
        # if the line contains an error, add a comment
        comment = "***BAD GEOMETRY***"
    else:
        comment = ""
    lines_new.append("%s, %0.3f, %s\n" % (line_str, sza, comment))

# save output file
OUT_FILEPATH = r"C:/Users/iant/Downloads/Clancy2017_volume_emission_rates_o2_singlet_delta_sza.csv"
with open(OUT_FILEPATH, "wb") as f:
    for line in lines_new:
        f.write(line)

# plot solar zenith angles
plt.figure()
plt.title("Solar Zenith Angles")
plt.ylabel("SZA (degrees)")
plt.plot(szas)
