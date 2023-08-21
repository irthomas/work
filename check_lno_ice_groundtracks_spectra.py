# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:39:40 2023

@author: iant

CHECK LNO GROUNDTRACK FOR ICE OBS DATA
"""


import re
import numpy as np
import matplotlib.pyplot as plt


from tools.file.hdf5_functions import make_filelist, open_hdf5_file

from tools.datasets.tes_albedo import get_TES_albedo_map, get_albedo


# regex = re.compile("20230..._.*_LNO_1_D._133")
# regex = re.compile("20230330_12.*_LNO_1_D._133")

regex = re.compile("20230327_122435_.*_LNO_1_D._132")
# regex = re.compile("20230327_122435_.*_LNO_1_D._133")



DPR = 180./np.pi


def get_lno_data(regex):
    file_level = "hdf5_level_1p0a"
    
    h5_fs, h5s, _ = make_filelist(regex, file_level)
    
    
    d = {"lats":[], "lons":[], "szas":[], "h5s":[], "ixs":[], "ys":[], "xs":[]}
    for file_ix, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):
        
        lats = h5_f["Geometry/Point0/Lat"][:, 0]
        lons = h5_f["Geometry/Point0/Lon"][:, 0]
        szas = h5_f["Geometry/Point0/SunSZA"][:, 0]
        y = h5_f["Science/YReflectanceFactorFlat"][...]
        x = h5_f["Science/X"][...]
        
        d["lats"].extend(list(lats))
        d["lons"].extend(list(lons))
        d["szas"].extend(list(szas))
        d["h5s"].extend([h5 for i in range(len(lats))])
        d["ixs"].extend([i for i in range(len(lats))])
        d["ys"].extend(list(y))
        d["xs"].extend([list(x) for i in range(len(lats))])
    
    for key in ["lats", "lons", "szas", "ixs", "ys", "xs"]:
        d[key] = np.asarray(d[key])
    
    return d

lno_dict = get_lno_data(regex)


#plot all groundtracks and szas
# fig0, ax0 = plt.subplots()
fig0, ax0 = plt.subplots()
scat = ax0.scatter((lno_dict["lons"]), lno_dict["lats"], c=lno_dict["szas"])
ax0.set_title("SZA map")
fig0.colorbar(scat)
for i in range(0, len(lno_dict["lons"]), 50):
    ax0.text(lno_dict["lons"][i], lno_dict["lats"][i], lno_dict["h5s"][i])



aft_ixs = np.where(lno_dict["szas"]<90.)[0]

list(set([lno_dict["h5s"][i] for i in aft_ixs]))


mean_ref = []

for aft_ix in aft_ixs:
    
    x = lno_dict["xs"][aft_ix, :]
    y = lno_dict["ys"][aft_ix, :]
    polyfit = np.polyfit(np.arange(320), y, 4)
    cont = np.polyval(polyfit, np.arange(320))
    
    mean_ref.append(np.mean(cont))
    


albedo_map, albedo_map_extents = get_TES_albedo_map()

# plot lat/lon map with orbits on top
fig1, ax1 = plt.subplots(figsize=(12, 8))
albedo_plot = ax1.imshow(albedo_map, extent=albedo_map_extents, vmin=0.1, vmax=0.4)
# ax1.scatter((lno_dict["lons"]), lno_dict["lats"], c=lno_dict["szas"])

ax1.set_title("MGS/TES albedo with order 132/133 mean reflectance factor overplotted")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_xlim((-180, 180))
ax1.set_ylim((-90, 90))
# cb1 = fig1.colorbar(albedoPlot)
# cb1.set_label("MGS/TES albedo", rotation=270, labelpad=10)
ax1.grid()
fig1.tight_layout()
plt.scatter(lno_dict["lons"][aft_ixs], lno_dict["lats"][aft_ixs], c=mean_ref)
plt.savefig("order132_133_on_tes_albedo.png")

tes_albedos = get_albedo(lno_dict["lons"], lno_dict["lats"], albedo_map)

#plot LNO ref fac vs TES at same location
plt.figure()
plt.plot(mean_ref, label="LNO mean reflectance factor per spectrum")
plt.plot(tes_albedos[aft_ixs]*0.7, label="TES albedo for same location on surface*0.7")
plt.title("LNO mean reflectance factor for order 133 vs TES albedo")
plt.xlabel("Frame number for all frames where SZA<45")
plt.ylabel("Mean LNO reflectance")
plt.legend()


# plot TES vs LNO mean ref fac
plt.figure()
plt.scatter(tes_albedos[aft_ixs], mean_ref, c=lno_dict["szas"][aft_ixs])
plt.colorbar()



fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(lno_dict["xs"][0, :], lno_dict["ys"].T, alpha=0.3)
