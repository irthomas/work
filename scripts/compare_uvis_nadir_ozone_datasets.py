# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:07:27 2026

@author: iant

JM: <70 deg SZA, units precipitable microns?
YW: +60 to -60 latitude, units Dobson (10x pr-um)
"""


from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import spiceypy as sp
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

from tools.spice.load_spice_kernels import load_spice_kernels
from tools.general.progress_bar import progress

yw_file_path = Path(r"C:\Users\iant\Documents\DOCUMENTS\science\yannick_jon_o3\Willame_JGR2025_withO3.txt")
# yw_file_path = Path(r"C:\Users\iant\Documents\DOCUMENTS\science\yannick_jon_o3\snippet.txt")
yw_h5_path = Path(r"C:\Users\iant\Documents\DOCUMENTS\science\yannick_jon_o3\Willame_JGR2025_withO3.h5")
jm_h5_path = Path(r"C:\Users\iant\Documents\DOCUMENTS\science\yannick_jon_o3\20260604_UVIS_ozone_filtered.h5")
comp_h5_path = Path(r"C:\Users\iant\Documents\DOCUMENTS\science\yannick_jon_o3\YW_JM_comparison.h5")

load_spice_kernels(planning=True)

JM_TO_YW_SCALER = 0.1

# %%

if not os.path.exists(comp_h5_path):
    if not os.path.exists(yw_h5_path):
        print("Loading YW file", yw_file_path)
        data = np.genfromtxt(
            yw_file_path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding=None,
            autostrip=True,
        )

        print("Finding columns")
        # columns = {column.strip().lower(): column for column in data.dtype.names}
        orbit_name, lat_in, lon_in, o3_in, ls_in, sza_in, my_in, o3err_in = data["Orbit_name"], data[
            "lat_Â°"], data["lon_Â°"], data["O3_Dobson"], data["Ls_Â°"], data["SZA_Â°"], data["MY"], data["Error_O3_Dobson"]

        def to_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return np.nan

        print("Sorting YW data")
        lat_yw = np.fromiter((to_float(value) for value in lat_in), dtype=float)
        lon_yw = np.fromiter((to_float(value) for value in lon_in), dtype=float)
        o3_yw = np.fromiter((to_float(value) for value in o3_in), dtype=float)
        o3err_yw = np.fromiter((to_float(value) for value in o3err_in), dtype=float)
        ls_yw = np.fromiter((to_float(value) for value in ls_in), dtype=float)
        sza_yw = np.fromiter((to_float(value) for value in sza_in), dtype=float)
        my_yw = np.fromiter((to_float(value) for value in my_in), dtype=float)
        # valid_yw = np.isfinite(lat_yw) & np.isfinite(lon_yw) & np.isfinite(o3_yw) & np.isfinite(ls_yw)

        print("Converting YW datetimes")
        dt_yw = [datetime.strptime(str("%0.6f" % ts_str), "%Y%m%d.%H%M%S") for ts_str in orbit_name]
        # dt_yw = np.asarray(dt_yw)
        ts = np.array([dt.timestamp() for dt in dt_yw], dtype="float64")

        print("Writing YW data to hdf5")
        with h5py.File(yw_h5_path, "w") as f:
            f.create_dataset("lat", dtype=float, data=lat_yw, compression="gzip", shuffle="true")
            f.create_dataset("lon", dtype=float, data=lon_yw, compression="gzip", shuffle="true")
            f.create_dataset("o3", dtype=float, data=o3_yw, compression="gzip", shuffle="true")
            f.create_dataset("o3err", dtype=float, data=o3err_yw, compression="gzip", shuffle="true")
            f.create_dataset("ls", dtype=float, data=ls_yw, compression="gzip", shuffle="true")
            f.create_dataset("sza", dtype=float, data=sza_yw, compression="gzip", shuffle="true")
            f.create_dataset("my", dtype=float, data=my_yw, compression="gzip", shuffle="true")
            # f.create_dataset("dt", dtype=str, data=dt_yw, compression="gzip", shuffle="true")
            f.create_dataset("ts", dtype=float, data=ts, compression="gzip", shuffle="true")

    else:
        print("Loading YW h5 file")
        with h5py.File(yw_h5_path, "r") as f:
            # print(f.keys())
            lat_yw = f["lat"][...]
            lon_yw = f["lon"][...]
            o3_yw = f["o3"][...]
            o3err_yw = f["o3err"][...]
            ls_yw = f["ls"][...]
            sza_yw = f["sza"][...]
            my_yw = f["my"][...]
            ts_yw = f["ts"][...]

        print("Converting YW datetimes")
        dt_yw = np.asarray([datetime.fromtimestamp(ts) for ts in ts_yw])

    print("Loading JM file")
    with h5py.File(jm_h5_path, "r") as f:
        # print(f.keys())

        # ls_jm = f["lsubs"][...]
        et_jm = f["ephemeris_time"][...]
        lat_jm = f["latitude"][...]
        lst_jm = f["lst"][...]
        o3_jm = f["ozone"][...]
        lon_jm = f["longitude"][...]

    print("Converting JM datetimes")
    dt_jm = [datetime.fromisoformat(sp.et2utc(et, "ISOC", 0)) for et in et_jm]
    dt_jm = np.asarray(dt_jm)

    print("Matching dates, lats and lons")
    # ixs_yw = np.arange(100000)
    ixs_yw = np.arange(len(dt_yw))
    last_ix_rough = 14500  # this needs to be specified - first index of YW observation in JM file minus a few hundred

    d = {"dt": [], "my": [], "ls": [], "lat": [], "lon": [], "o3_yw": [], "o3_jm": []}

    # loop through yw points, check if matching in JM data
    for ix_yw in progress(ixs_yw):
        # get approx matching index of YW observation in JM file
        # ix_rough = np.argmin(np.abs(dt_jm - dt_yw[ix_yw]))
        ix_rough = np.argmin(np.abs(dt_jm[last_ix_rough:last_ix_rough + 1000] - dt_yw[ix_yw])) + last_ix_rough

        # check JM data around this approx index
        ixs_rough = np.arange(ix_rough - 20, ix_rough + 20, 1)

        # get exact matching JM index
        ix = np.argmin(np.abs(lat_jm[ixs_rough] - lat_yw[ix_yw]) + np.abs(lon_jm[ixs_rough] - lon_yw[ix_yw])) + ixs_rough[0]

        # save first index for next iteration
        last_ix_rough = ixs_rough[0]

        # check if lat and lon match, if so save the match
        if np.abs(lat_jm[ix] - lat_yw[ix_yw]) < 0.1 and np.abs(lon_jm[ix] - lon_yw[ix_yw]) < 0.1:
            d["dt"].append(dt_yw[ix_yw])
            d["my"].append(my_yw[ix_yw])
            d["ls"].append(ls_yw[ix_yw])
            d["lat"].append(lat_yw[ix_yw])
            d["lon"].append(lon_yw[ix_yw])
            d["o3_yw"].append(o3_yw[ix_yw])
            d["o3_jm"].append(o3_jm[ix])
            # print("Found:", ix, dt_jm[ix], dt_yw[ix_yw], "%0.2f" % (lat_jm[ix] - lat_yw[ix_yw]),
            #       "%0.2f" % (lon_jm[ix] - lon_yw[ix_yw]), "%0.4f" % (o3_yw[ix_yw]), "%0.4f" % (o3_jm[ix] * JM_TO_YW_SCALER))
        # else:
            # print("Not:", ix, "lat=", lat_yw[ix_yw], "sza=", sza_yw[ix_yw], "dt=", dt_jm[ix], dt_yw[ix_yw],
            #       lat_jm[ix] - lat_yw[ix_yw], lon_jm[ix] - lon_yw[ix_yw], o3_yw[ix_yw], o3_jm[ix] / 7)

    # convert matches to arrays
    for key in d.keys():
        d[key] = np.asarray(d[key])

    print("Writing matches to hdf5")
    ts = np.array([dt.timestamp() for dt in d["dt"]], dtype="float64")
    with h5py.File(comp_h5_path, "w") as f:
        f.create_dataset("lat", dtype=float, data=d["lat"], compression="gzip", shuffle="true")
        f.create_dataset("lon", dtype=float, data=d["lon"], compression="gzip", shuffle="true")
        f.create_dataset("o3_yw", dtype=float, data=d["o3_yw"], compression="gzip", shuffle="true")
        f.create_dataset("o3_jm", dtype=float, data=d["o3_jm"], compression="gzip", shuffle="true")
        f.create_dataset("ls", dtype=float, data=d["ls"], compression="gzip", shuffle="true")
        f.create_dataset("my", dtype=float, data=d["my"], compression="gzip", shuffle="true")
        f.create_dataset("ts", dtype=float, data=ts, compression="gzip", shuffle="true")

else:
    print("Loading comparison h5 file")
    d = {}
    with h5py.File(comp_h5_path, "r") as f:
        # print(f.keys())
        d["lat"] = f["lat"][...]
        d["lon"] = f["lon"][...]
        d["o3_yw"] = f["o3_yw"][...]
        d["o3_jm"] = f["o3_jm"][...]
        d["ls"] = f["ls"][...]
        d["my"] = f["my"][...]
        # ts = f["ts"][...]

# %%

# make list of MY Ls start end combinations
start_id = 34
start_deg = 160
step = 20
num_items = 100

my_ls = [
    [
        start_id + (start_deg + i * step) // 360,
        [(start_deg + i * step) % 360, (start_deg + i * step) % 360 + step],
    ]
    for i in range(num_items)
]

print("Making plots from", my_ls[0], "to", my_ls[-1])
# loop through combinations
with PdfPages("uvis_nadir_ozone_comparison.pdf") as pdf:
    for my, ls in progress(my_ls):
        ixs = np.where((d["my"] == my) & (d["ls"] > ls[0]) & (d["ls"] < ls[1]))[0]

        # if data, plot and save figure to pdf
        if len(ixs) > 0:
            plt.figure(figsize=(10, 6))
            plt.title("Matched spectra MY%i Ls %0.1f to MY%i Ls %0.1f" % (my, ls[0], my, ls[1]))
            plt.plot(d["o3_yw"][ixs], label="YW Nadir O3 (Dobson)", alpha=0.3, rasterized=True)
            plt.plot(d["o3_jm"][ixs] * JM_TO_YW_SCALER, label="JM Nadir O3 * %0.2f" % JM_TO_YW_SCALER, alpha=0.5, rasterized=True)
            plt.legend()
            plt.xlabel("Matching spectrum index")
            plt.ylabel("O3 (YW Dobson, JM * %0.2f)" % JM_TO_YW_SCALER)
            plt.grid()

            pdf.savefig()

            plt.figure(figsize=(10, 6))
            plt.title("Matched spectra MY%i Ls %0.1f to MY%i Ls %0.1f" % (my, ls[0], my, ls[1]))
            plt.scatter(d["o3_jm"][ixs] * JM_TO_YW_SCALER, d["o3_yw"][ixs], c=d["lat"][ixs], cmap="brg", alpha=0.1, marker=".", rasterized=True)
            plt.plot([0.0, 5.0], [0.0, 5.0], "k--")
            cbar = plt.colorbar(label="Latitude (degrees)")
            cbar.solids.set_alpha(1)
            plt.xlabel("JM Nadir O3 * %0.2f" % JM_TO_YW_SCALER)
            plt.ylabel("YW Nadir O3 (Dobson)")
            plt.grid()

            pdf.savefig()

            plt.figure(figsize=(10, 6))
            plt.title("Matched spectra (zoomed) MY%i Ls %0.1f to MY%i Ls %0.1f" % (my, ls[0], my, ls[1]))
            plt.scatter(d["o3_jm"][ixs] * JM_TO_YW_SCALER, d["o3_yw"][ixs], c=d["lat"][ixs], cmap="brg", alpha=0.1, marker=".", rasterized=True)
            plt.plot([0.0, 5.0], [0.0, 5.0], "k--")
            cbar = plt.colorbar(label="Latitude (degrees)")
            cbar.solids.set_alpha(1)
            plt.xlabel("JM Nadir O3 * %0.2f" % JM_TO_YW_SCALER)
            plt.ylabel("YW Nadir O3 (Dobson)")
            plt.xlim([-0.1, 0.5])
            plt.ylim([-0.1, 0.5])
            plt.grid()

            pdf.savefig()

            plt.close('all')

# print("Plotting Ls", np.min(ls), "to", np.max(ls))
# fig, ax = plt.subplots(figsize=(9, 6))
# plot = ax.scatter(lon[valid], lat[valid], c=o3[valid], cmap="viridis", vmin=0, vmax=2)
# fig.colorbar(plot, ax=ax, label="O3")
# ax.set(xlabel="Longitude", ylabel="Latitude", title="UVIS Nadir O3 (Yannick), Ls %0.1f to %0.1f" % (np.min(ls), np.max(ls)))
# plt.tight_layout()
# plt.show()
