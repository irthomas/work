# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 09:00:22 2026

@author: iant

QUANTIFY IF ANY CHANGES IN NOMAD SO DARK CURRENT
"""


import re
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist2
from tools.plotting.colours import get_colours

regex = re.compile("20....0._......_0p1a_SO_1")
file_level = "hdf5_level_0p1a"
bin_number = 1

# h5fs, h5s, _ = make_filelist2(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")

# h5f = h5fs[0]
# h5 = h5s[0]

for i, (h5f, h5) in enumerate(zip(h5fs, h5s)):

    aotf = h5f["Channel/AOTFFrequency"][...]
    unique_aotfs = list(set(aotf))

    if 0.0 in unique_aotfs:
        # not background subtracted
        y = h5f["Science/Y"][...]
        y_mean = np.mean(y, axis=2)

        unique_aotfs = [0.0]

        light_starts = []

        if np.mean(y_mean[0:10]) < np.mean(y_mean[:-10]):
            # egress only

            for unique_aotf in unique_aotfs:
                ixs = np.where(aotf == unique_aotf)[0]

                diff = np.diff(y_mean[ixs, bin_number])
                diff_norm = diff/np.max(diff)

                light_starts.append(np.argmin(diff_norm < 0.05))

            light_start = np.min(light_starts)
            if light_start < 13:
                print("Skipping, light_start=", light_start, h5, i, "/", len(h5s))
            elif h5f["Channel/WindowTop"][0] != 120:
                print("Skipping, WindowTop=", h5f["Channel/WindowTop"][0], h5)
            else:
                temps_all = np.asarray([h5f["Housekeeping/SENSOR_%i_TEMPERATURE_SO" % i][1:] for i in range(1, 4)])
                temps = np.median(temps_all, axis=0)

                temp_start_ix = h5f["Telecommand20/SOStartScience1"][...] - 40
                temp = temps[temp_start_ix]
                for unique_aotf in unique_aotfs:

                    ixs = np.where(aotf == unique_aotf)[0]

                    dark_spectra = y[ixs[5:(light_start-5)], bin_number, :]

                    dark_offset = np.mean(dark_spectra, axis=1)

                    darks = dark_spectra - dark_offset[:, None]
                    # plt.figure()
                    # plt.title(unique_aotf)
                    # plt.plot(dark_spectra.T)

                    dark_std = np.std(darks)
                    dark_mean = np.mean(dark_offset)

                    if np.isnan(dark_std):
                        print("Error: std is nan")
                        continue
                    # print(h5, [(s, h5f["Channel"][s][0]) for s in h5f["Channel"].keys()])
                    if dark_std > 1000:
                        print("Skipping, dark_std=", dark_std, h5)
                        continue

                    # print(h5)
                    line = "%s\t%0.2f\t%i\t%0.4f\t%0.4f\n" % (h5, temp, unique_aotf, dark_mean, dark_std)
                    with open("nomad_so_dc_nobgsub2.txt", "a") as f:
                        f.write(line)


# %%

d = {"h5s": [], "ts": [], "aotfs": [], "dcs": [], "monthyears": []}
with open("nomad_so_dc_nobgsub2.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line_split = line.split()
        if float(line_split[2]) == 0.0:
            d["h5s"].append(line_split[0])
            d["ts"].append(float(line_split[1]))
            d["aotfs"].append(float(line_split[2]))
            d["dcs"].append(float(line_split[3]))
            d["monthyears"].append((float(line_split[0][0:4])*12 + float(line_split[0][4:6]))/12)

for key in ["ts", "aotfs", "dcs", "monthyears"]:
    d[key] = np.asarray(d[key])

plt.figure()
sc = plt.scatter(d["ts"], d["dcs"], c=d["monthyears"])
plt.colorbar(sc)
plt.grid()
plt.title("NOMAD SO background ADC counts when AOTF is off")
plt.ylabel("ADC counts")
plt.xlabel("Instrument temperature")

# counts = 56000
counts = d["dcs"]

n_accs = 12.0
binning = 3 + 1.0
it = 4.0

counts_per_pixel = counts / binning / n_accs

counts_0it = 1080  # LNO, approximately correct for SO
counts_sat = 13000  # LNO, approximately correct for SO

elec_0it = 0.0
elec_sat = 37.0e6

polyfit = np.polyfit(d["ts"], counts_per_pixel, 1)
tb_counts_exp_m15 = np.polyval(polyfit, -15.0)

elecs_per_pixel = elec_0it + (counts_per_pixel - counts_0it) * ((elec_sat - elec_0it) / (counts_sat - counts_0it))

plt.figure(figsize=(10, 6), constrained_layout=True)
sc = plt.scatter(d["ts"], elecs_per_pixel, c=d["monthyears"])
cbar = plt.colorbar(sc)
cbar.set_label("Observation Date", rotation=270, labelpad=20)
plt.grid()
plt.title("NOMAD-SO background electrons when AOTF is off")
plt.ylabel("Electrons")
plt.xlabel("Instrument temperature (C)")

polyfit = np.polyfit(d["ts"], elecs_per_pixel, 1)
polyvals = np.polyval(polyfit, d["ts"])
plt.plot(d["ts"], polyvals, label="Linear fit electrons vs SO temperature")

dc_elec_per_second = 6000.0  # electrons per second
dc_elec = dc_elec_per_second * n_accs * it/1000.0

plt.plot(d["ts"], polyvals + dc_elec * 3.6, label="Linear fit + dark current*3.6")
plt.plot(d["ts"], polyvals - dc_elec * 3.6, label="Linear fit - dark current*3.6")

plt.legend()
plt.savefig("NOMAD_SO_electrons_vs_temperature.png")

# readout_elec = 1000.0 # electrons
