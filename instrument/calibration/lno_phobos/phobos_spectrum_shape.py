# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 17:55:10 2025

@author: iant

READ PHOBOS SHAPE JSON
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import re

from scipy.signal import savgol_filter

from instrument.nomad_lno_instrument_v02 import nu_mp

# SAVE_JSON = True
SAVE_JSON = False

SG_LENGTH = 99

noisy_h5s = [
    "20250628_083013",
    "20241213_052910",
    "20241210_084211",
    "20240427_112108",
    "20240412_193543",
]

# open previously saved spectra
with open("lno_phobos_shape_output_all0.json", "r") as f:
    mean_shapes_d = json.load(f)

# regexes = [re.compile("202506.._......")]
# orders = [160, 162, 164, 166, 168, 170]

# regexes = [re.compile("202504.._......")]
# orders = [157, 160, 163, 166, 169, 172]

# regexes = [re.compile("202412.._......")] # some noisy
# orders = [160, 162, 164, 166, 168, 170]

# regexes = [re.compile("2024(10|09).._......")] # ok
# orders = [174, 175, 176, 190, 191, 192]

# regexes = [re.compile("2024(08|07|06).._......")] # all good
# orders = [163, 165, 167, 169]

# regexes = [re.compile("2024(06|05|04).._......")] # all good
# orders = [189, 190, 191, 192, 193, 201]


regexes = [
    re.compile("202506.._......"),
    re.compile("202504.._......"),
    re.compile("202412.._......"),
    re.compile("2024(10|09).._......"),
    re.compile("2024(08|07|06).._......"),
    re.compile("2024(06|05|04).._......"),
]
orders = [157, 160, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 174, 175, 176, 189, 190, 191, 192, 193, 201]
# orders = [160, 162, 164, 166, 168, 170]
# orders = [160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
# orders = [160, 162, 168, 169]
# orders = [163, 169]
# orders = [168]
# orders = [169]
# orders = [172]
# orders = [168, 169]

# regexes = [
#     re.compile("202(4|5)...._......"),
# ]
# orders = [157, 160, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 174, 175, 176, 189, 190, 191, 192, 193, 201]


# bins = [1]
# bins = [2]
bins = [0, 1, 2, 3, 4, 5]


# convert json to numpy arrays
for h5 in mean_shapes_d.keys():
    for order in mean_shapes_d[h5].keys():
        for bin_ in mean_shapes_d[h5][order].keys():
            mean_shapes_d[h5][order] = {int(k): np.asarray(v) for k, v in mean_shapes_d[h5][order].items()}

# all_orders = []
# for h5 in mean_shapes_d.keys():
#     for key in mean_shapes_d[h5].keys():
#         if int(key) not in all_orders:
#             all_orders.append(int(key))
# orders = sorted(all_orders)


mean_shapes_d2 = {}

plt.figure(figsize=(12, 6))
plt.title("Phobos spectral shapes of different orders")
for regex in regexes:

    for order_ix, order in enumerate(orders):

        waven = nu_mp(order, np.arange(320), -10)

        spectra = []
        h5s = []
        for h5 in mean_shapes_d.keys():
            if regex.search(h5):
                if str(order) in mean_shapes_d[h5].keys():
                    for bin_ in mean_shapes_d[h5][str(order)].keys():
                        if bin_ in bins:
                            if h5 not in noisy_h5s:
                                # print(h5, order, bin_)
                                spectra.append(mean_shapes_d[h5][str(order)][bin_] / np.mean(mean_shapes_d[h5][str(order)][bin_]))
                            h5s.append(h5)
                            # else:
                            #     # replace by adjacent h5 prefix
                            #     h5s.append(noisy_h5s[h5])

        if len(spectra) > 0:
            spectra = np.asarray(spectra)
            mean_spectrum = np.mean(spectra, axis=0)

            # smooth = savgol_filter(mean_spectrum, SG_LENGTH, 3)
            smooth = np.convolve(mean_spectrum, np.ones(59)/59, mode="same")
            smooth = np.convolve(smooth, np.ones(5)/5, mode="same")

            # plt.plot(spectra.T, color="C%i" % order_ix, alpha=0.3)
            # plt.plot(mean_spectrum, color="C%i" % order_ix, label="Order %i" % order)
            plt.plot(10000/waven, spectra.T, color="C%i" % order_ix, alpha=0.05)
            # plt.plot(waven, mean_spectrum, color="C%i" % order_ix, label="Order %i" % order)
            plt.plot(10000/waven, smooth, color="C%i" % order_ix, label="Order %i" % order)
            plt.text(10000/waven[160], smooth[160], "%s %i" % (regex.pattern, order))
            print(regex.pattern, order, np.where(smooth == np.max(smooth))[0])

            for h5 in h5s:
                if h5 not in mean_shapes_d2.keys():
                    mean_shapes_d2[h5] = {}

                mean_shapes_d2[h5][order] = [float(f) for f in smooth]

plt.xlabel("Wavelength")
plt.ylabel("Phobos mean spectra")
plt.grid()
plt.legend()

if SAVE_JSON:
    # TODO: try saving mean shape for each observation
    with open("lno_phobos_mean_shapes.json", "w", encoding="utf-8") as f:
        json.dump(mean_shapes_d2, f, ensure_ascii=False, indent=4)
