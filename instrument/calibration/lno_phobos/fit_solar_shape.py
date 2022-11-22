# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:59:52 2022

@author: iant

FIT SOLAR SPECTRUM SHAPE TO NOISY PHOBOS SPECTRA
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt



from instrument.nomad_lno_instrument_v01 import m_aotf
from instrument.calibration.lno_phobos.solar_inflight_cal import rad_cal_order

from tools.file.hdf5_functions import open_hdf5_file
from tools.general.normalise_values_to_range import normalise_values_to_range
from tools.datasets.get_phobos_crism_data import get_phobos_crism_data


bad_pixel_d = {
    (148, 1):[288, 294, ],
    (149, 1):[292, 310, 317, ],
    (150, 1):[301, ],
    (151, 1):[258, ],
    (152, 1):[],
    (153, 1):[5, 40, 245, ],
    (154, 1):[129, 135, 178, ],
    (155, 1):[125, 232, 308, ],
    
    
    (146, 3):[188, 228, 121, 236, 294, ],
    (149, 3):[292, 310, 317, 301, 258, ],
    (152, 3):[5, 245, 129, 135, 178, ],
    (155, 3):[125, 232, 308, 78, 254, 294, 291, ],

}



colour_order_dict = {
    142:{"colour":"C0"},
    148:{"colour":"C1"},
    153:{"colour":"C2"},
    154:{"colour":"C2"},
    158:{"colour":"C3"},
    160:{"colour":"C3"},
    164:{"colour":"C4"},
    166:{"colour":"C4"},
    170:{"colour":"C5"},
    172:{"colour":"C5"},
    177:{"colour":"C6"},
    178:{"colour":"C6"},
    184:{"colour":"C7"},

    174:{"colour":"C1"},
    175:{"colour":"C2"},
    176:{"colour":"C3"},
    189:{"colour":"C4"},
    190:{"colour":"C5"},
    191:{"colour":"C6"},

    192:{"colour":"C1"},
    193:{"colour":"C2"},
    201:{"colour":"C3"},
}




h5 = "20220710_200313_0p1a_LNO_1"
good_indices = [*range(20, 149)]

h5_f = open_hdf5_file(h5, path=r"E:\DATA\hdf5_phobos") #no detector offset!)

y = h5_f["Science/Y"][...]
unique_bins = sorted(h5_f["Science/Bins"][0, :, 0])
binning = unique_bins[1] - unique_bins[0]

aotf_f = h5_f["Channel/AOTFFrequency"][...]
unique_freqs = sorted(list(set(aotf_f)))

orders = np.array([m_aotf(i) for i in aotf_f])
unique_orders = sorted(list(set(orders)))


#correct bad pixels
for i, unique_bin in enumerate(unique_bins):
    bad_pixels = bad_pixel_d[(unique_bin, binning)]
    y[:, i, bad_pixels] = np.nan




# cal_h5 = "20201222_114725_1p0a_LNO_1_CF"
# # cal_d = {order:rad_cal_order(cal_h5, order, centre_indices=None) for order in unique_orders}
# cal_d = {order:rad_cal_order(cal_h5, order, centre_indices=range(100, 300)) for order in unique_orders}
# solar_scalars = {order:cal_d[order]["y_centre_mean"] / 2.0e6 for order in cal_d.keys()}




#plot solar calibration spectra
# plt.figure()
# plt.title("Solar calibration spectra")
# for order in cal_d.keys():
#     plt.plot(cal_d[order]["y_spectrum"], color = colour_order_dict[order]["colour"], label=order)
#     plt.plot([0, len(cal_d[order]["y_spectrum"])-1], [cal_d[order]["y_centre_mean"], cal_d[order]["y_centre_mean"]], \
#               color = colour_order_dict[order]["colour"], label=order)
# plt.legend()
    


from scipy.optimize import minimize

bin_ = 1

y_sun = cal_d[order]["y_spectrum"]
y_spectrum = y[good_indices[10], bin_, :]

scalar0 = np.nanmean(y_sun) / np.nanmean(y_spectrum)
offset0 = np.nanmean(y_spectrum)


def minimise(params, args=()):
    y_spectrum = args
    scalar = params[0]
    offset = params[1]
    print(y_spectrum[0], scalar, y_sun[0])
    out = np.nanmean(np.abs((y_spectrum + offset) - y_sun * scalar))
    return out


a = minimize(minimise, [scalar0, offset0], args=(y_spectrum))

scalar = a.x[0]
offset = a.x[1]

plt.plot(y_spectrum + offset)
plt.plot(y_sun / scalar)

# for bin_ in range(y.shape[1]):
#     plt.figure(figsize=(8, 4))
#     plt.title("Raw spectra for bin %i" %bin_)
#     plt.xlabel("Pixel number")
#     plt.ylabel("Signal (counts)")
#     for i, y_spectrum in enumerate(y[good_indices, bin_, :]):
#         colour = colour_order_dict[orders[i]]["colour"]
#         plt.plot(y_spectrum, alpha=0.3, color=colour)

