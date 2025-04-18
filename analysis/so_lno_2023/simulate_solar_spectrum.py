# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:33:09 2025

@author: iant

SIMULATE LNO RAW SOLAR SPECTRA TO CHECK AOTF SPECTRAL POSITION AND WIDTH
"""


import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from lmfit import Parameters


# compare absorption depths to forward model
from analysis.so_lno_2023.calibration import get_aotf, get_blaze_orders, get_calibration
from analysis.so_lno_2023.forward_model import forward_solar


centre_order = 167
aotf_freq = 24026.0
grat_t = -11.2611
aotf_t = -11.2611

good_px_ixs = np.arange(50, 320)

channel = "lno"

plot = ["aotf_blaze"]
# plot = []

if centre_order == 167:
    orders = np.arange(166, 169)


aotf_nu_offsets = np.arange(-3.0, 4.0, 1.0)

spectra = []
for aotf_nu_offset in aotf_nu_offsets:

    # get calibration info
    aotf = {"type": "sinc_gauss", "nu_offset": aotf_nu_offset}

    """initial parameters"""
    aotf_d = get_aotf(channel, aotf_freq, aotf_t, aotf=aotf)

    aotf_nu_centre = aotf_d["nu_centre"]

    orders_d = get_blaze_orders(channel, orders, aotf_nu_centre, grat_t, px_ixs=good_px_ixs)

    cal_d = get_calibration(channel, centre_order, aotf_d, orders_d, plot=plot)

    fw = forward_solar()
    fw.calibrate(cal_d)
    spectrum = fw.forward([], plot=[])

    spectra.append(spectrum)

plt.figure()
plt.grid()
plt.title("Effect of AOTF peak shift on raw LNO order %i solar spectrum" % centre_order)
plt.xlabel("Wavenumber")
plt.ylabel("Normalised response")
x = orders_d[centre_order]["px_nus"]
for aotf_nu_offset, spectrum in zip(aotf_nu_offsets, spectra):
    plt.plot(x, spectrum, label="AOTF shifted %0.1f cm-1" % aotf_nu_offset)
plt.legend()

# params = Parameters()
# params.add('mol_scaler', value=mol_scaler)

# fig1, (ax1a, ax1b) = plt.subplots(figsize=(25, 10), ncols=2, constrained_layout=True)

# trans = fw.forward_so(params, plot=["hr", "cont"], axes=[ax1a, ax1b])
