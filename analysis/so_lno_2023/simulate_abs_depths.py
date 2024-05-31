# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:16:31 2024

@author: iant

FORWARD MODEL EXPECTED ORDER CONTRIBUTIONS FOR DIFFERENT ALTITUDES
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# compare absorption depths to forward model
from analysis.so_lno_2023.calibration import get_aotf, get_blaze_orders, get_calibration
from analysis.so_lno_2023.molecules import get_molecules
# from analysis.so_lno_2023.geometry import get_geometry
from analysis.so_lno_2023.forward_model import forward
from analysis.so_lno_2023.functions.geometry import make_path_lengths


centre_order = 186
aotf_freq = 25430.0
grat_t = -11.2611
aotf_t = -11.2611

mol_scaler = 1.0

lat = 64.90912439185713
lon = 50.08229628349869
ls = 110.13582716894238
lst = 1.2522222222222221
myear = 36

alts_all = [np.arange(i, 101, 5) for i in range(10, 81, 3)]

good_px_ixs = np.arange(50, 320)

channel = "so"
# chosen_alt = 40.0 #km
# chosen_alt = 60.0 #km
# chosen_alt = 140.0 #km

molecules = {
    "CO": {"isos": [1, 2, 3, 4]},
    # "CO":{"isos":[1]},
}


if centre_order == 186:
    orders = np.arange(183, 193)
if centre_order == 195:
    orders = np.arange(190, 199)


# get calibration info
# aotf = {"type":"file", "filename":"4500um_closest_aotf.txt"}
aotf = {"type": "sinc_gauss"}

"""initial parameters"""
aotf_d = get_aotf(channel, aotf_freq, aotf_t, aotf=aotf)

aotf_nu_centre = aotf_d["nu_centre"]
orders_d = get_blaze_orders(channel, orders, aotf_nu_centre, grat_t, px_ixs=good_px_ixs)

cal_d = get_calibration(channel, centre_order, aotf_d, orders_d)


geom_d = {}

transs = []


with PdfPages("co_order_186.pdf") as pdf:
    for alts in alts_all:

        geom_d["alt"] = alts
        geom_d["myear"] = myear
        geom_d["ls"] = ls
        geom_d["lst"] = lst
        geom_d["lat"] = lat
        geom_d["lon"] = lon

        geom_d["alt_grid"] = geom_d["alt"]
        path_lengths = make_path_lengths(geom_d["alt_grid"])
        geom_d["path_lengths_km"] = path_lengths

        molecule_d = get_molecules(molecules, geom_d)

        # from lmfit import minimize
        from lmfit import Parameters

        fw = forward(raw=False)
        fw.calibrate(cal_d)
        fw.geometry(geom_d)
        fw.molecules(molecule_d)

        params = Parameters()
        params.add('mol_scaler', value=mol_scaler)

        fig1, (ax1a, ax1b) = plt.subplots(figsize=(25, 10), ncols=2, constrained_layout=True)

        trans = fw.forward_so(params, plot=["hr", "cont"], axes=[ax1a, ax1b])

        # plt.figure()
        # plt.plot(trans1)
        fig1.suptitle("Altitude range %0.3f-%0.3f" % (alts[0], alts[-1]))
        ax1b.set_ylim((0.9, 1.0))
        pdf.savefig()
        plt.close()

        transs.append(trans)


transs = np.asarray(transs)

plt.figure()
plt.plot(transs.T)

z = [l[0] for l in alts_all]
plt.figure()
for i in range(0, 270, 20):
    plt.scatter(z, transs[:, i])
    plt.plot(z, transs[:, i], label="Px %i" % i)
plt.legend()
plt.grid()
