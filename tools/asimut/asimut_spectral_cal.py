# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:16:48 2022

@author: iant

FIND PX-WVN FOR ASIMUT INPUT FROM ABSORPTION LINES
"""
import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import open_hdf5_file
from tools.spectra.baseline_als import baseline_als
from tools.spectra.molecular_spectrum_so import get_molecular_hr
from tools.general.get_minima_maxima import get_local_minima


h5 = "20220101_005247_1p0a_SO_A_E_136" #high water
molecules = {
    "h2o":{"smin":1.0e-28, "n_lines":9, "orders":[135, 136, 137]}, 
    "co2":{"smin":1.0e-29, "n_lines":9, "orders":[136]}
}

def lt22_waven(order, inter_temp, px_in):
    """spectral calibration Loic Feb 22"""

    cfpixel = [3.32e-8, 5.480e-4, 22.4701]

    p0_nu = -0.8276 * inter_temp #px/°C * T(interpolated)
    px_temp = px_in + p0_nu
    xdat  = np.polyval(cfpixel, px_temp) * order
    
    return xdat





h5_f = open_hdf5_file(h5) #open file


#find indices of spectra where 0.1 < median transmittance < 0.95
y_median = np.median(h5_f["Science/Y"][...], axis=1)
# indices = list(np.where((y_median > 0.1) & (y_median < 0.95))[0])
indices = list(np.where((y_median > 0.3) & (y_median < 0.7))[0])

y_mean = np.mean(h5_f["Science/Y"][indices, :], axis=0)

y_cont = baseline_als(y_mean)
y_cr = y_mean / y_cont


order = int(h5.split("_")[-1])
t = np.mean(h5_f["Channel/InterpolatedTemperature"][indices])

x = lt22_waven(order, t, np.arange(320.))

fig, axes = plt.subplots(nrows=3, figsize=(18, 10), sharex=True)
axes[0].plot(x, y_cr)



if "InterpolatedTemperature" not in h5_f["Channel"].keys():
    print("%s: Error temperatures not in file" %h5)



for i, molecule in enumerate(molecules):
    smin = molecules[molecule]["smin"]
    n_lines = molecules[molecule]["n_lines"]
    
    orders = molecules[molecule]["orders"]

    for order in orders:
        nu_hr = lt22_waven(order, t, np.arange(320.))
        mol_hr = get_molecular_hr(molecule.upper(), nu_hr, Smin=smin)

        axes[i+1].plot(x, mol_hr, label=f"{molecule} {order}")

        abs_ix_hr = get_local_minima(mol_hr) #get hitran absorption minima indices
        abs_nu_hrs = nu_hr[abs_ix_hr] #get absorption nu
        abs_y_hrs = mol_hr[abs_ix_hr] #get absorption depth


        #N strongest lines
        abs_y_cutoff = sorted(abs_y_hrs)[n_lines] #select only the n strongest lines
        
        for abs_index, (abs_nu_hr, abs_y_hr) in enumerate(zip(abs_nu_hrs, abs_y_hrs)):
            if abs_y_hr < abs_y_cutoff:
                axes[i+1].text(abs_nu_hr, abs_y_hr, abs_nu_hr)

    axes[i+1].set_yscale("log")
    axes[i+1].grid()
    axes[i+1].legend()
axes[0].grid()
