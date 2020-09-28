# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:55:08 2019

@author: iant
"""


# import os
#import sys

import numpy as np
# from scipy import interpolate
from matplotlib import pyplot as plt

# from tools.file.paths import paths
# from analysis.retrievals.pytran import pytran
#from analysis.retrievals.NOMADTOOLS import nomadtools
#from analysis.retrievals.NOMADTOOLS.nomadtools import gem_tools
#from analysis.retrievals.NOMADTOOLS.nomadtools.paths import NOMADParams
# from analysis.retrievals.NOMAD_instrument import freq_mp, F_blaze, F_aotf_3sinc
# from tools.spectra.solar_spectrum_so import get_solar_hr

# from instrument.nomad_so_instrument import nu_mp, F_blaze, F_aotf_goddard18b
from analysis.py_retrievals.so_simple_retrieval import simple_retrieval, forward_model

sim_spectra = {"HCl":{"scaler":4.0, "label":"HCl 4ppbv", "colour":"r--"},
               "H2O":{"scaler":1.0, "label":"H2O a priori", "colour":"m--"},
               "CO2":{"scaler":1.0, "label":"CO2 a priori", "colour":"c--"},
               "PH3":{"scaler":100.0, "label":"PH3 100ppbv", "colour":"b--"},
               }

y = np.ones((320)) + (np.random.rand(320)/1000) - (2/1000)
alt = 10.0
diffraction_order = 152
instrument_temperature = -5.0


fig, ax = plt.subplots(figsize=(9, 6))


# for molecule in ["H2O"]:
for molecule in ["PH3", "CO2", "H2O"]:
    retDict = simple_retrieval(y, alt, molecule, diffraction_order, instrument_temperature)
    retDict = forward_model(retDict, xa_fact=[sim_spectra[molecule]["scaler"]])
    sim_spectra[molecule]["spectrum"] = retDict["Y"][0, :]
    sim_spectra[molecule]["xa"] = retDict["xa"][0]
    sim_spectra[molecule]["xa_fact"] = retDict["xa_fact"][0]
    ax.plot(retDict["nu_p"], sim_spectra[molecule]["spectrum"], sim_spectra[molecule]["colour"], label=sim_spectra[molecule]["label"])
    print(molecule, sim_spectra[molecule]["xa"] * sim_spectra[molecule]["xa_fact"] * 1.0e6, "ppmv")
ax.legend()
plt.savefig("SO_order_%i_simulation_10km")


