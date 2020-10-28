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

# from instrument.nomad_lno_instrument import nu_mp, F_blaze, F_aotf_goddard18b
from analysis.py_retrievals.lno_reff_simple_retrieval import simple_retrieval, forward_model


y = np.ones((320)) + (np.random.rand(320)/1000) - (2/1000)
diffraction_order = 168
instrument_temperature = -5.0

fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(3,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax4 = fig.add_subplot(gs[0:2, 1])

retDict = simple_retrieval(y, diffraction_order, instrument_temperature)

dust_profiles = {
    "flat":
        {"dust":np.ones(len(retDict["nu_hr"])), "colour":"b"},
    "sloped 0.9":
        {"dust":np.linspace(1.0, 0.9, num=len(retDict["nu_hr"])), "colour":"orange"}
    }
                 

for dust_profile in dust_profiles.keys():
    retDict["Trans_hr"] = dust_profiles[dust_profile]["dust"] #dust
    colour = dust_profiles[dust_profile]["colour"]

    retDict = forward_model(retDict)

    ax1.set_title("Solar spectrum")
    ax1.plot(retDict["nu_hr"], retDict["I0_hr"], color=colour, label=dust_profile)
    
    ax2.set_title("Approx LNO AOTF function")
    ax2.plot(retDict["nu_hr"], retDict["W_aotf"], color=colour, label=dust_profile)
    
    ax3.set_title("Simulated dust profile")
    ax3.plot(retDict["nu_hr"], retDict["Trans_hr"], color=colour, label=dust_profile)
    ax3.set_xlabel("Wavenumbers cm-1")
    
    # ax4.plot(retDict["nu_p"], retDict["I0_p"])
    
    ax4.set_title("Simulated normalised reflectance inc AOTF and blaze")
    ax4.plot(retDict["nu_p"], retDict["Trans_p"]/np.max(retDict["Trans_p"]), color=colour, label=dust_profile)
    ax4.set_xlabel("Pixel wavenumbers cm-1")

ax3.legend()
ax4.legend()

plt.tight_layout()