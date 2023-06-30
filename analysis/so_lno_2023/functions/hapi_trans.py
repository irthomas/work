# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:22:40 2022

@author: iant

TEST HAPI
"""

import numpy as np
from matplotlib import pyplot as plt



from tools.spectra.hapi_functions import hapi_transmittance, get_hapi_nu_range, hapi_fetch
from instrument.nomad_so_instrument_v03 import lt22_waven


order_dict = {
    186:{"molecule":"CO"},
}

order = 186
t = -5.
alt = 50.

nu_px = lt22_waven(order, t)

molecule = order_dict[order]["molecule"]


hapi_nu_range = get_hapi_nu_range(molecule)
if nu_px[0]-1. < hapi_nu_range[0] or nu_px[-1]+1. > hapi_nu_range[1]:
    print("Refetching HAPI data")
    clear=True
else:
    clear=False
    
occ_sim_nu, occ_sim = hapi_transmittance(molecule, alt, [nu_px[0]-1., nu_px[-1]+1.], 0.001, clear=clear)
plt.plot(occ_sim_nu, occ_sim, "k")
