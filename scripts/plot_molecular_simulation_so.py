# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:41:42 2020

@author: iant
"""
import numpy as np
import matplotlib.pyplot as plt

from instrument.nomad_so_instrument_v03 import lt22_waven
from tools.spectra.molecular_spectrum_so import get_molecular_hr
from tools.spectra.solar_spectrum_so import nu_hr_grid
from tools.spectra.smooth_hr import smooth_hr

from tools.general.get_minima_maxima import get_local_minima

SMOOTHING_LEVEL = 350

#order = 129 
# order = 130
order = 134
molecule = "HCl"
molecule = "CO2"
molecule = "H2O"

instrument_temperature = 0.0

nu = lt22_waven(order, instrument_temperature)
nu_hr, _ = nu_hr_grid(order, 0, instrument_temperature)
#nu_hr = np.arange(nu[0], nu[-1], 0.001)

molecular_spectrum_hr = get_molecular_hr(molecule, nu_hr, Smin=1.0e-33)

molecular_spectrum = smooth_hr(molecular_spectrum_hr, window_len=(SMOOTHING_LEVEL-1))
normalised_molecular_spectrum = 1.0 - molecular_spectrum[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)]


indices = get_local_minima(normalised_molecular_spectrum)

for index in indices:
    relative_strength = (1.0-normalised_molecular_spectrum[index])/(1.0-np.min(normalised_molecular_spectrum[indices]))
    if relative_strength > 0.2:
        print("%0.3f, %0.3f" %(nu_hr[index], relative_strength))

plt.figure()
plt.plot(nu_hr, normalised_molecular_spectrum)
plt.title("%s Spectrum Order %i Simulation" %(molecule, order))
plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("Normalised Transmittance")
plt.savefig("%s_simulation_order_%i.png" %(molecule, order))

