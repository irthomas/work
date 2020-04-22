# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:41:42 2020

@author: iant
"""
import numpy as np
import matplotlib.pyplot as plt

from instrument.nomad_so_instrument import nu_mp
from tools.spectra.molecular_spectrum_so import get_molecular_hr
from tools.spectra.solar_spectrum_so import nu_hr_grid
from tools.spectra.smooth_hr import smooth_hr

SMOOTHING_LEVEL = 350

#order = 129 
order = 130 
molecule = "HCl"
molecule = "CO2"
molecule = "H2O"

instrument_temperature = 0.0

nu = nu_mp(order, np.arange(320), instrument_temperature)
nu_hr, _ = nu_hr_grid(order, 0, instrument_temperature)
#nu_hr = np.arange(nu[0], nu[-1], 0.001)

molecular_spectrum_hr = get_molecular_hr(molecule, nu_hr, Smin=1.0e-33)

molecular_spectrum = smooth_hr(molecular_spectrum_hr, window_len=(SMOOTHING_LEVEL-1))
normalised_molecular_spectrum = 1.0 - molecular_spectrum[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)]

plt.figure()
plt.plot(nu_hr, normalised_molecular_spectrum)
plt.title("%s Spectrum Order %i Simulation" %(molecule, order))
plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("Normalised Transmittance")
plt.savefig("%s_simulation_order_%i.png" %(molecule, order))