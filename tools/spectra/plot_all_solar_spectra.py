# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:51:28 2021

@author: iant

PLOT DIFFERENT SOLAR SPECTRA

"""
import os
import numpy as np
import matplotlib.pyplot as plt

from tools.file.paths import paths
from tools.spectra.solar_spectrum import get_solar_hr


# interpolate = False
interpolate = True

nu_hr = np.arange(2550., 4550.0, 0.2)
# nu_hr = np.arange(2550., 4550.0, 1.0)


solar_spectrum_files = [
    os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "Solar_irradiance_ACESOLSPEC_2015.dat"),
    os.path.join(paths["REFERENCE_DIRECTORY"], "nomad_solar_spectrum_solspec.txt"), #same as above
    os.path.join(paths["REFERENCE_DIRECTORY"], "pfsolspec_hr.dat"),
    os.path.join(paths["REFERENCE_DIRECTORY"], "psg_rad.txt"), #bad - no spectral lines
    os.path.join(paths["REFERENCE_DIRECTORY"], "ace-solar-spectrum.txt"),
]

#best solar lines from Goddard
lines = [
    2703.8,
    2733.3,
    2837.8,
    2927.1,
    2942.5,
    3172.9,
    3289.6,
    3414.5,
    3650.9,
    3750.1,
    3755.8,
    3787.9,
    4276.1,
    4383.5,
]


for solar_spectrum_file in solar_spectrum_files:
    
    basename = os.path.basename(solar_spectrum_file)
    
    if interpolate:
        I0_hr = get_solar_hr(nu_hr, solspec_filepath=solar_spectrum_file, interpolate=interpolate)
    else:
        nu_hr, I0_hr = get_solar_hr(nu_hr, solspec_filepath=solar_spectrum_file, interpolate=interpolate)
    
    plt.figure(figsize=(15,8))
    plt.plot(nu_hr, I0_hr)
    
    plt.title(basename)
    
    plt.axvline(x=2584.) #order 115 min
    plt.axvline(x=4530.) #order 200 max
    
    for line in lines:
        plt.axvline(x=line, c="r", linestyle=":", alpha=0.7)

    plt.savefig("%s_0.2cm-1_resolution.png" %basename)

