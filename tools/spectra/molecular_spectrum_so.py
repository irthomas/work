# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:43:33 2020

@author: iant

MAKE LNO MOLECULAR SPECTRA FOR A GIVEN ORDER/WAVENUMBER RANGE
"""
import numpy as np
import os
import matplotlib.pyplot as plt


from repos.pytran.pytran.hitran_utils import get_molecule_id
from repos.pytran.pytran.hitran import read_hitran2012_parfile, calculate_hitran_xsec
from repos.nomad_tools.nomadtools.rcsetup import defaultParams



def get_xsection(molecule, nu_hr, Smin=0.0, temperature=210.0, pressure=0.0):

    nu_hr_min = np.min(nu_hr)
    nu_hr_max = np.max(nu_hr)


    M = get_molecule_id(molecule)
    filename = os.path.join(defaultParams['paths.dirLP'], '%02i_hit16_2000-5000_CO2broadened.par' % M)
    if not os.path.exists(filename):
        filename = os.path.join(defaultParams['paths.dirLP'], '%02i_hit16_2000-5000.par' % M)
    
    if Smin == 0.0:
        if molecule == "CO2":
            Smin=1.0e-26
        elif molecule == "H2O":
            Smin=1.0e-27
        else:
            Smin=1.0e-33
        
    nlines = 999
    Smin /= 10.0
    while nlines>200:
        Smin *= 10.0
        LineList = read_hitran2012_parfile(filename, nu_hr_min, nu_hr_max, Smin=Smin)
        nlines = len(LineList['S'])
        print('Found %i lines for %s Smin of %0.1g' %(nlines, molecule, Smin))
    return calculate_hitran_xsec(LineList, M, nu_hr, T=temperature, P=pressure)


# #test xsec * path length to transmittance conversion
# nu_hr = np.arange(4100., 4200., 0.00042)
# molecule = "CO"
# Smin = 1.0e-28
# temperature = 210.
# pressure = 100. #Pa = 0.000986923 atm
# xsec = get_xsection(molecule, nu_hr, Smin=Smin, temperature=temperature, pressure=pressure) #cm2/molecule
# # xsec /= 1.0e4 #m2/molecule
# nd = 0.02504e27 * 0.000986923 #number density m-3 at 100 Pa

# for pathl in np.arange(6.0):
#     trans = 10**(-1.49791808045940E+18 * xsec * pathl + 3.38107922285908E-06)
#     plt.plot(nu_hr, trans, label=pathl)
#     print(np.min(trans))
# plt.legend()



def get_molecular_hr(molecule, nu_hr, Smin=0.0, temperature=210.0, pressure=100.0, path_length=100e3): #K, Pa, m
    
    pressure_atm = pressure * 0.00000986923 #atm
    

    pressures = {"H2O":113.0*1.0e-6*550, "CO2":550.0, "CO":876.0*1.0e-6*550, "HCl":0.003*1.0e-6*550} #previous: 0.24*1.0e3 pressure in Pa
    
    if pressure == 0.0:
        pressure = pressures[molecule]
    else:
        pressure *= 1.0e-6*550
    xsec = get_xsection(molecule, nu_hr, Smin=0.0, temperature=temperature, pressure=pressure_atm)
    transmittance = 10**(-1.49791808045940E+18 * xsec * path_length + 3.38107922285908E-06)

    return transmittance



