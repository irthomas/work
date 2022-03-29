# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:43:33 2020

@author: iant

MAKE LNO MOLECULAR SPECTRA FOR A GIVEN ORDER/WAVENUMBER RANGE
"""
import numpy as np
import os

from analysis.retrievals.pytran.pytran.hitran_utils import get_molecule_id
from analysis.retrievals.pytran.pytran.hitran import read_hitran2012_parfile, calculate_hitran_xsec
from analysis.retrievals.NOMADTOOLS.nomadtools.paths import NOMADParams


def get_molecular_hr(molecule, nu_hr, Smin=0.0):
    
    
    #H2O absorption should be 0.65 in 169
    #CO absorption should be 0.9 in 189
    #CO2 absorption should be 0 in order 165
    nu_hr_min = np.min(nu_hr)
    nu_hr_max = np.max(nu_hr)
    number_density = 1.895293e+17 #NT at surface

    pressures = {"H2O":113.0*1.0e-6*550, "CO2":550.0, "CO":876.0*1.0e-6*550, "HCl":3.0*1.0e-9*550} #previous: 0.24*1.0e3 pressure in Pa
#    scaling_factor = {"H2O":1.5, "CO2":20.0, "CO":40.0, "HCl":10}
    M = get_molecule_id(molecule)
    filename = os.path.join(NOMADParams['HITRAN_DIR'], '%02i_hit16_2000-5000_CO2broadened.par' % M)
    if not os.path.exists(filename):
        filename = os.path.join(NOMADParams['HITRAN_DIR'], '%02i_hit16_2000-5000.par' % M)
    
    if Smin == 0.0:
        if molecule == "CO2":
            Smin=1.0e-26
        elif molecule == "H2O":
            Smin=1.0e-27
        else:
            Smin=1.0e-33
        
    nlines = 999
    while nlines>200:
        Smin *= 10.0
        LineList = read_hitran2012_parfile(filename, nu_hr_min, nu_hr_max, Smin=Smin, silent=True)
        nlines = len(LineList['S'])
        print('Found %i lines for Smin of %0.1g' %(nlines, Smin))
#    molecular_spectrum_hr = calculate_hitran_xsec(LineList, M, nu_hr, T=210., P=pressures[molecule]) * number_density * scaling_factor[molecule]
    molecular_spectrum_hr = calculate_hitran_xsec(LineList, M, nu_hr, T=210., P=pressures[molecule]) * number_density

    return molecular_spectrum_hr




