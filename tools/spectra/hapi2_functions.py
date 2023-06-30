# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:40:03 2023

@author: iant

HAPI2 FUNCTIONS

MUST DOWNGRADE SQLALCHEMY TO AVOID ERRORS
pip install --force-reinstall sqlalchemy==1.4.46


"""

import numpy as np
from matplotlib import pyplot as plt



import sys
import io

sys.path.append(r"C:\Users\iant\Dropbox\NOMAD\Python")
from tools.datasets.get_gem_data import get_gem_data


import hapi2




# SETTINGS['api_key'] = getpass('Enter valid API key:')
# hapi2.fetch_info()

# mols = hapi2.fetch_molecules() # fetch molecules


# hapi2.Molecule('Water').dump()

# mol = hapi2.Molecule('Water')
# xss2 = hapi2.fetch_cross_sections(mol)

# print(mol.common_name, mol.ordinary_formula)


# hapi2.Molecule('Water').aliases

# isos = hapi2.fetch_isotopologues([hapi2.Molecule('Water'),])


# # from hapi2 import fetch_cross_section_spectra
    
# xss = hapi2.Molecule('water').cross_sections
# hapi2.fetch_cross_section_spectra(xss)


# isos = hapi2.Molecule('water').isotopologues
# hapi2.fetch_transitions(isos,2000,2500,'water_normal')

# stop()

def hapi_fetch(molecule, nu_start, nu_end):
    """fetch main isotope for each molecule"""
    
    print(molecule)
    table_name = molecule

    if molecule in ["H2O"]:
        component = {"H2O":1, "CO2":2, "O3":3, "CO":5, "CH4":6, "HCL":15}[molecule]
        hapi2.hapi.fetch(table_name, component, 1, nu_start, nu_end)
    else:  
        iso_codes = {"H2O":[1, 2, 3, 4, 5, 6], "CO2":[7, 8, 9, 10, 11, 12], "O3":[], "CO":[26, 36, 28, 27, 38, 37], "CH4":[], "HCL":[]}[molecule]
        hapi2.hapi.fetch_by_ids(table_name, iso_codes, nu_start, nu_end, ParameterGroups=['Voigt_CO2'])
    



def hapi_gem_trans(molecule, altitude, nu_range, nu_step, path_length_km=1e3, spec_res=0.2, l=10., myear=34, ls=180., lat=0., lon=0., lst=12., silent=False, clear=False):
    """make convolved transmittance spectrum for a given spectral range, altitude (km) and path length l (km), by fetching HAPI and GEM Mars data"""
    
    t, pressure, mol_ppmv, co2_ppmv = get_gem_tpvmr(molecule, altitude, myear, ls, lat, lon, lst)
    
    if not silent:
        print("Temperature = %0.1fK; pressure = %0.2e atm; %s ppmv = %0.1f; CO2 ppmv = %0.1f" %(t, pressure, molecule, mol_ppmv, co2_ppmv))
    
    
    nu, coef = get_abs_coeff(molecule, nu_range, nu_step, mol_ppmv, co2_ppmv, t, pressure, clear=clear)

    path_length_cm = 100.0 * l * path_length_km #km in cm
    
    nu, trans = hapi2.hapi.transmittanceSpectrum(nu, coef, Environment={'T':t, 'l':path_length_cm})
    nu_conv, trans_conv, i1, i2, slit = hapi2.hapi.convolveSpectrum(nu, trans, SlitFunction=hapi2.hapi.SLIT_GAUSSIAN, Resolution=spec_res, AF_wing=0.3)
    
    # plt.figure()
    # plt.plot(nu, trans)
    # plt.plot(nu_conv, trans_conv)
    
    return nu_conv, trans_conv
    
    

def get_hapi_nu_range(molecule):
    if molecule not in hapi2.hapi.tableList():
        return [0., 0.]

    old_stdout = sys.stdout # Memorize the default stdout stream
    sys.stdout = buffer = io.StringIO()
    
    
    hapi2.hapi.select(molecule)
    
    sys.stdout = old_stdout
    
    s = buffer.getvalue().split("\n")
    s[:] = [x for x in s if x]
    nu_start = np.float32(s[1].split()[1])
    nu_end = np.float32(s[-1].split()[1])
    
    return nu_start, nu_end



def fetch_hapi_data(molecule, nu_start, nu_end, clear=False):

    if molecule not in hapi2.hapi.tableList() or clear:
        hapi_fetch(molecule, nu_start, nu_end)
    if "CO2" not in hapi2.hapi.tableList() or clear:
        hapi_fetch("CO2", nu_start, nu_end)



def get_gem_tpvmr(molecule, altitude, myear, ls, lat, lon, lst):
    
    gem_d = get_gem_data(myear, ls, lat, lon, lst)
    
    t = np.interp(altitude, gem_d["z"][::-1], gem_d["t"][::-1])
    pressure = np.interp(altitude, gem_d["z"][::-1], gem_d["p"][::-1]) / 101300. #pa to atmosphere
    mol_ppmv = np.interp(altitude, gem_d["z"][::-1], gem_d[molecule.lower()][::-1]) #ppmv
    co2_ppmv = np.interp(altitude, gem_d["z"][::-1], gem_d["co2"][::-1]) #ppmv

    return t, pressure, mol_ppmv, co2_ppmv



def get_abs_coeff(molecule, nu_range, nu_step, mol_ppmv, co2_ppmv, t, pressure, clear=False):

    fetch_hapi_data(molecule, nu_range[0], nu_range[-1], clear=clear)
    
    
    component = {"H2O":1, "CO2":2, "O3":3, "CO":5, "CH4":6, "HCL":15}[molecule]
    scalar = 1.
    
    #TODO: extend calculation to other isotopologues

    if molecule == "CO2":
        Components=[
            (component, 1, mol_ppmv/co2_ppmv*hapi2.hapi.abundance(component, 1)),
            (component, 2, mol_ppmv/co2_ppmv*hapi2.hapi.abundance(component, 2)),
            (component, 3, mol_ppmv/co2_ppmv*hapi2.hapi.abundance(component, 3)),
        ]

    else:
        Components=[
            (component, 1, mol_ppmv/co2_ppmv*hapi2.hapi.abundance(component, 1)*scalar),
            (component, 2, mol_ppmv/co2_ppmv*hapi2.hapi.abundance(component, 2)*scalar),
            (component, 3, mol_ppmv/co2_ppmv*hapi2.hapi.abundance(component, 3)*scalar),
            (component, 4, mol_ppmv/co2_ppmv*hapi2.hapi.abundance(component, 4)*scalar),
            (component, 5, mol_ppmv/co2_ppmv*hapi2.hapi.abundance(component, 5)*scalar),
            (2, 1, (1-mol_ppmv/co2_ppmv))
        ]
        
    nu, coef = hapi2.hapi.absorptionCoefficient_Voigt(SourceTables=molecule, Components=Components, Diluent={'CO2':1.0}, \
                                                Environment={'T':t,'p':pressure}, WavenumberStep=nu_step, HITRAN_units=False)

    return nu, coef



"""examples"""
if __name__ == "__main__":

    # hapi_fetch("H2O", 3745, 3835)
    # hapi_fetch("CO2", 3745, 3835)
    # hapi_fetch("CO", 4000, 4500)
    
    # molecule = "H2O"
    # altitude = 50.
    # nu_range = [3745, 3835]
    # nu_range = [3057.02, 3081.44] #order 136
    # nu_step = 0.001
    
    molecule = "CO"
    altitude = 50.
    nu_range = [4200, 4300]
    nu_step = 0.001
    
    # molecule = "CO2"
    # altitude = 50.
    # nu_range = [3349.24, 3375.99]
    # nu_step = 0.001
    
    # get gem data for given altitude and generic lat/lon/my/lst then 
    occ_sim_nu, occ_sim = hapi_gem_trans(molecule, altitude, nu_range, nu_step, clear=True)
    
    plt.plot(occ_sim_nu, occ_sim)