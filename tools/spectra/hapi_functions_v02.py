# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:16:59 2023

@author: iant
"""


import numpy as np
from matplotlib import pyplot as plt

from tools.datasets.get_gem_data import get_gem_data

import hapi

import sys
import io


PARAMETER_GROUPS = ['Voigt_CO2']
ISO_IDS = {"H2O":[1, 2, 3, 4, 5, 6, 129], "CO2":[7, 8, 9, 10, 11, 12, 13, 14, 121, 15, 120, 122], "O3":[], "CO":[26, 27, 28, 29, 30, 31], "CH4":[32, 33, 34, 35], "HCL":[52, 53, 107, 108]}


def make_table_name(molecule, pressure):
    table_name = "%s_%0.3e" %(molecule, pressure)
    return table_name    


def hapi_fetch(molecule, pressure, nu_start, nu_end):
    """fetch main isotope for each molecule"""
    table_name = make_table_name(molecule, pressure)
    
    
    # if molecule in ["H2O"]:
    #     component = {"H2O":1, "CO2":2, "O3":3, "CO":5, "CH4":6, "HCL":15}[molecule]
    #     hapi.fetch(table_name, component, 1, nu_start, nu_end)
    # else:  
    iso_codes = ISO_IDS[molecule]
    hapi.fetch_by_ids(table_name, iso_codes, nu_start, nu_end, ParameterGroups=PARAMETER_GROUPS)
    


def hapi_transmittance(nu, coef, path_length_km, t, spec_res=None):

    path_length_cm = 100.0 * 1.0e3 * path_length_km #km in cm
    
    nu, trans = hapi.transmittanceSpectrum(nu, coef, Environment={'T':t, 'l':path_length_cm})
    
    if not spec_res:
        return nu, trans

    else:
        nu_conv, trans_conv, i1, i2, slit = hapi.convolveSpectrum(nu, trans, SlitFunction=hapi.SLIT_GAUSSIAN, Resolution=spec_res, AF_wing=0.3)
        return nu_conv, trans_conv
    


def hapi_gem_trans(molecule, altitude, nu_range, nu_step, isos=[1,2,3,4,5], path_length_km=10., spec_res=0.2, myear=34, ls=180., lat=0., lon=0., lst=12., silent=False, clear=False):
    """make convolved transmittance spectrum for a given spectral range, altitude (km) and path length l (km), by fetching HAPI and GEM Mars data"""
    
    t, pressure, mol_ppmv, co2_ppmv = get_gem_tpvmr(molecule, altitude, myear, ls, lat, lon, lst)
    
    if not silent:
        print("Temperature = %0.1fK; pressure = %0.2e atm; %s ppmv = %0.1f; CO2 ppmv = %0.1f" %(t, pressure, molecule, mol_ppmv, co2_ppmv))
    
    
    nu, coef = get_abs_coeff(molecule, nu_range, nu_step, mol_ppmv, co2_ppmv, t, pressure, isos=isos, clear=clear)
    nu, trans = hapi_transmittance(nu, coef, path_length_km, t, spec_res=spec_res)
    # plt.figure()
    # plt.plot(nu, trans)
    # plt.plot(nu_conv, trans_conv)
    
    return nu, trans
    
    

def get_hapi_nu_range(molecule):
    if molecule not in hapi.tableList():
        return [0., 0.]

    old_stdout = sys.stdout # Memorize the default stdout stream
    sys.stdout = buffer = io.StringIO()
    
    
    hapi.select(molecule)
    
    sys.stdout = old_stdout
    
    s = buffer.getvalue().split("\n")
    s[:] = [x for x in s if x]
    nu_start = np.float32(s[1].split()[1])
    nu_end = np.float32(s[-1].split()[1])
    
    return nu_start, nu_end



def fetch_hapi_data(molecule, pressure, nu_start, nu_end, clear=False):

    if molecule not in hapi.tableList() or clear:
        hapi_fetch(molecule, pressure, nu_start, nu_end)
    if molecule != "CO2":
        if "CO2" not in hapi.tableList() or clear:
            hapi_fetch("CO2", pressure, nu_start, nu_end)



def get_gem_tpvmr(molecule, altitude, myear, ls, lat, lon, lst, plot=False):
    
    gem_d = get_gem_data(myear, ls, lat, lon, lst, plot=plot)
    
    t = np.interp(altitude, gem_d["z"][::-1], gem_d["t"][::-1])
    pressure = np.interp(altitude, gem_d["z"][::-1], gem_d["p"][::-1]) / 101300. #pa to atmosphere
    mol_ppmv = np.interp(altitude, gem_d["z"][::-1], gem_d[molecule.lower()][::-1]) #ppmv
    co2_ppmv = np.interp(altitude, gem_d["z"][::-1], gem_d["co2"][::-1]) #ppmv

    return t, pressure, mol_ppmv, co2_ppmv



def get_abs_coeff(molecule, nu_range, nu_step, mol_ppmv, co2_ppmv, t, pressure, isos=["All"], clear=False):

    fetch_hapi_data(molecule, pressure, nu_range[0], nu_range[-1], clear=clear)
    
    component = {"H2O":1, "CO2":2, "O3":3, "CO":5, "CH4":6, "HCL":15}[molecule]
    
    if isos[0] == "All":
        isos = list(range(len(ISO_IDS[molecule])))
    
    Components=[(component, iso, mol_ppmv/co2_ppmv*hapi.abundance(component, iso)) for iso in isos]
    if molecule != "CO2":
        Components.append((2, 1, (1-mol_ppmv/co2_ppmv)))
    #     Components.append((2, 2, (1-mol_ppmv/co2_ppmv)))
    #     Components.append((2, 3, (1-mol_ppmv/co2_ppmv)))
        
    table_name = make_table_name(molecule, pressure)
    # print("table_name:", table_name)  
    print("molecule:", molecule, "component:", component, "table_name:", table_name)  
    print(Components)
    # print(t, pressure)
    # nu, coef = hapi.absorptionCoefficient_Voigt(SourceTables=molecule, Components=Components, Diluent={'CO2':1.0}, \
    #                                             Environment={'T':t,'p':pressure}, WavenumberStep=nu_step, HITRAN_units=False)
    nu, coef = hapi.absorptionCoefficient_Voigt(Components=Components, SourceTables=table_name, Diluent={'CO2':1.0}, \
                                                Environment={'T':t,'p':pressure}, WavenumberStep=nu_step, HITRAN_units=False)

    return nu, coef



"""examples"""
# if __name__ == "__main__":

    # hapi_fetch("H2O", pressure, 3745, 3835)
    # hapi_fetch("CO2", pressure, 3745, 3835)
    # hapi_fetch("CO", pressure, 4000, 4500)
    
    # molecule = "H2O"
    # altitude = 50.
    # nu_range = [3745, 3835]
    # nu_range = [3057.02, 3081.44] #order 136
    # nu_step = 0.001
    
    # molecule = "CO"
    # altitude = 50.
    # nu_range = [4100, 4300]
    # nu_step = 0.001
    
    # molecule = "CO2"
    # altitude = 50.
    # nu_range = [3349.24, 3375.99]
    # nu_step = 0.001
    
    # get gem data for given altitude and generic lat/lon/my/lst then 
    # occ_sim_nu, occ_sim = hapi_gem_trans(molecule, altitude, nu_range, nu_step, spec_res=None, isos=[1], clear=True)
    # plt.figure()
    # plt.plot(occ_sim_nu, occ_sim)
    
    # occ_sim_nu, occ_sim = hapi_gem_trans(molecule, altitude, nu_range, nu_step, spec_res=None, isos=[2,3,4,5], clear=True)
    # plt.figure()
    # plt.plot(occ_sim_nu, occ_sim)