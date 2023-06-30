# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:16:59 2023

@author: iant
"""


import numpy as np
from matplotlib import pyplot as plt

from tools.datasets.get_gem_data import get_gem_tpvmr

import hapi

import sys
import io


PARAMETER_GROUPS = ['Voigt_CO2']
ISO_IDS = {"H2O":[1, 2, 3, 4, 5, 6, 129], "CO2":[7, 8, 9, 10, 11, 12, 13, 14, 121, 15, 120, 122], "O3":[], "CO":[26, 27, 28, 29, 30, 31], "CH4":[32, 33, 34, 35], "HCL":[52, 53, 107, 108]}


def make_table_name(molecule, pressure):
    table_name = "tmp/%s_%011i" %(molecule, int(pressure*1e9))
    return table_name    


def hapi_fetch(molecule, pressure, nu_range, clear=False):
    #TODO: check why Voigt_CO2 doesn't work with H2O
    
    
    """fetch main isotope for each molecule"""
    table_name = make_table_name(molecule, pressure)
    
    
    # if molecule in ["H2O"]:
    #     component = {"H2O":1, "CO2":2, "O3":3, "CO":5, "CH4":6, "HCL":15}[molecule]
    #     hapi.fetch(table_name, component, 1, nu_range[0], nu_range[1])
    # else:  
    iso_codes = ISO_IDS[molecule]
    if table_name not in hapi.tableList() or clear:
        hapi.fetch_by_ids(table_name, iso_codes, nu_range[0], nu_range[1])#, ParameterGroups=PARAMETER_GROUPS)
        # print(table_name)
        # hapi.describeTable(table_name)


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



def fetch_hapi_data(molecule, pressure, nu_range, clear=False):

    hapi_fetch(molecule, pressure, nu_range, clear=clear)
    #also need CO2 for the diluent
    # if molecule != "CO2":
    #     hapi_fetch("CO2", pressure, nu_range, clear=clear)





def get_abs_coeff(molecule, nu_range, nu_step, mol_ppmv, co2_ppmv, t, pressure, isos=[1,2,3,4,5], clear=False):

    fetch_hapi_data(molecule, pressure, nu_range, clear=clear)
    
    component = {"H2O":1, "CO2":2, "O3":3, "CO":5, "CH4":6, "HCL":15}[molecule]
    
    # print(mol_ppmv, co2_ppmv)
    Components=[(component, iso, mol_ppmv/co2_ppmv*hapi.abundance(component, iso)) for iso in isos]
    # if molecule != "CO2":
    #     Components.append((2, 1, (1-mol_ppmv/co2_ppmv)))
    
    table_names = [make_table_name(molecule, pressure)]
    # if molecule != "CO2":
    #     table_names.append(make_table_name("CO2", pressure))

    # print("molecule:", molecule, "component:", component, "table_names:", table_names)  
    # print(Components)
    
    # print(t, pressure)
    nu, coef = hapi.absorptionCoefficient_Voigt(Components=Components, SourceTables=table_names, Diluent={'CO2':1.0}, \
                                                Environment={'T':t,'p':pressure}, WavenumberStep=nu_step, HITRAN_units=False)

    return nu, coef



"""examples"""
if __name__ == "__main__":
    
    level = 1
    
    
    if level == 0:
        #test basics
        molecule = "H2O"
        altitude = 50.
        nu_range = [3745, 3835]
        myear=34
        ls=180.
        lat=0.
        lon=0.
        lst=12.
        t, pressure, mol_ppmv, co2_ppmv = get_gem_tpvmr(molecule, altitude, myear, ls, lat, lon, lst)
        
        hapi_fetch(molecule, pressure, nu_range, clear=True)
        print(hapi.tableList())
    
        # hapi_fetch("H2O", pressure, [3745, 3835], clear=True)
        # hapi_fetch("CO2", pressure, [3745, 3835])
        # hapi_fetch("CO", pressure, [4000, 4500])

    if level == 1:    
        molecule = "H2O"
        altitude = 50.
        nu_range = [3745, 3835]
        # nu_range = [3057.02, 3081.44] #order 136
        nu_step = 0.001
        
        # molecule = "CO"
        # altitude = 50.
        # nu_range = [4100, 4300]
        # nu_step = 0.001
        
        # molecule = "CO2"
        # altitude = 50.
        # nu_range = [3349.24, 3375.99]
        # nu_step = 0.001
        
        # get gem data for given altitude and generic lat/lon/my/lst then 
        occ_sim_nu, occ_sim = hapi_gem_trans(molecule, altitude, nu_range, nu_step, spec_res=None, isos=[1], clear=True)
        plt.figure()
        plt.plot(occ_sim_nu, occ_sim)
        
        # occ_sim_nu, occ_sim = hapi_gem_trans(molecule, altitude, nu_range, nu_step, spec_res=None, isos=[2,3,4,5], clear=True)
        # plt.figure()
        # plt.plot(occ_sim_nu, occ_sim)