# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:41:12 2023

@author: iant
"""

from tools.datasets.get_gem_lut import get_gem_lut_tpvmr


def get_molecules(molecules, geom_d, plot=False):

    molecule_d = {}
    for molecule in molecules.keys():
        isos = molecules[molecule]["isos"]

        #get gem data, interpolate onto altitude grid        
        ts, pressures, mol_ppmvs, co2_ppmvs = get_gem_lut_tpvmr(
            molecule, geom_d["alt_grid"], geom_d["myear"], geom_d["ls"], \
            geom_d["lat"], geom_d["lon"], geom_d["lst"], plot=plot)

        molecule_d[molecule] = {}
        molecule_d[molecule]["ts"] = ts
        molecule_d[molecule]["pressures"] = pressures
        molecule_d[molecule]["mol_ppmvs"] = mol_ppmvs
        molecule_d[molecule]["co2_ppmvs"] = co2_ppmvs
        molecule_d[molecule]["isos"] = isos

        return molecule_d