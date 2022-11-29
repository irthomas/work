# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:22:40 2022

@author: iant

TEST HAPI
"""

import sys
import numpy as np
from matplotlib import pyplot as plt


from hapi import *

sys.path.append(r"C:\Users\iant\Dropbox\NOMAD\Python\repos\pytran")
import pytran




hitran21_fname = '20211005_CO_HITRAN_CO2_broadened.par' 
filename = hitran21_fname

data = pytran.read_hitran2012_parfile(filename, 4200, 4300, Smin=0)  # 1.e-25
data_pytran = {}

molec = 'CO'
molecule = molec
M = pytran.get_molecule_id(molecule)

wnstep = 0.001
wnrange = [4200, 4300]
wnmin, wnmax = (wnrange[0], wnrange[1])
dwn = wnstep  # 0.001
Nbwn = int(np.ceil((wnmax - wnmin) / dwn) + 1)
wns = np.linspace(wnmin, wnmax, Nbwn)
    


ind_molec = np.nonzero(data['M'] == M)[0]
data_pytran[molecule] = {key: value[ind_molec] for key, value in data.items()}




fetch_by_ids('CO',[26,36,28,27,38,37], 4200, 4300, ParameterGroups=['Voigt_CO2'])
# fetch('CO',,4200, 4300, ParameterGroups=['Voigt_CO2'])
# describeTable('CO')



# for pressure_exponential in range(-3, 3):

#     plt.figure(figsize=(15, 8))
#     pressure = 10.0 ** pressure_exponential
    
#     plt.title("%0.0e" %pressure)

#     nu, coef1 = absorptionCoefficient_Voigt(SourceTables='CO', Diluent={'CO2':1.0}, Environment={'T':296.,'p':pressure})
#     # plt.plot(nu+pressure_exponential, coef1, label='p=HAPI %0.1e' %(10.0 ** pressure_exponential))
#     plt.plot(nu, coef1, label='p=HAPI %0.1e' %(10.0 ** pressure_exponential))

#     xsec = pytran.hitran.calculate_hitran_xsec(data_pytran[molecule], M, wns, T=296., P=pressure * 101325.0)
#     plt.plot(wns+0.3, xsec, label='p=Pytran %0.1e' %(10.0 ** pressure_exponential))


#     plt.legend()




pressure_exponential = -2.0

plt.figure(figsize=(15, 8))
pressure = 10.0 ** pressure_exponential

plt.title("%0.0e" %pressure)

nu, coef1 = absorptionCoefficient_Voigt(SourceTables='CO', Diluent={'CO2':1.0}, Environment={'T':296.,'p':pressure}, WavenumberStep=0.001)
# plt.plot(nu+pressure_exponential, coef1, label='p=HAPI %0.1e' %(10.0 ** pressure_exponential))
plt.plot(nu, coef1, label='p=HAPI %0.1e' %(10.0 ** pressure_exponential))

xsec = pytran.hitran.calculate_hitran_xsec(data_pytran[molecule], M, wns, T=296., P=pressure * 101325.0)
plt.plot(wns+0.3, xsec, label='p=Pytran %0.1e' %(10.0 ** pressure_exponential))


plt.legend()
