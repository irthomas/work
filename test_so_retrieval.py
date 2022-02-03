# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:08:51 2022

@author: iant

SO RETRIEVALS
"""


import numpy as np
import os
import sys
from scipy import interpolate

import matplotlib.pyplot as plt



sys.path.append(r'C:\Users\iant\Dropbox\NOMAD\Python\repos\pytran')
import pytran
# import repos.pytran.pytran

# sys.path.append('/bira-iasb/projects/planetary/justint/NOMADTOOLS')
sys.path.append(r'C:\Users\iant\Dropbox\NOMAD\Python\repos\nomad_tools')
import nomadtools





"""new spectral calibration functions Aug/Sep 2021"""
def sinc_gd(dx,width,lobe,asym,offset):
    #goddard version
    sinc = (width*np.sin(np.pi*dx/width)/(np.pi*dx))**2.0
    ind = (abs(dx)>width).nonzero()[0]
    if len(ind)>0: sinc[ind] = sinc[ind]*lobe
    ind = (dx<=-width).nonzero()[0]
    if len(ind)>0: sinc[ind] = sinc[ind]*asym
    sinc += offset
    return sinc


def F_aotf3(dx, d):
    
    offset = d["aotfg"] * np.exp(-dx**2.0 / (2.0 * d["aotfgw"]**2.0))

    sinc = sinc_gd(dx,d["aotfw"],d["aotfs"],d["aotfa"], offset)
    
    return sinc


def F_blaze3(x, blazef, blazew):
    
    dx = x - blazef
    F = np.sinc((dx) / blazew)**2
    return F






def init_atmo(TangentAlt, atmo_filename=None, apriori_version='apriori_1_1_1_GEMZ_wz_mixed', apriori_zone='AllSeasons_AllHemispheres_AllTime'):
  

    import numpy as np
    from scipy import interpolate
    
    # NbZ = len(TangentAlt)
    
    if atmo_filename is None:
      atmo_filename = nomadtools.get_apriori_files(name='atmo', apriori_version=apriori_version, apriori_zone=apriori_zone) 

    atmo_in = {}
    atmo_in['Z'], atmo_in['T'], atmo_in['P'], atmo_in['NT'] = np.loadtxt(os.path.join(nomadtools.rcParams['paths.dirAtmosphere'], atmo_filename), comments='%', usecols=(0,1,2,3,), unpack=True)
    
    atmo = {}
    atmo['Z'] = TangentAlt[:]
    fun_T = interpolate.interp1d(atmo_in['Z'][::-1], atmo_in['T'][::-1])
    fun_P = interpolate.interp1d(atmo_in['Z'][::-1], np.log(atmo_in['P'][::-1]))
    fun_NT = interpolate.interp1d(atmo_in['Z'][::-1], np.log(atmo_in['NT'][::-1]))
    atmo['T'] = [fun_T(z) for z in TangentAlt]
    atmo['P'] = np.exp([fun_P(z) for z in TangentAlt])
    atmo['NT'] = np.exp([fun_NT(z) for z in TangentAlt])
    
    return atmo


LOG_LEVEL = 5

def init_molecules(mol_dict, nu_hr, apriori_version='apriori_1_1_1_GEMZ_wz_mixed', apriori_zone='AllSeasons_AllHemispheres_AllTime', 
                        nu_lp_min=None, nu_lp_max=None, fbord=25., **kwargs):

    nu_lp_min = self.nu_hr[0] - fbord
    
    if nu_lp_max is None:
      nu_lp_max = self.nu_hr[-1] + fbord
    
    #
    NbMol = len(mol_dict)
    xa_mol = {}
    sa_mol = {}
    sigma_mol = {}
    for mol in mol_dict:
    
      if LOG_LEVEL >= 2:
        print(mol, mol_dict[mol])
      molname = mol_dict[mol].get('molname', mol)
    
      #
      xa = mol_dict[mol].get('xa', 'gem')
      if xa == 'gem':
        xa_file, sa_file = nomadtools.get_apriori_files(name=molname, apriori_version=apriori_version, apriori_zone=apriori_zone)
        if LOG_LEVEL >= 2:
          print('Reading in apriori vmr from ', os.path.basename(xa_file))
        za_in, xa_in = np.loadtxt(os.path.join(nomadtools.rcParams['paths.dirAtmosphere'], xa_file), comments='%', usecols=(0,1,), unpack=True)
        xa_fun = interpolate.interp1d(za_in[::-1], xa_in[::-1])
        self.xa_mol[mol] = xa_fun(self.atmo['Z'])*1e-6
        #if mol == 'H2O_ice':
        #  self.xa_mol[mol] *= 7.79e-12  # correct for molecules -> patricles
      else:
        if LOG_LEVEL >= 2:
          print('Setting vmr to constant %f ppm'%xa)
        self.xa_mol[mol] = np.ones_like(self.atmo['Z'])*xa*1e-6
      if 'xfact' in mol_dict[mol]:
        self.xa_mol[mol] *= mol_dict[mol]['xfact']
    
      #
      sa = mol_dict[mol].get('sa', 'gem')
      if sa == 'gem':
        xa_file, sa_file = nomadtools.get_apriori_files(name=molname, apriori_version=apriori_version, apriori_zone=apriori_zone)
        if LOG_LEVEL >= 2:
          print('Reading in apriori sa from ', os.path.basename(sa_file))
        za_in, sa_in = np.loadtxt(os.path.join(nomadtools.rcParams['paths.dirAtmosphere'], sa_file), comments='%', usecols=(0,1,), unpack=True)
        sa_fun = interpolate.interp1d(za_in[::-1], sa_in[::-1])
        self.sa_mol[mol] = sa_fun(self.atmo['Z'])
      else:
        if LOG_LEVEL >= 2:
          print('Setting sa to constant %f'%xa)
        self.sa_mol[mol] = np.ones_like(self.atmo['Z'])*sa
      
    
      #
      abs_type = mol_dict[mol].get('type', 'hitran')
      if LOG_LEVEL >= 2:
        print(abs_type)
      if abs_type == 'hitran':  
        M = pytran.get_molecule_id(molname)
        str_min = mol_dict[mol].get('str_min',1e-26)
    
        HITRANDIR = nomadtools.rcParams['paths.dirLP']
        filename = os.path.join(HITRANDIR, '%02i_hit16_2000-5000_CO2broadened.par' % M)
        if not os.path.exists(filename):
          filename = os.path.join(HITRANDIR, '%02i_hit16_2000-5000.par' % M)
        #print(filename)
    
        LineList = pytran.read_hitran2012_parfile(filename, nu_lp_min, nu_lp_max, Smin=str_min)
        nlines = len(LineList['S'])
        if LOG_LEVEL >= 2:
          print('Found %i lines' % nlines)
        #self.LineList = LineList
    
        if M == 1:
          #print('adjusting HDO frac on %d lines' % sum(LineList['I']>3))
          LineList['S'][LineList['I']>3] *= 5        
    
        self.sigma_mol[mol] = np.zeros((self.NbZ,self.Nbnu_hr))
        if nlines > 0:
          for i in range(self.NbZ):
            if LOG_LEVEL >= 3:
              print("%d of %d" % (i, self.NbZ), self.xa_mol[mol][i])
            self.sigma_mol[mol][i,:] =  pytran.calculate_hitran_xsec(LineList, M, self.nu_hr, T=self.atmo['T'][i], P=self.atmo['P'][i]*1e2, qmix=self.xa_mol[mol][i])
    
      elif abs_type == 'xsec':
    
        #self.sigma_mol[mol] = np.zeros((NbZ,Nbnu_hr))
        self.sigma_mol[mol] = get_aero_xsec(self.nu_hr, molname)
    
      else:
        raise Exception("abs_type '%s' not recognized"%abs_type)





#2nd october slack
aotfwc  = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
aotfsc  = [ 8.10749274e-07, -3.30238496e-03,  4.08845247e+00] # sidelobes factor [scaler from AOTF frequency cm-1]
aotfac  = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
aotfgc  = [ 1.49266526e-07, -9.63798656e-04,  1.60097815e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]
# Calibration coefficients
cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])                  # Frequency of AOTF [cm-1 from kHz]
cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])            # Blaze free-spectral-range (FSR) [cm-1 from pixel]
ncoeff  = [-2.44383699e-07, -2.30708836e-05, -1.90001923e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
aotfts  = -6.5278e-5                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
blazep  = [-1.00162255e-11, -7.20616355e-09, 9.79270239e-06, 2.25863468e+01] # Dependence of blazew from AOTF frequency


d = {}

"""
d[wavenumber] = {b:blaze, a:aotf, nu_c:centre order nu}

"""
order_c = 189
pixels = np.arange(320)
tempa = -5.0
tempg = -5.0
blaze_shift = 0.0
line = 4383.5
aotf = 26600


mol_dict = {
   # 'CO' : {'xa':'gem', 'sa':'gem', },
   'H2O' : {'xa':'gem', 'sa':'gem', },
}
TangentAlt = np.arange(10., 100., 5.)


atmo = init_atmo(TangentAlt)
# mol_dict=mol_dict, order=order, adj_orders=0, pixel_shift=pixel_shift, spec_res=spec_res, TangentAlt=TangentAlt,
# nu_lp_min=nu_pm[0], nu_lp_max=nu_pm[-1]


px_hr = np.linspace(-10., 330, num=10000)

pixf_hr_c = np.polyval(cfpixel, px_hr) * order_c
pixf_hr_c += pixf_hr_c * np.polyval(ncoeff, tempg) + blaze_shift

for i, nu in enumerate(pixf_hr_c):
    d[nu] = []


for order in range(order_c - 3, order_c + 4):

    pixf_hr = np.polyval(cfpixel, px_hr) * order
    pixf_hr += pixf_hr * np.polyval(ncoeff, tempg) + blaze_shift
    
    
    blazew =  np.polyval(blazep, line - 3700.0)        # FSR (Free Spectral Range), blaze width [cm-1]
    blazew += blazew*np.polyval(ncoeff,tempg)        # FSR, corrected for temperature
    
    blazef = order*blazew
    F_blaze = F_blaze3(pixf_hr, blazef, blazew)
    
    
    aotff = np.polyval(cfaotf, aotf) + tempa*aotfts
    aotf_dict = {}
    aotf_dict["aotfw"] = np.polyval(aotfwc,aotff)
    aotf_dict["aotfs"] = np.polyval(aotfsc,aotff)
    aotf_dict["aotfa"] = np.polyval(aotfac,aotff)
    aotf_dict["aotfo"] = np.polyval(aotfoc,aotff)
    aotf_dict["aotfg"] = np.polyval(aotfgc,aotff)
    aotf_dict["aotfgw"] = 50. #offset width cm-1
    
    
    dx = pixf_hr - aotff
    F_aotf = F_aotf3(dx, aotf_dict)
    
    
    for i, nu in enumerate(pixf_hr_c):
        d[nu].append({"b":F_blaze[i], "a":F_aotf[i], "nu_c":pixf_hr[i]})

# #resolving power
# rp = 1000000.0

# #rp = nu/d_nu
# d_nu_hr = pixf[160] / rp

# #centre order high res grid
# nu_hr = np.arange(pixf[0] - 1.0, pixf[-1] + 1.0, d_nu_hr)


