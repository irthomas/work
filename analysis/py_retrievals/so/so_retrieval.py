# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 21:29:25 2021

@author: iant

ANALYSE OBSERVATION WITH RETRIEVAL:
    1) CHECK RAW SPECTRA FOR SPECTRAL DRIFT AND DEFINE X
    2) USE NEW PARAMETERS TO SIMULATE RAW SPECTRUM AND CALIBRATED SPECTRUM
    3) 
"""


import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
# import argparse
from scipy import interpolate
from datetime import datetime

from tools.plotting.colours import get_colours
from tools.file.hdf5_functions import make_filelist
from tools.file.get_hdf5_temperatures import get_interpolated_temperatures
from tools.general.get_nearest_index import get_nearest_index
from tools.asimut.ils_params import get_ils_params
from tools.spectra.baseline_als import baseline_als

from instrument.nomad_so_instrument import F_aotf_goddard18b, F_blaze, nu_mp

# from instrument.calibration.so_aotf_ils.simulation_functions import (get_file, get_data_from_file, select_data, fit_temperature,
# get_start_params, make_param_dict, calc_spectrum, fit_spectrum, area_under_curve, get_solar_spectrum)
# from instrument.calibration.so_aotf_ils.simulation_config import sim_parameters

# from instrument.calibration.so_aotf_ils.simulation_functions import get_absorption_line_indices

sys.path.append(r"C:\Users\iant\Dropbox\NOMAD\Python\repos\pytran")
from repos.pytran import pytran

HITRANDIR = r"C:\Users\iant\Documents\DATA\Radiative_Transfer\Auxiliary_Files\Spectroscopy"
NEW = True
NEW = False

filename = "20180626_142634_0p3k_SO_A_E_168"
atmo_filename = "20180626_142634_1p0a_SO_A_E_168"

# filename = "20180930_113957_1p0a_SO_A_I_189"
file_level = "hdf5_level_0p3k"

index = 132
pixels = np.arange(320)
NbP = len(pixels)
order = int(filename[-3:])
chosen_bin = 3
dnu = 0.003377402 #copy asimut 
# dnu = 0.001


t_aotf = 3.3759243
t_grating = 4.3769007# + 1.5


AOTF_OFFSET_SHAPE = "Constant"

orders = np.arange(order-4, order+5)

#do this better
if order == 168:
    nu_hr = np.arange(3691., 3896., dnu) #copy asimut psg comparison
if order == 189:
    nu_hr = np.arange(4150., 4350., dnu)

Nbnu_hr = len(nu_hr)







def get_all_x(temperatures, pixels, order):   
    
    
    x_array = np.zeros([len(temperatures), len(pixels)])
    
    #slack 29th August 2021
    cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])   # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    ncoeff  = [-1.76520810e-07, -2.26677449e-05, -1.93885521e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]

    for i, t in enumerate(temperatures):
        xdat  = np.polyval(cfpixel, pixels) * order
        xdat += xdat * np.polyval(ncoeff, t)
    
        x_array[i, :] = xdat
        
    return x_array


def calc_x_hdf5(hdf5_file):
    
    n_pixels = len(hdf5_file["Science/Y"][0, :])
    pixels = np.arange(n_pixels)
    order = hdf5_file["Channel/DiffractionOrder"][0]
    temperatures = get_interpolated_temperatures(hdf5_file, "so")
    
    x_array = get_all_x(temperatures, pixels, order)
    return x_array





def get_cal_params(aotf, orders, tempg, tempa):
    #from blazecalc.py 17/9/21

    # AOTF shape parameters
    aotfwc  = [-1.78088527e-07,  9.44266907e-04,  1.95991162e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    aotfsc  = [ 1.29304371e-06, -6.77032965e-03,  1.03141366e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
    aotfac  = [-1.96949242e-07,  1.48847262e-03, -1.40522510e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
    aotfgc  = [ 1.07865793e-07, -7.20862528e-04,  1.24871556e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]
    
    # Calibration coefficients
    cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])         # Frequency of AOTF [cm-1 from kHz]
    cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])   # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    ncoeff  = [-1.76520810e-07, -2.26677449e-05, -1.93885521e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
    blazep  = [-5.76161e-14,-2.01122e-10,2.02312e-06,2.25875e+01] # Dependence of blazew from AOTF frequency
    aotfts  = -6.5278e-5                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
    # norder  = 6                                                   # Number of +/- orders to be considered in order-addition
    
    d = {}
    # Calculate blaze parameters
    aotff = np.polyval(cfaotf, aotf) + tempa*aotfts  # AOTF frequency [cm-1], temperature corrected
    blazew =  np.polyval(blazep,aotf-22000.0)        # FSR (Free Spectral Range), blaze width [cm-1]
    blazew += blazew*np.polyval(ncoeff,tempg)        # FSR, corrected for temperature
    # order = round(aotff/blazew)                      # Grating order
    
    d["blazew"] = blazew
    d["aotff"] = aotff
    
    # Compute AOTF parameters
    d["aotfw"] = np.polyval(aotfwc,aotff)
    d["aotfs"] = np.polyval(aotfsc,aotff)
    d["aotfa"] = np.polyval(aotfac,aotff)
    d["aotfo"] = np.polyval(aotfoc,aotff)
    d["aotfg"] = np.polyval(aotfgc,aotff)
    d["aotfgw"] = 50. #offset width cm-1
    
    # Frequency of the pixels
    for order in orders:
        d[order] = {}
        pixf = np.polyval(cfpixel,range(320))*order
        pixf += pixf*np.polyval(ncoeff, tempg)
        d[order]["blazef"] = order*d["blazew"]                            # Center of the blaze
        d[order]["pixf"] = pixf
    
    return d
    

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


def F_blaze3(x, d, order):
    
    dx = x - d[order]["blazef"]
    F = np.sinc((dx) / d["blazew"])**2
    return F





regex = re.compile(filename) #(approx. orders 188-202) in steps of 8kHz

hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)

hdf5_file = hdf5_files[0]
hdf5_filename = hdf5_filenames[0]

temperatures = get_interpolated_temperatures(hdf5_file, "so")
x_all = calc_x_hdf5(hdf5_file)

y_all = hdf5_file["Science/Y"][:, :]
aotf_freq = hdf5_file["Channel/AOTFFrequency"][0]

alts_all = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
bins = hdf5_file["Science/Bins"][:, 0]
unique_bins = sorted(list(set(bins)))

bin_indices = np.where(bins == unique_bins[chosen_bin])[0]

y_bin = y_all[bin_indices, :]
x_bin = x_all[bin_indices, :]
alts_bin = alts_all[bin_indices]

y = y_bin[index, :]
x = x_bin[index]
chosen_alt = alts_bin[index]
t = temperatures[index]

y_cont = baseline_als(y)
y_cr = y / y_cont

index_all_bins = get_nearest_index(chosen_alt, alts_all)

# plt.figure()
# plt.plot(x, y)

d = get_ils_params(hdf5_filename, x, save_file=False)
d2 = get_cal_params(aotf_freq, orders, t_grating, t_aotf)





# def init_atmo(self, TangentAlt=None, atmo_filename=None, apriori_version='apriori_1_1_1_GEMZ_wz_mixed', apriori_zone='AllSeasons_AllHemispheres_AllTime', **kwargs):
    


TangentAlt = np.sort(np.append(np.arange(60., 130., 2.), chosen_alt))
alt_index = np.where(TangentAlt == chosen_alt)[0][0]

NbZ = len(TangentAlt)

atmo_dirpath = r"C:\Users\iant\Documents\DATA\retrievals\so_aotf_ils\CO\Atmosphere\gem-mars-a758"
atmo_filepath = os.path.join(atmo_dirpath, atmo_filename, "gem-mars-a758_%s_sp%04i.dat" %(atmo_filename, index_all_bins))
# atmo_filepath = r"C:\Users\iant\Documents\DATA\retrievals\so_aotf_ils\CO\Atmosphere\gem-mars-a585\apriori_1_1_1_GEMZ_wz_mixed\gem-mars-a585_AllSeasons_AllHemispheres_AllTime_mean_atmo.dat"

print('Reading in atmo from ', atmo_filepath)
atmo_in = {}
atmo_in['Z'], atmo_in['T'], atmo_in['P'], atmo_in['NT'] = np.loadtxt(atmo_filepath, comments='%', usecols=(0,1,2,3,), unpack=True)

atmo = {}
atmo['Z'] = TangentAlt[:]
fun_T = interpolate.interp1d(atmo_in['Z'][::-1], atmo_in['T'][::-1])
fun_P = interpolate.interp1d(atmo_in['Z'][::-1], np.log(atmo_in['P'][::-1]))
fun_NT = interpolate.interp1d(atmo_in['Z'][::-1], np.log(atmo_in['NT'][::-1]))
atmo['T'] = np.array([fun_T(z) for z in TangentAlt])
atmo['P'] = np.exp([fun_P(z) for z in TangentAlt])
atmo['NT'] = np.exp([fun_NT(z) for z in TangentAlt])

# import json
# atmo_out = {}
# for key, value in atmo.items():
#     print(key, type(value))
#     if type(value) == np.ndarray:
#         atmo_out[key] = value.tolist()
#     else:
#         atmo_out[key] = value
# with open("test_ian.json", "w") as f:
#     json.dump(atmo_out, f)

Rp = 3396.
s = np.zeros(NbZ)
dl = np.zeros((NbZ,NbZ))
for i in range(NbZ):
    s[i:] = np.sqrt((Rp+atmo['Z'][i:])**2-(Rp+atmo['Z'][i])**2)
    #print(i, s[i:])
    if i < NbZ-1:
        dl[i,i] = s[i+1] - s[i]
    if i < NbZ-2:
        dl[i,(i+1):-1] = s[(i+2):] - s[i:-2]
    dl[i,-1] = s[-1] - s[-2] + 2*10. /np.sqrt(1.-((Rp+atmo['Z'][i])/(Rp+atmo['Z'][-1]+1.))**2) 
    # print(dl[i,i:])
dl *= 1e5



  # def init_molecules(self, mol_dict={'CO2':{}}, apriori_version='apriori_1_1_1_GEMZ_wz_mixed', apriori_zone='AllSeasons_AllHemispheres_AllTime', 
  #                       nu_lp_min=None, nu_lp_max=None, fbord=25., **kwargs):


mol_dict = {}

if order == 168:
    mol_dict["H2O"] = {"xa":100.0, "sa":0.1, "par":"01_hitran16_Regalia19.par", "col":7, "str_min":1e-25}
    # mol_dict["CO2"] = {"xa":1.0, "sa":0.1, "par":"02_hit16_2000-5000_CO2broadened.par", "col":6}
if order == 189:
    mol_dict["CO"] = {'xa':'gem', 'sa':0.1, "col":5}

nu_lp_min = nu_hr[0]
nu_lp_max = nu_hr[-1]

    
if "sigma_mol" not in locals():
# if True:
    NbMol = len(mol_dict)
    xa_mol = {}
    sa_mol = {}
    sigma_mol = {}
    for mol in mol_dict:
    
        print(mol, mol_dict[mol])
        molname = mol_dict[mol].get('molname', mol)
        
        xa = mol_dict[mol].get('xa', 'gem')
        if xa == 'gem':
            print('Reading in apriori vmr from ', os.path.basename(atmo_filepath))
            
            # atmo_mean_filepath = r"C:\Users\iant\Documents\DATA\retrievals\so_aotf_ils\CO\Atmosphere\gem-mars-a585\apriori_1_1_1_GEMZ_wz_mixed\gem-mars-a585_AllSeasons_AllHemispheres_AllTime_mean_H2O.dat"
            # atmo_stdev_filepath = r"C:\Users\iant\Documents\DATA\retrievals\so_aotf_ils\CO\Atmosphere\gem-mars-a585\apriori_1_1_1_GEMZ_wz_mixed\gem-mars-a585_AllSeasons_AllHemispheres_AllTime_stdev_H2O.dat"
            
            # za_in, xa_in = np.loadtxt(atmo_mean_filepath, comments='%', usecols=(0,1,), unpack=True)
            za_in, xa_in = np.loadtxt(atmo_filepath, comments='%', usecols=(0,mol_dict[mol]["col"],), unpack=True)
            xa_fun = interpolate.interp1d(za_in[::-1], xa_in[::-1])
            xa_mol[mol] = xa_fun(atmo['Z'])*1e-6
            # print(xa_mol[mol])
            
        else:
            print('Setting vmr to constant %f ppm'%xa)
            xa_mol[mol] = np.ones_like(atmo['Z'])*xa*1e-6
        if 'xfact' in mol_dict[mol]:
            xa_mol[mol] *= mol_dict[mol]['xfact']
        
        #
        sa = mol_dict[mol]["sa"]#.get('sa', 'gem')
        # if sa == 'gem':
        #   xa_file, sa_file = nomadtools.get_apriori_files(name=molname, apriori_version=apriori_version, apriori_zone=apriori_zone)
        #   if LOG_LEVEL >= 2:
        #     print('Reading in apriori sa from ', os.path.basename(sa_file))
        #   za_in, sa_in = np.loadtxt(os.path.join(nomadtools.rcParams['paths.dirAtmosphere'], sa_file), comments='%', usecols=(0,1,), unpack=True)
        #   sa_fun = interpolate.interp1d(za_in[::-1], sa_in[::-1])
        #   self.sa_mol[mol] = sa_fun(self.atmo['Z'])
        # else:
        #   if LOG_LEVEL >= 2:
        print('Setting sa to constant %f' %sa)
        sa_mol[mol] = np.ones_like(atmo['Z'])*sa
        
        #get data from hitran
        M = pytran.get_molecule_id(molname)
        str_min = mol_dict[mol].get("str_min", 1e-26)
      
        hitran_filename = os.path.join(HITRANDIR, mol_dict[mol]["par"])
        
        print("Reading in hitran file %s" %os.path.basename(hitran_filename))
      
        LineList = pytran.read_hitran2012_parfile(hitran_filename, nu_lp_min, nu_lp_max, Smin=str_min)
        nlines = len(LineList['S'])
        print('Found %i lines' % nlines)
      
        # if M == 1:
        #     LineList['S'][LineList['I']>3] *= 5        
      
        sigma_mol[mol] = np.zeros((NbZ, Nbnu_hr))
        if nlines > 0:
            for i in range(NbZ):
                print("%d of %d" % (i, NbZ), xa_mol[mol][i])
                sigma_mol[mol][i,:] =  pytran.calculate_hitran_xsec(LineList, M, nu_hr, T=atmo['T'][i], P=atmo['P'][i]*1e2, qmix=xa_mol[mol][i])
    
    tau_hr = np.zeros((NbZ, Nbnu_hr))
    Trans_hr = np.zeros((NbZ, Nbnu_hr))
    
    
    sigma_hr = 0.0
    for mol in sigma_mol:
        sigma_hr += xa_mol[mol][:, None] * sigma_mol[mol]
    
    tau_hr[:, :] = 0.0
    for i in range(NbZ):
        for j in range(i, NbZ):
            tau_hr[i, :] += (atmo['NT'][j]*dl[i, j]) * sigma_hr[j, :]
        Trans_hr[i, :] = np.exp(-tau_hr[i, :])




"""spectral cal"""
def nu_mp2(m, p, t):

    cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])  # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    tcoeff  = np.array([-0.736363, -6.363908])                   # Blaze frequency shift due to temperature [pixel from Celsius]
    
   
    xdat  = np.polyval(cfpixel, p) * m
    dpix = np.polyval(tcoeff, t)
    xdat += dpix*(xdat[-1]-xdat[0])/320.0
    return xdat



def nu_mp3(m, p, t):

    cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])   # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    ncoeff  = np.array([-1.76520810e-07, -2.26677449e-05, -1.93885521e-04]) # Relative frequency shift coefficients [shift/frequency from Celsius]
        
    pixf  = np.polyval(cfpixel,p)*m
    pixf += pixf*np.polyval(ncoeff, t)
    
    return pixf


"""blaze cm-1"""
def F_blaze_centre(order):
    blaze_centre_168 = 3793.194407
    
    blaze_centre = blaze_centre_168 * (order / 168.)
    return blaze_centre    




def F_blaze2(x, order):
    
    blaze_centre = F_blaze_centre(order)
    
    fsr = 22.578538

    dx = x - blaze_centre
    F = np.sinc((dx) / fsr)**2
    return F







"""reverse AOTF asymmetry"""
def sinc(dx, amp, width, lobe, asym):
    # """asymetry switched 
 	sinc = amp * (width * np.sin(np.pi * dx / width) / (np.pi * dx))**2

 	ind = (abs(dx)>width).nonzero()[0]
 	if len(ind)>0: 
         sinc[ind] = sinc[ind]*lobe

 	# ind = (dx>=width).nonzero()[0]
 	ind = (dx<=width).nonzero()[0] 
 	if len(ind)>0: 
         sinc[ind] = sinc[ind]*asym

 	return sinc





def aotf_offset(variables, x):

    dx = x - variables["nu_offset"] + 1.0e-6

    if AOTF_OFFSET_SHAPE == "Constant":
        offset = variables["offset"]
    else:
        offset = variables["offset_height"] * np.exp(-dx**2.0/(2.0*variables["offset_width"]**2.0))

    return offset    


def F_aotf2(variables, x):
    
    dx = x - variables["nu_offset"] + 1.0e-6
    offset = aotf_offset(variables, x)
    
    sinc_raw = sinc(dx, variables["aotf_amplitude"], variables["aotf_width"], variables["sidelobe"], variables["asymmetry"])
    sinc_norm = sinc_raw / max(sinc_raw) * (1 - offset)
    
    F = sinc_norm + offset
    
    #normalise offset
    if AOTF_OFFSET_SHAPE != "Constant":
        variables["offset_height"] = variables["offset_height"] / np.max(F)

    return F


"""aotf temperature correction"""
# aotf_shift = aotfts*t_aotf*3790.4057


variables =  {
        "aotf_width":20.5049,
        "aotf_amplitude":1.0,
        "sidelobe":3.09414,
        "asymmetry":1.33845,
        "nu_offset":3790.4057,
}
if AOTF_OFFSET_SHAPE == "Constant":
    variables["offset"] = 0.04504





"""hr aotf"""
W_aotf = F_aotf2(variables, nu_hr)
W_aotf2 = sinc_gd(nu_hr - 3790.4057, 20.5049, 3.09414, 1.33845, 0.04504)

if NEW:
    W_aotf3 = F_aotf3(nu_hr - d2["aotff"], d2)

aotf2 = Trans_hr[alt_index, :] * W_aotf2
# aotf3 = Trans_hr[alt_index, :] * W_aotf3

# I_hr = 
plt.figure(figsize=(12,8), constrained_layout=True)
# plt.plot(nu_hr, Trans_hr[alt_index, :] * W_aotf , label="aotf")
plt.plot(nu_hr, aotf2/max(aotf2), label="aotf2")
nu, F= np.loadtxt(r"C:\Users\iant\Documents\DOCUMENTS\presentations\20180626_142634_1p0a_SO_A_E_168_94_test_fwd_SP1_FEN1_rad_aotf.dat", dtype=float, comments="%", usecols=(0,1,), unpack=True)
plt.plot(nu, F/max(F), label="rad_aotf")
plt.legend(loc="upper right")

# plt.figure()
# plt.plot(nu_hr, I_hr)

I_hr_norm = Trans_hr[alt_index, :] * W_aotf + (1 - W_aotf)
# plt.figure()
# plt.plot(nu_hr, I_hr_norm)

"""hr order addition"""
# fsr = 22.578538
# px_centre = nu_mp2(order, pixels, t)
# centre_indices = np.where((nu_hr > px_centre[0]) & (nu_hr < px_centre[-1]))[0]
# plt.plot(nu_hr[centre_indices], I_hr[centre_indices])


#add ILS

# nu_sp = np.arange(-1,1,0.01)    
# p = 160
# #make ils shape
# a1 = 0.0
# a2 = d["width"][p]
# a3 = 1.0
# a4 = d["displacement"][p]
# a5 = d["width"][p]
# a6 = d["amplitude"][p]
# ils0=a3 * np.exp(-0.5 * ((nu_sp + a1) / a2) ** 2)
# ils1=a6 * np.exp(-0.5 * ((nu_sp + a4) / a5) ** 2)
# ils = ils0 + ils1 

px_centre = nu_mp2(order, pixels, t_grating)
if NEW:
    px_centre = d2[order]["pixf"]

sconv = 0.2/2.355

# plt.figure(figsize=(12,8), constrained_layout=True)

W_conv = np.zeros((NbP, Nbnu_hr))
for iord in orders:
    nu_p = nu_mp2(iord, pixels, t_grating)
    if NEW:
        nu_p = d2[iord]["pixf"]
    print('order %d: %.1f to %.1f' % (iord, nu_p[0], nu_p[-1]))
    W_blaze = F_blaze2(nu_p, iord)
    for ip in pixels:
        inu1 = np.searchsorted(nu_hr, nu_p[ip] - 5.*sconv) #start index
        inu2 = np.searchsorted(nu_hr, nu_p[ip] + 5.*sconv) #end index
        
        nu_sp = nu_hr[inu1:inu2] - nu_p[ip]
        
        #make ils shape
        a1 = 0.0
        a2 = d["width"][ip]
        a3 = 1.0
        a4 = d["displacement"][ip]
        a5 = d["width"][ip]
        a6 = d["amplitude"][ip]
        
            
            
        ils0=a3 * np.exp(-0.5 * ((nu_sp + a1) / a2) ** 2)
        ils1=a6 * np.exp(-0.5 * ((nu_sp + a4) / a5) ** 2)
        
        ils = ils0 + ils1 

    
        # main_exp = 
        W_conv[ip,inu1:inu2] += (W_blaze[ip]) * ils
        # W_conv[ip,inu1:inu2] += (W_blaze[ip] * dnu)/(np.sqrt(2.0 * np.pi) * sconv) * np.exp(-(nu_hr[inu1:inu2] - nu_p[ip])**2 / (2. *sconv**2))
        # if ip == 319:
        #     plt.plot(nu_sp, ils + iord/1000.)
        #     plt.plot(nu_sp, (W_blaze[ip] * dnu)/(np.sqrt(2.0 * np.pi) * sconv) * np.exp(-(nu_hr[inu1:inu2] - nu_p[ip])**2 / (2. *sconv**2)) + iord/1000.)


I0_hr = W_aotf
I0_p = np.matmul(W_conv, I0_hr)  # np x 1
I_hr = I0_hr[None,:] * Trans_hr  # nz x nhr
I_p = np.matmul(W_conv, I_hr.T).T  # nz x np
Trans_p = I_p / I0_p[None,:]     # nz x np


fig = plt.figure(figsize=(18, 10), constrained_layout=True)
gs = fig.add_gridspec(5, 1)
ax1a = fig.add_subplot(gs[0:2, 0])
ax1b = fig.add_subplot(gs[2, 0], sharex=ax1a)
ax1c = fig.add_subplot(gs[3, 0], sharex=ax1a)
ax1d = fig.add_subplot(gs[4, 0], sharex=ax1a)


fig.suptitle("AOTF + Blaze + OA + Double ILS: blazecalc.py AOTF and blaze")
# fig.suptitle("AOTF + Blaze + OA + Double ILS: tcoeff=[-0.736363,-6.363908] (from aotf.py), t_grating=4.3769007")
# fig.suptitle("AOTF + Blaze + OA + Double ILS: ncoeff=[-1.76520810e-07,-2.26677449e-05,-1.93885521e-04] (from blazecalc.py), t_grating=4.3769007")

# plt.plot(px_centre, Trans_p[alt_index, :], label="%0.2fkm" %chosen_alt)
ax1a.plot(px_centre, y_cr, "k--", label="Science/Y")
ax1a.plot(px_centre, Trans_p[alt_index, :], label="%0.2fkm" %chosen_alt)
nu, asc = np.loadtxt(r"C:\Users\iant\Documents\DOCUMENTS\presentations\20180626_142634_1p0a_SO_A_E_168_94_test_fwd_SP1_FEN1_rad_forw.dat", dtype=float, comments="%", usecols=(0,1,), unpack=True)
ax1a.plot(nu, asc, label="ASIMUT")

psc = np.loadtxt(r"C:\Users\iant\Documents\DOCUMENTS\presentations\psg_aotfC.txt", dtype=float, comments="#", usecols=(0,1,2,), unpack=True).T
psc[:,1] = psc[:,1]/psc[:,2]

df = np.abs((psc[0,0] - psc[-1,0])/(len(psc[:,0])-1))
f0 = (psc[0,0] + psc[-1,0])/2.0
sigma = df*((f0/df)/17000.0)/2.355
gx = np.arange(-3*sigma, 3*sigma, df)
ker = np.exp(-(gx/sigma)**2/2)
ker = ker/np.sum(ker)
pps = np.convolve(psc[:,1], ker, mode='same')
ppp = np.interp(nu, np.flip(psc[:,0]), np.flip(pps))
xpix  = range(len(ppp))
dfp = np.polyval([-2.4333526e-06,0.0018259633,-0.031606901],xpix)*(f0/3700.0)
ghost = np.interp(nu+dfp,nu,ppp)
psp = (ppp + 0.27*ghost)/1.27

ax1a.plot(nu, psp, label="PSG")



# psc2 = np.loadtxt(r"C:\Users\iant\Documents\DOCUMENTS\presentations\20180626_142634_1p0a_SO_A_E_168_94_test_fwd_SP1_FEN1_rad_aotfC.dat", dtype=float, comments="%", usecols=(0,1,2,), unpack=True).T
# psc2[:,1] = psc2[:,1]/psc2[:,2]

# plt.plot(psc[:,0], psc[:,1])
# plt.plot(psc2[:,0], psc2[:,1])

# df = np.abs((psc[0,0] - psc[-1,0])/(len(psc[:,0])-1))
# f0 = (psc[0,0] + psc[-1,0])/2.0
# sigma = df*((f0/df)/17000.0)/2.355
# gx = np.arange(-3*sigma, 3*sigma, df)
# ker = np.exp(-(gx/sigma)**2/2)
# ker = ker/np.sum(ker)
# pps = np.convolve(psc[:,1], ker, mode='same')
# ppp = np.interp(nu, np.flip(psc[:,0]), np.flip(pps))
# xpix  = range(len(ppp))
# dfp = np.polyval([-2.4333526e-06,0.0018259633,-0.031606901],xpix)*(f0/3700.0)
# ghost = np.interp(nu+dfp,nu,ppp)
# psp = (ppp + 0.27*ghost)/1.27
# ax1a.plot(nu, psp, label="ASIMUT w/ SO calc")


ax1b.set_title("Residual PSG - ASIMUT")
ax1b.plot(nu, psp-asc)

px_centre_interp = np.interp(nu, px_centre, Trans_p[alt_index, :])
ax1c.set_title("Residual PSG - Python")
ax1c.plot(nu, psp-px_centre_interp)

ax1d.set_title("Residual ASIMUT - Python")
ax1d.plot(nu, asc-px_centre_interp)

# plt.plot(pixels, Trans_p[alt_index, :], label="%0.2fkm, tcoeff=[-0.736363,-6.363908] (from aotf.py), t_grating=4.3769007" %chosen_alt)
# # plt.plot(pixels, Trans_p[alt_index, :], label="%0.2fkm, ncoeff=[-1.76520810e-07,-2.26677449e-05,-1.93885521e-04] (from blazecalc.py), t_grating=4.3769007" %chosen_alt)
# plt.plot(pixels, y_cr, label="Science/Y")
# nu, T = np.loadtxt(r"C:\Users\iant\Dropbox\NOMAD\Python\reference_files\aotf\20180626_142634_1p0a_SO_A_E_168_94_test_fwd_SP1_FEN1_rad_forw.dat", dtype=float, comments="%", usecols=(0,1,), unpack=True)
# plt.plot(pixels, T, label="ASIMUT")

ax1d.set_xlabel("Wavenumber cm-1")
ax1a.grid()
ax1b.grid()
ax1c.grid()
ax1d.grid()
ax1a.legend(loc="lower right")
fig.savefig("asimut_psg_py_comparison_old.png")
# fig.savefig("asimut_psg_py_comparison_new.png")

# a = np.arange(10.0)
# b = np.array([0.5, 5.5, 7.5])

# c = np.searchsorted(a, b)


# d = combine_groups([a,a], [b,b])
plt.figure(figsize=(12,8), constrained_layout=True)


groups = []
for m in orders:
    # delta_nu = F_blaze_centre(order) - F_blaze_centre(m) #this is too simplistic, doesn't work
    px_m = nu_mp2(m, pixels, t_grating)
    indices_m = np.where((nu_hr > px_m[0]) & (nu_hr < px_m[-1]))[0]

    px_delta = px_centre - px_m
    
    #interpolate the deltas onto the hr grid
    hr_delta = np.interp(nu_hr[indices_m], px_m, px_delta)

    plt.plot(nu_hr[indices_m] + hr_delta, I_hr[alt_index, indices_m], label="Order %i" %m)
    
    groups.append([nu_hr[indices_m] + hr_delta, I_hr_norm[indices_m]])

plt.legend(loc="upper right")

combined_arrays = [np.concatenate([grp[idx] for grp in groups]) for idx in range(len(groups[0]))]
sort_indices = np.argsort(combined_arrays[0], kind="mergesort")
out = [arr[sort_indices] for arr in combined_arrays]
   
plt.figure(figsize=(12,8), constrained_layout=True)
plt.plot(out[0], out[1], label="Order addition")

nu, F, Fdiv = np.loadtxt(r"C:\Users\iant\Documents\DOCUMENTS\presentations\20180626_142634_1p0a_SO_A_E_168_94_test_fwd_SP1_FEN1_rad_aotfC.dat", dtype=float, comments="%", usecols=(0,1,2,), unpack=True)
plt.plot(nu, F/Fdiv, label="ASIMUT")
plt.xlim([3793, 3799])
plt.legend(loc="upper right")
# nu, F = np.loadtxt(r"C:\Users\iant\Documents\DOCUMENTS\presentations\20180626_142634_1p0a_SO_A_E_168_94_test_fwd_SP1_FEN1_rad_forw.dat", dtype=float, comments="%", usecols=(0,1,), unpack=True)
# plt.figure(figsize=(12,8), constrained_layout=True)
# plt.plot(nu, F)


# plt.figure()
# plt.title("HR AOTF and order addition")
# plt.plot(nu_sp + px_centre[p], ils)
# plt.plot(out[0], out[1])
# plt.xlim([px_centre[p]-1.5, px_centre[p]+1.5])
# np.convolve()


# psc = np.asarray([out[0], out[1]]).T

# df = -(psc[0,0] - psc[-1,0])/(len(psc[:,0])-1)
# f0 = (psc[0,0] + psc[-1,0])/2.0
# sigma = df*((f0/df)/17000.0)/2.355
# gx = np.arange(-3*sigma, 3*sigma, df)
# ker = np.exp(-(gx/sigma)**2/2)
# ker = ker/np.sum(ker)
# pps = np.convolve(psc[:,1], ker, mode='same')
# ppp = np.interp(asp[:,0], np.flip(psc[:,0]), np.flip(pps))
# xpix  = range(len(ppp))
# dfp = np.polyval([-2.4333526e-06,0.0018259633,-0.031606901],xpix)*(f0/3700.0)
# ghost = np.interp(asp[:,0]+dfp,asp[:,0],ppp)
# psp = (ppp + 0.27*ghost)/1.27
