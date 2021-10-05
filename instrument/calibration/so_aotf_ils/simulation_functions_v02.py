# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:41:48 2021

@author: iant

MINISCAN FITTING FUNCTIONS
"""


import numpy as np
import os
import sys
# import re
#from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
from scipy.signal import savgol_filter
# import scipy.signal as ss
import lmfit

import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist
# from tools.file.read_write_hdf5 import write_hdf5_from_dict, read_hdf5_to_dict
from tools.file.paths import paths
from tools.file.get_hdf5_temperatures import get_interpolated_temperatures


from tools.spectra.solar_spectrum import get_solar_hr
from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
# from tools.spectra.fit_polynomial import fit_polynomial

from tools.general.get_nearest_index import get_nearest_index

from instrument.nomad_so_instrument import t_nu_mp
# from instrument.nomad_so_instrument import F_blaze_goddard21, F_aotf_goddard21

# from instrument.nomad_so_instrument import m_aotf as m_aotf_so

from instrument.calibration.so_aotf_ils.simulation_config import AOTF_FROM_FILE, sim_parameters




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



def get_cal_params(d, tempg, tempa, pc={}):
    #from blazecalc.py 17/9/21
    
    if np.mod(d["i"], 100) == 0:
        print("Calculating Parameters %i" %d["i"])
    d["i"] += 1
        
    aotf = d["A"]
    nu_hr = d["nu_hr"]
    orders = d["orders"]

    #1st september slack
    # AOTF shape parameters
    # aotfwc  = [-1.78088527e-07,  9.44266907e-04,  1.95991162e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    # aotfsc  = [ 1.29304371e-06, -6.77032965e-03,  1.03141366e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
    # aotfac  = [-1.96949242e-07,  1.48847262e-03, -1.40522510e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    # aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
    # aotfgc  = [ 1.07865793e-07, -7.20862528e-04,  1.24871556e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]
    
    # # Calibration coefficients
    # cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])         # Frequency of AOTF [cm-1 from kHz]
    # cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])   # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    # ncoeff  = [-1.76520810e-07, -2.26677449e-05, -1.93885521e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
    # blazep  = [-5.76161e-14,-2.01122e-10,2.02312e-06,2.25875e+01] # Dependence of blazew from AOTF frequency
    # aotfts  = -6.5278e-5                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]


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


    
    # Calculate blaze parameters
    if "aotff" in pc.keys():
        aotff = pc["aotff"]
    else:
        aotff = np.polyval(cfaotf, aotf) + tempa*aotfts  # AOTF frequency [cm-1], temperature corrected
    # order = round(aotff/blazew)                      # Grating order
    
    d["aotff"] = aotff
    
    # Compute AOTF parameters. Note that changes in aotff are not passed to other AOTF parameters. 
    if "aotfw" in pc.keys():
        d["aotfw"] = pc["aotfw"]
    else:
        d["aotfw"] = np.polyval(aotfwc,aotff)

    if "aotfs" in pc.keys():
        d["aotfs"] = pc["aotfs"]
    else:
        d["aotfs"] = np.polyval(aotfsc,aotff)

    if "aotfa" in pc.keys():
        d["aotfa"] = pc["aotfa"]
    else:
        d["aotfa"] = np.polyval(aotfac,aotff)

    if "aotfo" in pc.keys():
        d["aotfo"] = pc["aotfo"]
    else:
        d["aotfo"] = np.polyval(aotfoc,aotff)

    if "aotfg" in pc.keys():
        d["aotfg"] = pc["aotfg"]
    else:
        d["aotfg"] = np.polyval(aotfgc,aotff)

    if "aotfgw" in pc.keys():
        d["aotfgw"] = pc["aotfgw"]
    else:
        d["aotfgw"] = 50. #offset width cm-1

    dx = nu_hr - aotff
    d["F_aotf"] = F_aotf3(dx, d)

    
    if "blaze_shift" in pc.keys():
        d["blaze_shift"] = pc["blaze_shift"]
    else:
        d["blaze_shift"] = 0.0
        
        
    #TODO: update this with different blaze values for each order
    if ("blaze_shift" in pc.keys()) or (d["i"]==1):
        print("Recalculating blaze")
        # blazew =  np.polyval(blazep,aotf-22000.0)        # FSR (Free Spectral Range), blaze width [cm-1]
        blazew =  np.polyval(blazep,d["line"]-3700.0)        # FSR (Free Spectral Range), blaze width [cm-1]
        blazew += blazew*np.polyval(ncoeff,tempg)        # FSR, corrected for temperature
        d["blazew"] = blazew


        # Frequency of the pixels
        for order in orders:
            d[order] = {}
            pixf = np.polyval(cfpixel,range(320))*order
            pixf += pixf*np.polyval(ncoeff, tempg) + d["blaze_shift"]
            blazef = order*blazew                        # Center of the blaze
            d[order]["pixf"] = pixf
            d[order]["blazef"] = blazef
            # print(order, blazef, blazew)
            
            blaze = F_blaze3(pixf, blazef, blazew)
            d[order]["F_blaze"] = blaze
        
    
        F_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan
        nu_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan
    
        for i, order in enumerate(d["orders"]):
        
            F_blaze = list(d[order]["F_blaze"])
            F_blazes[i*321:(i+1)*320+i] = F_blaze
            nu_blazes[i*321:(i+1)*320+i] = d[order]["pixf"]
            
        d["F_blazes"] = F_blazes
        d["nu_blazes"] = nu_blazes
        
        d = get_ils_params(d)
        d = blaze_conv(d)
    
    return d




def get_ils_params(d):
    """get ils params, add to dictionary"""
    
    aotff = d["aotff"]
    pixels = d["pixels"]

    #from ils.py on 6/7/21
    amp = 0.2724371566666666 #intensity of 2nd gaussian
    rp = 16939.80090831571 #resolving power cm-1/dcm-1
    disp_3700 = [-3.06665339e-06,  1.71638815e-03,  1.31671485e-03] #displacement of 2nd gaussian cm-1 w.r.t. 3700cm-1 vs pixel number
    
    A_w_nu0 = aotff / rp
    sconv = A_w_nu0/2.355
    
    
    disp_3700_nu = np.polyval(disp_3700, pixels) #displacement at 3700cm-1
    disp_order = disp_3700_nu / -3700.0 * aotff #displacement adjusted for wavenumber
    
    d["ils"] = {"width":np.tile(sconv, len(pixels)), "displacement":disp_order, "amplitude":np.tile(amp, len(pixels))}
    
    return d
        



def blaze_conv(d):
    #make blaze convolution function for each pixel
    
    nu_hr = d["nu_hr"]
    pixels = d["pixels"]

    W_conv = np.zeros((len(pixels), len(nu_hr)))
    
    for iord in d["orders"]:
        nu_p = d[iord]["pixf"]
        W_blaze = d[iord]["F_blaze"]
        
        # print('order %d: %.1f to %.1f' % (iord, nu_p[0], nu_p[-1]))
        
        for ip in pixels:
            inu1 = np.searchsorted(nu_hr, nu_p[ip] - 0.5) #start index
            inu2 = np.searchsorted(nu_hr, nu_p[ip] + 0.5) #end index
            
            nu_sp = nu_hr[inu1:inu2] - nu_p[ip]
            
            #make ils shape
            a1 = 0.0
            a2 = d["ils"]["width"][ip]
            a3 = 1.0
            a4 = d["ils"]["displacement"][ip]
            a5 = d["ils"]["width"][ip]
            a6 = d["ils"]["amplitude"][ip]
                
            ils0=a3 * np.exp(-0.5 * ((nu_sp + a1) / a2) ** 2)
            ils1=a6 * np.exp(-0.5 * ((nu_sp + a4) / a5) ** 2)
            ils = ils0 + ils1 
    
        
            W_conv[ip,inu1:inu2] += (W_blaze[ip]) * ils
            # W_conv[ip,inu1:inu2] += (W_blaze[ip] * dnu)/(np.sqrt(2.0 * np.pi) * sconv) * np.exp(-(nu_hr[inu1:inu2] - nu_p[ip])**2 / (2. *sconv**2))
            # if ip == 319:
            #     plt.plot(nu_sp, ils + iord/1000.)
            #     plt.plot(nu_sp, (W_blaze[ip] * dnu)/(np.sqrt(2.0 * np.pi) * sconv) * np.exp(-(nu_hr[inu1:inu2] - nu_p[ip])**2 / (2. *sconv**2)) + iord/1000.)
    
    d["W_conv"] = W_conv
    
    return d





def get_file(regex, file_level="hdf5_level_0p2a"):
    #get first file matching regex
    
    hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)
    hdf5_file = hdf5Files[0]
    hdf5_filename = hdf5Filenames[0]
    print(hdf5_filename)
    return hdf5_file, hdf5_filename




def get_data_from_file(hdf5_file, hdf5_filename, d):

    channel = hdf5_filename.split("_")[3].lower()
    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]

    detector_data_all = hdf5_file["Science/Y"][...]
    detector_centre_data = detector_data_all[:, [9,10,11,15], :] #chosen to avoid bad pixels
    spectra = np.mean(detector_centre_data, axis=1)

    d["hdf5_filename"] = hdf5_filename
    d["channel"] = channel
    d["aotf_freqs"] = aotf_freq
    d["spectra"] = spectra
    
    d["pixels"] = sim_parameters[d["line"]]["pixels"]
    d["centre_order"] = sim_parameters[d["line"]]["centre_order"]
    d["order_range"] = sim_parameters[d["line"]]["order_range"]
    d["orders"] = list(range(d["order_range"][0], d["order_range"][1]+1))
    
    d["i"] = 0
    
    return d




def select_data(d, index):
    
    d["index"] = index
    d["A"] = d["aotf_freqs"][index]
    d["spectrum"] = d["spectra"][index]
    d["spectrum_norm"] = d["spectrum"]/np.max(d["spectrum"])
    temperature = d["temperatures"][index]
    d["temperature"] = temperature

    # if AOTF_FROM_FILE:
    #     d = load_aotf_from_file(d)

    
    return d






def get_all_x(hdf5_file, d):
    #make wavenumber grid from temperature
    
    if "InterpolatedTemperature" in hdf5_file["Channel"].keys():
        print("Getting temperatures directly from file")
        temperatures = hdf5_file["Channel/InterpolatedTemperature"][...]
        
    # else:
    #     temperatures = get_interpolated_temperatures(hdf5_file, "so")

    pixels = d["pixels"]
    order = d["centre_order"]
    
    x_array = np.zeros([len(temperatures), len(pixels)])
    
    #slack 29th August 2021
    cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])   # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    # ncoeff  = [-1.76520810e-07, -2.26677449e-05, -1.93885521e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
    #slack 2nd october
    ncoeff  = [-2.44383699e-07, -2.30708836e-05, -1.90001923e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]

    for i, t in enumerate(temperatures):
        xdat  = np.polyval(cfpixel, pixels) * order
        xdat += xdat * np.polyval(ncoeff, t)
    
        x_array[i, :] = xdat
        
    d["temperatures"] = temperatures
    d["x_all"] = x_array 
    return d







def get_solar_spectrum(d, plot=False):
    
    nu_hr = np.arange(sim_parameters[d["line"]]["nu_range"][0], sim_parameters[d["line"]]["nu_range"][1], sim_parameters[d["line"]]["d_nu"])
    d["nu_hr"] = nu_hr
    
    solar_spectrum_filename = sim_parameters[d["line"]]["solar_spectra"][d["solar_spectrum"]]
    
    ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], solar_spectrum_filename)
    I0_solar_hr = get_solar_hr(d["nu_hr"], solspec_filepath=ss_file)

    #scale ACE to same level as PFS solar spectrum
    if d["solar_spectrum"] == "ACE":
        I0_solar_hr *= (4.266009 / 1.4507183e-006)

    d["I0_solar_hr"] = I0_solar_hr

    #pre-convolute solar spectrum to approximate level - only for fitting temperature and plotting
    I0_lr = savgol_filter(I0_solar_hr, sim_parameters[d["line"]]["filter_smoothing"], 1)
    d["I0_lr"] = I0_lr
    # # I0_cont = fit_polynomial(nu_hr, I0_low_res, degree=2)
    # # I0_cr = I0_low_res / I0_cont
    # # I0_low_res = I0_low_res/np.max(I0_low_res)
    # # plt.plot(nu_hr, I0_cr, label="Convolved solar line")

    
    
    
    
    
    """remove the chosen solar line from the solar spectrum"""
    sl_extent = [
        np.min(np.where(d["nu_hr"] > sim_parameters[d["line"]]["solar_line_nu_range"][0])), 
        np.max(np.where(d["nu_hr"] < sim_parameters[d["line"]]["solar_line_nu_range"][1]))
    ]
    d["sl_extent"] = sl_extent
    
    sl_indices = np.arange(sl_extent[0], sl_extent[1]+1)
    d["sl_indices"] = sl_indices
    
    
    # sl_flat = np.polyval(np.polyfit(sl_extent, [I0_lr[sl_extent[0]], I0_lr[sl_extent[1]]], 1), sl_indices)
    sl_flat = np.polyval(np.polyfit(sl_extent, [I0_solar_hr[sl_extent[0]], I0_solar_hr[sl_extent[1]]], 1), sl_indices)
    
    # I0_lr_slr = np.copy(I0_lr)
    I0_solar_slr = np.copy(I0_solar_hr)

    # I0_lr_slr[sl_indices] = sl_flat
    I0_solar_slr[sl_indices] = sl_flat
    
    # d["I0_lr_slr"] = I0_lr_slr
    d["I0_solar_slr"] = I0_solar_slr #slr = solar line removed


    # if plot:
    #     plt.figure()
    #     plt.plot(nu_hr, I0_lr)
    #     plt.plot(nu_hr, I0_lr_slr)
    #     sys.exit()

    return d




def get_absorption_line_indices(d):
    #get indices of all spectra where absorption line is present
    abs_indices = np.where(
        (d["aotf_freqs"] > sim_parameters[d["line"]]["solar_line_aotf_range"][0]) & 
        (d["aotf_freqs"] < sim_parameters[d["line"]]["solar_line_aotf_range"][1]))[0]

    return abs_indices



def find_absorption_minimum(d):
    index = d["index"]
    
    abs_indices = get_absorption_line_indices(d)
    
    #find index of spectrum w/ absorption closest to desired index
    absorption_line_fit_index = abs_indices[get_nearest_index(index, abs_indices)]
    
    # aotf_freqs = d["aotf_freqs"]
    spectra = d["spectra"]
    #remove continuum from miniscan spectrum to fit solar line minimum
    spectrum_w_absorption = spectra[absorption_line_fit_index]
    spectrum_cont = baseline_als(spectrum_w_absorption)
    spectrum_cr = spectrum_w_absorption[50:]/spectrum_cont[50:]
    smi = np.argmin(spectrum_cr) + 50 #spectrum min index
    
    d["spectrum_cr"] = spectrum_cr
    d["absorption_pixel"] = smi
    return d





def fit_temperature(d, hdf5_file, plot=False):
    """code to check absorption lines in solar spectrum"""

    d = find_absorption_minimum(d)

    d = get_all_x(hdf5_file, d)
    smi = d["absorption_pixel"]
    spectrum_cr = d["spectrum_cr"]
    
    # channel = d["channel"]
    nu_hr = d["nu_hr"]
    # temperature = spectrum_temperature(hdf5_file, channel, index)
    temperature = d["temperature"]
    
    order = d["centre_order"]
    
    
    """code to shift spectral cal to match absorption"""
    
    # pixels_nu = nu_mp(order, sim_parameters[d["line"]]["pixels"], temperature)
    pixels_nu = d["x_all"][d["index"], :]
    
    if np.abs(pixels_nu[smi]-d["line"]) > 2.0: #if not fitting to desired solar line
        # print("Warning: solar line at %0.2f and fitting to line at %0.2f" %(d["line"], pixels_nu[smi]))
        approx_smi = get_nearest_index(d["line"], pixels_nu) #320 px
        approx_smi_range = np.arange(max([0, approx_smi-10]), min([319, approx_smi+10]), 1)
        smi = np.argmin(spectrum_cr[approx_smi_range-50]) + approx_smi_range[0]
        # print("Desired solar line is at %0.2f and now fitting to line at %0.2f" %(d["line"], pixels_nu[smi]))

    x_hr, y_hr, min_position_nu = fit_gaussian_absorption(pixels_nu[smi-3:smi+4], spectrum_cr[smi-53:smi-46])
    absorption_depth = np.min(y_hr)
        
    
    
    #find nu of solar band
    ami = np.argmin(d["I0_lr"]) #absorption_min_index
    absorption_nu = nu_hr[ami] #near enough on convolved HR grid
    if np.abs(absorption_nu - d["line"]) > 2.0: #if not fitting to desired solar line
        # print("Warning: desired solar line at %0.2f and fitting to solar line at %0.2f" %(d["line"], absorption_nu))
        approx_ami = get_nearest_index(d["line"], nu_hr)
        approx_ami_range = np.arange(approx_ami-1000, approx_ami+1000, 1)
        ami = np.argmin(d["I0_lr"][approx_ami_range]) + approx_ami_range[0]
        absorption_nu = nu_hr[ami] #near enough on convolved HR grid
        # print("Desired solar line is at %0.2f and now fitting to solar line at %0.2f" %(d["line"], absorption_nu))
        
    
   
    if plot:
        plt.figure()
        plt.plot(pixels_nu[50:], spectrum_cr, label="Temperature spectral calibration")
        plt.plot(x_hr, y_hr, linestyle="--", label="Fit to miniscan absorption")
        sys.exit()
    
    
    
    delta_nu = absorption_nu - min_position_nu
    t_calc = t_nu_mp(order, absorption_nu, smi) #this is wrong now
    print("temperature=", temperature, "t_calc=", t_calc, "delta_nu=", delta_nu)
    # delta_t = temperature - t_calc
    # print("delta_t=", delta_t)
    
    d["absorption_depth"] = absorption_depth
    
    d["t_calc"] = t_calc
    # d["nu_hr"] = nu_hr
    
    
    
    return d





#for calculating area under the curves
def trapezium_area(y1, y2, dx=1.0):
    return 0.5 * (y1 + y2) * dx
    

def area_under_curve(curve1, curve2, dx=1.0):
    area = 0
    for i in range(len(curve1)-1):
        y1 = curve1[i] - curve2[i]
        y2 = curve1[i+1] - curve2[i+1]
    
        area += trapezium_area(y1, y2, dx=dx)
    return area







def make_param_dict(d, chosen_params):
    """choose starting value and range for each fitting parameter"""
    
    #best, min, max
    param_dict = {}
    if "aotff" in chosen_params:
        param_dict["aotff"] = [d["aotff"], d["aotff"]-5.0, d["aotff"]+5.0]
    if "aotfw" in chosen_params:
        param_dict["aotfw"] = [d["aotfw"], d["aotfw"]-1.0, d["aotfw"]+1.0]
    if "aotfs" in chosen_params:
        param_dict["aotfs"] = [d["aotfs"], d["aotfs"]-3.0, d["aotfs"]+3.0]
    if "aotfa" in chosen_params:
        param_dict["aotfa"] = [d["aotfa"], 0.001, d["aotfa"]*3.0]
    if "aotfo" in chosen_params:
        param_dict["aotfo"] = [d["aotfo"], d["aotfo"]-1.0, d["aotfo"]+1.0]
    if "aotfg" in chosen_params:
        param_dict["aotfg"] = [d["aotfg"], 0.0, d["aotfg"]*2.0]
    if "aotfgw" in chosen_params:
        param_dict["aotfgw"] = [d["aotfgw"], d["aotfgw"]-40.0, d["aotfgw"]+40.0]
        
    if "blaze_shift" in chosen_params:
        param_dict["blaze_shift"] = [d["blaze_shift"], d["blaze_shift"]-1.0, d["blaze_shift"]+1.0]
    
    return param_dict
    




def calc_spectrum(d, I0_solar_hr):
    """make simulated spectrum"""
    
    W_aotf = d["F_aotf"]
    W_conv = d["W_conv"]
    
    I0_hr = I0_solar_hr * W_aotf
    I0_p = np.matmul(W_conv, I0_hr)  # np x 1

    return I0_p
    
 


def fit_resid(params, d):
    """define the fit residual"""
    
    t = d["temperature"]
    
    variables = {}
    for key in params.keys():
        variables[key] = params[key].value
        
    d = get_cal_params(d, t, t, pc=variables)
        
    I0_p = calc_spectrum(d, d["I0_solar_hr"])

    return (I0_p/max(I0_p) - d["spectrum_norm"]) / d["sigma"]





def fit_spectrum(param_dict, d):
    # print("Fitting AOTF and blaze")
    print("Starting parameters:")
    params = lmfit.Parameters()
    for key, value in param_dict.items():
       params.add(key, d[key], min=value[1], max=value[2])
       print(key, d[key])

    lm_min = lmfit.minimize(fit_resid, params, args=(d,), method='leastsq')
    d["chisq"] = lm_min.chisqr
    # print("chisq=", chisq)

    print("Fitted parameters:")
    for key in params.keys():
        d[key] = lm_min.params[key].value
        print(key, d[key])

    return d



    
    