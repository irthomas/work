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
from tools.file.read_write_hdf5 import write_hdf5_from_dict, read_hdf5_to_dict
from tools.file.paths import paths

from tools.sql.get_sql_spectrum_temperature import get_sql_temperatures_all_spectra

from tools.spectra.solar_spectrum import get_solar_hr
from tools.spectra.baseline_als import baseline_als
from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
# from tools.spectra.fit_polynomial import fit_polynomial

from tools.general.get_nearest_index import get_nearest_index

from instrument.nomad_so_instrument import nu_mp, spec_res_order, t_nu_mp
from instrument.nomad_so_instrument import F_blaze_goddard21, F_aotf_goddard21

from instrument.nomad_so_instrument import m_aotf as m_aotf_so

from instrument.calibration.so_aotf_ils.simulation_config import AOTF_OFFSET_SHAPE, BLAZE_WIDTH_FIT, sim_parameters





def get_file(regex, file_level="hdf5_level_0p2a"):

    hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)
    hdf5_file = hdf5Files[0]
    hdf5_filename = hdf5Filenames[0]
    print(hdf5_filename)
    return hdf5_file, hdf5_filename




def get_data_from_file(hdf5_file, hdf5_filename):

    channel = hdf5_filename.split("_")[3].lower()
    aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]

    detector_data_all = hdf5_file["Science/Y"][...]
    detector_centre_data = detector_data_all[:, [9,10,11,15], :] #chosen to avoid bad pixels
    spectra = np.mean(detector_centre_data, axis=1)
    
    return {"hdf5_filename":hdf5_filename, "channel":channel, "aotf_freqs":aotf_freq, "spectra":spectra}




def select_data(d, index):
    
    d["index"] = index
    d["A"] = d["aotf_freqs"][index]
    d["spectrum"] = d["spectra"][index]
    d["spectrum_norm"] = d["spectrum"]/np.max(d["spectrum"])
    # d["centre_order"] = m_aotf_so(d["A"])
    d["centre_order"] = sim_parameters[d["line"]]["centre_order"]
    d["pixels"] = sim_parameters[d["line"]]["pixels"]
    d["m_range"] = sim_parameters[d["line"]]["order_range"]

    
    return d






def spectrum_temperature(hdf5_file, channel, index):
    """get temperature of a specific spectrum (by index)"""
    
    temperatures = get_sql_temperatures_all_spectra(hdf5_file, channel)
    # temperature = np.mean(temperatures)
    temperature = temperatures[index]
    return temperature

    


def get_solar_spectrum(d, plot=False):
    
    nu_hr = np.arange(sim_parameters[d["line"]]["nu_range"][0], sim_parameters[d["line"]]["nu_range"][1], sim_parameters[d["line"]]["d_nu"])
    d["nu_hr"] = nu_hr
    
    solar_spectrum_filename = sim_parameters[d["line"]]["solar_spectra"][d["solar_spectrum"]]
    
    ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], solar_spectrum_filename)
    I0_solar_hr = get_solar_hr(d["nu_hr"], solspec_filepath=ss_file)
    I0_lr = savgol_filter(I0_solar_hr, sim_parameters[d["line"]]["filter_smoothing"], 1)
    # I0_cont = fit_polynomial(nu_hr, I0_low_res, degree=2)
    # I0_cr = I0_low_res / I0_cont
    # I0_low_res = I0_low_res/np.max(I0_low_res)
    # plt.plot(nu_hr, I0_cr, label="Convolved solar line")
    
    
    d["I0_solar_hr"] = I0_solar_hr
    
    #pre-convolute solar spectrum to approximate level
    d["I0_lr"] = I0_lr
    
    I0_lr_slr = np.copy(I0_lr)
    sl_extent = [
        np.min(np.where(d["nu_hr"] > sim_parameters[d["line"]]["solar_line_nu_range"][0])), 
        np.max(np.where(d["nu_hr"] < sim_parameters[d["line"]]["solar_line_nu_range"][1]))
    ]
    d["sl_extent"] = sl_extent
    
    sl_indices = np.arange(sl_extent[0], sl_extent[1]+1)
    d["sl_indices"] = sl_indices
    
    
    sl_flat = np.polyval(np.polyfit(sl_extent, [I0_lr[sl_extent[0]], I0_lr[sl_extent[1]]], 1), sl_indices)
    
    I0_lr_slr[sl_indices] = sl_flat
    
    d["I0_lr_slr"] = I0_lr_slr

    if plot:
        plt.figure()
        plt.plot(nu_hr, I0_lr)
        plt.plot(nu_hr, I0_lr_slr)
        sys.exit()

    return d


def fit_temperature(d, hdf5_file, plot=False):
    """code to check absorption lines in solar spectrum"""

    index = d["index"]
    
    #get indices of all spectra where absorption line is present
    abs_indices = np.where(
        (d["aotf_freqs"] > sim_parameters[d["line"]]["solar_line_aotf_range"][0]) & 
        (d["aotf_freqs"] < sim_parameters[d["line"]]["solar_line_aotf_range"][1]))[0]
    #find index of spectrum w/ absorption closest to desired index
    absorption_line_fit_index = abs_indices[get_nearest_index(index, abs_indices)]
    
    # aotf_freqs = d["aotf_freqs"]
    spectra = d["spectra"]
    channel = d["channel"]
    nu_hr = d["nu_hr"]
    temperature = spectrum_temperature(hdf5_file, channel, index)
    
    
    #TODO: fix this
    # order = m_aotf_so(aotf_freqs[absorption_line_fit_index])

    order = d["centre_order"]
    
    
    """code to shift spectral cal to match absorption"""
    #remove continuum from miniscan spectrum to fit solar line minimum
    spectrum_w_absorption = spectra[absorption_line_fit_index]
    spectrum_cont = baseline_als(spectrum_w_absorption)
    spectrum_cr = spectrum_w_absorption[50:]/spectrum_cont[50:]
    smi = np.argmin(spectrum_cr) + 50 #spectrum min index
    pixels_nu = nu_mp(order, sim_parameters[d["line"]]["pixels"], temperature)
    
    if np.abs(pixels_nu[smi]-d["line"]) > 2.0: #if not fitting to desired solar line
        print("Warning: solar line at %0.2f and fitting to line at %0.2f" %(d["line"], pixels_nu[smi]))
        approx_smi = get_nearest_index(d["line"], pixels_nu) #320 px
        approx_smi_range = np.arange(max([0, approx_smi-10]), min([319, approx_smi+10]), 1)
        smi = np.argmin(spectrum_cr[approx_smi_range-50]) + approx_smi_range[0]
        print("Desired solar line is at %0.2f and now fitting to line at %0.2f" %(d["line"], pixels_nu[smi]))

    x_hr, y_hr, min_position_nu = fit_gaussian_absorption(pixels_nu[smi-3:smi+4], spectrum_cr[smi-53:smi-46])
    absorption_depth = np.min(y_hr)
        
    
    
    #find nu of solar band
    ami = np.argmin(d["I0_lr"]) #absorption_min_index
    absorption_nu = nu_hr[ami] #near enough on convolved HR grid
    if np.abs(absorption_nu - d["line"]) > 2.0: #if not fitting to desired solar line
        print("Warning: desired solar line at %0.2f and fitting to solar line at %0.2f" %(d["line"], absorption_nu))
        approx_ami = get_nearest_index(d["line"], nu_hr)
        approx_ami_range = np.arange(approx_ami-1000, approx_ami+1000, 1)
        ami = np.argmin(d["I0_lr"][approx_ami_range]) + approx_ami_range[0]
        absorption_nu = nu_hr[ami] #near enough on convolved HR grid
        print("Desired solar line is at %0.2f and now fitting to solar line at %0.2f" %(d["line"], absorption_nu))
        
    
   
    if plot:
        plt.figure()
        plt.plot(pixels_nu[50:], spectrum_cr, label="Temperature spectral calibration")
        plt.plot(x_hr, y_hr, linestyle="--", label="Fit to miniscan absorption")
        sys.exit()
    
    
    
    delta_nu = absorption_nu - min_position_nu
    print("delta_nu=", delta_nu)
    t_calc = t_nu_mp(order, absorption_nu, smi)
    delta_t = temperature - t_calc
    print("delta_t=", delta_t)
    
    d["absorption_depth"] = absorption_depth
    d["absorption_pixel"] = smi
    d["temperature"] = t_calc
    d["nu_hr"] = nu_hr
    d["t0"] = temperature
    
    
    
    return d






def trapezium_area(y1, y2, dx=1.0):
    return 0.5 * (y1 + y2) * dx
    

def area_under_curve(curve1, curve2, dx=1.0):
    area = 0
    for i in range(len(curve1)-1):
        y1 = curve1[i] - curve2[i]
        y2 = curve1[i+1] - curve2[i+1]
    
        area += trapezium_area(y1, y2, dx=dx)
    return area




def get_spec_res(d):
    
    c_order = int(np.mean(sim_parameters[d["line"]]["order_range"]))
    spec_res = spec_res_order(c_order)
    
    return spec_res


def calc_blaze(d):

    spec_res = get_spec_res(d)
    
    nu_hr = d["nu_hr"]
    hdf5_filename = d["hdf5_filename"]
    temperature = d["temperature"]
    
    #if convolution already saved to file
    h5_conv_filename = "conv_%s_order%i-%i_dnu%f_temp%.2f" %(hdf5_filename, sim_parameters[d["line"]]["order_range"][0], sim_parameters[d["line"]]["order"][1], sim_parameters[d["line"]]["d_nu"], temperature)
    if os.path.exists(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename+".h5")):
        print("Reading W_conv from existing file")
        W_conv = read_hdf5_to_dict(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename))[0]["W_conv"]
    
        
    else:
        print("Making file", h5_conv_filename)
        Nbnu_hr = len(nu_hr)
        NbP = len(sim_parameters[d["line"]]["pixels"])
        
        #old and new blaze functions are functional identical - use 2021 function only
        sconv = spec_res/2.355
        W_conv = np.zeros((NbP,Nbnu_hr))
        for iord in range(sim_parameters[d["line"]]["order_range"][0], sim_parameters[d["line"]]["order_range"][1]+1):
            print("Blaze order %i" %iord)
            nu_pm = nu_mp(iord, sim_parameters[d["line"]]["pixels"], temperature)
            W_blaze = F_blaze_goddard21(iord, sim_parameters[d["line"]]["pixels"], temperature)
            for ip in sim_parameters[d["line"]]["pixels"]:
                W_conv[ip,:] += (W_blaze[ip]*sim_parameters[d["line"]]["d_nu"])/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
                
        W_conv[W_conv < 1.0e-5] = 0.0 #remove small numbers
                
        write_hdf5_from_dict(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename), {"W_conv":W_conv}, {}, {}, {})
    
    d["W_conv"] = W_conv
    return d



def aotf_conv(d, variables):
    W_aotf = F_aotf_goddard21(0., d["nu_hr"], d["temperature"], 
                              A=d["A"] + variables["aotf_shift"], 
                              wd=variables["sinc_width"], 
                              sl=variables["sidelobe"], 
                              af=variables["asymmetry"]) + variables["offset"]
    I0_hr = W_aotf * d["I0_solar_hr"]
    I0_p = np.matmul(d["W_conv"], I0_hr)
    return I0_p/max(I0_p)








def get_start_params(d):
    """compute parameters from fits"""

    """blaze"""
    #pre-compute delta nu per pixel
    d["nu_mp_centre"] = nu_mp(d["centre_order"], d["pixels"], d["temperature"], p0=0)
    d["p_dnu"] = (d["nu_mp_centre"][-1] - d["nu_mp_centre"][0])/320.0


    d["p0"] = np.polyval([0.22,150.8], d["centre_order"])
    d["blaze_shift"] = np.polyval([-0.736363, -6.363908], d["temperature"]) # Blaze frequency shift due to temperature [pixel from Celsius]
    d["p0"] += d["blaze_shift"]
    
    d["p_width"] = 22.473422 / d["p_dnu"]
    
    """aotf"""
    d["A_nu0"] = np.polyval([1.34082e-7, 0.1497089, 305.0604], d["A"]) # Frequency of AOTF [cm-1 from kHz]
    d["aotf_shift"]  = -6.5278e-5 * d["temperature"] * d["A_nu0"] # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
    
    d["A_nu0"] += d["aotf_shift"]
    
    
    """code to find line in AOTF kHz"""
    # A = np.arange(13000., 30000., 1.)
    # A_nu = np.polyval([1.34082e-7, 0.1497089, 305.0604], A)
    # coeffs = np.polyfit(A_nu, A, 2)
    # A_line = np.polyval(coeffs, d["line"])
    # print("nu_line=", d["line"], "A_line=", A_line)
    # A_nus = np.polyval([1.34082e-7, 0.1497089, 305.0604], d["aotf_freqs"][0:256])
    # for i, (A, nu) in enumerate(zip(d["aotf_freqs"][0:256], A_nus)): print(i, A, nu)


    """set up initial fits"""
    # width = np.polyval([1.11085173e-06, -8.88538288e-03,  3.83437870e+01], A_nu0)
    # lobe  = np.polyval([2.87490586e-06, -1.65141511e-02,  2.49266314e+01], A_nu0)
    # asym  = np.polyval([-5.47912085e-07, 3.60576934e-03, -4.99837334e+00], A_nu0)
 
    # Email 6th July 2021
    d["width"] = np.polyval([-3.03468322e-07, 1.79966624e-03, 1.80434587e+01], d["A_nu0"])
    d["lobe"]  = np.polyval([ 1.96290411e-06, -1.19113254e-02, 2.00409867e+01], d["A_nu0"])
    d["asym"]  = np.polyval([ 5.64936782e-09, -5.32457230e-06, 1.27745735e+00], d["A_nu0"])
    d["offset"]  = np.polyval([ 3.05238507e-07, -1.80269235e-03, 2.85281370e+00], d["A_nu0"]) #not yet implemented

    # d["width"] = np.polyval([-2.85452723e-07,  1.66652129e-03,  1.83411690e+01], d["A_nu0"]) # Sinc width [cm-1 from AOTF frequency cm-1]
    # d["lobe"]  = np.polyval([ 2.19386777e-06, -1.32919656e-02,  2.18425092e+01], d["A_nu0"]) # sidelobes factor [scaler from AOTF frequency cm-1]
    # d["asym"]  = np.polyval([-3.35834373e-10, -6.10622773e-05,  1.62642005e+00], d["A_nu0"]) # Asymmetry factor [scaler from AOTF frequency cm-1]
    
    return d



def make_param_dict(d):
    #best, min, max
    param_dict = {
        "blaze_centre":[d["p0"], d["p0"]-20.0, d["p0"]+20.],
        "aotf_width":[d["width"], d["width"]-2., d["width"]+2.],
        "aotf_shift":[0.0, -3.0, 3.0],
        "sidelobe":[d["lobe"], 0.05, 20.0],
        "asymmetry":[d["asym"], 0.01, 2.0],
        }
    
    if AOTF_OFFSET_SHAPE == "Constant":
        param_dict["offset"] = [0.0, 0.0, 0.3]
        
    else:    
        param_dict["offset_height"] = [0.0, 0.0, 0.3]
        param_dict["offset_width"] = [40.0, 10.0, 300.0]

    if BLAZE_WIDTH_FIT:
        param_dict["blaze_width"] = [d["p_width"], d["p_width"]-20., d["p_width"]+20.]

    return param_dict
    
def make_param_dict_aotf(d):
    #best, min, max
    param_dict = {
        "aotf_width":[d["width"], d["width"]-2., d["width"]+2.],
        "aotf_shift":[0.0, -3.0, 3.0],
        "sidelobe":[d["lobe"], 0.05, 20.0],
        "asymmetry":[d["asym"], 0.01, 2.0],
        "nu_offset":[d["A_nu0"], d["A_nu0"]-150., d["A_nu0"]+150.],
        }
    
    if AOTF_OFFSET_SHAPE == "Constant":
        param_dict["offset"] = [0.0, 0.0, 0.3]
        
    else:    
        param_dict["offset_height"] = [0.0, 0.0, 0.3]
        param_dict["offset_width"] = [40.0, 10.0, 300.0]

    return param_dict


def F_blaze(variables, d):
    
    dp = d["pixels"] - variables["blaze_centre"]
    dp[dp == 0.0] = 1.0e-6
    if BLAZE_WIDTH_FIT:
        F = (variables["blaze_width"]*np.sin(np.pi*dp/variables["blaze_width"])/(np.pi*dp))**2
    else:
        F = (d["p_width"]*np.sin(np.pi*dp/d["p_width"])/(np.pi*dp))**2
    
    return F


"""reverse AOTF asymmetry"""
def sinc(dx, amp, width, lobe, asym):
    # """asymetry switched 
 	sinc = amp * (width * np.sin(np.pi * dx / width) / (np.pi * dx))**2

 	ind = (abs(dx)>width).nonzero()[0]
 	if len(ind)>0: 
         sinc[ind] = sinc[ind]*lobe

 	ind = (dx>=width).nonzero()[0]
 	if len(ind)>0: 
         sinc[ind] = sinc[ind]*asym

 	return sinc




def F_aotf(nu_pm, variables, d):

    dx = nu_pm - d["A_nu0"] - variables["aotf_shift"]
    # print(dx)
    
    if AOTF_OFFSET_SHAPE == "Constant":
        offset = variables["offset"]
    else:
        offset = variables["offset_height"] * np.exp(-dx**2.0/(2.0*variables["offset_width"]**2.0))
    
    F = sinc(dx, 1.0, variables["aotf_width"], variables["sidelobe"], variables["asymmetry"]) + offset
    
    
    
    return F
    # return F/max(F)






def calc_spectrum(variables, d, I0=[0]):
    """make simulated spectrum"""    

    if I0[0] == 0:
        I0 = d["I0_lr"]
    
    solar = np.zeros(len(d["pixels"]))
    for order in range(d["m_range"][0], d["m_range"][1]+1):
        
        nu_pm = nu_mp(order, d["pixels"], d["temperature"])

        F = F_blaze(variables, d)
        G = F_aotf(nu_pm, variables, d)
        I0_lr_p = np.interp(nu_pm, d["nu_hr"], I0)
        
        solar += F * G * I0_lr_p
    
    return solar#/max(solar)





def fit_resid(params, d):
    """define the fit residual"""
    variables = {}
    for key in params.keys():
        variables[key] = params[key].value
        
    fit = calc_spectrum(variables, d)

    return (fit/max(fit) - d["spectrum_norm"]) / d["sigma"]





def fit_spectrum(param_dict, variables, d):
    # print("Fitting AOTF and blaze")
    params = lmfit.Parameters()
    for key, value in param_dict.items():
       params.add(key, value[0], min=value[1], max=value[2])

    lm_min = lmfit.minimize(fit_resid, params, args=(d,), method='leastsq')
    chisq = lm_min.chisqr
    # print("chisq=", chisq)

    for key in params.keys():
        variables[key] = lm_min.params[key].value

    return variables, chisq



    
    