# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:41:48 2021

@author: iant

MINISCAN FITTING FUNCTIONS
"""


import numpy as np
import os
import re
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
from tools.spectra.fit_polynomial import fit_polynomial

from tools.general.get_nearest_index import get_nearest_index

from instrument.nomad_so_instrument import nu_grid, F_blaze, nu_mp, spec_res_order, F_aotf_goddard18b, t_nu_mp
from instrument.nomad_so_instrument import F_blaze_goddard21, F_aotf_goddard21

from instrument.nomad_so_instrument import m_aotf as m_aotf_so

from instrument.calibration.so_aotf_ils.simulation_config import ORDER_RANGE, D_NU, pixels





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
    d["aotf_freq"] = d["aotf_freqs"][index]
    d["spectrum"] = d["spectra"][index]
    d["spectrum_norm"] = d["spectrum"]/np.max(d["spectrum"])
    
    return d




def get_nu_hr():
    
    temperature = 0.0

    nu_range = [
        nu_mp(ORDER_RANGE[0], np.zeros(1), temperature)[0] - 5.0, \
        nu_mp(ORDER_RANGE[1], np.zeros(1)+319.0, temperature)[0] + 5.0            
            ]
    nu_hr = np.arange(nu_range[0], nu_range[1], D_NU)
    
    return nu_hr




def spectrum_temperature(hdf5_file, channel, index):
    
    
    temperatures = get_sql_temperatures_all_spectra(hdf5_file, channel)
    # temperature = np.mean(temperatures)
    temperature = temperatures[index]
    return temperature

    


def get_spec_res():
    
    c_order = int(np.mean(ORDER_RANGE))
    spec_res = spec_res_order(c_order)
    
    return spec_res


def fit_temperature(d, hdf5_file):
    """code to check absorption lines in solar spectrum"""

    index = d["index"]
    absorption_line_fit_index = get_nearest_index(index, np.arange(0,1492,256)) * 256
    
    aotf_freqs = d["aotf_freqs"]
    spectra = d["spectra"]
    channel = d["channel"]
    nu_hr = get_nu_hr()
    temperature = spectrum_temperature(hdf5_file, channel, index)
    
    # plt.figure()
    order = m_aotf_so(aotf_freqs[absorption_line_fit_index])
    spectrum_w_absorption = spectra[absorption_line_fit_index]
    spectrum_cont = baseline_als(spectrum_w_absorption)
    spectrum_cr = spectrum_w_absorption[50:]/spectrum_cont[50:]
    pixels_nu = nu_mp(order, pixels, temperature)
    # plt.plot(pixels_nu[50:], spectrum_cr, label="Temperature spectral calibration")
    
    
    
    ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "Solar_irradiance_ACESOLSPEC_2015.dat")
    I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)
    I0_low_res = savgol_filter(I0_solar_hr, 499, 1)
    # I0_cont = fit_polynomial(nu_hr, I0_low_res, degree=2)
    # I0_cr = I0_low_res / I0_cont
    # I0_low_res = I0_low_res/np.max(I0_low_res)
    # plt.plot(nu_hr, I0_cr, label="Convolved solar line")
    
    d["I0_solar_hr"] = I0_solar_hr
    
    
    """code to shift spectral cal to match absorption"""
    #find nu of solar band
    absorption_min_index = np.argmin(I0_low_res)
    absorption_nu = nu_hr[absorption_min_index] #near enough
    smi = np.argmin(spectrum_cr) #spectrum min index
    x_hr, y_hr, min_position_nu = fit_gaussian_absorption(pixels_nu[50:][smi-3:smi+4], spectrum_cr[smi-3:smi+4])
    absorption_depth = np.min(y_hr)
    
    # plt.plot(x_hr, y_hr, linestyle="--", label="Fit to miniscan absorption")
    
    
    
    delta_nu = absorption_nu - min_position_nu
    print("delta_nu=", delta_nu)
    t_calc = t_nu_mp(order, absorption_nu, smi+50)
    delta_t = temperature - t_calc
    print("delta_t=", delta_t)
    
    d["absorption_depth"] = absorption_depth
    d["temperature"] = t_calc
    d["nu_hr"] = nu_hr
    
    return d



def calc_blaze(d):

    spec_res = get_spec_res()
    
    nu_hr = d["nu_hr"]
    hdf5_filename = d["hdf5_filename"]
    temperature = d["temperature"]
    
    #if convolution already saved to file
    h5_conv_filename = "conv_%s_order%i-%i_dnu%f_temp%.2f" %(hdf5_filename, ORDER_RANGE[0], ORDER_RANGE[1], D_NU, temperature)
    if os.path.exists(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename+".h5")):
        print("Reading W_conv from existing file")
        W_conv = read_hdf5_to_dict(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename))[0]["W_conv"]
    
        
    else:
        print("Making file", h5_conv_filename)
        Nbnu_hr = len(nu_hr)
        NbP = len(pixels)
        
        #old and new blaze functions are functional identical - use 2021 function only
        sconv = spec_res/2.355
        W_conv = np.zeros((NbP,Nbnu_hr))
        for iord in range(ORDER_RANGE[0], ORDER_RANGE[1]+1):
            print("Blaze order %i" %iord)
            nu_pm = nu_mp(iord, pixels, temperature)
            W_blaze = F_blaze_goddard21(iord, pixels, temperature)
            for ip in pixels:
                W_conv[ip,:] += (W_blaze[ip]*D_NU)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
                
        W_conv[W_conv < 1.0e-5] = 0.0 #remove small numbers
                
        write_hdf5_from_dict(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename), {"W_conv":W_conv}, {}, {}, {})
    
    d["W_conv"] = W_conv
    return d

    W_aotf = F_aotf_goddard18b(self.order, self.nu_hr, offset=0.)
    I0_hr = W_aotf * self.I0_hr       # nhr
    I_hr = I0_hr[None,:] * self.Trans_hr  # nz x nhr

    I0_p = np.zeros(self.NbP)
    I_p = np.zeros((self.NbZ,self.NbP))
    for iord in range(self.NbTotalOrders):
      for ip in range(self.NbP):
        inu1 = self.W2_conv_inu1[iord,ip]
        inu2 = inu1 + self.Nbnu_w
        I0_p[ip] += np.sum(I0_hr[inu1:inu2]*self.W2_conv[iord,ip,:])

        for iz in range(self.NbZ):
          I_p[iz,ip] += np.sum(I_hr[iz,inu1:inu2]*self.W2_conv[iord,ip,:])
        
    self.Trans_p = I_p / I0_p[None,:]     # nz x np



def aotf_conv(d, variables):
    W_aotf = F_aotf_goddard21(0., d["nu_hr"], d["temperature"], 
                              A=d["aotf_freq"] + variables["aotf_shift"], 
                              wd=variables["sinc_width"], 
                              sl=variables["sidelobe"], 
                              af=variables["asymmetry"]) + variables["offset"]
    I0_hr = W_aotf * d["I0_solar_hr"]
    I0_p = np.matmul(d["W_conv"], I0_hr)
    return I0_p/max(I0_p)
