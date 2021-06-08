# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:57:27 2021

@author: iant

FIT AOTF TO MINISCAN OBSERVATIONS
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



D_NU = 0.001
ORDER_RANGE = [192, 198]
pixels = np.arange(320)


file_level = "hdf5_level_0p2a"
regex = re.compile("20190416_020948_0p2a_SO_1_C")



order_range = ORDER_RANGE

colours = plt.get_cmap("tab10")


hdf5Files, hdf5Filenames, _ = make_filelist(regex, file_level)
hdf5_file = hdf5Files[0]
hdf5_filename = hdf5Filenames[0]
print(hdf5_filename)




channel = hdf5_filename.split("_")[3].lower()
aotf_freq = hdf5_file["Channel/AOTFFrequency"][...]
detector_data_all = hdf5_file["Science/Y"][...]
detector_centre_data = detector_data_all[:, [9,10,11,15], :] #chosen to avoid bad pixels
spectra = np.mean(detector_centre_data, axis=1)

with open("fit_log_%s.tsv" %hdf5_filename, "a") as f:
    f.write("i\taotf_freq\ttemperature\tsinc_width\taotf_shift\tsidelobe\tasymmetry\toffset\tabs_sum\tabs_sum_fit\n")

# for index in sorted(np.concatenate([
#         np.arange(0,1492,256), 
#         np.arange(1,1492,256), 
#         np.arange(2,1492,256), 
#         np.arange(3,1492,256),
#         np.arange(4,1492,256),
#         np.arange(5,1492,256),
#         np.arange(6,1492,256),
#         ])):

for index in [0]:


    temperatures = get_sql_temperatures_all_spectra(hdf5_file, channel)
    # temperature = np.mean(temperatures)
    
    c_order = int(np.mean(order_range))
    spec_res = spec_res_order(c_order)
    
    # dim = detector_data_all.shape
    
    
    temperature = temperatures[index]
    # temperature -= 3.0
    
    
    """spectral grid and blaze functions of all orders"""
    dnu = D_NU
    t = 0.0 #just for making grid
    
    nu_range = [
        nu_mp(order_range[0],  np.zeros(1), t)[0] - 5.0, \
        nu_mp(order_range[1],  np.zeros(1) + 319.0, t)[0] + 5.0            
            ]
    nu_hr = np.arange(nu_range[0], nu_range[1], dnu)
    
    
    
    """code to check absorption lines in SO data"""
    # plt.figure()
    # for i in range(0, 255, 20):
    #     order = m_aotf_so(aotf_freq[i])
    #     spectrum = spectra[i]
    #     spectrum_cont = baseline_als(spectrum)
    #     spectrum_cr = spectrum[50:]/spectrum_cont[50:]
    #     pixels_nu = nu_mp(order, pixels, temperature)
    #     plt.plot(pixels_nu[50:], spectrum_cr, label=i)
    # plt.legend()
    
    """code to check absorption lines in solar spectrum"""
    absorption_line_fit_index = get_nearest_index(index, np.arange(0,1492,256)) * 256
    
    # plt.figure()
    order = m_aotf_so(aotf_freq[absorption_line_fit_index])
    spectrum_w_absorption = spectra[absorption_line_fit_index]
    spectrum_cont = baseline_als(spectrum_w_absorption)
    spectrum_cr = spectrum_w_absorption[50:]/spectrum_cont[50:]
    pixels_nu = nu_mp(order, pixels, temperature)
    # plt.plot(pixels_nu[50:], spectrum_cr, label="Temperature spectral calibration")
    
    
    ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "Solar_irradiance_ACESOLSPEC_2015.dat")
    I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)
    I0_low_res = savgol_filter(I0_solar_hr, 499, 1)
    I0_cont = fit_polynomial(nu_hr, I0_low_res, degree=2)
    I0_cr = I0_low_res / I0_cont
    # I0_low_res = I0_low_res/np.max(I0_low_res)
    # plt.plot(nu_hr, I0_cr, label="Convolved solar line")
    
    
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
    
    
    # pixels_nu = nu_mp(order, pixels, t_calc)
    # plt.plot(pixels_nu[50:], spectrum_cr, label="Fitted spectral calibration")
    # plt.legend()
    
    
    temperature = t_calc
    
    spectrum = spectra[index]
    
    
    ss_file = os.path.join(paths["RETRIEVALS"]["SOLAR_DIR"], "Solar_irradiance_ACESOLSPEC_2015.dat")
    I0_solar_hr = get_solar_hr(nu_hr, solspec_filepath=ss_file)
    
    
    #if convolution already saved to file
    h5_conv_filename = "conv_%s_order%i-%i_dnu%f_temp%.2f" %(hdf5_filename, order_range[0], order_range[1], D_NU, temperature)
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
        for iord in range(order_range[0], order_range[1]+1):
            print("Blaze order %i" %iord)
            nu_pm = nu_mp(iord, pixels, temperature)
            W_blaze = F_blaze_goddard21(iord, pixels, temperature)
            for ip in pixels:
                W_conv[ip,:] += (W_blaze[ip]*dnu)/(np.sqrt(2.*np.pi)*sconv)*np.exp(-(nu_hr-nu_pm[ip])**2/(2.*sconv**2))
                
        W_conv[W_conv < 1.0e-5] = 0.0 #remove small numbers
                
        write_hdf5_from_dict(os.path.join(paths["SIMULATION_DIRECTORY"], h5_conv_filename), {"W_conv":W_conv}, {}, {}, {})
    
    
    
    
    
    
    
    def aotf_conv(sinc_width, aotf_shift, sidelobe, asymmetry, offset):
        W_aotf = F_aotf_goddard21(0., nu_hr, temperature, A=aotf_freq[index]+aotf_shift, wd=sinc_width, sl=sidelobe, af=asymmetry) + offset
        I0_hr = W_aotf * I0_solar_hr
        I0_p = np.matmul(W_conv, I0_hr)
        return I0_p/max(I0_p)
    
    
    def aotf_fit_resid(params, spectrum_norm, sigma):
        sinc_width = params['sinc_width'].value
        aotf_shift = params['aotf_shift'].value
        sidelobe = params['sidelobe'].value
        asymmetry = params['asymmetry'].value
        offset = params['offset'].value
    
        return (aotf_conv(sinc_width, aotf_shift, sidelobe, asymmetry, offset) - spectrum_norm) / sigma
    
    
    #standard AOTF width
    W_aotf = F_aotf_goddard21(0., nu_hr, temperature, A=aotf_freq[index])
    I0_hr = W_aotf * I0_solar_hr
    I0_p = np.matmul(W_conv, I0_hr)
    solar_scaled = I0_p/max(I0_p)
    
    
    
    # popt, pcov = curve_fit(aotf_fit, pixels, spectrum)
    params = lmfit.Parameters()
    # nu0: 4382.640722753343
    # sinc width: 20.739074862906232
    # sidelobe factor: 7.770908740766124
    # asymmetry: 0.28037507899728986
    
    param_dict = {
        "sinc_width":[20.5, 15., 25.],
        "aotf_shift":[0.0, -100.0, 100.0],
        "sidelobe":[7.0, 0.001, 30.0],
        "asymmetry":[0.3, 0.001, 50.0],
        "offset":[0.01, 0.0, 0.5],
        }
    
    for key, value in param_dict.items():
        params.add(key, value[0], min=value[1], max=value[2])
    
    
    smi_fitted = smi + 50
    sigma = np.ones_like(spectrum)
    # sigma[smi_fitted-12:smi_fitted+13] = 0.01
    sigma[np.arange(50, 301, 50)] = 0.001
    print("Fitting to miniscan")
    spectrum_norm = spectrum/np.max(spectrum)
    lm_min = lmfit.minimize(aotf_fit_resid, params, args=(spectrum_norm,sigma), method='leastsq')
    sinc_width = lm_min.params["sinc_width"].value
    aotf_shift = lm_min.params["aotf_shift"].value
    sidelobe = lm_min.params["sidelobe"].value
    asymmetry = lm_min.params["asymmetry"].value
    offset = lm_min.params["offset"].value
    print(lm_min.params)
    
    #check if bounds hit
    error = False
    for key, value in param_dict.items():
        if np.abs(lm_min.params[key].value - value[1]) < 0.0001:
            error = True
        if np.abs(lm_min.params[key].value - value[2]) < 0.0001:
            error = True
    
    fitted_scaled = aotf_conv(sinc_width, aotf_shift, sidelobe, asymmetry, offset)
    
    
    abs_sum = np.sum(np.abs(spectrum_norm - solar_scaled))
    abs_sum_fit = np.sum(np.abs(spectrum_norm - fitted_scaled))
    
    print(abs_sum)
    print(abs_sum_fit)

    if not error:
        with open("fit_log_%s.tsv" %hdf5_filename, "a") as f:
            f.write(f"{index:.0f}\t{aotf_freq[index]:.0f}\t{temperature:.5f}\t{sinc_width:.5f}\t{aotf_shift:.5f}\t{sidelobe:.5f}\t{asymmetry:.5f}\t{offset:.5f}\t{abs_sum:.5f}\t{abs_sum_fit:.5f}\n")
    
    
    fig1, (ax1a, ax1b) = plt.subplots(ncols=2, figsize=(12, 5))
    W_aotf_fitted = F_aotf_goddard21(0., nu_hr, temperature, A=aotf_freq[index]+aotf_shift, wd=sinc_width, sl=sidelobe, af=asymmetry) + offset
    ax1b.plot(nu_hr, W_aotf, color=colours(0), label="Goddard 21 AOTF")
    ax1b.plot(nu_hr, W_aotf_fitted, color=colours(1), label="Best fit AOTF")
    ax1b.set_xlabel("Wavenumber cm-1")
    ax1b.legend()
    
    ax1a.plot(pixels, spectrum_norm, color=colours(2), label=hdf5_filename)
    ax1a.plot(pixels, solar_scaled, color=colours(0), label="Goddard 21 (asym reversed)")
    ax1a.plot(pixels, fitted_scaled, color=colours(1), label="Fitted")
    ax1a.plot(pixels[smi_fitted-6:smi_fitted+7], fitted_scaled[smi_fitted-6:smi_fitted+7], color=colours(3), linestyle="--")
    ax1a.scatter(pixels[np.arange(50, 301, 50)], fitted_scaled[np.arange(50, 301, 50)], color=colours(1), linestyle="--")
    ax1a.text(0, 1, f"chisq={lm_min.chisqr:.1f}")
    ax1a.legend()
    ax1a.set_xlabel("Pixel number")
    if error:
        print("ERROR")
        fig1.savefig(os.path.join(paths["SIMULATION_DIRECTORY"], "%s_%i_%i_error.png" %(hdf5_filename, aotf_freq[index], index)))
    else:
        fig1.savefig(os.path.join(paths["SIMULATION_DIRECTORY"], "%s_%i_%i.png" %(hdf5_filename, aotf_freq[index], index)))
    plt.close()