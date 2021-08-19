# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:58:49 2021

@author: iant

READ IN OUTPUT FROM AOTF FIT 
"""


import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as mcol

import lmfit
import os
# from scipy.signal import savgol_filter

from tools.plotting.colours import get_colours
from tools.general.get_nearest_index import get_nearest_index
from tools.spectra.non_uniform_savgol import non_uniform_savgol

from instrument.calibration.so_aotf_ils.simulation_config import AOTF_OFFSET_SHAPE, sim_parameters


line = 4383.5
# line = 4276.1
# line = 3787.9


# SAVE_OUTPUT = True
SAVE_OUTPUT = False


error_n_medians = sim_parameters[line]["error_n_medians"]
filenames = sim_parameters[line]["filenames"]



suffixes = sim_parameters[line]["solar_spectra"].keys()

# cmap = mcol.LinearSegmentedColormap.from_list("",["r","b"])


temperature_colours = np.arange(-8, 8, 0.1)
colours = get_colours(len(temperature_colours), cmap="brg")

###for fitting final AOTF shape

def make_param_dict_aotf(line):
    #best, min, max
    param_dict = {
        "aotf_width":[20., 18., 22.],
        "aotf_amplitude":[0.8, 0.5, 1.25],
        "sidelobe":[1., 0.05, 20.0],
        "asymmetry":[1.0, 0.01, 2.0],
        "nu_offset":[line, line-30., line+30],
        }
    
    if AOTF_OFFSET_SHAPE == "Constant":
        param_dict["offset"] = [0.0, 0.0, 0.3]
        
    else:    
        param_dict["offset_height"] = [0.0, 0.0, 0.3]
        param_dict["offset_width"] = [40.0, 10.0, 300.0]

    return param_dict



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
    
    F = sinc(dx, variables["aotf_amplitude"], variables["aotf_width"], variables["sidelobe"], variables["asymmetry"]) + offset
    
    #normalise offset
    variables["offset_height"] = variables["offset_height"] / np.max(F)
    return F/max(F)


def aotf_residual(params, x, y, sigma):
    """define the fit residual"""
    variables = {}
    for key in params.keys():
        variables[key] = params[key].value
        
    fit = F_aotf2(variables, x)

    return (fit/max(fit) - y) / sigma

def fit_aotf(param_dict, x, y, sigma):
    print("Fitting AOTF")
    params = lmfit.Parameters()
    for key, value in param_dict.items():
       params.add(key, value[0], min=value[1], max=value[2])
    
    lm_min = lmfit.minimize(aotf_residual, params, args=(x, y, sigma), method='leastsq')
    chisq = lm_min.chisqr

    variables = {}
    for key in params.keys():
        variables[key] = lm_min.params[key].value

    return variables, chisq
    




    
all_points = {"A_nu0":[], "F_aotf":[]}
# plt.figure(figsize=(15, 8))

for filename in filenames:
    
    for suffix in suffixes:
    
        file_path = os.path.join("output", "so_miniscan_aotf_fits", "%s_%s_%.0f_aotf_function.txt" %(filename, suffix, line))
        
        if os.path.exists(file_path):
            txt_in = np.loadtxt(file_path, skiprows=1, delimiter=",")
        else:
            print("Warning: file %s does not exist" %file_path)
            continue
            
        d_in = {"A":txt_in[:,0],"A_nu0":txt_in[:,1],"t0":txt_in[:,2],"t_calc":txt_in[:,3], "F_aotf":txt_in[:,4], "error":txt_in[:,5]}
        
        good_indices = np.where(d_in["error"] < np.median(d_in["error"]) * error_n_medians)[0]
        
    
        #remove bad points where chisq too high    
        d_good = {}
        for key,values in d_in.items():
            d_good[key] = values[good_indices]
            
        #sort together
        sort_ix = np.argsort(d_good["A"])
        for key,values in d_good.items():
            d_good[key] = d_good[key][sort_ix]
        
        
        c = [colours[get_nearest_index(t, temperature_colours)] for t in d_good["t0"]]
            

        all_points["A_nu0"].extend(d_good["A_nu0"])
        all_points["F_aotf"].extend(d_good["F_aotf"])

        
        # plt.scatter(d_good["A_nu0"], d_good["F_aotf"], c=c)
        
        
        
        # """histogram bin"""
        # nbins = 500
        # bins = np.linspace(4200., 4500., nbins+1)
        # ind = np.digitize(d_good["A_nu0"], bins)
        # F_aotf_binned = [np.mean(d_good["F_aotf"][ind == j]) for j in range(0, nbins+1)]
        
        # plt.scatter(bins, F_aotf_binned)
        
        
        """fit aotf function"""
        # param_dict = make_param_dict_aotf()
        # # fit_params, chisq = fit_aotf(param_dict, d_good["A_nu0"], d_good["F_aotf"])
        # fit_params = {
        #     "aotf_width":18.5,
        #     "aotf_amplitude":0.65,
        #     "sidelobe":4.,
        #     "asymmetry":1.8,
        #     "nu_offset":4385.0,
        #     "offset_height":0.02079,
        #     "offset_width":65.89586,
        #     }
    
        # x_range = np.arange(min(d_good["A_nu0"]), max(d_good["A_nu0"]), 0.01)
        # F_aotf_fitted = F_aotf2(fit_params, x_range)
        # plt.plot(x_range, F_aotf_fitted)
    
        # print(chisq)
        print(min(d_good["t0"]), max(d_good["t0"]))
    
# plt.xlim([4380, 4450])


all_points["A_nu0"] = np.array(all_points["A_nu0"])
all_points["F_aotf"] = np.array(all_points["F_aotf"])


"""histogram bin"""
bins_int = sim_parameters[line]["histogram_bins"]
ind = np.digitize(all_points["A_nu0"], bins_int)
bins = bins_int + (bins_int[1] - bins_int[0])
F_aotf_binned = np.array([np.mean(all_points["F_aotf"][ind == j]) for j in range(0, len(bins_int))])




"""smoothing filter"""
smooth = non_uniform_savgol(bins[~np.isnan(F_aotf_binned)], F_aotf_binned[~np.isnan(F_aotf_binned)], 25, 2)
bins_smoothed = bins[~np.isnan(F_aotf_binned)]
smooth_scalar = 1. / max(smooth)
smoothed_norm = smooth * smooth_scalar

# plt.figure()
# plt.scatter(bins, F_aotf_binned * smooth_scalar, c="k", marker="+", label="Raw binned data")
# plt.plot(bins_smoothed, smoothed_norm, label="Smoothed (Savitsky-Golay filter)")
# plt.xlabel("AOTF Wavenumber")
# plt.ylabel("Area of absorption line")
# plt.title("AOTF from fitting %.0fcm-1 solar line" %line)
# plt.legend()
# plt.grid()

"""fit aotf function to centre + main sidelobes"""

if line == 4383.5:
    # centre_nu_range = [4000., 5000.]
    # centre_nu_range = [4345., 4420.]
    centre_nu_range = [4331., 4444.]
    # centre_nu_range = [4320., 4480.]

if line == 4276.1:
    centre_nu_range = [4000., 5000.]
    # centre_nu_range = [4239., 4317.]
    # centre_nu_range = [4265., 4288.]
centre_indices = np.where((bins_smoothed > centre_nu_range[0]) & (bins_smoothed < centre_nu_range[1]))[0]

param_dict = make_param_dict_aotf(line)
sigma = np.ones_like(smoothed_norm[centre_indices])
fit_params, chisq = fit_aotf(param_dict, bins_smoothed[centre_indices], smoothed_norm[centre_indices], sigma)

print("chisq=", chisq)
"""fit aotf function"""
# fit_params = {
#     "aotf_width":19.455,
#     "aotf_amplitude":0.79,
#     "sidelobe":5.968,
#     "asymmetry":1.267,
#     "nu_offset":4384.799,
#     # "nu_offset":4276.1,
#     "offset_height":0.137,
#     "offset_width":37.193,
#     }

filename = "AOTF_from_fitting_%.0fcm-1_solar_line" %line


x_range = np.arange(min(bins_smoothed), max(bins_smoothed), 0.01)
F_aotf_fitted = F_aotf2(fit_params, x_range)
plt.figure(figsize=(7,5))
plt.plot(bins_smoothed, smoothed_norm, label="Smoothed (Savitsky-Golay filter)")
plt.plot(x_range, F_aotf_fitted, "k--", label="AOTF fit using existing parameters")
plt.plot(x_range, aotf_offset(fit_params, x_range), "g:", label="Gaussian offset")

text = "\n".join(["%s: %0.3f" %(key, value) for key, value in fit_params.items() if key != "nu_offset"])
plt.text(min(x_range), 0.4, text)
plt.xlabel("AOTF Wavenumber")
plt.ylabel("Area of absorption line")
plt.title("AOTF from fitting %.0fcm-1 solar line" %line)
plt.legend()
plt.grid()

if SAVE_OUTPUT:
    plt.savefig("%s_fit2.png" %filename)


"""interpolate smoothed onto regular grid"""
aotf_interp = np.interp(x_range, bins_smoothed, smoothed_norm)
x_range = x_range - x_range[np.where(aotf_interp == max(aotf_interp))[0][0]]
x_range = x_range[::-1]
plt.figure()
plt.plot(x_range, aotf_interp)
plt.xlabel("AOTF Wavenumber")
plt.ylabel("Area of absorption line")
plt.title("AOTF from fitting %.0fcm-1 solar line" %line)
plt.grid()

if SAVE_OUTPUT:
    plt.savefig("%s.png" %filename)

    np.savetxt("%s.txt" %filename, np.array([x_range, aotf_interp]).T, fmt="%.6f", delimiter=",", header="Wavenumber,AOTF function")
