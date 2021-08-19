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
from scipy.signal import savgol_filter

from tools.plotting.colours import get_colours
from tools.general.get_nearest_index import get_nearest_index


from instrument.calibration.so_aotf_ils.simulation_config import AOTF_OFFSET_SHAPE, sim_parameters


# line = 4383.5
line = 4276.1
# line = 3787.9





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
        "aotf_amplitude":[0.8, 0.75, 0.85],
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



def F_aotf2(variables, x):
    
    dx = x - variables["nu_offset"] + 1.0e-6

    if AOTF_OFFSET_SHAPE == "Constant":
        offset = variables["offset"]
    else:
        offset = variables["offset_height"] * np.exp(-dx**2.0/(2.0*variables["offset_width"]**2.0))
    
    F = sinc(dx, variables["aotf_amplitude"], variables["aotf_width"], variables["sidelobe"], variables["asymmetry"]) + offset
    return F


def aotf_residual(params, x, y):
    """define the fit residual"""
    variables = {}
    for key in params.keys():
        variables[key] = params[key].value
        
    fit = F_aotf2(variables, x)

    return (fit/max(fit) - y) #/ sigma

def fit_aotf(param_dict, x, y):
    print("Fitting AOTF")
    params = lmfit.Parameters()
    for key, value in param_dict.items():
       params.add(key, value[0], min=value[1], max=value[2])
    
    lm_min = lmfit.minimize(aotf_residual, params, args=(x, y), method='leastsq')
    chisq = lm_min.chisqr

    variables = {}
    for key in params.keys():
        variables[key] = lm_min.params[key].value

    return variables, chisq
    



def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


    
all_points = {"A_nu0":[], "F_aotf":[]}
plt.figure(figsize=(15, 8))

for filename in filenames:
    
    for suffix in suffixes:
    
        file_path = os.path.join("output", "so_miniscan_aotf_fits", "%s_%s_aotf_function.txt" %(filename, suffix))
    
        txt_in = np.loadtxt(file_path, skiprows=1, delimiter=",")
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

        
        plt.scatter(d_good["A_nu0"], d_good["F_aotf"], c=c)
        
        
        
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
nbins = 500
bins_int = np.linspace(4250., 4550., nbins+1)
ind = np.digitize(all_points["A_nu0"], bins_int)
bins = bins_int + (bins_int[1] - bins_int[0])
F_aotf_binned = np.array([np.mean(all_points["F_aotf"][ind == j]) for j in range(0, nbins+1)])




"""smoothing filter"""
smooth = non_uniform_savgol(bins[~np.isnan(F_aotf_binned)], F_aotf_binned[~np.isnan(F_aotf_binned)], 25, 2)
bins_smoothed = bins[~np.isnan(F_aotf_binned)]
smooth_scalar = 1. / max(smooth)
smoothed_norm = smooth * smooth_scalar

plt.figure()
plt.scatter(bins, F_aotf_binned * smooth_scalar, c="k", marker="+", label="Raw binned data")
plt.plot(bins_smoothed, smoothed_norm, label="Smoothed (Savitsky-Golay filter)")
plt.xlabel("AOTF Wavenumber")
plt.ylabel("Area of absorption line")
plt.title("AOTF from fitting 4383cm-1 solar line")
plt.legend()
plt.grid()

"""fit aotf function to centre + main sidelobes"""
centre_nu_range = [4345., 4420.]
centre_indices = np.where((bins_smoothed > centre_nu_range[0]) & (bins_smoothed < centre_nu_range[1]))[0]

param_dict = make_param_dict_aotf()
# fit_params, chisq = fit_aotf(param_dict, bins_smoothed, smoothed_norm)

"""fit aotf function"""
fit_params = {
    "aotf_width":19.5,
    "aotf_amplitude":0.95,
    "sidelobe":4.,
    "asymmetry":1.8,
    "nu_offset":4385.0,
    "offset_height":0.05,
    "offset_width":500,
    }



x_range = np.arange(min(bins_smoothed), max(bins_smoothed), 0.01)
F_aotf_fitted = F_aotf2(fit_params, x_range)
plt.figure()
plt.plot(bins_smoothed, smoothed_norm, label="Smoothed (Savitsky-Golay filter)")
plt.plot(x_range, F_aotf_fitted, "k--", label="AOTF fit using existing parameters")
plt.xlabel("AOTF Wavenumber")
plt.ylabel("Area of absorption line")
plt.title("AOTF from fitting 4383cm-1 solar line")
plt.legend()
plt.grid()

"""interpolate smoothed onto regular grid"""
aotf_interp = np.interp(x_range, bins_smoothed, smoothed_norm)
x_range = x_range - x_range[np.where(aotf_interp == max(aotf_interp))[0][0]]
# aotf_function = 

plt.figure()
plt.plot(x_range, aotf_interp)
plt.xlabel("AOTF Wavenumber")
plt.ylabel("Area of absorption line")
plt.title("AOTF from fitting 4383cm-1 solar line")
plt.grid()

filename = "AOTF_from_fitting_4383cm-1_solar_line"
plt.savefig("%s.png" %filename)

np.savetxt("%s.txt" %filename, np.array([x_range, aotf_interp]).T, fmt="%.6f", delimiter=",", header="Wavenumber,AOTF function")
