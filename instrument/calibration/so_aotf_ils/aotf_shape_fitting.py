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

from tools.plotting.colours import get_colours
from tools.general.get_nearest_index import get_nearest_index


from instrument.calibration.so_aotf_ils.simulation_config import AOTF_OFFSET_SHAPE

# max_chisq = 0.01

filenames = [
    "20180716_000706_0p2a_SO_1_C",
    "20181010_084333_0p2a_SO_2_C",
    "20181129_002850_0p2a_SO_2_C",
    "20181206_171850_0p2a_SO_2_C",
    "20190416_024455_0p2a_SO_1_C",
    "20210226_085144_0p2a_SO_2_C",
    "20210201_111011_0p2a_SO_2_C",
    "20190416_020948_0p2a_SO_1_C",
]    


# cmap = mcol.LinearSegmentedColormap.from_list("",["r","b"])


temperature_colours = np.arange(-8, 8, 0.1)
colours = get_colours(len(temperature_colours), cmap="brg")

###for fitting final AOTF shape

def make_param_dict_aotf():
    #best, min, max
    param_dict = {
        "aotf_width":[20., 18., 22.],
        "aotf_amplitude":[0.8, 0.75, 0.85],
        "sidelobe":[1., 0.05, 20.0],
        "asymmetry":[1.0, 0.01, 2.0],
        "nu_offset":[4383., 4360., 4400.],
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
    






    
    
plt.figure(figsize=(15, 8))

for filename in filenames:
    
    file_path = os.path.join("output", "so_miniscan_aotf_fits", "%s_aotf_function.txt" %filename)

    txt_in = np.loadtxt(file_path, skiprows=1, delimiter=",")
    d_in = {"A":txt_in[:,0],"A_nu0":txt_in[:,1],"t0":txt_in[:,2],"t_calc":txt_in[:,3], "F_aotf":txt_in[:,4], "error":txt_in[:,5]}
    
    good_indices = np.where(d_in["error"] < np.median(d_in["error"])*2.)[0]
    

    #remove bad points where chisq too high    
    d_good = {}
    for key,values in d_in.items():
        d_good[key] = values[good_indices]
        
    #sort together
    sort_ix = np.argsort(d_good["A"])
    for key,values in d_good.items():
        d_good[key] = d_good[key][sort_ix]
    
    
    c = [colours[get_nearest_index(t, temperature_colours)] for t in d_good["t0"]]
        
    
    plt.scatter(d_good["A_nu0"], d_good["F_aotf"], c=c)
    
    
    param_dict = make_param_dict_aotf()
    
    # fit_params, chisq = fit_aotf(param_dict, d_good["A_nu0"], d_good["F_aotf"])

    
    fit_params = {
        "aotf_width":18.5,
        "aotf_amplitude":0.65,
        "sidelobe":4.,
        "asymmetry":1.8,
        "nu_offset":4385.0,
        "offset_height":0.02079,
        "offset_width":65.89586,
        }

    x_range = np.arange(min(d_good["A_nu0"]), max(d_good["A_nu0"]), 0.01)
    F_aotf_fitted = F_aotf2(fit_params, x_range)
    plt.plot(x_range, F_aotf_fitted)

    # print(chisq)
    print(min(d_good["t0"]), max(d_good["t0"]))
    
# plt.xlim([4380, 4450])




    