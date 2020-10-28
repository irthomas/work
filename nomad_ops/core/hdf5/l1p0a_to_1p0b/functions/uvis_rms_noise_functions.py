# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:28:21 2020

@author: iant

FUNCTIONS FOR CALCULATING UVIS OCC RMS NOISE

"""

import numpy as np
import os
import re

from scipy.io import readsav
# from tools.spectra.savitzky_golay import savitzky_golay

from nomad_ops.core.hdf5.l1p0a_to_1p0b.config import THUMBNAILS_DESTINATION, UVIS_RMS_NOISE_DIRECTORY




def make_rms_dict(rms_dataset_name):
    
    rms_dataset_path = os.path.join(UVIS_RMS_NOISE_DIRECTORY, rms_dataset_name)
    
    idl_dict = readsav(rms_dataset_path)["x"]
    variable_names = idl_dict.dtype.names #get all dataset names
    
    rms_dict = {} #convert to python dictionary
    for variable_name in variable_names: #loop through names copying each to dict
        rms_dict[variable_name] = idl_dict[variable_name][0]
    return rms_dict


def get_rms_dict(horizontal_binning):
    """get RMS data"""
    #choose rms noise dataset
    rms_dataset_dict = {
        # 0:"transerr_err2_nt21_1024.idlsav", 
        # 3:"transerr_err2_nt21_256.idlsav", 
        # 7:"transerr_err2_nt21_128.idlsav",

        0:"transerr_err3_nt21_1024_md.idlsav", 
        3:"transerr_err3_nt21_256_md.idlsav", 
        7:"transerr_err3_nt21_128_md.idlsav",
        }
    
    if horizontal_binning in rms_dataset_dict.keys():
        rms_dataset_name = rms_dataset_dict[horizontal_binning]
        
        rms_dict = make_rms_dict(rms_dataset_name)
    
    else:
        print("Error: Binning scheme %i is unknown" %horizontal_binning)

    return rms_dict





def find_index(value_in, list_in):
    return next(x[0] for x in enumerate(list_in) if x[1] > value_in) - 1


# def smoothTriangle(data, degree, dropVals=False):
#     triangle=np.array(list(range(degree)) + [degree] + list(range(degree)[::-1])) + 1
#     smoothed=[]

#     for i in range(degree, len(data) - degree * 2):
#         point=data[i:i + len(triangle)] * triangle
#         smoothed.append(sum(point)/sum(triangle))
#     if dropVals:
#         return smoothed
#     smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
#     while len(smoothed) < len(data):
#         smoothed.append(smoothed[-1])
#     return smoothed

# def movingaverage(interval, window_size):
#     window = np.ones(int(window_size))/float(window_size)
#     return np.convolve(interval, window, 'same')

def convolve_smooth(x, span):
    
    convolution = np.convolve(x, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    if span > 2:
        print("Error")
        return []
    
    if span > 1:
        convolution[1] = convolution[2]
        convolution[-2] = convolution[-3]
    if span > 0:
        convolution[0] = convolution[1]
        convolution[-1] = convolution[-2]
       
    return convolution






def prepare_nadir_fig_tree(fig_name):
    
    fig_path = os.path.join(THUMBNAILS_DESTINATION, "lno_1p0a_radiance_factor")  
    
    m = re.match("(\d{4})(\d{2})(\d{2}).*", fig_name)
    year = m.group(1)
    month = m.group(2)
    path_fig = os.path.join(fig_path, year, month) #note: not split by day
    if not os.path.isdir(path_fig):
            os.makedirs(path_fig, exist_ok=True)
    return os.path.join(path_fig, fig_name)

