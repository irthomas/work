# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:11:14 2020

@author: iant
"""


import numpy as np




def read_hdf5_to_dict(hdf5_filename):

    import h5py

    with h5py.File(hdf5_filename+".h5",'r') as f:
    
        datasets = {}
        for node, dataset in f.items():
            path = node
        #    print(path)
            if isinstance(f[path], h5py.Dataset):
                datasets[path] = dataset[...]
            else:
                for node2, dataset2 in f[node].items():
                    path = node+"/"+node2
        #            print(path)
                    if isinstance(f[path], h5py.Dataset):
                        datasets[path] = dataset2[...]
                    else:
                        for node3, dataset3 in f[node][node2].items():
                            path = node+"/"+node2+"/"+node3
        #                    print(path)
                            if isinstance(f[path], h5py.Dataset):
                                datasets[path] = dataset3[...]
                                
        attributes = {}
        for name in f.attrs:
            attributes[name] = f.attrs[name]

    return datasets, attributes            




def get_temperature_corrected_mean_curve(temperature_in, curvature_dict):
    """get mean curve, shift peak to correct for temperature, interpolate back onto original pixel grid"""

    
    temperature_shift_coeffs = curvature_dict["temperature_shift_coeffs"]
    reference_temperature_peak = curvature_dict["reference_temperature_peak"]
    mean_curve = curvature_dict["mean_curve"]
    pixels = curvature_dict["pixels"]

    """shift curve to accommodate the temperature effect"""
    #shift the pixel peak to account for temperature
    pixel_temperature_shift = np.polyval(temperature_shift_coeffs, temperature_in) - reference_temperature_peak
    pixels_shifted = pixels + pixel_temperature_shift

    """After shifting, reinterpolate onto the pixel grid"""
    #reinterpolate temperature shifted curve onto original pixel grid
    mean_curve_shifted = np.interp(pixels, pixels_shifted, mean_curve)

    return mean_curve_shifted



def make_correction_curve(temperature, coeffs, pixels, degree=3):
    
    points = np.zeros((4, 2))
    for i in range(4):
        for j in range(2):
            points[i, j] = np.polyval(coeffs[i, j, :], temperature)
    
    polyfit = np.polyfit(points[:, 0], points[:, 1], degree)
    
    curve = np.polyval(polyfit, pixels)
    return curve    



