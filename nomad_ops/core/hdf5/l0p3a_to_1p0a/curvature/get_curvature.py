# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:31:24 2020

@author: iant

TEST LNO CURVATURE

"""

# import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from tools.file.hdf5_functions import make_filelist
# from tools.spectra.fit_polynomial import fit_polynomial
# from tools.spectra.smooth_hr import smooth_hr
# from tools.spectra.savitzky_golay import savitzky_golay
# from tools.plotting.colours import get_colours
# from tools.general.get_minima_maxima import get_local_maxima_or_equals

from tools.file.read_write_hdf5 import read_hdf5_to_dict


from nomad_ops.core.hdf5.l0p3a_to_1p0a.curvature.curvature_functions import get_temperature_corrected_mean_curve

MAX_SZA = 30.

FIG_X = 17
FIG_Y = 9


# SAVE_PDF = True
SAVE_PDF = False


diffraction_order = 189


#get data from hdf5 dict
curvature_dict = read_hdf5_to_dict("lno_reflectance_factor_curvature_order_%i" %diffraction_order)[0]

clear_nu = curvature_dict["clear_nu"]
# reference_temperature = curvature_dict["reference_temperature"]
# mean_curve_coeffs = curvature_dict["mean_curve_coeffs"]



#get list of files for testing
regex = re.compile("201806.*_LNO_._DP_%i" %diffraction_order)
file_level = "hdf5_level_1p0a"


hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)

#first remove low sza files
chosen_hdf5_files = []
chosen_hdf5_filenames = []

for file_index, (hdf5_filename, hdf5_file) in enumerate(zip(hdf5_filenames, hdf5_files)):
    
    sza = np.mean(hdf5_file["Geometry/Point0/IncidenceAngle"][...], axis=1)
    
    valid_ys = np.where(sza < MAX_SZA)[0]
    if len(valid_ys) == 0:
        continue
    
    chosen_hdf5_files.append(hdf5_file)
    chosen_hdf5_filenames.append(hdf5_filename)





"""apply to nadir data"""
for file_index, (hdf5_filename, hdf5_file) in enumerate(zip(chosen_hdf5_filenames[0:3], chosen_hdf5_files[0:3])):

    
    y = hdf5_file["Science/YReflectanceFactor"][...]
    x = hdf5_file["Science/X"][...]
    temperature = float(hdf5_file["Channel/MeasurementTemperature"][0][0])
    
    valid_ys = np.where(sza == np.min(sza))[0]
     
    y_selected = np.mean(y[valid_ys[0]-4:valid_ys[0]+5], axis=0)

    mean_curve_shifted = get_temperature_corrected_mean_curve(temperature, diffraction_order)

    """Find first good pixel"""
    valid_xs = np.where((clear_nu[0, 0] < x) & (x < clear_nu[0, 1]))[0]
    x_first_pixel = valid_xs[0] #the first pixel where the polynomial fit is made
    x_last_pixel = 320.0

    plt.figure()
    pixels = curvature_dict["pixels"]
    mean_curve = curvature_dict["mean_curve"]
    
    plt.plot(pixels[x_first_pixel:], mean_curve[x_first_pixel:], "k:")
    plt.plot(pixels[x_first_pixel:], y_selected[x_first_pixel:], label="LNO reflectance factor before curvature removal")

    plt.plot(pixels[x_first_pixel:], mean_curve[x_first_pixel:], "k:")
    plt.plot(pixels[x_first_pixel:], mean_curve_shifted[x_first_pixel:], "r:")
    plt.legend()
        
    plt.figure()
    plt.plot(x[x_first_pixel:], y_selected[x_first_pixel:], label="LNO reflectance factor before curvature removal")
    plt.plot(x[x_first_pixel:], y_selected[x_first_pixel:]/mean_curve[x_first_pixel:], label="LNO reflectance factor after curvature removal (no temperature shift)")
    plt.plot(x[x_first_pixel:], y_selected[x_first_pixel:]/mean_curve_shifted[x_first_pixel:], label="LNO reflectance factor after curvature removal")
    plt.legend()



# if SAVE_PDF:
#     with PdfPages("LNO_curvature_order_%i.pdf" %diffraction_order) as pdf: #open pdf
    
#         for file_index, (hdf5_filename, hdf5_file) in enumerate(zip(chosen_hdf5_filenames, chosen_hdf5_files)):
    
#             ax_index = np.mod(file_index, 4) #subplot index: 0,1,2,3,0,1,2,3,0,1, etc
    
#             #if first subplot on the page, open new figure
#             if ax_index == 0:
#                 fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 11))
    
#                 #fudge to make big x and y labels work
#                 fig.add_subplot(111, frameon=False)
#                 plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#                 plt.ylabel("Reflectance Factor                     ")
#                 plt.xlabel("Wavenumbers cm-1")
    
#             #choose subplot 0,1,2 or 3
#             ax = axes.flat[ax_index]
            
#             #plot to subplot
#             ax.set_title("%s" %(hdf5_filename))
    
#             pixels = np.arange(320.)
            
#             valid_ys = np.where(sza == np.min(sza))[0]
            
#             y = hdf5_file["Science/YReflectanceFactor"][...]
#             x = hdf5_file["Science/X"][...]
#             temperature = float(hdf5_file["Channel/MeasurementTemperature"][0][0])
            
#             y_selected = np.mean(y[valid_ys[0]-4:valid_ys[0]+5], axis=0)
        
        
#             #shift the pixel peak to account for temperature
#             pixel_temperature_shift = np.polyval(lno_curvature_dict[diffraction_order]["temperature_shift_coeffs"], temperature) - max_index
#             pixels_shifted = pixels + pixel_temperature_shift
        
#             #reinterpolate temperature shifted curve onto original pixel grid
#             mean_curve_shifted = np.interp(pixels, pixels_shifted, mean_curve)
        
#             # plt.figure()
#             # plt.plot(pixels[x_first_pixel:], mean_curve[x_first_pixel:], "k:")
#             # plt.plot(pixels_shifted[x_first_pixel:], mean_curve[x_first_pixel:], "b:")
#             # plt.plot(pixels[x_first_pixel:], mean_curve_shifted[x_first_pixel:], "r:")
        
#             ax.plot(x[x_first_pixel:], y_selected[x_first_pixel:], label="LNO reflectance factor before curvature removal")
#             # plt.plot(x[x_first_pixel:], y_selected[x_first_pixel:]/mean_curve[x_first_pixel:])
#             ax.plot(x[x_first_pixel:], y_selected[x_first_pixel:]/mean_curve_shifted[x_first_pixel:], label="LNO reflectance factor after curvature removal")
#             ax.legend(loc="lower left")
            
#             fig.tight_layout()
#             #after plotting 4th subplot, save and close figure
#             if ax_index == 3:
#                 pdf.savefig()
#                 plt.close()
        
#         #if last occultation is not in 4th subplot, save and close figure anyway
#         if ax_index != 3:
#             pdf.savefig()
#             plt.close()

