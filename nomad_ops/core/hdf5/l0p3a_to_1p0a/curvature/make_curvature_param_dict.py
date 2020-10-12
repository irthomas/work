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
# from matplotlib.backends.backend_pdf import PdfPages


from tools.file.hdf5_functions import make_filelist
from tools.spectra.fit_polynomial import fit_polynomial
# from tools.spectra.smooth_hr import smooth_hr
# from tools.spectra.savitzky_golay import savitzky_golay
from tools.plotting.colours import get_colours
# from tools.general.get_minima_maxima import get_local_maxima_or_equals

MAX_SZA = 30.

# SAVE_PDF = True
SAVE_PDF = False

lno_curvature_dict = {
168:{"clear_nu":[[3780., 3784.], [3785., 3796.], [3797., 3801.], [3802., 3810.]], "temperature_shift_coeffs":[-1.00061872, 88.19576276], },
189:{"clear_nu":[[4254., 4263.], [4265., 4267.], [4268., 4270.], [4272., 4274.], [4275.5, 4281.]], "temperature_shift_coeffs":[-0.84808798, 116.03337011], },

# 189:{"clear_nu":[]},
}

diffraction_order = 189



if diffraction_order not in lno_curvature_dict.keys():
    print("Error: diffraction order not yet ready")
    sys.exit()

regex = re.compile("20.*_LNO_._DP_%i" %diffraction_order)
file_level = "hdf5_level_1p0a"


hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)

plt.figure()



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


colours = get_colours(len(chosen_hdf5_filenames))


interpolation_pixels = np.arange(50.0, 300.0, 20.)
curvature_dict = {}
for interpolation_pixel in interpolation_pixels:
    curvature_dict[interpolation_pixel] = []



variables = {"temperature":[], "peak":[], "peak_shifted":[], "colours":[]}
for file_index, (hdf5_filename, hdf5_file) in enumerate(zip(chosen_hdf5_filenames, chosen_hdf5_files)):

    pixels = np.arange(320.)
    
    valid_ys = np.where(sza == np.min(sza))[0]
    
    y = hdf5_file["Science/YReflectanceFactor"][...]
    x = hdf5_file["Science/X"][...]
    temperature = float(hdf5_file["Channel/MeasurementTemperature"][0][0])

    y_selected = np.mean(y[valid_ys[0]-4:valid_ys[0]+5], axis=0)

    
    #find and remove water line indices
    valid_xs = []
    
    #if absorption lines not yet defined, plot raw data and stop
    if len(lno_curvature_dict[diffraction_order]["clear_nu"]) == 0:
        plt.plot(x, y_selected)
        
        if file_index == 100:
            sys.exit()
        
        continue
    
    else:
        
        for abs_line in lno_curvature_dict[diffraction_order]["clear_nu"]:
            valid_xs.extend(np.where((abs_line[0] < x) & (x < abs_line[1]))[0])
        
    

    x_mean = np.mean(x)
    x_centre = x - x_mean
    
    x_first_pixel = valid_xs[0] #the first pixel where the polynomial fit is made
    
    
    y_fit = np.polyval(np.polyfit(x_centre[valid_xs], y_selected[valid_xs], 4), x_centre)
    # y_fit = smooth_hr(y[valid_y, valid_xs], window_len=19)
    # y_fit = savitzky_golay(y[valid_y, valid_xs], 39, 2)
    # y_fit = fit_polynomial(x, y[valid_y, :], degree=2, indices=valid_xs)

    #for plotting the peak point - find pixel at the peak
    max_index = np.where(y_fit == np.max(y_fit[:150]))[0][0]
    
    #normalise curve to peak at 1.0 considering peak only in first 150 pixels
    y_fit_normalised = y_fit/np.max(y_fit[:150])
    
    #shift the pixel peak to account for temperature
    pixel_temperature_shift = np.polyval(lno_curvature_dict[diffraction_order]["temperature_shift_coeffs"], temperature) - max_index
    pixels_shifted = pixels + pixel_temperature_shift

    # print(max_index)
    # print(np.polyval(TEMPERATURE_CORRECTION_COEFFS, temperature))
    # print(pixel_temperature_shift)
    # print(max_index-pixel_temperature_shift)
    # stop()
    

    plt.title("Search pattern: %s" %regex.pattern)

    """"plot all"""
    # plt.plot(y_fit, color=colours[file_index], label=hdf5_filename[:15], alpha=0.4)
    # plt.scatter(max_index, y_fit[max_index], color=colours[file_index])

    """plot normalised and shifted for temperature"""
    plt.plot(pixels_shifted, y_fit_normalised, color=colours[file_index], label=hdf5_filename[:15], alpha=0.2)
    plt.scatter(max_index+pixel_temperature_shift, y_fit_normalised[max_index], color=colours[file_index])


    """make mean curvature spectrum from normalised + temperature adjusted curves: interpolate each normalised curve at set pixel points"""
    for interpolation_pixel in interpolation_pixels:
        y_interp = np.interp(interpolation_pixel, pixels_shifted, y_fit_normalised)
        curvature_dict[interpolation_pixel].append(y_interp)


    # plt.title(hdf5_filename)
    plt.xlabel("Pixel number")
    plt.ylabel("Reflectance factor")
    # plt.plot(x, y_selected)
    # plt.plot(x, y_fit, "k:")
    # plt.scatter(x[max_index], y_fit[max_index])


    
    
    variables["temperature"].append(temperature)
    variables["peak"].append(max_index)
    variables["peak_shifted"].append(max_index+pixel_temperature_shift)
    variables["colours"].append(colours[file_index])
    



mean_y_interps = []
"""make mean curvature"""
for interpolation_pixel in interpolation_pixels:
    #get mean fit 
    mean_y_interps.append(np.mean(curvature_dict[interpolation_pixel]))
    
mean_curve_coeffs = np.polyfit(interpolation_pixels, mean_y_interps, 4)
mean_curve = np.polyval(mean_curve_coeffs, pixels)
plt.plot(pixels, mean_curve, "k:", linewidth=3)


#find peak of mean_curve
peak_pixel = np.where(np.max(mean_curve[:150]) == mean_curve[:150])[0][0]
variables["mean_curve_peak_pixel"] = peak_pixel


plt.axvline(x=peak_pixel, color="k", linestyle=":")


plt.figure(figsize=(9,5))
plt.title("Search pattern: %s" %regex.pattern)
plt.xlabel("Instrument temperature")
plt.ylabel("Pixel position of polynomial peak (in first 150 pixels)")

plt.scatter(variables["temperature"], variables["peak"], color=variables["colours"])

coeffs = fit_polynomial(variables["temperature"], variables["peak"], coeffs=True)[1]
print(coeffs)
coeffs = fit_polynomial(variables["temperature"], variables["peak_shifted"], coeffs=True)[1]
print(coeffs)
plt.plot(variables["temperature"], fit_polynomial(variables["temperature"], variables["peak"]))

print(peak_pixel)
