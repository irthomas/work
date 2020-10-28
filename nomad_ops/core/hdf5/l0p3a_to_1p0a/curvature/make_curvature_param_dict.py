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

from tools.file.read_write_hdf5 import write_hdf5_from_dict


FIG_X = 17
FIG_Y = 9


MAX_SZA = 30.

# SAVE_PDF = True
SAVE_PDF = False

lno_curvature_dict = {
168:{"clear_nu":[[3780., 3784.], [3785., 3796.], [3797., 3801.], [3802., 3810.]], "temperature_shift_coeffs":[-1.00061872, 88.19576276], },
189:{"clear_nu":[[4254., 4263.], [4265., 4267.], [4268., 4270.], [4272., 4274.], [4275.5, 4281.]], "temperature_shift_coeffs":[ -1.25805608, 112.75089597], },

# 189:{"clear_nu":[]},
}

diffraction_order = 189


reference_temperature = 0.0


if diffraction_order not in lno_curvature_dict.keys():
    print("Error: diffraction order not yet ready")
    sys.exit()

regex = re.compile("20.*_LNO_._DP_%i" %diffraction_order)
file_level = "hdf5_level_1p0a"


hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)


fig1 = plt.figure(figsize=(FIG_X, FIG_Y))
gs = fig1.add_gridspec(2,2)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig1.add_subplot(gs[0, 1])
ax4 = fig1.add_subplot(gs[1, 1])



ax1.set_title("Search pattern: %s" %regex.pattern)
ax2.set_title("Search pattern: %s (temperature correction)" %regex.pattern)

ax1.set_xlabel("Pixel number")
ax1.set_ylabel("Reflectance factor")

ax2.set_xlabel("Pixel number")
ax2.set_ylabel("Reflectance factor")

ax2.set_xlabel("Pixel number")
ax2.set_ylabel("Reflectance factor")

ax3.set_title("Search pattern: %s" %regex.pattern)
ax3.set_xlabel("Instrument temperature")
ax3.set_ylabel("Pixel position of polynomial peak (in first 150 pixels)")

ax4.set_title("Search pattern: %s (temperature correction)" %regex.pattern)
ax4.set_xlabel("Instrument temperature")
ax4.set_ylabel("Pixel position of polynomial peak (in first 150 pixels)")


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


reference_temperature_peak = np.polyval(lno_curvature_dict[diffraction_order]["temperature_shift_coeffs"], reference_temperature)


variables = {"temperature":[], "peak":[], "peak_shifted":[], "shift":[], "colours":[]}
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
        ax1.plot(x, y_selected)
        
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

    #normalise curve to peak at 1.0 considering peak only in first 150 pixels
    y_fit_normalised = y_fit/np.max(y_fit[:150])

    #for plotting the peak point - find pixel at the peak
    max_index = np.where(y_fit_normalised == np.max(y_fit_normalised[:150]))[0][0]

    
    #shift the pixel peak to account for temperature -> shift to reference temperature
    pixel_temperature_shift = np.polyval(lno_curvature_dict[diffraction_order]["temperature_shift_coeffs"], temperature) - reference_temperature_peak
    pixels_shifted = pixels - pixel_temperature_shift

    # print(max_index)
    # print(np.polyval(TEMPERATURE_CORRECTION_COEFFS, temperature))
    # print(pixel_temperature_shift)
    # print(max_index-pixel_temperature_shift)
    # stop()
    


    """"plot all"""
    # plt.plot(y_fit, color=colours[file_index], label=hdf5_filename[:15], alpha=0.4)
    # plt.scatter(max_index, y_fit[max_index], color=colours[file_index])

    ax1.plot(pixels, y_fit_normalised, color=colours[file_index], label=hdf5_filename[:15], alpha=0.2)
    ax1.scatter(max_index, y_fit_normalised[max_index], color=colours[file_index])

    """plot normalised and shifted for temperature"""
    ax2.plot(pixels_shifted, y_fit_normalised, color=colours[file_index], label=hdf5_filename[:15], alpha=0.2)
    ax2.scatter(max_index-pixel_temperature_shift, y_fit_normalised[max_index], color=colours[file_index])

    # print(max_index, pixel_temperature_shift, temperature, max_index-pixel_temperature_shift)


    """make mean curvature spectrum from normalised + temperature adjusted curves: interpolate each normalised curve at set pixel points"""
    for interpolation_pixel in interpolation_pixels:
        y_interp = np.interp(interpolation_pixel, pixels_shifted, y_fit_normalised)
        curvature_dict[interpolation_pixel].append(y_interp)



    
    
    variables["temperature"].append(temperature)
    variables["peak"].append(max_index)
    variables["shift"].append(pixel_temperature_shift)
    variables["peak_shifted"].append(max_index-pixel_temperature_shift)
    variables["colours"].append(colours[file_index])
    



mean_y_interps = []
"""make mean curvature"""
for interpolation_pixel in interpolation_pixels:
    #get mean fit 
    mean_y_interps.append(np.mean(curvature_dict[interpolation_pixel]))
    
mean_curve_coeffs = np.polyfit(interpolation_pixels, mean_y_interps, 4)
mean_curve = np.polyval(mean_curve_coeffs, pixels)
ax2.plot(pixels, mean_curve, "k:", linewidth=3)


variables["pixels"] = pixels
variables["mean_curve_coeffs"] = mean_curve_coeffs
variables["mean_curve"] = mean_curve

#find peak of mean_curve
# peak_pixel = np.where(np.max(mean_curve[:150]) == mean_curve[:150])[0][0]
variables["reference_temperature_peak"] = reference_temperature_peak
variables["reference_temperature"] = reference_temperature

variables["temperature_shift_coeffs"] = lno_curvature_dict[diffraction_order]["temperature_shift_coeffs"]
variables["clear_nu"] = lno_curvature_dict[diffraction_order]["clear_nu"]


ax2.axvline(x=reference_temperature_peak, color="k", linestyle=":")



ax3.scatter(variables["temperature"], variables["peak"], color=variables["colours"])

coeffs = fit_polynomial(variables["temperature"], variables["peak"], coeffs=True)[1]
print(coeffs)
coeffs = fit_polynomial(variables["temperature"], variables["peak_shifted"], coeffs=True)[1]
print(coeffs)
ax3.plot(variables["temperature"], fit_polynomial(variables["temperature"], variables["peak"]))


ax4.scatter(variables["temperature"], variables["peak_shifted"], color=variables["colours"])
ax4.plot(variables["temperature"], fit_polynomial(variables["temperature"], variables["peak_shifted"]))

ax4.axvline(x=reference_temperature, color="k", linestyle=":")
ax4.axhline(y=reference_temperature_peak, color="k", linestyle=":")


print(reference_temperature_peak)

ax1b = ax1.twinx()
ax1b.hist(variables["peak"], alpha=0.5, color="grey") 
ax2b = ax2.twinx()
ax2b.hist(variables["peak_shifted"], alpha=0.5, color="grey")


#save to hdf5 file
for key in variables.keys():
    variables[key] = np.asarray(variables[key])

write_hdf5_from_dict("lno_reflectance_factor_curvature_order_%i" %diffraction_order, variables, {}, {}, {})