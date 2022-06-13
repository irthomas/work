# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:47:34 2022

@author: iant

GET SPECTRAL CALIBRATION FROM NOMAD DATA
"""

import matplotlib.pyplot as plt

import numpy as np




correction_dict = {
    136:{"px_range":[50, 310], "lines":[]}
}



fig, ax = plt.subplots(figsize=(10,5))

y_chosen = h5["Science/Y"][indices, :]

pixels = np.arange(px_range[0], px_range[1], 1)
n_spectra = y_all.shape[0]

#shift by 0.22cm-1 at start
x_shifted = x_old + 0.32


frame_index = get_nearest_index(chosen_alt, alts)
frame_indices = np.arange(frame_index-15, frame_index+16)

y = np.mean(y_all[frame_indices, px_range[0]:], axis=0)
y_cont = baseline_als(y)
y_cr = y / y_cont

# ax.plot(x, y_cr, label="X and Y from HDF5 file, mean of indices %i to %i" %(min(frame_indices), max(frame_indices)))
# ax.plot(x_shifted, y_cr, label="X+0.22cm-1 and Y from HDF5 file, mean of indices %i to %i" %(min(frame_indices), max(frame_indices)))
ax.plot(x_shifted, y_cr, label="CO spectrum at 30km")
ax.set_xlabel("Wavenumber cm-1")
ax.set_ylabel("Normalised Transmittance")
ax.set_title(hdf5_filename)

# spectral_lines_nu = spectral_lines_dict[order-1]
# order_delta = order_delta_nu(order, order-1, 0, 0.0)
# for spectral_line_nu in spectral_lines_nu:
#     ax.axvline(spectral_line_nu + order_delta, c="b", linestyle="--")

# spectral_lines_nu = spectral_lines_dict[order+1]
# order_delta = order_delta_nu(order, order+1, 0, 0.0)
# for spectral_line_nu in spectral_lines_nu:
#     ax.axvline(spectral_line_nu + order_delta, c="r", linestyle="--")

spectral_lines_nu = spectral_lines_dict[order]
for i, spectral_line_nu in enumerate(spectral_lines_nu):
    if i == 0:
        label = "Order 189 CO lines"
    else:
        label = ""
    ax.axvline(spectral_line_nu, c="k", linestyle="--", label=label)


absorption_points = np.where(y_cr < 0.985)[0]
all_local_minima = get_local_minima(y_cr)

local_minima = [i for i in all_local_minima if i in absorption_points]
# ax.scatter(x[local_minima], y_cr[local_minima], c="k", marker="+", label="Absorption Minima")

delta_nus = []
pixel_minima = []
nu_minima = []

for i, local_minimum in enumerate(local_minima):
    
    local_minimum_indices = np.arange(local_minimum-2, local_minimum+3, 1)
    
    x_hr, y_hr, x_min_position, chisq = fit_gaussian_absorption(x_shifted[local_minimum_indices], y_cr[local_minimum_indices], error=True)
    
    if i == 0:
        label = "Gaussian fit to absorption bands"
    else:
        label = ""
    # ax.plot(x_hr, y_hr, "r", label=label)

    closest_spectral_line = get_nearest_index(x_min_position, spectral_lines_nu)
    delta_nu = x_min_position - spectral_lines_nu[closest_spectral_line]
    if np.abs(delta_nu) > 0.5:
        print("Ignoring %0.1f line, too far from expected value" %x_min_position)
    else:
        print(delta_nu)
        delta_nus.append(delta_nu)
        #find pixel of minimum
        x_hr, y_hr, x_min_position, chisq = fit_gaussian_absorption(pixels[local_minimum_indices], y_cr[local_minimum_indices], error=True)
        pixel_minima.append(x_min_position)
        nu_minima.append(spectral_lines_nu[closest_spectral_line])
        
#remove bad points
delta_min_max = [np.mean(delta_nus) - np.std(delta_nus)*1.5, np.mean(delta_nus) + np.std(delta_nus)*1.5]
valid_indices = [i for i,v in enumerate(delta_nus) if v>delta_min_max[0] and v<delta_min_max[1]]

valid_pixel_minima = [v for i,v in enumerate(pixel_minima) if i in valid_indices]
valid_nu_minima = [v for i,v in enumerate(nu_minima) if i in valid_indices]

   

polyfit = np.polyfit(valid_pixel_minima, valid_nu_minima, 3)
    
x_new = np.polyval(polyfit, pixels)
x_new_all_px = np.polyval(polyfit, np.arange(320))


first_pixels = p0_nu_mp(order, x_new, pixels)
mean_first_pixel = np.mean(first_pixels)


