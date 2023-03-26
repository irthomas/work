# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:06:09 2023

@author: iant

LNO SPECTRAL CALIRATION PART 2 - READ IN FULLSCAN ANALYSIS RESULTS AND OUTPUT SPECTRAL AND TEMPERATURE COEFFICIENTS

spectral calibration is of the form
v / m = pxt^2 * A + pxt * B + C

pxt = px + (t * D) where px = the pixel number, t is the temperature and A, B, C and D are the coefficients to fit

First, find D: Take data for each order/line combination at a range of temperatures, the gradient is D

Then interpolate to find the zero celsius value i.e where pxt = px for each order/line combination
Then polyfit v/m versus the pxt pixels calculated
"""




import numpy as np
import matplotlib.pyplot as plt

filename = r"output\lno_spectral_calibration\LNO_fullscan_spectral_analysis3.txt" #3rd version i.e. after the 2nd iteration

input_dict = {"orders":[], "ts":[], "nus":[], "pxs":[], "nu_orders":[], "delta_nus":[]}

with open(filename, "r") as f:
    lines = f.readlines()
    
    for line in lines[1:]:
        line_split = line.split("\t")
    
        input_dict["orders"].append(float(line_split[1]))
        input_dict["ts"].append(float(line_split[2]))
        input_dict["nus"].append(float(line_split[3]))
        input_dict["pxs"].append(float(line_split[4]))
        input_dict["delta_nus"].append(float(line_split[5]))
        
        input_dict["nu_orders"].append(input_dict["nus"][-1]/input_dict["orders"][-1])
        


mean_delta_nu = np.mean(input_dict["delta_nus"])
print("mean delta nu = %0.3f" %mean_delta_nu)
std_delta_nu = np.std(input_dict["delta_nus"])
print("standard deviation delta nu = %0.3f" %std_delta_nu)



spectrum_px = np.arange(320)

unique_nu_orders = list(set(input_dict["nu_orders"]))


gradient_dict = {"orders":[], "nu_orders":[], "gradients":[], "px_zero_celsius":[]}

plt.figure(figsize=(12, 6), constrained_layout=True)
plt.grid()
plt.xlabel("Pixel number of absorption")
plt.ylabel("Instrument temperature C")
plt.title("LNO occultation fullscan: absorption shifts with temperature")
for unique_nu_order in unique_nu_orders:
    ixs = [i for i, v in enumerate(input_dict["nu_orders"]) if v == unique_nu_order]

    # print(len(ixs))
    if len(ixs) > 4:
        chosen_pxs = [input_dict["pxs"][i] for i in ixs]
        chosen_ts = [input_dict["ts"][i] for i in ixs]
        
        coeffs = np.polyfit(chosen_pxs, chosen_ts, 1)
        fit_ts = np.polyval(coeffs, chosen_pxs)
        
        "t = coeffs[0] * px + coeffs[1]"
        px_zero_celsius = -coeffs[1]/coeffs[0]
        
        # plt.scatter([i - np.mean(chosen_pxs) for i in chosen_pxs], chosen_ts)
        # plt.plot([i - np.mean(chosen_pxs) for i in chosen_pxs], fit_ts)
        plt.scatter(chosen_pxs, chosen_ts)
        plt.plot(chosen_pxs, fit_ts)
        
        gradient_dict["orders"].append(input_dict["orders"][ixs[0]])
        gradient_dict["nu_orders"].append(unique_nu_order)
        gradient_dict["gradients"].append(-1./coeffs[0])
        gradient_dict["px_zero_celsius"].append(px_zero_celsius)
        
plt.savefig("LNO_occultation_fullscan_absorption_shifts_with_temperature.png")
        
        # stop()
plt.figure(figsize=(12, 6), constrained_layout=True)
plt.grid()
plt.xlabel("Diffraction order")
plt.ylabel("D coefficient")
plt.title("LNO occultation fullscan: pixel shift per degree temperature change")
plt.scatter(gradient_dict["orders"], gradient_dict["gradients"])
gradient_coeffs = np.polyfit(gradient_dict["orders"], gradient_dict["gradients"], 1)
gradient_fits = np.polyval(gradient_coeffs, gradient_dict["orders"])
plt.plot(gradient_dict["orders"], gradient_fits)

plt.savefig("LNO_occultation_fullscan_pixel_shift_per_degree.png")

mean_gradient = np.mean(gradient_fits)

print("D = ", mean_gradient)

plt.figure(figsize=(12, 6), constrained_layout=True)
plt.grid()
plt.xlabel("Pixel number of absorption at 0 degrees C")
plt.ylabel("Absorption wavenumber / diffraction order")
plt.title("LNO occultation fullscan: pixel to wavenumber calibration")
plt.scatter(gradient_dict["px_zero_celsius"], gradient_dict["nu_orders"])
px_nu_coeffs = np.polyfit(gradient_dict["px_zero_celsius"], gradient_dict["nu_orders"], 2)
fit_px_nu_order = np.polyval(px_nu_coeffs, gradient_dict["px_zero_celsius"])
plt.plot(gradient_dict["px_zero_celsius"], fit_px_nu_order)

plt.savefig("LNO_occultation_fullscan_pixel_to_wavenumber_calibration.png")


print("A = ", px_nu_coeffs[0])
print("B = ", px_nu_coeffs[1])
print("C = ", px_nu_coeffs[2])


def it23_waven(diffraction_order, temperature):
    """LNO spectral calibration Ian January 22"""
    A =  3.942616548909985e-09
    B =  0.0005612789704190822
    C =  22.471896082567795
    D =  -0.851152813531944

    cfpixel = [A, B, C]

    p0_nu = D * temperature #px/°C * instrument temperature
    px_t_offset = np.arange(320.) + p0_nu
    nu  = np.polyval(cfpixel, px_t_offset) * diffraction_order
    
    return nu #wavenumbers of pixels


# A = 3.32e-8
# B = 5.480e-4
# C = 22.4701
# D = -0.8276


# for i in range(len(orders)):
    
    

#     nu_calc = it23_waven(order, inter_temp, px_in, A, B, C, D)

#     delta_nu = np.abs(nu_calc - nu)

