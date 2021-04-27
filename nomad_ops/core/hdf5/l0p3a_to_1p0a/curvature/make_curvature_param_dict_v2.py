# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:31:24 2020

@author: iant
ANALYSE LNO DATA AND DEFINE CURVE BASED ON 4 POINTS:
    PIXEL ON LEFT OF DETECTOR
    POSITION OF PEAK
    POSITION OF MINIMUM
    PIXEL ON RIGHT OF DETECTOR
    
    ALL ARE TEMPERATURE DEPENDENT
    
FROM THESE 4 POINTS CONSTRUCT A CUBIC CURVE FOR EACH TEMPERATURE AND ORDER

"""

# import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from datetime import datetime

from tools.file.hdf5_functions import make_filelist
from tools.spectra.fit_polynomial import fit_polynomial_errors
from tools.spectra.baseline_als import baseline_als
# from tools.spectra.smooth_hr import smooth_hr
from tools.spectra.savitzky_golay import savitzky_golay
from tools.plotting.colours import get_colours

# from tools.file.read_write_hdf5 import write_hdf5_from_dict


from nomad_ops.core.hdf5.l0p3a_to_1p0a.curvature.curvature_functions import make_correction_curve


FIG_X = 17
FIG_Y = 9


MAX_SZA = 30.

POLYFIT_DEGREE = 3 #must be 3 for 4 points


lno_curvature_dict = {



167:{"clear_nu":[[3752., 3756.], [3757., 3759.], [3760.5, 3765.], [3766.5, 3769.], [3771., 3779.], [3780., 3782.], [3783., 3785.]], "px_left":40, "px_right":310},
168:{"clear_nu":[[3780., 3784.], [3785., 3796.], [3797., 3801.], [3802., 3810.]], "px_left":40, "px_right":310},
169:{"clear_nu":[[3790., 3799.], [3799.5, 3801.], [3802., 3802.5], [3803.5, 3804.],  [3805., 3805.5], [3807.5, 3812.], [3813., 3815.], [3817., 3819.5], [3822., 3823.5], [3827., 3829.]], "px_left":40, "px_right":310},

186:{"clear_nu":[[4179., 4189.], [4191., 4191.5], [4196., 4199.], [4200.5, 4204.], [4205., 4213.]], "px_left":40, "px_right":290},
187:{"clear_nu":[[4200., 4208.], [4210., 4213.], [4215., 4218.], [4219., 4222.], [4223.5, 4226.], [4228., 4231.], [4233., 4235.5]], "px_left":40, "px_right":290},
188:{"clear_nu":[[4224., 4235.5], [4236.5, 4239.5], [4242.5, 4244.], [4245., 4247.5], [4249., 4250.5], [4253., 4254.], [4255., 4258.]], "px_left":40, "px_right":290},
   
  
189:{"clear_nu":[[4250., 4263.], [4265., 4267.], [4268., 4270.], [4272., 4274.], [4275.5, 4283.]], "px_left":40, "px_right":310}, #this one for 26th Nov analysis
190:{"clear_nu":[[4265., 4278.], [4279., 4280.], [4282.5, 4284.5], [4285.8, 4287.8], [4289., 4291.], [4292., 4294.], [4295.5, 4296.5], [4299.5, 4300.], [4301.5, 4303.]], "px_left":40, "px_right":290},
191:{"clear_nu":[[4294., 4303.], [4304., 4306.], [4307., 4308.5], [4310., 4311.5], [4312.4, 4314.], [4315., 4316.5], [4318., 4323.], [4324., 4326.]], "px_left":40, "px_right":290},

193:{"clear_nu":[[4335., 4371.]], "px_left":40, "px_right":290},
194:{"clear_nu":[[4355., 4382.], [4384., 4393.]], "px_left":40, "px_right":290},
196:{"clear_nu":[[4403., 4439.]], "px_left":40, "px_right":290},



}
# diffraction_order = 134
# diffraction_order = 136


# diffraction_order = 167
# diffraction_order = 168
# diffraction_order = 169
# diffraction_order = 170


# diffraction_order = 186 #not enough data
# diffraction_order = 187
# diffraction_order = 188 #not enough data

# diffraction_order = 189
# diffraction_order = 190
# diffraction_order = 191
# diffraction_order = 192
# diffraction_order = 193

# diffraction_order = 194
diffraction_order = 195
# diffraction_order = 196



if diffraction_order not in lno_curvature_dict.keys():
    lno_curvature_dict[diffraction_order] = {"clear_nu":[]}



def find_nearest_index(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx




if diffraction_order not in lno_curvature_dict.keys():
    print("Error: diffraction order not yet ready")
    sys.exit()

regex = re.compile("20.*_LNO_._D._%i" %diffraction_order)
# regex = re.compile("20200.*_LNO_._DF_%i" %diffraction_order)
file_level = "hdf5_level_1p0a"


hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)



#if absorption lines not yet defined, plot raw data and stop
if len(lno_curvature_dict[diffraction_order]["clear_nu"]) == 0:
    PLOT_RAW = True
    print("Plotting raw spectra only")
else:
    PLOT_RAW = False


fig3, axes3 = plt.subplots(ncols=2, figsize=(FIG_X, FIG_Y))
fig2, axes2 = plt.subplots(nrows=4, ncols=2, figsize=(FIG_X, FIG_Y))
fig1, axes1 = plt.subplots(ncols=2, figsize=(FIG_X, FIG_Y))
    


#first remove low sza files
chosen_hdf5_files = []
chosen_hdf5_filenames = []

for file_index, (hdf5_filename, hdf5_file) in enumerate(zip(hdf5_filenames, hdf5_files)):
    
    sza1 = np.mean(hdf5_file["Geometry/Point0/IncidenceAngle"][...], axis=1)
    
    valid_ys1 = np.where(sza1 < MAX_SZA)[0]
    if len(valid_ys1) == 0:
        continue
    
    chosen_hdf5_files.append(hdf5_file)
    chosen_hdf5_filenames.append(hdf5_filename)


colours = get_colours(len(chosen_hdf5_filenames))



variables = {"temperatures":[], "hdf5_filename":[], "y_fit":[], "y":[], "point_xys":[], "colours":[]}
#loop through low SZA files
for file_index, (hdf5_filename, hdf5_file) in enumerate(zip(chosen_hdf5_filenames, chosen_hdf5_files)):

    pixels = np.arange(320.)

    sza = np.mean(hdf5_file["Geometry/Point0/IncidenceAngle"][...], axis=1)
    valid_ys = np.where(sza < MAX_SZA)[0]
    
    
    
    y = hdf5_file["Science/YReflectanceFactor"][...]
    x = hdf5_file["Science/X"][...]
    temperature = float(hdf5_file["Channel/MeasurementTemperature"][0][0])

    if len(valid_ys) > 5:
        y_selected = np.mean(y[valid_ys, :], axis=0)
    else:
        #get spectra closest to min solar incidence angle
        valid_ys = np.where(sza == np.min(sza))[0]
        y_selected = np.mean(y[valid_ys[0]-8:valid_ys[0]+9, :], axis=0)

    
    
    if PLOT_RAW:
        axes1[0].plot(x, y_selected)
        poly_fit = np.polyfit(x, y_selected, POLYFIT_DEGREE)
        axes1[1].plot(x, y_selected/np.polyval(poly_fit, x), alpha=0.3)
        
        #plot first 100 only
        if file_index == 100 or file_index == len(chosen_hdf5_filenames)-1:
            axes1[1].set_ylim((0.5, 1.5))
            plt.show()
            input("Pausing")
    
    else:

        #find and remove atmospheric line indices
        valid_xs = []
        for abs_line in lno_curvature_dict[diffraction_order]["clear_nu"]:
            #get pixels not containing absorption line or too close to edges of detector
            valid_xs.extend(np.where((abs_line[0] < x) & (x < abs_line[1]))[0])
        

        #get error
        err = np.abs(y_selected[valid_xs]/np.polyval(np.polyfit(pixels[valid_xs], y_selected[valid_xs], 8), pixels[valid_xs])-1.0)
        als = baseline_als(err, lam=500.) * 0.66
        y_err = np.polyval(np.polyfit(pixels[valid_xs], als, 9), pixels)
        # y_err[valid_xs[-1]:320] = als[-1] #extrapolate to end of detector
        y_err[y_err < 0.0] = np.mean(err) #remove negatives
    
        # plt.figure()
        # # plt.plot(pixels[valid_xs], y_selected[valid_xs])
        # plt.plot(pixels[valid_xs], err, "g")
        # plt.plot(pixels[valid_xs], als, "b")
        # plt.plot(pixels, y_err, "b--")
        # stop()
    
        x = pixels
        #x relative to centre
        x_mean = np.mean(x)
        x_centre = x - x_mean
    
        #don't include error in fit
        # poly_fit = np.polyfit(x_centre[valid_xs], y_selected[valid_xs], POLYFIT_DEGREE)
        #include error in fit
        _, poly_fit = fit_polynomial_errors(x_centre[valid_xs], y_selected[valid_xs], y_err[valid_xs], degree=POLYFIT_DEGREE, coeffs=True)
        
        #make poly fit curve
        y_fit = np.polyval(poly_fit, x_centre)
    
        #plot time as colour
        colour = colours[file_index]
        
        # #plot temperature as colour
        # colour_grid = np.linspace(-12.0, 3.0, num=len(colours))
        # colour = colours[find_nearest_index(colour_grid, temperature)]
        
    
        #normalise curve to peak at 1.0 considering peak only in first 150 pixels
        y_fit_normalised = y_fit/np.max(y_fit[:150])
        als = baseline_als(y_selected, lam=500.)
        sg = savitzky_golay(y_selected, 59, 2, deriv=0)
        
        # plt.figure()
        # plt.plot(x_centre, y_selected)
        # plt.plot(x_centre, y_fit)
        # plt.plot(x_centre, als)
        # plt.plot(x_centre, sg)
        # stop()
        
        #find x,y positions of given pixels
        left_px = lno_curvature_dict[diffraction_order]["px_left"]
        left_y_px = y_fit_normalised[lno_curvature_dict[diffraction_order]["px_left"]]
        
        right_px = lno_curvature_dict[diffraction_order]["px_right"]
        right_y_px = y_fit_normalised[lno_curvature_dict[diffraction_order]["px_right"]]
    
        #find pixel number at the peak
        peak_px = np.where(y_fit_normalised == np.max(y_fit_normalised[:150]))[0][0]
        peak_y_px = y_fit_normalised[peak_px]
    
        #find pixel number at the trough
        trough_px = np.where(y_fit_normalised == np.min(y_fit_normalised[150:]))[0][0]
        trough_y_px = y_fit_normalised[trough_px]
        
        point_xy = np.zeros((4, 2))
        point_xy[:] = [[left_px, left_y_px], [peak_px, peak_y_px], [trough_px, trough_y_px], [right_px, right_y_px]]
        
        point_names = ["x left", "y left", "x peak", "y peak", "x trough", "y trough", "x_right", "y_right"]
    
    
    
    
        #if peak is not in first 20 pixels or all negative
        if peak_px > 20 and np.max(y_selected) > 0:
            
            variables["hdf5_filename"].append(hdf5_filename)
            variables["temperatures"].append(temperature)
            variables["y_fit"].append(y_fit)
            variables["y"].append(y_selected)
            variables["colours"].append(colour)
            variables["point_xys"].append(point_xy)
            

temperatures = variables["temperatures"] = np.asfarray(variables["temperatures"])
colours = variables["colours"] = np.asfarray(variables["colours"])
points = variables["point_xys"] = np.asfarray(variables["point_xys"])
y_fit = variables["y_fit"] = np.asfarray(variables["y_fit"])
y = variables["y"] = np.asfarray(variables["y"])




#test initial polynomial fit on some spectra

for file_index in range(0, len(temperatures), int(len(temperatures)/10.)):
    
    temperature = temperatures[file_index]
    axes3[0].plot(pixels, y[file_index, :] + file_index/100, color=colours[file_index], label="%0.1f" %temperature)
    axes3[0].plot(pixels, y_fit[file_index, :] + file_index/100, "--", color=colours[file_index])
    axes3[1].plot(pixels, y[file_index, :]/y_fit[file_index, :] + file_index/50, color=colours[file_index])
    
axes3[1].set_ylim((-1,file_index/50 + 2))

axes3[0].legend()
# stop()






#plot x,y, coefficient terms in subplots
coeffs = np.zeros((4, 2, 2))

for i in range(len(points[0, :, :].flatten())):
    axes2.flatten()[i].set_title(point_names[i])
    axes2.flatten()[i].scatter(temperatures, points.reshape((len(temperatures), 8))[:, i], color=colours)

    point_coeffs = np.polyfit(temperatures, points.reshape((len(temperatures), 8))[:, i], 1)
    mean_curve = np.polyval(point_coeffs, temperatures)

    axes2.flatten()[i].plot(temperatures, mean_curve)

    coeffs.reshape(8, 2)[i, :] = point_coeffs

variables["coeffs"] = coeffs

#print coeffs for inclusion in dictionary
h = "%i:{'coeffs':np.array([" %diffraction_order
for i in range(4):
    h += "["
    for j in range(2):
        h+= "[%0.9g, %0.9g]," %(coeffs[i, j, 0], coeffs[i, j, 1])
    h += "],"
h += "])},"
print(h)



#find bad fit points
# a = points[:, 3, 1]
# b = a.argmax()
# plt.figure()
# plt.plot(y[b, :])


#test polynomial fit on some spectra

for file_index in range(0, len(temperatures), int(len(temperatures)/10.)):
    
    temperature = temperatures[file_index]
    curve = make_correction_curve(temperature, coeffs, pixels)

    #plot points
    curve_points = np.zeros((4, 2))
    for i in range(4):
        for j in range(2):
            curve_points[i, j] = np.polyval(coeffs[i, j, :], temperature)
    axes1[0].scatter(curve_points[:, 0], curve_points[:, 1]* np.max(y_fit[file_index, :150]), color=colours[file_index])
    
    
    # als = baseline_als(y[file_index, :], lam=500.)
    # sg = savitzky_golay(y[file_index, :], 59, 2)
    
    axes1[0].plot(pixels, y[file_index, :], color=colours[file_index], label="Y %s" %variables["hdf5_filename"][file_index])
    axes1[0].plot(pixels, y_fit[file_index, :], color=colours[file_index])
    # axes1[0].plot(pixels, als, linestyle=":", color=colours[file_index])
    # axes1[0].plot(pixels, sg, linestyle=":", color=colours[file_index])

    axes1[0].plot(pixels, curve * np.max(y_fit[file_index, :150]), linestyle="--", color=colours[file_index], label="Curve")

    axes1[1].plot(pixels, y[file_index, :] / curve, color=colours[file_index])
    axes1[0].set_ylim((0.0, 0.6))
    axes1[1].set_ylim((0.0, 0.6))

axes1[0].legend()

#delete extra datasets and save rest to h5 file
# del variables["y_fit"]
# del variables["y"]
# del variables["hdf5_filename"]

# write_hdf5_from_dict("lno_reflectance_factor_curvature_order_%i" %diffraction_order, variables, {"Creation time":str(datetime.now())[:-7]}, {}, {})