# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 10:15:32 2021

@author: iant
"""


import re
import numpy as np
# import lmfit


import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# from instrument.nomad_so_instrument import nu_mp
# from tools.asimut.ils_params import get_ils_params

from instrument.calibration.so_aotf_ils.simulation_functions_v02 import (get_cal_params, get_file, get_data_from_file, 
     select_data, fit_temperature, make_param_dict, calc_spectrum, fit_spectrum, get_solar_spectrum, get_all_x, find_absorption_minimum)




"""choose a solar line"""
line = 4383.5
# line = 4276.1
# line = 3787.9


if line == 4383.5:
    regex = re.compile("20190416_020948_0p2a_SO_1_C") #choose a file
    index = 0
    # index = 80 #choose a spectrum in the file (where 0 = 1st spectrum)

if line == 4276.1:
    regex = re.compile("20180716_000706_0p2a_SO_1_C")
    index = 72



"""choose which parameters to fit - note: blaze changes will slow down the fitting a lot!"""
#AOTF: centre wavenumber; width; sidelobe; amplitude; gauss width
chosen_params = ["aotff", "aotfw", "aotfs", "aotfa", "aotfg", "aotfgw"]

#as above, plus blaze shift
# chosen_params = ["aotff", "aotfw", "aotfs", "aotfa", "aotfg", "aotfgw", "blaze_shift"]

#AOTF centre only
# chosen_params = ["aotff"]


"""choose a solar spectrum (ACE has no solar lines for orders>195)"""
# solar_spectrum = "ACE"
solar_spectrum = "PFS"




"""get data from the file
    note that you will need to modify the tools.file.paths file to specify the correct directory
    or open the h5 file yourself 
        e.g. hdf5_file = h5py.File(<path>, "r")
        hdf5_filename = os.path.basename(<path>)"""
        
hdf5_file, hdf5_filename = get_file(regex)


#make dictionary containing all important data
d = {}
d["line"] = line
d = get_data_from_file(hdf5_file, hdf5_filename, d)
d["solar_spectrum"] = solar_spectrum
d = get_solar_spectrum(d, plot=False)


d = get_all_x(hdf5_file, d)
d = select_data(d, index)
# d = fit_temperature(d, hdf5_file)
d = find_absorption_minimum(d)
# d = get_start_params(d)
# param_dict = make_param_dict(d)


"""spectral grid and blaze functions of all orders"""

t = d["temperature"]
d = get_cal_params(d, t, t)
# d = get_ils_params(d)
# d = blaze_conv(d)

param_dict = make_param_dict(d, chosen_params)



#sigma for preferentially fitting of certain pixels: smi = solar minimum index
d["sigma"] = np.ones_like(d["spectrum_norm"])
smi = d["absorption_pixel"]
d["sigma"][smi-18:smi+19] = 0.01 #give 100x weighting to pixels +-18 pixels from the centre of the solar line absorption
# d["sigma"][:100] = 10. #deweight edge of detector by 0.1x
# d["sigma"][280:] = 10.




print("m=", d["centre_order"])
print("temperature=", d["temperature"])
# print("t_calc=", d["t_calc"])
print("A= %0.1f kHz" %d["A"])




# print("p0=", d["p0"])
# print("p_width=", d["p_width"])
# print("aotf_shift=", d["aotf_shift"])
# print("blaze_shift=", d["aotf_shift"])
# print("A_nu0=", d["A_nu0"])



# set up tKinter
root = tk.Tk()
root.geometry('1900x600')
root.resizable(False, False)
root.title('AOTF Simulation')

root.columnconfigure((0, 3), weight=1)
root.columnconfigure((1, 4), weight=4)
root.columnconfigure((2, 5), weight=1)
root.rowconfigure(0, weight=8)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=1)



#only for placing sliders in the grid correctly
pos_dict = {}
for i, param in enumerate(chosen_params):
    #[0 or 3, 1 to 4]
    pos_dict[param] = [int(np.floor(i/5)*3), np.mod(i, 5)+1]








# set up tKinter
fig1 = plt.Figure(figsize=(10,5), dpi=100, constrained_layout=True)
ax1 = fig1.add_subplot(111)
chart_type = FigureCanvasTkAgg(fig1, root)
chart_type.get_tk_widget().grid(column=0, columnspan=3, row=0)
ax1.set_title('AOTF and Blaze Shapes')   
ax1.set_ylim(0, 1.4)

fig2 = plt.Figure(figsize=(10,5), dpi=100, constrained_layout=True)
ax2 = fig2.add_subplot(111)
chart_type = FigureCanvasTkAgg(fig2, root)
chart_type.get_tk_widget().grid(column=3, columnspan=3, row=0)
ax2.set_title('Simulation')   


#set up initial variables and sliders
def s_value(slider):
    return slider.get()

def s_str(slider):
    return '{: .2f}'.format(slider.get())



slider_d = {}
for key, value in param_dict.items():
    slider_d[key] = {}
    slider_d[key]["t_label"] = ttk.Label(root, text=key)
    slider_d[key]["v_label"] = ttk.Label(root, text=value[0])
    slider_d[key]["value_tk"] = tk.DoubleVar()

    slider_d[key]["value_tk"].set(value[0])

    slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
    slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
    slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]





#make the lines on the figures
W_aotf = d["F_aotf"]

F_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan
nu_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan


solar_fit = calc_spectrum(d, d["I0_solar_hr"])
solar_fit_slr = calc_spectrum(d, d["I0_solar_slr"])


line1a, = ax1.plot(d["nu_hr"], W_aotf)
line1b, = ax1.plot(d["nu_blazes"], d["F_blazes"])
line1c, = ax1.plot(d["nu_hr"], (d["I0_lr"]-min(d["I0_lr"]))/(max(d["I0_lr"]) - min(d["I0_lr"])))


line2a, = ax2.plot(d["pixels"], d["spectrum_norm"], label="%s i=%i" %(hdf5_filename, index))
line2b, = ax2.plot(d["pixels"], solar_fit/max(solar_fit), label="Simulation")
line2c, = ax2.plot(d["pixels"], solar_fit_slr/max(solar_fit), label="Sim No Solar Line")

ax2.legend()





def slider_changed(event):
    #update left hand figure with changed parameters
    
    variables = {}
    for key in slider_d.keys():
        slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
        slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
        slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]
        variables[key] = slider_d[key]["value"]
        
    get_cal_params(d, t, t, pc=variables)

    W_aotf = d["F_aotf"]
    line1a.set_ydata(W_aotf)

    line1b.set_ydata(d["F_blazes"])

    fig1.canvas.draw()
    fig1.canvas.flush_events()





for key, value in param_dict.items():
    slider_d[key]["slider"] = ttk.Scale(
            root,
            from_ = value[1],
            to = value[2],
            orient = "horizontal",
            variable = slider_d[key]["value_tk"],
            # command=slider_changed,
    )
    
    slider_d[key]["slider"].set(value[0])
    slider_d[key]["slider"].bind("<ButtonRelease-1>", slider_changed)

    slider_d[key]["t_label"].grid(column=pos_dict[key][0], row=pos_dict[key][1])
    slider_d[key]["slider"].grid(column=pos_dict[key][0]+1, row=pos_dict[key][1], sticky="EW")
    slider_d[key]["v_label"].grid(column=pos_dict[key][0]+2, row=pos_dict[key][1])






def button_simulate_f():
    #when the simulate button is pressed, change the right hand side figure
    solar = calc_spectrum(d, d["I0_solar_hr"])
    solar_slr = calc_spectrum(d, d["I0_solar_slr"])
    
    line2b.set_ydata(solar/max(solar))
    line2c.set_ydata(solar_slr/max(solar))
    fig2.canvas.draw()
    fig2.canvas.flush_events()








def button_fit_f():
    #when the fit button is pressed

    fit_spectrum(param_dict, d)
    
    solar_fit = calc_spectrum(d, d["I0_solar_hr"])
    line2b.set_ydata(solar_fit/max(solar_fit))
    solar_fit_slr = calc_spectrum(d, d["I0_solar_slr"])
    line2c.set_ydata(solar_fit_slr/max(solar_fit))



    for key in slider_d.keys():
        slider_d[key]["slider"].set(d[key])
        slider_d[key]["value"] = d[key]
        slider_d[key]["value_str"] = '{: .2f}'.format(d[key])
        slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]

    line1a.set_ydata(d["F_aotf"])
    line1b.set_ydata(d["F_blazes"])


    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()


#add the simulate and fit buttons
button_update = ttk.Button(root, text="Simulate Response", command=button_simulate_f)
button_update.grid(column=3, row=5)
button_fit = ttk.Button(root, text="Fit", command=button_fit_f)
button_fit.grid(column=4, row=5)


#run the gui
root.mainloop()
