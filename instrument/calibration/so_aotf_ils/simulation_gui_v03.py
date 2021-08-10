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


from instrument.calibration.so_aotf_ils.simulation_functions import get_file, get_data_from_file, select_data, fit_temperature
from instrument.calibration.so_aotf_ils.simulation_functions import get_start_params, make_param_dict, F_blaze, F_aotf, calc_spectrum, fit_spectrum

from instrument.calibration.so_aotf_ils.simulation_config import nu_range, AOTF_OFFSET_SHAPE



def s_value(slider):
    return slider.get()

def s_str(slider):
    return '{: .2f}'.format(slider.get())



regex = re.compile("20190416_020948_0p2a_SO_1_C")
index = 0

"""spectral grid and blaze functions of all orders"""

#get data, fit temperature and get starting function parameters
hdf5_file, hdf5_filename = get_file(regex)
d = get_data_from_file(hdf5_file, hdf5_filename)
d = select_data(d, index)
d = fit_temperature(d, hdf5_file)
d = get_start_params(d)
param_dict = make_param_dict(d)


#sigma for preferentially fitting of certain pixels
d["sigma"] = np.ones_like(d["spectrum_norm"])
smi = d["absorption_pixel"]
d["sigma"][smi-18:smi+19] = 0.01
# d["sigma"][:100] = 10.
# d["sigma"][280:] = 10.



print("m=", d["centre_order"])
print("t=", d["temperature"])
print("A= %0.1f kHz" %d["A"])




print("p0=", d["p0"])
print("p_width=", d["p_width"])
print("aotf_shift=", d["aotf_shift"])
print("blaze_shift=", d["aotf_shift"])
print("A_nu0=", d["A_nu0"])



# set up tKinter
root = tk.Tk()
root.geometry('1400x600')
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

pos_dict = {
    "blaze_centre":[0, 1],
    "blaze_width":[0, 2],
    "aotf_width":[0, 3],
    "aotf_shift":[0, 4],
    "sidelobe":[3, 1],
    "asymmetry":[3, 2],
    }
if AOTF_OFFSET_SHAPE == "Constant":
    pos_dict["offset"] = [3, 3]
else:
    pos_dict["offset_height"] = [3, 3]
    pos_dict["offset_width"] = [3, 4]








# set up tKinter
fig1 = plt.Figure(figsize=(6,5), dpi=100)
ax1 = fig1.add_subplot(111)
chart_type = FigureCanvasTkAgg(fig1, root)
chart_type.get_tk_widget().grid(column=0, columnspan=3, row=0)
ax1.set_title('AOTF and Blaze Shapes')   
ax1.set_ylim(0, 1.4)

fig2 = plt.Figure(figsize=(6,5), dpi=100)
ax2 = fig2.add_subplot(111)
chart_type = FigureCanvasTkAgg(fig2, root)
chart_type.get_tk_widget().grid(column=3, columnspan=3, row=0)
ax2.set_title('Simulation')   


#set up initial variables
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



variables = {}
for key, value in param_dict.items():
    variables[key] = value[0]




# for plotting
aotf_nu_range = np.arange(nu_range[0], nu_range[1], 0.1) #only for plotting

W_aotf = F_aotf(aotf_nu_range, variables, d)
line1a, = ax1.plot(aotf_nu_range, W_aotf)
W_blaze = F_blaze(variables, d)
line1b, = ax1.plot(d["nu_mp_centre"], W_blaze)


line2a, = ax2.plot(d["pixels"], d["spectrum_norm"], label="%s i=%i" %(hdf5_filename, index))
line2b, = ax2.plot(d["pixels"], np.zeros_like(d["pixels"]), label="Simulation")
line2c, = ax2.plot(d["pixels"], np.zeros_like(d["pixels"]), label="Sim No Solar Line")

ax2.legend()





def slider_changed(event):
    for key in slider_d.keys():
        slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
        slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
        slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]
        variables[key] = slider_d[key]["value"]

    W_aotf = F_aotf(aotf_nu_range, variables, d)
    line1a.set_ydata(W_aotf)

    W_blaze = F_blaze(variables, d)
    line1b.set_ydata(W_blaze)


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
    solar = calc_spectrum(variables, d)
    solar_slr = calc_spectrum(variables, d, I0=d["I0_lr_slr"])
    
    line2b.set_ydata(solar/max(solar))
    line2c.set_ydata(solar_slr/max(solar_slr))
    fig2.canvas.draw()
    fig2.canvas.flush_events()








def button_fit_f():
    # print(variables)
    fit_spectrum(param_dict, variables, d)
    
    solar_fit = calc_spectrum(variables, d)
    line2b.set_ydata(solar_fit/max(solar_fit))
    solar_fit_slr = calc_spectrum(variables, d, I0=d["I0_lr_slr"])
    line2c.set_ydata(solar_fit_slr/max(solar_fit_slr))



    for key in slider_d.keys():
        slider_d[key]["slider"].set(variables[key])
        slider_d[key]["value"] = variables[key]
        slider_d[key]["value_str"] = '{: .2f}'.format(variables[key])
        slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]

    W_aotf = F_aotf(aotf_nu_range, variables, d)
    line1a.set_ydata(W_aotf)

    W_blaze = F_blaze(variables, d)
    line1b.set_ydata(W_blaze)


    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()


button_update = ttk.Button(root, text="Simulate Response", command=button_simulate_f)
button_update.grid(column=3, row=5)
button_fit = ttk.Button(root, text="Fit", command=button_fit_f)
button_fit.grid(column=4, row=5)


root.mainloop()
