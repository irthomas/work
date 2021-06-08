# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:51:10 2021

@author: iant

SIMULATION GUI
"""
import re
import numpy as np

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from instrument.nomad_so_instrument import F_aotf_goddard21


from instrument.calibration.so_aotf_ils.simulation_functions import get_file, get_data_from_file, select_data, fit_temperature, calc_blaze, aotf_conv

from instrument.calibration.so_aotf_ils.simulation_config import pixels


def s_value(slider):
    return slider.get()

def s_str(slider):
    return '{: .2f}'.format(slider.get())

regex = re.compile("20190416_020948_0p2a_SO_1_C")
index = 0

hdf5_file, hdf5_filename = get_file(regex)
d = get_data_from_file(hdf5_file, hdf5_filename)
d = select_data(d, index)
d = fit_temperature(d, hdf5_file)
d = calc_blaze(d)



root = tk.Tk()
root.geometry('1400x600')
root.resizable(False, False)
root.title('AOTF simulation')

root.columnconfigure((0, 3), weight=1)
root.columnconfigure((1, 4), weight=4)
root.columnconfigure((2, 5), weight=1)
root.rowconfigure(0, weight=8)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)


param_dict = {
    "sinc_width":[20.5, 15., 25.],
    "aotf_shift":[0.0, -100.0, 100.0],
    "sidelobe":[7.0, 0.001, 30.0],
    "asymmetry":[0.3, 0.001, 10.0],
    "offset":[0.01, 0.0, 0.5],
    }

pos_dict = {
    "sinc_width":[0, 1],
    "aotf_shift":[0, 2],
    "sidelobe":[0, 3],
    "asymmetry":[3, 1],
    "offset":[3, 2],
    }

nu_hr = np.arange(-150, 150, 0.1) + 4382


fig1 = plt.Figure(figsize=(6,5), dpi=100)
ax1 = fig1.add_subplot(111)
chart_type = FigureCanvasTkAgg(fig1, root)
chart_type.get_tk_widget().grid(column=0, columnspan=3, row=0)
ax1.set_title('AOTF shape')   


fig2 = plt.Figure(figsize=(6,5), dpi=100)
ax2 = fig2.add_subplot(111)
chart_type = FigureCanvasTkAgg(fig2, root)
chart_type.get_tk_widget().grid(column=3, columnspan=3, row=0)
ax2.set_title('Simulation')   


#set up initial variables
slider_d = {}
variables = {}
for key, value in param_dict.items():
    slider_d[key] = {}
    slider_d[key]["t_label"] = ttk.Label(root, text=key)
    slider_d[key]["v_label"] = ttk.Label(root, text=value[0])
    slider_d[key]["value_tk"] = tk.DoubleVar()

    slider_d[key]["value_tk"].set(value[0])

    slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
    slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
    slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]
    
    variables[key] = slider_d[key]["value"]
    # print(slider_d[key]["value"])

def aotf(variables):

    t = 0.0
    m = 0.0
    aotf_freq = 26666
    print(variables)
    W_aotf = F_aotf_goddard21(m, nu_hr, t, 
                              A=aotf_freq + variables["aotf_shift"],
                              wd=variables["sinc_width"],
                              sl=variables["sidelobe"],
                              af=variables["asymmetry"]) + variables["offset"]

    return W_aotf


W_aotf = aotf(variables)
line1, = ax1.plot(nu_hr, W_aotf)


line2a, = ax2.plot(pixels, d["spectrum_norm"], label="%s i=%i" %(hdf5_filename, index))
line2b, = ax2.plot(pixels, np.zeros_like(pixels), label="Simulation")

ax2.legend()




def slider_changed(event):
    for key in slider_d.keys():
        slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
        slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
        slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]
        variables[key] = slider_d[key]["value"]
    W_aotf = aotf(variables)
        
    line1.set_ydata(W_aotf)
    fig1.canvas.draw()
    fig1.canvas.flush_events()



for key, value in param_dict.items():
    slider_d[key]["slider"] = ttk.Scale(
            root,
            from_=value[1],
            to=value[2],
            orient='horizontal',
            variable=slider_d[key]["value_tk"],
            # command=slider_changed,
    )
    slider_d[key]["slider"].set(value[0])
    slider_d[key]["slider"].bind("<ButtonRelease-1>", slider_changed)

    slider_d[key]["t_label"].grid(column=pos_dict[key][0], row=pos_dict[key][1])
    slider_d[key]["slider"].grid(column=pos_dict[key][0]+1, row=pos_dict[key][1], sticky="EW")
    slider_d[key]["v_label"].grid(column=pos_dict[key][0]+2, row=pos_dict[key][1])


def button_simulate_f():
    print("Simulating AOTF")
    W_aotf_conv = aotf_conv(d, variables)
    
    print("Changing plot")
    line2b.set_ydata(W_aotf_conv)
    print(W_aotf_conv[200])
    fig2.canvas.draw()
    fig2.canvas.flush_events()

def button_fit_f():
    print("Fitting AOTF")
    W_aotf = aotf(variables)
    
    line2b.set_ydata(W_aotf)
    fig2.canvas.draw()
    fig2.canvas.flush_events()


button_update = ttk.Button(root, text="Simulate Response", command=button_simulate_f)
button_update.grid(column=3, row=3)
button_fit = ttk.Button(root, text="Fit", command=button_fit_f)
button_fit.grid(column=4, row=3)


root.mainloop()
