# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:52:35 2021

@author: iant

SIMULATION GUI 2
"""

import numpy as np

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from instrument.nomad_so_instrument import F_aotf_goddard21


def s_value(slider):
    return slider.get()

def s_str(slider):
    return '{: .2f}'.format(slider.get())


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
    "asymmetry":[0.3, 0.001, 50.0],
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


#set up initial variables
slider_d = {}
variables = {}
for key, value in param_dict.items():
    slider_d[key] = {}
    
    slider_d[key]["t_label"] = ttk.Label(root, text=key)
    slider_d[key]["v_label"] = ttk.Label(root, text=value[0])
    slider_d[key]["value_tk"] = tk.DoubleVar()


    slider_d[key]["slider"] = ttk.Scale(
            root,
            from_=value[1],
            to=value[2],
            orient='horizontal',
            variable=slider_d[key]["value_tk"],
    )

    slider_d[key]["slider"].set(value[0])
    slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
    slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
    slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]

    slider_d[key]["t_label"].grid(column=pos_dict[key][0], row=pos_dict[key][1])
    slider_d[key]["slider"].grid(column=pos_dict[key][0]+1, row=pos_dict[key][1], sticky="EW")
    slider_d[key]["v_label"].grid(column=pos_dict[key][0]+2, row=pos_dict[key][1])

    
    variables[key] = slider_d[key]["value"]


def make_aotf(variables):

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


W_aotf = make_aotf(variables)
line, = ax1.plot(nu_hr, W_aotf)






def button_update_f():
    print("updating")
    for key in slider_d.keys():
        slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
        slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
        slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]
        
        variables[key] = slider_d[key]["value"]
    W_aotf = make_aotf(variables)
    
    # print(W_aotf[1500])
    line.set_ydata(W_aotf)
    fig1.canvas.draw()
    fig1.canvas.flush_events()


button_update = ttk.Button(root, text="Update", command=button_update_f)
button_update.grid(column=3, row=3)



# slider0_label = ttk.Label(root, text="Sinc_width")
# slider0_label.grid(column=0, row=1)
# slider0_value = ttk.Label(root, text=slider_values[0].get())
# slider0_value.grid(column=2, row=1)
# slider0 = ttk.Scale(
#     root,
#     from_=0,
#     to=100,
#     orient='horizontal',
#     variable=slider_values[0],
#     command=slider_changed,
# )
# slider0.grid(column=1, row=1, ipadx=100, sticky="EW")


root.mainloop()
