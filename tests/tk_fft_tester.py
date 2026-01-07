# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:50:01 2023

@author: iant

QT APP FOR 1-D SPECTRUM FFT EDITING

"""


import numpy as np
# import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# data = np.sin(np.arange(0, 31, 0.1)) + np.sin(np.arange(0, 62, 0.2))
# data = np.loadtxt("raw_aotf_20191002_000902-186-8.tsv")
data = np.loadtxt("spectrum.txt")


fft = np.fft.fft(data)
original_fft = np.copy(fft)

ifft = np.fft.ifft(fft)

points = np.arange(30)
n_cols = len(points)


data_min = np.min([np.min(fft.real[1:-1]), np.min(fft.imag[1:-1])])
data_max = np.max([np.max(fft.real[1:-1]), np.max(fft.imag[1:-1])])

data_min_max = np.max([np.abs(data_min), np.abs(data_max)])
# data_min_max = 1e4

# set up tKinter
root = tk.Tk()
root.geometry('1900x900')
root.resizable(False, False)
root.title('FFT simulation')

for i in range(n_cols):
    root.columnconfigure((i, 0), weight=1)
root.rowconfigure(0, weight=8)  # plots
root.rowconfigure(1, weight=1)  # label
root.rowconfigure(2, weight=1)  # slider real
root.rowconfigure(3, weight=1)  # value real
root.rowconfigure(4, weight=1)  # slider imag
root.rowconfigure(5, weight=1)  # value imag


# only for placing sliders in the grid correctly
pos_dict = {}
for i, param in enumerate(points):
    pos_dict[param] = [i, 1]


# set up tKinter
fig1 = plt.Figure(figsize=(10, 5), dpi=100, constrained_layout=True)
ax1 = fig1.add_subplot(111)
chart_type = FigureCanvasTkAgg(fig1, root)
chart_type.get_tk_widget().grid(column=0, columnspan=int(n_cols/2), row=0)
ax1.set_title('Data')
# ax1.set_ylim(0, 1.4)

fig2 = plt.Figure(figsize=(10, 5), dpi=100, constrained_layout=True)
ax2 = fig2.add_subplot(111)
chart_type = FigureCanvasTkAgg(fig2, root)
chart_type.get_tk_widget().grid(column=int(n_cols/2), columnspan=int(n_cols/2), row=0)
ax2.set_title('FFT')
# ax2.set_xlim(0, 20)
ax2.set_ylim((-data_min_max*1.1, data_min_max*1.1))


# set up initial variables and sliders
def s_value(slider):
    return slider.get()


def s_str(slider):
    return '{: .2f}'.format(slider.get())


slider_d = {}
for key in points:
    slider_d[key] = {}
    slider_d[key]["t_label"] = ttk.Label(root, text=key)
    slider_d[key]["v_label"] = ttk.Label(root, text=fft.real[key])

    slider_d[key]["value_tk"] = tk.DoubleVar()
    slider_d[key]["value_tk"].set(fft.real[key])

    slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
    slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
    slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]

    slider_d[key]["v_labeli"] = ttk.Label(root, text=fft.real[key])

    slider_d[key]["value_tki"] = tk.DoubleVar()
    slider_d[key]["value_tki"].set(fft.imag[key])

    slider_d[key]["valuei"] = s_value(slider_d[key]["value_tki"])
    slider_d[key]["value_stri"] = s_str(slider_d[key]["value_tki"])
    slider_d[key]["v_labeli"]["text"] = slider_d[key]["value_stri"]


line1a, = ax1.plot(data)
line1b, = ax1.plot(ifft)

line2a, = ax2.plot(original_fft)
line2b, = ax2.plot(fft.real)
line2c, = ax2.plot(fft.imag)


def slider_changed(event):
    # update left hand figure with changed parameters

    for key in slider_d.keys():
        slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
        slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
        slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]
        fft.real[key] = slider_d[key]["value"]

        slider_d[key]["valuei"] = s_value(slider_d[key]["value_tki"])
        slider_d[key]["value_stri"] = s_str(slider_d[key]["value_tki"])
        slider_d[key]["v_labeli"]["text"] = slider_d[key]["value_stri"]
        fft.imag[key] = slider_d[key]["valuei"]

    ifft = np.fft.ifft(fft)
    ifft_shifted = ifft - np.mean(ifft) + np.mean(data)

    line1b.set_ydata(ifft_shifted)

    line2b.set_ydata(fft.real)
    line2c.set_ydata(fft.imag)

    ax1.set_ylim((
        np.min([np.min(ifft_shifted), np.min(data)])/1.1,
        np.max([np.max(ifft.real), np.max(data)])*1.1,
    ))

    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()


for key in points:
    slider_d[key]["slider"] = ttk.Scale(
        root,
        from_=data_min_max,
        to=-data_min_max,
        orient="vertical",
        variable=slider_d[key]["value_tk"],
        # command=slider_changed,
    )

    slider_d[key]["slider"].set(fft.real[key])
    slider_d[key]["slider"].bind("<ButtonRelease-1>", slider_changed)

    slider_d[key]["t_label"].grid(column=pos_dict[key][0], row=pos_dict[key][1])
    slider_d[key]["slider"].grid(column=pos_dict[key][0], row=pos_dict[key][1]+1, sticky="EW")
    slider_d[key]["v_label"].grid(column=pos_dict[key][0], row=pos_dict[key][1]+2)

    slider_d[key]["slideri"] = ttk.Scale(
        root,
        from_=data_min_max,
        to=-data_min_max,
        orient="vertical",
        variable=slider_d[key]["value_tki"],
        # command=slider_changed,
    )

    slider_d[key]["slideri"].set(fft.imag[key])
    slider_d[key]["slideri"].bind("<ButtonRelease-1>", slider_changed)

    slider_d[key]["slideri"].grid(column=pos_dict[key][0], row=pos_dict[key][1]+3, sticky="EW")
    slider_d[key]["v_labeli"].grid(column=pos_dict[key][0], row=pos_dict[key][1]+4)


# def button_simulate_f():
#     #when the simulate button is pressed, change the right hand side figure
#     solar = calc_spectrum(d, d["I0_solar_hr"])
#     solar_slr = calc_spectrum(d, d["I0_solar_slr"])

#     line2b.set_ydata(solar/max(solar))
#     line2c.set_ydata(solar_slr/max(solar))
#     fig1.canvas.draw()
#     fig1.canvas.flush_events()
#     fig2.canvas.draw()
#     fig2.canvas.flush_events()


# add the simulate and fit buttons
# button_update = ttk.Button(root, text="Simulate Response", command=button_simulate_f)
# button_update.grid(column=3, row=5)


# run the gui
root.mainloop()
