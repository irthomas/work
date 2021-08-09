# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:38:39 2021

@author: iant
"""


import re
import numpy as np
import lmfit


import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from instrument.nomad_so_instrument import nu_mp


from instrument.calibration.so_aotf_ils.simulation_functions import get_file, get_data_from_file, select_data, fit_temperature#, calc_blaze, aotf_conv


from instrument.calibration.so_aotf_ils.simulation_config import ORDER_RANGE, pixels



def s_value(slider):
    return slider.get()

def s_str(slider):
    return '{: .2f}'.format(slider.get())

regex = re.compile("20190416_020948_0p2a_SO_1_C")
index = 50

"""spectral grid and blaze functions of all orders"""
nu_range = [4309.7670539950705, 4444.765043408191]
dnu = 0.005
nu_hr = np.arange(nu_range[0], nu_range[1], dnu)


hdf5_file, hdf5_filename = get_file(regex)
d = get_data_from_file(hdf5_file, hdf5_filename)
d = select_data(d, index)
d["nu_hr"] = nu_hr
d = fit_temperature(d, hdf5_file)
# d = calc_blaze(d)


m_range = ORDER_RANGE
m = d["centre_order"]
t = d["temperature"]
A = d["aotf_freqs"][index]
print("A= %0.1f kHz" %A)
print("t=", t)


"""compute parameters from fits"""
"""blaze"""
#pre-compute delta nu per pixel
nu_mp_centre = nu_mp(m, pixels, t, p0=0)
p_dnu = (nu_mp_centre[-1] - nu_mp_centre[0])/320.0


p0 = np.polyval([0.22,150.8], m)
blaze_shift = np.polyval([-0.736363, -6.363908], t) # Blaze frequency shift due to temperature [pixel from Celsius]
p0 += blaze_shift

p_width = 22.473422 / p_dnu
print("m=", m)
print("p0=", p0)
print("p_width=", p_width)
print("blaze_shift=", blaze_shift)

"""aotf"""
A_nu0 = np.polyval([1.34082e-7, 0.1497089, 305.0604], A) # Frequency of AOTF [cm-1 from kHz]
aotf_shift  = -6.5278e-5 * t * A_nu0 # AOTF frequency shift due to temperature [relative cm-1 from Celsius]

A_nu0 += aotf_shift

# width = np.polyval([1.11085173e-06, -8.88538288e-03,  3.83437870e+01], A_nu0)
# lobe  = np.polyval([2.87490586e-06, -1.65141511e-02,  2.49266314e+01], A_nu0)
# asym  = np.polyval([-5.47912085e-07, 3.60576934e-03, -4.99837334e+00], A_nu0)

width = np.polyval([-2.85452723e-07,  1.66652129e-03,  1.83411690e+01], A_nu0) # Sinc width [cm-1 from AOTF frequency cm-1]
lobe  = np.polyval([ 2.19386777e-06, -1.32919656e-02,  2.18425092e+01], A_nu0) # sidelobes factor [scaler from AOTF frequency cm-1]
asym  = np.polyval([-3.35834373e-10, -6.10622773e-05,  1.62642005e+00], A_nu0) # Asymmetry factor [scaler from AOTF frequency cm-1]







print("A_nu0=", A_nu0)

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
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=1)


#best, min, max
param_dict = {
    "blaze_centre":[p0, p0-20.0, p0+20.],
    "blaze_width":[p_width, p_width-20., p_width+20.],
    "aotf_width":[width, width-2., width+2.],
    "aotf_shift":[0.0, -3.0, 3.0],
    "sidelobe":[lobe, 0.05, 20.0],
    "asymmetry":[asym, 0.01, 2.0],
    # "offset":[0.0, 0.0, 0.3],
    "offset_height":[0.0, 0.0, 0.3],
    "offset_width":[40.0, 10.0, 100.0],
    }

pos_dict = {
    "blaze_centre":[0, 1],
    "blaze_width":[0, 2],
    "aotf_width":[0, 3],
    "aotf_shift":[0, 4],
    "sidelobe":[3, 1],
    "asymmetry":[3, 2],
    # "offset":[3, 3],
    "offset_height":[3, 3],
    "offset_width":[3, 4],
    }



fig1 = plt.Figure(figsize=(6,5), dpi=100)
ax1 = fig1.add_subplot(111)
chart_type = FigureCanvasTkAgg(fig1, root)
chart_type.get_tk_widget().grid(column=0, columnspan=3, row=0)
ax1.set_title('AOTF and Blaze Shapes')   


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



def F_blaze(variables):
    
    dp = pixels - variables["blaze_centre"]
    dp[dp == 0.0] = 1.0e-6
    F = (variables["blaze_width"]*np.sin(np.pi*dp/variables["blaze_width"])/(np.pi*dp))**2
    
    return F


def F_aotf(nu_pm, variables):
    """reverse AOTF asymmetry"""
    def sinc(dx, amp, width, lobe, asym):
        # """asymetry switched 
     	sinc = amp*(width*np.sin(np.pi*dx/width)/(np.pi*dx))**2

     	ind = (abs(dx)>width).nonzero()[0]
     	if len(ind)>0: 
            sinc[ind] = sinc[ind]*lobe

     	ind = (dx>=width).nonzero()[0]
     	if len(ind)>0: 
            sinc[ind] = sinc[ind]*asym

     	return sinc

    dx = nu_pm - A_nu0 - variables["aotf_shift"]
    # print(dx)
    
    offset = variables["offset_height"] * np.exp(-dx**2.0/(2.0*variables["offset_width"]**2.0))
    
    F = sinc(dx, 1.0, variables["aotf_width"], variables["sidelobe"], variables["asymmetry"]) + offset
    
    
    
    return F
    # return F/max(F)




I0_lr = d["I0_lr"]
I0_lr_slr = d["I0_lr_slr"]

def calc(variables, I0=[0]):
    
    
    if I0[0] == 0:
        I0 = I0_lr
    
    solar = np.zeros(len(pixels))
    for im in range(m_range[0], m_range[1]+1):

        nu_pm = nu_mp(im, pixels, t)
        
        F = F_blaze(variables)
        G = F_aotf(nu_pm, variables)
        I0_lr_p = np.interp(nu_pm, nu_hr, I0)
        
        solar += F * G * I0_lr_p
    
    return solar#/max(solar)




def fit_resid(params, spectrum_norm, sigma):

    variables = {}
    for key in params.keys():
        variables[key] = params[key].value
        
    fit = calc(variables)

    return (fit/max(fit) - spectrum_norm) / sigma



nu_pm_c = nu_mp(m, pixels, t)

aotf_nu_range = np.arange(nu_range[0], nu_range[1], 0.1)

W_aotf = F_aotf(aotf_nu_range, variables)
line1a, = ax1.plot(aotf_nu_range, W_aotf)
W_blaze = F_blaze(variables)
line1b, = ax1.plot(nu_pm_c, W_blaze)


line2a, = ax2.plot(pixels, d["spectrum_norm"], label="%s i=%i" %(hdf5_filename, index))
line2b, = ax2.plot(pixels, np.zeros_like(pixels), label="Simulation")
line2c, = ax2.plot(pixels, np.zeros_like(pixels), label="Sim No Solar Line")

ax2.legend()




def slider_changed(event):
    for key in slider_d.keys():
        slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
        slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
        slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]
        variables[key] = slider_d[key]["value"]

    W_aotf = F_aotf(aotf_nu_range, variables)
    line1a.set_ydata(W_aotf)

    W_blaze = F_blaze(variables)
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
    solar = calc(variables)
    solar_slr = calc(variables, I0=I0_lr_slr)
    
    line2b.set_ydata(solar/max(solar))
    line2c.set_ydata(solar_slr/max(solar_slr))
    fig2.canvas.draw()
    fig2.canvas.flush_events()




def button_fit_f():
    print("Fitting AOTF and blaze")
    params = lmfit.Parameters()
    for key, value in param_dict.items():
       params.add(key, value[0], min=value[1], max=value[2])
       
    sigma = np.ones_like(d["spectrum_norm"])
    smi = d["absorption_pixel"]
    sigma[smi-18:smi+19] = 0.01
    # sigma[:100] = 10.
    # sigma[280:] = 10.

    lm_min = lmfit.minimize(fit_resid, params, args=(d["spectrum_norm"], sigma), method='leastsq')

    for key in params.keys():
        variables[key] = lm_min.params[key].value

    # print(variables)

    
    solar_fit = calc(variables)
    line2b.set_ydata(solar_fit/max(solar_fit))
    solar_fit_slr = calc(variables, I0=I0_lr_slr)
    line2c.set_ydata(solar_fit_slr/max(solar_fit_slr))



    for key in slider_d.keys():
        slider_d[key]["slider"].set(variables[key])
        slider_d[key]["value"] = variables[key]
        slider_d[key]["value_str"] = '{: .2f}'.format(variables[key])
        slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]
    #     slider_d[key]["value"] = s_value(slider_d[key]["value_tk"])
    #     slider_d[key]["value_str"] = s_str(slider_d[key]["value_tk"])
    #     slider_d[key]["v_label"]["text"] = slider_d[key]["value_str"]
    #     variables[key] = slider_d[key]["value"]

    W_aotf = F_aotf(aotf_nu_range, variables)
    line1a.set_ydata(W_aotf)

    W_blaze = F_blaze(variables)
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
