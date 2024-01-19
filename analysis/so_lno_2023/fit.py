# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:29:24 2023

@author: iant

VARIABLES TO CONSIDER:
    AOTF FUNCTION 
    BLAZE FUNCTION
    ATMOS TEMPERATURE (MORE VARIABILITY FURTHER FROM LINE CENTRE, EFFECT IS SMALL)
    ISOTOPE RATIO (SHOULD BE SMALL)
    BENDING/BASELINE SHAPE
"""






import numpy as np
import matplotlib.pyplot as plt


from analysis.so_lno_2023.functions.h5 import read_h5
from analysis.so_lno_2023.calibration import get_aotf, get_blaze_orders, get_calibration
from analysis.so_lno_2023.molecules import get_molecules
from analysis.so_lno_2023.geometry import get_geometry
from analysis.so_lno_2023.forward_model import forward


from tools.general.get_nearest_index import get_nearest_index
from tools.spectra.baseline_als import baseline_als

from lmfit import minimize, Parameters



"""SO"""
h5 = "20220301_114833_1p0a_SO_A_I_186"
# h5 = "20220301_114833_1p0a_SO_A_I_185"
# h5 = "20220301_114833_1p0a_SO_A_I_184"
#without saturation alt=60km scaler=1.0


channel = "so"
# chosen_alt = 40.0 #km
chosen_alt = 60.0 #km
# chosen_alt = 140.0 #km

molecules = {
    "CO":{"isos":[1,2,3,4]},
    # "CO":{"isos":[1]},
}


alt_delta = 5.0 #each layer
alt_max = chosen_alt + 50.0

# orders = np.arange(183, 193)
orders = np.arange(180, 192)

px_ixs = np.arange(50, 300)
# px_ixs = np.arange(0, 320)

# mol_scaler = 0.3409
# mol_scaler = 0.3109
mol_scaler = 1.0
# mol_scaler = 2.5
# mol_scaler = 0.0


# plot = ["fit"]



fit_raw = True
# fit_raw = False






#get data from h5 obs file
h5_d = read_h5(h5)

ix = get_nearest_index(chosen_alt, h5_d["alts"])

y = h5_d["y"][ix][px_ixs]
y_raw = h5_d["y_raw"][ix][px_ixs]
y_cont = baseline_als(y)
y_flat = y / y_cont

centre_order = h5_d["order"]
aotf_freq = h5_d["aotf_freq"]
grat_t = h5_d["nomad_t"]
aotf_t = h5_d["nomad_t"]



#get calibration info
# aotf = {"type":"file", "filename":"4500um_closest_aotf.txt"}
aotf = aotf={"type":"sinc_gauss"}

"""initial parameters"""
aotf_d = get_aotf(channel, aotf_freq, aotf_t, aotf=aotf)
aotf_nu_centre = aotf_d["nu_centre"]
orders_d = get_blaze_orders(channel, orders, aotf_nu_centre, grat_t, px_ixs=px_ixs)

cal_d = get_calibration(channel, centre_order, aotf_d, orders_d)
geom_d = get_geometry(h5_d, ix, alt_max, alt_delta)
molecule_d = get_molecules(molecules, geom_d)




"""raw solar spectrum"""
# fw_raw = forward(raw=True)
# fw_raw.calibrate(cal_d)
# toa = fw_raw.forward_toa(plot=["aotf"])

# plt.figure()
# plt.plot(toa, label="Simulated top-of-atmosphere signal")
# plt.grid()


"""aotf temperature shifted transmittance"""


fw_raw = forward(raw=True)
fw_raw.calibrate(cal_d)
toa1 = fw_raw.forward_toa()

# recalibrate with different AOTF temperature
aotf_t_shift = 1.0
aotf_d = get_aotf(channel, aotf_freq, aotf_t+aotf_t_shift, aotf=aotf)

# recalibrate with different grating temperature
grat_t_shift = 1.0
orders_d = get_blaze_orders(channel, orders, aotf_nu_centre, grat_t+grat_t_shift, px_ixs=px_ixs)



cal_d2 = get_calibration(channel, centre_order, aotf_d, orders_d, plot=["aotf"])
fw_raw.calibrate(cal_d2)
toa2 = fw_raw.forward_toa()


plt.figure()
plt.plot(toa1, label="Simulated top-of-atmosphere signal (AOTF T=%0.1fC)" %aotf_t)
plt.plot(toa2, label="Simulated top-of-atmosphere signal (AOTF T=%0.1fC)" %(aotf_t+aotf_t_shift))
plt.legend()
plt.grid()


plt.figure()
plt.title(h5)
plt.plot(toa1 / toa2, label="Ratio TOA raw signals")
plt.legend()
plt.grid()

"""transmittance with molecules - modify atmos temperature"""


fw = forward(raw=False)
fw.calibrate(cal_d)
fw.geometry(geom_d)
fw.molecules(molecule_d)

params = Parameters()
params.add('mol_scaler', value=mol_scaler)

trans1 = fw.forward_so(params, plot=["hr", "cont"])

molecule_d2 = get_molecules(molecules, geom_d)
molecule_d2["CO"]["ts"] += 50.0 #modify atmospheric layer temperatures

fw2 = forward(raw=False)
fw2.calibrate(cal_d)
fw2.geometry(geom_d)
fw2.molecules(molecule_d2)

params = Parameters()
params.add('mol_scaler', value=mol_scaler)

trans2 = fw2.forward_so(params, plot=["hr", "cont"])

molecule_d3 = get_molecules(molecules, geom_d)
molecule_d3["CO"]["ts"] = np.ones_like(molecule_d3["CO"]["ts"]) + 120.0 #modify atmospheric layer temperatures

fw3 = forward(raw=False)
fw3.calibrate(cal_d)
fw3.geometry(geom_d)
fw3.molecules(molecule_d3)

params = Parameters()
params.add('mol_scaler', value=mol_scaler)

trans3 = fw3.forward_so(params, plot=["hr", "cont"])



plt.figure()
plt.plot(y_flat, label="%s %0.1fkm" %(h5, chosen_alt))
plt.plot(trans1, label="Simulated spectrum")
plt.plot(trans2, label="Simulated spectrum, T+50K")
plt.plot(trans3, label="Simulated spectrum, T=120K")
plt.legend()
plt.grid()




"""change AOTF shape"""
# cal_d = get_calibration(channel, centre_order, aotf_d, orders_d)
# fw = forward(raw=False)
# fw.calibrate(cal_d)
# fw.geometry(geom_d)
# fw.molecules(molecule_d)

# params = Parameters()
# params.add('mol_scaler', value=mol_scaler)


# trans1 = fw.forward_so(params, plot=["cont"])
# plt.figure()
# plt.plot(y_flat, label="%s %0.1fkm" %(h5, chosen_alt))
# plt.plot(trans1, label="Simulated spectrum")
# plt.legend()
# plt.grid()


# aotf_boost_nu_range = [4205., 4218.]
# aotf_d2 = get_aotf(channel, aotf_freq, aotf_t, aotf=aotf)
# aotf_nus = aotf_d2["aotf_nus"]
# ixs = np.where((aotf_nus > aotf_boost_nu_range[0]) & (aotf_nus < aotf_boost_nu_range[1]))[0]
# aotf_d2["F_aotf"][ixs] *= 2.0
# cal_d3 = get_calibration(channel, centre_order, aotf_d2, orders_d)

# fw2 = forward(raw=False)
# fw2.calibrate(cal_d3)
# fw2.geometry(geom_d)
# fw2.molecules(molecule_d)


# plt.figure()
# plt.plot(cal_d["aotf"]["F_aotf"])
# plt.plot(cal_d3["aotf"]["F_aotf"])



# params = Parameters()
# params.add('mol_scaler', value=mol_scaler)

# trans2 = fw2.forward_so(params, plot=["cont"])
# plt.figure()
# plt.plot(y_flat, label="%s %0.1fkm" %(h5, chosen_alt))
# plt.plot(trans2, label="Simulated spectrum")
# plt.legend()
# plt.grid()


# # save all figs
# for i in plt.get_fignums():
#     plt.figure(i)
#     plt.savefig('%i.png' % i)

# fw.fit(params, y)

# raw_spectrum2 = fw_raw.forward_so(params)

# # ratio = raw_spectrum1/raw_spectrum2

# plt.figure()
# plt.plot(raw_spectrum1)
# plt.plot(raw_spectrum2)

# plt.figure()
# plt.plot(px_ixs, ratio)
    
    

# fw_raw = forward(raw=True)
# fw_raw.calibrate(cal_d)
# fw_raw.geometry(geom_d)
# fw_raw.molecules(molecule_d)

    

    # fw_raw.forward_so(params, plot=["hr"])
    

    # fw_raw.fit(params, y_raw, plot=["fit"])

    
    
    
    # ssd = forward(params, channel, cal_d, molecule_d, y_flat)
    
    
    
    # out = minimize(forward, params, args=(channel, cal_d, molecule_d, y_flat), max_nfev=20)
    # print("number of iterations=", out.nfev)
    
    # # out.params.pretty_print()
    
    # params["mol_scaler"] = out.params["mol_scaler"]
    
    # diff = forward(params, channel, cal_d, molecule_d, y_flat, plot=["hr", "fit", "cont"])
    
    # # next steps: remove first 50 and last 20 pixels
    # # fit to microwindows?

