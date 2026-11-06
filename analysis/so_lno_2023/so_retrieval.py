# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:40:58 2026

@author: iant

FOLLOWING FROM SIMULATE ABS DEPTHS, DO RETRIEVAL WITH NEW MODULAR CALIBRATION
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import lmfit
# from lmfit import minimize
from lmfit import Parameters

# compare absorption depths to forward model
from analysis.so_lno_2023.calibration import get_aotf, get_blaze_orders, get_calibration
from analysis.so_lno_2023.molecules import get_molecules
# from analysis.so_lno_2023.geometry import get_geometry
from analysis.so_lno_2023.forward_model import forward
from analysis.so_lno_2023.functions.geometry import make_path_lengths
from analysis.so_lno_2023.functions.h5 import read_h5
from analysis.so_lno_2023.functions.baseline import baseline_als as baseline
from analysis.so_lno_2023.functions.spectral_correction import find_spectra_minima
from tools.general.get_minima_maxima import get_local_minima
from analysis.so_lno_2023.functions.spectral_cal import lt22_t_shift

MAX_NFEV = 99
MAX_SPECTRAL_SHIFT_FIT = 0.1  # cm-1

# h5 = "20260614_070009_1p0a_SO_A_I_134.h5"
# h5 = "20250623_031155_1p0a_SO_A_E_129.h5"
h5 = "20230930_090935_1p0a_SO_A_I_134.h5"  # high water content

DATA_PATH = r"c:\users\iant\dropbox\nomad\python"
DATA_PATH = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a\2023\09\30"

spec_ix = 90
channel = "so"
molecules = {
    "H2O": {"isos": [1, 2, 3, 4]},
    # "HCl": {"isos": [1, 2, 3, 4]},
}
h2o_mol_scaler = 1.0
hcl_mol_scaler = 1.0
good_px_ixs = np.arange(50, 320)
n_adj_orders = 2
t_offset = {
    "20260614_070009_1p0a_SO_A_I_134.h5": {"grat": -1.138, "aotf": 3.5},
    "20250623_031155_1p0a_SO_A_E_129.h5": {"grat": 0.0, "aotf": 0.0},
    "20230930_090935_1p0a_SO_A_I_134.h5": {"grat": 0.0, "aotf": 0.0},
}


h5_d = read_h5(h5, spec_ix=spec_ix, data_path=DATA_PATH)
grat_t = h5_d["nomad_t"]
aotf_t = h5_d["nomad_t"]
aotf_freq = h5_d["aotf_freq"]

for key in ["x", "y", "y_raw", "y_error"]:
    h5_d[key] = h5_d[key][good_px_ixs]
y_bg = baseline(h5_d["y"])
y_norm = h5_d["y"] / y_bg
x = h5_d["x"]
y_error = h5_d["y_error"] / y_bg

print("h5=", h5, "ix=", spec_ix, "alt=", h5_d["alt"], "T=", np.mean(y_bg))
if h5 in t_offset.keys():
    if "grat" in t_offset[h5].keys():
        grat_t += t_offset[h5]["grat"]
    if "aotf" in t_offset[h5].keys():
        aotf_t += t_offset[h5]["aotf"]


y_raw_norm = h5_d["y_raw"] / np.max(h5_d["y_raw"])


centre_order = h5_d["order"]
orders = np.arange(centre_order - n_adj_orders, centre_order + n_adj_orders + 1)

# get calibration info
# aotf = {"type":"file", "filename":"4500um_closest_aotf.txt"}
aotf = {"type": "sinc_gauss"}

"""initial parameters"""
aotf_d = get_aotf(channel, aotf_freq, aotf_t, aotf=aotf)
aotf_nu_centre = aotf_d["nu_centre"]
orders_d = get_blaze_orders(channel, orders, aotf_nu_centre, grat_t, px_ixs=good_px_ixs)
cal_d = get_calibration(channel, centre_order, aotf_d, orders_d)

geom_d = {}
geom_d["alt"] = np.arange(h5_d["alt"], 80, 5)
geom_d["myear"] = h5_d["myear"]
geom_d["ls"] = h5_d["ls"]
geom_d["lst"] = h5_d["lst"]
geom_d["lat"] = h5_d["lat"]
geom_d["lon"] = h5_d["lon"]
geom_d["alt_grid"] = geom_d["alt"]
path_lengths = make_path_lengths(geom_d["alt_grid"])
geom_d["path_lengths_km"] = path_lengths

molecule_d = get_molecules(molecules, geom_d)


"""check aotf shape in raw spectra"""
# fw_raw = forward(raw=True)
# fw_raw.calibrate(cal_d)
# fw_raw.geometry(geom_d)
# fw_raw.molecules(molecule_d)

# params = Parameters()
# params.add('h2o_mol_scaler', value=h2o_mol_scaler, min=0.001, max=5.0)

# raw1 = fw_raw.forward_so(params, plot=[], axes=[])

# fig2, ax2 = plt.subplots()
# ax2.plot(y_raw_norm)
# ax2.plot(raw1)
# aotf_chi_sq = np.sum((y_raw_norm - raw1)**2)
# print("AOTF fit chisq = %0.3f" % aotf_chi_sq)


"""check pixel shift, derive new grating temperature offset"""
fw = forward(raw=False)
fw.calibrate(cal_d)
fw.geometry(geom_d)
fw.molecules(molecule_d)

params = Parameters()
if "H2O" in molecules.keys():
    params.add('h2o_mol_scaler', value=h2o_mol_scaler, min=0.001, max=5.0)
if "HCL" in molecules.keys():
    params.add('hcl_mol_scaler', value=hcl_mol_scaler, min=0.001, max=50.0)

# fit spectral calibration
fig1, ax1 = plt.subplots()
ax1.plot(x, y_norm)
# find wavenumber of absorptions in data
nu_obs_minima, chi_sq_obs_all, logger_msg = find_spectra_minima(x, y_norm, min_sig=0.99, stds_sig=2, ax=ax1)

trans1 = fw.forward_so(params, plot=["hr"], axes=[])

ax1.plot(x, trans1)
# find wavenumber of absorptions in simulation
nu_sim_minima, chi_sq_sim_all, logger_sim_msg = find_spectra_minima(x, trans1, min_sig=0.99, stds_sig=2, ax=ax1)

chi_sq_obs_all = 1.0 - chi_sq_obs_all / np.max(chi_sq_obs_all)
chi_sq_sim_all = 1.0 - chi_sq_sim_all / np.max(chi_sq_sim_all)

matching_nu = []
for nu_obs_minimum, chi_sq_obs in zip(nu_obs_minima, chi_sq_obs_all):
    if chi_sq_obs > 0.5:
        # try to fit
        good_ixs = np.where(np.abs(nu_obs_minimum - nu_sim_minima) < MAX_SPECTRAL_SHIFT_FIT)[0]
        if len(good_ixs) == 1:
            if chi_sq_sim_all[good_ixs[0]] > 0.5:
                matching_nu.append([nu_obs_minimum, nu_sim_minima[good_ixs[0]]])

matching_nu = np.asarray(matching_nu)


delta_nu = np.mean(np.diff(matching_nu))
# convert nu error to temperature error
so_delta_nu = np.mean(np.diff(cal_d["orders"][centre_order]["px_nus"]))
delta_px = delta_nu / so_delta_nu
delta_t = lt22_t_shift(delta_px)
print("Temperature offset of %0.3fC found (%0.4f cm-1)" % (delta_t, so_delta_nu))

stop()

"""run retrieval"""
"""initiate parameters"""
aotf_d = get_aotf(channel, aotf_freq, aotf_t, aotf=aotf)
aotf_nu_centre = aotf_d["nu_centre"]
orders_d = get_blaze_orders(channel, orders, aotf_nu_centre, grat_t, px_ixs=good_px_ixs)
cal_d = get_calibration(channel, centre_order, aotf_d, orders_d)
x = cal_d["orders"][centre_order]["px_nus"]

geom_d = {}
geom_d["alt"] = np.arange(h5_d["alt"], 80, 5)
geom_d["myear"] = h5_d["myear"]
geom_d["ls"] = h5_d["ls"]
geom_d["lst"] = h5_d["lst"]
geom_d["lat"] = h5_d["lat"]
geom_d["lon"] = h5_d["lon"]
geom_d["alt_grid"] = geom_d["alt"]
path_lengths = make_path_lengths(geom_d["alt_grid"])
geom_d["path_lengths_km"] = path_lengths

molecules = {
    "H2O": {"isos": [1, 2, 3, 4]},
    # "HCl": {"isos": [1, 2, 3, 4]},
}
molecule_d = get_molecules(molecules, geom_d)

fw = forward(raw=False)
fw.calibrate(cal_d)
fw.geometry(geom_d)
fw.molecules(molecule_d)

params = Parameters()
if "H2O" in molecules.keys():
    params.add('h2o_mol_scaler', value=h2o_mol_scaler, min=0.001, max=5.0)
if "HCL" in molecules.keys():
    params.add('hcl_mol_scaler', value=hcl_mol_scaler, min=0.1, max=10.0)


def trans_residual(params, fw, y_norm, sigma):
    trans = fw.forward_so(params, plot=[], axes=[])
    return (trans - y_norm)**2 / sigma


sigma = 0.001

lm_min = lmfit.minimize(trans_residual, params, args=(fw, y_norm, sigma), method='leastsq', max_nfev=MAX_NFEV)
chisq = lm_min.chisqr


"""plot final values"""
variables = {}
for key in params.keys():
    variables[key] = lm_min.params[key].value

params_fit = Parameters()
if "H2O" in molecules.keys():
    params_fit.add('h2o_mol_scaler', value=variables['h2o_mol_scaler'])

    fw = forward(raw=False)
    fw.calibrate(cal_d)
    fw.geometry(geom_d)
    fw.molecules(molecule_d)
    trans = fw.forward_so(params_fit, plot=[], axes=[])

    plt.figure()
    plt.plot(x, trans, label="H2O")

params_fit = Parameters()
if "HCL" in molecules.keys():
    params_fit.add('hcl_mol_scaler', value=variables['hcl_mol_scaler'])

    fw = forward(raw=False)
    fw.calibrate(cal_d)
    fw.geometry(geom_d)
    fw.molecules(molecule_d)
    trans = fw.forward_so(params_fit, plot=[], axes=[])
    plt.plot(x, trans, label="HCL")

plt.plot(x, y_norm, "k")
plt.plot(x, y_norm - y_error, "k--", alpha=0.5)
plt.plot(x, y_norm + y_error, "k--", alpha=0.5)
