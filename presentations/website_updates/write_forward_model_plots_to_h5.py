# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:26:55 2026

@author: iant

MAKE PLOTS FOR WEBSITE FORWARD MODEL. WRITE ALL TO H5 FOR PLOTTING ON WEBSITE

1. GET SO/LNO PSA DATA
2. SELECT OBSERVATION IX
3. FROM AOTF FREQUENCY AND INST TEMPERATURE, GET AOTF CENTRE NU, SHAPE
4. GET BLAZE
5. GET HITRAN EXAMPLE E.G. WATER

"""
from analysis.so_lno_2023.molecules import get_molecules
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import h5py
from lmfit import Parameters
from analysis.so_lno_2023.forward_model import forward

from tools.spectra.savitzky_golay import savitzky_golay

import pds4_tools
from psa_utils import download, tap

from analysis.so_lno_2023.calibration import get_aotf, get_blaze_orders, get_calibration

psa = tap.PsaTap(tap_url="https://psa.esa.int/psa-tap/tap/")

lid = "nmd_cal_sc_so_20230930t090935-20230930t092801-a-i-134"
lid_full = "urn:esa:psa:em16_tgo_nmd:data_calibrated:%s" % lid
xml_path = lid + ".xml"

# version_id = "4.0"
channel = "SO"
# good_px_ixs = np.arange(50, 320)
good_px_ixs = np.arange(0, 320)

query = "SELECT TOP 1 * FROM psa.epn_core WHERE obs_id = '%s'" % (lid_full.lower())
# results = psa.query(query).to_dict("list")

if os.path.exists(xml_path):
    print("File already exists, skipping download")

else:
    print("Downloading PSA data")
    filepaths = download.download_by_query(query)

    print("Clearing up files")
    good_filepaths = [f for f in filepaths if lid.lower() in f.lower()]

    for filepath in filepaths:
        if filepath in good_filepaths:
            filename = os.path.basename(filepath)
            shutil.copy(filepath, filename)

print("Loading file")
structures = pds4_tools.read(xml_path)

data = structures["CAL_NOMAD_%s" % channel].data

# get a list of available data fields
available_fields = list(data.meta_data.keys())

# extract altitudes, wavenumbers and transmittances from product
tangent_altitudes = data["TangentAltAreoidStart0"]
wavenumbers = data["Wavenumber, Pixel wavenumber"]
transmittances = data["Transmittance, Pixel transmittance"]
transmittance_errors = data["Transmittance error, Pixel transmittance error"]
aotf_freq = data["AOTFFrequency"][0]
centre_order = data["DiffractionOrder"][0]
aotf_t = data["InstrumentTemperature"][0]
grat_t = data["InstrumentTemperature"][0]

aotf = {"type": "sinc_gauss"}
orders = np.arange(centre_order - 2, centre_order + 3, 1)


aotf_d = get_aotf(channel, aotf_freq, aotf_t, aotf=aotf)
aotf_nu_centre = aotf_d["nu_centre"]
orders_d = get_blaze_orders(channel, orders, aotf_nu_centre, grat_t, px_ixs=good_px_ixs)
cal_d = get_calibration(channel, centre_order, aotf_d, orders_d)


# plot x (spectral calibration of each pixel) vs transmittance for all the spectra in the file
# add a legend with the tangent altitude of each spectrum
# plt.figure()
# plt.title("NOMAD data: %s" % os.path.basename(xml_path))
# plt.xlabel("Wavenumber (cm-1)")
# plt.ylabel("Transmittance")

# for tangent_altitude, pixel_x, transmittance in zip(tangent_altitudes, x, transmittances):
#     plt.plot(pixel_x, transmittance, label="%0.1fkm" % tangent_altitude)

# plt.grid()
# plt.legend()

plt.figure()
plt.plot(aotf_d["aotf_nus"], aotf_d["F_aotf"])

for order in orders:
    plt.plot(orders_d[order]["px_nus"], orders_d[order]["F_blaze"])

# datasets to write to h5 file
so_h5 = {"aotf_nu": aotf_d["aotf_nus"], "aotf_f": aotf_d["F_aotf"]}
for order in orders:
    so_h5["blaze_nu_%i" % order] = orders_d[order]["px_nus"]
    so_h5["blaze_F_%i" % order] = orders_d[order]["F_blaze"]


# %%
# make isrf shapes and copy isrf + aotf order 134 to hdf5 and plot on website

isrf_filename = "Order_134_ILS.dat"
aotf_filename = "Order_134_gauss_w50_updated.dat"

with open(isrf_filename, "r") as f:
    isrf_data = np.asfarray([s.strip().split() for s in f.readlines()])[:, 1:]

with open(aotf_filename, "r") as f:
    aotf_data = np.asfarray([s.strip().split() for s in f.readlines()])

aotf_data[:, 1] = aotf_data[:, 1] / np.max(aotf_data[:, 1])


def make_isrf(nu_grid, c):

    isrf0 = c[2] * np.exp(-0.5 * ((nu_grid + c[0]) / c[1]) ** 2)
    isrf1 = c[5] * np.exp(-0.5 * ((nu_grid + c[3]) / c[4]) ** 2)
    isrf = isrf0 + isrf1

    return isrf


plt.plot(aotf_data[:, 0] + aotf_nu_centre, aotf_data[:, 1])


plt.figure()

px_nus = orders_d[centre_order]["px_nus"]

for px_ix, px_nu in enumerate(px_nus):

    nu_grid = np.arange(-0.5, 0.5, 0.01)
    isrf = make_isrf(nu_grid, isrf_data[px_ix, :])

    isrf = isrf / np.max(isrf)
    nus = nu_grid + px_nu

    plt.plot(nus, isrf)

    so_h5[px_ix] = {}
    so_h5[px_ix]["nu"] = nus
    so_h5[px_ix]["isrf"] = isrf

# also add pixel 100 isrf from all orders
plt.figure()

px_ix = 100
for order in orders:
    px_nu = orders_d[order]["px_nus"][px_ix]
    nu_grid = np.arange(-0.5, 0.5, 0.01)
    isrf = make_isrf(nu_grid, isrf_data[px_ix, :])

    isrf = isrf / np.max(isrf)
    nus = nu_grid + px_nu

    plt.plot(nus, isrf)

    so_h5["%i-%i" % (order, px_ix)] = {}
    so_h5["%i-%i" % (order, px_ix)]["nu"] = nus
    so_h5["%i-%i" % (order, px_ix)]["isrf"] = isrf

# %%
# add hitran high-res spectrum
molecules = {"H2O": {"isos": [1, 2, 3, 4]}}
geom_d = {"alt_grid": [30], "myear": 35, "ls": 1, "lat": 1, "lon": 1, "lst": 1, "path_lengths_km": [100]}
molecule_d = get_molecules(molecules, geom_d)

fw = forward(raw=False)
fw.calibrate(cal_d)
fw.geometry(geom_d)
fw.molecules(molecule_d)

params = Parameters()
params.add('h2o_mol_scaler', value=1.0, min=0.001, max=5.0)
transm = fw.forward_so(params, plot=["hr"], axes=[])

plt.figure()
hapi_nus = fw.hapi_nus
hapi_trans = fw.hapi_trans_total

plt.plot(hapi_nus, hapi_trans)
so_h5["water_hr_nu"] = hapi_nus
so_h5["water_hr_spec"] = hapi_trans

# after convolution
plt.figure()
plt.plot(orders_d[centre_order]["px_nus"], transm)

so_h5["water_conv_nu"] = orders_d[centre_order]["px_nus"]
so_h5["water_conv_spec"] = transm

# %%
with h5py.File("so_forward_134.h5", "w") as f:
    for k1, v1 in so_h5.items():
        if isinstance(v1, dict):
            for k2, v2 in so_h5[k1].items():
                if isinstance(k2, dict):
                    for k3, v3 in so_h5[k1][k2].items():
                        path = "%s/%s/%s" % (k1, k2, k3)
                        data = v3
                        f.create_dataset(path, dtype=np.float32, data=data, compression="gzip", shuffle=True)
                else:
                    path = "%s/%s" % (k1, k2)
                    data = v2
                    f.create_dataset(path, dtype=np.float32, data=data, compression="gzip", shuffle=True)
        else:
            path = "%s" % (k1)
            data = v1
            f.create_dataset(path, dtype=np.float32, data=data, compression="gzip", shuffle=True)

# %%

l2_path = r"X:\projects\work\NOMAD\Science\so_occ_h2o_1\ord134\RESULTS\2023\09\20230930_090935_1p0a_SO_A_I_134_Level2.h5"

indices_in_l1p0a = [72, 76, 80, 84, 88, 92, 96, 100, 104]
# manually write tangent altitudes from psa file
psa_tangent_altitudes = [10.02, 10.63, 11.25, 11.86, 12.48, 13.09, 13.70, 14.32, 14.93]

with h5py.File(l2_path, "r") as l2f:
    xl1 = np.squeeze(l2f["Science/X"][0, :])
    yfit = np.squeeze(l2f["Science/Y"][...])
    yobs = np.squeeze(l2f["Science/YObs"][...])
    alt = np.squeeze(l2f["Geometry/Point0/TangentAlt"][...])


plt.figure()
plt.plot(yobs.T)
plt.plot(yfit.T)

# check transmittance in l2 and psa file match
for i in indices_in_l1p0a:
    plt.plot(transmittances[i, :], "k--")
    print("%0.2f" % tangent_altitudes[i])

with h5py.File("so_retrieval_134.h5", "w") as f:
    f.create_dataset("x", dtype=np.float32, data=xl1, compression="gzip", shuffle=True)
    for i, alt in enumerate(psa_tangent_altitudes):
        index = indices_in_l1p0a[i]
        f.create_dataset("%i/alt" % index, dtype=np.float32, data=[alt])
        f.create_dataset("%i/obs" % index, dtype=np.float32, data=yobs[i, :], compression="gzip", shuffle=True)
        f.create_dataset("%i/fit" % index, dtype=np.float32, data=yfit[i, :], compression="gzip", shuffle=True)


# %%
# UVIS spectral resolution
# only binning types are 0, 1, 3, 7

occ_fwhm = np.asarray([
    [253.65, 1.38],
    [296.73, 1.42],
    [312.57, 1.53],
    [365.02, 1.56],
    [404.66, 1.47],
    [435.83, 1.48],
    [546.08, 1.60],
    [576.96, 1.51],
    [579.07, 1.52],
])

nad_fwhm = np.asarray([
    [253.65, 1.20],
    [296.73, 1.20],
    [312.57, 1.45],
    [365.02, 1.38],
    [404.66, 1.29],
    [435.83, 1.27],
    [546.08, 1.45],
    [576.96, 1.53],
    [579.07, 1.53],
])

# plt.plot(occ_fwhm[:, 0], occ_fwhm[:, 1])
# plt.plot(nadir_fwhm[:, 0], nadir_fwhm[:, 1])

# interpolate to pixel grid
with h5py.File("20180530_074202_0p3b_UVIS_I.h5", "r") as h5f:
    lambda_0bin = h5f["Science/X"][0, :]
# uvis_px = np.arange(len(uvis_x))


# spectrally binned
# 0 1 2 3 4 5 6 7

uvis_fwhm = {}
uvis_fwhm["ixs_0bin"] = np.arange(len(lambda_0bin))
uvis_fwhm["ixs_1bin"] = np.arange(0.5, 1023 - 0.49, 2.0)
uvis_fwhm["ixs_3bin"] = np.arange(1.5, 1023 - 1.49, 4.0)
uvis_fwhm["ixs_7bin"] = np.arange(3.5, 1023 - 3.49, 8.0)

# unsmoothed unbinned full range interpolation
fwhm_0bin_occ = np.interp(lambda_0bin, occ_fwhm[:, 0], occ_fwhm[:, 1])
fwhm_0bin_nad = np.interp(lambda_0bin, nad_fwhm[:, 0], nad_fwhm[:, 1])

# smooth
uvis_fwhm["fwhm_0bin_occ_smoothed"] = savitzky_golay(fwhm_0bin_occ, 59, 1)
uvis_fwhm["fwhm_0bin_nad_smoothed"] = savitzky_golay(fwhm_0bin_nad, 59, 1)

uvis_fwhm["fwhm_1bin_occ_smoothed"] = np.interp(uvis_fwhm["ixs_1bin"], uvis_fwhm["ixs_0bin"], uvis_fwhm["fwhm_0bin_occ_smoothed"]) * 2
uvis_fwhm["fwhm_1bin_nad_smoothed"] = np.interp(uvis_fwhm["ixs_1bin"], uvis_fwhm["ixs_0bin"], uvis_fwhm["fwhm_0bin_nad_smoothed"]) * 2

uvis_fwhm["fwhm_3bin_occ_smoothed"] = np.interp(uvis_fwhm["ixs_3bin"], uvis_fwhm["ixs_0bin"], uvis_fwhm["fwhm_0bin_occ_smoothed"]) * 4
uvis_fwhm["fwhm_3bin_nad_smoothed"] = np.interp(uvis_fwhm["ixs_3bin"], uvis_fwhm["ixs_0bin"], uvis_fwhm["fwhm_0bin_nad_smoothed"]) * 4

uvis_fwhm["fwhm_7bin_occ_smoothed"] = np.interp(uvis_fwhm["ixs_7bin"], uvis_fwhm["ixs_0bin"], uvis_fwhm["fwhm_0bin_occ_smoothed"]) * 8
uvis_fwhm["fwhm_7bin_nad_smoothed"] = np.interp(uvis_fwhm["ixs_7bin"], uvis_fwhm["ixs_0bin"], uvis_fwhm["fwhm_0bin_nad_smoothed"]) * 8

plt.plot(uvis_fwhm["ixs_0bin"], fwhm_0bin_occ)
plt.scatter(uvis_fwhm["ixs_0bin"], uvis_fwhm["fwhm_0bin_occ_smoothed"])
plt.scatter(uvis_fwhm["ixs_1bin"], uvis_fwhm["fwhm_1bin_occ_smoothed"])
plt.scatter(uvis_fwhm["ixs_3bin"], uvis_fwhm["fwhm_3bin_occ_smoothed"])
plt.scatter(uvis_fwhm["ixs_7bin"], uvis_fwhm["fwhm_7bin_occ_smoothed"])


plt.plot(uvis_fwhm["ixs_0bin"], fwhm_0bin_nad)
plt.scatter(uvis_fwhm["ixs_0bin"], uvis_fwhm["fwhm_0bin_nad_smoothed"])
plt.scatter(uvis_fwhm["ixs_1bin"], uvis_fwhm["fwhm_1bin_nad_smoothed"])
plt.scatter(uvis_fwhm["ixs_3bin"], uvis_fwhm["fwhm_3bin_nad_smoothed"])
plt.scatter(uvis_fwhm["ixs_7bin"], uvis_fwhm["fwhm_7bin_nad_smoothed"])

with h5py.File("uvis_forward.h5", "w") as f:
    for key, value in uvis_fwhm.items():
        f.create_dataset(key, dtype=np.float32, data=value, compression="gzip", shuffle=True)

h = ""
headers = ["Pixel number", "UVIS solar occultation FWHM (nm)", "UVIS nadir and limb FWHM (nm)"]
binnings = {0: "unbinned", 1: "half-resolution", 3: "quarter-resolution", 7: "eighth-resolution"}

for key, value in binnings.items():
    h += "<h2>UVIS approximate spectral resolution FWHM for binning=%i (%s)</h2>\n" % (key, value)
    h += "<table border='2'><tbody>\n"
    h += "<tr>"
    for header in headers:
        h += "<th>%s</th>" % header
    h += "</tr>\n"

    for nm, occ, nad in zip(uvis_fwhm["ixs_%ibin" % key], uvis_fwhm["fwhm_%ibin_occ_smoothed" % key], uvis_fwhm["fwhm_%ibin_nad_smoothed" % key]):
        h += "<tr><td>%i</td><td>%0.2f</td><td>%0.2f</td></tr>\n" % (nm, occ, nad)
    h += "</tbody></table>\n"
    h += "<br>\n"
    h += "<br>\n"
