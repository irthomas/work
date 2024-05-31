# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:49:18 2023

@author: iant

READ IN PHOBOS JSON OUTPUT AND ANALYSE
"""

import numpy as np
import matplotlib.pyplot as plt
import json

from instrument.nomad_lno_instrument_v01 import nu_mp
from tools.datasets.get_phobos_crism_data import get_phobos_crism_data


crism_d = get_phobos_crism_data()


"""search the json made by script phobos_rad_cal_fullscan.py
orders_to_find = find observations containing these order combinations
nominal_orders = interpolate the nomian
orders_to_scale = 
"""
selected_orders = {
    
    # "Hydration band 1":{"orders_to_find":[[160, 162, 164, 166, 168, 170]], "nominal_orders":[160, 162, 164, 166, 168, 170], "orders_to_scale":[164, 166, 168, 170]},
    # "Hydration band 2":{"orders_to_find":[[157, 160, 163, 166, 169, 172]], "nominal_orders":[157, 160, 163, 166, 169, 172], "orders_to_scale":[166, 169, 172]},
    "Carbonates":{"orders_to_find":[[174, 175, 176, 190, 191, 192]], "nominal_orders":[174, 175, 176, 190, 191, 192], "orders_to_scale":[174, 175, 176, 190, 191, 192], "linestyle":"none"},
    }


#open previously saved spectra
with open("lno_phobos_output.json", "r") as f:
    phobos_d = json.load(f)

#convert json to numpy arrays
for h5 in phobos_d.keys():
    for key in phobos_d[h5].keys():
        phobos_d[h5][key] = np.asarray(phobos_d[h5][key])


fig1, ax1a = plt.subplots(figsize=(12, 7))
fig1.suptitle("LNO Phobos Observations Radiometric Scaling")

fig2, ax2a = plt.subplots(figsize=(12, 7))
fig2.suptitle("Phobos Radiometric Scaling")
ax2a.scatter(crism_d["x"], crism_d["phobos_red"], color="tab:red", marker="x", alpha=0.7, label="CRISM Phobos red (Fraeman et al. 2014)")
ax2a.scatter(crism_d["x"], crism_d["phobos_blue"], color="tab:blue", marker="x", alpha=0.7, label="CRISM Phobos blue (Fraeman et al. 2014)")


order_text_on_plot = []
for name in selected_orders.keys():

    nominal_orders = np.asarray(selected_orders[name]["nominal_orders"])
    
    
    #convert order to micron for a given temperature grating e.g. -5C
    nominal_um = 10000.0 / np.asarray([nu_mp(order, 160.0, -5.0) for order in nominal_orders])
    
    
    
    
    interps = []
    
    for h5 in phobos_d.keys():
        orders = phobos_d[h5]["orders"]
        
        if list(orders) in selected_orders[name]["orders_to_find"]:
            
            norm_vals = phobos_d[h5]["norm_vals"]
            norm_vals_norm = norm_vals/np.max(np.mean(norm_vals[-3:]))
            norm_vals_interp = np.interp(nominal_orders, orders, norm_vals_norm)
        
            ax1a.scatter(nominal_um, norm_vals_interp)
            
            interps.append(norm_vals_interp)
    
    print(name, "%i observations found matching orders" %len(interps), selected_orders[name]["orders_to_find"])
    
    interps = np.asarray(interps)
    
    interp_mean = np.mean(interps, axis=0)
    interp_std = np.std(interps, axis=0)
    
    ax1a.errorbar(nominal_um, y=interp_mean, yerr=interp_std, color="k")
    
    orders_to_scale = selected_orders[name]["orders_to_scale"]
    orders_to_scale_um = 10000.0 / np.asarray([nu_mp(order, 160.0, -5.0) for order in orders_to_scale])
    order_ixs = [i for i, order in enumerate(nominal_orders) if order in orders_to_scale]
    
    # order_ums = [
    #     2.769762434224507519,
    #     2.669650539011573454,
    #     2.637868984975721531,
    #     2.606835232211300646,
    # ]
    # order_ixs = [0,3,4,5]
    
    #calculate scaling factor to calibrate to crism red
    matching_crism_indices = np.where((crism_d["x"] > np.min(orders_to_scale_um)) & (crism_d["x"] < np.max(orders_to_scale_um)))[0]
    crism_red_mean = np.mean(crism_d["phobos_red"][matching_crism_indices])
    crism_blue_mean = np.mean(crism_d["phobos_blue"][matching_crism_indices])
    lno_mean = np.mean(interp_mean[order_ixs])
    
    red_scalar = crism_red_mean / lno_mean
    blue_scalar = crism_blue_mean / lno_mean
    
    y_column_mean_norm_red = interp_mean*red_scalar
    y_column_mean_norm_blue = interp_mean*blue_scalar
    y_column_std_norm_red = interp_std*red_scalar
    y_column_std_norm_blue = interp_std*blue_scalar
    
    
    x_plt = nominal_um
    
    y_plt1 = interp_mean*red_scalar
    y_err1 = interp_std*red_scalar

    y_plt2 = interp_mean*blue_scalar
    y_err2 = interp_std*blue_scalar
    
    if "linestyle" in selected_orders[name].keys():
        ls = selected_orders[name]["linestyle"]
    
        ax2a.errorbar(x_plt, y=y_plt1, yerr=y_err1, color="darkred", capsize=2, ls=ls, label="LNO scaled to Phobos red")
        ax2a.errorbar(x_plt, y=y_plt2, yerr=y_err2, color="darkblue", capsize=2, ls=ls,label="LNO scaled to Phobos blue")

    ax2a.scatter(x_plt, y_plt1, color="darkred")
    ax2a.scatter(x_plt, y_plt2, color="darkblue")
    
    for x, y, order in zip(x_plt-0.015, y_plt2+(x_plt/40.0)-0.06, nominal_orders):
        #just label each order once
        if order not in order_text_on_plot:
            ax2a.text(x, y, order)
            order_text_on_plot.append(order)
    
ax2a.grid()
ax2a.legend()
ax2a.set_xlim((2, 3.1))
ax2a.set_ylim((0.025, 0.075))
ax2a.set_xlabel("Wavelength (microns)")
ax2a.set_ylabel("CRISM Phobos I/F")
# fig2.subplots_adjust(bottom=0.15)
    
fig2.savefig("lno_phobos_radiometric_calibration.png")
    
    
    
        
    # for h5 in phobos_d.keys():
        
    #     snr = phobos_d[h5]["norm_vals"] / phobos_d[h5]["norm_stds"]
    #     um = phobos_d[h5]["um"]
        
    #     orders = phobos_d[h5]["orders"]
    #     norm_vals = phobos_d[h5]["norm_vals"]
    #     norm_vals_norm = norm_vals/np.max(np.mean(norm_vals[-3:]))
    #     norm_vals_interp = np.interp(nominal_orders, orders, norm_vals_norm)
    
    #     ax1a.scatter(nominal_um, norm_vals_interp)
    
    
        # red_scaled = phobos_d[h5]["red_scaled"]
        # red_err_scaled = red_scaled / snr
        
        # blue_scaled = phobos_d[h5]["blue_scaled"]
        # blue_err_scaled = blue_scaled / snr
    
    
        
        # ax2a.errorbar(um, y=red_scaled, yerr=red_err_scaled, color="darkred", capsize=2, label=h5, alpha=0.5)
        # ax2a.scatter(um, red_scaled, color="darkred", alpha=0.5)
    
        # ax2a.errorbar(um, y=blue_scaled, yerr=blue_err_scaled, color="darkblue", capsize=2, label=h5, alpha=0.5)
        # ax2a.scatter(um, blue_scaled, color="darkblue", alpha=0.5)
    
stop()


poggiali_mixtures = np.asarray([
# x,SAC,sCR,MIX04 10,MIX05 10,MIX05 30,MIX 06 1
[1.99608,1.20269,1.08232,1.05671,1.1909,1.14201,1.17677 ],
[2.00453,1.20537,1.08379,1.05809,1.1946,1.14385,1.18076 ],
[2.0073,1.20625,1.08428,1.05854,1.19582,1.14446,1.18207 ],
[2.00785,1.20642,1.08437,1.05863,1.19606,1.14458,1.18233],
[2.00833,1.20657,1.08446,1.0587,1.19627,1.14468,1.18256 ],
[2.00842,1.2066,1.08447,1.05872,1.19631,1.1447,1.1826   ],
[2.07428,1.22845,1.09607,1.06939,1.22338,1.15942,1.21005],
[2.07807,1.22985,1.09675,1.07,1.22469,1.16031,1.21127   ],
[2.07812,1.22986,1.09676,1.07001,1.22471,1.16032,1.21128],
[2.08056,1.23078,1.0972,1.0704,1.22554,1.1609,1.21204   ],
[2.12427,1.24816,1.10511,1.07739,1.23841,1.1716,1.22426 ],
[2.13351,1.25172,1.1067,1.07885,1.24102,1.17396,1.22687 ],
[2.1398,1.25407,1.10777,1.07983,1.24285,1.17559,1.22873 ],
[2.14066,1.25439,1.10792,1.07997,1.2431,1.17582,1.22899 ],
[2.14638,1.25646,1.10889,1.08086,1.24482,1.17731,1.2308 ],
[2.15129,1.2582,1.10972,1.08161,1.24633,1.17861,1.23245 ],
[2.21003,1.27776,1.11944,1.08966,1.26602,1.19445,1.25657],
[2.22219,1.28174,1.12142,1.09096,1.27021,1.19777,1.26029],
[2.22791,1.28362,1.12234,1.09151,1.27216,1.19933,1.26177],
[2.2451,1.28933,1.1251,1.09289,1.27798,1.20402,1.26563  ],
[2.24979,1.2909,1.12585,1.09319,1.27955,1.2053,1.26659  ],
[2.27233,1.29848,1.12939,1.09419,1.28702,1.2114,1.27121 ],
[2.30372,1.30909,1.13422,1.09497,1.29719,1.21974,1.2786 ],
[2.31525,1.31297,1.13598,1.0953,1.30088,1.22275,1.2816  ],
[2.31944,1.31438,1.13661,1.09546,1.30222,1.22383,1.28273],
[2.34389,1.32259,1.14029,1.0972,1.30994,1.23002,1.28964 ],
[2.34565,1.32318,1.14056,1.09741,1.3105,1.23045,1.29017 ],
[2.36607,1.33002,1.14362,1.10095,1.31689,1.23542,1.29641],
[2.38524,1.33644,1.14651,1.10546,1.32286,1.23993,1.30267],
[2.40259,1.34224,1.14914,1.10923,1.32823,1.24386,1.30869],
[2.4135,1.34588,1.1508,1.11098,1.33161,1.24625,1.31272  ],
[2.41535,1.3465,1.15108,1.11122,1.33218,1.24665,1.31342 ],
[2.4697,1.36424,1.15966,1.11455,1.34913,1.25743,1.33388 ],
[2.4704,1.36446,1.15978,1.11458,1.34936,1.25755,1.3341  ],
[2.47259,1.36515,1.16014,1.11467,1.35005,1.25794,1.33476],
[2.48816,1.37003,1.16277,1.11545,1.35501,1.2606,1.33909 ],
[2.48992,1.37059,1.16307,1.11556,1.35558,1.2609,1.33955 ],
[2.49767,1.37301,1.16444,1.11615,1.35808,1.26218,1.34152],
[2.54738,1.38986,1.17488,1.12508,1.37562,1.27173,1.35305],
[2.55068,1.39114,1.17576,1.12597,1.37699,1.27266,1.35379],
[2.55262,1.3919,1.1763,1.12649,1.37782,1.27325,1.35423  ],
[2.55412,1.3925,1.17673,1.12689,1.37848,1.27372,1.35456 ],
[2.5613,1.39547,1.17903,1.1288,1.3819,1.27638,1.35618   ],
[2.56565,1.39734,1.18067,1.12998,1.3843,1.27839,1.35715 ],
[2.57253,1.40039,1.18345,1.13211,1.38917,1.28229,1.35869],
[2.57497,1.4015,1.18441,1.13302,1.39142,1.2839,1.35924  ],
[2.59172,1.40946,1.18994,1.13926,1.40302,1.29618,1.36297],
[2.59535,1.41124,1.19084,1.14003,1.40372,1.29822,1.36377],
[2.60175,1.41442,1.19212,1.14091,1.40455,1.3005,1.36516 ],
[2.60309,1.41508,1.19234,1.14103,1.40469,1.30077,1.36545],
[2.60968,1.41836,1.19315,1.14139,1.40526,1.3013,1.36683 ],
[2.61825,1.42259,1.1937,1.14129,1.40589,1.30082,1.36854 ],
[2.62439,1.42549,1.19384,1.14072,1.40634,1.30008,1.37008],
[2.63167,1.42869,1.1938,1.13906,1.40695,1.29911,1.37286 ],
[2.63735,1.43092,1.19368,1.13784,1.40756,1.29853,1.37524],
[2.64178,1.43249,1.19359,1.13781,1.40816,1.2984,1.37711 ],
[2.64563,1.43375,1.19362,1.13829,1.40882,1.29878,1.37871],
[2.6503,1.43516,1.19412,1.13914,1.40991,1.30055,1.38063 ],
[2.65537,1.43657,1.19593,1.14015,1.41169,1.30487,1.38265],
[2.6677,1.43967,1.20278,1.14191,1.42232,1.31256,1.38709 ],
[2.66794,1.43972,1.20284,1.14191,1.42266,1.31258,1.38717],
[2.67053,1.44033,1.20315,1.14178,1.42532,1.31257,1.38795],
[2.67067,1.44036,1.20315,1.14176,1.42539,1.31256,1.38799],
[2.6714,1.44053,1.20311,1.14165,1.42565,1.3125,1.3882   ],
[2.67205,1.44068,1.20302,1.14151,1.42576,1.31243,1.38837],
[2.68021,1.44253,1.19881,1.13456,1.42447,1.31063,1.38968],
[2.68588,1.44377,1.19322,1.1202,1.42353,1.30866,1.38887 ],
[2.68798,1.44423,1.19063,1.11332,1.42328,1.30782,1.38808],
[2.69081,1.44484,1.18647,1.10345,1.42303,1.30659,1.38666],
[2.6951,1.44575,1.17559,1.08848,1.42282,1.30455,1.384   ],
[2.69573,1.44589,1.17251,1.08636,1.4228,1.30423,1.38358 ],
[2.69759,1.44628,1.16188,1.08029,1.42278,1.30326,1.38234],
[2.69786,1.44634,1.16072,1.07942,1.42278,1.30311,1.38216],
[2.70162,1.44713,1.15231,1.0676,1.42283,1.301,1.37964   ],
[2.70208,1.44722,1.15166,1.06619,1.42284,1.30073,1.37933],
[2.70285,1.44739,1.15063,1.06383,1.42287,1.30027,1.37882],
[2.70702,1.44826,1.14547,1.05063,1.42309,1.29762,1.37611],
[2.71073,1.44903,1.13986,1.04038,1.42337,1.29499,1.37381],
[2.71142,1.44917,1.13848,1.03917,1.42343,1.29447,1.37341],
[2.71485,1.44988,1.13263,1.03457,1.42376,1.29167,1.37144],
[2.72224,1.4514,1.12807,1.02825,1.4246,1.28431,1.36781  ],
[2.73401,1.4538,1.126,1.0215,1.42624,1.2704,1.36511     ],
[2.73577,1.45416,1.12587,1.02072,1.42651,1.26867,1.36533],
[2.73677,1.45437,1.12582,1.02031,1.42667,1.26777,1.36556],
[2.75343,1.45776,1.12687,1.01804,1.42953,1.25815,1.38187],
[2.75462,1.458,1.1272,1.01833,1.42975,1.25766,1.38329   ],
[2.75613,1.45831,1.12774,1.01881,1.43003,1.25706,1.38492],
[2.7644,1.45999,1.13576,1.02252,1.43163,1.25404,1.39087 ],
[2.76955,1.46105,1.13847,1.0246,1.43266,1.25226,1.39332 ],
[2.77282,1.46172,1.13935,1.02551,1.43334,1.25114,1.39473],
[2.77537,1.46224,1.13983,1.02598,1.43387,1.25028,1.39581],
[2.78404,1.46403,1.14072,1.02656,1.43573,1.24755,1.3997 ],
[2.78553,1.46434,1.14079,1.02655,1.43605,1.24713,1.40043],
[2.79256,1.46581,1.141,1.02623,1.43763,1.2454,1.40415   ],
[2.79895,1.46715,1.14107,1.02568,1.4391,1.24436,1.40778 ],
[2.80971,1.46944,1.14123,1.02457,1.44167,1.24451,1.41411],
[2.81743,1.4711,1.14157,1.02407,1.44357,1.24704,1.41864 ],
[2.82506,1.47276,1.14239,1.02414,1.44549,1.25143,1.42296],
[2.82965,1.47377,1.14317,1.02452,1.44667,1.25442,1.42541],
[2.84214,1.47656,1.14619,1.02649,1.44995,1.26146,1.43121],
[2.84278,1.4767,1.14635,1.02662,1.45012,1.26174,1.43147 ],
[2.84977,1.4783,1.14813,1.02824,1.452,1.26432,1.43408   ],
[2.86286,1.48134,1.15074,1.03064,1.45558,1.26764,1.43808],
[2.87368,1.48393,1.15222,1.03105,1.45859,1.26953,1.44092],
[2.87469,1.48418,1.15233,1.03108,1.45888,1.26968,1.44118],
[2.88008,1.48549,1.15288,1.03124,1.46039,1.27043,1.44251],
[2.89149,1.48833,1.15389,1.03162,1.46361,1.27179,1.44526],
[2.89408,1.48898,1.15411,1.03173,1.46434,1.27207,1.44588],
[2.89625,1.48953,1.15429,1.03183,1.46496,1.27231,1.44639],
[2.92603,1.49727,1.15736,1.03457,1.47338,1.27634,1.4533 ],
[2.93515,1.49969,1.15866,1.03596,1.47595,1.27826,1.45539],
[2.93965,1.50089,1.1594,1.03673,1.47721,1.27935,1.45642 ],
[2.95219,1.50426,1.16175,1.03901,1.48072,1.28284,1.45932],
[2.95636,1.50538,1.16262,1.0398,1.48188,1.28413,1.46029 ],
[2.97901,1.51147,1.16789,1.04463,1.48808,1.29199,1.4656 ],
[2.9791,1.51149,1.16791,1.04465,1.4881,1.29202,1.46562  ],
[2.98462,1.51297,1.16931,1.04611,1.48959,1.29407,1.46694],
[2.99463,1.51564,1.1719,1.04904,1.49226,1.29784,1.46935 ],
[2.99557,1.51589,1.17215,1.04932,1.49251,1.2982,1.46957 ],
])

ref_ix = 55
# ref_ix = 14
ref_refl = 0.0333
ref_refl = 0.04325
ax2a.plot(poggiali_mixtures[:, 0], poggiali_mixtures[:, 1]/poggiali_mixtures[ref_ix, 1]*ref_refl, label="SAC")
ax2a.plot(poggiali_mixtures[:, 0], poggiali_mixtures[:, 2]/poggiali_mixtures[ref_ix, 2]*ref_refl, label="sCR")
ax2a.plot(poggiali_mixtures[:, 0], poggiali_mixtures[:, 3]/poggiali_mixtures[ref_ix, 3]*ref_refl, label="Simulant 10% mix4")
ax2a.plot(poggiali_mixtures[:, 0], poggiali_mixtures[:, 4]/poggiali_mixtures[ref_ix, 4]*ref_refl, label="Simulant 10% mix5")
ax2a.plot(poggiali_mixtures[:, 0], poggiali_mixtures[:, 5]/poggiali_mixtures[ref_ix, 5]*ref_refl, label="Simulant 30% mix5")
ax2a.plot(poggiali_mixtures[:, 0], poggiali_mixtures[:, 6]/poggiali_mixtures[ref_ix, 6]*ref_refl, label="Simulant 1% mix6")
plt.legend()

fig2.savefig("lno_phobos_radiometric_calibration_mixtures.png")
