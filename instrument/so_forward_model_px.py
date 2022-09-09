# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:58:37 2022

@author: iant

PIXEL-BASED FORWARD MODEL

EACH PIXEL HAS A HR GRID WITH ALL WAVENUMBERS HITTING IT IN ALL ORDERS
THEN EACH PIXEL IN EACH ORDER HAS AN AOTF SCALAR AND BLAZE SCALAR PER PIXEL

SPECTRAL CAL
ILS PER PIXEL

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from tools.plotting.colours import get_colours

from tools.spectra.molecular_spectrum_so import get_xsection
from tools.datasets.get_gem_data import get_gem_data



def lt22_waven(order, t, pixels, channel="so", coeffs=False):
    """spectral calibration Loic Feb 22. Get pixel wavenumbers from order + temperature"""

    def lt22_p0_shift(t):
        """first pixel (temperature shift) Loic Feb 22"""
        
        p0 = -0.8276 * t #px/°C * T(interpolated)
        return p0


    
    px_shifted = pixels + lt22_p0_shift(t)

    cfpixel = {"so":[3.32e-8, 5.480e-4, 22.4701], "lno":[3.32e-8, 5.480e-4, 22.4701]}
    xdat  = np.polyval(cfpixel[channel], px_shifted) * order
    
    if coeffs:
        return cfpixel[channel]
    
    else:
        return xdat





"""order to aotf frequency"""
A_aotf = {
    110:14332, 111:14479, 112:14627, 113:14774, 114:14921, 115:15069, 116:15216, 117:15363, 118:15510, 119:15657,
    120:15804, 121:15951, 122:16098, 123:16245, 124:16392, 125:16539, 126:16686, 127:16832, 128:16979, 129:17126,
    130:17273, 131:17419, 132:17566, 133:17712, 134:17859, 135:18005, 136:18152, 137:18298, 138:18445, 139:18591,
    140:18737, 141:18883, 142:19030, 143:19176, 144:19322, 145:19468, 146:19614, 147:19761, 148:19907, 149:20052,
    150:20198, 151:20344, 152:20490, 153:20636, 154:20782, 155:20927, 156:21074, 157:21219, 158:21365, 159:21510,
    160:21656, 161:21802, 162:21947, 163:22093, 164:22238, 165:22384, 166:22529, 167:22674, 168:22820, 169:22965,
    170:23110, 171:23255, 172:23401, 173:23546, 174:23691, 175:23836, 176:23981, 177:24126, 178:24271, 179:24416,
    180:24561, 181:24706, 182:24851, 183:24996, 184:25140, 185:25285, 186:25430, 187:25575, 188:25719, 189:25864,
    190:26008, 191:26153, 192:26297, 193:26442, 194:26586, 195:26731, 196:26875, 197:27019, 198:27163, 199:27308,
    200:27452, 201:27596, 202:27740, 203:27884, 204:28029, 205:28173, 206:28317, 207:28461, 208:28605, 209:28749,
    210:28893,
}

def aotf_freq_to_order(aotf_khz):
    """aotf frequency kHz to nearest order"""
    res_key, res_val = min(A_aotf.items(), key=lambda x: abs(aotf_khz - x[1]))
    return res_key





def F_ils(hr_grid, width, displacement, amplitude):

    #make ils shape
    a1 = 0.0
    a2 = width
    a3 = 1.0
    a4 = displacement
    a5 = width
    a6 = amplitude
        
    ils0 = a3 * np.exp(-0.5 * ((hr_grid + a1) / a2) ** 2)
    ils1 = a6 * np.exp(-0.5 * ((hr_grid + a4) / a5) ** 2)
    ils = ils0 + ils1 

    return ils





def get_ils_params(aotff, pixels):
    """get ils params"""
    

    #from ils.py on 6/7/21
    amp = 0.2724371566666666 #intensity of 2nd gaussian
    rp = 16939.80090831571 #resolving power cm-1/dcm-1
    disp_3700 = [-3.06665339e-06,  1.71638815e-03,  1.31671485e-03] #displacement of 2nd gaussian cm-1 w.r.t. 3700cm-1 vs pixel number
    
    spec_res = aotff / rp #wavenumber / resolving power = spectral resolution
    sconv = spec_res/2.355
    
    
    disp_3700_nu = np.polyval(disp_3700, pixels) #displacement at 3700cm-1
    disp_order = disp_3700_nu / -3700.0 * aotff #displacement adjusted for wavenumber
    
    width = sconv
    displacement = disp_order #1 value per pixel
    amplitude = amp
    
    return width, displacement, amplitude




def ils_px(width, displacement, amplitude):
    """make ils functions for a single pixel in one order"""

    d_nu = 0.001
    nu_range = 0.7
    
    
    hr_grid = np.arange(-nu_range, nu_range, d_nu)
    
    ils = F_ils(hr_grid, width, displacement, amplitude)

    return hr_grid, ils



def aotf_shape(aotf_khz, t, orders):
    #AOTF t correction
    aotfts  = -6.5278e-5                                   # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
    aotf_shift = aotfts * t
    
    
    
    #AOTF in cm-1
    cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])                  # Frequency of AOTF [cm-1 from kHz]
    aotff = np.polyval(cfaotf, aotf_khz)
    
    aotff += aotf_shift
    
    d_aotf = {}
    d_aotf["aotf_centre"] = aotff
    d_aotf["aotf_shift"] = aotf_shift
    
    d_aotf["aotffs"] = {}
    
    
    #AOTF shape
    #2nd october slack
    aotfwc  = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    aotfsc  = [ 8.10749274e-07, -3.30238496e-03,  4.08845247e+00] # sidelobes factor [scaler from AOTF frequency cm-1]
    aotfac  = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
    aotfgc  = [ 1.49266526e-07, -9.63798656e-04,  1.60097815e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]
    
    d_aotf["aotfw"] = np.polyval(aotfwc, aotff)
    d_aotf["aotfs"] = np.polyval(aotfsc, aotff)
    d_aotf["aotfa"] = np.polyval(aotfac, aotff)
    d_aotf["aotfo"] = np.polyval(aotfoc,aotff)
    d_aotf["aotfg"] = np.polyval(aotfgc,aotff)
    d_aotf["aotfgw"] = 50. #offset width cm-1



    #find aotf centre cm-1 for all orders for blaze calcs
    for order in orders:
        aotf_khz_m = A_aotf[order]
        aotff_m = np.polyval(cfaotf, aotf_khz_m) + aotf_shift #use same shift
        d_aotf["aotffs"][order] = aotff_m
    
    return d_aotf




def sinc_gd(dx, width, lobe, asym, offset):
    """new spectral calibration functions Aug/Sep 2021"""
    #goddard version
    sinc = (width * np.sin(np.pi * dx / width)/(np.pi * dx))**2.0
    ind = (abs(dx) > width).nonzero()[0]
    if len(ind) > 0: sinc[ind] = sinc[ind] * lobe
    ind = (dx <= -width).nonzero()[0]
    if len(ind) > 0: sinc[ind] = sinc[ind] * asym
    sinc += offset
    return sinc




def F_aotf3(dx, d):
    offset = d["aotfg"] * np.exp(-dx**2.0 / (2.0 * d["aotfgw"]**2.0))
    sinc = sinc_gd(dx, d["aotfw"], d["aotfs"], d["aotfa"], offset)
    return sinc




def F_blaze3(x, blazef, blazew):
    dx = x - blazef
    F = np.sinc((dx) / blazew)**2
    return F


def make_path_lengths(zs):
    radius = 3396.
    pl = []
    for z in zs:
        pl.append(2 * (np.sqrt((radius + z)**2 - (radius + zs[0])**2)) - np.sum(pl))
    pl.pop(0)
    pl.append(pl[-1])

    pl = np.asfarray(pl)
    pl *= 1e5 #cm?
    return pl




# aotf_khz = 25864.
aotf_khz = 25285. #order 185
t_nomad = -5.0

molecule = "CO"
xa = 1.0 * 1.0e-6

atmos_dict = get_gem_data(36, 180.0, 0.0, 0.0, 12.0, plot=False)

atmos_ix = 44
z_alt = atmos_dict["z"][atmos_ix]
p = atmos_dict["p"][atmos_ix]
t = atmos_dict["t"][atmos_ix]
nd = atmos_dict["nd"][atmos_ix]


z_max = z_alt + 30.001 #6 heights above only
dz = 5.
zs = np.arange(z_alt, z_max, dz)
ps = np.zeros_like(zs)
nds = np.zeros_like(zs)

scale_height = 11.1

for z_ix, z in enumerate(zs): #loop through altitudes above chosen alt
    exp = np.exp(-(z-z_alt) / scale_height)
    ps[z_ix] = p * exp
    nds[z_ix] = nd * exp


pls = make_path_lengths(zs)


# def so_cal(aotf_khz, t):
"""
#FSRpeak = P0 + P1dv + P2dv2 + P3dv3, where dv is vAOTF 3700 cm 1, and P0, P1, P2, P3 are 2.25863468E+01, 9.79270239E 06,  7.20616355E 09, and  1.00162255E 11 respectively 
#FSRi = vi/m = F0 + F1·i + F2·i2, where i is the pixel number (0 to 319), vi is the frequency at pixel i,
#m is the order, and the coefficients F0, F1 and F2 are 2.24734E+01, 5.55953E 04, 1.75128E 08 respectively.
#The order number (m) is simply the AOTF frequency (in wavenumbers) divided by the FSR at that pixel.
#grooves spacing should expand as FSR’ = FSR·[1+K(T)], where K(T) is a scaling correction factor for temperature T [ºC]. In our analysis of all calibration data from 2018 to 2021 of FSRpeak from the full-scans and the line-positions from the mini-scans, we observe the same K(T) function, which we determine to be: K(T) = K0 + K1T + K2T2, where K0, K1 and K2 are  1.90001923E 04,  2.30708836E 05, and  2.44383699E 07 respectively 
"""

pixels = np.arange(0., 320., 10.0)
n_orders = 3
centre_order = aotf_freq_to_order(aotf_khz)
orders = [int(order) for order in np.arange(centre_order - n_orders, centre_order + n_orders + 1, 1)]

colours = get_colours(len(orders), colours=["blue", "yellow", "red"])



d_aotf = {**aotf_shape(aotf_khz, t_nomad, orders), **{int(px):{int(order):{} for order in orders} for px in pixels}}


d_px = {int(px):{int(order):{} for order in orders} for px in pixels}
d_ils = {int(px):{int(order):{} for order in orders} for px in pixels}
d_blaze = {int(px):{int(order):{} for order in orders} for px in pixels}

d_px2 = {}


#make HR ILS grid 1 x order per pixel

for order_ix, order in enumerate(orders):

    aotff = d_aotf["aotffs"][order]

    #ils 
    width, displacements, amplitude = get_ils_params(aotff, pixels)

    for px_ix, pixel in enumerate(pixels):
        d_ils[pixel][order] = {"width":width, "displacement":displacements[px_ix], "amplitude":amplitude}
        

    #blaze width
    blazep  = [-1.00162255e-11, -7.20616355e-09, 9.79270239e-06, 2.25863468e+01] # Dependence of blazew from AOTF frequency
    blazew =  np.polyval(blazep, aotff - 3700.0)        # FSR (Free Spectral Range), blaze width [cm-1]

    #blaze temperature shift
    ncoeff  = [-2.44383699e-07, -2.30708836e-05, -1.90001923e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
    blazew += blazew * np.polyval(ncoeff, t_nomad)        # FSR, corrected for temperature {1+K(t)}
    
    
    #GV wavenumber cal
    # cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])            # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    # pixf = np.polyval(cfpixel,np.arange(320)) * order #pixel wavenumbers
    # pixf += pixf * np.polyval(ncoeff, t_nomad) #pixel wavenumbers temp corrected
    
    #LT wavenumber cal
    pixf = lt22_waven(order, t_nomad, pixels)
               
               
    blazef = order * blazew #center of the blaze
    
    #make blaze function, one value per pixel
    blaze = F_blaze3(pixf, blazef, blazew)
    
    #make aotf, one value per pixel
    dx = pixf - d_aotf["aotf_centre"]
    aotf = F_aotf3(dx, d_aotf)

    for px_ix, pixel in enumerate(pixels):
        d_ils[pixel][order]["pixf"] = pixf[px_ix] #pixel centres
        d_blaze[pixel][order] = blaze[px_ix] #blaze
        d_aotf[pixel][order] = aotf[px_ix] #aotf



        hr_grid, ils = ils_px(
            d_ils[pixel][order]["width"], 
            d_ils[pixel][order]["displacement"], 
            d_ils[pixel][order]["amplitude"]
        )


        d_px[pixel][order] = {"nu_hr":hr_grid + pixf[px_ix], "ils":ils}
        
        #make hr nu grid (1 for each px containing all orders)
        if pixel not in d_px2.keys():
            empty_nu_grid = np.zeros((len(hr_grid)*len(orders)))
            empty_ils_grid = np.zeros((len(hr_grid)*len(orders)))
            empty_orders = np.zeros(len(orders))
            empty_ix_range = np.zeros((len(orders), 2), dtype=int)

            d_px2[pixel] = {"nu_hr":empty_nu_grid, "ils":empty_ils_grid, "order":empty_orders, "ix_range":empty_ix_range}
            
        #record start/end indices
        ix_range = [order_ix * len(hr_grid), (order_ix + 1) * len(hr_grid)]

        d_px2[pixel]["nu_hr"][order_ix * len(hr_grid):(order_ix + 1) * len(hr_grid)] = hr_grid + pixf[px_ix]
        d_px2[pixel]["ils"][order_ix * len(hr_grid):(order_ix + 1) * len(hr_grid)] = ils
        d_px2[pixel]["order"][order_ix] = order
        d_px2[pixel]["ix_range"][order_ix] = ix_range





plt.figure()
plt.title("AOTF and blaze contribution across all orders")
plt.xlabel("Wavenumber cm-1")
plt.ylabel("Relative contribution")
#plot all orders
print("Getting molecules")
for pixel in pixels:
    for order_ix, order in enumerate(orders):
    
        pixf = d_ils[pixel][order]["pixf"] #single value cm-1 of pixel
        plt.scatter(pixf, d_aotf[pixel][order], color="k", marker=".", label="Order %s" %order)
        plt.scatter(pixf, d_blaze[pixel][order], color=colours[order_ix], marker=".")



    #run through every pixel (orders grouped), get sigma for each altitude layer
    d_px2[pixel]["sigma"] = {}
    d_px2[pixel]["tau"] = np.zeros_like(d_px2[pixel]["nu_hr"])
    for z, p, nd, pl in zip(zs, ps, nds, pls):
        sigma_hr = get_xsection(molecule, d_px2[pixel]["nu_hr"], Smin=1.0e-25, temperature=t, pressure=p*1e2)
        d_px2[pixel]["sigma"][int(z)] = sigma_hr

        tau_hr = xa * nd * pl * sigma_hr
        d_px2[pixel]["tau"] += tau_hr
        
    d_px2[pixel]["mol"] = np.exp(-d_px2[pixel]["tau"])

    
    toa_hr = np.ones_like(d_px2[pixel]["mol"])
    d_px2[pixel]["toa"] = toa_hr


# plt.figure()
for pixel in pixels:
    for order_ix, order in enumerate(orders):
        
        ix_range = np.arange(d_px2[pixel]["ix_range"][order_ix, 0], d_px2[pixel]["ix_range"][order_ix, 1])
        d_px[pixel][order]["mol"] = d_px2[pixel]["mol"][ix_range]
        d_px[pixel][order]["toa"] = d_px2[pixel]["toa"][ix_range]
    

    # plt.plot(d_px2[pixel]["nu_hr"], d_px2[pixel]["mol"])
    # plt.plot(d_px2[pixel]["nu_hr"], d_px2[pixel]["toa"])
    



plt.figure()
plt.title("Relative contribution of different orders to signal")
plt.xlabel("Pixel number")
plt.ylabel("Raw contribution")
#each pixel (orders grouped)
#convolve everything
d = {int(px):{int(order):{} for order in orders} for px in pixels}
d_atm = {int(px):{int(order):{} for order in orders} for px in pixels}
d_toa = {int(px):{int(order):{} for order in orders} for px in pixels}
y_atm = np.ones_like(pixels)
y_toa = np.ones_like(pixels)
for pixel_ix, pixel in enumerate(pixels):
    for order_ix, order in enumerate(orders):
        
        # ix_range = np.arange(d_px2[pixel]["ix_range"][order_ix, 0], d_px2[pixel]["ix_range"][order_ix, 1])
    
        atm = np.sum(d_px[pixel][order]["ils"] * d_px[pixel][order]["mol"]) * d_aotf[pixel][order] * d_blaze[pixel][order]
        toa = np.sum(d_px[pixel][order]["ils"] * d_px[pixel][order]["toa"]) * d_aotf[pixel][order] * d_blaze[pixel][order]

    
        d_atm[pixel][order] = atm
        d_toa[pixel][order] = toa
        # d[pixel][order] = atm / toa #doesn't work - must do order addition first
        
        #sum up raw spectra for all orders
        y_atm[pixel_ix] += atm
        y_toa[pixel_ix] += toa
        
        if pixel_ix == 0:
            label = "Order %s" %order
        else:
            label = ""

        plt.scatter(pixel, atm, color=colours[order_ix], marker=".", label=label)
        plt.scatter(pixel, toa, color=colours[order_ix], marker=".")
    
#scale contribution from each order to relative level
for pixel_ix, pixel in enumerate(pixels):
    for order_ix, order in enumerate(orders):
        
        line_strength = 1.0 - (d_atm[pixel][order] / d_toa[pixel][order])
        scalar_toa = d_toa[pixel][order] / np.sum([d_toa[pixel][order] for order in orders])
        scalar_atm = d_atm[pixel][order] / np.sum([d_atm[pixel][order] for order in orders])
        
        d[pixel][order] = 1.0 - line_strength * np.mean([scalar_toa, scalar_atm])


plt.plot(pixels, y_atm, "k")
plt.plot(pixels, y_toa, "k")
plt.legend()
plt.grid()





#cm-1 for centre order
x = [d_ils[pixel][centre_order]["pixf"] for pixel in pixels]


fig1, ax1 = plt.subplots()
fig1.suptitle("Convolved spectrum order contributions")
ax1.set_xlabel("Wavenumber cm-1")
ax1.set_ylabel("Absorption strength")


y_old = np.zeros_like(x)
for order_ix, order in enumerate(orders):
    y = np.array([d[pixel][order] - 1.0 for pixel in pixels]) #subtract 1 to make bar chart work correctly
    
    ax1.bar(x, y, bottom=y_old, width=1/13, color=colours[order_ix], linewidth=0, label="Order %i" %order)
    
    y_old += y


y_sim = y_atm / y_toa
ax1.plot(x, y_sim -1.0)

ax1.grid()
ax1.legend()

#change labels to add 1
ticks = ax1.get_yticks()
ax1.set_yticklabels(ticks+1.0)

import json
with open("%s_z=%0.1f_t=%0.1f_p=%0.1f.json" %(molecule, z, t, p), "w") as f:
    f.write(json.dumps(d, indent=2))


# spectrum = np.zeros_like(pixels)
# for px_ix, pixel in enumerate(pixels):
#     spectrum[px_ix] = np.sum([d[pixel][order] for order in orders])

# x = [d_ils[pixel][centre_order]["pixf"] for pixel in pixels]

# plt.plot(x, spectrum)
    