# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:04:31 2022

@author: iant

GET HIGH RES (HR) ABSORPTION LINES
MAKE HR AOTF FUNCTION AND MULTIPLY HR LINES

FOR EACH PIXEL:
    FOR EACH ORDER:
        CONVOLVE HR LINES, MULTIPLY BY BLAZE
        

    SUM UP ALL ORDERS
    


"""

import numpy as np
import matplotlib.pyplot as plt






def lt22_waven(order, t, channel="so", coeffs=False):
    """spectral calibration Loic Feb 22. Get pixel wavenumbers from order + temperature"""

    def lt22_p0_shift(t):
        """first pixel (temperature shift) Loic Feb 22"""
        
        p0 = -0.8276 * t #px/°C * T(interpolated)
        return p0


    
    px_shifted = np.arange(320.0) + lt22_p0_shift(t)

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





def ils_hr_grid(hr_grid, width, displacement, amplitude):

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





def get_ils_params(aotff):
    """get ils params"""
    
    pixels = np.arange(320.0)

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




def ils_order(nu_hr, nu_px, width, displacements, amplitude):
    """make ils functions for a single order"""
    
    
    ils_grid = np.zeros((len(nu_px), len(nu_hr)))
    
    for px, px_nu in enumerate(nu_px):
        displacement = displacements[px]

        #find indices in hr grid covering +- 0.5 cm-1 of pixel wavenumber
        nu_hr_start_ix = np.searchsorted(nu_hr, px_nu - 0.5) #start index
        nu_hr_end_ix = np.searchsorted(nu_hr, px_nu + 0.5) #end index
        
        hr_grid = nu_hr[nu_hr_start_ix:nu_hr_end_ix] - px_nu
        
        ils = ils_hr_grid(hr_grid, width, displacement, amplitude)

        ils_grid[px, nu_hr_start_ix:nu_hr_end_ix] = ils
    
    return ils_grid




def blaze_conv(nu_hr, aotff):
    #make blaze convolution function for each pixel
    
    nu_hr = d["nu_hr"]
    pixels = d["pixels"]

    W_conv = np.zeros((len(pixels), len(nu_hr)))
    
    for iord in d["orders"]:
        nu_p = d[iord]["pixf"]
        W_blaze = d[iord]["F_blaze"]
        
        # print('order %d: %.1f to %.1f' % (iord, nu_p[0], nu_p[-1]))
        
        for ip in pixels:
            inu1 = np.searchsorted(nu_hr, nu_p[ip] - 0.5) #start index
            inu2 = np.searchsorted(nu_hr, nu_p[ip] + 0.5) #end index
            
            hr_grid = nu_hr[inu1:inu2] - nu_p[ip]
            
            
        
            W_conv[ip,inu1:inu2] += (W_blaze[ip]) * ils
            # W_conv[ip,inu1:inu2] += (W_blaze[ip] * dnu)/(np.sqrt(2.0 * np.pi) * sconv) * np.exp(-(nu_hr[inu1:inu2] - nu_p[ip])**2 / (2. *sconv**2))
            # if ip == 319:
            #     plt.plot(nu_sp, ils + iord/1000.)
            #     plt.plot(nu_sp, (W_blaze[ip] * dnu)/(np.sqrt(2.0 * np.pi) * sconv) * np.exp(-(nu_hr[inu1:inu2] - nu_p[ip])**2 / (2. *sconv**2)) + iord/1000.)
    
    d["W_conv"] = W_conv
    
    return d




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
    offset = d["aotf"]["aotfg"] * np.exp(-dx**2.0 / (2.0 * d["aotf"]["aotfgw"]**2.0))
    sinc = sinc_gd(dx, d["aotf"]["aotfw"], d["aotf"]["aotfs"], d["aotf"]["aotfa"], offset)
    return sinc




def F_blaze3(x, blazef, blazew):
    dx = x - blazef
    F = np.sinc((dx) / blazew)**2
    return F





if True:
    aotf_khz = 25285.
    t = -5.0
# def so_cal(aotf_khz, t):
    """
    #FSRpeak = P0 + P1dv + P2dv2 + P3dv3, where dv is vAOTF 3700 cm 1, and P0, P1, P2, P3 are 2.25863468E+01, 9.79270239E 06,  7.20616355E 09, and  1.00162255E 11 respectively 
    #FSRi = vi/m = F0 + F1·i + F2·i2, where i is the pixel number (0 to 319), vi is the frequency at pixel i,
    #m is the order, and the coefficients F0, F1 and F2 are 2.24734E+01, 5.55953E 04, 1.75128E 08 respectively.
    #The order number (m) is simply the AOTF frequency (in wavenumbers) divided by the FSR at that pixel.
    #grooves spacing should expand as FSR’ = FSR·[1+K(T)], where K(T) is a scaling correction factor for temperature T [ºC]. In our analysis of all calibration data from 2018 to 2021 of FSRpeak from the full-scans and the line-positions from the mini-scans, we observe the same K(T) function, which we determine to be: K(T) = K0 + K1T + K2T2, where K0, K1 and K2 are  1.90001923E 04,  2.30708836E 05, and  2.44383699E 07 respectively 
    """

    n_orders = 3
    centre_order = aotf_freq_to_order(aotf_khz)
    orders = [int(order) for order in np.arange(centre_order - n_orders, centre_order + n_orders + 1, 1)]

    d = {"aotf":{}, "orders":orders, "centre_order":centre_order}


    

        

    #AOTF t correction
    aotfts  = -6.5278e-5                                   # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
    aotf_shift = aotfts * t


    
    #AOTF in cm-1
    cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])                  # Frequency of AOTF [cm-1 from kHz]
    aotff = np.polyval(cfaotf, aotf_khz)
    
    aotff += aotf_shift
    
    d["aotf"]["aotf_centre"] = aotff
    d["aotf"]["aotf_shift"] = aotf_shift


    #AOTF shape
    #2nd october slack
    aotfwc  = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    aotfsc  = [ 8.10749274e-07, -3.30238496e-03,  4.08845247e+00] # sidelobes factor [scaler from AOTF frequency cm-1]
    aotfac  = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
    aotfgc  = [ 1.49266526e-07, -9.63798656e-04,  1.60097815e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]

    d["aotf"]["aotfw"] = np.polyval(aotfwc, aotff)
    d["aotf"]["aotfs"] = np.polyval(aotfsc, aotff)
    d["aotf"]["aotfa"] = np.polyval(aotfac, aotff)
    d["aotf"]["aotfo"] = np.polyval(aotfoc,aotff)
    d["aotf"]["aotfg"] = np.polyval(aotfgc,aotff)
    d["aotf"]["aotfgw"] = 50. #offset width cm-1

    


    

    for order in orders:
        d[order] = {}

        aotf_khz= A_aotf[order] + aotf_shift
        aotff = np.polyval(cfaotf, aotf_khz)
        d[order]["aotff"] = aotff
        #blaze width
        blazep  = [-1.00162255e-11, -7.20616355e-09, 9.79270239e-06, 2.25863468e+01] # Dependence of blazew from AOTF frequency
        blazew =  np.polyval(blazep, aotff - 3700.0)        # FSR (Free Spectral Range), blaze width [cm-1]
    
        #blaze temperature shift
        ncoeff  = [-2.44383699e-07, -2.30708836e-05, -1.90001923e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
        blazew += blazew * np.polyval(ncoeff, t)        # FSR, corrected for temperature {1+K(t)}
        d[order]["blazew"] = blazew
        
        
        #GV wavenumber cal
        # cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])            # Blaze free-spectral-range (FSR) [cm-1 from pixel]
        # pixf = np.polyval(cfpixel,np.arange(320)) * order #pixel wavenumbers
        # pixf += pixf * np.polyval(ncoeff, t) #pixel wavenumbers temp corrected
        
        #LT wavenumber cal
        pixf = lt22_waven(order, t)
        d[order]["pixf"] = pixf
                   
                   
        blazef = order * blazew #center of the blaze
        d[order]["blazef"] = blazef
        # print(order, blazef, blazew)
        
        #make blaze function
        blaze = F_blaze3(pixf, blazef, blazew)
        d[order]["F_blaze"] = blaze

        #ils 
        width, displacements, amplitude = get_ils_params(aotff)
        d[order]["ils"] = {"width":width, "displacements":displacements, "amplitude":amplitude}
        
    
    nu_min = d[orders[0]]["pixf"][0] - 5.0
    nu_max = d[orders[-1]]["pixf"][-1] + 5.0

    

    #make AOTF
    nu_hr = np.arange(nu_min, nu_max, 0.001)
    dx = nu_hr - d["aotf"]["aotf_centre"]
    d["aotf"]["F_aotf"] = F_aotf3(dx, d)
    d["aotf"]["nu_hr"] = nu_hr


    #don't make ils grid (ils per pixel on hr grid). Instead loop through and convolve ILS per pixel per order
    
    # for order in orders:
    #     d[order]["ils_grid"] = ils_order(nu_hr, pixf, \
    #           d[order]["ils"]["width"], d[order]["ils"]["displacements"], d[order]["ils"]["amplitude"])


    
    # F_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan
    # nu_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan
    
    # for i, order in enumerate(d["orders"]):
    
    #     F_blaze = list(d[order]["F_blaze"])
    #     F_blazes[i*321:(i+1)*320+i] = F_blaze
    #     nu_blazes[i*321:(i+1)*320+i] = d[order]["pixf"]
        
    # d["F_blazes"] = F_blazes
    # d["nu_blazes"] = nu_blazes
        
    
        # d = blaze_conv(d)
    
    # return d



# d = so_cal(25285., -5.0)

fig1, (ax1a, ax1b, ax1c) = plt.subplots(figsize=(12, 9), nrows=3, sharex=True)
ax1a.plot(d["aotf"]["nu_hr"], d["aotf"]["F_aotf"])

for order in d["orders"]:
    ax1a.plot(d[order]["pixf"], d[order]["F_blaze"], label=order)
ax1a.legend()

from tools.spectra.molecular_spectrum_so import get_molecular_hr

mol_hr = 1.0 - get_molecular_hr("CO", d["aotf"]["nu_hr"], Smin=1.0e-25, scalar=10.)

ax1b.plot(d["aotf"]["nu_hr"], mol_hr)

# ax1b.set_yscale("log")

mol_aotf = mol_hr * d["aotf"]["F_aotf"]
ax1c.plot(d["aotf"]["nu_hr"], mol_aotf)




