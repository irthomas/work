# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:33:45 2021

@author: iant


CALIBRATION PARAMETERS

"""
import numpy as np


d = {}
temp = 0.0



def sinc_gd(dx,width,lobe,asym,offset):
    #goddard version
	sinc = (width*np.sin(np.pi*dx/width)/(np.pi*dx))**2.0
	ind = (abs(dx)>width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*lobe
	ind = (dx<=-width).nonzero()[0]
	if len(ind)>0: sinc[ind] = sinc[ind]*asym
	sinc += offset
	return sinc


def F_aotf3(dx, d):
    
    offset = d["aotfg"] * np.exp(-dx**2.0 / (2.0 * d["aotfgw"]**2.0))

    sinc = sinc_gd(dx,d["aotfw"],d["aotfs"],d["aotfa"], offset)
    
    return sinc


def F_blaze3(x, blazef, blazew):
    
    dx = x - blazef
    F = np.sinc((dx) / blazew)**2
    return F






def get_ils_params(d):
    """get ils params, add to dictionary"""
    
    aotff = d["aotff"]
    pixels = d["pixels"]

    #from ils.py on 6/7/21
    amp = 0.2724371566666666 #intensity of 2nd gaussian
    rp = 16939.80090831571 #resolving power cm-1/dcm-1
    disp_3700 = [-3.06665339e-06,  1.71638815e-03,  1.31671485e-03] #displacement of 2nd gaussian cm-1 w.r.t. 3700cm-1 vs pixel number
    
    A_w_nu0 = aotff / rp
    sconv = A_w_nu0/2.355
    
    
    disp_3700_nu = np.polyval(disp_3700, pixels) #displacement at 3700cm-1
    disp_order = disp_3700_nu / -3700.0 * aotff #displacement adjusted for wavenumber
    
    d["ils"] = {"width":np.tile(sconv, len(pixels)), "displacement":disp_order, "amplitude":np.tile(amp, len(pixels))}
    
    return d
        



def blaze_conv(d):
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
            
            nu_sp = nu_hr[inu1:inu2] - nu_p[ip]
            
            #make ils shape
            a1 = 0.0
            a2 = d["ils"]["width"][ip]
            a3 = 1.0
            a4 = d["ils"]["displacement"][ip]
            a5 = d["ils"]["width"][ip]
            a6 = d["ils"]["amplitude"][ip]
                
            ils0=a3 * np.exp(-0.5 * ((nu_sp + a1) / a2) ** 2)
            ils1=a6 * np.exp(-0.5 * ((nu_sp + a4) / a5) ** 2)
            ils = ils0 + ils1 
    
        
            W_conv[ip,inu1:inu2] += (W_blaze[ip]) * ils
            # W_conv[ip,inu1:inu2] += (W_blaze[ip] * dnu)/(np.sqrt(2.0 * np.pi) * sconv) * np.exp(-(nu_hr[inu1:inu2] - nu_p[ip])**2 / (2. *sconv**2))
            # if ip == 319:
            #     plt.plot(nu_sp, ils + iord/1000.)
            #     plt.plot(nu_sp, (W_blaze[ip] * dnu)/(np.sqrt(2.0 * np.pi) * sconv) * np.exp(-(nu_hr[inu1:inu2] - nu_p[ip])**2 / (2. *sconv**2)) + iord/1000.)
    
    d["W_conv"] = W_conv
    
    return d











def get_cal_params(d, temp):
# if True:
    #from blazecalc.py 17/9/21
    
    aotf = d["A"]
    nu_hr = d["nu_hr"]
    orders = d["orders"]

    #1st september slack
    # AOTF shape parameters
    # aotfwc  = [-1.78088527e-07,  9.44266907e-04,  1.95991162e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    # aotfsc  = [ 1.29304371e-06, -6.77032965e-03,  1.03141366e+01] # sidelobes factor [scaler from AOTF frequency cm-1]
    # aotfac  = [-1.96949242e-07,  1.48847262e-03, -1.40522510e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    # aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
    # aotfgc  = [ 1.07865793e-07, -7.20862528e-04,  1.24871556e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]
    
    # # Calibration coefficients
    # cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])         # Frequency of AOTF [cm-1 from kHz]
    # cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])   # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    # ncoeff  = [-1.76520810e-07, -2.26677449e-05, -1.93885521e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
    # blazep  = [-5.76161e-14,-2.01122e-10,2.02312e-06,2.25875e+01] # Dependence of blazew from AOTF frequency
    # aotfts  = -6.5278e-5                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]


    #2nd october slack
    aotfwc  = [-1.66406991e-07,  7.47648684e-04,  2.01730360e+01] # Sinc width [cm-1 from AOTF frequency cm-1]
    aotfsc  = [ 8.10749274e-07, -3.30238496e-03,  4.08845247e+00] # sidelobes factor [scaler from AOTF frequency cm-1]
    aotfac  = [-1.54536176e-07,  1.29003715e-03, -1.24925395e+00] # Asymmetry factor [scaler from AOTF frequency cm-1]
    aotfoc  = [            0.0,             0.0,             0.0] # Offset [coefficients for AOTF frequency cm-1]
    aotfgc  = [ 1.49266526e-07, -9.63798656e-04,  1.60097815e+00] # Gaussian peak intensity [coefficients for AOTF frequency cm-1]
    # Calibration coefficients
    cfaotf  = np.array([1.34082e-7, 0.1497089, 305.0604])                  # Frequency of AOTF [cm-1 from kHz]
    cfpixel = np.array([1.75128E-08, 5.55953E-04, 2.24734E+01])            # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    ncoeff  = [-2.44383699e-07, -2.30708836e-05, -1.90001923e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
    aotfts  = -6.5278e-5                                          # AOTF frequency shift due to temperature [relative cm-1 from Celsius]
    blazep  = [-1.00162255e-11, -7.20616355e-09, 9.79270239e-06, 2.25863468e+01] # Dependence of blazew from AOTF frequency


    
    # Calculate blaze parameters
    aotff = np.polyval(cfaotf, aotf) + temp*aotfts  # AOTF frequency [cm-1], temperature corrected
    d["aotff"] = aotff
   
    d["aotfw"] = np.polyval(aotfwc,aotff)
    d["aotfs"] = np.polyval(aotfsc,aotff)
    d["aotfa"] = np.polyval(aotfac,aotff)
    d["aotfo"] = np.polyval(aotfoc,aotff)
    d["aotfg"] = np.polyval(aotfgc,aotff)
    d["aotfgw"] = 50. #offset width cm-1

    dx = nu_hr - aotff
    d["F_aotf"] = F_aotf3(dx, d)

    
    d["blaze_shift"] = 0.0
        
        
    #TODO: update this with different blaze values for each order
    # blazew =  np.polyval(blazep,aotf-22000.0)        # FSR (Free Spectral Range), blaze width [cm-1]
    blazew =  np.polyval(blazep,d["line"]-3700.0)        # FSR (Free Spectral Range), blaze width [cm-1]
    blazew += blazew*np.polyval(ncoeff,temp)        # FSR, corrected for temperature
    d["blazew"] = blazew

    order = round(aotff/blazew)                      # Grating order


    # Frequency of the pixels
    for order in orders:
        d[order] = {}
        pixf = np.polyval(cfpixel,range(320))*order
        pixf += pixf*np.polyval(ncoeff, temp) + d["blaze_shift"]
        blazef = order*blazew                        # Center of the blaze
        d[order]["pixf"] = pixf
        d[order]["blazef"] = blazef
        # print(order, blazef, blazew)
        
        blaze = F_blaze3(pixf, blazef, blazew)
        d[order]["F_blaze"] = blaze
    

    F_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan
    nu_blazes = np.zeros(320 * len(d["orders"]) + len(d["orders"]) -1) * np.nan

    for i, order in enumerate(d["orders"]):
    
        F_blaze = list(d[order]["F_blaze"])
        F_blazes[i*321:(i+1)*320+i] = F_blaze
        nu_blazes[i*321:(i+1)*320+i] = d[order]["pixf"]
        
    d["F_blazes"] = F_blazes
    d["nu_blazes"] = nu_blazes
    
    d = get_ils_params(d)
    d = blaze_conv(d)
    

    return d


