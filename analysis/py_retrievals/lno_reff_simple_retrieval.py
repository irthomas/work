# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:55:08 2019

@author: iant
"""


import os
#import sys

import numpy as np
# from scipy import interpolate
#from matplotlib import pyplot as plt

from tools.file.paths import paths
# from analysis.retrievals.pytran import pytran
#from analysis.retrievals.NOMADTOOLS import nomadtools
#from analysis.retrievals.NOMADTOOLS.nomadtools import gem_tools
#from analysis.retrievals.NOMADTOOLS.nomadtools.paths import NOMADParams
#from analysis.retrievals.NOMAD_instrument import freq_mp, F_blaze, F_aotf_3sinc
from tools.spectra.solar_spectrum_lno import get_solar_hr

from instrument.nomad_lno_instrument import nu_mp, F_blaze

from instrument.nomad_lno_instrument import F_aotf_goddard19draft as F_aotf
# from instrument.nomad_so_instrument import F_aotf_goddard18b as F_aotf



def simple_retrieval(y_in, order, instrument_temperature):
    """AOTF and blaze function simulation with solar spectrum.
       Add dust spectrum later"""

    
    resolving_power = 10000.
    
    retDict = {}
    retDict["instrument_temperature"] = instrument_temperature
    

    adj_orders = 2
    retDict["order"] = order
    retDict["adj_orders"] = adj_orders

    nu_hr_min = nu_mp(order-adj_orders, 0., retDict["instrument_temperature"]) - 5.
    nu_hr_max = nu_mp(order+adj_orders, 320., retDict["instrument_temperature"]) + 5.
    nu_centre = nu_mp(order, 160.0, retDict["instrument_temperature"])
    dnu = 0.001
    Nbnu_hr = int(np.ceil((nu_hr_max-nu_hr_min)/dnu)) + 1
    retDict["nu_hr"] = np.linspace(nu_hr_min, nu_hr_max, Nbnu_hr)
    dnu = retDict["nu_hr"][1]-retDict["nu_hr"][0]

    # read in solar 
    solspecFilepath = os.path.join(paths["REFERENCE_DIRECTORY"], "nomad_solar_spectrum_solspec.txt")
    I0_solar_hr = get_solar_hr(retDict["nu_hr"], solspecFilepath)

    retDict["Nbnu_hr"] = Nbnu_hr
    retDict["dnu"] = dnu
    retDict["I0_hr"] = I0_solar_hr


    retDict["pixels"] = np.arange(len(y_in))
    NbP = len(retDict["pixels"])
    retDict["NbP"] = NbP
    retDict["nu_p"] = nu_mp(order, retDict["pixels"], retDict["instrument_temperature"])


    print("Computing convolution matrix")
    W_conv_old = np.zeros((NbP,Nbnu_hr))
    W_conv = np.zeros((NbP,Nbnu_hr))
    
    retDict["spectral_resolution"] = nu_centre / resolving_power
    retDict["sconv"] = retDict["spectral_resolution"]/2.355 #0.40/2.355
    
    
    for iord in range(order-adj_orders, order+adj_orders+1):
        retDict["%i" %iord] = {}
        retDict["%i" %iord]["nu_pm"] = nu_mp(iord, retDict["pixels"], retDict["instrument_temperature"])
        W_blaze = F_blaze(iord, retDict["pixels"], retDict["instrument_temperature"])
        retDict["%i" %iord]["W_blaze"] = W_blaze
        for pixelIndex, _ in enumerate(retDict["pixels"]):
            gaussian = (retDict["%i" %iord]["W_blaze"][pixelIndex]*retDict["dnu"])/(np.sqrt(2.*np.pi)*retDict["sconv"])*np.exp(-(retDict["nu_hr"]-retDict["%i" %iord]["nu_pm"][pixelIndex])**2/(2.*retDict["sconv"]**2))
            retDict["%i" %iord]["gaussian"] = gaussian
            W_conv_old[pixelIndex,:] += gaussian
            W_conv[pixelIndex,:] += gaussian
    retDict["W_conv"] = W_conv
    retDict["W_conv_old"] = W_conv_old

    retDict["Trans_hr"] = np.linspace(1.0, 0.9, num=len(retDict["nu_hr"])) #dust
    
    return retDict



def forward_model(retDict):
    

    retDict["W_aotf"] = F_aotf(retDict["order"], retDict["nu_hr"], retDict["instrument_temperature"])
    I0_hr = retDict["W_aotf"] * retDict["I0_hr"]            # nhr
    retDict["I0_p"] = np.matmul(retDict["W_conv"], I0_hr)    # np x 1
    I_hr = I0_hr * retDict["Trans_hr"]  # nz x nhr
    retDict["I_p"] = np.matmul(retDict["W_conv"], I_hr.T).T  # nz x np
    retDict["Trans_p"] = retDict["I_p"] / retDict["I0_p"]       # nz x np

    # retDict = fit_background(retDict)

    # retDict["Y"] = retDict["Trans_p"] * retDict["Trans_background"]

    # y = retDict["Y"].ravel()
    # retDict["y"] = y
    return retDict



# def Jacobian_y(retDict, xa_fact=[None]):

#     if xa_fact[0] == None:
#         xa_fact = retDict["xa_fact"]

#     Ky = np.zeros((retDict["NbZ"]*retDict["NbP"], len(xa_fact))) #jacobian matrix of forward model (nAltitudes x nPixels) by nAltitudes

#     W_aotf = F_aotf(retDict["order"], retDict["nu_hr"])
#     I0_hr = W_aotf * retDict["I0_hr"]             # high res solar spectrum
#     I0_p = np.matmul(retDict["W_conv"], I0_hr)    # convolved pixel solar spectrum

#     for ispec in range(retDict["NbZ"]):
#         for iz in range(ispec, retDict["NbZ"]): #loop through 0 to nAltitudes, 1 to nAltitudes, 2 to nAltitudes etc.

#             iy1 = ispec * retDict["NbP"]
#             iy2 = (ispec+1) * retDict["NbP"]

#             J_hr = (-retDict["xa"][iz]*retDict["atmo"]['NT'][iz]*retDict["dl"][ispec,iz])*retDict["sigma_hr"][iz,:]*I0_hr*retDict["Trans_hr"][ispec,:]
#             J_p = np.matmul(retDict["W_conv"], J_hr.T).T #convolve jacobian to pixel resolution
#             Ky[iy1:iy2,iz] = (J_p / I0_p) * retDict["Trans_background"][ispec,:] #low res jacobian / solar spectrum * transmission
# #            retDict["Ky"] = Ky
        
#     retDict["Ky"] = Ky.copy()
#     return retDict



# def fit_background(retDict):

#     x = retDict["nu_p"] - np.mean(retDict["nu_p"]) #relative wavenumbers

#     """matrix version (faster and uses YError)"""
#     X = np.ones((retDict["NbP"],retDict["background_degree"]+1))

#     for iz in range(retDict["NbZ"]):

#         for ib in range(0,retDict["background_degree"]+1):
#             X[:,ib] = retDict["Trans_p"][iz,:]*x**ib
#         y = retDict["YObs"][iz,:]
#         W = np.diag(1.0 / retDict["YError"][iz,:]**2)
#         yp = np.matmul(X.T, np.matmul(W, y))
#         A_LHS = np.matmul(X.T, np.matmul(W, X))
#         p = np.linalg.solve(A_LHS, yp)

#         retDict["background_coeffs"][iz,:] = p
#         for ib in range(0,retDict["background_degree"]+1):
#             X[:,ib] = x**ib
#         retDict["Trans_background"][iz,:] = np.matmul(X, p)

    """polyfit version"""
#    #find where no absorption line
#    #take mean of all altitudes modelled line
#    mean_spectrum = np.mean(retDict["Trans_p"], axis=0)
#    good_pixels = np.where(mean_spectrum > 0.9999)[0]
#    
#    if len(good_pixels) < 100:
#        print("Warning: only %i pixels used for fit" %len(good_pixels))
#
#    for iz in range(retDict["NbZ"]):
#        
#        spectrum = retDict["YObs"][iz, :]
#
#        p = np.polyfit(x[good_pixels], spectrum[good_pixels], retDict["background_degree"])
#        retDict["background_coeffs"][iz,:] = p
#
#        retDict["Trans_background"][iz,:] = np.polyval(p, x)
    
    """baseline fitting version"""
    
    
#    def baseline_als(y, lam=250.0, p=0.95, niter=10):
#        from scipy import sparse
#        from scipy.sparse.linalg import spsolve
#    
#        L = len(y)
#        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
#        w = np.ones(L)
#        for i in range(niter):
#            W = sparse.spdiags(w, 0, L, L)
#            Z = W + lam * D.dot(D.transpose())
#            z = spsolve(Z, w*y)
#            w = p * (y > z) + (1-p) * (y < z)
#    
#        return z
#
#
#    for iz in range(retDict["NbZ"]):
#        
#        spectrum = retDict["YObs"][iz, :]
#
#        p = baseline_als(spectrum)
#        retDict["Trans_background"][iz,:] = p
    

    return retDict



# def Rodgers_OEM(retDict, niter_max=5, chi_tol=1.0e-6, alpha=0.1):

#     retDict = fit_background(retDict)
#     retDict["Y"] = retDict["Trans_background"]
#     chisq_old = np.sqrt(np.sum(((retDict["YObs"]-retDict["Y"])/retDict["YError"])**2) / (retDict["NbZ"]*retDict["NbP"] - retDict["NbZ"])) 
#     print("step 0: chi^2 = ", chisq_old)
     
    
#     """Se = np.diag(retDict["YError.ravel()**2)"""
#     Se_inv = np.diag(1.0/retDict["YError"].ravel()**2)

#     Sa = np.diag(retDict["sa"]**2)
#     Sa_inv = np.linalg.inv(Sa)

#     for step in range(niter_max):
#         retDict = forward_model(retDict) #simulate using xa and xa_fact
#         retDict = Jacobian_y(retDict) #update jacobians
        
#         y = retDict["y"]
#         Ky = retDict["Ky"]
#         chisq = np.sqrt(np.sum(((retDict["YObs"]-retDict["Y"])/retDict["YError"])**2) / (retDict["NbZ"]*retDict["NbP"] - retDict["NbZ"])) 
#         print("step %d: chi^2 = "%step, chisq)

#         W1 = np.matmul(Se_inv, Ky)
#         W2 = Sa_inv + np.matmul(Ky.T, W1)
#         w1 = np.matmul(Ky, retDict["xa_fact"] - np.ones(len(retDict["xa_fact"])))
#         w2 = np.matmul(W1.T, retDict["YObs"].ravel() - y + w1)
#         dx = np.linalg.solve(W2, w2)

#         """w = 0.1"""
#         xa_fact = 1.0 + dx
#         xa_fact[dx<-1.] = 0.001 #
#         retDict["xa_fact"] = (1.0-alpha)*retDict["xa_fact"] + alpha*xa_fact

#         print("ppb xa * xa_fact=", retDict["xa"] * 1.0e9 * xa_fact)


#     retDict["S"] = np.linalg.inv(W2)
    
#     return retDict
    
    

