# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:55:08 2019

@author: iant
"""


import os
#import sys

import numpy as np
from scipy import interpolate
#from matplotlib import pyplot as plt

from tools.file.paths import paths
from analysis.retrievals.pytran import pytran
#from analysis.retrievals.NOMADTOOLS import nomadtools
#from analysis.retrievals.NOMADTOOLS.nomadtools import gem_tools
#from analysis.retrievals.NOMADTOOLS.nomadtools.paths import NOMADParams
#from analysis.retrievals.NOMAD_instrument import freq_mp, F_blaze, F_aotf_3sinc
from tools.spectra.solar_spectrum_so import get_solar_hr

from instrument.nomad_so_instrument import nu_mp, F_blaze, F_aotf_goddard18b

def simple_retrieval(y_in, alt, molecule, order, instrument_temperature, snr=500.0):
    
    resolving_power = 10000.
    
    retDict = {}
#    TangentAlt = obsDict["alt"]
#    retDict["XObs"] = obsDict["x"]
#    retDict["YObs"] = y_in
#    retDict["NbZ"] = NbZ
#    print("TangentAlt=", TangentAlt)

    TangentAlt = np.arange(alt, 100.0, 5.0)
    NbZ = len(TangentAlt)
    retDict["NbZ"] = NbZ
    retDict["instrument_temperature"] = instrument_temperature
    
    retDict["YObs"] = np.ones((NbZ, len(y_in)))
    retDict["YObs"][0, :] = y_in
    retDict["YError"] = np.zeros_like(retDict["YObs"]) + y_in / snr

#    atmolist = gem_tools.get_observation_atmolist_filename(obsDict)
#    with open(os.path.join(paths["RETRIEVALS"]["APRIORI_FILE_DESTINATION"], atmolist), 'r') as f:
#        all_atmofiles = f.readlines()
    atmofile = os.path.join(paths["RETRIEVALS"]["APRIORI_FILE_DESTINATION"], "generic.dat")

    atmo = {}
    atmo['Z'] = TangentAlt
    atmo['T'] = np.zeros(NbZ)
    atmo['P'] = np.zeros(NbZ)
    atmo['NT'] = np.zeros(NbZ)
    Zin, Tin, Pin, NTin = np.loadtxt(atmofile, comments='%', usecols=(0,1,2,3), unpack=True)
    for iz in range(NbZ):
        iz2 = np.argmax(atmo['Z'][iz]>Zin)

        f = (Zin[iz2-1]-atmo['Z'][iz])/(Zin[iz2-1]-Zin[iz2])
        atmo['T'][iz] = (1.-f)*Tin[iz2-1] + f*Tin[iz2]
        atmo['P'][iz] = np.exp((1.-f)*np.log(Pin[iz2-1]) + f*np.log(Pin[iz2]))
        atmo['NT'][iz] = np.exp((1.-f)*np.log(NTin[iz2-1]) + f*np.log(NTin[iz2]))
    retDict["atmo"] = atmo

    gem_version_string = 'gem-mars-a585'
#    apriori_version = 'apriori_1_1_1_GEMZ_wz_mixed'
    apriori_version = 'apriori_4_2_4_GEMZ_wz_mixed'
#    geom = nomadtools.get_obs_geometry(obsDict)
    apriori_zone = "Spring_Southern_Day" #apriori_zone='AllSeasons_AllHemispheres_AllTime'
    print("apriori_zone=", apriori_zone)

#    molecule = obsDict["molecule"]
    
    if molecule == "CH4":
        retDict["xa"] = np.ones(len(atmo['Z'])) * 1.0e-9 #1ppb constant, converted to dimensionless quantity
        retDict["sa"] = np.ones(len(atmo['Z'])) * 10.0 #10x variation allowed above/below.  #for CH4, don't scale CO2 allowed deviation
    elif molecule == "HCl":
        retDict["xa"] = np.ones(len(atmo['Z'])) * 1.0e-9 #1ppb constant, converted to dimensionless quantity
        retDict["sa"] = np.ones(len(atmo['Z'])) * 10.0 #10x variation allowed above/below.  #for CH4, don't scale CO2 allowed deviation
    elif molecule == "PH3":
        retDict["xa"] = np.ones(len(atmo['Z'])) * 1.0e-9 #1ppb constant, converted to dimensionless quantity
        retDict["sa"] = np.ones(len(atmo['Z'])) * 10.0 #10x variation allowed above/below.  #for CH4, don't scale CO2 allowed deviation
    else:
        name = molecule
        atmo_dir = os.path.join(gem_version_string, apriori_version)
        mean_file = gem_version_string + '_' + apriori_zone + '_mean_'+ name +'.dat'
        stdev_file = gem_version_string + '_' + apriori_zone + '_stdev_'+ name +'.dat'
        xa_file = os.path.join(atmo_dir, mean_file)
        sa_file = os.path.join(atmo_dir, stdev_file)

        za_in, xa_in = np.loadtxt(os.path.join(paths["RETRIEVALS"]['ATMOSPHERE_DIR'], xa_file), comments='%', usecols=(0,1,), unpack=True) #read in ppm
        sa_in = np.loadtxt(os.path.join(paths["RETRIEVALS"]['ATMOSPHERE_DIR'], sa_file), comments='%', usecols=(1,))
        
        xa_fun = interpolate.interp1d(za_in[::-1], xa_in[::-1])
        sa_fun = interpolate.interp1d(za_in[::-1], sa_in[::-1])
        
        retDict["xa"] = xa_fun(atmo['Z']) * 1.0e-6 #state vector: get a priori VMR in ppm from GEM. Convert to no units
        retDict["sa"] = sa_fun(atmo['Z']) 
        

    retDict["xa_fact"] = np.ones_like(retDict["xa"])


    adj_orders = 0
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


    M = pytran.get_molecule_id(molecule)
    filename = os.path.join(paths["RETRIEVALS"]['HITRAN_DIR'], '%02i_hit16_2000-5000_CO2broadened.par' % M)
    if not os.path.exists(filename):
        filename = os.path.join(paths["RETRIEVALS"]['HITRAN_DIR'], '%02i_hit16_2000-5000.par' % M)


    if molecule == "CO2" and order > 140:
        Smin=1.0e-26
    elif molecule == "H2O":
        Smin=1.0e-27
    else:
        Smin=1.0e-33
        
    nlines = 999
    while nlines>200:
        Smin *= 10.0
        LineList = pytran.read_hitran2012_parfile(filename, nu_hr_min, nu_hr_max, Smin=Smin, silent=True)
        nlines = len(LineList['S'])
        print('Found %i lines' % nlines)

    
    retDict["LineList"] = LineList
    
    retDict["sigma_hr"] = np.zeros((NbZ,Nbnu_hr))
    for i in range(NbZ):
        if np.mod(i, 10) == 0:
            print("%d of %d" % (i, NbZ))
        retDict["sigma_hr"][i,:] =  pytran.calculate_hitran_xsec(LineList, M, retDict["nu_hr"], T=atmo['T'][i], P=atmo['P'][i]*1e3)


    #limit range
    retDict["pixels"] = np.arange(len(y_in))
    NbP = len(retDict["pixels"])
    retDict["NbP"] = NbP
    retDict["nu_p"] = nu_mp(order, retDict["pixels"], retDict["instrument_temperature"])
    retDict["Trans_p"] = np.ones((NbZ,NbP))


    background_degree = 4
    retDict["background_degree"] = background_degree
    retDict["background_coeffs"] = np.zeros((NbZ,background_degree+1))
    retDict["Trans_background"] = np.ones((NbZ,NbP))

    print("Computing convolution matrix")
    W_conv_old = np.zeros((NbP,Nbnu_hr))
    W_conv = np.zeros((NbP,Nbnu_hr))
    
    retDict["spectral_resolution"] = nu_centre / resolving_power
    retDict["sconv"] = retDict["spectral_resolution"]/2.355 #0.40/2.355
    
#    if obsDict["gaussian_scalar"] is not None:
#        scalar = obsDict["gaussian_scalar"]
#        xoffset = obsDict["gaussian_xoffset"]
#        width = retDict["sconv"] * obsDict["gaussian_width"]
    
    for iord in range(order-adj_orders, order+adj_orders+1):
        retDict["%i" %iord] = {}
        retDict["%i" %iord]["nu_pm"] = nu_mp(iord, retDict["pixels"], retDict["instrument_temperature"])
        W_blaze = F_blaze(iord, retDict["pixels"], retDict["instrument_temperature"])
        retDict["%i" %iord]["W_blaze"] = W_blaze
        for pixelIndex, _ in enumerate(retDict["pixels"]):
            #(W_blaze[pixelIndex]*dnu) / (np.sqrt(2.*np.pi)*sconv) * np.exp(-(nu_hr-nu_pm[pixelIndex])**2/(2.*sconv**2))
            gaussian = (retDict["%i" %iord]["W_blaze"][pixelIndex]*retDict["dnu"])/(np.sqrt(2.*np.pi)*retDict["sconv"])*np.exp(-(retDict["nu_hr"]-retDict["%i" %iord]["nu_pm"][pixelIndex])**2/(2.*retDict["sconv"]**2))
#            if obsDict["gaussian_scalar"] is not None:
#                gaussian2 = (retDict["%i" %iord]["W_blaze"][pixelIndex]*retDict["dnu"]*scalar)/(np.sqrt(2.*np.pi)*width)*np.exp(-(retDict["nu_hr"]-retDict["%i" %iord]["nu_pm"][pixelIndex]+xoffset)**2/(2.*width**2))
            retDict["%i" %iord]["gaussian"] = gaussian
#            if obsDict["gaussian_scalar"] is not None:
#                retDict["%i" %iord]["gaussian2"] = gaussian2
            W_conv_old[pixelIndex,:] += gaussian
            W_conv[pixelIndex,:] += gaussian
#            if obsDict["gaussian_scalar"] is not None:
#                W_conv[pixelIndex,:] += gaussian2
    retDict["W_conv"] = W_conv
    retDict["W_conv_old"] = W_conv_old
    
    return retDict



def forward_model(retDict, xa_fact=[None]):
    
    """xa_fact, NbZ, "atmo", Nbnu_hr, xa, sigma_hr, order, I0_hr, W_conv"""

    if xa_fact[0]==None:
        xa_fact = retDict["xa_fact"]
    elif len(xa_fact) == 1:
        xa_fact = xa_fact * retDict["NbZ"]
        retDict["xa_fact"] = xa_fact
    else:
        retDict["xa_fact"] = xa_fact


        
    #get relative path lengths etc.
    Rp = 3396.
    s = np.zeros(retDict["NbZ"])
    dl = np.zeros((retDict["NbZ"],retDict["NbZ"]))
    for i in range(retDict["NbZ"]):
        s[i:] = np.sqrt((Rp+retDict["atmo"]['Z'][i:])**2-(Rp+retDict["atmo"]['Z'][i])**2)
        if i < retDict["NbZ"]-1:
            dl[i,i] = s[i+1] - s[i]
        if i < retDict["NbZ"]-2:
            dl[i,(i+1):-1] = s[(i+2):] - s[i:-2]
        dl[i,-1] = s[-1] - s[-2] + 2*10. /np.sqrt(1.-((Rp+retDict["atmo"]['Z'][i])/(Rp+retDict["atmo"]['Z'][-1]+1.))**2) 
    dl *= 1e5
    retDict["dl"] = dl

    retDict["tau_hr"] = np.zeros((retDict["NbZ"],retDict["Nbnu_hr"]))
    retDict["Trans_hr"] = np.ones((retDict["NbZ"],retDict["Nbnu_hr"]))
   
    for i in range(retDict["NbZ"]): #loop altitudes
        for j in range(i,retDict["NbZ"]): #loop altitudes above current altitude
            retDict["tau_hr"][i,:] += (xa_fact[j]*retDict["xa"][j]*retDict["atmo"]['NT'][j]*dl[i,j])*retDict["sigma_hr"][j,:]
        retDict["Trans_hr"][i,:] = np.exp(-retDict["tau_hr"][i,:])

    W_aotf = F_aotf_goddard18b(retDict["order"], retDict["nu_hr"], retDict["instrument_temperature"])
    I0_hr = W_aotf * retDict["I0_hr"]            # nhr
    I0_p = np.matmul(retDict["W_conv"], I0_hr)    # np x 1
    I_hr = I0_hr[None,:] * retDict["Trans_hr"]  # nz x nhr
    I_p = np.matmul(retDict["W_conv"], I_hr.T).T  # nz x np
    retDict["Trans_p"] = I_p / I0_p[None,:]       # nz x np

    retDict = fit_background(retDict)

    retDict["Y"] = retDict["Trans_p"] * retDict["Trans_background"]

    y = retDict["Y"].ravel()
    retDict["y"] = y
    return retDict



def Jacobian_y(retDict, xa_fact=[None]):

    if xa_fact[0] == None:
        xa_fact = retDict["xa_fact"]

    Ky = np.zeros((retDict["NbZ"]*retDict["NbP"], len(xa_fact))) #jacobian matrix of forward model (nAltitudes x nPixels) by nAltitudes

    W_aotf = F_aotf_goddard18b(retDict["order"], retDict["nu_hr"])
    I0_hr = W_aotf * retDict["I0_hr"]             # high res solar spectrum
    I0_p = np.matmul(retDict["W_conv"], I0_hr)    # convolved pixel solar spectrum

    for ispec in range(retDict["NbZ"]):
        for iz in range(ispec, retDict["NbZ"]): #loop through 0 to nAltitudes, 1 to nAltitudes, 2 to nAltitudes etc.

            iy1 = ispec * retDict["NbP"]
            iy2 = (ispec+1) * retDict["NbP"]

            J_hr = (-retDict["xa"][iz]*retDict["atmo"]['NT'][iz]*retDict["dl"][ispec,iz])*retDict["sigma_hr"][iz,:]*I0_hr*retDict["Trans_hr"][ispec,:]
            J_p = np.matmul(retDict["W_conv"], J_hr.T).T #convolve jacobian to pixel resolution
            Ky[iy1:iy2,iz] = (J_p / I0_p) * retDict["Trans_background"][ispec,:] #low res jacobian / solar spectrum * transmission
#            retDict["Ky"] = Ky
        
    retDict["Ky"] = Ky.copy()
    return retDict



def fit_background(retDict):

    x = retDict["nu_p"] - np.mean(retDict["nu_p"]) #relative wavenumbers

    """matrix version (faster and uses YError)"""
    X = np.ones((retDict["NbP"],retDict["background_degree"]+1))

    for iz in range(retDict["NbZ"]):

        for ib in range(0,retDict["background_degree"]+1):
            X[:,ib] = retDict["Trans_p"][iz,:]*x**ib
        y = retDict["YObs"][iz,:]
        W = np.diag(1.0 / retDict["YError"][iz,:]**2)
        yp = np.matmul(X.T, np.matmul(W, y))
        A_LHS = np.matmul(X.T, np.matmul(W, X))
        p = np.linalg.solve(A_LHS, yp)

        retDict["background_coeffs"][iz,:] = p
        for ib in range(0,retDict["background_degree"]+1):
            X[:,ib] = x**ib
        retDict["Trans_background"][iz,:] = np.matmul(X, p)

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



def Rodgers_OEM(retDict, niter_max=5, chi_tol=1.0e-6, alpha=0.1):

    retDict = fit_background(retDict)
    retDict["Y"] = retDict["Trans_background"]
    chisq_old = np.sqrt(np.sum(((retDict["YObs"]-retDict["Y"])/retDict["YError"])**2) / (retDict["NbZ"]*retDict["NbP"] - retDict["NbZ"])) 
    print("step 0: chi^2 = ", chisq_old)
     
    
    """Se = np.diag(retDict["YError.ravel()**2)"""
    Se_inv = np.diag(1.0/retDict["YError"].ravel()**2)

    Sa = np.diag(retDict["sa"]**2)
    Sa_inv = np.linalg.inv(Sa)

    for step in range(niter_max):
        retDict = forward_model(retDict) #simulate using xa and xa_fact
        retDict = Jacobian_y(retDict) #update jacobians
        
        y = retDict["y"]
        Ky = retDict["Ky"]
        chisq = np.sqrt(np.sum(((retDict["YObs"]-retDict["Y"])/retDict["YError"])**2) / (retDict["NbZ"]*retDict["NbP"] - retDict["NbZ"])) 
        print("step %d: chi^2 = "%step, chisq)

        W1 = np.matmul(Se_inv, Ky)
        W2 = Sa_inv + np.matmul(Ky.T, W1)
        w1 = np.matmul(Ky, retDict["xa_fact"] - np.ones(len(retDict["xa_fact"])))
        w2 = np.matmul(W1.T, retDict["YObs"].ravel() - y + w1)
        dx = np.linalg.solve(W2, w2)

        """w = 0.1"""
        xa_fact = 1.0 + dx
        xa_fact[dx<-1.] = 0.001 #
        retDict["xa_fact"] = (1.0-alpha)*retDict["xa_fact"] + alpha*xa_fact

        print("ppb xa * xa_fact=", retDict["xa"] * 1.0e9 * xa_fact)


    retDict["S"] = np.linalg.inv(W2)
    
    return retDict
    
    

