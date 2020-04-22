# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:15:03 2018

@author: iant



SCRIPT TO BUILD SO AND LNO SPECTRAL CALIBRATION TABLE TO BE PLACED IN PFM_AUXILIARY_FILES/SPECTRAL_CALIBRATION DIRECTORY
SEE EXM-NO-TNO-AER-00083-iss0rev0-Spectral_calibration_coefficients_analysis_170322



"""

import os
import h5py
import numpy as np
from datetime import datetime
#import matplotlib.pyplot as plt


#figx=18
#figy=9

VERSION = "07"

if os.path.exists(os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")):
    BASE_DIRECTORY = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")
elif os.path.exists(os.path.normcase(r"C:\Users\ithom\Dropbox\NOMAD\Python")):
    BASE_DIRECTORY = os.path.normcase(r"C:\Users\ithom\Dropbox\NOMAD\Python")
elif os.path.exists(os.path.normcase(r"/home/iant/linux")):
    BASE_DIRECTORY = os.path.normcase(r"/home/iant/linux")


#Name:[[coefficients], [inputVariables], outputUnits]
soCoefficientDict = {
#"AOTFSincWidthCoefficients":[[-2.1837e-7, 5.82007e-4, 21.8543], ["AOTFCentre"], "cm-1"],
#"AOTFSidelobeRatioCoefficients":[[4.25071e-7, -2.24849e-3, 4.24031], ["AOTFCentre"], "cm-1"],
#"AOTFBaselineOffsetCoefficients":[[-3.51707e-8, 2.80952e-4, -0.499704], ["AOTFCentre"], "cm-1"],
"AOTFWnCoefficients":[[1.34082e-7, 0.1497089, 305.0604], ["AOTFFrequency"], "cm-1"],
"AOTFCentreTemperatureShiftCoefficients":[[0.0, -6.5278e-5, 0.0], ["AOTFCentre","Temperature"], "cm-1"],

"AOTFOrderCoefficients":[[5.6669740E-09, 6.6230100E-03, 1.3913770E+01], ["AOTFFrequency"], "order"],
"ResolvingPowerCoefficients":[[2.15817e-3, -17.3554, 4.59995e4], ["AOTFCentre"], "none"],

"BlazeFunction":[[0.0, 0.0, 22.473422, 0.0001238771664, 22.55759504, 0.003147426214], ["Order"], "cm-1"], #i.e. FSR and centre of grating. Replace with new coefficients in wavenumbers

"PixelSpectralCoefficients":[[1.751279e-8, 5.559526e-4, 22.473422], ["X", "Order"], "cm-1"],
"FirstPixelCoefficients":[[0.0, -7.299039e-1, -6.267734], ["Temperature"], "px"],

"AOTFSincWidthCoefficients":[[-2.18387e-7, 5.82007e-4, 2.18543e1], ["AOTFCentre"], "cm-1"],
"AOTFSidelobeRatioCoefficients":[[4.25071e-7, -2.24849e-3, 4.24031], ["AOTFCentre"], "cm-1"],
"AOTFOffsetCoefficients":[[], ["AOTFCentre"], "cm-1"], #not implemented
}

#coefficientDict = {"PixelSpectralCoefficients":[1.751279e-8, 5.559526e-4, 22.473422],
#"FirstPixelCoefficients":[0.0, -7.299039e-1, -6.267734]}

# TODO: check temperature shift, resolving power
lnoCoefficientDict = {
"AOTFWnCoefficients":[[9.409476e-8, 0.1422382, 300.67657], ["AOTFFrequency"], "cm-1"],
"AOTFCentreTemperatureShiftCoefficients":[[0.0, -6.5278e-5, 0.0], ["AOTFCentre","Temperature"], "cm-1"],

"AOTFOrderCoefficients":[[3.9186850E-09, 6.3020400E-03, 1.3321030E+01], ["AOTFFrequency"], "order"],
"ResolvingPowerCoefficients":[[-1.898669696e-05, 0.2015505624, 16509.58391], ["AOTFCentre"], "none"], #calculated below from figure in Liuzzi et al

"BlazeFunction":[[0.0, 0.0, 22.478113, 0.0001245622383, 22.56190161, 0.00678411387], ["Order"], "cm-1"], #i.e. FSR and centre of grating. Replace with new coefficients in wavenumbers

"PixelSpectralCoefficients":[[3.774791e-8, 5.508335e-4, 22.478113], ["X", "Order"], "cm-1"],
"FirstPixelCoefficients":[[0.0, -6.556383e-1, -8.024164], ["Temperature"], "px"],

"AOTFCoefficientsLiuzzi":[[0.6290016297432226, 18.188122, 0.37099837025677734, 12.181137], ["N/A"], "cm-1"] #i0, w, ig, sigmag
}


#calculate blaze function centre in wavenumbers
#blaze_wavenumbers = []
blaze_coefficients_px = [0.0, 0.22, 150.80]
for channel, coefficientDict in zip(["so", "lno"], [soCoefficientDict, lnoCoefficientDict]):
    pixel_spectral_coefficients = coefficientDict["PixelSpectralCoefficients"][0]
    orders = np.arange(100,220)
    blaze_centre_px = np.polyval(blaze_coefficients_px, orders)
    blaze_centre_waven = orders * np.polyval(pixel_spectral_coefficients, blaze_centre_px)    
    blaze_coefficients_waven = np.polyfit(orders, blaze_centre_waven, 2)
    print(channel + ": " + ", ".join("%.10g" % f for f in blaze_coefficients_waven))
    
    
#calculate LNO resolving power coefficients as a function of wavenumber
#linear interpolation of spectral resolution figure from Goddard
specResX=np.linspace(2500.0, 4600.0, num=100)
specResY=np.linspace(0.148, 0.27, num=100)
resolvingPower = specResX / specResY
resolvingPowerCoefficients = np.polyfit(specResX, resolvingPower, 2)
print(", ".join("%.10g" % f for f in resolvingPowerCoefficients))
    
    

#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(orders, blaze_centre_waven)
#plt.figure()
#plt.plot(orders, np.polyval(blaze_coefficients_waven, orders)-blaze_centre_waven)


for channel in ["so", "lno"]:

    title = "%s_Spectral_Calibration_Table" %channel.upper()
    coefficientDict = {"so":soCoefficientDict, "lno":lnoCoefficientDict}[channel]
    
    #make arrays of coefficients for given calibration date
    #at present, values don't change over time. Therefore copy values for dates 2 and 3
    calibrationTimes = ["2015 JAN 01 00:00:00.000", "2017 JAN 01 00:00:00.000"]
    
    
    
    outputFilename = "%s" %(title.replace(" ","_"))
       
    hdf5File = h5py.File(os.path.join(BASE_DIRECTORY, outputFilename + "_v%s.h5" %VERSION), "w")
    
    for calibrationTime in calibrationTimes:
        hdf5Group = hdf5File.create_group(calibrationTime)
            
        for key, value in coefficientDict.items():
            coefficients, inputVariables, outputUnits = value
            hdf5Dataset = hdf5Group.create_dataset(key, data=coefficients, dtype=np.float64)
            hdf5Dataset.attrs["units"] = outputUnits
            for variableIndex, inputVariable in enumerate(inputVariables):
                hdf5Dataset.attrs["inputVariable%i" %variableIndex] = inputVariable
            

    
#    if channel=="lno":
#        comments = "Analysis by A. Mahieux (NOMAD_LNO_calib_report.docx, 2017-02-03) and G. Villaneuva (nomad_calib_gsfc.pdf, 2017-08-28)"
#    elif channel=="so":
#        comments = "Analysis by G. Villaneuva (nomad_calib_gsfc.pdf, 2017-08-28)"
#    hdf5File.attrs["Comments"] = comments
    hdf5File.attrs["DateCreated"] = str(datetime.now())
    hdf5File.close()
    
    


