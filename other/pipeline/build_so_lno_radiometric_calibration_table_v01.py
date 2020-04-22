# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:15:03 2018

@author: iant



SCRIPT TO BUILD SO AND LNO RADIOMETRIC CALIBRATION TABLE TO BE PLACED IN PFM_AUXILIARY_FILES/RADIOMETRIC_CALIBRATION DIRECTORY

CURRENTLY JUST BLANK VALUES TO SIMULATE PIPELINE 

DATA IS STRUCTURED BY TIME (IN GROUP) AND THEN IN A 2D MATRIX OF ORDER X PIXEL NUMBER.

2 DATASETS ARE PRESENT: Y AND Y ERROR, EACH CONVERTED TO 1MS INTEGRATION TIME PER PIXEL

AT PRESENT, ASSUME THAT LNO NADIR IS VERTICALLY BINNED 144 ROWS INTO ONE SPECTRUM PER MEASUREMENT

NO DATA YET FOR SO


"""

import os
import h5py
import numpy as np
#import numpy.linalg as la
#import gc
from datetime import datetime
#import matplotlib.pyplot as plt


figx=18
figy=9

if os.path.exists(os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")):
    BASE_DIRECTORY = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")
elif os.path.exists(os.path.normcase(r"C:\Users\ithom\Dropbox\NOMAD\Python")):
    BASE_DIRECTORY = os.path.normcase(r"C:\Users\ithom\Dropbox\NOMAD\Python")


channel="so"
#channel="lno"    
    

title = "%s_Radiometric_Calibration_Table" %channel.upper()


if channel=="lno":
    
    diffractionOrders = np.arange(0,230,1)
    
    pixels = np.arange(320)

    yCountsToRadiances = np.ones((len(diffractionOrders),len(pixels))) #array of error values lookup table
    yRadianceFactorCounts = np.ones((len(diffractionOrders),len(pixels))) #array of error values lookup table
    
    yErrorSingleValue = 5.0 
    yErrorRadiances = np.ones((len(diffractionOrders),len(pixels))) #array of error values lookup table
    yErrorRadianceFactors = np.ones((len(diffractionOrders),len(pixels))) #array of error values lookup table

elif channel=="so":
    
    diffractionOrders = np.arange(0,230,1)
    
    pixels = np.arange(320)

    yErrorSingleValue = 5.0 
    yErrorRadiances = np.ones((len(diffractionOrders),len(pixels))) #array of error values lookup table
    yErrorRadianceFactors = np.ones((len(diffractionOrders),len(pixels))) #array of error values lookup table
   
    
              


outputFilename = "%s" %(title.replace(" ","_"))


calibrationTimes = []

#make arrays of coefficients for given calibration date
calibrationTimes.append(b"2015 JAN 01 00:00:00.000")
#at present, values don't change over time. Therefore copy values for dates 2 and 3
calibrationTimes.append(b"2016 JAN 01 00:00:00.000")
calibrationTimes.append(b"2017 JAN 01 00:00:00.000")


#now write to file
#open file for writing
hdf5File = h5py.File(os.path.join(BASE_DIRECTORY,outputFilename+".h5"), "w")

#loop manually through calibration times. Not expecting many calibrations over time!
calibrationTime = calibrationTimes[0]
hdf5Group1 = hdf5File.create_group(calibrationTime)
hdf5Group1.create_dataset("DiffractionOrder",data=diffractionOrders,dtype=np.float)
hdf5Group1.create_dataset("Pixels",data=pixels,dtype=np.float)
if channel=="lno": hdf5Group1.create_dataset("YCountsToRadiances",data=yCountsToRadiances,dtype=np.float)
if channel=="lno": hdf5Group1.create_dataset("YRadianceFactorCounts",data=yRadianceFactorCounts,dtype=np.float)
hdf5Group1.create_dataset("YErrorSingleValue",data=yErrorSingleValue,dtype=np.float)
hdf5Group1.create_dataset("YErrorRadiances",data=yErrorRadiances,dtype=np.float)
hdf5Group1.create_dataset("YErrorRadianceFactors",data=yErrorRadianceFactors,dtype=np.float)



calibrationTime = calibrationTimes[1]
hdf5Group2 = hdf5File.create_group(calibrationTime)
hdf5Group2.create_dataset("DiffractionOrder",data=diffractionOrders,dtype=np.float)
hdf5Group2.create_dataset("Pixels",data=pixels,dtype=np.float)
if channel=="lno": hdf5Group2.create_dataset("YCountsToRadiances",data=yCountsToRadiances,dtype=np.float)
if channel=="lno": hdf5Group2.create_dataset("YRadianceFactorCounts",data=yRadianceFactorCounts,dtype=np.float)
hdf5Group2.create_dataset("YErrorSingleValue",data=yErrorSingleValue,dtype=np.float)
hdf5Group2.create_dataset("YErrorRadiances",data=yErrorRadiances,dtype=np.float)
hdf5Group2.create_dataset("YErrorRadianceFactors",data=yErrorRadianceFactors,dtype=np.float)






calibrationTime = calibrationTimes[2]
hdf5Group3 = hdf5File.create_group(calibrationTime)
hdf5Group3.create_dataset("DiffractionOrder",data=diffractionOrders,dtype=np.float)
hdf5Group3.create_dataset("Pixels",data=pixels,dtype=np.float)
if channel=="lno": hdf5Group3.create_dataset("YCountsToRadiances",data=yCountsToRadiances,dtype=np.float)
if channel=="lno": hdf5Group3.create_dataset("YRadianceFactorCounts",data=yRadianceFactorCounts,dtype=np.float)
hdf5Group3.create_dataset("YErrorSingleValue",data=yErrorSingleValue,dtype=np.float)
hdf5Group3.create_dataset("YErrorRadiances",data=yErrorRadiances,dtype=np.float)
hdf5Group3.create_dataset("YErrorRadianceFactors",data=yErrorRadianceFactors,dtype=np.float)





if channel=="lno":
    comments = "Dummy calibration at present for testing purposes"
elif channel=="so":
    comments = "No SO radiometric calibration at present"
hdf5File.attrs["Comments"] = comments
hdf5File.attrs["DateCreated"] = str(datetime.now())
hdf5File.close()









