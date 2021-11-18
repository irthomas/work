# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:12:58 2021

@author: iant

REMAKE LNO COUNTS PER RADIANCE LNO RADIOMETRIC CALIBRATION CURVES FOR GUILLAUME

"""


import os
import h5py
import numpy as np
from datetime import datetime
import re


import matplotlib.pyplot as plt

from tools.file.paths import paths
from tools.file.hdf5_functions import make_filelist

from instrument.nomad_lno_instrument import nu0_aotf, m_aotf

"""pfm ground/inflight cal"""


input_dicts = {
    "20150426_054602_0p1a_LNO_1":{"file_level":"hdf5_level_0p1a", "bb_temp":150.+273., "orders":range(116, 198)},
    "20150426_030851_0p1a_LNO_1":{"file_level":"hdf5_level_0p1a", "bb_temp":150.+273., "orders":range(116, 198)},
}





def opticalTransmission(csl_window=False):
    #0:Wavelength, 1:Lens ZnSe, 2:Lens Si, 3: Lens Ge, 4:AOTF, 5:Par mirror, 6:Planar miror, 7:Detector, 8:Cold filter, 9:Window transmission function
    #10:CSL sapphire window
    optics_all = np.loadtxt(paths["BASE_DIRECTORY"]+os.sep+"reference_files"+os.sep+"nomad_optics_transmission.csv", skiprows=1, delimiter=",")
    if csl_window==False:
        optics_transmission_total = (optics_all[:,1]) * (optics_all[:,2]**3.) * (optics_all[:,3]**2.) * (optics_all[:,4]) * (optics_all[:,5]**2.) * (optics_all[:,6]**4.) * (optics_all[:,7]) * (optics_all[:,8]) * (optics_all[:,9])
    elif csl_window=="only":
        optics_transmission_total = optics_all[:,10]
    else:
        optics_transmission_total = (optics_all[:,1]) * (optics_all[:,2]**3.) * (optics_all[:,3]**2.) * (optics_all[:,4]) * (optics_all[:,5]**2.) * (optics_all[:,6]**4.) * (optics_all[:,7]) * (optics_all[:,8]) * (optics_all[:,9]) * (optics_all[:,10])
    optics_wavenumbers =  10000. / optics_all[:,0]
    return optics_wavenumbers, optics_transmission_total


def planck(xscale, temp, units): #planck function W/cm2/sr/spectral unit
    if units=="microns" or units=="um" or units=="wavel":
        c1=1.191042e8
        c2=1.4387752e4
        return c1/xscale**5.0/(np.exp(c2/temp/xscale)-1.0) / 1.0e4 # m2 to cm2
    elif units=="wavenumbers" or units=="cm-1" or units=="waven":
        c1=1.191042e-5
        c2=1.4387752
        return ((c1*xscale**3.0)/(np.exp(c2*xscale/temp)-1.0)) / 1000.0 / 1.0e4 #mW to W, m2 to cm2



#loop through files
plt.figure(figsize=(12, 7), constrained_layout=True)

lines = ["hdf5_filename,diffraction_order,counts_per_radiance"]
for regex_txt, input_dict in input_dicts.items():
    
    file_level = input_dict["file_level"]
    bb_temp = input_dict["bb_temp"]
    good_orders = input_dict["orders"]
    regex = re.compile(regex_txt)

    hdf5_files, hdf5_filenames, hdf5_paths = make_filelist(regex, file_level, full_path=True)


    hdf5_file = hdf5_files[0]
    hdf5_filename = hdf5_filenames[0]

    y = hdf5_file["Science/Y"][...]
    aotf = hdf5_file["Channel/AOTFFrequency"][...]
    its = hdf5_file["Channel/IntegrationTime"][...]
    binnings = hdf5_file["Channel/Binning"][...]
    naccs = hdf5_file["Channel/NumberOfAccumulations"][...]

    it = float(its[0]) / 1.0e3 #microseconds to seconds
    nacc = float(naccs[0])/2.0 #assume LNO nadir background subtraction is on
    binning = float(binnings[0]) + 1.0 #binning starts at zero
    
    print("integrationTimeFile = %i" %its[0])
    print("integrationTime = %0.2f" %it)
    print("nAccumulation = %i" %nacc)
    print("binning = %i" %binning)


    #convert aotf to orders
    orders = np.asfarray([m_aotf(a) for a in aotf])
    #choose only orders with a signal (not below 115)
    indices = [i for i,v in enumerate(orders) if v in good_orders]
    
    aotf = aotf[indices]
    y = y[indices, :, :]
    orders = orders[indices]
    

    n_spectra = y.shape[0]
    
    #normalise to 1s integration time per pixel
    y = y / (it * nacc) / binning

    """correct obvious detector offset in order 194 data"""
    if hdf5_filename == "20150426_054602_0p1a_LNO_1":
        #find 194
        ix = np.where(orders == 194)[0]
        y[ix, :] = y[ix, :] - 1.05 #fudge to correct bad point


    #find mean of detector bins and chosen spectral pixels
    DETECTOR_BINS_TO_SUM = np.arange(1,24)
    y_mean = np.mean(y[:, DETECTOR_BINS_TO_SUM, :], axis=1)
    y_mean_centre = np.mean(y_mean[:, 160:240], axis=1)
    print("File contains %i detector frames" %n_spectra)
        
    #get central wavenumber of each order
    nus = np.asfarray([nu0_aotf(a) for a in aotf])
    #convert cm-1 to radiance for blackbody
    plancks = planck(nus, bb_temp, "cm-1")


    #account for transmittance of window on the TVac chamber at CSL
    window_nu, window_trans = opticalTransmission(csl_window="only")
    window_interp = np.interp(nus, window_nu[::-1], window_trans[::-1])
    plancks_window = plancks * window_interp     


    
    counts_per_radiance = y_mean_centre / plancks_window
    plt.scatter(orders, counts_per_radiance, label="%s, 150C blackbody" %hdf5_filename)

    #save data for writing to file
    lines.extend(["%s,%i,%0.1f" %(hdf5_filename, order, count) for order, count in zip(orders, counts_per_radiance)])


plt.legend()
plt.grid()
plt.title("Radiometric calibration of LNO from ground calibration data")
plt.ylabel("Mean counts per pixel (for pixels 160-240) per second per radiance (inverse W/cm2/sr/cm-1)")
plt.xlabel("Diffraction order")
plt.savefig("counts_per_radiance.png")

with open("counts_per_radiance.txt", "w") as f:
    for line in lines:
        f.write(line+"\n")