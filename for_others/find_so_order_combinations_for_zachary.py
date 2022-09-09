# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:42:40 2022

@author: iant


FIND ALL OBSERVATIONS WHERE A SPECIFIC COMBINATION OF DIFFRACTION ORDERS WERE MEASURED

SO TYPICAL OBSERVATION:
    5 OR 6 DIFFRACTION ORDERS (EXCEPT IF FULLSCAN)
    A H5 FILE IS MADE FOR EACH DIFFRACTION ORDER (EXCEPT IF FULLSCAN)
    DETECTOR FOV SPLIT INTO 4 DETECTOR BINS (EACH WITH SLIGHTLY DIFFERENT POINTING AND HENCE TANGENT ALTITUDE)
    
    AT LEVEL1.0A, THE SPECTRA ARE SORTED BY ALTITUDE, SO THE BINS ARE NOT ORDER SEQUENTIALLY
    
    IN SOME OCCULTATIONS, ONE ORDER IS MEASURED >50KM AND ANOTHER IS MEASURED <50KM
    THIS WAS COMMON AT THE START OF THE MISSION BUT NOT USED NOW
    
    THE HIGHEST ALTITUDE IN EACH H5 FILE DEPENDS ON THE ALTITUDE RANGE OF THE ABSORPTION BANDS WITHIN THAT ORDER
    E.G. ORDER 165 FILES (STRONGEST CO2) HAVE MORE SPECTRA THAN ORDER 134 (WEAK H2O)

"""

import os
import glob
import h5py
import numpy as np

import matplotlib.pyplot as plt

root_dir = r"E:\DATA\hdf5\hdf5_level_1p0a"
data_dir = r"E:\DATA\hdf5\hdf5_level_1p0a\2018\05"

#get list of SO files
h5_paths = sorted(glob.glob(data_dir+os.sep+"**"+os.sep+"*_SO_*.h5", recursive=True))


#get list of H5 filenames from paths
h5s = [os.path.splitext(os.path.basename(s))[0] for s in h5_paths]

#group same observations together in a dictionary
observation_dts = {}

#for now, ignore fullscans and orders that do not measure the full wavelength range
files_to_ignore = ["I_S", "E_S", "_SO_H_", "_SO_L"] #if a filename contains any of these, skip it


#loop through list of filenames
for h5 in h5s:
    
    #check that the filename does not contain any of the strings to be ignored
    if sum([1 if s in h5 else 0 for s in files_to_ignore]) ==0:
        
        #get prefix of h5 filename without order number e.g. 20180531_194558_1p0a_SO_A_E
        observation = h5[0:27]
        
        #get diffraction order
        order = int(h5.rsplit("_", 1)[1])
        
        #if this prefix is new, create a new dictionary entry
        if observation not in observation_dts:
            observation_dts[observation] = []
            
        #add to dictionary
        observation_dts[observation].append(order)


#reverse this - i.e. search for instances where order combinations are the same
#now the diffraction order combination becomes the dictionary key and the values are the filename prefixes
order_combinations = {}


for key, value in observation_dts.items():
    
    if tuple(value) not in order_combinations:
        order_combinations[tuple(value)] = []
        
    order_combinations[tuple(value)].append(key)
    
    

#choose an order combination to search for
chosen_order_combination = (121, 134, 149, 165, 167, 190)

#find list of matching filename prefixes
chosen_observations = order_combinations[chosen_order_combination]

#now loop through prefixes, making the filenames for each order and extracting the data
#load files
for chosen_observation in chosen_observations:
    
    data = {}
    
    for order in chosen_order_combination:
        #add order to filename prefix
        h5_filename = chosen_observation + "_%s.h5" %order
        
        #make path to file
        year = h5_filename[0:4]
        month = h5_filename[4:6]
        day = h5_filename[6:8]
        
        #open file
        with h5py.File(os.path.join(root_dir, year, month, day, h5_filename), "r") as h5_f:
            
            #choose a detector bin (0 to 3)
            chosen_bin = 3 #3 is the most popular
            
            #get list of bins from file, 1 per spectrum. The values correspond to the top detector row
            bins = h5_f["Science/Bins"][:, 0]
            
            #get list of unique bins i.e. 4 values
            unique_bins = sorted(list(set(bins)))
            
            #get indices of spectra where the chosen bin was measured
            bin_ixs = np.where(bins == unique_bins[chosen_bin])[0]
            
            #then get Y and tangent altitude data from h5 file just for those indices
            #take mean of transmittance (y)
            #TODO: must remove atmospheric bands!!
            y = np.mean(h5_f["Science/Y"][bin_ixs, 160:240], axis=1)
            alts = h5_f["Geometry/Point0/TangentAltAreoid"][bin_ixs, 0]

            # spectral calibration is the same for all spectra -> just get first row
            #take mean of wavenumber (x)
            x = np.mean(h5_f["Science/X"][0, :])
            
            #add to dictionary for that order
            data[order] = {"mean_cm-1":x, "trans":y, "alts":alts}
    
    # for alt in 
    
    plt.figure()
    plt.title(chosen_observation)
    for order in data.keys():
        plt.plot(data[order]["trans"], data[order]["alts"], label=order)
        
    plt.legend()
    plt.grid()
    
    #next step: select transmittances for a given altitude
    
    
         