# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:01:42 2020

@author: iant

LIST FILES AND THEIR PARAMETERS
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import h5py

from tools.file.hdf5_functions import make_filelist
from tools.plotting.colours import get_colours


regex = re.compile(".*_UVIS_L")
file_level = "hdf5_level_1p0a"


def check_uvis_limbs(regex, file_level):
    """plot and print uvis limbs with minimum tangent height and spectral binning data"""
    hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)
    colours = get_colours(len(hdf5_filenames))
    
    fig, ax = plt.subplots()
    
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
        
        tangent_alts = np.mean(hdf5_file["Geometry/Point0/TangentAltAreoid"][...], axis=1)
        valid_indices = np.where(tangent_alts>-990.0)[0]
        
        ax.plot(tangent_alts[valid_indices], label=hdf5_filename, color=colours[file_index])
        
        binning = hdf5_file["Channel/HorizontalAndCombinedBinningSize"][0]
        
        n_spectra = len(tangent_alts)
        
        centre_indices = np.arange(int(n_spectra/4), int(n_spectra*3/4))
        min_tangent_alt = np.min(tangent_alts[centre_indices])
    
        print(hdf5_filename, "min alt=%0.1f" %min_tangent_alt, "binning=%i" %binning)   
        
    ax.legend()
    
    
    
    
    
    
regex = re.compile("........_......_.*_(SO|LNO)_._[IEGS].*")
file_level = "hdf5_level_1p0a"


def check_invalid_temperatures(regex, file_level):
    """check for invalid values in interpolated temperature field. Make .sh script to reprocess"""
    hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level, silent=True, open_files=False)
    
    # i = 0
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
        # i += 1
        
        # if np.mod(file_index, 100) == 0:
        #     print("%i/%i" %(file_index, len(hdf5_filenames)))
    
        with h5py.File(hdf5_file, "r") as h5:
            if "InterpolatedTemperature" in h5["Channel"].keys():
                interpolated_ts = h5["Channel/InterpolatedTemperature"][...]
            else:
                print("# No temperatures:")
                start = "%s-%s-%sT%s:%s:%s" %(hdf5_filename[0:4], hdf5_filename[4:6], hdf5_filename[6:8], hdf5_filename[9:11], hdf5_filename[11:13], hdf5_filename[13:15])
                end = "%s-%s-%sT%s:%s:%i" %(hdf5_filename[0:4], hdf5_filename[4:6], hdf5_filename[6:8], hdf5_filename[9:11], hdf5_filename[11:13], int(hdf5_filename[13:15])+1)
                print("./scripts/run_as_nomadr ./scripts/run_pipeline.py --log INFO make --from $FROM --to $TO --beg %s --end %s --filter='........_......_.*_(SO|LNO)_._[IEGS].*' --n_proc=8 --all" %(start, end))
    
        #check if 
        mask = ~((-20.0 < interpolated_ts) & (interpolated_ts < 15.0))
        if np.sum(mask) > 0:
    
            start = "%s-%s-%sT%s:%s:%s" %(hdf5_filename[0:4], hdf5_filename[4:6], hdf5_filename[6:8], hdf5_filename[9:11], hdf5_filename[11:13], hdf5_filename[13:15])
            end = "%s-%s-%sT%s:%s:%i" %(hdf5_filename[0:4], hdf5_filename[4:6], hdf5_filename[6:8], hdf5_filename[9:11], hdf5_filename[11:13], int(hdf5_filename[13:15])+1)
            print("./scripts/run_as_nomadr ./scripts/run_pipeline.py --log INFO make --from $FROM --to $TO --beg %s --end %s --filter='........_......_.*_(SO|LNO)_._[IEGS].*' --n_proc=8 --all" %(start, end))
    
    print("Done")




# regex = re.compile("........_......_.*_LNO_._D.*")
regex = re.compile("........_......_.*_UVIS_D")
file_level = "hdf5_level_1p0a"


def check_off_nadir(regex, file_level):
    """check for off-nadir pointings during dayside nadirs"""
    
    hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level, silent=True, open_files=False)
    
    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
        
        if np.mod(file_index, 100) == 0:
            print("%i/%i" %(file_index, len(hdf5_filenames)))
    
        with h5py.File(hdf5_file, "r") as h5:
            latitude = h5["Geometry/Point0/Lat"][:, 0]
            
            n_nans = np.count_nonzero(np.isnan(latitude))
            if n_nans > 0:
                print(hdf5_filename, ":", n_nans)
                
                
check_off_nadir(regex, file_level)