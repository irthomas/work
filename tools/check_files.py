# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:01:42 2020

@author: iant

LIST FILES AND THEIR PARAMETERS
"""
import re
import numpy as np
import matplotlib.pyplot as plt

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
    
    
