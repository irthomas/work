# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:02:23 2022

@author: iant

RUN LEVEL 0.3K TO 1.0A OF NOMAD DATA PIPELINE ON A FILE
"""



import os
import sys
import logging
from datetime import datetime

from nomad_ops.config import NOMAD_TMP_DIR



# DIRECTORY_STRUCTURE = False
DIRECTORY_STRUCTURE = True

# level = "l0p3k_to_1p0a"
# level = "l0p2a_to_0p3a"
# level = "l0p3a_to_1p0a"
# level = "l0p1d_to_0p1e"
level = "l0p1e_to_0p2a"

# path_to_data_dir = r"C:\Users\iant\Dropbox\NOMAD\Python\other\pipeline"
path_to_data_dir = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_0p1e_unbinned"




if level == "l0p3k_to_1p0a": # occultations
    from nomad_ops.core.hdf5.l0p3k_to_1p0a.l0p3k_to_1p0a_v23 import convert
elif level == "l0p2a_to_0p3a": # SO and LNO
    from nomad_ops.core.hdf5.l0p2a_to_0p3a.l0p2a_to_0p3a_v23 import convert
elif level == "l0p3a_to_1p0a": # LNO nadirs
    from nomad_ops.core.hdf5.l0p3a_to_1p0a.l0p3a_to_1p0a_v23 import convert
elif level == "l0p1d_to_0p1e": # SO and LNO
    from nomad_ops.core.hdf5.l0p1d_to_0p1e.l0p1d_to_0p1e_v23 import convert
elif level == "l0p1e_to_0p2a": # SO and LNO
    from nomad_ops.core.hdf5.l0p1a_0p1e_to_0p2a.l0p1a_0p1e_to_0p2a_v23 import convert


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger( __name__ )


#paths to file
hdf5_filepaths = [
    # "20180428_013814_0p3k_SO_A_E_190.h5",
    # "20180422_003456_0p3a_LNO_1_D_167.h5",
    # "20180422_003456_0p3a_LNO_1_D_169.h5",
    # "20180423_155230_0p3a_LNO_1_D_136.h5",
    # "20221003_231028_0p3a_LNO_1_D_119.h5",

    # "20221125_062726_0p1d_LNO_1_D_189.h5",
    "20221125_082524_0p1e_LNO_1_D_189.h5",
    
    # "20220915_132618_0p1d_LNO_1_F.h5",

]

#loop, making each file
for hdf5_filepath in hdf5_filepaths:
    
    if DIRECTORY_STRUCTURE:
        year = hdf5_filepath[0:4]
        month = hdf5_filepath[4:6]
        day = hdf5_filepath[6:8]
        file_path = os.path.join(path_to_data_dir, year, month, day, hdf5_filepath)
    else:
        file_path = os.path.join(path_to_data_dir, hdf5_filepath)


    
    #make output file path
    hdf5_basename = os.path.basename(hdf5_filepath)
    
    # #set make_plots=False to turn off plots to speed up conversion (optional)
    # out = l1f.TransmittancesAlgo(hdf5_filepath, new_hdf5_filepath, make_plots=True)
    
    print("%s || Starting conversion: %s" %(str(datetime.now())[:-6], hdf5_basename))
    hdf5FilepathOuts = convert(file_path)
    
    for hdf5FilepathOut in hdf5FilepathOuts:
    
        #clean up filename
        old_short_level_name = level[1:5]
        new_short_level_name = level[9:13]
        
        
        output_hdf5_filepath = os.path.join(hdf5FilepathOut)
        output_hdf5_basename = os.path.basename(output_hdf5_filepath)
        new_hdf5_basename = output_hdf5_basename.replace(old_short_level_name, new_short_level_name)
        new_hdf5_filepath = os.path.join(NOMAD_TMP_DIR, new_hdf5_basename)
    
        if os.path.exists(new_hdf5_filepath):
            os.remove(new_hdf5_filepath)
        os.rename(output_hdf5_filepath, new_hdf5_filepath)
        
        print("%s || Done: %s converted to %s" %(str(datetime.now())[:-6], hdf5_basename, new_hdf5_filepath))
        
        # for plotting results
        dsets_to_plot = ["Science/Y", "Science/YReflectanceFactorFlat", "Science/YReflectanceFactorOld", "Science/YReflectanceFactor"]
        
        plot_type = ""
        # plot_type = "spectra simple"
        # plot_type = "spectra by bin"
        
        import h5py
        import numpy as np
        import matplotlib.pyplot as plt
        with h5py.File(new_hdf5_filepath, "r") as h5_f:
            for dset_name in dsets_to_plot:
                if dset_name in h5_f.keys():
                    y = h5_f[dset_name][...]
                    if plot_type == "spectra simple":
                        plt.figure()
                        plt.plot(y.T, alpha=0.1)
                        # plt.ylim((0, 0.5))
                        plt.title(dset_name)
            
                    if plot_type == "spectra by bin":
                        #plot each bin e.g. for bad pixel checks
                        bins = h5_f["Science/Bins"][:, 0]
                        unique_bins = sorted(list(set(bins)))
                        print("Number of bins = ", h5_f.attrs["NBins"])
                        for unique_bin in unique_bins:
                            ixs = np.where(bins == unique_bin)[0]
                            plt.figure()
                            plt.title("%s: bin %i" %(dset_name, unique_bin))
                            plt.plot(y[ixs, :].T, alpha=0.1)