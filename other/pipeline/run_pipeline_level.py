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

# OUTPUT_DIR_STRUCTURE = False
OUTPUT_DIR_STRUCTURE = True

# level = "l0p3k_to_1p0a"
# level = "l0p2a_to_0p3a"
# level = "l0p3a_to_1p0a"
# level = "l0p1d_to_0p1e"
level = "l0p1e_to_0p2a"

# path_to_data_dir = r"C:\Users\iant\Dropbox\NOMAD\Python\other\pipeline"


# path_to_data_dir = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_0p1e"
path_to_data_dir = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_0p1e_unbinned"
# path_to_data_dir = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_0p1e_5geoms" #change N_OBS_DATETIMES = 5
# path_to_data_dir = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_0p1e_unbinned_5geoms" #change N_OBS_DATETIMES = 5



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
hdf5_filenames = [
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
for hdf5_filename in hdf5_filenames:
    
    if DIRECTORY_STRUCTURE:
        year = hdf5_filename[0:4]
        month = hdf5_filename[4:6]
        day = hdf5_filename[6:8]
        hdf5_filepath = os.path.join(path_to_data_dir, year, month, day, hdf5_filename)
    else:
        hdf5_filepath = os.path.join(path_to_data_dir, hdf5_filename)


    
    #make output file path
    hdf5_basename = os.path.basename(hdf5_filename)
    
    # #set make_plots=False to turn off plots to speed up conversion (optional)
    # out = l1f.TransmittancesAlgo(hdf5_filepath, new_hdf5_filepath, make_plots=True)
    
    print("%s || Starting conversion: %s" %(str(datetime.now())[:-6], hdf5_basename))
    hdf5FilepathOuts = convert(hdf5_filepath)
    
    for hdf5FilepathOut in hdf5FilepathOuts:
    
        #clean up filename
        old_short_level_name = level[1:5]
        new_short_level_name = level[9:13]
        
        
        output_hdf5_filepath = hdf5FilepathOut
        
        if OUTPUT_DIR_STRUCTURE:
            new_hdf5_filepath = hdf5_filepath.replace(old_short_level_name, new_short_level_name)
            os.makedirs(os.path.dirname(new_hdf5_filepath), exist_ok=True)
        else:
            output_hdf5_basename = os.path.basename(output_hdf5_filepath)
            new_hdf5_basename = output_hdf5_basename.replace(old_short_level_name, new_short_level_name)
            new_hdf5_filepath = os.path.join(NOMAD_TMP_DIR, new_hdf5_basename)
    
        #remove existing file(s)
        if os.path.exists(new_hdf5_filepath):
            os.remove(new_hdf5_filepath)
        os.rename(output_hdf5_filepath, new_hdf5_filepath)
        
        print("%s || Done: %s converted to %s" %(str(datetime.now())[:-6], hdf5_basename, new_hdf5_filepath))
        
        # for plotting results
        # dsets_to_plot = ["Science/Y", "Science/YReflectanceFactorFlat", "Science/YReflectanceFactorOld", "Science/YReflectanceFactor"]
        dsets_to_plot = ["Science/YReflectanceFactorFlat"]
        
        plot_type = ""
        plot_type = "surface"
        # plot_type = "spectra simple"
        # plot_type = "spectra by bin"
        
        import h5py
        import numpy as np
        import matplotlib.pyplot as plt
        with h5py.File(new_hdf5_filepath, "r") as h5_f:
            if plot_type == "spectra simple":
                for dset_name in dsets_to_plot:
                    if dset_name in h5_f.keys():
                        y = h5_f[dset_name][...]
                        plt.figure()
                        plt.plot(y.T, alpha=0.1)
                        # plt.ylim((0, 0.5))
                        plt.title(dset_name)
            
            if plot_type == "spectra by bin":
                for dset_name in dsets_to_plot:
                    if dset_name in h5_f.keys():
                        y = h5_f[dset_name][...]
                        #plot each bin e.g. for bad pixel checks
                        bins = h5_f["Science/Bins"][:, 0]
                        unique_bins = sorted(list(set(bins)))
                        print("Number of bins = ", h5_f.attrs["NBins"])
                        for unique_bin in unique_bins:
                            ixs = np.where(bins == unique_bin)[0]
                            plt.figure()
                            plt.title("%s: bin %i" %(dset_name, unique_bin))
                            plt.plot(y[ixs, :].T, alpha=0.1)
                            
            if plot_type == "surface":
                plt.figure(figsize=(10, 8))
                n_points = int([s for s in list(h5_f["Geometry"].keys()) if "Point" in s][-1].replace("Point", ""))
                lats = [h5_f["Geometry/Point%i/Lat" %i][...] for i in range(n_points+1)]
                lons = [h5_f["Geometry/Point%i/Lon" %i][...] for i in range(n_points+1)]
                bins = h5_f["Science/Bins"][:, 0]
                unique_bins = sorted(list(set(bins)))
                
                # for lat_arr, lon_arr in zip(lats[0:1], lons[0:1]): #loop through PointX
                # for lat_arr, lon_arr in zip(lats, lons): #loop through PointX, giving a 2d array of start and end lat/lons
                for k, unique_bin in enumerate(unique_bins):
                    ixs = np.where(unique_bin == bins)[0] #loop through bins
            
                    for i in ixs[0:2]:
                        for j in range(lons[0].shape[1]):
                            rectangle = np.asarray([
                                [lons[1][i, j], lats[1][i, j]], \
                                [lons[2][i, j], lats[2][i, j]], \
                                [lons[3][i, j], lats[3][i, j]], \
                                [lons[4][i, j], lats[4][i, j]], \
                                [lons[1][i, j], lats[1][i, j]], \
                            ])
                                
                            if j == 0 and i == ixs[0]:
                                label = "Bin %i edges" %k
                            else:
                                label = ""
                                
                            plt.plot(rectangle[:, 0], rectangle[:, 1], "C%i" %k, label=label)
                            plt.scatter(lons[0][i, j], lats[0][i, j], c="C%i" %k, label=label.replace("edges", "centre"))
                    
                         
                    
                        # plt.scatter(lon_arr, lat_arr, c=np.repeat(bins, 2).reshape((-1, 2)))
                # plt.scatter(lon_arr[0, 0], lat_arr[0, 0], label=lon_arr[0, 0])
                # plt.scatter(lon_arr[1, 0], lat_arr[1, 0])
                plt.legend(loc="right")
                plt.grid()
                plt.title("%s geometry" %os.path.join(*hdf5_filepath.split(os.sep)[-5:]))
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                        
