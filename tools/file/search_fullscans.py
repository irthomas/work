# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:06:13 2020

@author: iant

FIND SPECTRA IN FULLSCANS

"""



# import os
import matplotlib.pyplot as plt
# import numpy as np
import re
# import h5py

# from tools.file.paths import paths, FIG_X, FIG_Y
from tools.file.hdf5_functions import make_filelist
# from tools.spectra.fit_polynomial import fit_polynomial
# from tools.general.get_nearest_index import get_nearest_index
from tools.plotting.colours import get_colours

# from tools.spectra.so_non_linearity_correction import make_so_correction_dict, correct_so_observation
from matplotlib.backends.backend_pdf import PdfPages


regex = re.compile(".*_SO_._S") #row120 used 
file_level = "hdf5_level_1p0a"

search_diffraction_order = 152

# vertical_line_nu = 0.0
vertical_line_nu = 3423.93 #add a line on the plots in a given position

y_limits = [0.05, 1.01]



with PdfPages("order_%s_fullscans.pdf" %search_diffraction_order) as pdf: #open pdf

    
    
    hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level, silent=True)
    
    for hdf5_file, hdf5_filename in zip(hdf5_files, hdf5_filenames):
        print(hdf5_filename)
        
        fig, ax = plt.subplots(figsize=(12, 9))

        diffraction_orders = hdf5_file["Channel/DiffractionOrder"][...]
        alts = hdf5_file["Geometry/Point0/TangentAltAreoid"][:, 0]
        
        indices = [i for i,x in enumerate(diffraction_orders) if x == search_diffraction_order]
        
        if len(indices) == 0: #if diff order not in file
            plt.close()
            continue
        
        x = hdf5_file["Science/X"][indices[0], :]
        
        colours = get_colours(len(indices))
        
        spectra = 0
        for i, index in enumerate(indices):
            y = hdf5_file["Science/Y"][index, :]
            if max(y) < y_limits[1] and min(y) > y_limits[0]:
                spectra += 1
                alt = alts[index]
                ax.plot(x, y, color=colours[i], label="i=%i, alt=%0.1f" %(index, alt))
                if vertical_line_nu > 0.0:
                    ax.axvline(x=vertical_line_nu, color="k", linestyle="--")
            
        if spectra>0:
            ax.legend(loc="lower right")
            ax.set_ylabel("Transmittance")
            ax.set_xlabel("Wavenumber cm-1")
            # fig.tight_layout()
            pdf.savefig(dpi=100)
        plt.close()
