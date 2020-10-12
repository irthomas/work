# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:47:58 2020

@author: iant
"""

import os
import platform

if platform.system() == "Windows":
    SYSTEM = "Windows"
else:
    SYSTEM = "Linux"





WAVELENGTH_TO_PLOT = 270.0

PLOT_FIGS = True




    

"""set paths to calibration files"""
if SYSTEM == "Windows":
    from tools.file.paths import paths
    PFM_AUXILIARY_FILES = paths["PFM_AUXILIARY_FILES"]
    NOMAD_TMP_DIR = os.path.join(paths["BASE_DIRECTORY"], "output")
#    ROOT_STORAGE_PATH = os.path.join(paths["BASE_DIRECTORY"], "output")
    
    THUMBNAILS_DESTINATION = os.path.join(paths["BASE_DIRECTORY"], "output")
    import other.pipeline.generic_functions as generics

else:
    import matplotlib
    matplotlib.use('Agg')

    from nomad_ops.config import NOMAD_TMP_DIR, PFM_AUXILIARY_FILES, THUMBNAILS_DESTINATION

    import nomad_ops.core.hdf5.generic_functions as generics






"""set paths to calibration files"""

#input directories
UVIS_RMS_NOISE_DIRECTORY = os.path.join(PFM_AUXILIARY_FILES, "uvis_rms_noise")




"""set constants"""
NA_VALUE = -999

FIG_X = 17
FIG_Y = 9
