# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:41:34 2022

@author: iant

LIST ALL FILES IN SUBDIRECTORIES
"""

import os
import glob



def list_files(path, extension, filenames_only=False):

    file_paths = glob.glob(r"%s/**/*.%s" %(path, extension), recursive=True)
    
    if filenames_only:
        file_names = [os.path.basename(d) for d in file_paths]
        
        return file_names
    
    return file_paths