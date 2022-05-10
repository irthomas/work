# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:33:45 2022

@author: iant

SCRIPT TO DELETE 1 MONTH OF DATA + OPTIONAL CACHE.DB
"""

import shutil
import os
import platform

from nomad_ops.config import ROOT_STORAGE_PATH

windows = platform.system() == "Windows"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('month', type=str, help='Enter month/year e.g. 04/2018')
    parser.add_argument('year', type=str, help='Enter month/year e.g. 04/2018')
    parser.add_argument('short_level', type=str, help='Enter short level name')
    parser.add_argument('-delete_cache', action='store_true', help='Delete cache db for each level')
    args = parser.parse_args()
    
month = args.month
year = args.year
short_level = args.short_level

if args.delete_cache:
    delete_cache = True
else:
    delete_cache = False

short_level = "hdf5_10a"

level = "hdf5_level_%sp%s%s" %(short_level[5], short_level[6], short_level[7])

cache_db_path = os.path.join(ROOT_STORAGE_PATH, "hdf5", level, "cache.db")


path_to_delete = os.path.join(ROOT_STORAGE_PATH, "hdf5", level, year, month)
print("Deleting path %s" %path_to_delete)

if not windows:
    if os.path.exists(path_to_delete):
        shutil.rmtree(path_to_delete)
    else:
        print("Error: path %s does not exist" %path_to_delete)
    
    if delete_cache:
        if os.path.exists(cache_db_path):
            os.remove(cache_db_path)
        else:
            print("Error: cache %s does not exist" %cache_db_path)
            


