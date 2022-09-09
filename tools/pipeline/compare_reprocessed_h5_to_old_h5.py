# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:24:01 2022

@author: iant
"""

import os
import h5py
import glob

from tools.file.paths import paths

def progress(iterable, length=50):
    total = len(iterable)

    def printProgressBar (iteration):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = "*" * filledLength + '-' * (length - filledLength)
        print(f'\r |{bar}| {percent}% ', end = "\r")

    printProgressBar(0)

    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    print()
    
    

def rename(name):
    
    return name.replace("hdf5_level_1p0a_old", "hdf5_level_1p0a")




def compare(year):

    old_dir = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_1p0a_old/%i/" %year
    
    print("Getting paths of old files")
    h5_paths = sorted(glob.glob(old_dir+r"/**/*.h5", recursive=True))
    
    parent_dir = os.path.split(os.path.dirname(old_dir))[1]
    
    with open("log_%s.txt" %parent_dir, "a") as f:
        f.write("Files which are not found in new directory or have wrong number of spectra\n")
    
    print("Checking if new versions exist")
    for h5_path in progress(h5_paths):
        
        h5_basename = os.path.basename(h5_path)
        
        new_path = rename(h5_path)
        
        
        found = False
        check_shape = True
        
        if os.path.exists(new_path):
            found = True
        else:
            #see if DP replaced by DF
            if "LNO_1_DP" in new_path:
                if os.path.exists(new_path.replace("LNO_1_DP", "LNO_1_DF")):
                    found = True
                    new_path = new_path.replace("LNO_1_DP", "LNO_1_DF")
            if "LNO_1_DF" in new_path:
                if os.path.exists(new_path.replace("LNO_1_DF", "LNO_1_DP")):
                    found = True
                    new_path = new_path.replace("LNO_1_DF", "LNO_1_DP")
    
            if "LNO_1_F" in new_path:
                found = True
                check_shape = False
        
                                                       
        if not found:
            print("%s not found" %h5_basename)
            with open("log_%s.txt" %parent_dir, "a") as f:
                f.write("%s not found\n" %h5_basename)
        
        else:
            
            if check_shape:
                with h5py.File(h5_path, "r") as h5_old:
                    if "Y" in h5_old["Science"].keys():
                        dataset = "Science/Y"
                    elif "YReflectanceFactorFlat" in h5_old["Science"].keys():
                        dataset = "Science/YReflectanceFactorFlat"
                    
                    
                    old_shape = h5_old[dataset][:, 0].shape[0]
        
                with h5py.File(new_path, "r") as h5_new:
                    new_shape = h5_new[dataset][:, 0].shape[0]
                    
                if new_shape != old_shape:
                    text = "%s different shape %i => %i" %(h5_basename, old_shape, new_shape)
                    print(text)
                    with open("log_%s.txt" %parent_dir, "a") as f:
                        f.write(text+"\n")



def read_logs():

    for year in [2018, 2019, 2020, 2021, 2022]:
        log_name = "log_%i.txt" %year
        log_path = os.path.join(paths["REFERENCE_DIRECTORY"], log_name)
        
        with open(log_path, "r") as f:
            lines = f.readlines()

        print(year, len(lines))
            
            
        for line in lines:
            if "not found" in line:
                print(line.split(" ")[0])
                
