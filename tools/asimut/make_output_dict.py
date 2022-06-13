# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:32:04 2022

@author: iant

READ ASIMUT OUTPUT TO A DICTIONARY
"""

import numpy as np
import configparser
import h5py



def read_inp_to_dict(inp_filepath):
    """read the necessary info from the .inp file into a dictionary"""
    
    
    inp_config = configparser.ConfigParser(allow_no_value=True)
    inp_config.read(inp_filepath)
    inp_config_dict = inp_config._sections
    
    d = {}
    
    d["A_nu0"] = float(inp_config_dict["SP1"]["aotfcentralwnb"].replace("[", "").replace("]", ""))
    d["YError"] = inp_config_dict["SP1"]["datayerrorselect"]
    
    vals = inp_config_dict["SP1"]["spectraid_list"]
    vals = vals.replace("val[","").replace("]","").split(" ")
    d["indices"] = [int(v)-1 for v in vals] #0 indexed for python


    return d




def read_h5_output(out_filepath):
    """read the necessary info from the asimut hdf5 output file into a dictionary"""

    with h5py.File(out_filepath) as h5_fout:
    
        n = len(h5_fout.keys())
    
        y_all = np.zeros((n, 320))
        yr_all = np.zeros((n, 320))
        
    
        #loop through passes in file. 1 pass per spectrum retrieved
        for key in h5_fout.keys():
            ix = int(key.replace("Pass_",""))
            y = h5_fout[key]["Fit_0"]["Y"][...]
            yr = h5_fout[key]["Fit_0"]["YCALC"][...]
            y_all[n - ix, :] = y #store in reverse order
            yr_all[n - ix, :] = yr
        

    return y_all, yr_all


    

def make_output_dict(inp_filepath, out_filepath, h5_f):
    """make a dictionary containing data from the input file, hdf5 output file and the original 1.0a file"""

    d = read_inp_to_dict(inp_filepath)
    y, yr = read_h5_output(out_filepath)

    d["y"] = y
    d["yr"] = yr

    x = h5_f["Science/X"][0,:]
    y_error = h5_f["Science/%s" %d["YError"]][...]
    alts = np.mean(h5_f["Geometry/Point0/TangentAltAreoid"][...], axis=1)
    
    alts_retrieved = alts[d["indices"]]
    y_error_retrieved = y_error[d["indices"], :]
    
    d["x_in"] = x
    d["y_error"] = y_error_retrieved
    d["alts"] = alts_retrieved
        
    return d

