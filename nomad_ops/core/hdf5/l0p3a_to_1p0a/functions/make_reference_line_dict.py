# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:15:17 2020

@author: iant
"""



def make_reference_line_dict(rad_fact_order_dict, hr_simulation_filepath):
    """get dict of high res smoothed reference spectra from files and additional information"""
    
    import numpy as np
    
    
    hr_spectra = np.loadtxt(hr_simulation_filepath, delimiter=",")
    
    reference_dict = {}
    reference_dict["nu_hr"] = hr_spectra[:, 0]
    reference_dict["solar"] = hr_spectra[:, 1]
    reference_dict["molecular"] = hr_spectra[:, 2]
    
    if "solar_molecular" in rad_fact_order_dict.keys():
        solar_molecular = rad_fact_order_dict["solar_molecular"]
        reference_dict["solar_molecular"] = solar_molecular
    else:
        solar_molecular = ""
        reference_dict["solar_molecular"] = ""
        
    
    if solar_molecular == "molecular":
        reference_dict["reference_hr"] = [reference_dict["molecular"]]
        reference_dict["molecule"] = rad_fact_order_dict["molecule"]
        

    elif solar_molecular == "solar":
        reference_dict["reference_hr"] = [reference_dict["solar"]]
        reference_dict["molecule"] = ""
 
    elif solar_molecular == "both":
        reference_dict["reference_hr"] = [reference_dict["solar"], reference_dict["molecular"]]
        reference_dict["molecule"] = rad_fact_order_dict["molecule"]

    else:
        reference_dict["reference_hr"] = [np.array(0.0)]
        reference_dict["molecule"] = ""

    if solar_molecular != "":
        #add other keys to dictionary
        for key_name in ["mean_sig","min_sig","stds_sig","stds_ref"]:
            reference_dict[key_name] = rad_fact_order_dict[key_name]
    else:
        for key_name in ["mean_sig","min_sig","stds_sig","stds_ref"]:
            reference_dict[key_name] = 0.0

    return reference_dict


