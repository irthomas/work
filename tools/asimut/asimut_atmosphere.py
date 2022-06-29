# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:14:06 2022

@author: iant

WRITE ASIMUT APRIORI FILE
"""



KEYS_TO_WRITE = ["Z", "T", "P", "ND", "Ar", "CO", "CO2", "H2O", "N2", "O2", "O3", "d0.1", "d1.5", "d10", "H2O_ice"]

CONVERSIONS = {"P":0.01, "ND":1.0e-6}


def asimut_atmosphere(atmos_dict, atmos_filepath):
    """write apriori atmosphere from gem data dictionary"""
    
    n_lines = len(atmos_dict["z"])
    
    #make list of lines to write to file
    lines = []
    
    #make header
    lines.append("%# " + " ".join(KEYS_TO_WRITE) + "\n")
    
    #loop through altitudes
    for i in range(n_lines):
        line = ""
        
        #loop through columns
        for key in KEYS_TO_WRITE:

            #if data available for a column, write to list
            if key.lower() in atmos_dict.keys():
                
                if key in CONVERSIONS:
                    value = atmos_dict[key.lower()][i] * CONVERSIONS[key]
                else:
                    value = atmos_dict[key.lower()][i]
                
                line += "%0.5e " %value
                
            else:
                #if no data, write zero
                line += "%0.5e " %0.0
                
        line += "\n"
        lines.append(line)
        
    #write to file
    with open(atmos_filepath, "w") as f:
        f.writelines(lines)

        
    