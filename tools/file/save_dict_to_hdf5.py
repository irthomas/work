# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:03:05 2020

@author: iant

SAVE DICTIONARY TO HDF5
"""





# correction_dict = {
#     0:{"spectra":np.ones((190, 320)), "coefficients":np.ones((2,320))}, 
#     1:{"spectra":np.ones((190, 320)), "coefficients":np.ones((2,320))}, 
#     2:{"spectra":np.ones((190, 320)), "coefficients":np.ones((2,320))},
#        }

# input_dict = correction_dict


def save_dict_to_hdf5(input_dict, hdf5_filename):
    """convert a nested dictionary containing numpy arrays to hdf5"""

    import h5py
    import numpy as np
    
    def to_string(key_in):
        
        if type(key_in) == int:
            key_str = str(key_in)
        else:
            key_str = key_in
            
        return key_str
    
    
    def new_level(hdf5_in, key_in):
        
        if key_in in list(hdf5_in.keys()):
            return hdf5_in
        else:
            hdf5_in.create_group(key_in)
            return hdf5_in
    
    
    with h5py.File(hdf5_filename+".h5", "w") as hdf5_file_out:
    
        for key1, value in input_dict.items():
            key1_str = to_string(key1)
                
            if type(value) == np.ndarray:
                hdf5_file_out[key1_str] = value
            elif type(value) == dict:
                # print(key1, "contains a dict")
                for key2, value2 in input_dict[key1].items():
                    key2_str = to_string(key2)
                    hdf5_file_out = new_level(hdf5_file_out, key1_str)
         
                    if type(value2) == np.ndarray:
                        hdf5_file_out[key1_str][key2_str] = value2
                    elif type(value2) == dict:
                        # print(key2, "contains a dict")
                        for key3, value3 in input_dict[key1][key2].items():
                            key3_str = to_string(key3)
        
                            if type(value3) == np.ndarray:
                                hdf5_file_out[key1_str][key2_str][key3_str] = value3
                            elif type(value3) == dict:
                                print(key3, "contains a dict - needs more levels")
                            else:
                                print(key3, "type is unknown")
                    else:
                        print(key2, "type is unknown")
            else:
                print(key1, "type is unknown")
            
        
