# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:18:12 2020

@author: iant

READ IN OZONE OCCULTATION DATA AND PLOT COMPARISONS
"""

import numpy as np
import os
import h5py




NUM_MATCH1 = r"((?:(?:|-|\+)\D\d+[.]\d+e(?:-|\+)\d+|(?:|-|\+)\d+[.]\d+))"
NUM_MATCH = r"((?:(?:|-|\+)\D\d+[.]\d+e(?:-|\+)\d+|(?:|-|\+)\d+[.]\d+)|\d+.)"
TEXT_MATCH = r"(\S+)"

                                 
def get_goddard_dict(filepath):
    """save Goddard data to a dictionary, where each HDF5 file is an entry"""
    

    regex = r"\s+".join([NUM_MATCH1] + [NUM_MATCH]*2 + [TEXT_MATCH])


    # regex = "\s+(\d+[.]\d+)\s+([- ]\d+[.]\d+)\s+(\d+[.]\d+)\s+(\d+[.]\d+)\s+((?:\d+[.]\d+e.\d+|\d+[.]))\s+((?:\d+[.]\d+e[+]\d+|\d+[.]))\s+(\S+)\s"
    data = np.fromregex(filepath, regex, 
        dtype={'names': ('alt', 'o3', 'o3_error', 'filename'),
               'formats': (np.float, np.float, np.float, 'U30')}
        )

    filenames = data["filename"].tolist()
    filenames_unique = list(set(filenames))
    
    goddard_dict = {}
    for each_filename in filenames_unique:
        indices = [index for index,value in enumerate(filenames) if value == each_filename]
        # middle_index = int(len(indices)/2)
        goddard_dict[each_filename] = {
            # "ls":data["ls"][indices[0]], 
            # "lat":data["lat"][indices[middle_index]], 
            # "lst":data["lst"][indices[middle_index]], 
            "alt":data["alt"][indices], 
            "o3":data["o3"][indices],
            "o3_error":data["o3_error"][indices],
            }
    return goddard_dict



def get_ssi_dict(dirpath):
    """save SSI data to a dictionary, where each HDF5 file is an entry"""

    dirlist = sorted(os.listdir(dirpath)) 
#    regex = "\s+(\d+[.]\d+)\s+[- ]\d+[.]\d+\s+([- ]\d+[.]\d+)\s+(\d+[.]\d+)\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+\d+[.]\d+e.\d+\s+\d+[.]\d+e.\d+\s+((?:\d+[.]\d+e.\d+|\d+[.]))\s+\S+"
    regex = "\s+(\d+[.]\d+)\s+[- ]\d+[.]\d+\s+([- ]\d+[.]\d+)\s+(\d+[.]\d+)\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+\d+[.]\d+e.\d+\s+\d+[.]\d+e.\d+\s+((?:\d+[.]\d+e.\d+|\d+[.]))\s+((?:\d+[.]\d+e.\d+|\d+[.]))\s+\S+"
    ssi_dict = {}
    for each_filename in dirlist:
        h5_filename = each_filename[:27]
        data = np.fromregex(os.path.join(dirpath, each_filename), regex, 
            dtype={'names': ('alt', 'lat', 'lst', 'o3','err'),
                   'formats': (np.float, np.float, np.float, np.float, np.float)}
            )
        #get ls from header line in file
        ls_data = np.fromregex(os.path.join(dirpath, each_filename), "\s+(\d+.\d+)\s+-\s+L_S", [('ls', np.float)])
        ls = ls_data["ls"][0]
        
        #not always data!!
        if len(data["alt"])>0:
            middle_index = int(len(data["alt"])/2)
            
            ssi_dict[h5_filename] = {"ls":ls, "lat":data["lat"][middle_index], "lst":data["lst"][middle_index], "alt":data["alt"], "o3":data["o3"], "err":data["err"]}
            
    return ssi_dict






def get_bira_dict(filepath):
    """save BIRA data to a dictionary, where each HDF5 file is an entry - new version where retrievals all in same file"""

    bira_dict = {}
    
    with h5py.File(filepath, "r") as hdf5_file:
    
        #1D datasets
        geometry_group = hdf5_file["Geometry"]
        filenames = geometry_group["orbit"][...]
        lat = geometry_group["Lat"][0, :]
        # lon = geometry_group["Lon"][0, :]
        ls = geometry_group["Ls"][0, :]
        lst = geometry_group["LST"][0, :]
    
        #2D datasets
        science_group = hdf5_file["Science"]
        o3 = science_group["Nd_O3"][...]
        alt = science_group["zf"][...]
        err = science_group["DnO3_Err"][...]
        dof = science_group["DOF"][...]
        
    for file_index, hdf5_filename in enumerate(filenames):
        
        #ignore zero values
        non_zero_indices = np.where(alt[:, file_index] > 0.0)
        
        #save to dictionary
        bira_dict[hdf5_filename.decode()] = {"ls":ls[file_index], "lat":lat[file_index], "lst":lst[file_index], "alt":np.ndarray.flatten(alt[non_zero_indices, file_index]), "o3":np.ndarray.flatten(o3[non_zero_indices, file_index]), "err":np.ndarray.flatten(err[non_zero_indices, file_index]), "dof":np.ndarray.flatten(dof[non_zero_indices, file_index])}

    return bira_dict

def get_bira_ac_dict(filepath):
    """save BIRA data to a dictionary, where each HDF5 file is an entry - new version where retrievals all in same file"""

    bira_dictAC = {}
    filenames = []
    with h5py.File(filepath, "r") as hdf5_file:
    
        #1D datasets
        geometry_group = hdf5_file["Geometry"]
        filenames = geometry_group["orbit"][...]          
        lat = geometry_group["Lat"][0, :]
        # lon = geometry_group["Lon"][0, :]
        ls = geometry_group["Ls"][0, :]
        lst = geometry_group["LST"][0, :]
#        year = geometry_group["Year"][: ,0]
#        month = geometry_group["Month"][:, 0]
#        day = geometry_group["Day"][:, 0]
        
        #2D datasets
        science_group = hdf5_file["Science"]
        o3 = science_group["Nd_O3"][...]
        alt = science_group["zf"][...]
        err = science_group["DnO3_Err"][...]
        dof = science_group["DOF"][...]
    
#    filenames = ['']*np.size(timeAC)
#    Time  = ['']*np.size(timeAC)
#    for ii in range(0,np.size(timeAC)):  
#        if len(str(int(timeAC[ii])))<6: 
#            if len(str(int(timeAC[ii])))==5:
#                  Time[ii] ='0'+str(int(timeAC[ii]))
#            else:
#                  Time[ii] ='00'+str(int(timeAC[ii])) 
#        else:
#            Time[ii] =str(int(timeAC[ii]))
#        filenames[ii]=str(int(year[ii]))+str(int(month[ii]))+str(int(day[ii]))+'_'+Time[ii]+'_1p0a_UVIS_'
#    print(filenames)   
    
    for file_index, hdf5_filename in enumerate(filenames):
#        print(hdf5_filename)
#        hdf5_filename = hdf5_filename[:19] + "a" + hdf5_filename[20:]
        #ignore zero values
        non_zero_indices = np.where(alt[:, file_index] > 0.0)

        #save to dictionary
        bira_dictAC[hdf5_filename.decode()] = {"ls":ls[file_index], "lat":lat[file_index], "lst":lst[file_index], "alt":np.ndarray.flatten(alt[non_zero_indices, file_index]), "o3":np.ndarray.flatten(o3[non_zero_indices, file_index]), "err":np.ndarray.flatten(err[non_zero_indices, file_index]), "dof":np.ndarray.flatten(dof[non_zero_indices, file_index])}

    return bira_dictAC


#NO3_err_mean NO3_err_rms NO3_err_rand NO3_err_sys

def get_ou_dict(filepath):
    """save OU data to a dictionary, where each HDF5 file is an entry"""
    
    regex = r"\s+".join([NUM_MATCH]*9 + [TEXT_MATCH])
    
    data = np.fromregex(filepath, regex, 
        dtype={'names': ('ls', 'lat', 'lst', 'alt', 'o3', 'o3_error_mean', 'o3_error_rms', 'o3_error_rand', 'o3_error_sys', 'filename'),
               'formats': (np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, 'U30')}
        )
    
    filenames = data["filename"].tolist()
    filenames_unique = list(set(filenames))
    
    ou_dict = {}
    for each_filename in filenames_unique:
        indices = [index for index,value in enumerate(filenames) if value == each_filename]
        middle_index = int(len(indices)/2)
        ou_dict[each_filename] = {
            "ls":data["ls"][indices[0]], 
            "lat":data["lat"][indices[middle_index]], 
            "lst":data["lst"][indices[middle_index]], 
            "alt":data["alt"][indices], 
            "o3":data["o3"][indices],
            "o3_error":data["o3_error_rand"][indices],
            }
    return ou_dict




