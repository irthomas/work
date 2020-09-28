# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:18:12 2020

@author: iant

READ IN OZONE OCCULTATION DATA AND PLOT COMPARISONS
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

from matplotlib.backends.backend_pdf import PdfPages



DATA_ROOT_DIR_PATH = os.path.normcase(r"D:\DATA\Radiative_Transfer\UVIS\Occultation\Comparison_Retrievals") #Ian extrnal HDD. Change as required

#note that SSI data needs extracting from tar gz before reading in
# GODDARD_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"Goddard\GEM_khayat_fd-b204.dat"))
GODDARD_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"Goddard\GSFC_retrieved_o3.dat"))
SSI_EXTRACTED_DIR_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"SSI\ozone_010c_t010")) #Mike Wolff data is in lots of files in this directory
BIRA_H5_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"BIRA\retrievals_All.h5"))
# OU_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"OU\occ_retrievals_ou.dat"))
OU_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"OU\occ_retrievals_ou_err.dat"))


                                 
def get_goddard_dict(filepath):
    """save Goddard data to a dictionary, where each HDF5 file is an entry"""
    
    # regex = "\s+(\d+[.]\d+)\s+([- ]\d+[.]\d+)\s+(\d+[.]\d+)\s+(\d+[.]\d+)\s+((?:\d+[.]\d+e.\d+|\d+[.]))\s+\d+[.]\d+e.\d+\s+\d+[.]\d+e.\d+\s+(\S+)\s+\S+"
    #new version with 1 sigma error
    regex = "\s+(\d+[.]\d+)\s+([- ]\d+[.]\d+)\s+(\d+[.]\d+)\s+(\d+[.]\d+)\s+((?:\d+[.]\d+e.\d+|\d+[.]))\s+((?:\d+[.]\d+e[+]\d+|\d+[.]))\s+(\S+)\s"
    data = np.fromregex(filepath, regex, 
        dtype={'names': ('ls', 'lat', 'lst', 'alt', 'o3', 'o3_error', 'filename'),
               'formats': (np.float, np.float, np.float, np.float, np.float, np.float, 'U30')}
        )


    filenames = data["filename"].tolist()
    filenames_unique = list(set(filenames))
    
    goddard_dict = {}
    for each_filename in filenames_unique:
        indices = [index for index,value in enumerate(filenames) if value == each_filename]
        middle_index = int(len(indices)/2)
        goddard_dict[each_filename] = {
            "ls":data["ls"][indices[0]], 
            "lat":data["lat"][indices[middle_index]], 
            "lst":data["lst"][indices[middle_index]], 
            "alt":data["alt"][indices], 
            "o3":data["o3"][indices],
            "o3_error":data["o3_error"][indices],
            }
    return goddard_dict






def get_ssi_dict(dirpath):
    """save SSI data to a dictionary, where each HDF5 file is an entry"""

    dirlist = sorted(os.listdir(dirpath))
    # regex = "\s+(\d+[.]\d+)\s+[- ]\d+[.]\d+\s+([- ]\d+[.]\d+)\s+(\d+[.]\d+)\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+\d+[.]\d+e.\d+\s+\d+[.]\d+e.\d+\s+((?:\d+[.]\d+e.\d+|\d+[.]))\s+\S+"
    regex = "\s+(\d+[.]\d+)\s+[- ]\d+[.]\d+\s+([- ]\d+[.]\d+)\s+(\d+[.]\d+)\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+(?:\d+[.]\d+e.\d+|\d+[.])\s+\d+[.]\d+e.\d+\s+\d+[.]\d+e.\d+\s+((?:\d+[.]\d+e.\d+|\d+[.]))\s+((?:\d+[.]\d+e.\d+|\d+[.]))\s+\S+"
    ssi_dict = {}
    for each_filename in dirlist:
        h5_filename = each_filename[:27]
        
        data = np.fromregex(os.path.join(dirpath, each_filename), regex, 
            dtype={'names': ('alt', 'lat', 'lst', 'o3', 'o3_error'),
                   'formats': (np.float, np.float, np.float, np.float, np.float)}
            )

        #get ls from header line in file
        ls_data = np.fromregex(os.path.join(dirpath, each_filename), "\s+(\d+.\d+)\s+-\s+L_S", [('ls', np.float)])
        ls = ls_data["ls"][0]
        
        #not always data!!
        if len(data["alt"])>0:
            middle_index = int(len(data["alt"])/2)
            
            ssi_dict[h5_filename] = {
                "ls":ls, 
                "lat":data["lat"][middle_index], 
                "lst":data["lst"][middle_index], 
                "alt":data["alt"], 
                "o3":data["o3"],
                "o3_error":data["o3_error"],
                }
            
    return ssi_dict




# def get_bira_dict_old(dirpath):
#     """save BIRA data to a dictionary, where each HDF5 file is an entry - old version where each retrieval in separate h5 file"""

#     bira_dict = {}
    
#     for each_path, _, filenames in os.walk(dirpath):
#         for each_filename in filenames:
#             # relative_path = os.path.relpath(each_path, dirpath)
#             relative_filepath = os.path.join(each_path, each_filename)
    
#             hdf5_filename = each_filename[:27]
    
#             with h5py.File(relative_filepath, "r") as hdf5_file:
#                 ls = 0.0 #TODO: calculate Ls (not included in Arianna's files) or take it from other datasets
    
    
#                 o3 = hdf5_file['Pass_1']['Fit_0']['O3']['x_den'][...]
#                 alt = hdf5_file['Pass_1']['Fit_0']['O3']['zf'][...]
#                 lat = hdf5_file['Pass_1']['Fen_01']['Geometry']['Geo_Lat'][...]
#                 # lon = hdf5_file['Pass_1']['Fen_01']['Geometry']['Geo_Lon'][...]
#                 lst = hdf5_file['Pass_1']['Fen_01']['Geometry']['Geo_LST'][...]
    
#             bira_dict[hdf5_filename] = {"ls":ls, "lat":float(lat), "lst":float(lst), "alt":alt, "o3":o3}

#     return bira_dict


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
        o3_error = science_group["DnO3_Err"][...]
        alt = science_group["zf"][...]
        dof = science_group["DOF"][...]
        
    for file_index, hdf5_filename in enumerate(filenames):
        
        #ignore zero values
        non_zero_indices = np.where(alt[:, file_index] > 0.0)
        
        #save to dictionary
        bira_dict[hdf5_filename.decode()] = \
            {"ls":ls[file_index], 
             "lat":lat[file_index], 
             "lst":lst[file_index], 
             "alt":np.ndarray.flatten(alt[non_zero_indices, file_index]), 
             "o3":np.ndarray.flatten(o3[non_zero_indices, file_index]),
             "o3_error":np.ndarray.flatten(o3_error[non_zero_indices, file_index]),
             "dof":np.ndarray.flatten(dof[non_zero_indices, file_index]),
             }

    return bira_dict



def get_ou_dict(filepath):
    """save OU data to a dictionary, where each HDF5 file is an entry"""
    
    filepath = OU_DAT_FILE_PATH
    # regex = "(\d+[.]\d+)\s*([- ]\d+[.]\d+)\s+(\d+[.]\d+)\s+(\d+[.]\d+)\s+((?:\d+[.]\d+e[+]\d+|\d+[.]\d+))\s+(\S+)"
    regex = "(\d+[.]\d+)\s*([- ]\d+[.]\d+)\s+(\d+[.]\d+)\s+(\d+[.]\d+)\s+((?:\d+[.]\d+e[+]\d+|\d+[.]\d+))\s+((?:\d+[.]\d+e[+]\d+|\d+[.]\d+))\s+(\S+)"
    data = np.fromregex(filepath, regex, 
        dtype={'names': ('ls', 'lat', 'lst', 'alt', 'o3', 'o3_error', 'filename'),
               'formats': (np.float, np.float, np.float, np.float, np.float, np.float, 'U30')}
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
            "o3_error":data["o3_error"][indices],
            }
    return ou_dict




# #get data in dictionary form - one key per HDF5 file    
# goddard_data = get_goddard_dict(GODDARD_DAT_FILE_PATH)
# ssi_data = get_ssi_dict(SSI_EXTRACTED_DIR_PATH)
# bira_data = get_bira_dict(BIRA_H5_FILE_PATH)
# ou_data = get_ou_dict(OU_DAT_FILE_PATH)




# #get list of HDF5 filenames
# goddard_filenames = list(goddard_data.keys())
# ssi_filenames = list(ssi_data.keys())
# bira_filenames = list(bira_data.keys())
# ou_filenames = list(ou_data.keys())


# #find common HDF5 files present in all comparison datasets

# #find which are present in both Goddard and SSI datasets
# matching_filenames1 = sorted(list(set(goddard_filenames).intersection(ssi_filenames)))

# #find which are present in both BIRA and OU datasets
# matching_filenames2 = sorted(list(set(bira_filenames).intersection(ou_filenames)))

# #find which are present in all datasets
# matching_filenames = sorted(list(set(matching_filenames1).intersection(matching_filenames2)))




"""old figure code - choose 9 random files to compare"""
# file_indices = np.linspace(0, len(matching_filenames)-1, 9)
                          
# #make 9 subplots
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 9))

# #fudge to make big x and y labels work
# fig.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.ylabel("Altitude")
# plt.xlabel("O3 Number Density")

# alpha=0.2
# #loop through each subplot, adding data from the two dictionaries
# for ax_index, ax in enumerate(axes.flat):
    
#     matching_filename = matching_filenames[int(file_indices[ax_index])]

#     ax.set_title("%s (Ls:%0.1f, lat:%0.0f)" %(matching_filename, goddard_data[matching_filename]["ls"], goddard_data[matching_filename]["lat"]))
    
#     ax.plot(goddard_data[matching_filename]["o3"], goddard_data[matching_filename]["alt"], label="Goddard", color="b")
#     x1 = goddard_data[matching_filename]["o3"] - goddard_data[matching_filename]["o3_error"]
#     x2 = goddard_data[matching_filename]["o3"] + goddard_data[matching_filename]["o3_error"]
#     ax.fill_betweenx(goddard_data[matching_filename]["alt"], x1, x2, color="b", alpha=alpha)
    
 
#     ax.plot(ssi_data[matching_filename]["o3"], ssi_data[matching_filename]["alt"], label="SSI", color="orange")
#     x1 = ssi_data[matching_filename]["o3"] - ssi_data[matching_filename]["o3_error"]
#     x2 = ssi_data[matching_filename]["o3"] + ssi_data[matching_filename]["o3_error"]
#     ax.fill_betweenx(ssi_data[matching_filename]["alt"], x1, x2, color="orange", alpha=alpha)

#     ax.plot(bira_data[matching_filename]["o3"], bira_data[matching_filename]["alt"], label="BIRA", color="g")
#     x1 = bira_data[matching_filename]["o3"] - bira_data[matching_filename]["o3_error"]
#     x2 = bira_data[matching_filename]["o3"] + bira_data[matching_filename]["o3_error"]
#     ax.fill_betweenx(bira_data[matching_filename]["alt"], x1, x2, color="g", alpha=alpha)

#     ax.plot(ou_data[matching_filename]["o3"], ou_data[matching_filename]["alt"], label="OU", color="r")
#     x1 = ou_data[matching_filename]["o3"] - ou_data[matching_filename]["o3_error"]
#     x2 = ou_data[matching_filename]["o3"] + ou_data[matching_filename]["o3_error"]
#     ax.fill_betweenx(ou_data[matching_filename]["alt"], x1, x2, color="r", alpha=alpha)
    
#     ax2 = ax.twiny()
#     ax2.plot(bira_data[matching_filename]["dof"], bira_data[matching_filename]["alt"], linestyle="--", color="k")
#     ax2.set_xlim((0,1))
#     ax.set_xscale('log')
#     ax.legend()
    

# fig.tight_layout()
# fig.savefig("ozone_comparison.png")



# """save all plots to pdf in 2x2 subplots"""
alpha=0.2
with PdfPages('ozone_comparisons.pdf') as pdf: #open pdf

    #loop through all matching filenames
    for file_index, matching_filename in enumerate(matching_filenames):
        print(matching_filename)
        
        ax_index = np.mod(file_index, 4) #subplot index: 0,1,2,3,0,1,2,3,0,1, etc

        #if first subplot on the page, open new figure
        if ax_index == 0:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 11))

            #fudge to make big x and y labels work
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel("Altitude                     ")
            plt.xlabel("O3 Number Density")

        #choose subplot 0,1,2 or 3
        ax = axes.flat[ax_index]
        
        #plot to subplot
        ax.set_title("%s (Ls:%0.1f, lat:%0.0f)" %(matching_filename, goddard_data[matching_filename]["ls"], goddard_data[matching_filename]["lat"]))

        ax.plot(goddard_data[matching_filename]["o3"], goddard_data[matching_filename]["alt"], label="Goddard", color="b")
        x1 = goddard_data[matching_filename]["o3"] - goddard_data[matching_filename]["o3_error"]
        x2 = goddard_data[matching_filename]["o3"] + goddard_data[matching_filename]["o3_error"]
        ax.fill_betweenx(goddard_data[matching_filename]["alt"], x1, x2, color="b", alpha=alpha)
        
     
        ax.plot(ssi_data[matching_filename]["o3"], ssi_data[matching_filename]["alt"], label="SSI", color="orange")
        x1 = ssi_data[matching_filename]["o3"] - ssi_data[matching_filename]["o3_error"]
        x2 = ssi_data[matching_filename]["o3"] + ssi_data[matching_filename]["o3_error"]
        ax.fill_betweenx(ssi_data[matching_filename]["alt"], x1, x2, color="orange", alpha=alpha)
    
        ax.plot(bira_data[matching_filename]["o3"], bira_data[matching_filename]["alt"], label="BIRA", color="g")
        x1 = bira_data[matching_filename]["o3"] - bira_data[matching_filename]["o3_error"]
        x2 = bira_data[matching_filename]["o3"] + bira_data[matching_filename]["o3_error"]
        ax.fill_betweenx(bira_data[matching_filename]["alt"], x1, x2, color="g", alpha=alpha)
    
        ax.plot(ou_data[matching_filename]["o3"], ou_data[matching_filename]["alt"], label="OU", color="r")
        x1 = ou_data[matching_filename]["o3"] - ou_data[matching_filename]["o3_error"]
        x2 = ou_data[matching_filename]["o3"] + ou_data[matching_filename]["o3_error"]
        ax.fill_betweenx(ou_data[matching_filename]["alt"], x1, x2, color="r", alpha=alpha)


        ax2 = ax.twiny()
        ax2.plot(bira_data[matching_filename]["dof"], bira_data[matching_filename]["alt"], linestyle="--", color="k")
        ax2.set_xlim((0,1))

        ax.set_ylim((0, 80))
        ax.set_xlim((1e4, 1e11))
        ax.set_xscale('log')
        ax.legend(loc="lower left")
        
        fig.tight_layout()
        
        #after plotting 4th subplot, save and close figure
        if ax_index == 3:
            pdf.savefig()
            plt.close()
    
    #if last occultation is not in 4th subplot, save and close figure anyway
    if ax_index != 3:
        pdf.savefig()
        plt.close()
        






