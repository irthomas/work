# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:18:12 2020

@author: iant

READ IN OZONE OCCULTATION DATA AND PLOT COMPARISONS
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.backends.backend_pdf import PdfPages

from get_ozone_retrieval_data import get_goddard_dict, get_ssi_dict, get_bira_dict, get_ou_dict, get_bira_ac_dict


OUTPUT_DIRECTORY = os.getcwd()



# DATA_ROOT_DIR_PATH = os.path.normcase(r"/bira-iasb/projects/NOMAD/Science/Radiative_Transfer/UVIS/Occultation/Comparison_Retrievals") #BIRA server
DATA_ROOT_DIR_PATH = os.path.normcase(r"D:\DATA\Radiative_Transfer\UVIS\Occultation\Comparison_Retrievals") #Ian extrnal HDD. Change as required


#note that SSI data needs extracting from tar gz before reading in
#GODDARD_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"Goddard/GEM_khayat_fd-b204.dat"))
GODDARD_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"Goddard/GSFC_retrieved_o3.dat"))
SSI_EXTRACTED_DIR_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"SSI/ozone_010c_t010")) #Mike Wolff data is in lots of files in this directory
# BIRA_H5_FILE_PATH_apriori = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"BIRA/retrievals_All_112019_testSa.h5"))
BIRA_H5_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"BIRA/retrievals_All.h5"))
BIRA_AC_H5_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"BIRA/Retrieval_acv_2019_11_v6.h5"))

OU_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"OU/occ_retrievals_ou.dat")) #new
# OU_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"OU/occ_retrievals_ou_err.dat")) #old


#NOVEMBER 2019

#note that SSI data needs extracting from tar gz before reading in
GODDARD_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"Goddard/201911final/occ_retrievals_gsfc_201911_err_rand.dat"))
SSI_EXTRACTED_DIR_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"SSI/ozone_010c_t010")) #Mike Wolff data is in lots of files in this directory
BIRA_H5_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"BIRA/retrievals_All_112019_testErrRandom.h5"))
OU_DAT_FILE_PATH = os.path.join(DATA_ROOT_DIR_PATH, os.path.normcase(r"OU/occ_retrievals_ou_201911.dat")) #new



# #get data in dictionary form - one key per HDF5 file
print("Getting Goddard data")
goddard_data = get_goddard_dict(GODDARD_DAT_FILE_PATH)
print("Getting SSI data")
# ssi_data = get_ssi_dict(SSI_EXTRACTED_DIR_PATH)
print("Getting BIRA data")
bira_data = get_bira_dict(BIRA_H5_FILE_PATH)
bira_ac_data_temp = get_bira_ac_dict(BIRA_AC_H5_FILE_PATH)
# bira_data_apriori = get_bira_dict(BIRA_H5_FILE_PATH_apriori)
print("Getting OU data")
ou_data = get_ou_dict(OU_DAT_FILE_PATH)




#fudge to correct filenames
bira_ac_data = {}
for key, value in bira_ac_data_temp.items():
    new_key = key.replace("1p0a", "1p0b")
    bira_ac_data[new_key] = value





#get list of HDF5 filenames
goddard_filenames = list(goddard_data.keys())
# ssi_filenames = list(ssi_data.keys())
bira_filenames = list(bira_data.keys())
bira_ac_filenames = list(bira_ac_data.keys())
ou_filenames = list(ou_data.keys())


#find common HDF5 files present in all comparison datasets

#find which are present in both Goddard and SSI datasets
matching_filenames1 = sorted(list(set(goddard_filenames).intersection(bira_ac_filenames)))

#find which are present in both BIRA and OU datasets
matching_filenames2 = sorted(list(set(bira_filenames).intersection(ou_filenames)))

##find which are present in both BIRA and BIRA_AC datasets
#matching_filenames3 = sorted(list(set(bira_filenames).intersection(bira_AC_filenames)))

#find which are present in both BIRA and BIRA_AC datasets
# matching_filenames3 = sorted(list(set(matching_filenames2).intersection(bira_filenames_apriori)))

#find which are present in all datasets
matching_filenames = sorted(list(set(matching_filenames1).intersection(matching_filenames2)))




## """save all plots to pdf in 2x2 subplots"""
alpha=0.2
with PdfPages(OUTPUT_DIRECTORY+os.sep+"ozone_comparisons_V8.pdf") as pdf: #open pdf

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
        ax.set_title("%s (Ls:%0.1f, lat:%0.0f)" %(matching_filename, bira_data[matching_filename]["ls"], bira_data[matching_filename]["lat"]))

        ax.plot(goddard_data[matching_filename]["o3"], goddard_data[matching_filename]["alt"], label="Goddard", color="b")
        x1 = goddard_data[matching_filename]["o3"] - goddard_data[matching_filename]["o3_error"]
        x2 = goddard_data[matching_filename]["o3"] + goddard_data[matching_filename]["o3_error"]
        ax.fill_betweenx(goddard_data[matching_filename]["alt"], x1, x2, color="b", alpha=alpha)
        
     
        # ax.plot(ssi_data[matching_filename]["o3"], ssi_data[matching_filename]["alt"], label="SSI", color="orange")
        # x1 = ssi_data[matching_filename]["o3"] - ssi_data[matching_filename]["err"]
        # x2 = ssi_data[matching_filename]["o3"] + ssi_data[matching_filename]["err"]
        # ax.fill_betweenx(ssi_data[matching_filename]["alt"], x1, x2, color="orange", alpha=alpha)
    
        ax.plot(bira_data[matching_filename]["o3"], bira_data[matching_filename]["alt"], label="BIRA", color="g")
        x1 = bira_data[matching_filename]["o3"] - bira_data[matching_filename]["err"]
        x2 = bira_data[matching_filename]["o3"] + bira_data[matching_filename]["err"]
        ax.fill_betweenx(bira_data[matching_filename]["alt"], x1, x2, color="g", alpha=alpha)
    
        ax.plot(ou_data[matching_filename]["o3"], ou_data[matching_filename]["alt"], label="OU", color="r")
        x1 = ou_data[matching_filename]["o3"] - ou_data[matching_filename]["o3_error"]
        x2 = ou_data[matching_filename]["o3"] + ou_data[matching_filename]["o3_error"]
        ax.fill_betweenx(ou_data[matching_filename]["alt"], x1, x2, color="r", alpha=alpha)

        ax.plot(bira_ac_data[matching_filename]["o3"], bira_ac_data[matching_filename]["alt"], label="BIRA_AC", color="limegreen")
        x1 = bira_ac_data[matching_filename]["o3"] - bira_ac_data[matching_filename]["err"]
        x2 = bira_ac_data[matching_filename]["o3"] + bira_ac_data[matching_filename]["err"]
        ax.fill_betweenx(bira_ac_data[matching_filename]["alt"], x1, x2, color="limegreen", alpha=alpha)        

        ax2 = ax.twiny()
        ax2.plot(bira_data[matching_filename]["dof"], bira_data[matching_filename]["alt"], linestyle="--", color="k")
#        ax2.plot(bira_data_apriori[matching_filename]["dof"], bira_data_apriori[matching_filename]["alt"], linestyle="--", color="grey")
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





