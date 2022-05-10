# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:18:12 2020

@author: iant

READ IN OZONE OCCULTATION DATA AND PLOT COMPARISONS
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import matplotlib


from get_ozone_retrieval_data import get_goddard_dict, get_ssi_dict, get_bira_dict, get_ou_dict

OUTPUT_DIRECTORY = os.getcwd()



# DATA_ROOT_DIR_PATH = os.path.normcase(r"/bira-iasb/projects/NOMAD/Science/Radiative_Transfer/UVIS/Occultation/Comparison_Retrievals") #BIRA server
DATA_ROOT_DIR_PATH = os.path.normcase(r"D:\DATA\Radiative_Transfer\UVIS\Occultation\Comparison_Retrievals") #Ian extrnal HDD. Change as required


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
# bira_dataAC = get_biraAC_dict(BIRA_AC_H5_FILE_PATH)
# bira_data_apriori = get_bira_dict(BIRA_H5_FILE_PATH_apriori)
print("Getting OU data")
ou_data = get_ou_dict(OU_DAT_FILE_PATH)




#get list of HDF5 filenames
goddard_filenames = list(goddard_data.keys())
# ssi_filenames = list(ssi_data.keys())
bira_filenames = list(bira_data.keys())
# bira_filenames_apriori = list(bira_data_apriori.keys())
# bira_AC_filenames = list(bira_dataAC.keys())
ou_filenames = list(ou_data.keys())


#find common HDF5 files present in all comparison datasets

#find which are present in both Goddard and SSI datasets
# matching_filenames1 = sorted(list(set(goddard_filenames).intersection(ssi_filenames)))

#find which are present in both BIRA and OU datasets
matching_filenames2 = sorted(list(set(bira_filenames).intersection(ou_filenames)))

##find which are present in both BIRA and BIRA_AC datasets
#matching_filenames3 = sorted(list(set(bira_filenames).intersection(bira_AC_filenames)))

#find which are present in both BIRA and BIRA_AC datasets
# matching_filenames3 = sorted(list(set(matching_filenames2).intersection(bira_filenames_apriori)))

#find which are present in all datasets
matching_filenames = sorted(list(set(goddard_filenames).intersection(matching_filenames2)))






############################################################
############################################################
#  Create a contour plot of differences
############################################################        
fig = plt.figure()
gridspec.GridSpec(3,3)
plt.title('Differences', fontsize=24)
ax1 = plt.subplot2grid((2,3), (0,0), colspan=3, rowspan=1)
ax2 = plt.subplot2grid((2,3), (1,0), colspan=3, rowspan=1)

ax1.set_ylabel('Height, km', fontsize=22)
ax2.set_ylabel('Height, km', fontsize=22)
ax1.set_ylim([0,70])
ax2.set_ylim([0,70])
ax1.set_xlim([0,np.size(matching_filenames)])
ax2.set_xlim([0,np.size(matching_filenames)])
norm1 = matplotlib.colors.Normalize(vmin=-100,vmax=100) 
c_m1 = matplotlib.cm.coolwarm
s_m1 = matplotlib.cm.ScalarMappable(cmap=c_m1, norm=norm1)
s_m1.set_array([])
norm2 = matplotlib.colors.Normalize(vmin=-100,vmax=100) 
c_m2 = matplotlib.cm.coolwarm
s_m2 = matplotlib.cm.ScalarMappable(cmap=c_m2, norm=norm2)
s_m2.set_array([])
obs_dates =np.zeros(np.size(matching_filenames))
ii=0 
#for ii in range(0,100):
for file_index,matching_filename in enumerate(matching_filenames):
     ii = file_index 
#     print(ii)
     print(matching_filename)
     obs_date, obs_time, obs_level,obs_channel, obs_type_code = matching_filename.split('_')
     obs_dates[ii] = obs_date
#     matching_filename = matching_filenames[ii]
#     print(np.around(bira_data[matching_filename]["alt"],decimals=0))
#     print(np.around(goddard_data[matching_filename]["alt"],decimals=0))
     goddard_alt=np.around(goddard_data[matching_filename]["alt"], decimals=1) 
     goddard_O3 = goddard_data[matching_filename]["o3"]
     bira_alt = np.around(bira_data[matching_filename]["alt"],decimals=1)
     bira_O3 = bira_data[matching_filename]["o3"]
     ou_alt=np.around(ou_data[matching_filename]["alt"], decimals=1) 
     ou_O3 = ou_data[matching_filename]["o3"]
     
     matching_alt1 = sorted(list(set(goddard_alt).intersection(bira_alt)))
     bira_alt_int = bira_alt[np.in1d(bira_alt, matching_alt1)]
     goddard_alt_int = goddard_alt[np.in1d(goddard_alt, matching_alt1)] 
     goddard_alt_int = goddard_alt_int[::-1]
     bira_O3_int = bira_O3[np.in1d(bira_alt, matching_alt1)]
     goddard_O3_int  = goddard_O3[np.in1d(goddard_alt, matching_alt1)]
     goddard_O3_int = goddard_O3_int[::-1]
     matching_alt2 = sorted(list(set(goddard_alt).intersection(ou_alt)))
     ou_alt_int = ou_alt[np.in1d(ou_alt, matching_alt2)]
     goddard_alt_int2 = goddard_alt[np.in1d(goddard_alt, matching_alt2)]
     goddard_alt_int2 = goddard_alt_int2[::-1]
     ou_alt_int2 = ou_alt[np.in1d(ou_alt, matching_alt2)]
     ou_O3_int2  = ou_O3[np.in1d(ou_alt, matching_alt2)]
     goddard_O3_int2  = goddard_O3[np.in1d(goddard_alt, matching_alt2)]
     goddard_O3_int2  = goddard_O3_int2 [::-1]
     Diff_bira_goddard = (bira_O3_int-goddard_O3_int)*1E-6
#     print(Diff_bira_goddard,matching_alt1)
     Diff_ou_goddard = (ou_O3_int2-goddard_O3_int2)*1E-6
     
     ax1.scatter(np.ones(np.size(matching_alt1))*ii, bira_alt_int,c=s_m1.to_rgba(Diff_bira_goddard),s=10,edgecolors='none') 
     ax2.scatter(np.ones(np.size(matching_alt2))*ii, ou_alt_int2,c=s_m1.to_rgba(Diff_ou_goddard),s=10,edgecolors='none') 
     
#     ii=ii+1
cbar1 = plt.colorbar(s_m1,orientation='vertical', ax=ax1)
cbar1.set_label('(bira-goddard)$\cdot 10^6$', fontsize=20) 
cbar2 = plt.colorbar(s_m2,orientation='vertical', ax=ax2)
cbar2.set_label('(ou-goddard)$\cdot 10^6$', fontsize=20)

max_index = np.min((len(obs_dates)-1, 1500))

# ax1.set_xticklabels([int(obs_dates[0]),int(max_index/3),int(max_index*2/3),int(max_index),
#                     int(obs_dates[2000]),int(obs_dates[2500]),int(obs_dates[3000])], fontsize=16) 
# ax2.set_xticklabels([int(obs_dates[0]),int(obs_dates[500]),int(obs_dates[1000]),int(obs_dates[1500]),
#                     int(obs_dates[2000]),int(obs_dates[2500]),int(obs_dates[3000])], fontsize=16)     

fig.set_size_inches(14, 10) 
fig.tight_layout()
fig.savefig(OUTPUT_DIRECTORY+os.sep+"Differences_V3.png", dpi=fig.dpi) 
#fig.savefig(dir_out+ 'Differences'+'.eps',format='eps', dpi=fig.dpi)



############################################################
#  Create a contour plot of ratio
############################################################        
fig = plt.figure()
gridspec.GridSpec(3,3)
plt.title('Differences', fontsize=24)
ax1 = plt.subplot2grid((2,3), (0,0), colspan=3, rowspan=1)
ax2 = plt.subplot2grid((2,3), (1,0), colspan=3, rowspan=1)

ax1.set_ylabel('Height, km', fontsize=22)
ax2.set_ylabel('Height, km', fontsize=22)
ax1.set_ylim([0,70])
ax2.set_ylim([0,70])
ax1.set_xlim([0,np.size(matching_filenames)])
ax2.set_xlim([0,np.size(matching_filenames)])
norm1 = matplotlib.colors.Normalize(vmin=0,vmax=2) 
c_m1 = matplotlib.cm.coolwarm
s_m1 = matplotlib.cm.ScalarMappable(cmap=c_m1, norm=norm1)
s_m1.set_array([])
norm2 = matplotlib.colors.Normalize(vmin=0,vmax=2) 
c_m2 = matplotlib.cm.coolwarm
s_m2 = matplotlib.cm.ScalarMappable(cmap=c_m2, norm=norm2)
s_m2.set_array([])
obs_dates =np.zeros(np.size(matching_filenames))
ii=0 
#for ii in range(0,100):
for file_index,matching_filename in enumerate(matching_filenames):
     ii = file_index 
#     print(ii)
     print(matching_filename)
     obs_date, obs_time, obs_level,obs_channel, obs_type_code = matching_filename.split('_')
     obs_dates[ii] = obs_date
#     matching_filename = matching_filenames[ii]
#     print(np.around(bira_data[matching_filename]["alt"],decimals=0))
#     print(np.around(goddard_data[matching_filename]["alt"],decimals=0))
     goddard_alt=np.around(goddard_data[matching_filename]["alt"], decimals=1) 
     goddard_O3 = goddard_data[matching_filename]["o3"]
     bira_alt = np.around(bira_data[matching_filename]["alt"],decimals=1)
     bira_O3 = bira_data[matching_filename]["o3"]
     ou_alt=np.around(ou_data[matching_filename]["alt"], decimals=1) 
     ou_O3 = ou_data[matching_filename]["o3"]
     
     matching_alt1 = sorted(list(set(goddard_alt).intersection(bira_alt)))
     bira_alt_int = bira_alt[np.in1d(bira_alt, matching_alt1)]
     goddard_alt_int = goddard_alt[np.in1d(goddard_alt, matching_alt1)] 
     goddard_alt_int = goddard_alt_int[::-1]
     bira_O3_int = bira_O3[np.in1d(bira_alt, matching_alt1)]
     goddard_O3_int  = goddard_O3[np.in1d(goddard_alt, matching_alt1)]
     goddard_O3_int = goddard_O3_int[::-1]
     matching_alt2 = sorted(list(set(goddard_alt).intersection(ou_alt)))
     ou_alt_int = ou_alt[np.in1d(ou_alt, matching_alt2)]
     goddard_alt_int2 = goddard_alt[np.in1d(goddard_alt, matching_alt2)]
     goddard_alt_int2 = goddard_alt_int2[::-1]
     ou_alt_int2 = ou_alt[np.in1d(ou_alt, matching_alt2)]
     ou_O3_int2  = ou_O3[np.in1d(ou_alt, matching_alt2)]
     goddard_O3_int2  = goddard_O3[np.in1d(goddard_alt, matching_alt2)]
     goddard_O3_int2  = goddard_O3_int2 [::-1]
     Ratio_bira_goddard = (bira_O3_int/goddard_O3_int)
#     print(Diff_bira_goddard,matching_alt1)
     Ratio_ou_goddard = (ou_O3_int2/goddard_O3_int2)
     
     ax1.scatter(np.ones(np.size(matching_alt1))*ii, bira_alt_int,c=s_m1.to_rgba(Ratio_bira_goddard),s=10,edgecolors='none') 
     ax2.scatter(np.ones(np.size(matching_alt2))*ii, ou_alt_int2,c=s_m1.to_rgba(Ratio_ou_goddard),s=10,edgecolors='none') 
     
#     ii=ii+1
cbar1 = plt.colorbar(s_m1,orientation='vertical', ax=ax1)
cbar1.set_label('(bira/goddard)', fontsize=20) 
cbar2 = plt.colorbar(s_m2,orientation='vertical', ax=ax2)
cbar2.set_label('(ou/goddard)', fontsize=20)   
# ax1.set_xticklabels([int(obs_dates[0]),int(obs_dates[500]),int(obs_dates[1000]),int(obs_dates[1500]),
#                     int(obs_dates[2000]),int(obs_dates[2500]),int(obs_dates[3000])], fontsize=16) 
# ax2.set_xticklabels([int(obs_dates[0]),int(obs_dates[500]),int(obs_dates[1000]),int(obs_dates[1500]),
#                     int(obs_dates[2000]),int(obs_dates[2500]),int(obs_dates[3000])], fontsize=16)     

fig.set_size_inches(14, 10) 
fig.tight_layout()
fig.savefig(OUTPUT_DIRECTORY+os.sep+"Ratios_V1.png", dpi=fig.dpi) 

    
print('End')





