#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:42:30 2020

@author: guillaumecruzmermy
"""

import os
import csv
import numpy as np
from numpy import genfromtxt
import h5py
from hdf5_functions_v02 import get_hdf5_attributes,get_dataset_contents,get_hdf5_attribute
import scipy
from scipy import stats
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

import datetime

def load_data(filename, skipr= 0, comments='#'):
    """Load the data file. 2 types of files are supported: h5 and 2-columns ascii files
    ---------------------
        PARAMETERS: 
    ---------------------
    filename: Complete path to the file of data
    skipr: integer in case some rows have to be skipped
    comments: character used to comment lines in the input file
    ---------------------
        RETURNS:
    ---------------------
    data : dictionary containing x and y values    
    """
    
    import os
    
    path_f, file_extension = os.path.splitext(filename)
    
    if file_extension == '.h5':
        h5_file = h5py.File(filename, 'r')
        x = h5_file['Science']['X'][:]
        y = h5_file['Science']['Y'][:]
    
    elif file_extension == '.npz':
         npzfile = np.load(filename)
         x = np.asarray(npzfile['arr_0']).squeeze()
         #(wavenb,solarspec,wavelen)
         y = np.asarray(npzfile['arr_1']).squeeze()
    else:
        try:
            ascii_file = np.loadtxt(filename, skiprows=skipr, comments='#', usecols=[0, 1])
            x = ascii_file[:,0]
            y = ascii_file[:,1]
        
        except TypeError:
            print('The format is not recognized. Please use hdf5 or ascii files.')

    raw_data = {'x': x, 'y': y}
    return raw_data


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def remove_continuum_ALS(data, lam, p, niter):
            
    L = len(data)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*data)
        w = p * (data > z) + (1-p) * (data < z)
        
    estimated_continuum = z;
    #estimated_signal = data/z * np.mean(estimated_continuum);
    return estimated_continuum


#### READ ALL FILENAME OF THE SAME ORDER : example with order 189
all_filename=[]
for root, dirs, files in os.walk("/Users/guillaumecruzmermy/Documents/ExoMars/DATA/NOMAD_LNO/Dayside_nadir/hdf5_level_1p0a"):
    for file in files:
        if file.endswith("189.h5"):
            all_filename = np.append(all_filename,file)
            


##CONSTRUCTION OF MATRIX THAT CONTAINS ALL DATA (you can skip this part if you calibrate file by file)
## Extracting all atributes

dim_spectra = 500  ###Ensure to have enough size to store all spectra per orbit, if the number of spectra is < 500 it's ok cause it's fill with NaN

all_wavenumbers=np.full([np.size(all_filename),320],np.nan)

all_inc_angle = np.full([np.size(all_filename),dim_spectra],np.nan)

all_lat=np.full([np.size(all_filename),4,dim_spectra,2],np.nan)
all_long=np.full([np.size(all_filename),4,dim_spectra,2],np.nan)
all_lst=np.full([np.size(all_filename),4,dim_spectra,2],np.nan)

all_noa=np.full([np.size(all_filename),dim_spectra],np.nan)
all_int_time=np.full([np.size(all_filename),dim_spectra],np.nan)
all_bins=np.full([np.size(all_filename),dim_spectra,2],np.nan)
all_spec_res=np.full([np.size(all_filename),dim_spectra],np.nan)

all_long_sol=np.full([np.size(all_filename),dim_spectra,2],np.nan)

data_cube = np.full([np.size(all_filename), dim_spectra,320],np.nan)

all_data_norm_count = np.full([np.size(all_filename), dim_spectra,320],np.nan)
all_data_yrad = np.full([np.size(all_filename), dim_spectra,320],np.nan)
all_data_yrad_simple = np.full([np.size(all_filename), dim_spectra,320],np.nan)
all_data_yrad_fact = np.full([np.size(all_filename), dim_spectra,320],np.nan)
all_data_yreflec = np.full([np.size(all_filename), dim_spectra,320],np.nan)

all_dist_to_sun=np.full([np.size(all_filename),dim_spectra,2],np.nan)

all_filenames = np.full([np.size(all_filename)],np.nan)

all_timestamp= np.full([np.size(all_filename),dim_spectra],np.nan)


"""set a working directory"""

WORKING_DIRECTORY=os.path.normcase(r"///Users/guillaumecruzmermy/Documents/ExoMars/DATA/NOMAD_LNO/Dayside_nadir")


"""select a file number (0 or 1 for LNO)"""

file_index=0

data_folders=["hdf5_level_1p0a"]


for i in range(0,np.size(all_filename)):
    filenames=all_filename[i]
    """make lists of filenames, file levels and filepaths"""
    
    filename_list=[]
    filelevel_list=[]
    filepaths=[]
    for data_folder in data_folders:
        for filename in filenames:
            year=filenames[0:4]
            month=filenames[4:6]
            day=filenames[6:8]
            
            filepaths.append(os.path.normcase(WORKING_DIRECTORY+os.sep+data_folder+os.sep+year+os.sep+month+os.sep+day+os.sep+filenames)) #si spectres dans différent dossier Année/Mois/Jour
            
            filename_list.append(filenames)
            filelevel_list.append(data_folder)
            
    """read the h5 file"""
    
    filepath_hdf5 = filepaths[0]
    hdf5_file=h5py.File(filepath_hdf5,"r")
        
    """extract attributes"""
            
    wavenumbers_all=hdf5_file["Science"]["X"][:] #spectral axis
    detector_data_all=hdf5_file["Science"]["YUnmodified"][:] #datas
    diff_order=hdf5_file["Channel"]["DiffractionOrder"][:] #get diffraction order
    order=diff_order[0]
        
    noa=hdf5_file['Channel']['NumberOfAccumulations'][:]
    int_time=hdf5_file['Channel']['IntegrationTime'][:] 
    binning=hdf5_file['Channel']['Binning'][:]
    spec_res=hdf5_file['Channel']['SpectralResolution'][:] 
    
    long_sol=hdf5_file['Geometry']['LSubS'][:] 
    
    npoints = get_hdf5_attribute(hdf5_file,"GeometryPoints")
    geometry_names = ["Geometry/Point%i" %point for point in range(1,npoints)]
    geometry_points_lons = np.asfarray([get_dataset_contents(hdf5_file,"Lon",chosen_group=geometry_name)[0] for geometry_name in geometry_names])
    geometry_points_lats = np.asfarray([get_dataset_contents(hdf5_file,"Lat",chosen_group=geometry_name)[0] for geometry_name in geometry_names])
    geometry_points_lst = np.asfarray([get_dataset_contents(hdf5_file,"LST",chosen_group=geometry_name)[0] for geometry_name in geometry_names])
        
    #Raw counts unmodified
    data_unmodif = hdf5_file["Science"]["YUnmodified"][:]
  
    #Raw nadir counts normalised to counts per pixel per secon
    data_norm_count = hdf5_file["Science"]["YNormalisedCounts"][:]
    
    #Radiance calculated from lab blackbody measurements with AOTF + Blaze : not reliable at high diffraction orders 
    data_yrad= hdf5_file["Science"]["YRadiance"][:]
    
    #Radiance calibrated from lab blackbody measurements without AOTF + Blaze : derived from a simple planck function at T° of the blackbody and wavnumbr of the pixel
    data_yrad_simple = hdf5_file["Science"]["YRadianceSimple"][:]
    
    data_yreflec = hdf5_file["Science"]["YReflectanceFactor"][:]
    
    inc_angle= hdf5_file["Geometry"]["MeanIncidenceAngle"][:]
    
    bins=hdf5_file["Science"]["Bins"][:]
    
    dist_to_sun=hdf5_file["Geometry"]["DistToSun"][:]
    
    timestamp=hdf5_file["Timestamp"][:]
    
    nb_spectra=np.size(detector_data_all[:,0])

    data_cube[i,0:nb_spectra,:]=detector_data_all
    
    all_filenames[i]=filenames[0:8]

    all_timestamp[i,0:nb_spectra]=timestamp
        
    
    all_inc_angle[i,0:nb_spectra]=inc_angle
    
    all_lat[i,:,0:nb_spectra,:]=geometry_points_lats
    all_long[i,:,0:nb_spectra,:]=geometry_points_lons
    all_lst[i,:,0:nb_spectra,:]=geometry_points_lst
    
    
    all_noa[i,0:nb_spectra]=noa
    all_int_time[i,0:nb_spectra]=int_time
    all_bins[i,0:nb_spectra,:]=bins
    all_spec_res[i,0:nb_spectra]=spec_res    
    
    all_long_sol[i,0:nb_spectra,:]=long_sol
    
    all_wavenumbers[i,:]=wavenumbers_all
        
    
    all_data_norm_count[i,0:nb_spectra,:] = data_norm_count
    all_data_yrad[i,0:nb_spectra,:] = data_yrad
    all_data_yrad_simple[i,0:nb_spectra,:] =data_yrad_simple
    all_data_yreflec[i,0:nb_spectra,:]=data_yreflec
    
    all_dist_to_sun[i,0:nb_spectra,:]=dist_to_sun
    

### GET THE TEMPERATURE FROM THE CSV FILE
   
temperature_lno = np.loadtxt(fname=r'''/Users/guillaumecruzmermy/Documents/ExoMars/DATA/heaters_temp_2018-03-24T000131_to_2020-10-04T235949.csv''',delimiter=',',skiprows=1,usecols=(1,2,3,4,5))
time_lno_txt = np.loadtxt(fname=r'''/Users/guillaumecruzmermy/Documents/ExoMars/DATA/heaters_temp_2018-03-24T000131_to_2020-10-04T235949.csv''',delimiter=',',skiprows=1,usecols=(0),dtype='datetime64')
time_lno = (time_lno_txt.astype('float'))/1E6

temp_lno_1=temperature_lno[:,2]
temp_lno_2=temperature_lno[:,3]

### IMPORT IRRADIANCE SOLAR SPECTRUM
filename_solar ='/Users/guillaumecruzmermy/Documents/ExoMars/DATA/Solar/irrad_spectrale_1_5_UA_ACE_kurucz.npz'
raw_data = load_data(filename_solar, skipr=0, comments='#')  

wavenb_kurucz=raw_data['x']

irrad_mars=raw_data['y']
true_irrad_mars = np.zeros([np.size(all_filenames),np.size(irrad_mars)])
for i in range (0,np.size(all_filenames)):
    true_irrad_mars[i,:]=irrad_mars[:]*((1.524**2)/(np.nanmean(all_dist_to_sun[i,:,0],axis=0))**2)  ## Correction of the distance for the irradiance for each filenames
  


### Read data to correct for the temperature in the radiometric calibration
filepath_fact ='/Users/guillaumecruzmermy/Documents/ExoMars/DATA/NOMAD_LNO/Dayside_nadir/correction_radiometric_calibration_factor.h5'
hdf5_fact = h5py.File(filepath_fact, "r")
    
diffraction_order=hdf5_fact['Diffraction Order'][:]
A1 = hdf5_fact['A1'][:]  #Coefficient A for the Linear regression using temperature sensor 1
B1 = hdf5_fact['B1'][:]  #Coefficient B for the Linear regression using temperature sensor 1
A2 = hdf5_fact['A2'][:]  #Coefficient A for the Linear regression using temperature sensor 2
B2 = hdf5_fact['B2'][:]  #Coefficient B for the Linear regression using temperature sensor 2

pos_order = np.where(diffraction_order==189)



obs_norm=np.zeros([nb_obs,np.size(data_cube[0,:,0]),np.size(data_cube[0,0,:])])
baseline_2=np.zeros([nb_obs,np.size(data_cube[0,:,0]),np.size(data_cube[0,0,:])])
baseline_obs=np.zeros([nb_obs,np.size(data_cube[0,:,0]),np.size(data_cube[0,0,:])])
obs_corr=np.zeros([nb_obs,np.size(data_cube[0,:,0]),np.size(data_cube[0,0,:])])
obs_final=np.zeros([nb_obs,np.size(data_cube[0,:,0]),np.size(data_cube[0,0,:])])
obs_calib=np.zeros([nb_obs,np.size(data_cube[0,:,0]),np.size(data_cube[0,0,:])])

nb_obs = np.size(data_cube[:,0,0])
for i in range (0,nb_obs):
    nb_spec= (np.size(data_cube[i,:,0])-np.count_nonzero(np.isnan(data_cube[i,:,0])))
    for k in range (0,nb_spec):

        #Normalisation
        obs_norm[i,k,:]=data_cube[i,k,:]/((all_bins[i,k,1] - all_bins[i,k,0])+1)/((all_noa[i,k])*(all_int_time[i,k]/1000)*all_spec_res[i,k])
    
        # Baseline removal
        baseline_obs[i,k,:]=remove_continuum_ALS(obs_norm[i,k,:],1E3,0.9,niter=10)

        obs_corr[i,k,:]=(obs_norm[i,k,:]/baseline_obs[i,k,:])*np.mean(baseline_obs[i,k,:])
        
        #Test to check if temepratures data are available for the filename, if not we take the mean of all temperature (not realistic but it's better than not accounting the temperature)
        if all_timestamp[i,nb_spec-1]>time_lno[-1]:
            temp_inter=np.mean(temp_lno_1)
        else:
            time = all_timestamp[i,k]
            #Linear interpolation of the temperature to get the temporal resolution of the temperature sensor
            temp_interp=np.interp(time,time_lno,temp_lno_1)  #Here I used temperature sensor 1
        
        new_fact =  A1[pos_order]*temp_interp +B1[pos_order]
        
        #Application of the radiometric factor
        obs_calib[i,k,:]=obs_corr[i,k,:]*new_fact
        
        #Account for the incident angle
        obs_final[i,k,:]=obs_calib[i,k,:]/np.cos(all_inc_angle[i,k]*np.pi/180)
    
  

### Inteprolation of the NOMAD wavenumbers on the irradiance spectra (Kurucz-ACE) to compute the reflectance
kurucz_wvnb=np.zeros([np.size(all_filenames),np.size(all_wavenumbers[0,:])])
for i in range (0,np.size(all_filenames)):
    f = interpolate.interp1d(wavenb_kurucz,true_irrad_mars[i,:])
    kurucz_wvnb[i,:]=f(all_wavenumbers[i,:])


### Only use data where incident angle < 80
pos_80=np.where((all_inc_angle<80))

### Reflectance computation

reflectance =np.pi* obs_final[pos_80[0],pos_80[1],:]/kurucz_wvnb[pos_80[0],:]
reflectance[np.where(reflectance>1)]=1
reflectance[np.where(reflectance<0)]=0
