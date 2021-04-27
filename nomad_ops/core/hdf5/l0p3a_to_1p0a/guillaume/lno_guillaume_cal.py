# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:44:05 2020

@author: iant

ADD GUILLAUME CAL TO LNO NADIR FILES

"""




import numpy as np
import os
from datetime import datetime
# import matplotlib.pyplot as plt
import h5py

from nomad_ops.core.hdf5.l0p3a_to_1p0a.config import RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, HDF5_TIME_FORMAT
from nomad_ops.core.hdf5.l0p3a_to_1p0a.guillaume.temp_lin_reg import get_temp_lin_reg
from nomad_ops.core.hdf5.l0p3a_to_1p0a.functions.baseline_als import baseline_als

def load_old_solar_spectrum():
    
    solar_spectrum_file = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, "irrad_spectrale_1_5_UA_ACE_kurucz.npz")
    
    npzfile = np.load(solar_spectrum_file)
    x = np.asarray(npzfile['arr_0']).squeeze()
    y = np.asarray(npzfile['arr_1']).squeeze()

    return x, y

def load_solar_spectrum():
    
    solar_spectrum_file = os.path.join(RADIOMETRIC_CALIBRATION_AUXILIARY_FILES, "irrad_spectrale_1_5_UA_ACE_kurucz.h5")
    
    with h5py.File(solar_spectrum_file, "r") as f:
        x = f["x"][...]
        y = f["y"][...]

    return x, y


def convert_ref_fac_guillaume(hdf5_file, inc_angle):
    

    wavenumbers = hdf5_file["Science"]["X"][:] #spectral axis
    detector_data_all = hdf5_file["Science"]["Y"][:] #data
    diffraction_order = hdf5_file["Channel"]["DiffractionOrder"][0] #get diffraction order
    bins = hdf5_file["Science"]["Bins"][:]
    
    temperature_datetime_str = hdf5_file["Temperature/TemperatureDateTime"][...]
    temperatures = hdf5_file["Temperature/NominalLNO"][...]
    temperature_datetimes = [datetime.strptime(string.decode(), HDF5_TIME_FORMAT) for string in temperature_datetime_str]
    temperature_datetime_delta = [(dt - datetime(year=2018, month=1, day=1)).total_seconds() for dt in temperature_datetimes]
    
    obs_datetime_str = hdf5_file["Geometry/ObservationDateTime"][:, 0]
    obs_datetimes = [datetime.strptime(string.decode(), HDF5_TIME_FORMAT) for string in obs_datetime_str]
    obs_datetime_delta = [(dt - datetime(year=2018, month=1, day=1)).total_seconds() for dt in obs_datetimes]
    
    
    dist_to_sun=hdf5_file["Geometry"]["DistToSun"][0, 0]
    noa=hdf5_file['Channel']['NumberOfAccumulations'][:]
    int_time=hdf5_file['Channel']['IntegrationTime'][:] 
    # binning=hdf5_file['Channel']['Binning'][:]
    spec_res=hdf5_file['Channel']['SpectralResolution'][:]
    
    # inc_angle= hdf5_file["Geometry"]["Point0"]["IncidenceAngle"][:, 0]
    
    
    
    # wavenb_kurucz, irrad_mars = load_old_solar_spectrum()
    wavenb_kurucz, irrad_mars = load_solar_spectrum()
    
    true_irrad_mars = irrad_mars * ((1.524**2) / (np.nanmean(dist_to_sun, axis=0))**2)  ## Correction of the distance for the irradiance for each filenames
    
    
    nb_spec = len(inc_angle)
    obs_norm = np.zeros_like(detector_data_all)
    baseline_obs = np.zeros_like(detector_data_all)
    obs_corr = np.zeros_like(detector_data_all)
    obs_calib = np.zeros_like(detector_data_all)
    obs_final = np.zeros_like(detector_data_all)
    for k in range(nb_spec):
     
        #Normalisation
        n_rows = (bins[k, 1] - bins[k, 0]) + 1.0
        obs_norm[k, :] = detector_data_all[k, :] / n_rows / (noa[k] * (int_time[k] / 1000.0) * spec_res[k])
    
        # Baseline removal
        baseline_obs[k,:] = baseline_als(obs_norm[k, :], lam=1.0e3, p=0.9, niter=10)
        
        obs_corr[k, :] = (obs_norm[k, :] / baseline_obs[k, :]) * np.mean(baseline_obs[k, :])
        
        obs_temperature = np.interp(obs_datetime_delta[k], temperature_datetime_delta, temperatures)
        
        lin_reg_coeffs = get_temp_lin_reg(diffraction_order)
        
        new_fact =  np.polyval(lin_reg_coeffs, obs_temperature)
        
        #Application of the radiometric factor
        obs_calib[k, :] = obs_corr[k, :] * new_fact
        
        #Account for the incident angle
        obs_final[k, :] = obs_calib[k, :] / np.cos(inc_angle[k] * np.pi / 180.0)
    
    
    
    ### Interpolation of the NOMAD wavenumbers on the irradiance spectra (Kurucz-ACE) to compute the reflectance
    kurucz_wvnb = np.interp(wavenumbers, wavenb_kurucz, true_irrad_mars)
    
    
    # ### Only use data where incident angle < 80
    # pos_80=np.where((inc_angle<80))
    # ### Reflectance computation
    # reflectance = np.pi * obs_final[pos_80[0], :] / kurucz_wvnb[pos_80[0], :]

    reflectance = np.pi * obs_final / kurucz_wvnb
    reflectance[np.where(reflectance > 1)] = 1
    reflectance[np.where(reflectance < 0)] = 0

    return reflectance




#code to save lin reg coeffs to py file
# hdf5_fact = h5py.File("correction_radiometric_calibration_factor.h5")
# diffraction_order=hdf5_fact['Diffraction Order'][:]
# A1 = hdf5_fact['A1'][:]  #Coefficient A for the Linear regression using temperature sensor 1
# B1 = hdf5_fact['B1'][:]  #Coefficient B for the Linear regression using temperature sensor 1
# for order,a,b in zip(diffraction_order, A1, B1):
#     print("%i:[%0.9g,%0.9g]," %(order, a, b))

#code to test on file
# import h5py
# hdf5_file = h5py.File("20200705_031603_0p3a_LNO_1_D_189.h5", "r")

# mean_incidence_angles = hdf5_file["Geometry"]["Point0"]["IncidenceAngle"][:, 0]

# reflectance = convert_ref_fac_guillaume(hdf5_file, mean_incidence_angles)
# # plt.plot(reflectance.T)

# pos_80=np.where((mean_incidence_angles<80))
# plt.plot(reflectance[pos_80[0], :].T)
