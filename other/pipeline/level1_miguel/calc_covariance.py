# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:39:11 2022

@author: iant
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:14:11 2022

@author: iant
"""
# import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl
# import h5py
from copy import deepcopy #replace this
import pandas as pd #remove pandas for data pipeline

########################################################################################################
################################################ Inputs ################################################
########################################################################################################
#
# path_to_hdf5_files:             Directory with the hdf5 files 
# single_file:                    Name of the file to analyze                 
# 
# min_alt:                        The sudy will be done above this altitude (integer)
# max_alt:                        The sudy will be done below this altitude (integer). If negative, max_alt will be the last altitude
#
#
# plot_data:                      True = Plot spectrum at each altitude and the bending correction (False = Don't plot any spectra)
# normalize_cov:                  True = Normalize covariance matrix (False = Don't show normalized matrix)
#
# bending_correction:             True = Correct the bending in NOMAD data (False = Don't correct bending)
# bending_correction_method:      Methods for bending correction: 1 = Polyfit  (Fitting a degree 4 polynomial and sustracting it from the observed spectrum)
#                                                                 2 = Smoothed running mean  (Computing running mean of the spectrum and smoothing it. This will be substracted from the observed spectrum)
# kernel_size_for_smooth:         Size of the kernel to smooth the running mean for the bending correction
#
# dont_move_baseline:             False = Correct the baseline and put it at Transm = 1 (True = don't move baseline)
# MW_for_baseline_i:              MW limits (in pixels) for the central MicroWindow used for identifying the baseline (integers)
# MW_for_baseline_f:
#
# spec_alt_correlations:          True = Generate spectral covariance matrix (False = Don't generate matrix)
#
########################################################################################################
################################################ Outputs ###############################################
########################################################################################################
# all_c:                         Covariance matrix
# c_jk_std:                      Standard deviation computed from the diagonal of the covariance matrix
# new_errTRA:                    New error for the data
########################################################################################################


# path_to_hdf5_files = r'E:\DATA\hdf5\hdf5_level_1p0a\2022\05\05'


plot_data = False                     # Plot spectrum at each altitude and the bending correction
# plot_data = True                      # Plot spectrum at each altitude and the bending correction
normalize_cov = True                  # Normalize covariance matrix

bending_correction = True             # Correct the bending in NOMAD dats
bending_correction_method = 2         # Methods for bending correction: 1 = Polyfit 
                                      #                                 2 = Smoothed running mean

dont_move_baseline = False            # False = Correct the baseline and put it at Transm = 1 (to compute spectral correlations)
MW_for_baseline_i = 140               # MW limits for the central MicroWindow used for identifying the baseline
MW_for_baseline_f = 180

spec_alt_correlations = True          # Generate spectral covariance matrix





def calc_covariance(x, y, yerr, alts, min_alt, h5_basename):
    """inputs for 1 bin in ascending altitude order"""
    kernel_size_for_smooth = 20           # Size of the kernel to smooth the running mean for the bending correction
    
        
    toa_alt_ixs = np.where(alts >= min_alt)[0]
    nfrecs = x.shape[0] #X must be 1d
    # datTRA = y
    errTRA = yerr
    # waven = x
    read_alts = alts[:]
    
    all_trans = y[toa_alt_ixs, :]
    all_error = yerr[toa_alt_ixs, :]
    
    # all_waven = x[:]
    wvn = x[:]
        
    all_alts = alts[toa_alt_ixs]
   
    # Sepctrum at TOA #
    spec_at_toa = all_trans[-1]
    
    # Bending correction at TOA#
    if bending_correction:
        ## Polynomial fitting ##
        if bending_correction_method==1:
            correction_pol_deg = 4
            fit_at_toa = np.polyfit(wvn,spec_at_toa,correction_pol_deg)
            fitfunction_at_toa = np.poly1d(fit_at_toa)
            y_at_toa = fitfunction_at_toa(wvn)
            diff_to_1_at_toa = [1-fit for fit in y_at_toa]
            bending_corrected_spec_at_toa = [transm+diff for transm,diff in zip(spec_at_toa,diff_to_1_at_toa)]
        
        ## Running mean + smoothing ##
        elif bending_correction_method==2:
            spec_for_correction_at_toa = pd.Series(spec_at_toa)
            mean_spec_for_correction_at_toa = spec_for_correction_at_toa.rolling(20,center=True).mean()
            X_at_toa = wvn[int(20/2):-int(20/2)]
            Y_at_toa = mean_spec_for_correction_at_toa[int(20/2):-int(20/2)]
            f_at_toa = interpolate.interp1d(X_at_toa, Y_at_toa, fill_value='extrapolate')
            # y_raw_at_toa = f_at_toa(wvn)
            # kernel = np.ones(kernel_size_for_smooth) / kernel_size_for_smooth
            # y_smoothed_at_toa = np.convolve(y_raw_at_toa, kernel, mode='same')
            X_at_toa = wvn[int(kernel_size_for_smooth/2):-int(kernel_size_for_smooth/2)]
            Y_at_toa = mean_spec_for_correction_at_toa[int(kernel_size_for_smooth/2):-int(kernel_size_for_smooth/2)]
            f_at_toa = interpolate.interp1d(X_at_toa, Y_at_toa, fill_value='extrapolate')
            y_at_toa = f_at_toa(wvn)
            diff_to_1_at_toa = [1-fit for fit in y_at_toa]
            bending_corrected_spec_at_toa = [transm+diff for transm,diff in zip(spec_at_toa,diff_to_1_at_toa)]
        
        if dont_move_baseline:
            baseline_original_at_toa = np.mean(spec_at_toa[MW_for_baseline_i:MW_for_baseline_f])
            baseline_corrected_at_toa = np.mean(bending_corrected_spec_at_toa[MW_for_baseline_i:MW_for_baseline_f])
            delta_baseline_at_toa = baseline_corrected_at_toa-baseline_original_at_toa
            corrected_spec_at_toa = bending_corrected_spec_at_toa-delta_baseline_at_toa
        else:
            corrected_spec_at_toa = bending_corrected_spec_at_toa
    else:
        corrected_spec_at_toa = spec_at_toa



    # Plot TOA spectrum before and after bending correction #
    if plot_data:
        plt.figure(figsize=(10,10))
        plt.title('TOA - '+str(np.round(all_alts[-1],2))+'km')
        if bending_correction:
            plt.plot(wvn,y_at_toa)
            plt.plot(wvn,corrected_spec_at_toa,label='after correction')
        plt.plot(wvn,spec_at_toa,label='before correction')
        plt.legend()
        plt.show()
    
    
    ## Bending correction at every altitude ##
    all_corrected_trans = []
    all_means = []
    for spec,alt in zip(all_trans,all_alts):
        if bending_correction:
            ## Polynomial fitting ##
            if bending_correction_method==1:
                correction_pol_deg = 4
                fit = np.polyfit(wvn,spec,correction_pol_deg)
                fitfunction = np.poly1d(fit)
                y = fitfunction(wvn)
                diff_to_1 = [1-fit for fit in y]
                bending_corrected_spec = [transm+diff for transm,diff in zip(spec,diff_to_1)]
            
            ## Running mean + spline fitting ##
            elif bending_correction_method==2:
                spec_for_correction = pd.Series(spec)
                mean_spec_for_correction = spec_for_correction.rolling(20,center=True).mean()
                X = wvn[int(20/2):-int(20/2)]
                Y = mean_spec_for_correction[int(20/2):-int(20/2)]
                f = interpolate.interp1d(X, Y, fill_value='extrapolate')
                # y_raw = f(wvn)
                kernel_size_for_smooth = 20
                # kernel = np.ones(kernel_size_for_smooth) / kernel_size_for_smooth
                # y_smoothed = np.convolve(y_raw, kernel, mode='same')
                X = wvn[int(kernel_size_for_smooth/2):-int(kernel_size_for_smooth/2)]
                Y = mean_spec_for_correction[int(kernel_size_for_smooth/2):-int(kernel_size_for_smooth/2)]
                f = interpolate.interp1d(X, Y, fill_value='extrapolate')
                y = f(wvn)
                diff_to_1 = [1-fit for fit in y]
                bending_corrected_spec = [transm+diff for transm,diff in zip(spec,diff_to_1)]
            
            ## Baseline correction ##
            if dont_move_baseline:
                baseline_original = np.mean(spec[MW_for_baseline_i:MW_for_baseline_f])
                baseline_corrected = np.mean(bending_corrected_spec[MW_for_baseline_i:MW_for_baseline_f])
                delta_baseline = baseline_corrected-baseline_original
                corrected_spec = bending_corrected_spec-delta_baseline
            else:
                corrected_spec = bending_corrected_spec
                
        else: 
            corrected_spec = spec
    
        corrected_mean = np.mean(corrected_spec)
        all_corrected_trans.append(corrected_spec)
        all_means.append(corrected_mean)
    
    
        ## Plot transmittance before and after the bending correction at each altitude ##
        if plot_data:
            plt.figure(figsize=(10,10))
            plt.title(str(np.round(alt,2))+'km')
            if bending_correction:
                plt.plot(wvn,y)
                plt.plot(wvn,corrected_spec,label='after correction')
            plt.plot(wvn,spec,label='before correction')
            plt.legend()
            plt.show()
    
    
    ############################
    ##   COVARIANCE MATRIX    ##
    ############################
    c_cube = []
    norm_c_cube = []
    c_cube_diag = []
    
    if spec_alt_correlations:
        # N = Window for averaging in the vertical dimension
        N = len(all_alts)-1
        all_corrected_trans = np.asarray(all_corrected_trans)
        print('Study for '+str(N)+' spectra')
        
        # Loop over all N altitudes (centering window in altitude n) #
        n = int(N/2)
        corrected_spec = all_corrected_trans[n]
        # wvn = all_waven[n]
        alt = all_alts[n]
        alt_0 = all_alts[n-int(N/2)]
        alt_f = all_alts[n+int(N/2)]
        print('Mean alt index = '+str(n))
        print('Min. alt = '+str(alt_0))
        print('Mean alt = '+str(alt))
        print('Max. alt = '+str(alt_f))
        print('Creating covariance matrix, please wait...')
        all_c = np.zeros((nfrecs,nfrecs),dtype=float)
        # Calculating covariance #
        Nj = N
        Nk = N
        for j in range(0,nfrecs):
            window_j_i = n-int(Nj/2)
            window_j_f = n+int(Nj/2)
            x_in_window_j = all_corrected_trans[window_j_i:window_j_f,j]
            x_mean_j = np.mean(x_in_window_j)
            sum_j = (x_in_window_j-x_mean_j)
            for k in range(0,nfrecs):
                window_k_i = n-int(Nk/2)
                window_k_f = n+int(Nk/2)
                x_in_window_k = all_corrected_trans[window_k_i:window_k_f,k]
                x_mean_k = np.mean(x_in_window_k)               
                sum_k = (x_in_window_k-x_mean_k)
                sum_jk = sum(sum_k*sum_j)
                # Covariance #
                c_jk = (float(1/(float(N)))) * sum_jk
                all_c[k,j] = deepcopy(c_jk)
        pd.DataFrame(all_c).replace(0, np.nan, inplace=True)
        c_cube.append(all_c)

        # Plot covariance matrix #
        plt.figure(figsize=(10,10))
        vmin = -1.0e-7
        vmax = 1.0e-7
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.title('Covariance matrix @ '+str(np.round(alt_0,2))+' - '+str(np.round(alt_f,2))+' km: %s' %h5_basename)
        plt.imshow(all_c,cmap='seismic',norm=norm)
        plt.xlim(0,320)
        plt.ylim(0,320)
        plt.colorbar()
        plt.show()
        plt.savefig("%s_covariance_matrix_%i.png" %(h5_basename, n))

        # Plot variances (diagonal of covariance matrix) #
        c_jk_diag = np.diagonal(all_c)
        c_jk_std = [np.sqrt(c_jk) for c_jk in c_jk_diag]
        c_cube_diag.append(c_jk_diag)      
        plt.figure(figsize=(10,10))
        plt.title('Covariance matrix diagonal '+str(np.round(alt_0,2))+' - '+str(np.round(alt_f,2))+' km')
        plt.plot(c_jk_diag)
        plt.show()

        # Plot spectrum and errors @ altitude [n] (middle altitude in the range) #
        pd_corrected_trans = pd.Series(all_corrected_trans[n])
        pd_corrected_trans_STD = pd_corrected_trans.rolling(20,center=True).std()
        corrected_trans_mean = np.mean(all_corrected_trans[n])
        plt.figure(figsize=(10,10))
        plt.title('Spectrum @ '+str(np.round(all_alts[n],2))+' km')
        plt.plot(all_corrected_trans[n],label='NOMAD data')
        plt.plot(corrected_trans_mean+all_error[n],ls='--',c='gray',label='YError')
        plt.plot(corrected_trans_mean-all_error[n],ls='--',c='gray')
        plt.plot(corrected_trans_mean+np.asarray(c_jk_std),ls='dotted',c='orange',label='STD from cov.')
        plt.plot(corrected_trans_mean-np.asarray(c_jk_std),ls='dotted',c='orange')
        plt.plot(corrected_trans_mean+np.asarray(pd_corrected_trans_STD),ls='dashdot',c='green',label='Running STD (20)')
        plt.plot(corrected_trans_mean-np.asarray(pd_corrected_trans_STD),ls='dashdot',c='green')
        plt.legend()
        plt.xlabel('# pixel')
        plt.ylabel('Transmittance')
        plt.show()
        plt.savefig("%s_spectrum_and_errors_%i.png" %(h5_basename, n))

        # Normailze covariance matrix to enhance diagonal and non-diagonal elements #
        if normalize_cov:
            norm_all_c = np.zeros((nfrecs,nfrecs),dtype=float)
            for j in range(0,nfrecs):
                for k in range(0,nfrecs):
                    c_jk = all_c[k,j]
                    norm_c_jk = c_jk/float(np.sqrt(c_jk_diag[j])*np.sqrt(c_jk_diag[k]))
                    norm_all_c[j,k] = deepcopy(norm_c_jk)
            norm_c_cube.append(norm_all_c)
            
            # Plot normalized covariance matrix #
            plt.figure(figsize=(10,10))
            vmin = -1
            vmax = 1
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            plt.title('Normalized cov matrix @ '+str(np.round(alt_0,2))+' - '+str(np.round(alt_f,2))+' km')
            plt.imshow(norm_all_c,cmap='jet',norm=norm)
            plt.xlim(0,320)
            plt.ylim(0,320)
            plt.colorbar()
            plt.show()
    
    
    ################################
    # Slices of covariance matrix ##
    ################################
    if spec_alt_correlations:
        pxl = 200
        cube_slice = np.asarray(c_cube)[:,:,pxl]
        pd.DataFrame(cube_slice).replace(0, np.nan, inplace=True)
    
        # Slice of covariance at altitude pixel=pxl #
        nalt1 = 0
    
        cube_slice = np.asarray(c_cube)[:,:,pxl]
        covaiance_nalt1 = cube_slice[nalt1,:]
        covaiance_nalt1_no_diag = deepcopy(cube_slice[nalt1,:])
        covaiance_nalt1_no_diag[pxl] = deepcopy(0)
        cov_std = np.std(covaiance_nalt1_no_diag)
        cov_mean = np.mean(covaiance_nalt1_no_diag)
    
        plt.figure(figsize=(10,3),dpi=100)
        plt.title('Covariance of pxl #'+str(pxl))
        plt.plot(covaiance_nalt1,label='covariance')
        plt.axhline(y=0,ls='--',c='gray')
        plt.axvline(x=pxl,ls='--',c='gray')
        plt.axhline(y=cov_mean+cov_std,c='black',ls='dotted',label='STD')
        plt.axhline(y=cov_mean-cov_std,c='black',ls='dotted')
        plt.legend()
        plt.show()
        print('Number of alts = '+str(len(all_alts)))
        print('Covariance STD (pxl #'+str(pxl)+') = '+str(cov_std))
    
    
    #############################################################
    ## Extending STD (from covariance matrix) to low altitudes ##
    #############################################################
    print('Extrapolating TOA STD to all the altitdues...')
    
    cov_err = [c_jk_std for alt in range(0,len(read_alts))]
    new_errTRA = []
    scale_factor = []
    for CovErrTOA,NominalErrTOA,NominalErr0 in zip(cov_err[-1],errTRA[-1],errTRA[0]):
            factor = (CovErrTOA-NominalErr0)/(NominalErrTOA-NominalErr0)
            scale_factor.append(factor)
    for alt_i in range(0,len(read_alts)):
        cov_err_scaled_i = []
        for pixel in range(0,320):
            NominalErr = errTRA[alt_i][pixel]
            NominalErr0 = errTRA[0][pixel]
            factor = scale_factor[pixel]
            cov_err_scaled = NominalErr0 + (NominalErr-NominalErr0)*factor
            cov_err_scaled_i.append(cov_err_scaled)
        new_errTRA.append(cov_err_scaled_i)
    
    ## Plot comparison between new and nominal error ##
    plt.figure(figsize=(10,10))
    plt.plot(new_errTRA[0],label='New YError',c='red')
    plt.plot(errTRA[0],ls='--',label='Nominal YError',c='blue')
    for alt in range(1,len(new_errTRA)):
        plt.plot(new_errTRA[alt],c='red')
        plt.plot(errTRA[alt],ls='--',c='blue')
    plt.legend()
    plt.show()
    
    
    return c_cube


# if __name__ == "__main__":
#     with h5py.File("20180502_133902_1p0a_SO_A_E_190.h5") as h5_f:
        
#         bins = h5_f['Channel/IndBin'][...]
#         unique_bins = sorted(list(set(bins)))
        
#         bin_ix = 1
#         bin_ixs = np.where(bins == unique_bins[bin_ix])[0]
        
#         x = h5_f["Science/X"][0, :] #x is the same throughout the occultation
#         y = h5_f["Science/Y"][bin_ixs, :]
#         yerr = h5_f["Science/YError"][bin_ixs, :]
#         alts = h5_f["Geometry/Point0/TangentAltAreoid"][bin_ixs, 0] #just get altitudes at start of acquisition
#         min_alt = 100.
        
#     calc_covariance(x, y, yerr, alts, min_alt)