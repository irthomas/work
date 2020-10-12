# -*- coding: utf-8 -*-


"""Requires:
module load java/1.8
export JAVA_HOME=/bira-iasb/softs/opt/java/jre1.8.0_121
To validate PDS products"""


import logging
import os.path
#import sys
import h5py
import numpy as np

from nomad_ops.config import NOMAD_TMP_DIR
import nomad_ops.core.hdf5.generic_functions as generics
from nomad_ops.config import PFM_AUXILIARY_FILES


from nomad_ops.core.hdf5.l1p0a_to_1p0b.make_gem_apriori import get_observation_atmolist_filename
from nomad_ops.core.hdf5.l1p0a_to_1p0b.fit_polynomial import fit_polynomial

from nomad_ops.core.hdf5.l1p0a_to_1p0b.uvis_rms_noise import uvis_rms_noise


__project__   = "NOMAD"
__author__    = "Ian Thomas"
__contact__   = "ian.thomas@oma.be"

#============================================================================================
# 7. CONVERT HDF5 LEVEL 1.0A TO 1.0B
#
# DONE:
#
# GENERATE AND VALIDATE PSA PRODUCTS
# MAKE A PRIORI LIST FOR OBSERVATIONS
#
# STILL TO DO:
#
# ADVANCED BAD PIXEL REMOVAL FROM SO
# FIXED PATTERN NOISE REMOVAL FROM SO
# PREPARE INPUTS FOR SO/LNO RETRIEVALS
#
#============================================================================================

logger = logging.getLogger( __name__ )

VERSION = 80
OUTPUT_VERSION = "1.0B"

# GENERATE_PSA = True
GENERATE_PSA = False

#MAKE_APRIORI = True
MAKE_APRIORI = False

MAKE_HDF5 = True
# MAKE_HDF5 = False



#LNO recalibration is now done in 0.3A->1.0A level - removed from here
SO_ABSORPTION_LINE_DICTIONARY = {133:3010.21, 134:3025.75, 135:3049.06, 136:3067.0}

#SO pixel correction 
SO_PIXEL_CORRECTION_AUX_FILES = {
    129:os.path.join(PFM_AUXILIARY_FILES, "pixel_correction", "px_correction_order129_windowtop=120.h5"),
    130:os.path.join(PFM_AUXILIARY_FILES, "pixel_correction", "px_correction_order130_windowtop=120.h5"),
    }

SO_INDICES_WITHOUT_ABSORPTIONS = {
    129:list(range(100))+list(range(120, 320)),
    130:list(range(64))+list(range(74, 121))+list(range(131,164))+list(range(181,191))+list(range(200,215))+list(range(226, 320)),
    }




def get_px_correction_dict(diffraction_order):

    pixel_correction_dict = {}
    if diffraction_order in list(SO_PIXEL_CORRECTION_AUX_FILES.keys()):
        with h5py.File(SO_PIXEL_CORRECTION_AUX_FILES[diffraction_order], "r") as f:
            bin_indices_str = list(f.keys())
            bin_indices = [int(i) for i in bin_indices_str]
            for bin_index, bin_index_str in zip(bin_indices, bin_indices_str):
                coefficients = f[bin_index_str]["coefficients"][...]
                pixel_correction_dict[bin_index] = coefficients
    
    else:
        print("Error: no pixel correction data for diffraction order %i" %diffraction_order)
    return pixel_correction_dict


# UVIS I, E, D; SO I, E; and all LNO measurement files can pass here
def convert(hdf5file_path): #passes future filename to pipeline

    hdf5_basename = os.path.splitext(os.path.basename(hdf5file_path))[0]
    
    # logger.info("convert: %s", hdf5_basename)
    tmp_file_path = os.path.join(NOMAD_TMP_DIR, hdf5_basename+".h5")  # the path has to be changed to .tmp

    hdf5FileIn = h5py.File(hdf5file_path, "r")

    observationType = generics.getObservationType(hdf5FileIn)
    channel, channelType = generics.getChannelType(hdf5FileIn)
    
    
    if GENERATE_PSA:
        import nomad_ops.core.psa.l1p0a_to_psa.l1p0a_to_psa as l1p0a_to_psa
 
        #first check if this file really should be converted or not
        generate = False
        
        if channel == "so" and observationType in ["I", "E"]:

            #check for grazing - don't convert these
            hdf5_basename_split = hdf5_basename.split("_")
            if len(hdf5_basename_split) == 7:
                if hdf5_basename_split[4] == "G":
                    generate = False
                    logger.info("%s is a grazing occultation. Skipping", hdf5_basename)
                else:
                    generate = True
            else:
                generate = True
                
            
        elif channel == "lno" and observationType in ["D"]:
            generate = False
        elif channel == "uvis" and observationType in ["I", "E", "D"]:
            generate = True
        else:
            logger.info("PSA product not generated for %s file %s of type %s", channel, hdf5_basename, observationType)
        
        #if channel/obs type combination is good, proceed to generation and validation of file.
        if generate:
            """first run psa converter"""
            l1p0a_to_psa.convert(hdf5file_path, hdf5FileIn) #generate PSA and validate for SO and LNO files

#            elif channel == "uvis":
#                """first run psa converter"""
#                l1p0a_to_psa.convert_uvis(hdf5file_path) #generate PSA and validate for UVIS files
            
    if MAKE_APRIORI:
        if observationType in ["E","I"] and channel=="so":
            logger.info("Making a priori files")
            get_observation_atmolist_filename(hdf5FileIn, hdf5_basename)
    
    #for SO / LNO observations, perform more advanced corrections in preparation for retrievals
    
    if MAKE_HDF5:
        if observationType in ["E","I"] and channel=="so": #lno bad pixel / retrievals not yet implemented
                
            diffraction_order = hdf5FileIn["Channel/DiffractionOrder"][0]
    #         if diffraction_order in SO_ABSORPTION_LINE_DICTIONARY.keys():
                
    #             logger.info("Creating 1.0b SO file")
            
            
    #             with h5py.File(tmp_file_path, "w") as hdf5FileOut:
    #                 generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, []) #copy all attributes
    
    #                 """advanced bad pixel removal"""
    #                 #read in required datasets
    #                 x = hdf5FileIn["Science/X"][...]
    #                 y = hdf5FileIn["Science/Y"][...]
    #                 bins = hdf5FileIn["Science/Bins"][...][:, 0]
    
    #                 pixelSpectralCoefficients = hdf5FileIn["Channel/PixelSpectralCoefficients"][0, :]
    #                 firstPixel = hdf5FileIn["Channel/FirstPixel"][...]
    
    #                 wavenumbers = x[0, :]
    #                 pixels = np.arange(0.0, len(wavenumbers))
        
    #                 #find unique bins and assign bin indices
    #                 unique_bins = sorted(list(set(bins))) #bin tops
    # #                unique_indices = range(len(unique_bins)) #0 to 3
    #                 indBin = np.zeros_like(bins)
    #                 for bin_index, unique_bin in enumerate(unique_bins):
    #                     indBin[bins == unique_bin] = bin_index #assign 0-3 to each bin
                        
    #                 #extra bad pixel correction
    #                 #correct bad pixel in bin2
    #                 bin2_indices = np.where(indBin == 2)[0]
    #                 if bin_index == 2:
    #                     y[bin2_indices, 269] = (y[bin2_indices, 268] + y[bin2_indices, 270])/2.0
    #                     y[bin2_indices, 84] = (y[bin2_indices, 83] + y[bin2_indices, 85])/2.0
            
    #                 """advanced spectral cal"""
    #                 POLYFIT_DEGREE = 3
                
    #                 h20_line_nu = SO_ABSORPTION_LINE_DICTIONARY[diffraction_order]
    #                 h20_line_px = pixels[np.abs(wavenumbers - h20_line_nu).argmin()] #pixel closest to 
    #                 px_search_range = range(int(h20_line_px-5), int(h20_line_px+6), 1) #search +- 5 pixels
                    
                    
    #                 #fit polynomial to first transmittance above T=0.3
    #                 y_mean = np.mean(y[:, 160:240], axis=1)
    #                 atmos_index = np.min(np.where(y_mean > 0.3)[0])
    #                 fit = np.polyval(np.polyfit(pixels, y[atmos_index, :], POLYFIT_DEGREE), pixels)
    #                 y_norm = y[atmos_index, :] - fit #y_norm is flattened centred on 0
                    
    #                 absorption_px = np.argmin(y_norm[px_search_range]) + px_search_range[0] #pixel with abs minimum in data
    
    
    #                 #pixel shift is not as good as wavenumber shift converted to pixel
    # #                    shift_px = absorption_px - h20_line_px #pixel shift
    #                 shift_nu = wavenumbers[absorption_px] - h20_line_nu #shift in wavenumbers
    #                 delta_nu = wavenumbers[absorption_px] - wavenumbers[absorption_px - 1] #wavenumbers per pixel
    #                 real_shift_px = shift_nu / delta_nu
                    
    #                 if real_shift_px > 5.0:
    #                     logger.warning("Pixel shift is %0.3f. Could be too high", real_shift_px)
                    
                    
    #                 pixel1_out = firstPixel - real_shift_px
    #                 x_out_line = np.polyval(pixelSpectralCoefficients, pixels + pixel1_out[0]) * np.float(diffraction_order) #new wavenumber grid
    #                 x_out = np.tile(x_out_line, [len(firstPixel), 1])
                    
    #                 logger.info("Absorption found at %0.3fcm-1, should be %0.3fcm-1. Correcting to %0.3fcm-1 (shifting %0.1f pixels)", wavenumbers[absorption_px], h20_line_nu, x_out_line[absorption_px], real_shift_px)
    #                 logger.info("Wavenumber of first pixel shifted from %0.3f to %0.3fcm-1 (%0.3fcm-1 shift)", wavenumbers[0], x_out_line[0], wavenumbers[0]-x_out_line[0])
               
    #                 #copy datasets to new file
    #                 for dset_path_1, dset_1 in generics.iter_datasets(hdf5FileIn):
    #                     generics.createIntermediateGroups(hdf5FileOut, dset_path_1.split("/")[:-1])
    
       
    #                     dsetcopy=np.array(dset_1)
    #                     if len(dsetcopy.shape) == 0:
    #                         hdf5FileOut.create_dataset(dset_path_1, dtype=dset_1.dtype, data=dsetcopy) #no compression for datasets of size 1
    #                     else:
    #                         if dset_path_1 == "Science/X":
    #                             dsetcopy = x_out
    #                         elif dset_path_1 == "Channel/FirstPixel":
    #                             dsetcopy = pixel1_out
    
    #                         hdf5FileOut.create_dataset(dset_path_1, dtype=dset_1.dtype, data=dsetcopy, compression="gzip", shuffle=True)
    
    #             hdf5FileIn.close()
    #             return [tmp_file_path]
    #         else:
    #             logger.info("Solar occultation advanced bad pixel / spectral cal correction not yet implemented for diffraction order %i (file %s)", diffractionOrder, hdf5_basename)




            if diffraction_order in list(SO_PIXEL_CORRECTION_AUX_FILES.keys()):
                
                logger.info("%s: creating 1.0b SO file", hdf5_basename)
                pixel_correction_dict = get_px_correction_dict(diffraction_order)
                indices_without_absorptions = SO_INDICES_WITHOUT_ABSORPTIONS[diffraction_order]
                
            
                with h5py.File(tmp_file_path, "w") as hdf5FileOut:
                    generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, []) #copy all attributes

                    y = hdf5FileIn["Science/Y"][...]
                    bins = hdf5FileIn["Science/Bins"][:, 0]
                    window_top = hdf5FileIn["Channel/WindowTop"][0]
                    
                    #TODO: check if WindowTop == 120
                    if window_top != 120:
                        logger.warning("%s uses different detector row %i", hdf5_basename, window_top)
                        return []
                        

                    pixels = np.arange(320.0)
                    unique_bins = np.array(sorted(list(set(bins))))
    
    
                    y_corrected = np.zeros_like(y)
                    #loop through spectra in file
                    for spectrum_index, spectrum in enumerate(y):
                        
                        #find bin number of spectrum
                        bin_index = np.where(bins[spectrum_index] == unique_bins)[0][0]
                        
                        
                        #get continuum
                        spectrum_continuum = fit_polynomial(pixels, spectrum, 5, indices=indices_without_absorptions)
                        spectrum_out = np.zeros_like(spectrum)
                        #loop through pixels, applying coefficients for that bin from dictionary based on continuum value
                        for pixel in pixels:
                            coefficients = pixel_correction_dict[bin_index][:, int(pixel)]
                            deviation = np.polyval(coefficients, spectrum_continuum[int(pixel)])
                            spectrum_out[int(pixel)] = spectrum[int(pixel)] - deviation
                    
                        y_corrected[spectrum_index, :] = spectrum_out 


                    #copy datasets to new file
                    for dset_path_1, dset_1 in generics.iter_datasets(hdf5FileIn):
                        generics.createIntermediateGroups(hdf5FileOut, dset_path_1.split("/")[:-1])
    
       
                        dsetcopy=np.array(dset_1)
                        if len(dsetcopy.shape) == 0:
                            hdf5FileOut.create_dataset(dset_path_1, dtype=dset_1.dtype, data=dsetcopy) #no compression for datasets of size 1
                        else:
                            if dset_path_1 == "Science/Y":
                                dsetcopy = y_corrected
    
                            hdf5FileOut.create_dataset(dset_path_1, dtype=dset_1.dtype, data=dsetcopy, compression="gzip", shuffle=True)

                    #save old unmodified dataset
                    hdf5FileOut.create_dataset("Science/YUncorrected", dtype=y.dtype, data=y, compression="gzip", shuffle=True)
    
                hdf5FileIn.close()
                return [tmp_file_path]
            # else:
            #     logger.info("%s: solar occultation pixel correction not yet implemented for diffraction order %i", hdf5_basename, diffraction_order)

        if observationType in ["E","I"] and channel=="uvis": #UVIS RMS noise


            logger.info("%s: creating 1.0b UVIS occultation file", hdf5_basename)
            rms_noise_dict = uvis_rms_noise(hdf5_basename, hdf5FileIn)
        
            with h5py.File(tmp_file_path, "w") as hdf5FileOut:
                generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, []) #copy all attributes

                #copy datasets to new file
                for dset_path_1, dset_1 in generics.iter_datasets(hdf5FileIn):
                    generics.createIntermediateGroups(hdf5FileOut, dset_path_1.split("/")[:-1])

   
                    dsetcopy=np.array(dset_1)
                    if len(dsetcopy.shape) == 0:
                        hdf5FileOut.create_dataset(dset_path_1, dtype=dset_1.dtype, data=dsetcopy) #no compression for datasets of size 1
                    else:
                        hdf5FileOut.create_dataset(dset_path_1, dtype=dset_1.dtype, data=dsetcopy, compression="gzip", shuffle=True)

                #save old unmodified dataset
                rms_noise = rms_noise_dict["rms_noise"]
                hdf5FileOut.create_dataset("Science/YErrorMeanRMS", dtype=rms_noise.dtype, data=rms_noise, compression="gzip", shuffle=True)

            hdf5FileIn.close()
            return [tmp_file_path]

        
            

    
    
        # elif observationType in ["L", "N", "F"] and channel=="lno": #avoid limbs, nightsides and fullscans for now
        #     logger.info("%s: LNO observation is of type %s. Skipping", hdf5_basename, observationType)
    
        # else:
        #     if channel in ["so", "lno"]:
        #         logger.info("File %s is not of type I, E (for SO) or D (for LNO). Advanced bad pixel / spectral cal correction not yet implemented", hdf5_basename)

    hdf5FileIn.close()
    return []




# """check output"""
# file1 = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5\hdf5_level_1p0a\2020\07\08\20200708_034050_1p0a_SO_A_E_130.h5"
# file2 = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\hdf5\hdf5_level_1p0b\2020\07\08\20200708_034050_1p0b_SO_A_E_130.h5"
# index = 123

# import matplotlib.pyplot as plt
# import h5py

# with h5py.File(file1, "r") as f:
#     y1 = f["Science/Y"][...]
# with h5py.File(file2, "r") as f:
#     y2 = f["Science/Y"][...]
    
# print(y1[:, 200])
# plt.figure()
# plt.plot(y1[index, :], label="Before")
# plt.plot(y2[index, :], label="After")

# plt.legend()