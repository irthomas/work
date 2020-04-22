# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:43:18 2020

@author: iant
"""


def make_so_correction_dict(regex, file_level, toa_alt):
    """takes a list of spectra defined by a regex and uses them 
    to derive a pixel correction dictionary of linear fits to transmittance vs 
    deviation from 5th order polynomial continuum fit"""

    import numpy as np
    from tools.file.hdf5_functions_v04 import makeFileList
    from tools.file.get_hdf5_data_v01 import getLevel1Data
    from tools.spectra.fit_polynomial import fit_polynomial
    
    hdf5_files, hdf5_filenames, _ = makeFileList(regex, file_level, silent=True)

    pixels = np.arange(320.0)
    bin_indices = list(range(4))
    
    correction_dict = {}
    #calibrate transmittances for all 4 bins
    for bin_index in bin_indices:
        spectra_in_bin = [] 
        for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
        
            
            #use mean method, returns dictionary
            obs_dict = getLevel1Data(hdf5_file, hdf5_filename, bin_index, silent=True, top_of_atmosphere=toa_alt)
       
            #get all spectra between 0.1<T<0.9
            y_mean = np.mean(obs_dict["y_mean"], axis=1)
            good_indices = np.where((y_mean > 0.1) & (y_mean < 0.9))[0]
            
            
            for spectrum_index in good_indices:
                spectra_in_bin.append(obs_dict["y_mean"][spectrum_index, :])
            
        #derive correction
        spectra_in_bin = np.asfarray(spectra_in_bin)
        correction_dict[bin_index] = {}
        correction_dict[bin_index]["spectra"] = spectra_in_bin
    
        continuum = np.zeros_like(spectra_in_bin)
        deviation = np.zeros_like(spectra_in_bin)
        
        #loop through spectra in this bin. Store continuum polyfit and deviation from fit
        for spectrum_index, spectrum in enumerate(spectra_in_bin):
            polyfit = np.polyfit(pixels, spectrum, 5)
            continuum[spectrum_index, :] = np.polyval(polyfit, pixels)
            deviation[spectrum_index, :] = spectrum - continuum[spectrum_index, :]
    
        correction_dict[bin_index]["continuum"] = continuum
        correction_dict[bin_index]["deviation"] = deviation
    
        #fit deviation from continuum vs continuum transmittance for each pixel
        fit_coefficients = []
        for pixel in pixels:
            pixel = int(pixel)
            linear_fit, coefficients = fit_polynomial(continuum[:, pixel], deviation[:, pixel], coeffs=True)
            fit_coefficients.append(coefficients)
    
        fit_coefficients = np.asfarray(fit_coefficients).T
        correction_dict[bin_index]["coefficients"] = fit_coefficients

    return correction_dict




def correct_so_observation(input_filepath, output_filepath, correction_dict, indices_without_absorptions):
    """apply correction to all spectra in Y dataset in a hdf5 filename
    Save to new file"""

    import numpy as np
    import h5py
    
    from tools.spectra.fit_polynomial import fit_polynomial
    from tools.file.read_write_hdf5 import read_hdf5_to_dict, write_hdf5_from_dict


    with h5py.File(input_filepath+".h5", "r") as hdf5_file:
        y = hdf5_file["Science/Y"][...]
        bins = hdf5_file["Science/Bins"][:, 0]


    pixels = np.arange(320.0)
    unique_bins = np.array(sorted(list(set(bins))))
    
    
    y_corrected = np.zeros_like(y)
    #loop through spectra in file
    for spectrum_index, spectrum in enumerate(y):
        
        #find bin number of spectrum
        bin_index = np.where(bins[spectrum_index] == unique_bins)[0]
        
        #get continuum
        spectrum_continuum = fit_polynomial(pixels, spectrum, 5, indices=indices_without_absorptions)
        spectrum_out = np.zeros_like(spectrum)
        #loop through pixels, applying coefficients for that bin from dictionary based on continuum value
        for pixel in pixels:
            coefficients = correction_dict[int(bin_index)]["coefficients"][:, int(pixel)]
            deviation = np.polyval(coefficients, spectrum_continuum[int(pixel)])
            spectrum_out[int(pixel)] = spectrum[int(pixel)] - deviation
    
        y_corrected[spectrum_index, :] = spectrum_out 


    #write spectra to new file
    replace_datasets = {"Science/Y":y_corrected}
    replace_attributes = {}
    hdf5_datasets, hdf5_attributes = read_hdf5_to_dict(input_filepath)
    write_hdf5_from_dict(output_filepath, hdf5_datasets, hdf5_attributes, replace_datasets, replace_attributes)






