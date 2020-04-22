
import os
import h5py
import numpy as np
import numpy.linalg as la
import gc

import bisect
from scipy.optimize import curve_fit,leastsq
from mpl_toolkits.basemap import Basemap


from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import struct

#import spicewrappers as sw #use cspice wrapper version
from hdf5_functions_v02 import get_hdf5_attributes,get_dataset_contents,write_to_hdf5
from analysis_functions_v01 import interpolate_bad_pixel,sg_filter,find_order,spectral_calibration_simple,write_log,get_filename_list,stop


channel="LNO"
figx=18
figy=9

obspaths = ["20161123_025550_LNO","20161123_192550_LNO","20161122_033050_LNO","20161124_013550_LNO",\
        "20161125_183550_LNO","20161125_195550_LNO","20161125_203550_LNO","20161123_033550_LNO","20161124_025550_LNO",\
        "20161127_152550_LNO","20161127_160550_LNO","20161123_200550_LNO","20161122_225550_LNO","20161124_033550_LNO",\
        "20161123_152550_LNO","20161123_225550_LNO","20161122_233550_LNO","20161123_160550_LNO","20161123_233550_LNO",\
        "20161125_155550_LNO","20161123_005550_LNO","20161123_172550_LNO","20161127_172550_LNO","20161127_180550_LNO",\
        "20161124_005550_LNO","20161123_013550_LNO","20161127_192550_LNO","20161127_200550_LNO"]

hdf5_files=[]
for obspath in obspaths:
    year = obspath[0:4] #get the date from the filename to find the file
    month = obspath[4:6]
    day = obspath[6:8]
#            filename=os.path.normcase(DATA_DIRECTORY+os.sep+"hdf5_level_0p1c"+os.sep+year+os.sep+month+os.sep+day+os.sep+obspath+".h5") #choose a file
    filename=os.path.normcase("insert directory here"+os.sep+year+os.sep+month+os.sep+day+os.sep+obspath+".h5") #choose a file
    hdf5_files.append(h5py.File(filename, "r")) #open file, add to list






"""plot 2d grid of correct aotf vs sg filter abs depth"""

#aotf to wavenumber coefficients from arnaud
aotf2wvn = [-4.7146921e-7,0.168101589,0.001357194]


#make empty arrays to hold all data
nfiles = len(obspaths)
nframes = len(obspaths)*510
pixels = range(320)
all_data_interpolated = np.zeros((nframes,320))
all_aotfs_interpolated = np.zeros(nframes)
aotf_starts=[]
aotf_ends=[]

#    xshifts=[0,3,3,3,3,3,3,0,0,0,0,0,0,5,5,0] 
xshifts=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #shift the spectra in each file by x pixels, so that they match up with the 
absorption_range = 6 #when interpolating a continuum across an absorption band, 



#now loop through files, reading in the detector data and aotf frequencies from each and storing them in the empty arrays
for file_index,hdf5_file in enumerate(hdf5_files):

    """get data from file"""
    print "Reading in file %i: %s" %(file_index,obspaths[file_index])
    detector_data_bins = get_dataset_contents(hdf5_file,"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
    measurement_temperature = np.mean(get_dataset_contents(hdf5_file,"AOTF_TEMP_%s" %channel.upper())[0][2:10])
    print "AOTF range %i to %i at %0.1fC" %(min(aotf_freq_all),max(aotf_freq_all),measurement_temperature)
    if aotf_freq_all[0]<16000: print "Warning: AOTFs in %s are too low - small signal" %obspaths[file_index]
    hdf5_file.close()
    
    #code to correct incorrect telecommand stuck onto detector data by SINBAD
    aotf_freq_range = np.arange(min(aotf_freq_all),min(aotf_freq_all)+2*256,2)
    aotf_freq_corrected = np.append(np.append(aotf_freq_range,aotf_freq_range),aotf_freq_range[0:28])
    if max(aotf_freq_all)-min(aotf_freq_all) != 510:
        print "Error: AOTFs may not be correct" #print error and stop program if there is a problem
        stop()
        
    #average vertical detector data from rows 6 to 18 to make one spectrum per AOTF frequency. Bins will be done later
    detector_data_binned = np.mean(detector_data_bins[:,6:18,:],axis=1) 
    #each file contains two and a bit sweeps through AOTF frequencies. Just take first run through
    detector_data_binned_mean = detector_data_binned[0:256,:]

    """shift spectra in x direction to match files"""
    for frame_index in range(256):
        xshift = xshifts[file_index]
        spectrum = detector_data_binned_mean[frame_index]
        if xshift==0:
            spectrum_shifted = spectrum
        elif xshift>0.:
            spectrum_shifted = np.asfarray(list(spectrum[xshift::]) + [spectrum[-1]]*xshift)
        elif xshift<0.:
            spectrum_shifted = np.asfarray([spectrum[0]]*(xshift*-1)+list(spectrum[:xshift:]))
        detector_data_binned_mean[frame_index] = spectrum_shifted


    #interpolate aotf frequencies and detector data into 1kHz steps and store frequencies and spectra in a big array for all files
    all_aotfs_interpolated[file_index*510:(file_index+1)*510] = np.arange(aotf_freq_range[0],aotf_freq_range[-1],1.0)
    for pixel_index in range(detector_data_binned_mean.shape[1]):
        all_data_interpolated[file_index*510:(file_index+1)*510,pixel_index] = np.interp(all_aotfs_interpolated[file_index*510:(file_index+1)*510],aotf_freq_range,detector_data_binned_mean[:,pixel_index])

    aotf_starts.append(int(aotf_freq_range[0])) #record first and last aotf frequency in each file
    aotf_ends.append(int(aotf_freq_range[-1]))


"""determine start and end points of overlapping region"""
#make a list of all overlapping indices i.e. where a given aotf frequency is present in 2 files. The overlapping region will be deleted from the 2d array so that the files are continuous
range_starts=[0]
range_ends=[0]
overlapping_indices=[]
for index in range(nfiles-1):
    range_starts.append(aotf_ends[index] - aotf_starts[index] + range_starts[-1])
    range_ends.append(range_starts[-1] + aotf_ends[index] - aotf_starts[index+1])
    overlapping_indices.extend(range(range_starts[-1],range_ends[-1]))


#make more empty array to hold data
spectra_uncorrected = np.zeros((nframes,len(pixels)))
spectra_corrected = np.zeros((nframes,len(pixels)))
spectrum1_sg = np.zeros((nframes,len(pixels)))
spectra_absorptions = np.zeros((nframes,len(pixels)))

#    absorption_indices_all=[]

  
"""Loop through each frame"""
for frame_index in range(nframes):
    spectrum1 = all_data_interpolated[frame_index,:]



    
    """pixel 40 is always bad, and can affect the fitting routine, so remove it by interpolation of pixels 39 and 40"""
    spectrum1[40] = np.mean([spectrum1[39],spectrum1[41]])
    spectra_uncorrected[frame_index,:] = spectrum1

    """fit savitsky golay filter (running mean) to spectrum"""
    spectrum1_sg[frame_index,:] = sg_filter(spectrum1, window_size=29, order=2)
#        spectrum1_div = (spectrum1 - spectrum1_sg[frame_index,:])/spectrum1
#        spectrum1_abs_div = np.abs((spectrum1 - spectrum1_sg[frame_index,:])/spectrum1)

#        """find local minima, maxima"""        
#        local_maxima = (np.diff(np.sign(np.diff(spectrum1_div))) < 0).nonzero()[0] + 1 # local max
#        local_minima = (np.diff(np.sign(np.diff(spectrum1_div))) > 0).nonzero()[0] + 1 # local max
    
#        """find points where divided initial spectrum deviates from filtered line and are also local minima. Assume these are absorption bands"""
#        absorption_indices = [local_minimum for local_minimum in local_minima if spectrum1_abs_div[local_minimum]>0.015 and local_minimum<311 and local_minimum>10]
#        absorption_indices_all.append(absorption_indices)
#        absorption_indices=[57,89,94,99,100,108,251,260,267,306] #manually select absorption bands
#        absorption_regions = [[absorption_index-absorption_range,absorption_index+absorption_range] for absorption_index in absorption_indices]
            
#        """for each absorption found, make linear fit across region bounded by absorption"""
#        spectrum2 = spectrum1[:]
#        for absorption_index in absorption_indices:
#            pixel_number_absorption = [absorption_index-absorption_range,absorption_index+absorption_range] #find two points in x
#            pixel_absorption = [spectrum1[absorption_index-absorption_range],spectrum1[absorption_index+absorption_range]] #find two points in y
        
#            """generate linear spectrum between the two points"""
#            coeffs = np.polyfit(pixel_number_absorption,pixel_absorption,1)
#            spectrum2[(absorption_index-absorption_range):(absorption_index+absorption_range)] = np.polyval(coeffs, range(absorption_index-absorption_range,absorption_index+absorption_range,1))
    
    
#        """now that absorptions have been interpolated over, re-run continuum fitting"""
#        spectrum2_sg = sg_filter(spectrum2, window_size=29, order=2)
#        spectrum2_div = (spectrum2 - spectrum2_sg)/spectrum2
#        spectrum2_abs_div = np.abs((spectrum2 - spectrum2_sg)/spectrum2)
    
    """divide original spectrum by this filtered spectrum to remove the general shape, leaving only the absorption bands"""
    spectrum2_absorption = spectrum1/spectrum1_sg[frame_index,:] 

#        spectra_corrected[frame_index,:] = spectrum2_sg
    spectra_absorptions[frame_index,:] = spectrum2_absorption #store in variable

"""delete overlapping regions"""
spectra_absorptions = np.delete(spectra_absorptions,overlapping_indices,axis=0)
aotfs_corrected = np.delete(all_aotfs_interpolated,overlapping_indices)
            
wavenumbers = np.polyval(aotf2wvn, aotfs_corrected)
    
x = wavenumbers
y = pixels
X,Y=np.meshgrid(x,y)

plt.figure(figsize = (figx,figy))
plt.pcolormesh(X,Y,1.0-(np.transpose(spectra_absorptions)))#, aspect=np.int(nfiles/1.5), interpolation=None)
plt.axis([x.min(), x.max(), y[0], y[-1]])

