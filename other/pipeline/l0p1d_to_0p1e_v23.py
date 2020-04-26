# -*- coding: utf-8 -*-
# pylint: disable=E1103
# pylint: disable=C0301

import logging
import os.path

import h5py
import numpy as np
import spiceypy as sp

from nomad_ops.config import NOMAD_TMP_DIR

from nomad_ops.config import PFM_AUXILIARY_FILES
import nomad_ops.core.hdf5.generic_functions as generics

__project__ = "NOMAD"
__author__ = "Ian Thomas & Roland Clairquin"
__contact__ = "roland.clairquin@oma.be"

#============================================================================================
# 3. CONVERT HDF5 LEVEL 0.1D TO LEVEL 0.1E
#
# DONE:
#
# VERTICALLY BIN Y DATA AND OTHER FIELDS IN LNO NADIR CASE
# APPLY LNO OFFSET CORRECTION
#
# MODIFY NSPEC ATTRIBUTE TO REFLECT NADIR BINNED DATA
#
#
# INTERPOLATE OVER BAD PIXELS
# ADD BAD PIXEL MAP TO FILE
# CORRECT NON-LINEARITY - NOT DONE NOW
#============================================================================================

logger = logging.getLogger(__name__)

VERSION = 80
OUTPUT_VERSION = "0.1E"

LNO_FLAG_FILE = os.path.join(PFM_AUXILIARY_FILES,"non_linearity_corrections","LNO_Detector_Offset_Correction_Flags_v01")
SO_FLAG_FILE = os.path.join(PFM_AUXILIARY_FILES,"non_linearity_corrections","SO_Detector_Offset_Correction_Flags_v02")

ARCMINS_TO_RADIANS = 57.29577951308232 * 60.0
SPICE_ABERRATION_CORRECTION = "None"
SPICE_STRING_FORMAT = "C"
SPICE_TIME_PRECISION = 3
SPICE_OBSERVER = "-143"


NA_VALUE = -999


SAVE_UNMODIFIED_DATASETS = True
#SAVE_UNMODIFIED_DATASETS = False

CORRECT_DETECTOR_OFFSET = True
#CORRECT_DETECTOR_OFFSET = False

#################################
# from pipeline_mappings_v04.py #
################################################################################################
DATASET_TO_BE_RESHAPED = set([
    "Time",
    "Date",
    "Timestamp",
    "Name",
    "Channel/BackgroundSubtraction",
    "Channel/IntegrationTime",
    "Channel/WindowTop",
    "Channel/NumberOfAccumulations",
    "Channel/WindowHeight",
    "Channel/Binning",
    "Channel/Exponent",
    "Channel/AOTFFrequency",
    "Channel/DiffractionOrder",
    "Channel/Pixel1",
    "Science/X",
    "Science/Y",
    "Science/XUnitFlag",
    "Science/YUnitFlag",
    "Science/YNb",
    "Science/YTypeFlag",
    "Geometry/ObservationDateTime",
    ])

################################################################################################

"""function to read calibration parameters from txt file"""
def readFlagFile(flag_file_name):
    flagFile = "%s.txt" %flag_file_name
#    logger.info("Opening detector correction calibration file %s for reading", flagFile)

    filesFound = np.zeros(14)
    with open(flagFile) as f:
        for index,line in enumerate(f):
            content = line.strip('\n')
            if len(content) > 0:
                if content[0] !="#":
                    if "REGION_TO_SUBTRACT=" in content:
                        region_to_subtract = content.split("=")[1].strip()
                        filesFound[0] = 1
                    elif "NUMBER_OF_PIXELS_TO_SUBTRACT=" in content:
                        number_of_pixels_to_subtract = int(content.split("=")[1].strip())
                        filesFound[1] = 1
                    elif "DETECTOR_CENTRE_START=" in content:
                        detector_centre_start = int(content.split("=")[1].strip())
                        filesFound[2] = 1
                    elif "DETECTOR_CENTRE_END=" in content:
                        detector_centre_end = int(content.split("=")[1].strip())
                        filesFound[3] = 1

                    elif "DETECTOR_OFFSET_FILE=" in content:
                        detector_offset_file_name = content.split("=")[1].strip()
                        filesFound[4] = 1
                    elif "STRAYLIGHT_MAX_DEVIATION=" in content:
                        straylight_max_deviation = float(content.split("=")[1].strip())
                        filesFound[5] = 1

                    elif "STRAYLIGHT_X_MAX_VECTOR=" in content:
                        straylight_x_max_vector = float(content.split("=")[1].strip())
                        filesFound[6] = 1
                    elif "STRAYLIGHT_X_MIN_VECTOR=" in content:
                        straylight_x_min_vector = float(content.split("=")[1].strip())
                        filesFound[7] = 1
                    elif "STRAYLIGHT_Y_MAX_VECTOR=" in content:
                        straylight_y_max_vector = float(content.split("=")[1].strip())
                        filesFound[8] = 1
                    elif "STRAYLIGHT_Y_MIN_VECTOR=" in content:
                        straylight_y_min_vector = float(content.split("=")[1].strip())
                        filesFound[9] = 1
                    elif "STRAYLIGHT_Z_MAX_VECTOR=" in content:
                        straylight_z_max_vector = float(content.split("=")[1].strip())
                        filesFound[10] = 1
                    elif "STRAYLIGHT_Z_MIN_VECTOR=" in content:
                        straylight_z_min_vector = float(content.split("=")[1].strip())
                        filesFound[11] = 1

                    elif "BAD_PIXEL_MAP=" in content:
                        bad_pixel_map_file_name = content.split("=")[1].strip()
                        filesFound[12] = 1
                    elif "NON_LINEARITY_FILE" in content:
                        non_linearity_file_name = content.split("=")[1].strip()
                        filesFound[13] = 1

    #check if all flags found in file
    for index, fileFound in enumerate(filesFound):
        if fileFound == 0:
#            print("Error: %s is missing flag %i" %(flag_file_name, index))
            logger.error("Error: %s is missing flag %i", flag_file_name, index)


    return region_to_subtract,number_of_pixels_to_subtract,detector_centre_start,detector_centre_end,detector_offset_file_name,\
            straylight_max_deviation,[straylight_x_min_vector,straylight_x_max_vector,straylight_y_min_vector,\
            straylight_y_max_vector,straylight_z_min_vector,straylight_z_max_vector],bad_pixel_map_file_name,non_linearity_file_name

def getBadPixelMap(channel_in,bad_pixel_map_name):
#    logger.info("Opening bad pixel file %s.h5 for reading", bad_pixel_map_name)
    calibration_file = h5py.File(os.path.join(PFM_AUXILIARY_FILES,"bad_pixel","%s.h5" % bad_pixel_map_name), "r")
    bad_pixel_map = calibration_file["Bad_Pixel_Map"][...]

    calibration_file.close()
    return bad_pixel_map


def getNonLinearityFile(non_linearity_file_name):
    if non_linearity_file_name != "None":
#        logger.info("Opening non-linearity file %s.h5 for reading", non_linearity_file_name)
        calibration_file = h5py.File(os.path.join(PFM_AUXILIARY_FILES,"non_linearity_corrections","%s.h5" % non_linearity_file_name), "r")
        non_linearity_file = calibration_file["Non_Linearity_Lookup_Table"][...]
        calibration_file.close()
        return non_linearity_file
    else:
        return None


def getDetectorOffsetFile(detector_offset_file_name):
#    logger.info("Opening detector offset file %s.h5 for reading", detector_offset_file_name)
    calibration_file = h5py.File(os.path.join(PFM_AUXILIARY_FILES,"non_linearity_corrections","%s.h5" % detector_offset_file_name), "r")
    detector_offset_file = calibration_file["Detector_Offsets/RatioOfSubtractionRegionToCentre"][...]

    calibration_file.close()
    return detector_offset_file


def reshapeDataset(dset_in, current_nbins): #this is not for the y dataset
    new_nbins = 1
#    logger.debug("Reshaping dataset: %s", dset_in.name)
#    print("Reshaping dataset: %s"% dset_in.name)
#    print("ndims=%i"%np.ndim(dset_in))

    if dset_in.dtype.char != "S": #if integer or float array
        if np.ndim(dset_in) == 1: #e.g. if diffraction order then do the mean not sum
            if new_nbins == 1:
                dset_out = np.mean(np.reshape(dset_in, (-1, current_nbins)), dtype=dset_in.dtype, axis=1)
            else:
                logger.error("Other new dimensions not yet implemented")
                return None
        elif np.ndim(dset_in) == 2:
#            print("2D dataset %s"%dset_in.name)
            if new_nbins == 1:
                first_dimension = dset_in.shape[0]
                second_dimension = dset_in.shape[1]
                dset_out = np.zeros((int(first_dimension/current_nbins), second_dimension),
                                    dtype=dset_in.dtype)
                row_iter = zip(range(0, first_dimension, current_nbins),
                               range(current_nbins, first_dimension+current_nbins, current_nbins))

                for rowIndex, (row_start, row_end) in enumerate(row_iter):
                    dset_out[rowIndex, :] = np.mean(dset_in[row_start:row_end, :], axis=0)
            else:
                logger.error("Other new dimensions not yet implemented")
                return None
        else:
            logger.error("Incorrect dimensions of dataset")
            return None
    else: #if string array
        if np.ndim(dset_in) == 1:
            dset_out = np.reshape(dset_in, (-1, current_nbins))[:, 0]
        elif np.ndim(dset_in) == 2:
            second_dimension = dset_in.shape[1]
            dset_out = np.reshape(dset_in, (-1, current_nbins*second_dimension))[:, [0, -1]]
        else: #can't have 3-d string array
            logger.error("Incorrect dimensions of dataset")
            return None
    return dset_out

def correctDetectorOffset(y_data, diffraction_orders, detector_offset_file_name, \
                          region_to_subtract, number_of_pixels_to_subtract, detector_centre_start, detector_centre_end, limb=False):
    """Correct spectrum offsets:
    Find mean of region of detector where signal is almost zero.
    Subtract this from whole spectrum.
    Add offset to correct for non-zero values in chosen region:
    Find mean of region where signal is high.
    Using known scaling factor (calculated elsewhere) between high signal region and low signal region
    Multiply whole spectrum by 1/(1-ratio) to re-add offset to whole spectrum"""

    detectorOffsetRatios = getDetectorOffsetFile(detector_offset_file_name)

    ratio_subtraction_region_to_centre = np.asfarray([detectorOffsetRatios[diffraction_order] for diffraction_order in diffraction_orders]) #do it per order in case of fullscan

    nPixels = y_data.shape[1]
    #yMeanRegion = mean of first or last region of detector, containing offsets for every spectrum.
    if region_to_subtract == "FIRST":
        yMeanRegion = np.mean(y_data[:, 0:number_of_pixels_to_subtract], axis=1)
#        logger.info("region_to_subtract=%s, number_of_pixels_to_subtract=%i", region_to_subtract, number_of_pixels_to_subtract)
    elif region_to_subtract == "LAST":
        yMeanRegion = np.mean(y_data[:, (-1*number_of_pixels_to_subtract):-1], axis=1)
#        logger.info("region_to_subtract=%s, number_of_pixels_to_subtract=%i", region_to_subtract, number_of_pixels_to_subtract)
    else:
        logger.error("Error: region_to_subtract %s is unknown", region_to_subtract)

    yMeanRegionArray = np.tile(yMeanRegion, (nPixels, 1)).T
    #yNew = new dataset after subtraction of offsets
    yNew = y_data - yMeanRegionArray
    #yMeanSpectrumOffset = mean values at centre of detector
#    logger.info("Detector centre pixels for offset correction are %i-%i", detector_centre_start, detector_centre_end)
#    logger.info("Lowest diffraction order=%0.1f, highest=%0.1f, lowest centre-subtraction region ratio=%0.3f, highest=%0.3f", \
#            np.min(diffraction_orders), np.max(diffraction_orders), \
#            np.min(ratio_subtraction_region_to_centre), np.max(ratio_subtraction_region_to_centre))

    if limb: #for limb measurements, there is no solar profile, therefore baseline should be corrected so that first/last region of detector is zero
        yMeanSpectrumOffset = 0.0
    else: #else, for solar reflectance, first 50 pixels are non-zero, so add a correction factor to scale whole curve up
        yMeanSpectrumOffset = np.mean(yNew[:, detector_centre_start:detector_centre_end], axis=1) * 1.0 / (1.0 / ratio_subtraction_region_to_centre - 1.0)
        yMeanSpectrumOffsetArray = np.tile(yMeanSpectrumOffset, (nPixels, 1)).T
        #add offset to shift whole spectrum back up to correct level
        yNew = yNew + yMeanSpectrumOffsetArray

#    plt.figure(figsize=(10,8))
#    plt.plot(y_data.T, alpha=0.3)
#    plt.xlabel("Pixel Number")
#    plt.ylabel("Detector Counts")
#    plt.title("Spectra Before Correction")
#    plt.figure(figsize=(10,8))
#    plt.plot(yNew.T, alpha=0.3)
#    plt.xlabel("Pixel Number")
#    plt.ylabel("Detector Counts")
#    plt.title("Spectra After Correction")

#    plt.figure(figsize=(10,8))
#    plt.plot(yMeanRegion, label="Mean of first pixels")
##    plt.plot(yMeanCentreRegion, label="Mean of centre pixels")
#    plt.legend()
#    plt.figure(figsize=(10,8))
#    plt.plot(np.mean(yNew[:,0:number_of_pixels_to_subtract], axis=1), label="Mean of first pixels - mean of first pixels")
#    plt.plot(np.mean(yNew[:,detector_centre_start:detector_centre_end], axis=1), label="Mean of centre pixels - mean of first pixels")
#    plt.legend()

#    logger.info("Mean offset subtracted=%0.2f, mean offset added=%0.2f", np.mean(yMeanRegion), np.mean(yMeanSpectrumOffset))
    return yNew, yMeanRegion, yMeanSpectrumOffset



def removeStraylight(y_data,bin_data,datetimes,straylight_max_deviation,straylight_vectors):
    """search through frames and find instances where spectra look strange
    and sun angle is within region where straylight can enter LNO channel.
    If found, set yvalidflag and write quality flag"""

    foundAlreadyInFile = False #flag to avoid repeated logger comments
    
    #find last bin indices
    last_bin_indices = np.where(bin_data[:, 0] == np.max(bin_data[:, 0]))[0]
    nBins = last_bin_indices[1] - last_bin_indices[0]
    nSpectraBinned = len(last_bin_indices)
    yValidFlagArray = np.ones(nSpectraBinned)
    y_data_out = np.copy(y_data)

#    plt.figure(figsize=(10,8))
#    logger.info("straylight_max_deviation=%0.1f", straylight_max_deviation)
    #loop through detector frames
    for frame_index,last_bin_index in enumerate(last_bin_indices):
        #assume last bin of frame is always free of straylight
        good_spectrum = y_data[last_bin_index]
        good_spectrum_std = np.std(good_spectrum)

        #loop through bins within frame except last one
        bins_indices_to_check = range(last_bin_index - (nBins - 1), last_bin_index)
        for bin_index_to_check in bins_indices_to_check:
            spectrum_to_check = y_data[bin_index_to_check]
            if np.std(np.abs(spectrum_to_check - good_spectrum)) > good_spectrum_std * straylight_max_deviation:
                straylightTime = datetimes[frame_index*nBins]
                #write to log at present. To be changed later
#                logger.warning("Possible straylight found at time %s", straylightTime)
#                logger.warning("%0.1f - %0.1f" %(np.std(np.abs(spectrum_to_check - good_spectrum)),good_spectrum_std))
                #determine angle between LNO_OCC boresight and sun direction
                obs2SunVector = sp.spkpos("SUN", sp.utc2et(straylightTime), "TGO_NOMAD_LNO_OPS_OCC", SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0]
                obs2SunUnitVector = obs2SunVector / sp.vnorm(obs2SunVector)
#                logger.warning(obs2SunUnitVector)
#                obs2SunAngle = sp.vsep(obs2SunUnitVector, np.asfarray([0.0, 0.0, 1.0]))
#                logger.warning("Angle between occultation boresight and sun = %0.2f degrees", (obs2SunAngle * sp.dpr()))

#                logger.info("straylight_vectors[Xmin,Xmax,Ymin,Ymax,Zmin,Zmax]=\
#                            [%0.1f,%0.1f,%0.1f,%0.1f,%0.1f,%0.1f]",\
#                    straylight_vectors[0],straylight_vectors[1],\
#                    straylight_vectors[2],straylight_vectors[3],\
#                    straylight_vectors[4],straylight_vectors[5])

                if straylight_vectors[0] < obs2SunUnitVector[0] < straylight_vectors[1] and \
                straylight_vectors[2] < obs2SunUnitVector[1] < straylight_vectors[3] and \
                straylight_vectors[4] < obs2SunUnitVector[2] < straylight_vectors[5]:

    #                plt.plot(spectrum_to_check)
                    if not foundAlreadyInFile: #flag to avoid repeated logger comments
                        logger.warning("Straylight removed at time %s", straylightTime)
                        foundAlreadyInFile = True
                    yValidFlagArray[frame_index] = 0
                    y_data_out[bins_indices_to_check,:] = np.nan
                    y_data_out[last_bin_index,:] = np.nan


                    #also remove spectrum before and after detected straylight (if not first or last spectrum)
                    if frame_index != 0:
                        bins_indices_to_remove = range(last_bin_index - 2 * nBins + 1, last_bin_index - nBins + 1)
                        y_data_out[bins_indices_to_remove, :] = np.nan
                        yValidFlagArray[frame_index - 1] = 0
                    if frame_index != nSpectraBinned and frame_index != nSpectraBinned - 1:
                        bins_indices_to_remove = range(last_bin_index + 1, last_bin_index + nBins + 1)
#                        logger.info("frame_index=%i, nSpectraBinned=%i, last_bin_index=%i, nBins=%i", frame_index, nSpectraBinned, last_bin_index, nBins)
                        logger.info(bins_indices_to_remove)
                        y_data_out[bins_indices_to_remove, :] = np.nan
                        yValidFlagArray[frame_index + 1] = 0


    return yValidFlagArray, y_data_out



def correctBadPixel(detector_data, bin_number, bad_pixel_list):
    """detect bad pixels in data and compare the bad pixel list. If anomalous reading detected, interpolate across adjacent values"""
    POLYNOMIAL_ORDER = 5
    N_STANDARD_DEVIATIONS = 1.5
    MODE = "Linear"
    nPixels = detector_data.shape[1]
    obsStandardDeviation = np.nanstd(detector_data, axis=0)
    polynomialValues = np.polyval(np.polyfit(range(nPixels), obsStandardDeviation, POLYNOMIAL_ORDER, full=True)[0],range(nPixels))
    obsPolynomialDifference = obsStandardDeviation - polynomialValues
    polynomialDifferenceStandardDeviation = np.std(obsPolynomialDifference)
    maxDeviation = polynomialDifferenceStandardDeviation * N_STANDARD_DEVIATIONS
    bad_pixels = list(np.where(obsPolynomialDifference > maxDeviation)[0])

#    logger.info("Potential bad pixels found: "+"%i "*len(bad_pixels) %tuple(bad_pixels))
    detector_data_new = np.copy(detector_data)
    bad_pixels_removed = []
    for bad_pixel in bad_pixels:
        #if pixel on edge of detector, ignore. if not, find adjacent non-bad pixels
        if bad_pixel + 1 != nPixels and bad_pixel - 1 >= 0 and bad_pixel in bad_pixel_list:
#            logger.info("Bad pixel %i corrected", bad_pixel)
#            logger.error("Bad pixel %i removed", bad_pixel)
            bad_pixels_removed.append(bad_pixel)
            pixel1ToInterpolate = bad_pixel - 1
            if pixel1ToInterpolate in bad_pixels:
                pixel1ToInterpolate -= 1
            pixel2ToInterpolate = bad_pixel + 1

            if pixel2ToInterpolate in bad_pixels:
                pixel2ToInterpolate += 1
            for rowIndex, detectorRow in enumerate(detector_data):
                if MODE == "Linear":
                    detector_data_new[rowIndex,bad_pixel] = np.polyval(np.polyfit([pixel1ToInterpolate, pixel2ToInterpolate], [detectorRow[pixel1ToInterpolate], detectorRow[pixel2ToInterpolate]], 1, full=True)[0], bad_pixel)
#                    logger.error("bin %i bad pixel %i from %0.1f to %0.1f", bin_number, bad_pixel, detector_data[rowIndex,bad_pixel], detector_data_new[rowIndex,bad_pixel])
                else:
                    logger.error("Error: MODE not defined")
#            logger.error(bad_pixel, detector_data[:,bad_pixel])
#            logger.error(bad_pixel, detector_data_new[:,badPixel])


    return detector_data_new, bad_pixels_removed

def correctBadPixels(y_data, bin_data, current_nbins, bad_pixel_map): #only for Y dataset
    """loop through all Y data, checking bad pixels against a bad pixel map"""
#    logger.info("Correcting Bad Pixels")
    bad_pixels_found=False

    y_data_corrected = np.zeros_like(y_data)
    bad_pixel_list_out = []
    #loop through bins, checking each individually
    for binNumber in range(current_nbins):
        bin_indices = np.where(bin_data[:, 0] == np.min(bin_data[:, 0]))[0] + binNumber
#        logger.error(bin_indices)
#        logger.info("%i to %i", bin_data[binNumber,0], bin_data[binNumber,1])
        #find matching list of bad pixels for the detector bin region
        bad_pixel_list = np.where(np.sum(bad_pixel_map[bin_data[binNumber,0]:bin_data[binNumber,1] + 1], axis=0) > 0)[0]
#        logger.info(bad_pixel_list)
        #pass detector values and list of potential bad pixels to function
        y_data_corrected[bin_indices, :], bad_pixels_removed = correctBadPixel(y_data[bin_indices, :], binNumber, bad_pixel_list)
        bad_pixel_list_out.append(bad_pixels_removed)

        if len(bad_pixels_removed) > 0:
            bad_pixels_found = True
    return y_data_corrected, bad_pixel_list_out, bad_pixels_found


def correctBadPixels2(y_data, bin_data, number_of_bins, channel): #only for Y dataset
    """version 2: from analysis of datasets, specify which pixels are bad"""

    y_data_corrected = np.zeros_like(y_data)
    bad_pixel_list_out = [[] for i in range(number_of_bins)]

    if channel == "so":
        for frameIndex, (y, binStartEnd) in enumerate(zip(y_data, bin_data)):
            binTop = binStartEnd[0]
            """mtp02 off pointing"""
            if binTop == 122:
                bad_pixels = [112]
                for bad_pixel in bad_pixels:
                    y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                    bad_pixel_list_out = [[112], [84, 200, 269], [124], [157]]
            if binTop == 126:
                bad_pixels = [84, 200, 269]
                for bad_pixel in bad_pixels:
                    y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                    bad_pixel_list_out = [[112], [84, 200, 269], [124], [157]]
            if binTop == 130:
                bad_pixels = [124]
                for bad_pixel in bad_pixels:
                    y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                    bad_pixel_list_out = [[112], [84, 200, 269], [124], [157]]
            if binTop == 134:
                bad_pixels = [157]
                for bad_pixel in bad_pixels:
                    y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                    bad_pixel_list_out = [[112], [84, 200, 269], [124], [157]]

            """mtp03+4 off pointing"""
            if binTop == 123:
                bad_pixels = [101]
                for bad_pixel in bad_pixels:
                    y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                    bad_pixel_list_out = [[101], [84, 124, 269], [], [152, 157]]
            if binTop == 127:
                bad_pixels = [84, 124, 269]
                for bad_pixel in bad_pixels:
                    y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                    bad_pixel_list_out = [[101], [84, 124, 269], [], [152, 157]]
            if binTop == 135:
                bad_pixels = [152, 157]
                for bad_pixel in bad_pixels:
                    y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                    bad_pixel_list_out = [[101], [84, 124, 269], [], [152, 157]]

            """nominal pointing"""
            if binTop == 120:
                bad_pixels = [256]
                for bad_pixel in bad_pixels:
                    y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                    bad_pixel_list_out = [[256], [], [84, 269, 124], []]
            if binTop == 128:
                bad_pixels = [84, 269, 124]
                for bad_pixel in bad_pixels:
                    y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                    bad_pixel_list_out = [[256], [], [84, 269, 124], []]

            y_data_corrected[frameIndex,:] = y


    if channel == "lno":
        if number_of_bins == 12:
            bad_pixel_list_out = [[82], [], [155], [64], [], [], [40], [], [142], [83, 306], [], [47, 76]]
            for frameIndex, (y, binStartEnd) in enumerate(zip(y_data, bin_data)):
                binTop = binStartEnd[0]
                if binTop == 80:
                    bad_pixels = [82]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))

                if binTop == 104:
                    bad_pixels = [155]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 116:
                    bad_pixels = [64]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))


                if binTop == 152:
                    bad_pixels = [40]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))

                if binTop == 176:
                    bad_pixels = [142]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 188:
                    bad_pixels = [83, 306]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))

                if binTop == 212:
                    bad_pixels = [47, 76]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))

                y_data_corrected[frameIndex,:] = y


        elif number_of_bins == 8: #3SUBD
            bad_pixel_list_out = [[82], [], [64], [], [40], [142], [83], [76]]
            for frameIndex, (y, binStartEnd) in enumerate(zip(y_data, bin_data)):
                binTop = binStartEnd[0]
                if binTop == 80:
                    bad_pixels = [82]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))

                if binTop == 116:
                    bad_pixels = [64]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))

                if binTop == 152:
                    bad_pixels = [40]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 170:
                    bad_pixels = [142]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 188:
                    bad_pixels = [83]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 206:
                    bad_pixels = [76]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))

                y_data_corrected[frameIndex,:] = y


        elif number_of_bins == 6: #4SUBD
            bad_pixel_list_out = [[82], [64], [], [40], [83], [47, 76]]
            for frameIndex, (y, binStartEnd) in enumerate(zip(y_data, bin_data)):
                binTop = binStartEnd[0]
                if binTop == 80:
                    bad_pixels = [82]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 104:
                    bad_pixels = [64]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))

                if binTop == 152:
                    bad_pixels = [40]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 176:
                    bad_pixels = [83]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 200:
                    bad_pixels = [47, 76]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))

                y_data_corrected[frameIndex,:] = y


        elif number_of_bins == 4: #6SUBD
            bad_pixel_list_out = [[82], [64], [40, 142], [47, 76]]
            for frameIndex, (y, binStartEnd) in enumerate(zip(y_data, bin_data)):
                binTop = binStartEnd[0]
                if binTop == 80:
                    bad_pixels = [82]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 116:
                    bad_pixels = [64]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 152:
                    bad_pixels = [40, 142]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                if binTop == 188:
                    bad_pixels = [47, 76]
                    for bad_pixel in bad_pixels:
                        y[bad_pixel] = np.mean((y[bad_pixel-1],y[bad_pixel+1]))
                y_data_corrected[frameIndex,:] = y


    bad_pixels_found = True
    return y_data_corrected, bad_pixel_list_out, bad_pixels_found



def correct_non_linearity(y_data, bin_data, current_nbins, number_of_accumulations, non_linearity_data):
#    logger.info("Correcting Non Linearity")
    HIGHEST_NON_LINEAR_COUNT_SINGLE_PIXEL = non_linearity_data.shape[0]
    N_SPECTRA = y_data.shape[0]
    N_PIXELS = y_data.shape[1]

    y_data_corrected = np.copy(y_data)

    def func(x, a, b):
        return a * np.exp(-b * x) + 1.0

    nFound = 0
    nNotFound = 0
    for spectrumIndex in range(N_SPECTRA): #loop through spectra in dataset
        binsInFrame = bin_data[spectrumIndex] #get data on which detector rows were measured
        firstBin = binsInFrame[0]
#        print(firstBin)
        lastBin = binsInFrame[1] + 1 #include last pixel
#        print(lastBin)
        nPixelsBinned = lastBin - firstBin
#        print(nPixelsBinned)
        nAccumulations = number_of_accumulations[spectrumIndex] #get number of accumulations for that bin
#        print(nAccumulations)

        for pixelIndex in range(N_PIXELS): #loop through pixels
            yValueUnbinned = y_data[spectrumIndex, pixelIndex] / nPixelsBinned / nAccumulations #calculate unbinned pixel value
            yValueUnbinnedInt = int(np.round(yValueUnbinned))
            if yValueUnbinned > 0 and yValueUnbinned < HIGHEST_NON_LINEAR_COUNT_SINGLE_PIXEL: #if value is positive and in non-linear range
                yValueUnbinnedNew = np.mean(non_linearity_data[yValueUnbinnedInt, firstBin:lastBin, pixelIndex]) #get new value from lookup table. taken mean of values in binned pixels
                yValueBinnedNew = yValueUnbinnedNew * nPixelsBinned * nAccumulations #multiply by number of pixels in bin
                y_data_corrected[spectrumIndex, pixelIndex] = yValueBinnedNew #save new value to corrected dataset
                nFound += 1
            else:
                nNotFound += 1
    logger.info("Non linearity: %i pixels corrected, %i not corrected", nFound, nNotFound)
#    logger.error("Non linearity: %i pixels corrected, %i not corrected", nFound, nNotFound)
    return y_data_corrected



def correct_non_linearity2(y_data, bin_data, current_nbins, number_of_accumulations, non_linearity_data):
    """new version of the non-linearity correction, using an exponential function derived from measurements (no lookup table)"""
#    logger.info("Correcting Non Linearity")
    N_SPECTRA = y_data.shape[0]
    N_PIXELS = y_data.shape[1]

    y_data_corrected = np.copy(y_data)

    def func(x):
        a=3.419764
        b=0.003491
        return a * np.exp(-b * x) + 1.0

    for spectrumIndex in range(N_SPECTRA): #loop through spectra in dataset
        binsInFrame = bin_data[spectrumIndex] #get data on which detector rows were measured
        firstBin = binsInFrame[0]
#        print(firstBin)
        lastBin = binsInFrame[1] + 1 #include last pixel
#        print(lastBin)
        nPixelsBinned = lastBin - firstBin
#        print(nPixelsBinned)
        nAccumulations = number_of_accumulations[spectrumIndex] #get number of accumulations for that bin
#        print(nAccumulations)

        for pixelIndex in range(N_PIXELS): #loop through pixels
            yValueUnbinned = y_data[spectrumIndex, pixelIndex] / nPixelsBinned / nAccumulations #calculate unbinned pixel value
            yValueUnbinnedNew = yValueUnbinned / func(yValueUnbinned)
            yValueBinnedNew = yValueUnbinnedNew * nPixelsBinned * nAccumulations
            y_data_corrected[spectrumIndex, pixelIndex] = yValueBinnedNew

#    logger.info("Non linearity corrected using exponential function")
#    logger.error("Non linearity: %i pixels corrected, %i not corrected", nFound, nNotFound)
    return y_data_corrected



def reshapeY(y_data, current_nbins):
    """sum counts for all bins in LNO nadir"""

    first_dimension = y_data.shape[0]
    second_dimension = y_data.shape[1]
    y_data_out = np.zeros((int(first_dimension/current_nbins), second_dimension),
                        dtype=y_data.dtype)
    row_iter = zip(range(0, first_dimension, current_nbins),
                   range(current_nbins, first_dimension+current_nbins, current_nbins))

    for rowIndex, (row_start, row_end) in enumerate(row_iter):
        y_data_out[rowIndex, :] = np.sum(y_data[row_start:row_end, :], axis=0)


    return y_data_out



def writeBadPixelQualityFlag(hdf5_file_out, bad_pixels_found):
    """write quality flags in file to reflect if bad pixels found/not found.
    At present only horizontal interpolation is applied"""

    #if bad pixels found, set quality flag to 1, else set equal to 0
    if bad_pixels_found:
#        logger.info("Bad pixels corrected. Writing relevant flags to file")
        hdf5_file_out.create_dataset("QualityFlag/BadPixelsHInterpolated", dtype=np.int,
                                data=1)
        hdf5_file_out.create_dataset("QualityFlag/BadPixelsVInterpolated", dtype=np.int,
                                data=0)
        hdf5_file_out.create_dataset("QualityFlag/BadPixelsMasked", dtype=np.int,
                                data=0)
    else:
        logger.info("Bad pixels uncorrected. Writing relevant flags to file")
        hdf5_file_out.create_dataset("QualityFlag/BadPixelsHInterpolated", dtype=np.int,
                                data=0)
        hdf5_file_out.create_dataset("QualityFlag/BadPixelsVInterpolated", dtype=np.int,
                                data=0)
        hdf5_file_out.create_dataset("QualityFlag/BadPixelsMasked", dtype=np.int,
                                data=0)

    return





def convert(hdf5file_path):
    logger.info("convert: %s", hdf5file_path)
#    logger.error("converting: %s", hdf5file_path)

    hdf5FileIn = h5py.File(hdf5file_path, "r")
    channel, channelType = generics.getChannelType(hdf5FileIn)
    observationType = generics.getObservationType(hdf5FileIn)
    hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
    hdf5FileOut = h5py.File(hdf5FilepathOut, "w")
    numberOfBins = hdf5FileIn.attrs["NBins"]
    bins = hdf5FileIn["Science/Bins"][...] #read in bin numbers
    numberOfAccumulations = hdf5FileIn["Channel/NumberOfAccumulations"][...] #read in number of accumulations for non-linearity correction
    generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION)

    #get flags from auxiliary file
    if channel == "so":
        flagFileName = SO_FLAG_FILE
    if channel == "lno":
        flagFileName = LNO_FLAG_FILE

    region_to_subtract, number_of_pixels_to_subtract, detector_centre_start, \
    detector_centre_end, detector_offset_file_name, straylight_max_deviation, \
    straylight_vectors, bad_pixel_map_file_name, non_linearity_file_name = readFlagFile(flagFileName) #get parameters from flag file


    #get bad pixel map from auxiliary files
    badPixelMap = getBadPixelMap(channel, bad_pixel_map_file_name)
    #add bad pixel map to file
    hdf5FileOut.create_dataset("Science/BadPixelMap", data=badPixelMap, compression="gzip", shuffle=True)



    #loop through datasets
    for dset_path, dset in generics.iter_datasets(hdf5FileIn):
        new_dset = None
        """if LNO nadir measurement, reshape datasets"""
        if channel == "lno" and observationType in ["D", "N"]:
            if dset_path in DATASET_TO_BE_RESHAPED:

                #special case: perform corrections to nadir Y dataset
                if dset_path == "Science/Y":
#                    logger.info("Checking LNO nadir Y dataset for straylight and detector voltage offsets")

                    if SAVE_UNMODIFIED_DATASETS:
                        #save old dataset to file
                        hdf5FileOut.create_dataset("Science/YUnmodified", data=dset[...],
                                                   compression="gzip", shuffle=True)


                    observationDTimeStarts = hdf5FileIn["Geometry/ObservationDateTime"][:,0]
                    #function to detect and remove straylight from data before reshaping
                    yValidFlag, intermediate_dset = removeStraylight(dset,bins,observationDTimeStarts,straylight_max_deviation,straylight_vectors)
                    #write new 1D array of 1 or 0 to show where straylight frames have been removed
                    hdf5FileOut.create_dataset("Science/YValidFlag", data=yValidFlag,
                                               compression="gzip", shuffle=True)

                    #if not straylight found, set quality flag to 0, else set equal to 1
                    if np.min(yValidFlag) == 1:
                        hdf5FileOut.create_dataset("QualityFlag/LNOStraylight", dtype=np.int,
                                                data=0)
                    else:
                        hdf5FileOut.create_dataset("QualityFlag/LNOStraylight", dtype=np.int,
                                                data=1)



                    #remove bad pixels ### correctBadPixels2(y_data, bin_data, number_of_bins, channel)
                    intermediate_2_dset, badPixelList, badPixelsFound = correctBadPixels2(intermediate_dset, bins, numberOfBins, channel)
                    #write bad pixels to attributes, one line per bin. Convert to array first
                    listMaxLength = max([len(list) for list in badPixelList]) #find length of longest list
                    badPixelArray = np.array([i + [0]*(listMaxLength-len(i)) for i in badPixelList]) #make array, filling in bad pixel values
                    hdf5FileOut.attrs["BadPixelsPerBin"] = badPixelArray

                    if SAVE_UNMODIFIED_DATASETS:
                        #save old dataset to file
                        hdf5FileOut.create_dataset("Science/YBadPixel", data=intermediate_2_dset,
                                                   compression="gzip", shuffle=True)


                    #reshape dataset - sum all bins vertically
                    intermediate_3_dset = reshapeY(intermediate_2_dset, numberOfBins)

                    #now correct LNO detector rows in reshaped dataset and write applied offsets to attributes (if required later)
                    diffractionOrders = reshapeDataset(hdf5FileIn["Channel/DiffractionOrder"], numberOfBins) #need to reshape diffraction orders first
                    
                    if CORRECT_DETECTOR_OFFSET:
                        new_dset, offsets_subtracted, offsets_added = correctDetectorOffset(intermediate_3_dset, diffractionOrders, \
                          detector_offset_file_name, region_to_subtract,number_of_pixels_to_subtract, \
                          detector_centre_start, detector_centre_end, limb=False)
                        hdf5FileOut.attrs["DetectorOffsetsSubtracted"] = offsets_subtracted
                        hdf5FileOut.attrs["DetectorOffsetsAdded"] = offsets_added
                    else:
                        new_dset = intermediate_3_dset

                    #write quality flags to file to reflect bad pixels
                    writeBadPixelQualityFlag(hdf5FileOut, badPixelsFound)


                else:
                    new_dset = reshapeDataset(dset, numberOfBins)


            elif dset_path == "Science/Bins": #special case for shaping Bins dataset
                second_dimension = dset.shape[1]
                new_dset = np.reshape(dset, (-1, numberOfBins*second_dimension))[:, [0, -1]]

        elif channel == "lno" and observationType in ["L"]:
            """special case: perform corrections to limb Y dataset"""
            if dset_path == "Science/Y":
#                logger.info("Checking LNO limb Y dataset for detector voltage offsets")

                if SAVE_UNMODIFIED_DATASETS:
                    #save old dataset to file
                    hdf5FileOut.create_dataset("Science/YUnmodified", data=dset[...],
                                               compression="gzip", shuffle=True)

#                region_to_subtract, number_of_pixels_to_subtract, detector_centre_start, \
#                detector_centre_end, detector_offset_file_name, \
#                straylight_max_deviation,straylight_vectors,bad_pixel_map_file_name,non_linearity_file_name = readFlagFile(channel)

                #remove bad pixels
#                intermediate_dset, badPixelList, badPixelsFound = correctBadPixels(dset[...], bins, numberOfBins, badPixelMap)
                intermediate_dset, badPixelList, badPixelsFound = correctBadPixels2(dset[...], bins, numberOfBins, channel)
                #write bad pixels to attributes, one line per bin. Convert to array first
                listMaxLength = max([len(list) for list in badPixelList]) #find length of longest list
                badPixelArray = np.array([i + [0]*(listMaxLength-len(i)) for i in badPixelList]) #make array, filling in bad pixel values
                hdf5FileOut.attrs["BadPixelsPerBin"] = badPixelArray

                #now correct LNO detector rows in reshaped dataset and write applied offsets to attributes (if required later)
                diffractionOrders = hdf5FileIn["Channel/DiffractionOrder"][...] #need to reshape diffraction orders first
                if CORRECT_DETECTOR_OFFSET:
                    new_dset, offsets_subtracted, offsets_added = correctDetectorOffset(intermediate_dset, diffractionOrders, \
                      detector_offset_file_name, region_to_subtract, number_of_pixels_to_subtract, \
                      detector_centre_start, detector_centre_end, limb=True) #if limb, don't add solar profile offset to data!
                    hdf5FileOut.attrs["DetectorOffsetsSubtracted"] = offsets_subtracted
                    hdf5FileOut.attrs["DetectorOffsetsAdded"] = offsets_added
                else:
                    new_dset = intermediate_3_dset

                #set all yValidFlag to 1 and write to file
                yValidFlag = np.ones(new_dset.shape[0])
                hdf5FileOut.create_dataset("Science/YValidFlag", data=yValidFlag,
                                           compression="gzip", shuffle=True)

                #write quality flags to file to reflect bad pixels
                writeBadPixelQualityFlag(hdf5FileOut, badPixelsFound)

                #set straylight quality flag to 0
                hdf5FileOut.create_dataset("QualityFlag/LNOStraylight", dtype=np.int,
                                                data=0)


        elif channel == "lno" and observationType != "C":
            """for all other LNO datasets except calibration - remove bad pixel only"""
            if dset_path == "Science/Y":


                if SAVE_UNMODIFIED_DATASETS:
                    #save old dataset to file
                    hdf5FileOut.create_dataset("Science/YUnmodified", data=dset[...],
                                               compression="gzip", shuffle=True)

                #remove bad pixels
#                new_dset, badPixelList, badPixelsFound = correctBadPixels(dset[...], bins, numberOfBins, badPixelMap)
                new_dset, badPixelList, badPixelsFound = correctBadPixels2(dset[...], bins, numberOfBins, channel)
                #write bad pixels to attributes, one line per bin. Convert to array first
                listMaxLength = max([len(list) for list in badPixelList]) #find length of longest list
                badPixelArray = np.array([i + [0]*(listMaxLength-len(i)) for i in badPixelList]) #make array, filling in bad pixel values
                hdf5FileOut.attrs["BadPixelsPerBin"] = badPixelArray

                #set all yValidFlag to 1 and write to file
                yValidFlag = np.ones(new_dset.shape[0])
                hdf5FileOut.create_dataset("Science/YValidFlag", data=yValidFlag,
                                           compression="gzip", shuffle=True)

                #write quality flags to file to reflect bad pixels
                writeBadPixelQualityFlag(hdf5FileOut, badPixelsFound)

                #set straylight quality flag to 0
                hdf5FileOut.create_dataset("QualityFlag/LNOStraylight", dtype=np.int,
                                                data=0)


        elif channel == "so" and observationType != "C":
            """for all other SO datasets except calibration - remove bad pixels and correct for non-linearity if flag is set"""
            if dset_path == "Science/Y":

                if SAVE_UNMODIFIED_DATASETS: #save old and new datasets to file?
                    #save original dataset to file
                    hdf5FileOut.create_dataset("Science/YUnmodified", data=dset[...],
                                               compression="gzip", shuffle=True)

                """non linearity correction"""
                backgroundSubtraction = hdf5FileIn["Channel/BackgroundSubtraction"][...]
                #assume first value is true for all values
                if backgroundSubtraction[0] == 1: #if bg subtracted, skip non linearity correction
#                    logger.info("SO measurement with background subtraction. Cannot correct non-linearity")
                    intermediate_dset = dset[...]

                    #TODO: estimate dark background values, add to light values, do non-linearity correction on all and subtract again
                else:
                    #save old dataset to file

                    #load non-linearity file. If ==None, then don't perform the correction
                    nonLinearityData = getNonLinearityFile(non_linearity_file_name)
                    if nonLinearityData != None: #if flag set not to None in auxiliary file, do non lin correction

                        #version 1: use lookup table
#                        old_correction = correct_non_linearity(dset[...], bins, numberOfBins, numberOfAccumulations, nonLinearityData)
                        #save old correction
#                        hdf5FileOut.create_dataset("Science/YNonLinearOld", data=old_correction, compression="gzip", shuffle=True)

                        #version 2: use function instead of lookup table
#                        logger.info("Correcting non-linearity in SO Y dataset, then removing bad pixels")
                        intermediate_dset = correct_non_linearity2(dset[...], bins, numberOfBins, numberOfAccumulations, nonLinearityData)
                        #save corrected data
                        hdf5FileOut.create_dataset("Science/YNonLinear", data=intermediate_dset,
                                                   compression="gzip", shuffle=True)
                    else:
                        intermediate_dset = dset[...] #not correcting non-linearity

                """bad pixel"""
#                #remove bad pixels
#                new_dset, badPixelList, badPixelsFound = correctBadPixels(intermediate_dset, bins, numberOfBins, badPixelMap)
                new_dset, badPixelList, badPixelsFound = correctBadPixels2(intermediate_dset, bins, numberOfBins, channel)
                #write bad pixels to attributes, one line per bin. Convert to array first
                listMaxLength = max([len(list) for list in badPixelList]) #find length of longest list
                badPixelArray = np.array([i + [0]*(listMaxLength-len(i)) for i in badPixelList]) #make array, filling in bad pixel values
                hdf5FileOut.attrs["BadPixelsPerBin"] = badPixelArray

                """write new datasets"""
                #set all yValidFlag to 1 and write to file
                yValidFlag = np.ones(new_dset.shape[0])
                hdf5FileOut.create_dataset("Science/YValidFlag", data=yValidFlag,
                                           compression="gzip", shuffle=True)

                #write quality flags to file to reflect bad pixels
                writeBadPixelQualityFlag(hdf5FileOut, badPixelsFound)

                #set straylight quality flag to 0
                hdf5FileOut.create_dataset("QualityFlag/LNOStraylight", dtype=np.int,
                                                data=0)

        """write existing datasets"""
        #make all groups
        dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
        #write new datasets to file (if exists)
        if new_dset is not None:
            hdf5FileOut.create_dataset(dset_path, data=new_dset,
                                       compression="gzip", shuffle=True)
        else:
            hdf5FileIn.copy(dset_path, dest)


    #add number of spectra to file
    hdf5FileOut.attrs["NSpec"] = hdf5FileOut["Science/Y"].shape[0]

    hdf5FileOut.close()
    hdf5FileIn.close()
    return [hdf5FilepathOut]
