# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:27:56 2023

@author: iant
"""

import os
import h5py
import numpy as np
from datetime import datetime

from tools.file.paths import paths

ALTITUDE_FIELD = "TangentAltAreoid"
MAX_PERCENTAGE_ERROR = 2.0
TRANSMITTANCE_FIELD = "Y"
DETECTOR_DATA_FIELD = "YUnmodified"
DETECTOR_DATA_ERROR = "YError"


class h5_obj:

    """Convert the selected bin to transmittance and output a dictionary
    Inputs:
        h5f = an open level 1.0a hdf5 file
        h5 = filename (not path)
        bin_index = bin number i.e. 0, 1, 2 or 3
        silent = remove verbose message
        top_of_atmosphere = tangent altitude of Sun region
    Output:
        a dictionary containing selected fields
        y_mean = calculated transmittance
        x = wavenumbers
        alt = tangent altitude areoid
        label = a label for adding to a legend
        etc.
    """
    
    def __init__(self, h5, silent=True):
        self.h5 = h5
        self.silent = silent
        
        #output dictionary
        # self.h5_d = {}
        self.error = False
        self.h5_path = None
        
        #get info from filename
        self.obspath_split = self.h5.split("_")
        self.obs_type = self.obspath_split[5]

        #if fullscan or calibration
        self.calibration = False
        self.fullscan = False
        
        if self.obspath_split[5] == "S":
            self.fullscan = True
        elif "C" in self.obspath_split[5]:
            self.calibration = True
            
        if not self.fullscan and not self.calibration:
            self.diffraction_order = self.obspath_split[6]
        
    def set_h5_path(self, h5_path):
        
        self.h5_path = h5_path
        print("Using path %s" %h5_path)

        
    def open_h5f(self):
        
        if not self.h5_path:
            self.h5_path = paths["DATA_DIRECTORY"]
            
            
        year = self.h5[0:4] #get the date from the filename to find the file
        month = self.h5[4:6]
        day = self.h5[6:8]
        file_level = "hdf5_level_%s" %self.h5[16:20]
        self.h5_filepath = os.path.join(self.h5_path, file_level, year, month, day, self.h5+".h5")
        h5f = h5py.File(self.h5_filepath, "r")
        
        return h5f


    def read_h5(self, bin_ixs):

        if not self.silent:
            print("Reading file %s" %self.h5)
        
        h5f = self.open_h5f()
        
        h5_d = {}
        for bin_ix in bin_ixs:
        
            bins = h5f["Science/Bins"][:, 0]
            unique_bins = sorted(list(set(bins)))
            unique_bin_ixs = np.where(bins == unique_bins[bin_ix])[0]
            
            h5_d[bin_ix] = {}
        
            h5_d[bin_ix]["y"] = h5f["Science/%s" %TRANSMITTANCE_FIELD][unique_bin_ixs, :]
            
            if not self.calibration:
                h5_d[bin_ix]["y_raw"] = h5f["Science/%s" %DETECTOR_DATA_FIELD][unique_bin_ixs, :]
                h5_d[bin_ix]["y_error"] = h5f["Science/%s" %DETECTOR_DATA_ERROR][unique_bin_ixs, :]
                h5_d[bin_ix]["lon"] = h5f["Geometry/Point0/Lon"][unique_bin_ixs, 0]
                h5_d[bin_ix]["lat"] = h5f["Geometry/Point0/Lat"][unique_bin_ixs, 0]
                h5_d[bin_ix]["alt"] = h5f["Geometry/Point0/%s" %ALTITUDE_FIELD][unique_bin_ixs, 0]
            else:
                h5_d[bin_ix]["lon"] = np.zeros(h5_d[bin_ix]["y"].shape[0]) - 999.0
                h5_d[bin_ix]["lat"] = np.zeros(h5_d[bin_ix]["y"].shape[0]) - 999.0
                h5_d[bin_ix]["alt"] = np.zeros(h5_d[bin_ix]["y"].shape[0]) - 999.0

        if not self.calibration:
            h5_d["ls"] = h5f["Geometry/LSubS"][0, 0] #get first value
            h5_d["lst"] = h5f["Geometry/Point0/LST"][0, 0] #get first value
        else:
            h5_d["ls"] = -999.0
            h5_d["lst"] = -999.0

        h5_d["x"] = h5f["Science/X"][0, :] #get first row (all the same)

        h5_d["T"] = h5f["Channel/MeasurementTemperature"][0] #get first value

        h5_d["aotfs"] = h5f["Channel/AOTFFrequency"][...]
        h5_d["orders"] = h5f["Channel/DiffractionOrder"][...]
        
        
        h5_d["p0"] = h5f["Channel/FirstPixel"][0] #get first value
        h5_d["dt"] = datetime.strptime(self.h5[0:15], '%Y%m%d_%H%M%S')
        
        h5f.close()
        
        return h5_d


    def h5_to_dict(self, bin_ixs):
        """read data from h5 file into a dictionary"""
        
        self.h5_d = self.read_h5(bin_ixs)

    def cut_pixels(self, bin_ixs, px_ixs):
        """cut detector columns e.g. noisy first pixels"""
        
        for dset in ["y", "y_raw", "y_error"]:
            for bin_ix in bin_ixs:
                self.h5_d[bin_ix][dset] = self.h5_d[bin_ix][dset][:, px_ixs]
                
        for dset in ["x"]:
            self.h5_d[dset] = self.h5_d[dset][px_ixs]
        
        
    def trans_recal(self, bin_ixs, top_of_atmosphere):
        
        self.top_of_atmosphere = top_of_atmosphere


        for bin_ix in bin_ixs:
            #get indices for top of atmosphere
            top_ixs = np.where(self.h5_d[bin_ix]["alt"] > top_of_atmosphere)[0]
            if len(top_ixs) < 10:
                print("Error: Insufficient points %s above %i. n points = %i" %(self.h5, top_of_atmosphere, len(top_ixs)))
                self.error = True
                
            #l10a: data is sorted altitude ascending
            self.h5_d[bin_ix]["y_sun_mean"] = np.mean(self.h5_d[bin_ix]["y_raw"][top_ixs[:10], :], axis=0)
            self.h5_d[bin_ix]["y_mean"] = self.h5_d[bin_ix]["y_raw"] / self.h5_d[bin_ix]["y_sun_mean"]
        
        
    def test_trans_recal(self, bin_ixs, test_altitude):

        #check error
        for bin_ix in bin_ixs:
            
            y_sun = self.h5_d[bin_ix]["y_sun_mean"]
            y_raw = self.h5_d[bin_ix]["y_raw"]
            alt = self.h5_d[bin_ix]["alt"]
        
            test_ix = np.abs(alt - test_altitude).argmin()
            y_error = np.abs((y_sun[200] - y_raw[test_ix, 200]) / y_raw[test_ix, 200])
            if not self.silent: 
                print("%ikm error = %0.2f" %(test_altitude, y_error * 100) + r"%")

            if y_error * 100 > MAX_PERCENTAGE_ERROR:
                print("Warning: %s bin %i error too large. %ikm error = %0.1f" %(self.h5, bin_ix, test_altitude, y_error * 100))
                self.error = True
 
    def solar_cal(self, bins_to_average):
        
        for bin_ix in bins_to_average:
            self.h5_d[bin_ix]["y_mean"] = self.h5_d[bin_ix]["y"]
            
        self.yall = np.asarray([self.h5_d[bin_ix]["y_mean"] for bin_ix in bins_to_average])
        

            
    def make_label(self, bin_ix):
        
        alt = self.h5_d[bin_ix]["alt"]
        
        #find start/end lat/lons for label
        index_toa = np.abs(alt - self.top_of_atmosphere).argmin()
        index_0 = np.abs(alt - 0.0).argmin()
        lat_range = [self.h5_d[bin_ix]["lat"][index_toa], self.h5_d[bin_ix]["lat"][index_0]]        
        lon_range = [self.h5_d[bin_ix]["lon"][index_toa], self.h5_d[bin_ix]["lon"][index_0]]        
        self.label_full_out = self.h5[0:15]+"_"+self.obs_type+" lat=%0.0f to %0.0f, lon=%0.0f to %0.0f" %(lat_range[0],lat_range[1],lon_range[0],lon_range[1])
        self.label_out = self.h5[0:15]+"_"+self.obs_type+" order %s" %self.diffraction_order
        
      
    # def bad_pixel(self, bin_ix):
        
        # outputDict["alt"] = alt[ix_range[0]:ix_range[-1]]
        # outputDict["x"] = np.tile(wavenumbers, [len(binIndex), 1])[ix_range[0]:ix_range[-1], :]
        # outputDict["y_mean"] = detector_data_transmittance[ix_range[0]:ix_range[-1], :]
        # outputDict["y"] = transmittance[ix_range[0]:ix_range[-1], :]
        # outputDict["y_raw"] = detector_data[ix_range[0]:ix_range[-1], :]
        # outputDict["y_error"] = detector_error[ix_range[0]:ix_range[-1], :]
        # outputDict["order"] = int(diffraction_order)
        # outputDict["temperature"] = measurementTemperature
        # outputDict["first_pixel"] = firstPixel
        # outputDict["obs_datetime"] = obsDatetime
        # outputDict["label"] = label_out
        # outputDict["label_full"] = label_full_out
        # outputDict["bin_index"] = np.asarray([bin_index] * len(binIndex))[ix_range[0]:ix_range[-1]]
        # outputDict["ls"] = np.tile(ls, len(binIndex))[ix_range[0]:ix_range[-1]]
        # outputDict["lst"] = np.tile(lst, len(binIndex))[ix_range[0]:ix_range[-1]]
        # outputDict["longitude"] = lon[ix_range[0]:ix_range[-1]]
        # outputDict["latitude"] = lat[ix_range[0]:ix_range[-1]]

    # return outputDict


