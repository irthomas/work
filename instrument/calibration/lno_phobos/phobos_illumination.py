# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:23:48 2022

@author: iant


GET PHOBOS ILLUMINATION

IF OFF-POINTING: GET VECTOR TO PHOBOS AND CALCULATE EXPECTED FOV

"""

import re
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap


import itertools
import spiceypy as sp
from datetime import datetime

from tools.file.hdf5_functions import open_hdf5_file
from tools.datasets.get_phobos_viking_data import get_phobos_viking_data
from tools.plotting.colours import get_colours

from tools.spice.load_spice_kernels import load_spice_kernels
from tools.spice.make_offset_vector import make_offset_vector


from tools.spice.make_perimeter_bounds import make_perimeter_bounds
from tools.spice.rotation_matrix_from_vectors import rotation_matrix_from_vectors


bin_numbers = [1,2,3]


LNO_DETECTOR_CENTRE_LINE = 152


ARCMINS_TO_RADIANS = 57.29577951308232 * 60.0
KILOMETRES_TO_AU = 149597870.7
SP_DPR = sp.dpr()

NA_VALUE = np.nan#-999 #value to be used for NaN

# body-fixed, body-centered reference frame associated with the SPICE_TARGET body
SPICE_PLANET_REFERENCE_FRAME = "IAU_PHOBOS"
SPICE_ABERRATION_CORRECTION = "None"
SPICE_PLANET_ID = 401
# et2lst: form of longitude supplied by the variable lon
SPICE_LONGITUDE_FORM = "PLANETOCENTRIC"
# spkpos: reference frame relative to which the output position vector
# should be expressed
SPICE_REFERENCE_FRAME = "J2000"
#et2utc: string format flag describing the output time string. 'C' Calendar format, UTC
SPICE_STRING_FORMAT = "C"
# et2utc: number of decimal places of precision to which fractional seconds
# (for Calendar and Day-of-Year formats) or days (for Julian Date format) are to be computed
SPICE_TIME_PRECISION = 3

SPICE_TARGET = "PHOBOS"
SPICE_OBSERVER = "-143"

SPICE_SHAPE_MODEL_METHOD = "DSK/UNPRIORITIZED"
SPICE_INTERCEPT_METHOD = "INTERCEPT/DSK/UNPRIORITIZED"

# SPICE_SHAPE_MODEL_METHOD = "Ellipsoid"
# SPICE_INTERCEPT_METHOD = "INTERCEPT/ELLIPSOID"

DREF = "TGO_NOMAD_LNO_OPS_NAD"


load_spice_kernels()



def make_obs_dict(observationDatetimes):
    #make non-point dictionary
    d = {}
    d_tmp = {}
    
    d["et_s"] = np.asfarray([sp.utc2et(observationDatetime[0].decode()) for observationDatetime in observationDatetimes])
    # d["et_e"] = np.asfarray([sp.utc2et(observationDatetime[1].decode()) for observationDatetime in observationDatetimes])
    # d["et"] = np.vstack((d["et_s"], d["et_e"])).T
    
    
    
    
    d_tmp["obs_subpnt_s"] = [sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,et,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER) for et in d["et_s"]]
    # d_tmp["obs_subpnt_e"] = [sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,et,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,SPICE_OBSERVER) for et in d["et_e"]]
    
    d_tmp["obs_subpnt_xyz_s"] = [obs_subpnt[0] for obs_subpnt in d_tmp["obs_subpnt_s"]]
    d_tmp["obs_reclat_s"] = [sp.reclat(obs_subpnt_xyz) for obs_subpnt_xyz in d_tmp["obs_subpnt_xyz_s"]]
    d_tmp["obs_lon_s"] = [obs_reclat[1] for obs_reclat in d_tmp["obs_reclat_s"]]
    d_tmp["obs_lat_s"] = [obs_reclat[2] for obs_reclat in d_tmp["obs_reclat_s"]]
    d["obs_lon_s"] = np.asfarray(d_tmp["obs_lon_s"]) * SP_DPR
    d["obs_lat_s"] = np.asfarray(d_tmp["obs_lat_s"]) * SP_DPR
    d_tmp["sun_subpnt_s"] = [sp.subpnt(SPICE_INTERCEPT_METHOD,SPICE_TARGET,et,SPICE_PLANET_REFERENCE_FRAME,SPICE_ABERRATION_CORRECTION,"SUN") for et in d["et_s"]]
    d_tmp["sun_subpnt_xyz_s"] = [sun_subpnt[0] for sun_subpnt in d_tmp["sun_subpnt_s"]]
    d_tmp["sun_reclat_s"] = [sp.reclat(sun_subpnt_xyz) for sun_subpnt_xyz in d_tmp["sun_subpnt_xyz_s"]]
    d_tmp["sun_lon_s"] = [sun_reclat[1] for sun_reclat in d_tmp["sun_reclat_s"]]
    d_tmp["sun_lat_s"] = [sun_reclat[2] for sun_reclat in d_tmp["sun_reclat_s"]]
    d["sun_lon_s"] = np.asfarray(d_tmp["sun_lon_s"]) * SP_DPR
    d["sun_lat_s"] = np.asfarray(d_tmp["sun_lat_s"]) * SP_DPR
    

    return d
   




def make_obs_point_dict(ets, new_vectors):

    dp = {}
    dp_tmp = {}
    
    
    for point, fov in enumerate(new_vectors):
    # for point, fov in enumerate(boresightVectorbounds):
    
        dp[point] = {}
        dp_tmp[point] = {}
    
        dp_tmp[point]["sincpt_s"] = []
        dp_tmp[point]["surf_s"] = []
        dp_tmp[point]["ilumin_s"] = []
        dp[point]["ph_angle_s"] = []
        dp[point]["inc_angle_s"] = []
        dp[point]["em_angle_s"] = []
    
        dp[point]["lon_s"] = []
        dp[point]["lat_s"] = []
    
    
        for et in ets:
    
    
        
            try:
                sincpt = sp.sincpt(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, DREF, fov)[0]
            except sp.stypes.NotFoundError:
                sincpt = (np.zeros(3) + NA_VALUE, NA_VALUE, np.zeros(3) + NA_VALUE)[0]
    
    
    
            if not np.isnan(sincpt[1]):# != NA_VALUE:
                ilumin = sp.ilumin(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, sincpt)
                dp[point]["ph_angle_s"].append(ilumin[2] * SP_DPR)
                dp[point]["inc_angle_s"].append(ilumin[3] * SP_DPR)
                dp[point]["em_angle_s"].append(ilumin[4] * SP_DPR)
                
                
                reclat = sp.reclat(sincpt)
                dp[point]["lon_s"].append(reclat[1] * SP_DPR)
                dp[point]["lat_s"].append(reclat[2] * SP_DPR)
                
                
            else:
                dp[point]["ph_angle_s"].append(NA_VALUE)
                dp[point]["inc_angle_s"].append(NA_VALUE)
                dp[point]["em_angle_s"].append(NA_VALUE)
    
                dp[point]["lon_s"].append(NA_VALUE)
                dp[point]["lat_s"].append(NA_VALUE)
    
    
    return dp


if __name__ == "__main__":
    # h5 = "20220710_200313_0p3a_LNO_1_P_148"
    # h5 = "20220713_164911_0p3a_LNO_1_P_148"
    h5 = "20230225_220121_0p1a_LNO_1"
    # h5_f = open_hdf5_file(h5)
    
    h5_f = open_hdf5_file(h5, path=r"E:\DATA\hdf5_phobos")
    
    obs_dt_strs_all = h5_f["Geometry/ObservationDateTime"][...]
    
    if "0p1a" in h5:
        bins = np.ndarray.flatten(h5_f["Science/Bins"][:, :, 0]) #detector row of top of each bin
    else:
        bins = h5_f["Science/Bins"][:, 0] #detector row of top of each bin

    bin_height = h5_f["Channel/Binning"][0] + 1 #number of arcminutes per bin
    unique_bins = sorted(list(set(bins)))
    
    
    for bin_number in bin_numbers:
        # bin_number = 0
        bin_top = unique_bins[bin_number]
        
        
        detector_offset = np.float32(bin_top) - np.float32(LNO_DETECTOR_CENTRE_LINE) #offset in pixels from detector centre
        # detectorOffset[1] += 1 #2nd value is 1 pixel more in -ve direction (down detector)
        
        min_x = -2.
        max_x = 2.
        min_y = detector_offset
        max_y = detector_offset + bin_height
        
        perimeter_points = list(make_perimeter_bounds(min_x, min_y, max_x, max_y, delta=0.25))
        perimeter_points.append(perimeter_points[0]) #repeat first point to make polygon
        offset_vectors = [make_offset_vector(v) for v in perimeter_points]
        
        
        
        bin_indices = np.where(bin_top == bins)
        # stop()
        
        if "0p1a" in h5:
            obs_dt_strs = obs_dt_strs_all[:]
        else:
            obs_dt_strs = obs_dt_strs_all[bin_indices]
        
        
        d = make_obs_dict(obs_dt_strs)
        colours = get_colours(len(d["et_s"]))
        et = d["et_s"][50] #choose a time within the observation
        
        
        channelId = sp.bods2c(DREF) #find channel id number
        [channelShape, name, boresightVector, nvectors, boresightVectorbounds] = sp.getfov(channelId, 4)
        # boresightVector #should be array([0., 0., 1.]),
        
        
        #update fov to point to centre of Phobos
        phobos_pos_lno = sp.spkpos(SPICE_TARGET, et, DREF, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0]
        
        phobos_vec_lno = phobos_pos_lno / sp.vnorm(phobos_pos_lno)
        
        
        rot_mats = [rotation_matrix_from_vectors(boresightVector, offset_vector) for offset_vector in offset_vectors]
        new_vectors = [np.dot(rot_mat, phobos_vec_lno) for rot_mat in rot_mats]
        
        
        
        dp = make_obs_point_dict(d["et_s"], new_vectors)
        
        
        
        #plot ground tracks on surface model
        fig1, ax1 = plt.subplots(figsize=(12, 7), constrained_layout=True)
        fig1.suptitle("Phobos surface height and groundtrack for %s bin %i\nRed = observation start; blue = observation end" %(h5, bin_number))
        ax1.grid()
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_xlim((-180, 180))
        ax1.set_ylim((-90, 90))
        
        
        for i in range(0, len(d["et_s"]), 10): #plot every 10 FOVs
        
            rect = [(dp[point]["lon_s"][i], dp[point]["lat_s"][i]) for point in range(len(dp.keys()))]
            ax1.add_patch(Polygon(rect, alpha=0.7, fill=False, closed=False, color=colours[i], linewidth=3))
        
        ax1.scatter(d["obs_lon_s"], d["obs_lat_s"], color=colours, marker="x", label="Sub-Observer")
        ax1.scatter(d["sun_lon_s"], d["sun_lat_s"], color=colours, label="Sub-Solar")
        
        
        
        phobos_surface = get_phobos_viking_data()
        im1 = ax1.imshow(phobos_surface, extent=(-180, 180, -90, 90), alpha=0.7, aspect=1.1, cmap="binary_r")
        cbar1 = fig1.colorbar(im1, orientation='vertical')
        cbar1.set_label("Surface Height (km)", rotation=270, labelpad=15)
        
        fig1.savefig("phobos_surface_height_%s_bin_%i.png" %(h5, bin_number))
        
        
        
        
        
        
        
        
        #calculate illumination on each lat/lon pair and shade map appropriately
        lons = np.arange(-180, 180, 1.)
        lats = np.arange(-90, 90, 1.)
        
        
        lonlats = np.array(list(itertools.product(lons / SP_DPR, lats / SP_DPR)))
        surf_xyzs = sp.latsrf(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, lonlats) #doesn't change with time
        
        
        #plot ground tracks on illumination model at start and end of observation
        for frame_index in [0, -1]:
            et = d["et_s"][frame_index]
        
            fig2, ax2 = plt.subplots(figsize=(12, 7), constrained_layout=True)
            fig2.suptitle("Phobos solar zenith angles for %s bin %i (%s)\nRed = observation start; blue = observation end" %(h5, bin_number, {0:"start", -1:"end"}[frame_index]))
            ax2.grid()
            ax2.set_xlabel("Longitude")
            ax2.set_ylabel("Latitude")
            ax2.set_xlim((-180, 180))
            ax2.set_ylim((-90, 90))
        
            phobos_illumination = np.zeros((len(lats), len(lons)))
        
            for lon_ix, lon in enumerate(lons):
                for lat_ix, lat in enumerate(lats):
                    index = lon_ix * len(lats) + lat_ix
                    ilumin = sp.ilumin(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, surf_xyzs[index])
                    phobos_illumination[len(lats)-1-lat_ix, lon_ix] = ilumin[3] * SP_DPR
        
        
            for i in range(0, len(d["et_s"]), 10):
            
                rect = [(dp[point]["lon_s"][i], dp[point]["lat_s"][i]) for point in range(len(dp.keys()))]
                ax2.add_patch(Polygon(rect, alpha=0.7, fill=False, closed=False, color=colours[i], linewidth=3))
            
            ax2.scatter(d["obs_lon_s"], d["obs_lat_s"], color=colours, marker="x", label="Sub-Observer")
            ax2.scatter(d["sun_lon_s"], d["sun_lat_s"], color=colours, label="Sub-Solar")
        
        
            cmap = LinearSegmentedColormap.from_list("", colors=["white", "black", "black"], N=256)
        
            im2 = ax2.imshow(phobos_illumination, extent=(-180, 180, -90, 90), alpha=1.0, aspect=1.1, cmap=cmap)
            cbar2 = fig2.colorbar(im2, orientation='vertical')
            cbar2.set_label("Solar Zenith Angle", rotation=270, labelpad=15)
            
            ax2.legend()
            fig2.savefig("phobos_sza_%s_bin_%i_%s.png" %(h5, bin_number, {0:"start", -1:"end"}[frame_index]))
        
    
        #save illumination conditions of footprint perimeters to file
        for i in range(0, len(d["et_s"])):
        
            line = "%s\t" %obs_dt_strs[i]
            lonlats = np.array([(dp[point]["lon_s"][i] / SP_DPR, dp[point]["lat_s"][i] / SP_DPR) for point in range(len(dp.keys()))])
            
            lonlats = lonlats[~np.isnan(lonlats)].reshape(-1, 2)
            
            if len(lonlats) != 0:
                surf_xyzs = sp.latsrf(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, lonlats) #doesn't change with time
                for surf_xyz in surf_xyzs:
                    ilumin = sp.ilumin(SPICE_SHAPE_MODEL_METHOD, SPICE_TARGET, et, SPICE_PLANET_REFERENCE_FRAME, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER, surf_xyz)
                    line += "%0.1f\t" %(ilumin[3] * SP_DPR)
                    
                with open("%s_bin_%i.txt" %(h5, bin_number), "a") as f:
                    f.write(line + "\n")