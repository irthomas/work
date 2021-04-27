# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:05:20 2020

@author: iant
"""
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp



from tools.file.paths import FIG_X, FIG_Y
from tools.file.hdf5_functions import open_hdf5_file, get_files_from_datastore

from tools.spice.load_spice_kernels import load_spice_kernels
from tools.spice.datetime_functions import utc2et

load_spice_kernels()


"""user modifiable"""
MESHGRID = False
# MESHGRID = True

# channel = "so"
channel = "lno"
"""end"""



SPICE_ABERRATION_CORRECTION = "None"
SPICE_OBSERVER = "-143"

DETECTOR_CENTRE_LINES = {"so":128, "lno":152}




linescan_dict = {
        "so":{
#            "MCO-1":["20161120_231420_0p1a_SO_1", "20161121_012420_0p1a_SO_1"],
            "Old SO boresight (Apr 2018)":["20180428_023343_0p1a_SO_1", "20180511_084630_0p1a_SO_1"],
            "UVIS-prime (Aug 2018)":["20180821_193241_0p1a_SO_1", "20180828_223824_0p1a_SO_1"],
            "UVIS-prime (Dec 2018)":["20181219_091740_0p1a_SO_1", "20181225_025140_0p1a_SO_1"], #not a nomad linescan
            "UVIS-prime (Jan 2019)":["20190118_183336_0p1a_SO_1", "20190125_061434_0p1a_SO_1"],
            "SO-prime (Oct 2019)":["20191022_013944_0p1a_SO_1", "20191028_003815_0p1a_SO_1"],
            "SO-prime (Feb 2020)":["20200226_024225_0p1a_SO_1", "20200227_041530_0p1a_SO_1"],
            "SO-prime (Dec 2020)":["20201224_011635_0p1a_SO_1", "20210102_092937_0p1a_SO_1"],
        },
         "lno":{
            "SO-prime (June 2016)":["20161121_000420_0p1a_LNO_1", "20161121_021920_0p1a_LNO_1"], \
            # "MTP001":["201905", "20190704"],
            # "MTP015":["", ""],
            # "SO-prime (Jul 2020)":["20200724_125331_0p1a_LNO_1", "20200728_144718_0p1a_LNO_1"],
        },
}


#    referenceFrame = "TGO_NOMAD_UVIS_OCC"
referenceFrame = "TGO_NOMAD_SO"
# referenceFrame = "TGO_NOMAD_LNO_OPS_OCC"
# referenceFrame = "TGO_SPACECRAFT"


def get_vector(date_time, reference_frame):
    # print("SUN", date_time, reference_frame, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)
    obs2SunVector = sp.spkpos("SUN", date_time, reference_frame, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0]
    obs2SunUnitVector = obs2SunVector / sp.vnorm(obs2SunVector)
    return -1 * obs2SunUnitVector #-1 is there to switch the directions to be like in cosmographia


if channel == "so":
    fig1, axes = plt.subplots(nrows=2, ncols=4, figsize=(FIG_X+5, FIG_Y+2), sharex=True, )
    axes = axes.flatten()
else:
    fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_X, FIG_Y))


fig1.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("%s FRAME X" %referenceFrame)
plt.ylabel("%s FRAME Y" %referenceFrame)
for linescan_index, (title, hdf5_filenames) in enumerate(linescan_dict[channel].items()):

    xs = []
    ys = []
    zs = []
    
    if channel == "so":
        ax = axes[linescan_index]

    for scan_index, hdf5_filename in enumerate(hdf5_filenames):

        
        try:
            hdf5_file = open_hdf5_file(hdf5_filename)
        except OSError:
            get_files_from_datastore([hdf5_filename])
            hdf5_file = open_hdf5_file(hdf5_filename)

        detector_centre_line = DETECTOR_CENTRE_LINES[channel]


        detector_data_all = hdf5_file["Science/Y"][...]
        datetime_all = hdf5_file["Geometry/ObservationDateTime"][...]
        window_top_all = hdf5_file["Channel/WindowTop"][...]
        window_height = hdf5_file["Channel/WindowHeight"][0]+1
        binning = hdf5_file["Channel/Binning"][0]+1
        sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]

        if binning==2: #stretch array
            detector_data_all=np.repeat(detector_data_all,2,axis=1)
            detector_data_all /= 2
        if binning==4: #stretch array
            detector_data_all=np.repeat(detector_data_all,4,axis=1)
            detector_data_all /= 4
            
        if sbsf == 0:
            detector_data_all -= 50000.0
    
        #convert data to times and boresights using spice
        et_all=np.asfarray([np.mean([utc2et(i[0]), utc2et(i[1])]) for i in datetime_all])

        #find which window top contains the line
        unique_window_tops = list(set(window_top_all))
        for unique_window_top in unique_window_tops:
            if unique_window_top <= detector_centre_line <= (unique_window_top + window_height):
                centre_window_top = unique_window_top
                centre_row_index = detector_centre_line - unique_window_top
    
        window_top_indices = np.where(window_top_all == centre_window_top)[0]
        detector_data_line = detector_data_all[window_top_indices, centre_row_index, :]
        et_line = et_all[window_top_indices]
    
        detector_line_mean = np.mean(detector_data_line[:, 160:240], axis=1)
        detector_line_min = (np.max(detector_line_mean) + np.min(detector_line_mean)) * 0.65
        detector_line_max = np.max(detector_line_mean)
        detector_line_mean[detector_line_mean < detector_line_min] = detector_line_min
        detector_line_mean[detector_line_mean > detector_line_max] = detector_line_max

        
        print("%s: max value = %0.0f, min value = %0.0f" %(hdf5_filename, np.max(detector_line_mean), np.min(detector_line_mean)))
        
        unitVectors = np.asfarray([get_vector(datetime,referenceFrame) for datetime in et_line])
#        marker_colour = np.log(detector_line_mean)
        marker_colour = detector_line_mean
        
        if not MESHGRID:
            ax.scatter(unitVectors[:,0], unitVectors[:,1], c=marker_colour, alpha=1, cmap="gnuplot", linewidths=0)


        
        xs.extend(unitVectors[:,0])
        ys.extend(unitVectors[:,1])
        zs.extend(marker_colour)

    if not MESHGRID:
        circle1 = plt.Circle((0, 0), 0.0016, color='yellow', alpha=0.1)
        ax.add_artist(circle1)
    ax.set_xlim([-0.004,0.004])
    ax.set_ylim([-0.004,0.004])
    ax.set_aspect("equal")
    ax.set_title("%s: %s & %s" %(title, hdf5_filenames[0][:8], hdf5_filenames[1][:8]))
    ax.set_title("%s" %(title))
    ax.grid()

        
    if MESHGRID:
        
        """plot interpolation"""

        xs = np.asfarray(xs)
        ys = np.asfarray(ys)
        zs = np.asfarray(zs)
        
        # Create grid values first.
        ngridx = 100
        ngridy = 100
        xi = np.linspace(-0.004, 0.004, ngridx)
        yi = np.linspace(-0.004, 0.004, ngridy)
        
        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        import matplotlib.tri as tri
        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)
        
        ax.contourf(xi, yi, zi, levels=50, cmap="gnuplot")


fig1.tight_layout()
plt.savefig("%s_linescan_boresight.png" %channel, dpi=300)
#fig1.suptitle("SO Linescans")



