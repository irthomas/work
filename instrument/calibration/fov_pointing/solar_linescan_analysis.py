# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:22:05 2021

@author: iant

GET NEW BORESIGHT

WGC:
    http://spice.esac.esa.int/webgeocalc/#NewCalculation
    State Vector
    OPS
    Target: Sun
    Observer: -143
    Frame: TGO_NOMAD_SO
    Time system: TDB / seconds past J2000 / list of times
    Rectangular
    <Calculate>
    Download CSV
    
    

"""

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp

import numpy.linalg as la
from scipy.signal import savgol_filter


from tools.file.paths import FIG_X, FIG_Y
from tools.file.hdf5_functions import open_hdf5_file, get_files_from_datastore

from tools.spice.load_spice_kernels import load_spice_kernels
from tools.spice.datetime_functions import utc2et

from tools.spice.read_webgeocalc import read_webgeocalc

# load_spice_kernels()
load_spice_kernels()


"""user modifiable"""
#make meshgrid of Sun signal
MESHGRID = False
# MESHGRID = True

#use WebGeoCalc state vectors. Must be made first
WGC = True
# WGC = False

#Print out ets so that they can be put in to the WGC
# PRINT_WGC_ETS = True
PRINT_WGC_ETS = False

"""end"""

channel = "so"


SPICE_ABERRATION_CORRECTION = "None"
SPICE_OBSERVER = "-143"

DETECTOR_CENTRE_LINES = {"so":128, "lno":152}




linescan_dict = {
    # "SO-prime (Dec 2020)":["20201224_011635_0p1a_SO_1", "20210102_092937_0p1a_SO_1"],
    "MTP040":["20210430_095159_0p1a_SO_1", "20210502_011033_0p1a_SO_1"]
}


#    referenceFrame = "TGO_NOMAD_UVIS_OCC"
referenceFrame = "TGO_NOMAD_SO"
# referenceFrame = "TGO_NOMAD_LNO_OPS_OCC"
# referenceFrame = "TGO_SPACECRAFT"



def get_vector2(hdf5_filename):
    dt, obs2SunVector = read_webgeocalc(hdf5_filename, "spkpos")
    print(dt[0])
    obs2SunUnitVector =  obs2SunVector / np.tile(la.norm(obs2SunVector, axis=1), (3, 1)).T
    return -1 * obs2SunUnitVector #-1 is there to switch the directions to be like in cosmographia
    
    

def get_vector(date_time, reference_frame):
    # print("SUN", date_time, reference_frame, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)
    obs2SunVector = sp.spkpos("SUN", date_time, reference_frame, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0]
    obs2SunUnitVector = obs2SunVector / sp.vnorm(obs2SunVector)
    return -1 * obs2SunUnitVector #-1 is there to switch the directions to be like in cosmographia


    


for linescan_index, (title, hdf5_filenames) in enumerate(linescan_dict.items()):
    
    fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 9))
    fig1.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("%s FRAME X (slit long edge)" %referenceFrame)
    plt.xlabel("%s FRAME Y (slit short edge)" %referenceFrame)

    xs = []
    ys = []
    zs = []
    xmax = []
    xmax_y = []
    ymax = []
    ymax_x = []
    
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
        
        print(window_top_all[0], window_height, binning, sbsf)

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

        # #find which window top contains the line - this is not correct for binning
        # unique_window_tops = list(set(window_top_all))
        # for unique_window_top in unique_window_tops:
        #     if unique_window_top <= detector_centre_line <= (unique_window_top + window_height):
        #         centre_window_top = unique_window_top
        #         centre_row_index = detector_centre_line - unique_window_top
    
        # window_top_indices = np.where(window_top_all == centre_window_top)[0]
        # detector_data_line = detector_data_all[window_top_indices, centre_row_index, :]
        # et_line = et_all[window_top_indices]
        
        
        #if all binned like an occultation
        detector_data_line = np.mean(detector_data_all[:, 7:9, :], axis=1)
        et_line = et_all[:]
        
        if PRINT_WGC_ETS:
            with open("%s_ets.txt" %hdf5_filename, "w") as f:
                for et in et_line:
                    f.write("%0.3f\n" %et)
            continue
    
        detector_line_mean = np.mean(detector_data_line[:, 160:240], axis=1)
        # detector_line_min = (np.max(detector_line_mean) + np.min(detector_line_mean)) * 0.65
        # detector_line_max = np.max(detector_line_mean)
        # detector_line_mean[detector_line_mean < detector_line_min] = detector_line_min
        # detector_line_mean[detector_line_mean > detector_line_max] = detector_line_max

        
        print("%s: max value = %0.0f, min value = %0.0f" %(hdf5_filename, np.max(detector_line_mean), np.min(detector_line_mean)))
        
        if not WGC:
            unitVectors = np.asfarray([get_vector(datetime,referenceFrame) for datetime in et_line])
        
        else:
            unitVectors = get_vector2(hdf5_filename)
            print(et_line[0])
            
        


#        marker_colour = np.log(detector_line_mean)
        marker_colour = detector_line_mean
        
        if not MESHGRID:
            ax.scatter(unitVectors[:, 1], unitVectors[:, 0], c=marker_colour, alpha=1, cmap="gnuplot", linewidths=0)


        #determine scan direction - check abs(x gradient) vs abs(y gradient). Higher number is scan direction
        x_gradient = np.gradient(unitVectors[:, 0])
        y_gradient = np.gradient(unitVectors[:, 1])
        
    
        if np.mean(np.abs(x_gradient)) > np.mean(np.abs(y_gradient)):
            scan_direction = "x" #y direction 
            savgol = savgol_filter(x_gradient, 199, 1)
            grad_mean = [np.mean(np.abs(x_gradient)), np.mean(np.abs(x_gradient))*-1.]
        else:
            scan_direction = "y" #x direction
            savgol = savgol_filter(y_gradient, 199, 1)
            grad_mean = [np.mean(np.abs(y_gradient)), np.mean(np.abs(y_gradient))*-1.]
            
        # plt.figure()
        # plt.plot(x_gradient)
        # plt.axhline(x_grad_mean[0])
        # plt.axhline(x_grad_mean[1])
        
        # plt.plot(savgol)
        
        #get indices
        idx = [
            np.argwhere(np.diff(np.sign(savgol - grad_mean[0]))).flatten(),
            np.argwhere(np.diff(np.sign(savgol - grad_mean[1]))).flatten()
            ]
        # plt.plot(np.arange(len(x_gradient))[idx[0]], savgol[idx[0]], 'ro')
        # plt.plot(np.arange(len(x_gradient))[idx[1]], savgol[idx[1]], 'go')
        
        # if scan_direction == "x":
        for i, ix in enumerate(idx[0]):
            i_next = i+1
            j_next = np.searchsorted(idx[1], ix+1)
            if j_next < len(idx[1]) and i_next < len(idx[0]):
                if idx[1][j_next] > idx[0][i_next]: #if next upper bound comes before lower bound
                    line_idx = np.arange(ix, idx[0][i_next], 1)

                    max_signal_ix = np.where(marker_colour[line_idx] == max(marker_colour[line_idx]))[0][0] + line_idx[0]
                    # print(unitVectors[max_signal_ix, 0], unitVectors[max_signal_ix, 1])
                    # ax.plot(unitVectors[max_signal_ix, 0], unitVectors[max_signal_ix, 1], "go")

                    if scan_direction == "x":
                        xmax.append(unitVectors[max_signal_ix, 0])
                        xmax_y.append(unitVectors[max_signal_ix, 1])
                    elif scan_direction == "y":
                        ymax.append(unitVectors[max_signal_ix, 1])
                        ymax_x.append(unitVectors[max_signal_ix, 0])

        for j, jx in enumerate(idx[1]):
            j_next = j+1
            i_next = np.searchsorted(idx[0], jx+1)
            if j_next < len(idx[1]) and i_next < len(idx[0]):
                if idx[1][j_next] < idx[0][i_next]: #if next upper bound comes before lower bound
                    line_idx = np.arange(jx, idx[1][j_next], 1)
            
                    max_signal_ix = np.where(marker_colour[line_idx] == max(marker_colour[line_idx]))[0][0] + line_idx[0]
                    # print(unitVectors[max_signal_ix, 0], unitVectors[max_signal_ix, 1])
                    
                    if scan_direction == "x":
                        xmax.append(unitVectors[max_signal_ix, 0])
                        xmax_y.append(unitVectors[max_signal_ix, 1])
                    elif scan_direction == "y":
                        ymax.append(unitVectors[max_signal_ix, 1])
                        ymax_x.append(unitVectors[max_signal_ix, 0])


        #closest point
        pt_ix = np.where((unitVectors[:, 1] > 0.00005) & (unitVectors[:, 1] < 0.00006) & (unitVectors[:, 0] > -0.000322) & (unitVectors[:, 0] < -0.000321))[0]
        pt_ix_centre = np.where((unitVectors[:, 1] > -0.000075) & (unitVectors[:, 1] < -0.00005) & (unitVectors[:, 0] > -0.25e-5) & (unitVectors[:, 0] < 0.00000))[0]

        if len(pt_ix) > 0:
            ax.text(unitVectors[pt_ix[0], 1], unitVectors[pt_ix[0], 0], "Closest to best point: %s" %datetime_all[pt_ix[0], 0].decode())
            ax.scatter(unitVectors[pt_ix[0], 1], unitVectors[pt_ix[0], 0], color="k", marker="x")

        if len(pt_ix_centre) > 0:
            ax.text(unitVectors[pt_ix_centre[0], 1], unitVectors[pt_ix_centre[0], 0], "Closest to origin: %s" %datetime_all[pt_ix_centre[0], 0].decode())
            ax.scatter(unitVectors[pt_ix_centre[0], 1], unitVectors[pt_ix_centre[0], 0], color="k", marker="x")
        
        ax.text(unitVectors[0, 1], unitVectors[0, 0], "Start: %s" %datetime_all[0, 0].decode())

        
        xs.extend(unitVectors[:, 0])
        ys.extend(unitVectors[:, 1])
        zs.extend(marker_colour)
        plt.figure()
        plt.title(scan_direction)
        for line in [0, 4, 8, 12]:
            plt.plot(detector_data_all[:, line, 200], label=line)
        plt.legend()
        
        # stop()
        # from tools.plotting.anim import make_frame_anim
        # make_frame_anim(detector_data_all, 0, 70000, "anim_test", ymax=16)


    ax.plot(xmax_y, xmax, "go")
    ax.plot(ymax, ymax_x, "bo")
    ax.axhline(y=np.mean(xmax), color="g", linestyle="--")
    ax.axvline(x=np.mean(ymax), color="b", linestyle="--")
    
    
    ax.scatter(0.0, 0.0, color="k", marker="x")
    
    print("x=", np.mean(xmax), "(slit long edge)")
    print("y=", np.mean(ymax), "(slit short edge)")

    if not MESHGRID:
        circle1 = plt.Circle((np.mean(ymax), np.mean(xmax)), 0.0016, color='yellow', alpha=0.1)
        ax.add_artist(circle1)
    ax.set_xlim([-0.004, 0.004])
    ax.set_ylim([-0.004, 0.004])
    ax.set_aspect("equal")
    # ax.set_title("%s: %s & %s" %(title, hdf5_filenames[0][:8], hdf5_filenames[1][:8]))
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
        
        ax.contourf(yi, xi, zi, levels=50, cmap="gnuplot")



fig1.tight_layout()
# plt.savefig("%s_linescan_boresight.png" %channel, dpi=300)
#fig1.suptitle("SO Linescans")



