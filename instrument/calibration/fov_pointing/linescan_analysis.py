# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:03:33 2020

@author: iant

Field of view simulations


"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import spiceypy as sp

from tools.file.paths import paths, FIG_X, FIG_Y
from tools.file.hdf5_functions import open_hdf5_file, get_files_from_datastore
from tools.spice.load_spice_kernels import load_spice_kernels
from tools.spice.datetime_functions import et2utc, utc2et

linescan_dict = {
        "so":{
            "MCO-1":["20161120_231420_0p1a_SO_1", "20161121_012420_0p1a_SO_1"],
            "MTP001":["20180428_023343_0p1a_SO_1", "20180511_084630_0p1a_SO_1"],
            "MTP005":["20180821_193241_0p1a_SO_1", "20180828_223824_0p1a_SO_1"],
            "MTP009":["20181219_091740_0p1a_SO_1", "20181225_025140_0p1a_SO_1"],
            "MTP010":["20190118_183336_0p1a_SO_1", "20190125_061434_0p1a_SO_1"],
            "MTP020":["20191022_013944_0p1a_SO_1", "20191028_003815_0p1a_SO_1"],
#            "MTP025":["", ""],
        },
         "lno":{
            "MCO-1":["20161121_000420_LNO", "20161121_021920_LNO"], \
            "MTP001":["201905", "20190704"],
#            "MTP001":["", ""],
#            "MTP001":["", ""],
        },
}

detector_centre_lines = {"so":128, "lno":152}
#boresights = {"so":[-0.92156, -0.38819, 0.00618]}
boresights = {"so":[-0.9218973, -0.38738526, 0.00616719]} #June 2018 onwards

channel = "so"
#title = "MCO-1"
#hdf5_filenames = linescan_dict[channel][title]


load_spice_kernels()



def find_cmatrix(ets_in,tolerance_in):
    """find list of c-matrices given ephemeris times and time errors"""

    obs="-143"
    ref="J2000"
    inst = int(obs) * 1000
    matrices = []
    if type(tolerance_in) == int or type(tolerance_in) == float:
        tolerance_in = "%s" %tolerance_in
    sctol = sp.sctiks(int(obs), tolerance_in)

    for et in ets_in:
        scticks = sp.sce2c(int(obs), et)
        [matrix, sc_time] = sp.ckgp(inst, scticks, sctol, ref)
        matrices.append(matrix)
    return matrices



def find_boresight(ets_in, tolerance_in, boresight_in):
    """return array of boresight pointing directions given ephemeris times, 
    time errors and TGO-to-channel boresight vector"""
    cmatrices = find_cmatrix(ets_in, tolerance_in)
    boresights = np.zeros((len(cmatrices), 3))
    for index, cmatrix in enumerate(cmatrices):
        boresights[index,:] = np.dot(np.transpose(cmatrix), boresight_in)
    return boresights



for title, hdf5_filenames in linescan_dict[channel].items():
    fig1, ax1 = plt.subplots(figsize=(FIG_X, FIG_Y))
    fig2, ax2 = plt.subplots(figsize=(FIG_X, FIG_Y))
    for hdf5_filename in hdf5_filenames:
        
        try:
            hdf5_file = open_hdf5_file(hdf5_filename)
        except OSError:
            get_files_from_datastore([hdf5_filename])
            hdf5_file = open_hdf5_file(hdf5_filename)
    
        
        
        
    
    
    ##    plot_both=False
    #    plot_both=True #flag to store values from orientation A so that results of both A and B can be plotted together.
    #    
    #    time_error=1
    #    if checkout=="NEC":
    #        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #define so boresight in tgo reference frame
    #    elif checkout=="MCC":
    #        if title=="SO Raster 1A" or title=="SO Raster 1B":
    #            boresight_to_tgo=(-0.92083,-0.38997,0.00042) #define so boresight in tgo reference frame
    #        elif title=="SO Raster 4A" or title=="SO Raster 4B": 
    #            boresight_to_tgo=(-0.92191,-0.38736,0.00608) #define so boresight in tgo reference frame
    #            if title=="SO Raster 4A":
    #                time_raster_centre=sp.utc2et("2016JUN15-23:15:00.000") #time of s/c pointing to centre
    #            elif title=="SO Raster 4B":
    #                time_raster_centre=sp.utc2et("2016JUN16-01:25:00.000") #time of s/c pointing to centre
    #            centre_theoretical=find_boresight([time_raster_centre],time_error,boresight_to_tgo)
    #            centre_theoretical_lat_lon=sp.reclat(centre_theoretical[0][0:3])[1:3]
    #        elif title=="LNO Raster 1A" or title=="LNO Raster 1B":
    #            boresight_to_tgo=(-0.92134,-0.38875,0.00076)
    #        elif title=="LNO Raster 4A" or title=="LNO Raster 4B":
    #            boresight_to_tgo=(-0.92163,-0.38800,0.00653)
    #        elif title=="SO-UVIS Raster 2A":
    #            boresight_to_tgo=(-0.92107,-0.38941,0.00093) #define so boresight in tgo reference frame
    #        elif title=="SO-UVIS Raster 3A": #team opposite uvis
    #            boresight_to_tgo=(-0.92207,-0.38696,0.00643) #define so boresight in tgo reference frame
    #    elif checkout=="MCO":
    #        if title=="SO Raster A" or title=="SO Raster B":
    #            boresight_to_tgo=(-0.92156, -0.38819, 0.00618) #define so boresight in tgo reference frame
    #        elif title=="LNO Raster 1" or title=="LNO Raster 2":
    #            boresight_to_tgo=(-0.92148, -0.38838, 0.00628) #define so boresight in tgo reference frame
        
        detector_centre_line = detector_centre_lines[channel]
        boresight = boresights[channel]
    
        detector_data_all = hdf5_file["Science/Y"][...]
        datetime_all = hdf5_file["Geometry/ObservationDateTime"][...]
        window_top_all = hdf5_file["Channel/WindowTop"][...]
        window_height = hdf5_file["Channel/WindowHeight"][0]+1
        binning = hdf5_file["Channel/Binning"][0]+1
    #    hdf5_file.close()
    
        if binning==2: #stretch array
            detector_data_all=np.repeat(detector_data_all,2,axis=1)
        if binning==4: #stretch array
            detector_data_all=np.repeat(detector_data_all,4,axis=1)
    
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
        ax1.plot(detector_line_mean)
        
        normalised_detector_line_mean = detector_line_mean / np.max(detector_line_mean)
    
        time_error=1    
        boresights_all=find_boresight(et_line, time_error, boresight)
        
        lon_lats=np.asfarray([sp.reclat(boresight)[1:3] for boresight in list(boresights_all)])
    
    
        ax2.scatter(lon_lats[:,0], lon_lats[:,1], c=normalised_detector_line_mean, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
    #    plt.scatter(lon_lats[:,0], lon_lats[:,1], c=marker_colour_b, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
    #        plt.scatter(centre_theoretical_lat_lon_a[0], centre_theoretical_lat_lon_a[1], c='r', marker='*', linewidth=0, s=120)
    #        plt.scatter(centre_theoretical_lat_lon_b[0], centre_theoretical_lat_lon_b[1], c='r', marker='*', linewidth=0, s=120)
        ax2.set_xlabel("Solar System Longitude (degrees)")
        ax2.set_ylabel("Solar System Latitude (degrees)")
        ax2.set_title(title)
        
    
#    #sum all detector data
#    detector_sum=np.sum(detector_data_all[:,:,:], axis=(1,2))
#    detector_sum[detector_sum<500000]=500000#np.mean(detector_sum)
    
    #find indices where centre of detector is
#    meas_indices=[]
#    detector_sum=[]
#    for index,window_top in enumerate(window_top_all):
#        if detector_centre_line in range(window_top,window_top+16*binning):
#            detector_line=detector_centre-window_top
#            meas_indices.append(index)
#            pixel_value=detector_data_all[index,detector_line,228]
#            if pixel_value<100:
#                pixel_value=100
#            detector_sum.append(pixel_value)
#    detector_sum=np.asfarray(detector_sum)
#    chosen_boresights=boresights_all[meas_indices,:]
#    lon_lats=np.asfarray([sp.reclat(chosen_boresight)[1:3] for chosen_boresight in list(chosen_boresights)])
    


    
#    marker_colour_a=np.log(1+detector_sum_a-min(detector_sum_a))
#    fig = plt.figure(figsize=(9,9))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(chosen_boresights_a[:,0], chosen_boresights_a[:,1], chosen_boresights_a[:,2], c=marker_colour_a, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
#    marker_colour_b=np.log(1+detector_sum_b-min(detector_sum_b))
#    ax.scatter(chosen_boresights_b[:,0], chosen_boresights_b[:,1], chosen_boresights_b[:,2], c=marker_colour_b, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
#    ax.azim=-108
#    ax.elev=-10
#    plt.gca().patch.set_facecolor('white')
#    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
#    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
#    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
##        plt.title(title_a+" & "+title_b+": Signal on pixel %i,228" %detector_centre)
#    plt.title(channel.upper()+" Solar Line Scan: Signal Measured on Detector Centre")
#
#    if save_figs: plt.savefig(title_a+"_"+title_b+"_Signal_on_pixel_%i,228_in_J2000.png" %detector_centre, dpi=600)
#        
#        
#        
#        
#        plt.figure(figsize=(10,8))
#        plt.scatter(lon_lats_a[:,0], lon_lats_a[:,1], c=marker_colour_a, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
#        plt.scatter(lon_lats_b[:,0], lon_lats_b[:,1], c=marker_colour_b, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
##        plt.scatter(centre_theoretical_lat_lon_a[0], centre_theoretical_lat_lon_a[1], c='r', marker='*', linewidth=0, s=120)
##        plt.scatter(centre_theoretical_lat_lon_b[0], centre_theoretical_lat_lon_b[1], c='r', marker='*', linewidth=0, s=120)
#        plt.xlabel("Solar System Longitude (degrees)")
#        plt.ylabel("Solar System Latitude (degrees)")
##        plt.title(title_a+" & "+title_b+": Signal on pixel %i,228" %detector_centre)
#        plt.title(channel.upper()+" Solar Line Scan: Signal Measured on Detector Centre")
#        cbar = plt.colorbar()
#        cbar.set_label("Log(Signal on Detector)", rotation=270, labelpad=20)
#        if save_figs: plt.savefig(title_a+"_"+title_b+"_Signal_on_pixel_%i,228_in_lat_lons.png" %detector_centre, dpi=600)
        
