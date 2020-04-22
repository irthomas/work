# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:26:11 2017

@author: iant

MAKE PLOTS FOR OBSERVATION PLANNING DISCUSSION: FOR JAN 2019

OCCULTATION ANIMATION
OCCULTATION LAT LONS COLOUR CODED BY TIME

OCCULTATION DURATION

GRAZING OCCULTATIONS



NADIR GROUND TRACK REPEAT - MARKER COLOURS FOR TIME, SUBSOLAR LONGITUDE
NADIR WITH OCCULTATIONS PLOTTED


NIGHTSIDE NADIR DURATIONS

"""


import numpy as np
import numpy.linalg as la
#import itertools
import os

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.patches as patches
#from mpl_toolkits.basemap import Basemap
#import struct
#from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import ListedColormap


#from spice_functions_v01 import find_cmatrix,find_boresight,py_ang,find_rad_lon_lat
from pipeline_config_v03b import BASE_DIRECTORY,KERNEL_DIRECTORY,figx,figy
from pipeline_mappings_v03b import METAKERNEL_NAME
import spiceypy as sp

#option=1
#option=2
option=3

#save_figs=False
save_figs=True
save_files=False
#save_files=True
#plot_figs=False
plot_figs=True

abcorr="None"
tolerance="1"
method="Intercept: ellipsoid"
formatstr="C"
prec=3
shape="Ellipsoid"

os.chdir(KERNEL_DIRECTORY)
sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
print(sp.tkvrsn("toolkit"))
os.chdir(BASE_DIRECTORY)

LOG_PATH1 = os.path.normcase(BASE_DIRECTORY+os.sep+"occultation_log.txt")
LOG_PATH2 = os.path.normcase(BASE_DIRECTORY+os.sep+"nadir_log.txt")

"""function to append log file"""
def write_log(log_number,line_to_write):
    global LOG_PATH1,LOG_PATH2
    if log_number==1:
        log_file=open(LOG_PATH1, 'a')
        log_file.write(line_to_write+'\n')
        log_file.close()
    elif log_number==2:
        log_file=open(LOG_PATH2, 'a')
        log_file.write(line_to_write+'\n')
        log_file.close()
#    print(line_to_write



if option==1:
    sun="SUN"
    step=10 #time delta (s)
    precooling_steps = int(np.ceil(600.0 / np.float(step)))
    wait_steps = int(np.ceil(10.0 / np.float(step)))
    max_altitude = 250.0 #km
    max_nadir_time = 2400 #seconds
    max_nadir_incidence_angle = 70 #degrees
    
    number_of_loops = 1
    duration_seconds = 3600 * 24 * 10
    suffix = "_10days"
    
    if save_files:
#        write_log(1,"Time,Tangent Longitude,Tangent Latitude,Tangent Altitude")
        write_log(2,"Time,SubSat Longitude,SubSat Latitude,Surface Incidence Angle,Observation Type")

    
    vmin=0
    vmax=6
    cmap = ListedColormap(["gold","deepskyblue","blue","lawngreen","forestgreen","gray","black"])

#    utcstring_start="2018MAY01-00:00:00 UTC"; azim=5; elev=16; #very short occultations

#    utcstring_start="2018NOV30-01:30:00 UTC"; azim=-14; elev=21; #longer occultations

#    utcstring_start="2018NOV30-09:30:00 UTC"; azim=-19; elev=22 #normal and then long grazing occultations

#    utcstring_start="2018DEC01-01:15:00 UTC"; azim=-14; elev=23; #grazing then no occultations

#    utcstring_start="2018DEC02-01:00:00 UTC"; azim=-18; elev=21; #grazing then no occultations

    utcstring_start="2019JAN01-00:45:00 UTC"; azim=5; elev=16; #very short occultations


    for loop_number in range(number_of_loops):
        print("Loop %i" %(loop_number+1))
        
        utctime_start=sp.str2et(utcstring_start) + duration_seconds*loop_number
        utctime_end=utctime_start + duration_seconds
    
            
            
      
        
        total_seconds=utctime_end-utctime_start
        nsteps=int(np.floor(total_seconds/step))
        
        times=np.arange(nsteps) * step + utctime_start
    
        print("Calculating positions")
        ref="IAU_MARS"
        observer="MARS"
        target="-143"
        mars2tgo_pos=np.asfarray([sp.spkpos(target,time,ref,abcorr,observer)[0] for time in list(times)]) #get tgo pos in mars frame
    
        observer="-143"
        target="SUN"
        tgo2sun_pos=np.asfarray([sp.spkpos(target,time,ref,abcorr,observer)[0] for time in list(times)]) #get sun pos in mars frame
    
    #    tgo_dist = la.norm(mars2tgo_pos,axis=1) #get tgo distance
    #    pericentre_indices = list((np.diff(np.sign(np.diff(tgo_dist))) > 0).nonzero()[0] + 1)
    
        print("Calculating occulations")
        observer="-143"
        target="MARS"
        occultation_status = np.asfarray([sp.occult(target, shape, "IAU_MARS", sun, shape, "IAU_SUN", abcorr, observer, time) for time in list(times)]) #get occultation statuses
        
        mars_axes = sp.bodvrd("MARS", "RADII", 3)[1] #get mars axis values
        line_points = mars2tgo_pos
        
        #make pointing vectors between tgo and sun positions
        line_vectors = tgo2sun_pos #np.asfarray([[tgo_vec[0]-sun_vec[0],tgo_vec[1]-sun_vec[1],tgo_vec[2]-sun_vec[2]] for tgo_vec,sun_vec in zip(list(tgo_pos),list(sun_pos))])
        
        #calculate surface tangent point using pointing vectors and tgo position vs time
        print("Calculating tangent points")
        tangent_coords = np.asfarray([sp.npedln(mars_axes[0], mars_axes[1], mars_axes[2], line_point, line_vector)[0] for line_point,line_vector in zip(list(line_points),list(line_vectors))])
        tangent_lonlats1 = np.asfarray([sp.reclat(tangent_coord) for tangent_coord in list(tangent_coords)])
        tangent_lonlats = np.asfarray([[tangent_lonlat[1]*sp.dpr(),tangent_lonlat[2]*sp.dpr()] for tangent_lonlat in list(tangent_lonlats1)])
        tangent_lsts = [sp.et2lst(time,499,(tangent_lonlat[0] / sp.dpr()),"PLANETOCENTRIC")[3] for time,tangent_lonlat in zip(list(times),list(tangent_lonlats))]
        tangent_lsts_hours = np.asfarray([np.float(tangent_lst[0:2]) + np.float(tangent_lst[3:5])/60.0 + np.float(tangent_lst[6:8])/3600.0 for tangent_lst in tangent_lsts])
        
        nadir_lonlats = np.asfarray([sp.reclat(mars2tgo) for mars2tgo in list(mars2tgo_pos)])[:,1:3] * sp.dpr()
        nadir_lsts = [sp.et2lst(time,499,(nadir_lonlat[0] / sp.dpr()),"PLANETOCENTRIC")[3] for time,nadir_lonlat in zip(list(times),list(nadir_lonlats))]
        nadir_lsts_hours = np.asfarray([np.float(nadir_lst[0:2]) + np.float(nadir_lst[3:5])/60.0 + np.float(nadir_lst[6:8])/3600.0 for nadir_lst in nadir_lsts])
        nadir_incidence_angles = np.asfarray([sp.ilumin(shape,target,time,"IAU_MARS",abcorr,observer,mars2tgo)[3] * sp.dpr() for time,mars2tgo in zip(list(times),list(mars2tgo_pos))])
        
        
        print("Calculating altitudes")
        #calculate tangent point altitude
        altitudes = np.asfarray([sp.npedln(mars_axes[0], mars_axes[1], mars_axes[2], line_point, line_vector)[1] for line_point,line_vector in zip(list(line_points),list(line_vectors))])
    
        print("Checking if occultations real or not")
        #check that tgo is really behind mars as seen from sun: calculate angle between tgo-sun vector and tgo-mars centre vector
        sep_angles = np.asfarray([sp.vsep(sp.vsub(line_coord,  line_point), line_vector) for line_coord,line_point,line_vector in zip(list(tangent_coords),list(line_points),list(line_vectors))])
        #if angle greater than 90 then tgo is between mars and sun, not behind mars
        valid_occ_indices = np.asfarray([1 if sep_angle <= sp.halfpi() else 0 for sep_angle in list(sep_angles)])    
    
        print("Removing invalid points")
        #calculate all valid tangent altitudes
        tangent_altitudes = np.asfarray([altitude if valid_occ==1 else np.nan for altitude,valid_occ in zip(list(altitudes),list(valid_occ_indices))])
        #calculate all valid tangent points
        tangentpt_lonlats = np.asfarray([tangent_lonlat if valid_occ==1 else [np.nan,np.nan] for tangent_lonlat,valid_occ in zip(list(tangent_lonlats),list(valid_occ_indices))])
    
        print("Checking if occultations within atmosphere")
        #if valid altitudes are within max height of atmosphere
        valid_altitude_indices = np.asfarray([1 if (tangent_altitude <= max_altitude and tangent_altitude > 0.0) else 0 for tangent_altitude in list(tangent_altitudes)])
        
        #remove points where above atmospheric height
        print("Removing points outside atmosphere")
        valid_altitudes = np.asfarray([tangent_altitude if valid_altitude==1 else np.nan for tangent_altitude,valid_altitude in zip(list(tangent_altitudes),list(valid_altitude_indices))])
        valid_tangent_lonlats = np.asfarray([tangent_lonlat if valid_altitude==1 else [np.nan,np.nan] for tangent_lonlat,valid_altitude in zip(list(tangent_lonlats),list(valid_altitude_indices))])
        valid_occultation_times = np.asfarray([time if valid_altitude==1 else np.nan for time,valid_altitude in zip(list(times),list(valid_altitude_indices))])
    
        
    
        #find start and end valid times for each occultation
        print("Calculating occultation start/end indices")
        occ_start_indices = []
        occ_end_indices = []
        for index,(valid_time,valid_time_offset) in enumerate(zip(list(valid_occultation_times),list(valid_occultation_times)[1::]+[valid_occultation_times[0]])):
            if np.isnan(valid_time) and not np.isnan(valid_time_offset):
                occ_start_indices.append(index)
            if not np.isnan(valid_time) and np.isnan(valid_time_offset):
                occ_end_indices.append(index)
    
        """now calculate occultation precooling"""
        occ_pc_start_indices = []
        occ_pc_end_indices = []
        daynadir_start_indices = []    
        daynadir_end_indices = []    
        daynadir_pc_start_indices = []    
        daynadir_pc_end_indices = []    
        daynadir_ends=[]
        daynadir_starts=[]
        nightnadir_start_indices = []    
        nightnadir_end_indices = []    
        nightnadir_pc_start_indices = []    
        nightnadir_pc_end_indices = []    
        nightnadir_ends=[]
        nightnadir_starts=[]
        
        #checks to perform:
        #is time between occultations too short for precooling?
        new_occ_start_indices = [occ_start_indices[0]]+[occ_start_indices[index] for index in range(1,len(occ_start_indices[:-1:])) if (occ_start_indices[index]-occ_end_indices[index-1]) > (precooling_steps+wait_steps)]
        new_occ_end_indices = [occ_end_indices[index] for index in range(len(occ_start_indices[:-1:])) if (occ_start_indices[index+1]-occ_end_indices[index]) > (precooling_steps+wait_steps)]
        occ_start_indices = new_occ_start_indices
        occ_end_indices = new_occ_end_indices
        
        occ_starts = [times[index] for index in occ_start_indices]
        occ_ends = [times[index] for index in occ_end_indices]
    
        
    
               
            
        
        nomad_obs_type = np.zeros_like(times)
        for index,(occ_start_index,occ_end_index) in enumerate(zip(occ_start_indices,occ_end_indices)):
            """start from 2nd occultation"""
            if index==0:
                
                occ_pc_end_index = occ_start_index - 1
                occ_pc_end_indices.append(occ_pc_end_index)
                occ_pc_start_index = occ_pc_end_index - precooling_steps - wait_steps
                occ_pc_start_indices.append(occ_pc_start_index)
                nomad_obs_type[occ_pc_start_index:occ_pc_end_index]=1
                nomad_obs_type[occ_start_index:occ_end_index]=2
                
            if index>0:
                
                if occ_start_index - occ_end_indices[index-1] < precooling_steps:
                    print("Warning: Only %i steps between previous occultation and this occultation" %(occ_start_index - occ_end_indices[index-1]))
    
                occ_pc_end_index = occ_start_index - 1
                occ_pc_end_indices.append(occ_pc_end_index)
                occ_pc_start_index = occ_pc_end_index - precooling_steps - wait_steps
                occ_pc_start_indices.append(occ_pc_start_index)
                nomad_obs_type[occ_pc_start_index:occ_pc_end_index]=1
                nomad_obs_type[occ_start_index:occ_end_index]=2
    
                nadir_end_index = occ_pc_start_index - 1
                nadir_pc_start_index = occ_end_indices[index-1] + wait_steps #end of previous occ plus 1 plus 10 seconds wait time
                nadir_pc_end_index = nadir_pc_start_index + precooling_steps
                nadir_start_index = nadir_pc_end_index + 1
    
    
                """check if dayside or nightside"""
                nadir_mid_index = int(np.mean([nadir_start_index,nadir_end_index]))
                nadir_mid_lst = nadir_lsts_hours[nadir_mid_index]
                
                """check if dayside nadir is so long that night is also measured - change to measure only the dayside part"""
                if (times[nadir_end_index] - times[nadir_start_index]) > max_nadir_time:
                    valid_angles = nadir_incidence_angles[nadir_start_index:nadir_end_index]
                    new_nadir_start_index = min(np.where(valid_angles < max_nadir_incidence_angle)[0]) + nadir_start_index
                    new_nadir_end_index = max(np.where(valid_angles < max_nadir_incidence_angle)[0]) + nadir_start_index
    
                    nadir_start_index = new_nadir_start_index
                    nadir_end_index = new_nadir_end_index
                    
                    nadir_pc_end_index = nadir_start_index - 1
                    nadir_pc_start_index = nadir_pc_end_index - precooling_steps - wait_steps
                
                """if nadir observation is too short - not enough time for precooling"""
                if (nadir_end_index - nadir_start_index) > 1: 
                    if nadir_mid_lst < 6.0 or nadir_mid_lst > 18.0:
                        nightnadir_end_indices.append(nadir_end_index)
                        nightnadir_ends.append(times[nadir_end_index])
            
                        nightnadir_pc_start_indices.append(nadir_pc_start_index)
                        nightnadir_pc_end_indices.append(nadir_pc_end_index)
                        nightnadir_start_indices.append(nadir_start_index)
                        nightnadir_starts.append(times[nadir_start_index])
                        
                        nomad_obs_type[nadir_pc_start_index:nadir_pc_end_index]=5
                        nomad_obs_type[nadir_start_index:nadir_end_index]=6
            
                    else:
                        daynadir_end_indices.append(nadir_end_index)
                        daynadir_ends.append(times[nadir_end_index])
            
                        daynadir_pc_start_indices.append(nadir_pc_start_index)
                        daynadir_pc_end_indices.append(nadir_pc_end_index)
                        daynadir_start_indices.append(nadir_start_index)
                        daynadir_starts.append(times[nadir_start_index])
                        
                        nomad_obs_type[nadir_pc_start_index:nadir_pc_end_index]=3
                        nomad_obs_type[nadir_start_index:nadir_end_index]=4
                
                
        
        
    #    for index in range(3):
    #        print("Index=%i" %index
    #        print("nadir_pc_start %i" %nadir_pc_start_indices[index]
    #        print("nadir_pc_end %i" %nadir_pc_end_indices[index]
    #        print("nadir_start %i" %nadir_start_indices[index]
    #        print("nadir_end %i" %nadir_end_indices[index]
    #        print("occ_pc_start %i" %occ_pc_start_indices[index]
    #        print("occ_pc_end %i" %occ_pc_end_indices[index]
    #        print("occ_start %i" %occ_start_indices[index+1]
    #        print("occ_end %i" %occ_end_indices[index+1]
        
    
        """may need to shift list elements by one if start time occurs within an occultation!"""
        print("Occultations:")
        for occ_start,occ_end,occ_start_index,occ_end_index in zip(occ_starts,occ_ends,occ_start_indices,occ_end_indices):
            occ_length = occ_end - occ_start
            occ_min_altitude = np.min(valid_altitudes[occ_start_index+1:occ_end_index+1])
            occ_max_altitude = np.max(valid_altitudes[occ_start_index+1:occ_end_index+1])
            print("%s - %s (%0.1fs duration, %0.1f-%0.1fkm altitude)" %(sp.et2utc(occ_start, formatstr, prec), sp.et2utc(occ_end, formatstr, prec), occ_length, occ_min_altitude, occ_max_altitude))
    
        print("Day Nadirs:")
        for nadir_start,nadir_end,nadir_start_index,nadir_end_index in zip(daynadir_starts,daynadir_ends,daynadir_start_indices,daynadir_end_indices):
            nadir_length = nadir_end - nadir_start
            print("%s - %s (%0.1fs duration)" %(sp.et2utc(nadir_start, formatstr, prec), sp.et2utc(nadir_end, formatstr, prec), nadir_length))
        
        print("Night Nadirs:")
        for nadir_start,nadir_end,nadir_start_index,nadir_end_index in zip(nightnadir_starts,nightnadir_ends,nightnadir_start_indices,nightnadir_end_indices):
            nadir_length = nadir_end - nadir_start
            print("%s - %s (%0.1fs duration)" %(sp.et2utc(nadir_start, formatstr, prec), sp.et2utc(nadir_end, formatstr, prec), nadir_length))
        
        
        if plot_figs:
    
        #    plt.figure(figsize=(figx,figy))
        #    plt.scatter(times-np.min(times),occultation_status*100)
        #    plt.scatter(times-np.min(times),valid_occ_indices*300,marker="x",s=50,color="r")
        #    plt.plot(times-np.min(times),tangent_altitudes)
        #    plt.scatter(times-np.min(times),valid_altitudes,marker="x",s=50,color="g")
        #    plt.xlabel("Time")
        #    plt.ylabel("Tangent Altitude")
            
        
        
            chosen_obs_type = 2
            chosen_xy = valid_tangent_lonlats / sp.dpr()
            chosen_z = (times-np.min(times))/3600.0
            title = ("Solar occultations from %s" %(utcstring_start)).replace(":","-").replace(" UTC","")
            fig = plt.figure(figsize=(figx,figy))
            ax = fig.add_subplot(111, projection="mollweide")
            ax.grid(True)
            plot1 = ax.scatter(chosen_xy[np.where(nomad_obs_type[:]==chosen_obs_type),0], chosen_xy[np.where(nomad_obs_type[:]==chosen_obs_type),1], c=chosen_z[np.where(nomad_obs_type[:]==chosen_obs_type)], cmap=plt.cm.jet, marker='o', linewidth=0)
            cbar = fig.colorbar(plot1,fraction=0.046, pad=0.04)
            cbar.set_label("Time after %s (hours)" %utcstring_start, rotation=270, labelpad=20)
            fig.tight_layout()
            if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_time"+suffix+".png")
        
        #    plt.figure(figsize=(figx,figy))
        #    plt.title("Solar Occultation Tangent Location vs Time")
        #    plt.xlabel("Longitude (degrees)")
        #    plt.ylabel("Latitude (degrees)")
        #    plt.xlim(-180,180)
        #    plt.scatter(valid_tangent_lonlats[:,0],valid_tangent_lonlats[:,1], c=marker_colour, cmap=plt.cm.jet, marker='o', linewidth=0)
        #    cbar = plt.colorbar(ticks=range(0,int(np.round(total_seconds/3600.0)),int(np.round(total_seconds/3600.0/24.0))))
        #    cbar.set_label("Time after %s (hours)" %utcstring_start)
        
        
            chosen_obs_type = 2
            chosen_xy = valid_tangent_lonlats / sp.dpr()
            chosen_z = valid_altitudes
            title = ("Solar occultations from %s" %(utcstring_start)).replace(":","-").replace(" UTC","")
            fig = plt.figure(figsize=(figx,figy))
            ax = fig.add_subplot(111, projection="mollweide")
            ax.grid(True)
            plot1 = ax.scatter(chosen_xy[np.where(nomad_obs_type[:]==chosen_obs_type),0], chosen_xy[np.where(nomad_obs_type[:]==chosen_obs_type),1], c=chosen_z[np.where(nomad_obs_type[:]==chosen_obs_type)], cmap=plt.cm.jet, marker='o', linewidth=0)
            cbar = fig.colorbar(plot1,fraction=0.046, pad=0.04)
            cbar.set_label("Tangent Point Altitude (km) Min=%0.1f, Max=%0.1f" %(np.nanmin(valid_altitudes),np.nanmax(valid_altitudes)), rotation=270, labelpad=20)
            fig.tight_layout()
            if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_altitude"+suffix+".png")
        
        
            title = ("Observation type from %s" %(utcstring_start)).replace(":","-").replace(" UTC","")
            fig = plt.figure(figsize=(figx,figy))
            ax = fig.add_subplot(111, projection="mollweide")
            ax.grid(True)
            plot1 = ax.scatter(nadir_lonlats[:,0] / sp.dpr(), nadir_lonlats[:,1] / sp.dpr(), c=nomad_obs_type, cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax), marker='o', linewidth=0)
        #    cbar = fig.colorbar(plot1,fraction=0.046, pad=0.04)
        #    cbar.set_label("Tangent Point Altitude (km) Min=%0.1f, Max=%0.1f" %(np.nanmin(valid_altitudes),np.nanmax(valid_altitudes)), rotation=270, labelpad=20)
            fig.tight_layout()
            if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+suffix+".png")
        
            chosen_obs_type = 4
            chosen_xy = nadir_lonlats / sp.dpr()
            chosen_z = nadir_incidence_angles
            title = ("Dayside nadir incidence angles from %s" %(utcstring_start)).replace(":","-").replace(" UTC","")
            fig = plt.figure(figsize=(figx,figy))
            ax = fig.add_subplot(111, projection="mollweide")
            ax.grid(True)
            plot1 = ax.scatter(chosen_xy[np.where(nomad_obs_type[:]==chosen_obs_type),0], chosen_xy[np.where(nomad_obs_type[:]==chosen_obs_type),1], c=chosen_z[np.where(nomad_obs_type[:]==chosen_obs_type)], cmap=plt.cm.jet, marker='o', linewidth=0)
            cbar = fig.colorbar(plot1,fraction=0.046, pad=0.04)
            cbar.set_label("Solar Incidence Angle (degrees) Min=%0.1f, Max=%0.1f" %(np.nanmin(chosen_z[np.where(nomad_obs_type[:]==chosen_obs_type)]),np.nanmax(chosen_z[np.where(nomad_obs_type[:]==chosen_obs_type)])), rotation=270, labelpad=20)
            fig.tight_layout()
            if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+suffix+".png")
        
            chosen_obs_type = 2
            chosen_xy = valid_tangent_lonlats / sp.dpr()
            chosen_z = tangent_lsts_hours
            title = ("Solar occultation Tangent Point LST from %s" %(utcstring_start)).replace(":","-").replace(" UTC","")
            fig = plt.figure(figsize=(figx,figy))
            ax = fig.add_subplot(111, projection="mollweide")
            ax.grid(True)
            plot1 = ax.scatter(chosen_xy[np.where(nomad_obs_type[:]==chosen_obs_type),0], chosen_xy[np.where(nomad_obs_type[:]==chosen_obs_type),1], c=chosen_z[np.where(nomad_obs_type[:]==chosen_obs_type)], cmap=plt.cm.jet, marker='o', linewidth=0)
            cbar = fig.colorbar(plot1,fraction=0.046, pad=0.04)
            cbar.set_label("Local Solar Time (hours) Min=%0.1f, Max=%0.1f" %(np.nanmin(chosen_z[np.where(nomad_obs_type[:]==chosen_obs_type)]),np.nanmax(chosen_z[np.where(nomad_obs_type[:]==chosen_obs_type)])), rotation=270, labelpad=20)
            fig.tight_layout()
            if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+suffix+".png")
        
        
        
        #    plt.figure(figsize=(figx,figy))
        #    plt.title("Solar Occultation Tangent Location vs Altitude")
        #    plt.xlabel("Longitude (degrees)")
        #    plt.ylabel("Latitude (degrees)")
        #    plt.xlim(-180,180)
        #    plt.scatter(valid_tangent_lonlats[:,0],valid_tangent_lonlats[:,1], c=marker_colour, cmap=plt.cm.jet, marker='o', linewidth=0)
        #    cbar = plt.colorbar(ticks=range(0,int(max_altitude),int(max_altitude/10)))
        #    cbar.set_label("Tangent Point Altitude (km) Min=%0.1f, Max=%0.1f" %(np.nanmin(valid_altitudes),np.nanmax(valid_altitudes)))
           
           
        #    max_plot_altitude=25
        #    plt.figure(figsize=(figx,figy))
        #    marker_colour = np.clip(valid_altitudes,0,max_plot_altitude)
        #    plt.title("Solar Occultation Tangent Location vs Altitude")
        #    plt.xlabel("Longitude (degrees)")
        #    plt.ylabel("Latitude (degrees)")
        #    plt.scatter(valid_tangent_lonlats[:,0],valid_tangent_lonlats[:,1], c=marker_colour, cmap=plt.cm.jet, marker='o', linewidth=0)
        #    cbar = plt.colorbar(ticks=np.arange(np.nanmin(valid_altitudes),max_plot_altitude,(max_plot_altitude-np.nanmin(valid_altitudes))/10))
        #    cbar.set_label("Tangent Point Altitude (km) Min=%0.1f, Max=%0.1f" %(np.nanmin(valid_altitudes),np.nanmax(valid_altitudes)))
        #    
            
        #    ref="IAU_MARS"
            ref="MARSIAU"
            observer="MARS"
            target="-143"
            tgo_pos=np.transpose(np.asfarray([sp.spkpos(target,time_in,ref,abcorr,observer)[0] for time_in in list(times)]))
            target="SUN"
            sun_pos=np.transpose(np.asfarray([sp.spkpos(target,time_in,ref,abcorr,observer)[0] for time_in in list(times)]))
        
        
            segments = np.zeros((len(times),2,3))
            segments[0,0,0] = tgo_pos[0,0]
            segments[0,0,1] = tgo_pos[1,0]
            segments[0,0,2] = tgo_pos[2,0]
            for index,(x,y,z) in enumerate(zip(list(tgo_pos[0,:-1:]),list(tgo_pos[1,:-1:]),list(tgo_pos[2,:-1:]))):
                segments[index,1,0] = x
                segments[index,1,1] = y
                segments[index,1,2] = z
            
                segments[index+1,0,0] = x
                segments[index+1,0,1] = y
                segments[index+1,0,2] = z
            segments[-1,1,0] = tgo_pos[0,-1]
            segments[-1,1,1] = tgo_pos[1,-1]
            segments[-1,1,2] = tgo_pos[2,-1]
        
            fig = plt.figure(figsize=(9,9))
            ax = p3.Axes3D(fig)
        
            tgo_point=ax.plot(tgo_pos[0, 0:1], tgo_pos[1, 0:1], tgo_pos[2, 0:1],'g*')[0]
            sun_line=ax.plot(sun_pos[0, 0:1], sun_pos[1, 0:1], sun_pos[2, 0:1],'y*')[0]
            
            tgo_line = Line3DCollection(segments[:2,:,:], array=nomad_obs_type, cmap = cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
            ax.add_collection(tgo_line)
            
            
            sun_arrow=ax.plot([tgo_pos[0,0],sun_pos[0,0]],[tgo_pos[1,0],sun_pos[1,0]],[tgo_pos[2,0],sun_pos[2,0]],'y')[0]
        
            limit=5e3
            ax.set_xlim((-1*limit,limit))
            ax.set_ylim((-1*limit,limit))
            ax.set_zlim((-1*limit,limit))
        
            mars_radius=3390
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            
            x = mars_radius * np.outer(np.cos(u), np.sin(v))
            y = mars_radius * np.outer(np.sin(u), np.sin(v))
            z = mars_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(x, y, z, color="r")
            ax.azim=azim
            ax.elev=elev
        
            plot_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
            plot_text2 = ax.text2D(0.05, 0.90, "", transform=ax.transAxes)
            plot_text3 = ax.text2D(0.05, 0.85, "", transform=ax.transAxes)
            title = ("Orbit_track_from_%s" %(utcstring_start)).replace(":","-").replace(" UTC","")
        
            def update_lines(num, tgo_line, sun_line, sun_arrow, tgo_point):
                global tgo_pos,sun_pos,valid_altitude_indices,valid_altitudes,nomad_obs_type,nadir_lonlats,nadir_lsts
                global segments
                if np.mod(num,100)==0:
                    print(num)
                tgo_line.set_segments(segments[:num, :,:])
                sun_line.set_data(sun_pos[0:2, :num]); sun_line.set_3d_properties(sun_pos[2, :num]) #add next part (in z) to line
                tgo_point.set_data(tgo_pos[0:2, num]); tgo_point.set_3d_properties(tgo_pos[2, num]) #add next part (in z) to line
                if valid_altitude_indices[num]==1:
                    sun_arrow.set_data([tgo_pos[0,num],sun_pos[0,num]],[tgo_pos[1,num],sun_pos[1,num]]); sun_arrow.set_3d_properties([tgo_pos[2,num],sun_pos[2,num]])
                    plot_text2.set_text("Altitude=%ikm" %valid_altitudes[num])
                else:
                    sun_arrow.set_data([0,0],[0,0]); sun_arrow.set_3d_properties([0,0])
                    plot_text2.set_text("Altitude=0km")
                
                if nomad_obs_type[num]==0:
                    plot_text.set_text("No Observation")
                    plot_text3.set_text("")
                elif nomad_obs_type[num]==1:
                    plot_text.set_text("Occultation Precooling")
                    plot_text3.set_text("")
                elif nomad_obs_type[num]==2:
                    plot_text.set_text("Occultation Science")
                    plot_text3.set_text("")
                elif nomad_obs_type[num]==3:
                    plot_text.set_text("Dayside Nadir Precooling")
                    plot_text3.set_text("")
                elif nomad_obs_type[num]==4:
                    plot_text.set_text("Dayside Nadir Science (%0.1f,%0.1f) degrees" %(nadir_lonlats[num,0],nadir_lonlats[num,1]))
                    plot_text3.set_text("LST = %s, Incidence Angle = %0.1f" %(nadir_lsts[num],nadir_incidence_angles[num]))
                elif nomad_obs_type[num]==5:
                    plot_text.set_text("Nightside Nadir Precooling")
                    plot_text3.set_text("")
                elif nomad_obs_type[num]==6:
                    plot_text.set_text("Nightside Nadir Science (%0.1f,%0.1f) degrees" %(nadir_lonlats[num,0],nadir_lonlats[num,1]))
                    plot_text3.set_text("LST = %s, Incidence Angle = %0.1f" %(nadir_lsts[num],nadir_incidence_angles[num]))
                return 0
            
            line_ani = animation.FuncAnimation(fig, update_lines, len(sun_pos[0]), fargs=(tgo_line, sun_line, sun_arrow, tgo_point), interval=1, blit=False)
                                               
        #    if save_figs: line_ani.save(title+".mp4", fps=50, extra_args=['-vcodec', 'libx264'])
            plt.show()    
        
        
        
        
        
        if save_files:
            
#            for index,nomad_obs in enumerate(nomad_obs_type):
#                if nomad_obs == 2:
#                    if not np.isnan(valid_tangent_lonlats[index,0]):
#                        output_text = "%s,%s,%s,%s" %(sp.et2utc(times[index],formatstr,prec),valid_tangent_lonlats[index,0],valid_tangent_lonlats[index,1],valid_altitudes[index])
#                        write_log(1,output_text)
            for index,nomad_obs in enumerate(nomad_obs_type):
                if nomad_obs == 4 or nomad_obs == 3:
                    if nomad_obs == 3:
                        obs_text = "Precooling"
                    elif nomad_obs == 4:
                        obs_text = "Science"
                        
                    output_text = "%s,%s,%s,%s,%s" %(sp.et2utc(times[index],formatstr,prec),nadir_lonlats[index,0],nadir_lonlats[index,1],nadir_incidence_angles[index],obs_text)
                    write_log(2,output_text)
   

if option==2:

    occ_lines=[]
    f=open(LOG_PATH1,"r")
    occ_lines=f.readlines()
    del occ_lines[0]
    f.close()
    
    
    
#    np.loadtxt(LOG_PATH, skiprows=1, dtype={"names": ("lons","lats","alts"), "formats": ("S25","<9f","<f9","<f9")})#np.str,np.float,np.float,np.float])
    
    tangent_lons = np.asfarray([occ_line.split(",")[1] for occ_line in occ_lines])
    tangent_lats = np.asfarray([occ_line.split(",")[2] for occ_line in occ_lines])
    tangent_alts = np.asfarray([occ_line.split(",")[3] for occ_line in occ_lines])
    
    plt.figure(figsize=(figx,figy))
    plt.scatter(tangent_lons,tangent_lats,c=tangent_alts,linewidth=0)
    plt.tight_layout()
    cbar = plt.colorbar()
    cbar.set_label("Tangent Altitude km")
    plt.ylim([-13,4])
    plt.xlim([129,146])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    
    
    
    
if option==3:

    """plot sun size, and occultation and nadir boresights"""
    """TIRVIM SIZE IS NOT CORRECT"""
    
    ref="J2000"
    observer="-143" #observer
    target = "SUN"
    step=3600 #time delta (s)
    
    utcstring_start="01MAY2018 12:00:00.000"
    utcstring_end="03MAY2020 12:00:00.000"
    utctime_start=sp.str2et(utcstring_start)
    utctime_end=sp.str2et(utcstring_end)
    
    total_seconds=utctime_end-utctime_start
    nsteps=int(np.floor(total_seconds/step))
    times=np.arange(nsteps) * step + utctime_start
    time_strings=[sp.et2utc(time_in, formatstr, prec) for time_in in list(times)]
    

    tgo_pos=np.asfarray([sp.spkpos(target,time,ref,abcorr,observer)[0] for time in list(times)])
    tgo_dist = la.norm(tgo_pos,axis=1)
    code = sp.bodn2c(target)
    pradii = sp.bodvcd(code, 'RADII', 3) # 10 = Sun
    sun_radius = pradii[1][0]
    sun_diameter_arcmins = np.arctan(sun_radius/tgo_dist) * sp.dpr() * 60.0 * 2.0
    
    fig1 = plt.figure(figsize=(figx-6,figy-5))
    plt.plot(sun_diameter_arcmins)
    plt.xlabel("UTC time")
    plt.ylabel("Solar diameter as seen from TGO (arcminutes)")
    title="Apparent diameter of Sun over 1 Martian year"
    plt.title(title)
    x_tick_indices = range(0,len(times),24*183)
    x_tick_names = [time_strings[x_tick_index] for x_tick_index in x_tick_indices]
    plt.xticks(x_tick_indices,x_tick_names)

    if save_figs: plt.savefig(title.replace(" ","_").replace(":","").replace(",","")+".png")
    
    

    ref='TGO_NOMAD_SO' 
    observer='-143' #observer
    target = 'SUN'
    et = sp.utc2et("01MAY2018 12:00:00.000")
    [pos_tgo_sun,_] = sp.spkpos(target,et,ref,abcorr,observer)
    D = la.norm(pos_tgo_sun)
    code = sp.bodn2c(target)
    pradii = sp.bodvcd(code, 'RADII', 3) # 10 = Sun
    R = pradii[1][0]  

    room=4
    # retrieve the UVIS FOV parameters
    code = sp.bodn2c('TGO_NOMAD_UVIS_OCC')
    [shape_uvis, frame_uvis, bsight_uvis, nvectors_uvis, bounds_uvis] = sp.getfov(code, room)
    # retrieve the LNO FOV parameters
    code = sp.bodn2c('TGO_NOMAD_LNO_OPS_OCC')
    [shape_lno, frame_lno, bsight_lno, nvectors_lno, bounds_lno] = sp.getfov(code, room)
    # retrieve the SO FOV parameters
    code = sp.bodn2c('TGO_NOMAD_SO')
    [shape_so, frame_so, bsight_so, nvectors_so, bounds_so] = sp.getfov(code, room)
    #retrive NIR FOV parameters
    code = sp.bodn2c('TGO_ACS_NIR_OCC')
    [shape_nir, frame_nir, bsight_nir, nvectors_nir, bounds_nir] = sp.getfov(code, room)
    #retrieve MIR FOV parameters
    code =sp.bodn2c('TGO_ACS_MIR')
    [shape_mir, frame_mir, bsight_mir, nvectors_mir, bounds_mir] = sp.getfov(code, room)
    #retrieve TIRVIM FOV parameters
    code = sp.bodn2c('TGO_ACS_TIRVIM_OCC')
    [shape_tirvim, frame_tirvim, bsight_tirvim, nvectors_tirvim, bounds_tirvim] = sp.getfov(code, room)

    
    # Check the boresights in the s/c frame %%%%%%%%%%%%%%%%%%%%%%%%
    uvis2ref = sp.pxform('TGO_NOMAD_UVIS_OCC', ref, et) # matrix to convert from UVIS frame to ref frame
    lno2ref = sp.pxform('TGO_NOMAD_LNO_OPS_OCC', ref, et) # matrix to convert from LNO frame to ref frame
    so2ref = sp.pxform('TGO_NOMAD_SO', ref, et) # matrix to convert from SO frame to ref frame
    nir2ref = sp.pxform('TGO_ACS_NIR_OCC', ref, et)
    mir2ref = sp.pxform('TGO_ACS_MIR', ref, et)
    tirvim2ref = sp.pxform('TGO_ACS_TIRVIM_OCC', ref, et)
    

    bsight_uvis2ref = np.dot(uvis2ref,bsight_uvis)
    bsight_lno2ref = np.dot(lno2ref,bsight_lno)
    bsight_so2ref = np.dot(so2ref,bsight_so)
    bsight_nir2ref = np.dot(nir2ref,bsight_nir)
    bsight_mir2ref = np.dot(mir2ref,bsight_mir)
    bsight_tirvim2ref = np.dot(tirvim2ref,bsight_tirvim)

    bounds_lno2ref = np.zeros((room,3))
    bounds_so2ref=np.zeros((room,3))
    bounds_nir2ref=np.zeros((room,3))
    bounds_mir2ref=np.zeros((room,3))
    #no uvis or tirvim required (circular)
    
    bounds_uvis2ref = np.dot(uvis2ref,bounds_uvis[0])
    bounds_tirvim2ref = np.dot(tirvim2ref,bounds_tirvim[0])
    
    for index in range(room):
        bounds_lno2ref[index,:] = np.dot(lno2ref,bounds_lno[index,:])  # Bounds of the lno fov in ref frame
        bounds_so2ref[index,:] = np.dot(so2ref,bounds_so[index,:])  # Bounds of the lno fov in ref frame
        bounds_nir2ref[index,:] = np.dot(nir2ref,bounds_nir[index,:])  # Bounds of the lno fov in ref frame
        bounds_mir2ref[index,:] = np.dot(mir2ref,bounds_mir[index,:])  # Bounds of the lno fov in ref frame
#        plt.scatter(bounds_lno2ref[index,0],bounds_lno2ref[index,1])
#        plt.scatter(bounds_so2ref[index,0],bounds_so2ref[index,1])
#        plt.scatter(bounds_nir2ref[index,0],bounds_nir2ref[index,1])
#        plt.scatter(bounds_mir2ref[index,0],bounds_mir2ref[index,1])

    fig1 = plt.figure(figsize=(figx-10,figy-4))
    ax1 = fig1.add_subplot(111, aspect='equal')
    
    palpha=0.4
    lno_patch=patches.Polygon(bounds_lno2ref[:,0:2],True,alpha=palpha,color='b', label="NOMAD LNO")
    so_patch=patches.Polygon(bounds_so2ref[:,0:2],True,alpha=palpha,color='r', label="NOMAD SO")
    uvis_patch=patches.Circle((bsight_uvis2ref[0], bsight_uvis2ref[1]), bounds_uvis[0][0],alpha=0.8,color='g', label="NOMAD UVIS")
    nir_patch = patches.Polygon(bounds_nir2ref[:,0:2],True,alpha=palpha,color='c', label="ACS NIR")
    mir_patch = patches.Polygon(bounds_mir2ref[:,0:2],True,alpha=palpha,color='m', label="ACS MIR")
#    tirvim_patch=patches.Circle((bsight_tirvim2ref[0], bsight_tirvim2ref[1]), bounds_tirvim[0][0],alpha=palpha,color='k', label="ACS TIRVIM")
    sun_patch = patches.Circle((0.0, 0.0), R/D,color='y', label="Sun on 01/05/2018")

    ax1.add_patch(sun_patch)
    ax1.add_patch(lno_patch)
    ax1.add_patch(so_patch)
    ax1.add_patch(uvis_patch)
#    ax1.add_patch(nir_patch)
#    ax1.add_patch(mir_patch)
#    ax1.add_patch(tirvim_patch)
    
    limit=3e-2
    ax1.set_xlim((-1*limit,limit))
    ax1.set_ylim((-1*limit/2,limit/2))

    ax1.set_xlabel(ref+' X-axis')
    ax1.set_ylabel(ref+' Y-axis')
    title="NOMAD solar occultation boresight alignment"
    ax1.set_title(title)
    plt.legend(loc="lower right")
    if save_figs: plt.savefig(title.replace(" ","_").replace(":","").replace(",","")+".png")

    """repeat for nadir"""
    
    ref='TGO_NOMAD_LNO_OPS_NAD'
    room=4
    # retrieve the UVIS FOV parameters
    code = sp.bodn2c('TGO_NOMAD_UVIS_NAD')
    [shape_uvis, frame_uvis, bsight_uvis, nvectors_uvis, bounds_uvis] = sp.getfov(code, room)
    # retrieve the LNO FOV parameters
    code = sp.bodn2c('TGO_NOMAD_LNO_OPS_NAD')
    [shape_lno, frame_lno, bsight_lno, nvectors_lno, bounds_lno] = sp.getfov(code, room)
    # retrieve the SO FOV parameters
    code = sp.bodn2c('TGO_ACS_NIR_NAD')
    [shape_nir, frame_nir, bsight_nir, nvectors_nir, bounds_nir] = sp.getfov(code, room)
    #retrieve TIRVIM FOV parameters
    code = sp.bodn2c('TGO_ACS_TIRVIM_NAD')
    [shape_tirvim, frame_tirvim, bsight_tirvim, nvectors_tirvim, bounds_tirvim] = sp.getfov(code, room)

    
    # Convert the boresights into the chosen reference frame
    uvis2ref = sp.pxform('TGO_NOMAD_UVIS_NAD', ref, et) # matrix to convert from UVIS frame to ref frame
    lno2ref = sp.pxform('TGO_NOMAD_LNO_OPS_NAD', ref, et) # matrix to convert from LNO frame to ref frame
    nir2ref = sp.pxform('TGO_ACS_NIR_NAD', ref, et)
#    tirvim2ref = sp.pxform('TGO_ACS_TIRVIM_NAD', ref, et)
    

    bsight_uvis2ref = np.dot(uvis2ref,bsight_uvis)
    bsight_lno2ref = np.dot(lno2ref,bsight_lno)
    bsight_nir2ref = np.dot(nir2ref,bsight_nir)
#    bsight_tirvim2ref = np.dot(tirvim2ref,bsight_tirvim)

    bounds_lno2ref = np.zeros((room,3))
    bounds_nir2ref=np.zeros((room,3))
    #no uvis or tirvim required (circular)
    
    bounds_uvis2ref = np.dot(uvis2ref,bounds_uvis[0])
#    bounds_tirvim2ref = np.dot(tirvim2ref,bounds_tirvim[0])
    
    for index in range(room):
        bounds_lno2ref[index,:] = np.dot(lno2ref,bounds_lno[index,:])  # Bounds of the lno fov in ref frame
        bounds_nir2ref[index,:] = np.dot(nir2ref,bounds_nir[index,:])  # Bounds of the lno fov in ref frame

    fig2 = plt.figure(figsize=(figx-10,figy-4))
    ax2 = fig2.add_subplot(111, aspect='equal')
    
    palpha=0.4
    lno_patch=patches.Polygon(bounds_lno2ref[:,0:2],True,alpha=palpha,color='b', label="NOMAD LNO")
    uvis_patch=patches.Circle((bsight_uvis2ref[0], bsight_uvis2ref[1]), bounds_uvis[0][0],alpha=palpha,color='g', label="NOMAD UVIS")
    nir_patch = patches.Polygon(bounds_nir2ref[:,0:2],True,alpha=palpha,color='c', label="ACS NIR")
#    tirvim_patch=patches.Circle((bsight_tirvim2ref[0], bsight_tirvim2ref[1]), bounds_tirvim[0][0],alpha=palpha,color='k', label="ACS TIRVIM")
    sun_patch = patches.Circle((0.0, 0.0), R/D,color='y', label="Sun")

    ax2.add_patch(lno_patch)
    ax2.add_patch(uvis_patch)
#    ax2.add_patch(nir_patch)
#    ax2.add_patch(tirvim_patch)
    
    limit=3e-2
    ax2.set_xlim((-1*limit,limit))
    ax2.set_ylim((-1*limit/2,limit/2))

    ax2.set_xlabel(ref+' X-axis')
    ax2.set_ylabel(ref+' Y-axis')
    title="NOMAD nadir boresight alignment"
    ax2.set_title(title)
    plt.legend(loc="lower right")
    if save_figs: plt.savefig(title.replace(" ","_").replace(":","").replace(",","")+".png")



os.chdir(KERNEL_DIRECTORY)
#sp.unload(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
