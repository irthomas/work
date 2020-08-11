# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 07:30:32 2016

@author: iant

"""

import os
import h5py
import numpy as np
#import numpy.linalg as la
if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
    import getpass
    import pysftp


from matplotlib import rcParams
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D

from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from hdf5_functions_v03 import BASE_DIRECTORY, FIG_X, FIG_Y, stop, getFile, makeFileList, printFileNames
from hdf5_functions_v03 import getFilesFromDatastore
#from analysis_functions_v01b import spectralCalibration,write_log,get_filename_list,stop
from filename_lists_v01 import getFilenameList


if not os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):# and not os.path.exists(os.path.normcase(r"X:\linux\Data")):
    print("Running on windows")
    import spiceypy as sp

    #load spiceypy kernels if required
    KERNEL_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\local_spice_kernels\kernels\mk")
    #KERNEL_DIRECTORY = os.path.normcase(r"X:\linux\Data\kernels\kernels\mk")
    METAKERNEL_NAME = "em16_plan_win.tm"
    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    print("KERNEL_DIRECTORY=%s" %KERNEL_DIRECTORY)
    SPICE_ABERRATION_CORRECTION = "None"
    SPICE_OBSERVER = "-143"


rcParams["axes.formatter.useoffset"] = False

SAVE_FIGS = False
#SAVE_FIGS = True
#save_figs=False
#save_figs=True
#save_files=False
#save_files=True
PASSWORD = ""








#option=1

#checkout="Ground"
#checkout="NEC"
#checkout="MCC"
#checkout="MCO1"
#checkout="MCO2"


#multiple=False #ignore


"""Enter location of data"""

"""Ground calibration"""

"""NEC data"""

"""MCC data"""

"""MCO-1 data"""
#title="SO Raster A"; option=35; file_level="hdf5_level_0p1c"
#title="LNO Raster A"; option=4
#title="SO Raster B"; option=4
#title="LNO Raster B"; option=4

#title="LNO Limb Scan 1"; option=29
#title="LNO Limb Scan 2"; option=29
#title="SO ACS Solar Pointing Test"; option=35
#title="SO LNO Boresight Checks"; option=49; file_level="hdf5_level_1p0a" #check SO/LNO miniscan for boresight position on detector


"""MCO-2 data"""
#title="LNO Limb Scan Spectra for ESA Press Release"; option=36
#title="LNO Limb Scan 1"; option=37; file_level="hdf5_level_0p1c"
#title="LNO Limb Scan 2"; option=37; file_level="hdf5_level_0p1c"
#title="LNO MIR Boresight Check"; option=42; file_level="hdf5_level_0p3a"

"""MTP001"""
#obspaths = ["20180428_023343_0p2a_SO_1_C","20180511_084630_0p2a_SO_1_C","20180522_221149_0p2a_SO_1_C"]
obspaths = ["20180511_084630_0p2a_SO_1_C","20180522_221149_0p2a_SO_1_C"]
#obspaths = ["20180428_023343_0p2a_SO_1_C"] #cut off by atmosphere
#obspaths = ["20180511_084630_0p2a_SO_1_C"] #UVIS
#obspaths = ["20180522_221149_0p2a_SO_1_C"]
#obspaths = ["*2018*_0p2a*_SO_1_C"]
obsTitles = obspaths
fileLevel = "hdf5_level_0p2a"











def polynomialFitSimple(x_in, y_in, order_in, x_out=np.asfarray([0])):
    if x_out.shape[0] == 1:
        return np.polyval(np.polyfit(x_in, y_in, order_in), x_in)
    else:
        return np.polyval(np.polyfit(x_in, y_in, order_in), x_out)
        
def normalise(array_in):
    """normalise input array"""
    return array_in/np.max(array_in)

def et2utc(et):
    return sp.et2utc(et, "C", 0)


def printBoresights(angleSeparationA, angleSeparationB):
    """input manual rotation angles from SPICE kernels to calculate new and old boresight"""
    oldSoBoresight = [0.0, 0.0, 1.0]
    oldUVISBoresight = [0.0, 0.0, 1.0]
    rotationMatrixSoUVIS = sp.pxform("TGO_NOMAD_SO", "TGO_NOMAD_UVIS_OCC", sp.utc2et("2018 APR 01 00:00:00 UTC"))
    oldSoBoresightUVIS = np.dot(oldSoBoresight, rotationMatrixSoUVIS.T)
    oldBoresightSeparation = sp.vsep(oldUVISBoresight, oldSoBoresightUVIS) * sp.dpr() * 60.0
    print("oldBoresightSeparation")
    print(oldBoresightSeparation)
 
    
   
    print("angleSeparationB")
    print(angleSeparationB)
    #####SAVE THIS IT WORKS!!!######
    newSoBoresightTGO = np.asfarray([
            -1.0 * np.sin(angleSeparationB / sp.dpr()), \
            np.sin(angleSeparationA / sp.dpr()) * np.cos(angleSeparationB / sp.dpr()), \
            np.cos(angleSeparationA / sp.dpr()) * np.cos(angleSeparationB / sp.dpr())]) 
    
    print("newSoBoresightTGO, vnorm = %0.6f" %sp.vnorm(newSoBoresightTGO))
    print(newSoBoresightTGO)
    
    newUVISBoresightTGO = np.asfarray([-0.922221097920913, -0.386613383297695, 0.006207330031467])
    oldSoBoresightTGO = np.asfarray([-0.92156, -0.38819, 0.00618])   
    oldUVISBoresightTGO = np.asfarray([-0.92207347097, -0.3869614566418, 0.0064300242046])   
    
    oldNewSoBoresightSeparation = sp.vsep(newSoBoresightTGO, oldSoBoresightTGO) * sp.dpr() * 60.0
    print("oldNewSoBoresightSeparation")
    print(oldNewSoBoresightSeparation)
    
    oldNewUVISBoresightSeparation = sp.vsep(newUVISBoresightTGO, oldUVISBoresightTGO) * sp.dpr() * 60.0
    print("oldNewUVISBoresightSeparation")
    print(oldNewUVISBoresightSeparation)
    
    newSoUVISBoresightSeparation = sp.vsep(newSoBoresightTGO, newUVISBoresightTGO) * sp.dpr() * 60.0
    print("newSoUVISBoresightSeparation")
    print(newSoUVISBoresightSeparation)
    
    oldSoUVISBoresightSeparation = sp.vsep(oldSoBoresightTGO, oldUVISBoresightTGO) * sp.dpr() * 60.0
    print("oldSoUVISBoresightSeparation")
    print(oldSoUVISBoresightSeparation)





def get_vector(date_time, reference_frame):
    obs2SunVector = sp.spkpos("SUN", date_time, reference_frame, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0]
    obs2SunUnitVector = obs2SunVector / sp.vnorm(obs2SunVector)
    return obs2SunUnitVector




frameIndex = 0
def animateFrames(hdf5Files,hdf5Filenames,titles):
#    referenceFrame = "TGO_NOMAD_UVIS_OCC"
    referenceFrame = "TGO_NOMAD_SO"
    #referenceFrame = "TGO_SPACECRAFT"

    for fileIndex, hdf5File in enumerate(hdf5Files):
#        if hdf5Filenames[fileIndex] in ["20180511_084630_0p2a_SO_1_C"]:
#            animationSpeed = 1
#        elif hdf5Filenames[fileIndex] in ["20180428_023343_0p2a_SO_1_C"]:
#            animationSpeed = 1
#        else:
#            print("Error: please add filename and indices to list")
        animationSpeed = 1

        
        hdf5File = hdf5Files[fileIndex]
        print("Reading in file %i: %s" %(fileIndex + 1, obspaths[fileIndex]))
        
        detectorDataAll = get_dataset_contents(hdf5File, "Y")[0]
        binsAll = get_dataset_contents(hdf5File, "Bins")[0]
        observationTimeStringsAll = get_dataset_contents(hdf5File, "ObservationDateTime")[0]
        hdf5File.close()
        
        centreBinIndices = np.where(binsAll[:,0,0] == 116)[0] #window stepping, so find 1st bin
#        centreRowIndex = np.where(binsAll[0,:,0] == 128)[0] #then find which row is the centre of the detector in that bin
#        centrePixelIndex = 200
        
        detectorDataCentreFrames = detectorDataAll[centreBinIndices,:,:] #get data for frames containing centre bins
        observationTimeStringsCentreFrames = observationTimeStringsAll[centreBinIndices,:]
        observationTimesCentreFrames = np.asfarray([np.mean([sp.utc2et(timeCentreFrame[0]),sp.utc2et(timeCentreFrame[1])]) for timeCentreFrame in list(observationTimeStringsCentreFrames)])
        
#        detectorDataCentrePixel = detectorDataCentreFrames[:,centreRowIndex,centrePixelIndex] #get data for detector centre row

        nFrames = detectorDataCentreFrames.shape[0]
        maxValue = np.max(detectorDataCentreFrames)
    
        def update_frame(frame_index):
        #    global detectorDataCentreFrames
            return detectorDataCentreFrames[frame_index,:,:]
        
        
        def update_text(date_time):
            obs2SunVector = sp.spkpos("SUN", date_time, referenceFrame, SPICE_ABERRATION_CORRECTION, SPICE_OBSERVER)[0]
            obs2SunUnitVector = obs2SunVector / sp.vnorm(obs2SunVector)
        #    obs2SunAngle = sp.vsep(obs2SunUnitVector, np.asfarray([0.0, 0.0, 1.0]))
            return obs2SunUnitVector
    
        
        global frameIndex
        
        fig = plt.figure(figsize=(FIG_X, FIG_Y))
        ax = fig.add_subplot(111)
        im = ax.imshow(update_frame(frameIndex), animated=True, aspect=6, vmin=0.0, vmax=maxValue, cmap=plt.cm.GnBu)
        txt = ax.text(10,10,update_text(observationTimesCentreFrames[frameIndex]))
        
        def updatefig(*args):
            global frameIndex
            
            if frameIndex < nFrames:
                frameIndex += 1
            else:
                frameIndex = 0
#            print(frameIndex)
        #    print(obs2SunUnitVector)
            im.set_array(update_frame(frameIndex))
            txt.set_text("%0.5f, %0.5f, %0.5f" %tuple(update_text(observationTimesCentreFrames[frameIndex])) + ", frame=%i" %frameIndex)
            return im, txt, 
        
        return animation.FuncAnimation(fig, updatefig, interval=animationSpeed, blit=True)
#        plt.show()
#        return ani



def analyseLineScan(hdf5Files,hdf5Filenames,titles):
#    referenceFrame = "TGO_NOMAD_UVIS_OCC"
    referenceFrame = "TGO_NOMAD_SO"
#    referenceFrame = "TGO_SPACECRAFT"


    fig, ax = plt.subplots(figsize=(FIG_X, FIG_Y))
    #new boresight X direction is assumed to be zero if not calculated
    meanParFit = 0.0
    #new boresight Y direction is assumed to be zero if not calculated
    meanPerpFit = 0.0
    for fileIndex, hdf5File in enumerate(hdf5Files):
        
        #add indices bounding each pass across the sun
        if hdf5Filenames[fileIndex] in ["20180511_084630_0p2a_SO_1_C"]:
            lineScanCentreIndicesAll = [range(60,120,1),range(280,340,1),range(400,490,1),range(630,720,1),range(760,850,1),range(990,1060,1),range(1150,1200,1)]
            completeLineScanIndices = range(7)
            analyse=True
            parallel=True
        elif hdf5Filenames[fileIndex] in ["20180428_023343_0p2a_SO_1_C"]:
            lineScanCentreIndicesAll = [range(230,370,1),range(380,520,1),range(520,680,1)]
            completeLineScanIndices = range(3)
            analyse=False
            parallel=False
        elif hdf5Filenames[fileIndex] in ["20180522_221149_0p2a_SO_1_C"]:
            lineScanCentreIndicesAll = [range(280,300,1),range(440,470,1),range(590,650,1),range(760,815,1),range(925,970,1),range(1100,1120,1)]
            completeLineScanIndices = range(1,5,1)
            analyse=True
            parallel=False
        else:
            print("Error: please add filename and indices to list")
    
        hdf5File = hdf5Files[fileIndex]
        print("Reading in file %i: %s" %(fileIndex + 1, obspaths[fileIndex]))
        
        detectorDataAll = get_dataset_contents(hdf5File, "Y")[0]
        binsAll = get_dataset_contents(hdf5File, "Bins")[0]
        observationTimeStringsAll = get_dataset_contents(hdf5File, "ObservationDateTime")[0]
        hdf5File.close()
        
        centreBinIndices = np.where(binsAll[:,0,0] == 116)[0] #window stepping, so find 1st bin
        centreRowIndex = np.where(binsAll[0,:,0] == 128)[0] #then find which row is the centre of the detector in that bin
        centrePixelIndex = 200
        
        detectorDataCentreFrames = detectorDataAll[centreBinIndices,:,:] #get data for frames containing centre bins
        observationTimeStringsCentreFrames = observationTimeStringsAll[centreBinIndices,:]
        observationTimesCentreFrames = np.asfarray([np.mean([sp.utc2et(timeCentreFrame[0]),sp.utc2et(timeCentreFrame[1])]) for timeCentreFrame in list(observationTimeStringsCentreFrames)])
        
        detectorDataCentrePixel = detectorDataCentreFrames[:,centreRowIndex,centrePixelIndex] #get data for detector centre row
        
    
    
        centreLineCentrePixel = detectorDataCentreFrames[:, centreRowIndex, 200].flatten()
        nFrames = centreLineCentrePixel.shape[0]
        maxValuesFrameIndices = []
        maxValuesPixelCounts = []
        
        plt.figure(figsize=(FIG_X, FIG_Y))
        for lineScanCentreIndices in lineScanCentreIndicesAll:
            centreLineCentrePixelFit = polynomialFitSimple(np.arange(len(centreLineCentrePixel[lineScanCentreIndices])), centreLineCentrePixel[lineScanCentreIndices], 5)
            indexMaxValue = centreLineCentrePixelFit.argmax()
            maxValuesFrameIndices.append(lineScanCentreIndices[indexMaxValue])
            maxValuesPixelCounts.append(centreLineCentrePixelFit[indexMaxValue])
            plt.plot(lineScanCentreIndices,centreLineCentrePixelFit)
        plt.plot(centreLineCentrePixel)
        plt.scatter(maxValuesFrameIndices, maxValuesPixelCounts, c="k")
        
        maxValuesFrameIndices = np.asarray(maxValuesFrameIndices)
        maxValuesPixelCounts = np.asarray(maxValuesPixelCounts)
        
        
        maxValuesVectors = np.asfarray([get_vector(datetime,referenceFrame) for datetime in observationTimesCentreFrames[maxValuesFrameIndices]])
        
        #find peak of curves (but don't use for calculation, only for plottingS)
        maxValuesPerpFit = polynomialFitSimple(maxValuesFrameIndices[completeLineScanIndices], maxValuesPixelCounts[completeLineScanIndices], 2, x_out=np.arange(nFrames))
        plt.plot(np.arange(nFrames), maxValuesPerpFit)
        maxPerpIndex = maxValuesPerpFit.argmax()
        plt.scatter(maxPerpIndex, maxValuesPerpFit[maxPerpIndex])
        
        
        unitVectors = np.asfarray([get_vector(datetime,referenceFrame) for datetime in observationTimesCentreFrames])
        marker_colour = np.log(detectorDataCentrePixel+1000).flatten()
        ax.set_xlim([-0.006,0.006])
        ax.set_ylim([-0.006,0.006])
        ax.set_xlabel("%s FRAME X" %referenceFrame)
        ax.set_ylabel("%s FRAME Y" %referenceFrame)
        ax.scatter(unitVectors[:,0], unitVectors[:,1], c=marker_colour, alpha=0.6)
        ax.set_aspect("equal")
        
        
        ax.scatter(maxValuesVectors[:,0], maxValuesVectors[:,1], c="k")
            

        if analyse:
            if parallel:
                #mean X
                meanParFit = np.mean(maxValuesVectors[:,0])
                
                ax.scatter(0.0,0.0, marker="x", c="k", s=160)
                ax.plot([meanParFit,meanParFit],[maxValuesVectors[0,1],maxValuesVectors[-1,1]], "k")
                
                vectorOrigin = [0.0,0.0,1.0]
                vectorNew = [meanParFit, meanPerpFit, 0.0]
                vectorNew[2] = np.sqrt(1.0-vectorNew[0]**2-vectorNew[1]**2)
                print("vectorNew (1 direction only)")
                print(vectorNew)
                
                angleSeparation = sp.vsep(vectorOrigin,vectorNew) * sp.dpr() * 60.0
                print("angleSeparation (1 direction only)")
                print(angleSeparation)
            else:
                #mean Y
                meanPerpFit = np.mean(maxValuesVectors[:,1])

                ax.scatter(0.0,0.0, marker="x", c="k", s=160)
                ax.plot([maxValuesVectors[0,0],maxValuesVectors[-1,0]],[meanPerpFit,meanPerpFit], "k")
                
                vectorOrigin = [0.0,0.0,1.0]
                vectorNew = [meanParFit, meanPerpFit, 0.0]
                vectorNew[2] = np.sqrt(1.0-vectorNew[0]**2-vectorNew[1]**2)
                print("vectorNew (1 direction only)")
                print(vectorNew)
                
                angleSeparation = sp.vsep(vectorOrigin,vectorNew) * sp.dpr() * 60.0
                print("angleSeparation (1 direction only)")
                print(angleSeparation)

    if meanParFit != 0.0 and meanPerpFit != 0.0:
        ax.scatter(0.0,0.0, marker="x", c="k", s=160)
        ax.scatter(meanParFit,meanPerpFit, marker="x", c="k", s=160)
        ax.plot([meanParFit,meanParFit],[maxValuesVectors[0,1],maxValuesVectors[-1,1]], "k")
        ax.plot([meanParFit,meanParFit],[np.min(unitVectors[:,1]),np.max(unitVectors[:,1])], "k")
        circle1 = plt.Circle((0.0,0.0), 10.0/60.0/sp.dpr(), color='k', alpha=0.1)
#        circle1 = plt.Circle((meanParFit,meanPerpFit), 10.0/60.0/sp.dpr(), color='k', alpha=0.1)
        ax.add_artist(circle1)
        offset = 0.0001
        ax.text(0.0+offset,0.0+offset*1.5,"Origin")
        ax.text(meanParFit+offset,meanPerpFit-offset*2.0,"New Boresight")


hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
#analyseLineScan(hdf5Files, hdf5Filenames, titles)
#ani = animateFrames(hdf5Files, hdf5Filenames, titles); plt.show(plt.show())

#don't trust the calculation - angles estimated manually from data
angleSeparationA = -89.08792599580298
angleSeparationB = 67.15504926474100 + 3.0/60.0 #old value plus 3 arcminutes
printBoresights(angleSeparationA, angleSeparationB)











#channel={"SO ":"so", "SO-":"so", "LNO":"lno", "UVI":"uvis"}[title[0:3]]
#detector_centre={"so":128, "lno":152, "uvis":0}[channel] #or 152 for lno??
#nec_sun_detector_centre={"so":130, "lno":157, "uvis":0}[channel] #for static measurements during NEC using the ground calibration

#
#
#if option==2:
#    """plot intensity vs time"""
#    if channel=="so" or channel=="lno":
#        detector_data,_,_ = get_dataset_contents(hdf5_file,"YBins")
#        exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
#        binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
#        first_window_top = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0][0]
#        hdf5_file.close()
#        
#        if binning==2: #stretch array
#            detector_data=np.repeat(detector_data,2,axis=1)
#           
#        if checkout=="NEC" or checkout=="MCC":
#            nlines=16
#            nsteps=16
#        elif channel=="lno":
#            nlines=24
#            nsteps=7
#        else:
#            print("Error: don't use for MCO SO")
#
#            
#            
#        sum_centre_all=[]
#        exponent_all=[]
#        full_frame_all=[]
#        vert_slice_all=[]
#        for index2 in range(int(detector_data.shape[0]/(nsteps/binning))): #loop through frames
#            full_frame = np.zeros((nlines*nsteps,320))
#            for index in range(int(nsteps/binning)): #loop through window subframes
#                full_frame[(index*nlines*binning):((index+1)*nlines*binning),:]=detector_data[(index+(index2*nsteps/binning)),:,:]
#                exponent_all.append(exponent)
#    #        sum_centre_all.append(np.sum(full_frame[(detector_centre-4):(detector_centre+4),220:236]))
#            sum_centre_all.append(np.sum(full_frame[detector_centre-first_window_top,228]))
#            full_frame_all.append(full_frame)
#        
#        #plt.figure(figsize=(10,8))
#        #plt.imshow(full_frame_all[64])
#        #plt.colorbar()
#        
#        #plt.figure(figsize=(10,8))
#        #plt.imshow(full_frame_all[64])
#        #plt.colorbar()
#        
#        time=np.arange(int(detector_data.shape[0]/(nsteps/binning)))*(nsteps/binning)
#        
#        plt.figure(figsize=(10,8))
#        plt.plot(time,sum_centre_all)
##        plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
#        plt.ylabel("Sum signal ADU for pixel %i,228" %(detector_centre))
#        plt.xlabel("Approx time after pre-cooling ends (seconds)")
#        plt.title(title)
#        plt.yscale("log")
#        if save_figs: plt.savefig(title+"_intensity_versus_time_raster_scan_log.png")
#        
#    #    np.savetxt(title+".txt", np.transpose(np.asfarray([time,sum_centre_all])), delimiter=",")
#    
#        plt.figure(figsize=(10,8))
#        plt.plot(time,sum_centre_all)
##        plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
#        plt.ylabel("Sum signal ADU for pixel %i,228" %(detector_centre))
#        plt.xlabel("Approx time after pre-cooling ends (seconds)")
#        plt.title(title)
#        if save_figs: plt.savefig(title+"_intensity_versus_time_raster_scan.png")
#        
#        
#        plt.figure(figsize=(10,8))
#        plt.plot(2**exponent)
#        plt.ylabel("Exponent ADU")
#        plt.xlabel("Approx time after pre-cooling ends (seconds)")
#        plt.title(title)
#        if save_figs: plt.savefig(title+"_exponent_versus_time_raster_scan.png")
#
#if option==3:
#    """v1 plot animations of chosen raster scan"""
#    """this should be changed so a function generates the new frames"""
#    detector_data,_,_ = get_dataset_contents(hdf5_file,"YBins")
#    time_data = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
#    binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
#    hdf5_file.close()
#    
#    if binning==2: #stretch array
#        detector_data=np.repeat(detector_data,2,axis=1)
#    
#    fig=plt.figure(figsize=(10,8))
#    ax = fig.add_subplot(111)
#    
#    full_frame_all=[]
#    for index2 in range(int(len(time_data)/(16/binning))):
#        full_frame = np.zeros((256,320))
#        for index in range(int(16/binning)):
#            full_frame[(index*16*binning):((index+1)*16*binning),:]=detector_data[(index+(index2*16/binning)),:,:]
#        frame = ax.imshow(full_frame, vmin=0, vmax=1e4, animated=True)
#        t = ax.annotate(time_data[(index+(index2*16/binning))],(50,50),size=50)#time_data[(index+(index2*16))])
#        full_frame_all.append([frame,t])
#    
#    ani = animation.ArtistAnimation(fig, full_frame_all, interval=50, blit=True)
#    if save_figs: ani.save(title+" Detector_Frame.mp4", fps=20, extra_args=['-vcodec', 'libx264'])
#    plt.show()
#
#    print("Done")
#    gc.collect() #clear animation from memory
#    
#if option==4:
#    """plot intensity vs position in raster"""
#
#
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
#
#    orientation=title[-1].lower() #find orientation from last letter of title
#
#    #get data
#    if multiple:
#        hdf5_file = hdf5_files[0]
#    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"YBins")
#    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
#    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
#    window_top_all = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0]
#    binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
#    hdf5_file.close()
#
#    if binning==2: #stretch array
#        detector_data_all=np.repeat(detector_data_all,2,axis=1)
#
#    #convert data to times and boresights using spice
#    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
#    time_error=1    
#    boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)
#    
##    #sum all detector data
##    detector_sum=np.sum(detector_data_all[:,:,:], axis=(1,2))
##    detector_sum[detector_sum<500000]=500000#np.mean(detector_sum)
#    
#    #find indices where centre of detector is
#    meas_indices=[]
#    detector_sum=[]
#    for index,window_top in enumerate(window_top_all):
#        if detector_centre in range(window_top,window_top+16*binning):
#            detector_line=detector_centre-window_top
#            meas_indices.append(index)
#            pixel_value=detector_data_all[index,detector_line,228]
#            if pixel_value<100:
#                pixel_value=100
#            detector_sum.append(pixel_value)
#    detector_sum=np.asfarray(detector_sum)
#    chosen_boresights=boresights_all[meas_indices,:]
#    lon_lats=np.asfarray([sp.reclat(chosen_boresight)[1:3] for chosen_boresight in list(chosen_boresights)])
#    
#    if plot_both and orientation=="a":
#        detector_sum_a=detector_sum
#        chosen_boresights_a=chosen_boresights
#        lon_lats_a=lon_lats
#        title_a=title
##        centre_theoretical_a=centre_theoretical
##        centre_theoretical_lat_lon_a=centre_theoretical_lat_lon
#    if plot_both and orientation=="b":
#        detector_sum_b=detector_sum
#        chosen_boresights_b=chosen_boresights
#        lon_lats_b=lon_lats
#        title_b=title
##        centre_theoretical_b=centre_theoretical
##        centre_theoretical_lat_lon_b=centre_theoretical_lat_lon
#
#
#    if not plot_both:
#        marker_colour=np.log(1+detector_sum-min(detector_sum))
#        fig = plt.figure(figsize=(9,9))
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(chosen_boresights[:,0], chosen_boresights[:,1], chosen_boresights[:,2], c=marker_colour, marker='o', linewidth=0)
#        ax.azim=-108
#        ax.elev=-10
#        plt.title(title+": Signal on pixel %i,228" %detector_centre)
#        if save_figs: plt.savefig(title+"_Signal_on_pixel_%i,228_in_J2000.png" %detector_centre)
#
#        plt.figure(figsize=(9,9))
#        plt.scatter(lon_lats[:,0], lon_lats[:,1], c=marker_colour, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
#        plt.scatter(centre_theoretical_lat_lon[0], centre_theoretical_lat_lon[1], c='r', marker='*', linewidth=0, s=120)
#        plt.xlabel("Solar System Longitude (degrees)")
#        plt.ylabel("Solar System Latitude (degrees)")
#        plt.title(title+": Signal on pixel %i,228" %detector_centre)
#        if save_figs: plt.savefig(title+"_Signal_on_pixel_%i,228_in_lat_lons.png" %detector_centre)
#
#    
#    if plot_both and orientation=="b":
#        marker_colour_a=np.log(1+detector_sum_a-min(detector_sum_a))
#        fig = plt.figure(figsize=(9,9))
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(chosen_boresights_a[:,0], chosen_boresights_a[:,1], chosen_boresights_a[:,2], c=marker_colour_a, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
#        marker_colour_b=np.log(1+detector_sum_b-min(detector_sum_b))
#        ax.scatter(chosen_boresights_b[:,0], chosen_boresights_b[:,1], chosen_boresights_b[:,2], c=marker_colour_b, cmap=plt.cm.gnuplot, marker='o', linewidth=0)
#        ax.azim=-108
#        ax.elev=-10
#        plt.gca().patch.set_facecolor('white')
#        ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
#        ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
#        ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
##        plt.title(title_a+" & "+title_b+": Signal on pixel %i,228" %detector_centre)
#        plt.title(channel.upper()+" Solar Line Scan: Signal Measured on Detector Centre")
#
#        if save_figs: plt.savefig(title_a+"_"+title_b+"_Signal_on_pixel_%i,228_in_J2000.png" %detector_centre, dpi=600)
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
#        
#        
#
#if option==6:
#    """make vertical detector plots where sun is seen to determine slit position and time when in centre"""
##    so_boresight_to_tgo=(-0.92136,-0.38866,0.00325) #define so boresight in tgo reference frame
##    lno_nadir_boresight_to_tgo=(-0.00685,-0.99993,0.00945) #define lno boresight in tgo reference frame
#
#    if checkout=="NEC":
#        detector_data_all,_,_ = get_dataset_contents(hdf5_file,"YBins")
#        time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
#        date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
#        window_top_all = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0]
#        binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
#    else:
#        detector_data_all,_,_ = get_dataset_contents(hdf5_file,"Y")
#        time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0][:,0]
#        date_data_all = get_dataset_contents(hdf5_file,"ObservationDate")[0][:,0]
#        window_top_all = get_dataset_contents(hdf5_file,"WindowTop")[0]
#        binning = get_dataset_contents(hdf5_file,"Binning")[0][0]+1
#    hdf5_file.close()
#    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
#    time_error=1    
##    boresights_all=find_boresight(epoch_times_all,time_error,so_boresight_to_tgo)
##    detector_sum=np.sum(detector_data_all[:,:,:], axis=(1,2))
##    detector_sum=detector_data_all[:,0,230]
##    detector_sum[detector_sum<500000]=500000 #to plot better
#
#    
#    sun_indices1=0
#    sun_indices2=0
#    if title=="SO ACS Raster 1":
#        sun_indices1=range(580,700) #plot all data
#        sun_indices2=range(880,990) #plot all data
#        window_tops=[96,112,128,144]
##        sun_indices1=range(640,680) #plot limited data
##        sun_indices2=range(920,960) #plot limited data
#    elif title=="SO ACS Raster 2":
#        sun_indices1=range(100,400)
#        sun_indices2=range(100,200)
#        window_tops=[96,112,128,144]
#    elif title=="LNO ACS Raster 1":
##        sun_indices1=range(405) #plot all data
##        sun_indices2=range(405,1020) #plot all data
#        sun_indices1=range(1020) #plot all data on same plot
#        window_tops=[64,80,96,112,128,144,160,176,192,208,224] #plot all data
##        sun_indices1=range(475,530) #plot limited data
##        window_tops=[128,144,160]
#
#    elif title=="SO Raster 1" or title=="SO Raster 2":
#        sun_indices1=range(880,1150) #plot all data
##        sun_indices1=range(1080,1120) #plot limited data
#        window_tops=[96,112,128,144]
#    elif title=="LNO Raster 1" or title=="LNO Raster 2":
#        sun_indices1=range(0,2100) #plot all data
#        window_tops=[80,96,112,128,144,160,176,192,208] #plot all data
##        sun_indices1=range(1040,1110) #plot limited data
##        window_tops=[128,144,160]
#    elif title=="SO Raster 4A" or title=="SO Raster 4B":
#        sun_indices1=range(0,2100) #plot all data
#        window_tops=[96,112,128,144] #plot all data
#    elif title=="LNO Raster 1A":
#        sun_indices1=range(0,2100) #plot all data
#        window_tops=[32,64,96,128,160,192,224] #plot all data
#    elif title=="LNO Raster 4A":
#        sun_indices1=range(0,2100) #plot all data
#        window_tops=[32,64,96,128,160,192,224] #plot all data
#
#
#    if not sun_indices1==0:
#        indices=[]
#        for index,window_top in enumerate(window_top_all):
#            if window_top in window_tops:
#                if index in sun_indices1:
#                    indices.append(index)
#    
#        times=time_data_all[indices]
#        dates=date_data_all[indices]
#        window_tops_selected=window_top_all[indices]
#        epochs=epoch_times_all[indices]
#        detector_counts=detector_data_all[indices,:,:]
#
#        xs=[]    
#        for window_top in window_tops_selected:
#            xs.append(np.arange(16)*binning+window_top)
#        xs=np.asarray(xs)
#    
#        detector_v_centre=230
#        vert_slices=detector_counts[:,:,detector_v_centre]
#    
#        
##        plt.figure(figsize=(10,8))
##        plt.plot(np.transpose(xs),np.transpose(vert_slices))
##        plt.ylabel("Sum signal ADU for pixels in detector column %i" %detector_v_centre)
##        plt.xlabel("Vertical Pixel Number")
##        if channel=="lno":
##            plt.xlim((60,240))
##        elif channel=="so":
##            plt.xlim((100,150))
###        plt.legend(times)
##        plt.title(title+" pass 1: vertical columns on detector where sun is seen")
##        if save_figs: 
##            plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_vertical_columns_on_detector_where_sun_is_seen.png")
#
#    if not sun_indices2==0:
#        indices=[]
#        for index,window_top in enumerate(window_top_all):
#            if window_top in window_tops:
#                if index in sun_indices2:
#                    indices.append(index)
#        
#        times=time_data_all[indices]
#        dates=date_data_all[indices]
#        window_tops_selected=window_top_all[indices]
#        epochs=epoch_times_all[indices]
#        detector_counts=detector_data_all[indices,:,:]
#    
#        xs=[]    
#        for window_top in window_tops_selected:
#            xs.append(np.arange(16)*binning+window_top)
#        xs=np.asarray(xs)
#    
#        detector_v_centre=230
#        vert_slices=detector_counts[:,:,detector_v_centre]
#        
#        plt.figure(figsize=(10,8))
#        plt.plot(np.transpose(xs),np.transpose(vert_slices))
#        plt.ylabel("Sum signal ADU for pixels in detector column %i" %detector_v_centre)
#        plt.xlabel("Vertical Pixel Number")
#        plt.legend(times)
#        plt.title(title+" pass 2: vertical columns on detector where sun is seen")
#        if save_figs: 
#            plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_pass_2_vertical_columns_on_detector_where_sun_is_seen.png")
#
#
#    """check detector smile"""
#    if title=="LNO Raster 4A" or title=="LNO Raster 4B":
#        indices = [index for index,window_top in enumerate(window_top_all) if window_top in window_tops]
#        continuum_range = [203,209,217,223]
#        signal_minimum = 300000
#    if title=="SO Raster 4A" or title=="SO Raster 4B":
#        indices = [index for index,window_top in enumerate(window_top_all) if window_top in window_tops]
#        continuum_range = [210,215,223,228]
#        signal_minimum = 200000
#
#    detector_data_selected = detector_data_all[indices,:,:]
#    window_top_selected = window_top_all[indices]
#        
#    plt.figure(figsize = (figx,figy))
#    absorption_minima=[]
#    detector_rows=[]
#    for frame_index in range(len(detector_data_selected[:,0,0])):
#        for bin_index in range(len(detector_data_selected[0,:,0])):
#            if detector_data_selected[frame_index,bin_index,200]>signal_minimum:
#                detector_data_normalised = (detector_data_selected[frame_index,bin_index,:]-np.min(detector_data_selected[frame_index,bin_index,:]))/(np.max(detector_data_selected[frame_index,bin_index,:])-np.min(detector_data_selected[frame_index,bin_index,:]))
#                plt.plot(detector_data_normalised)
#                
#                pixels=np.arange(320)
#                
#                continuum_pixels = pixels[range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3])]    
#                continuum_spectra = detector_data_selected[frame_index,bin_index,range(continuum_range[0],continuum_range[1])+range(continuum_range[2],continuum_range[3])]
#                
#                #fit polynomial to continuum on either side of absorption band
#                coefficients = np.polyfit(continuum_pixels,continuum_spectra,2)
#                continuum = np.polyval(coefficients,pixels[range(continuum_range[0],continuum_range[3])])
#                absorption = detector_data_selected[frame_index,bin_index,range(continuum_range[0],continuum_range[3])]/continuum
#                absorption_pixel = np.arange(20)
##                plt.plot(absorption_pixel,absorption)
#                
#                abs_coefficients = np.polyfit(pixels[range(continuum_range[0],continuum_range[3])][6:12],absorption[6:12],2)
#                detector_row = window_top_selected[frame_index]+bin_index*binning
#                
#                absorption_minima.append((-1*abs_coefficients[1]) / (2*abs_coefficients[0]))
#                
#                detector_rows.append(detector_row)
#                
#    plt.figure(figsize = (figx-10,figy-2))
#    plt.scatter(detector_rows,absorption_minima,marker="o",linewidth=0,alpha=0.5)
#    
#    fit_coefficients = np.polyfit(detector_rows,absorption_minima,1)
#    fit_line = np.polyval(fit_coefficients,detector_rows)
#    
#    plt.plot(detector_rows,fit_line,"k", label="Line of best fit, min=%0.1f, max=%0.1f" %(np.min(fit_line),np.max(fit_line)))
#    plt.legend()
#    plt.ylabel("Pixel column number at minimum of quadratic fit to absorption line")
#    plt.xlabel("Detector row")
#    plt.title(title+" Detector Smile: Quadratic fits to absorption line")
#    plt.ylim((continuum_range[1],continuum_range[2]))
#    plt.tight_layout()
#    if save_figs: 
#        plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_detector_smile.png")
#
#
##    plt.figure(figsize=(9,9))
##    plt.plot(boresights_all[:,0],boresights_all[:,1],'b.')
#
##    plot partially reconstructed frames or raw 16 lines with imshow
##    plt.figure(figsize=(10,8))
##    plt.imshow(full_frame_all[64])
##    plt.colorbar()
##    
##    plt.figure(figsize=(10,8))
##    plt.imshow(full_frame_all[64])
##    plt.colorbar()
#
#if option==7:
#    """convert peak sun time to boresight"""
#    date1=["2016-04-13"]; time1=["03-25-05.505"]    
#    date2=["2016-04-13"]; time2=["03-20-17.509","03-20-33.509"] #half way between these two
#    
#    so_boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical
#
#    epoch_time_sun=convert_hdf5_time_to_spice_utc(time1,date1)[0]
#    
#    cmatrix_sun=sp.ckgp(epoch_time_sun,1)
#    
#    observer="-143" #tgo
#    target="SUN"
#    relative_sun_position=sp.spkpos(observer,target,epoch_time_sun)
#    sun_distance = la.norm(relative_sun_position)
#    sun_pointing_vector = relative_sun_position / sun_distance
#    
#    new_boresight = tuple(np.dot(cmatrix_sun,sun_pointing_vector))
#    print("boresight="+"%.10f "*3 %new_boresight)
#    print("angle_difference=%.10f arcmins" %(py_ang(so_boresight_to_tgo,new_boresight) * 180 * 60 / np.pi))
#    
#    
#    
#    
#if option==8:
#    """calculate time when theoretical boresight pointed to sun"""
#
#    step=0.1 #16 minutes #plot limited
#    nsteps=40 * 120 
##    step=0.5#40 minutes #plot whole range
#    if title=="SO Raster 1":
#        date1=["2016-04-11"]; time1=["19-58-24.998"]     #SO calculated
##        epoch_time_start=sp.utc2et("2016APR11-19:40:00.998") #for SO full range
#        epoch_time_start=sp.utc2et("2016APR11-19:50:00.998") #for SO limited range
#        step=0.1
#        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical
#        nsteps=80 * 120 
#    if title=="SO ACS Raster 1":
#        date1=["2016-04-13"]; time1=["03-25-05.505"]     #SO calculated
##        epoch_time_start=sp.utc2et("2016APR13-03:05:05.505") #for SO whole time range
#        epoch_time_start=sp.utc2et("2016APR13-03:20:05.505") #for SO limited range
#        step=0.1
#        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical
#    if title=="LNO Raster 1":
#        date1=["2016-04-11"]; time1=["20-48-10.958"]     #LNO calculated
##        epoch_time_start=sp.utc2et("2016APR11-19:40:00.998") #for SO full range
#        epoch_time_start=sp.utc2et("2016APR11-20:40:00.958") #for SO limited range
#        step=0.1
#        boresight_to_tgo=(-0.92126,-0.38890,0.00368) #theoretical
#        nsteps=80 * 120 
#    if title=="LNO ACS Raster 1":
#        date1=["2016-04-13"]; time1=["04-32-55.944"]     #LNO calculated
##        epoch_time_start=sp.utc2et("2016APR13-03:05:05.505") #for SO whole time range
#        epoch_time_start=sp.utc2et("2016APR13-04:27:54.944") #for SO limited range
#        step=0.1
#        boresight_to_tgo=(-0.92126,-0.38890,0.00368) #theoretical
#    if title=="UVIS ACS Raster 1":
#        date1=["2016-04-13"]; time1=["03-25-08.431"]     #UVIS calculated
#        epoch_time_start=sp.utc2et("2016APR13-03:20:08.431") #for UVIS limited range
#        step=0.1
#        boresight_to_tgo=(-0.921550000000000,-0.388220000000000,0.003710000000000) #theoretical
#
#
#    epoch_time_peak=convert_hdf5_time_to_spice_utc(time1,date1)[0]
#    utc_time_peak=sp.et2utc(epoch_time_peak, "C", 0)
#    
#
#    cmatrices=sp.ckgp(epoch_time_start,1,step,nsteps)
#    times=sw.et2utcx(epoch_time_start,step,nsteps)
#    boresights_all=[]
#    for cmatrix in cmatrices:
#        boresights_all.append(np.dot(np.transpose(cmatrix),boresight_to_tgo))
#    boresights_all=np.asfarray(boresights_all)
#    [_,boresight_lons,boresight_lats] = find_rad_lon_lat(boresights_all)
#    
#    observer="-143" #tgo
#    target="SUN"
#    relative_sun_position=sw.spkposx(observer,target,epoch_time_start,step,nsteps)
#    sun_distance = la.norm(relative_sun_position[0,:])
#    sun_pointing_vector = relative_sun_position / sun_distance
#    
#    angles=[]
#    for boresight,sun in zip(boresights_all,sun_pointing_vector):
#        angles.append(py_ang(boresight,sun) * 180 * 60 / np.pi)
#    angles=np.asfarray(angles)
#
#    fig = plt.figure(figsize=(9,9))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(boresights_all[:,0],boresights_all[:,1],boresights_all[:,2], marker='.', linewidth=0)
##    ax.scatter(sun_pointing_vector[:,0],sun_pointing_vector[:,1],sun_pointing_vector[:,2], marker='o', linewidth=0, c='y')
#    
#    index_when_peak_signal=times.index(utc_time_peak) #find index of peak signal
#    index_when_centred_on_sun=np.abs(angles-0).argmin() #find index when raster scan is in centre
#    print("time of peak signal= %s" %times[index_when_peak_signal])
#    print("time of theoretical sun centre= %s" %times[index_when_centred_on_sun])
#    ax.scatter(sun_pointing_vector[index_when_peak_signal,0],sun_pointing_vector[index_when_peak_signal,1],sun_pointing_vector[index_when_peak_signal,2], marker='o', linewidth=0, c='r')
#    ax.scatter(sun_pointing_vector[index_when_centred_on_sun,0],sun_pointing_vector[index_when_centred_on_sun,1],sun_pointing_vector[index_when_centred_on_sun,2], marker='o', linewidth=0, c='g')
#    
#    angular_offset=py_ang(boresights_all[index_when_peak_signal,:],boresights_all[index_when_centred_on_sun,:]) * 180 * 60 / np.pi
#    print("horizontal offset between theoretical and true boresight=%f" %angular_offset)
#    plt.figure()
#    plt.plot(angles)
#
#    plt.figure(figsize=(9,9))
#    plt.scatter(boresight_lons,boresight_lats, marker='.', linewidth=0, c='b')
#    plt.scatter(boresight_lons[0:100],boresight_lats[0:100], marker='.', linewidth=0, c='y')
#    plt.scatter(boresight_lons[index_when_peak_signal],boresight_lats[index_when_peak_signal],marker='o', linewidth=0, c='r')
#    plt.scatter(boresight_lons[index_when_centred_on_sun],boresight_lats[index_when_centred_on_sun],marker='o', linewidth=0, c='g')
#    print("%f %f" %(boresight_lons[index_when_centred_on_sun],boresight_lats[index_when_centred_on_sun]))
#    
#if option==9:
#    """find peak for UVIS vertically binned data"""
#    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"Y")
#    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:]
#    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:]
#    hdf5_file.close()
#    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
#    time_error=1    
##    boresights_all=find_boresight(epoch_times_all,time_error,so_boresight_to_tgo)
#    detector_sum=np.sum(detector_data_all[:,0,:], axis=(1))
#    detector_sum[0:2100]=0 #otherwise peak is found in first pass
#    index_when_peak_signal=np.abs(detector_sum-0).argmax()
#    print("max signal at %s %s" %(time_data_all[index_when_peak_signal],date_data_all[index_when_peak_signal]))
#    
#    
#if option==10:
#    """plot proposed boresights on ACS raster scan with detector sum"""
#    detector_data_all,_,_ = get_dataset_contents(hdf5_file,"Y")
#    if channel=="uvis":
#        time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:]
#        date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:]
#        detector_sum=np.sum(detector_data_all[:,0,:], axis=(1))
#        detector_sum[0]=detector_sum[1] #fudge because 1st frame is bias
#    elif channel=="so" or channel=="lno":
#        time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
#        date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
#        #sum all detector data
#        detector_sum=np.sum(detector_data_all[:,:,:], axis=(1,2))
#        detector_sum[detector_sum<500000]=500000#np.mean(detector_sum)
#    hdf5_file.close()
#    detector_data_all=[]
#    
#    print("Calculating times")
#    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
#    marker_colour=np.log(1+detector_sum-min(detector_sum))  
#    
#    old_so_boresight_to_tgo=(-0.92136,-0.38866,0.00325) #define so boresight in tgo reference frame
##    lno_nadir_boresight_to_tgo=(-0.00685,-0.99993,0.00945) #define lno boresight in tgo reference frame
#    old_lno_boresight_to_tgo=(-0.92126,-0.38890,0.00368)
#    old_uvis_boresight_to_tgo=(-0.921550000000000,-0.388220000000000,0.003710000000000)
#    
#    
#    new_uvis_boresight_same_to_tgo=(-0.921039704,-0.389467349,0.001023578) #calculated
#    new_uvis_boresight_opposite_to_tgo=(-0.922066875,-0.386977726,0.006396717) #calculated
#    new_so_boresight_same_to_tgo=(-0.920827113,-0.389970833,0.000420369) #calculated
#    new_so_boresight_opposite_to_tgo=(-0.921909199,-0.387358312,0.006079966) #calculated
#    new_lno_boresight_same_to_tgo=(-0.921341439,-0.38875361,0.000764119) #calculated
#    new_lno_boresight_opposite_to_tgo=(-0.921634644,-0.388003786,0.006530345) #calculated
#    new_mir_boresight_to_tgo=(-0.92148,-0.38842,-0.00112) #calculated
#    #convert data to times and boresights using spice
#    time_error=1
#    print("Calculating boresights")
#    boresights_all=find_boresight(epoch_times_all,time_error,old_so_boresight_to_tgo)
#    print("Calculating lat lons")
#    [_,boresight_lons,boresight_lats] = find_rad_lon_lat(boresights_all)
#
#    #convert data to times and boresights using spice
#    if title=="SO ACS Raster 1":
#        time_raster_centre=sp.utc2et("2016APR13-03:25:00.000") #time of s/c pointing to centre
#    elif title=="UVIS ACS Raster 1":
#        time_raster_centre=sp.utc2et("2016APR13-03:25:00.000") #time of s/c pointing to centre
#    elif title=="LNO ACS Raster 1":
#        time_raster_centre=sp.utc2et("2016APR13-04:33:00.000") #time of s/c pointing to centre
#    time_error=1
#    centre_boresight_theoretical=find_boresight([time_raster_centre],time_error,old_so_boresight_to_tgo)
#    
#    uvis_centre_calc_old=find_boresight([time_raster_centre],time_error,old_uvis_boresight_to_tgo)
#    so_centre_calc_old=find_boresight([time_raster_centre],time_error,old_so_boresight_to_tgo)
#    lno_centre_calc_old=find_boresight([time_raster_centre],time_error,old_lno_boresight_to_tgo)
#
#    uvis_centre_calc_same_new=find_boresight([time_raster_centre],time_error,new_uvis_boresight_same_to_tgo)
#    uvis_centre_calc_opposite_new=find_boresight([time_raster_centre],time_error,new_uvis_boresight_opposite_to_tgo)
#    so_centre_calc_same_new=find_boresight([time_raster_centre],time_error,new_so_boresight_same_to_tgo)
#    so_centre_calc_opposite_new=find_boresight([time_raster_centre],time_error,new_so_boresight_opposite_to_tgo)
#    lno_centre_calc_same_new=find_boresight([time_raster_centre],time_error,new_lno_boresight_same_to_tgo)
#    lno_centre_calc_opposite_new=find_boresight([time_raster_centre],time_error,new_lno_boresight_opposite_to_tgo)
#    mir_centre_calc_new=find_boresight([time_raster_centre],time_error,new_mir_boresight_to_tgo)
#
#    [_,centre_boresight_lon,centre_boresight_lat]=find_rad_lon_lat(centre_boresight_theoretical)
#    [_,uvis_centre_calc_old_lon,uvis_centre_calc_old_lat]=find_rad_lon_lat(uvis_centre_calc_old)
#    [_,uvis_centre_calc_new_lon,uvis_centre_calc_new_lat]=find_rad_lon_lat(uvis_centre_calc_same_new)
#
#
#    fig = plt.figure(figsize=(9,9))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(boresights_all[:,0],boresights_all[:,1],boresights_all[:,2], marker='.', linewidth=0, c=marker_colour)
#
#    ax.scatter(uvis_centre_calc_old[:,0],uvis_centre_calc_old[:,1],uvis_centre_calc_old[:,2], marker='o', linewidth=0, c='r')
#    ax.scatter(so_centre_calc_old[:,0],so_centre_calc_old[:,1],so_centre_calc_old[:,2], marker='o', linewidth=0, c='k')
#    ax.scatter(lno_centre_calc_old[:,0],lno_centre_calc_old[:,1],lno_centre_calc_old[:,2], marker='o', linewidth=0, c='c')
#
#    ax.scatter(uvis_centre_calc_same_new[:,0],uvis_centre_calc_same_new[:,1],uvis_centre_calc_same_new[:,2], marker='o', linewidth=0, c='r')
#    ax.scatter(uvis_centre_calc_opposite_new[:,0],uvis_centre_calc_opposite_new[:,1],uvis_centre_calc_opposite_new[:,2], marker='o', linewidth=0, c='r')
#    ax.scatter(so_centre_calc_same_new[:,0],so_centre_calc_same_new[:,1],so_centre_calc_same_new[:,2], marker='o', linewidth=0, c='k')
#    ax.scatter(so_centre_calc_opposite_new[:,0],so_centre_calc_opposite_new[:,1],so_centre_calc_opposite_new[:,2], marker='o', linewidth=0, c='k')
#    ax.scatter(lno_centre_calc_same_new[:,0],lno_centre_calc_same_new[:,1],lno_centre_calc_same_new[:,2], marker='o', linewidth=0, c='c')
#    ax.scatter(lno_centre_calc_opposite_new[:,0],lno_centre_calc_opposite_new[:,1],lno_centre_calc_opposite_new[:,2], marker='o', linewidth=0, c='c')
#    ax.scatter(mir_centre_calc_new[:,0],mir_centre_calc_new[:,1],mir_centre_calc_new[:,2], marker='o', linewidth=0, c='g')
#
#
#    ax.set_title(title+" sum of detector counts during raster scan plotted in solar system coordinates")
#    ax.set_xlabel("X in S.S. reference frame")
#    ax.set_ylabel("Y in S.S. reference frame")
#    ax.set_zlabel("Z in S.S. reference frame")
#
#    plt.figure(figsize=(9,9))
#    plt.scatter(boresight_lons,boresight_lats, marker='.', linewidth=0, c=marker_colour)
#    plt.scatter(centre_boresight_lon,centre_boresight_lat, marker='o', linewidth=0, c='b')
#    plt.scatter(uvis_centre_calc_old_lon,uvis_centre_calc_old_lat, marker='o', linewidth=0, c='k')
##    plt.scatter(uvis_centre_calc_new_lon,uvis_centre_calc_new_lat, marker='o', linewidth=0, c='r')
#    plt.xlabel("Longitude (degrees)")
#    plt.ylabel("Latitude (degrees)")
#    plt.title(title+" sum of detector counts during raster scan plotted in solar system lat/lon")
##    plt.savefig(title+"_sum_detector_counts__lat_lon.png")
#
#if option==11:
#    """try to figure out which way to move the boresights by plotting tgo coordinates in 3d and using model"""
#    hdf5_file.close()
#
#    time=sp.utc2et("2016APR13-03:25:00.000") #ACS raster scan centre
#    cmatrix=sw.ckgp1(time,1)
#    tgo_x_ss = np.dot(np.transpose(cmatrix),(1,0,0))
#    tgo_y_ss = np.dot(np.transpose(cmatrix),(0,1,0))
#    tgo_z_ss = np.dot(np.transpose(cmatrix),(0,0,1))
#    
#    old_so_boresight_to_tgo=(-0.92136,-0.38866,0.00325)
#    old_so_bs_ss = np.dot(np.transpose(cmatrix),old_so_boresight_to_tgo)
#
#    observer="-143" #tgo
#    target="SUN"
#    sun_pos_ss = sw.spkpos1(observer,target,time)
#    sun_distance = la.norm(sun_pos_ss)
#    sun_vector_ss = sun_pos_ss/sun_distance
#    
#    [tgo_x_radius,tgo_x_lon,tgo_x_lat]=sw.reclat(tgo_x_ss)
#    [tgo_y_radius,tgo_y_lon,tgo_y_lat]=sw.reclat(tgo_y_ss)
#    [tgo_z_radius,tgo_z_lon,tgo_z_lat]=sw.reclat(tgo_z_ss)
#    [sun_vector_radius,sun_vector_lon,sun_vector_lat]=sw.reclat(sun_vector_ss)
#    
#    #actual observed sun locations during SO ACS Raster 1: lon=21.40907405, lat=10.84657821
#
#    sun_observed_lon=21.2846285676596+(21.40907405-21.2846285676596)*100
#    sun_observed_lat=10.973230250359+(10.84657821-10.973230250359)*100
#    sun_obs_vector_ss=sw.latrec(1,sun_observed_lon,sun_observed_lat)
#
#    plt.figure(figsize=(9,9))
#    plt.scatter(tgo_x_lon,tgo_x_lat, marker='o', linewidth=0, c='r')
#    plt.scatter(tgo_y_lon,tgo_y_lat, marker='o', linewidth=0, c='g')
#    plt.scatter(tgo_z_lon,tgo_z_lat, marker='o', linewidth=0, c='b')
#    plt.scatter(sun_vector_lon,sun_vector_lat, marker='o', linewidth=0, c='y')
#    plt.scatter(sun_observed_lon,sun_observed_lat, marker='o', linewidth=0, c='orange')
#    plt.plot((0,tgo_x_lon),(0,tgo_x_lat), c='r')
#    plt.plot((0,tgo_y_lon),(0,tgo_y_lat), c='g')
#    plt.plot((0,tgo_z_lon),(0,tgo_z_lat), c='b')
#    plt.plot((0,sun_vector_lon),(0,sun_vector_lat), c='y')
#    plt.plot((0,sun_observed_lon),(0,sun_observed_lat), c='orange')
# 
#    fig = plt.figure(figsize=(9,9))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot((0,tgo_x_ss[0]),(0,tgo_x_ss[1]),(0,tgo_x_ss[2]), c='r')
#    ax.plot((0,tgo_y_ss[0]),(0,tgo_y_ss[1]),(0,tgo_y_ss[2]), c='g')
#    ax.plot((0,tgo_z_ss[0]),(0,tgo_z_ss[1]),(0,tgo_z_ss[2]), c='b')
#    ax.plot((0,sun_vector_ss[0]),(0,sun_vector_ss[1]),(0,sun_vector_ss[2]), c='y')
#    ax.plot((0,old_so_bs_ss[0]/2),(0,old_so_bs_ss[1]/2),(0,old_so_bs_ss[2]/2), c='k')
#    ax.plot((0,sun_obs_vector_ss[0]),(0,sun_obs_vector_ss[1]),(0,sun_obs_vector_ss[2]), c='orange')
#    
#    ax.text(tgo_x_ss[0],tgo_x_ss[1],tgo_x_ss[2],"TGO X")
#    ax.text(tgo_y_ss[0],tgo_y_ss[1],tgo_y_ss[2],"TGO Y")
#    ax.text(tgo_z_ss[0],tgo_z_ss[1],tgo_z_ss[2],"TGO Z")
#    ax.text(sun_vector_ss[0],sun_vector_ss[1],sun_vector_ss[2],"Sun")
#    ax.text(old_so_bs_ss[0]/2,old_so_bs_ss[1]/2,old_so_bs_ss[2]/2,"SO Old BS")
#    ax.text(sun_obs_vector_ss[0],sun_obs_vector_ss[1],sun_obs_vector_ss[2],"Sun Observed")
#    
#    rectan=sw.latrec(1,20,15)
#    out=sw.reclat(rectan)
#    
#    
#if option==16:
#    """extrapolate sun shape to find pixel containing sun centre"""
#    """1. plot vertical slices, define dark and light values and sun width to calculate position of sun on detector and whether fully illuminated
#    2. define crossing time indices to detect when sun crosses detector centre (forward scans) or when sun illumination peaks (for reverse scans) 
#    3. plot straight line through crossing points to define measured sun position
#    4. 
#    """
##    from scipy.interpolate import UnivariateSpline
#    detector_data,_,_ = get_dataset_contents(hdf5_file,"YBins")
#    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
#    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
#    window_top_all = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0]
#    binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
#    hdf5_file.close()
#    
#    if binning==2: #stretch array
#        detector_data=np.repeat(detector_data,2,axis=1)
#    window_size=16*binning
#
#    epoch_times_all=convert_hdf5_time_to_spice_utc(time_data_all,date_data_all)
#    time_error=1    
#    nframes=detector_data.shape[0]
#    
#    fine_time_step=0.1
#    if title=="LNO ACS Raster 1":
#        illuminated_value=150000
#        dark_value=100000
#        half_max_value=100000
#        spectral_line_index=228
##        sun_width=22
##        smoothing=100
#        sun_width=25
#        smoothing=200
#        chosen_window_top=144
#        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical
#        boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)
#    elif title=="SO ACS Raster 1" or title=="SO ACS Raster 2":
#        illuminated_value=100000
#        dark_value=50000
#        half_max_value=75000
#        spectral_line_index=228
##        sun_width=22
##        smoothing=100
#        sun_width=25
#        smoothing=200
#        chosen_window_top=128
#        boresight_to_tgo=(-0.92136,-0.38866,0.00325) #theoretical
#        boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)
#    elif title=="SO Raster 4A" or title=="SO Raster 4B":
##        illuminated_value=100000
##        dark_value=50000
##        half_max_value=75000
#        illuminated_value=250000
#        dark_value=50000
#        half_max_value=150000
#        spectral_line_index=228
#        smoothing=100
#
#        sun_width=25
#        chosen_window_top=128
#        boresight_to_tgo=(-0.92191,-0.38736,0.00608) #theoretical
#        boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)
#
#        fine_position_step=fine_time_step/3.0
#        
#        if title=="SO Raster 4A":       
#            time_indices=[[0,4],[5,8],[9,13],[14,18],[19,22]]
#            centre_indices=[56,57]
#            reverse=False
#        elif title=="SO Raster 4B":
#            reverse=True
#            time_indices=[[24,36],[37,49],[51,62],[63,75],[76,88],[89,101]]
#            centre_indices=[55,56]
#
#    elif title=="LNO Raster 4A" or title=="LNO Raster 4B":
##        illuminated_value=100000
##        dark_value=50000
##        half_max_value=75000
#        illuminated_value=300000
#        dark_value=50000
#        half_max_value=200000
#        spectral_line_index=228
#        smoothing=100
#
#        sun_width=25
#        chosen_window_top=128
#        boresight_to_tgo=(-0.92163,-0.38800,0.00653) #theoretical
#        boresights_all=find_boresight(epoch_times_all,time_error,boresight_to_tgo)
#
#        fine_position_step=0.08/2.0
#        
#        if title=="LNO Raster 4A":       
#            time_indices=[[0,17],[18,48],[50,73]]
#            centre_indices=[111,112]
#            reverse=False
#        elif title=="LNO Raster 4B":
#            reverse=True
#            time_indices=[[70,90],[95,125],[130,160],[165,195]]
#            centre_indices=[111,112]
#
#    
#    detector_vlines=detector_data[:,:,spectral_line_index]
#    
#    centres=[]
#    boresights=[]
#    times=[]
#    plt.figure(figsize=(9,9))
#    for frame_index in range(nframes):
#        detector_vline = detector_vlines[frame_index,:]
#        vpixel_number = np.asfarray(range(window_top_all[frame_index],window_top_all[frame_index]+window_size))
#        if detector_vline[0]<dark_value and detector_vline[window_size-1]>illuminated_value: #if rising
#            """interpolate to find pixel value where rising edge is seen"""
#            vpix1=max(vpixel_number[np.where(detector_vline<half_max_value)[0]])
#            pixval1=detector_vline[int(vpix1-window_top_all[frame_index])]
#            vpix2=min(vpixel_number[np.where(detector_vline>half_max_value)[0]])
#            pixval2=detector_vline[int(vpix2-window_top_all[frame_index])]
#            vpix_interp = np.interp(half_max_value,[pixval1,pixval2],[vpix1,vpix2])
##            print(vpix_interp
#            centres.append(vpix_interp+sun_width/2)
#            boresights.append(boresights_all[frame_index])
#            times.append(epoch_times_all[frame_index])
#            plt.scatter(vpix_interp,half_max_value)
#            print("rising %i" %frame_index)
#            plt.plot(vpixel_number,detector_vline)
#        elif detector_vline[0]>illuminated_value and detector_vline[window_size-1]<dark_value: #if falling
#            vpix1=max(vpixel_number[np.where(detector_vline>half_max_value)[0]])
#            pixval1=detector_vline[int(vpix1-window_top_all[frame_index])]
#            vpix2=min(vpixel_number[np.where(detector_vline<half_max_value)[0]])
#            pixval2=detector_vline[int(vpix2-window_top_all[frame_index])]
#            vpix_interp = np.interp(half_max_value,[pixval2,pixval1],[vpix2,vpix1])
##            print(vpix_interp
#            centres.append(vpix_interp-sun_width/2)
#            boresights.append(boresights_all[frame_index])
#            times.append(epoch_times_all[frame_index])
#            plt.scatter(vpix_interp,half_max_value)
#            print("falling %i" %frame_index)
#            plt.plot(vpixel_number,detector_vline)
#    plt.xlabel("Vertical Pixel Number")
#    plt.ylabel("Signal ADU for horizontal pixel %i" %spectral_line_index)
#    plt.title(title+": Vertical detector slices where Sun is seen")    
#    if save_figs: plt.savefig(title+"_vertical_detector_slices_where_Sun_is_seen.png")
#    
#    #find indices where centre of detector is
#    meas_indices=[]
#    for index,window_top in enumerate(window_top_all):
#        if window_top==chosen_window_top:
#            meas_indices.append(index)
#    detector_sum=np.sum(detector_data[:,detector_centre-chosen_window_top,:],axis=1)
#
#    marker_colour=np.log(1+detector_sum[meas_indices]-min(detector_sum[meas_indices]))
#    
#
#    if not reverse:
#        fine_centres=[]
#        fine_times=[]
#        for time_groups in range(len(time_indices)):
#            m,x=np.polyfit(times[time_indices[time_groups][0]:time_indices[time_groups][1]],centres[time_indices[time_groups][0]:time_indices[time_groups][1]],1)
#            fine_time=np.arange(times[time_indices[time_groups][0]],times[time_indices[time_groups][1]],fine_time_step)
#            fine_centres.extend(fine_time*m + x)
#            fine_times.extend(fine_time)
#
##        fine_centres=30.0*np.sin(fine_times/63.0-12.5)+128.0
#        fine_centres = np.asfarray(fine_centres)
#        fine_times = np.asfarray(fine_times)
#
#        detector_times=fine_times[np.where((fine_centres>detector_centre-fine_position_step) & (fine_centres<detector_centre+fine_position_step))[0]]
#        plt.figure(figsize=(9,9))
##        plt.plot(np.asfarray(times)[np.abs(np.asfarray(centres)-122.5)>0.5],np.asfarray(centres)[np.abs(np.asfarray(centres)-122.5)>0.5],'*')
#        plt.plot(times,centres,'*')
#        plt.plot(fine_times,fine_centres)
#        plt.plot([times[0],times[-1]],[detector_centre,detector_centre])
#        plt.plot([detector_times,detector_times],[min(centres),max(centres)])
#        plt.xlabel("Ephemeris time")
#        plt.ylabel("Calculated pixel row where sun is centred")
#        plt.title(title+": Position of sun centre on detector vs time")
#        if save_figs: plt.savefig(title+"_position_of_sun_centre_on_detector_vs_time.png")
#    
#        previous_time=0
#        crossing_times=[]
#        for time_loop in range(len(detector_times)):
#            if (detector_times[time_loop]-previous_time)>5: #remove values that are very close together
#                print("Calculated peak sun on detector row %i at %s" %(detector_centre,sw.et2utc(detector_times[time_loop])))
#                plt.text(detector_times[time_loop],110.0+time_loop,"%s" %sw.et2utc(detector_times[time_loop]))
#                crossing_times.append(detector_times[time_loop])
#            previous_time=detector_times[time_loop]
#
#    if reverse:
##        spl = UnivariateSpline(times,centres)
##        spl.set_smoothing_factor(smoothing)
##        fine_times=np.arange(times[0],times[-1],fine_time_step)
##        fine_centres=spl(fine_times)
#
#        crossing_times=[]
#        plt.figure(figsize=(9,9))
#        plt.title(title+" sum of detector row %i during raster scan" %chosen_window_top)
#        plt.xlabel("Time")
#        plt.ylabel("Sum of signal ADU")
#        for time_groups in range(len(time_indices)):
##            plt.figure(figsize=(9,9))
##            plt.title(title+" detector row %i for vertical lines during raster scan" %chosen_window_top)
##            plt.xlabel("Vertical pixel")
##            plt.ylabel("Signal ADU")
##            for plot_loop in range((detector_vlines[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]],:]).shape[0]):
##                plt.plot(np.arange(16), detector_vlines[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]],:][plot_loop,:], marker='o', linewidth=0)#,label=sw.et2utc(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]][plot_loop]))
##            plt.legend()
##            if save_figs: plt.savefig(title+"_detector_row_%i_for_vertical_lines_during_raster_scan.png" %chosen_window_top)
# 
#            """for non-gaussian shapes
#            spl = UnivariateSpline(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),detector_sum[meas_indices[time_indices[0][0]:time_indices[0][1]]])
#            spl.set_smoothing_factor(0.001)
#            fine_times=np.arange(epoch_times_all[meas_indices[time_indices[0][0]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),epoch_times_all[meas_indices[time_indices[0][1]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),fine_time_step)
#            fine_centres=spl(fine_times)
#    
#            m2,m1,x = np.polyfit(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),detector_sum[meas_indices[time_indices[0][0]:time_indices[0][1]]],2)
#            fine_times=np.arange(epoch_times_all[meas_indices[time_indices[0][0]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),epoch_times_all[meas_indices[time_indices[0][1]]]-np.mean(epoch_times_all[meas_indices[time_indices[0][0]:time_indices[0][1]]]),fine_time_step)
#            fine_centres=fine_times**2 * m2 + fine_times * m1 + x
#            """
#            
#            fine_times=np.arange(epoch_times_all[meas_indices[time_indices[time_groups][0]]],epoch_times_all[meas_indices[time_indices[time_groups][1]]],fine_time_step)
#            #Gaussian function
#            def gaussian(x, a, x0, sigma):
#                return a*np.exp(-(x-x0)**2/(2*sigma**2))
#            from scipy.optimize import curve_fit
#            mean = np.mean(epoch_times_all[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]])
#            sigma = 100
#            popt, pcov = curve_fit(gaussian, epoch_times_all[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]], detector_sum[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]], p0 = [np.mean(detector_sum[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]]), mean, sigma])
#            fine_centres =  gaussian(fine_times, *popt)
#            crossing_times.append(popt[1])
#            
#            plt.plot(epoch_times_all[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]], detector_sum[meas_indices[time_indices[time_groups][0]:time_indices[time_groups][1]]], marker='o', linewidth=0)
#            plt.plot(fine_times, fine_centres)
#            plt.text(popt[1],popt[0],sw.et2utc(popt[1]))
#            if save_figs: plt.savefig(title+"_sum_of_detector_row_%i_during_raster_scan_gaussian.png" %chosen_window_top)
#
#
#
#    
#    observer="-143" #tgo
#    target="SUN"
#    angles=np.zeros(nframes)
#    sun_pointing_vector=np.zeros((nframes,3))
#    
##old matlab version
##    for time_loop in range(nframes):
##        relative_sun_position=sw.spkpos1(observer,target,epoch_times_all[time_loop])
##        sun_distance = la.norm(relative_sun_position)
##        sun_pointing_vector = relative_sun_position / sun_distance
#    relative_sun_position=sw.spkposx(observer,target,epoch_times_all)
#    for time_loop in range(nframes):
#        sun_distance = la.norm(relative_sun_position[time_loop])
#        sun_pointing_vector[time_loop,:] = relative_sun_position[time_loop] / sun_distance #normalise
#
#        angles[time_loop] = py_ang(boresights_all[time_loop],sun_pointing_vector[time_loop,:]) * 180 * 60 / np.pi
#        
#    print("Raster %f arcmins from centre at %s" %(angles[angles.argmin()],sw.et2utc(epoch_times_all[angles.argmin()])))
#    print("Raster started at %s" %sw.et2utc(epoch_times_all[0]))
#    print("Raster ended at %s" %sw.et2utc(epoch_times_all[-1]))
#
#
#    plt.figure(figsize=(9,9))
#    plt.plot(angles)
#
##    plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
##    plt.xlabel("Approx time after pre-cooling ends (seconds)")
##    plt.title(title)
##    plt.yscale("log")
#    
##    np.savetxt(title+".txt", np.transpose(np.asfarray([time,sum_centre_all])), delimiter=",")
##    plt.savefig(title+"_intensity_versus_time_raster_scan_log.png")
#    
#
#    fig = plt.figure(figsize=(9,9))
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(boresights_all[meas_indices,0],boresights_all[meas_indices,1],boresights_all[meas_indices,2], c=marker_colour, marker='o', linewidth=0)
#    ax.text(boresights_all[0,0],boresights_all[0,1],boresights_all[0,2], "Raster Start")
#    ax.text(boresights_all[-1,0],boresights_all[-1,1],boresights_all[-1,2], "Raster End")
#    centre_boresight=find_boresight([epoch_times_all[angles.argmin()]],time_error,boresight_to_tgo)
#    crossing_boresights=find_boresight(crossing_times,time_error,boresight_to_tgo)
#    ax.scatter(crossing_boresights[:,0],crossing_boresights[:,1],crossing_boresights[:,2], marker='s', linewidth=0, c='b')
#    ax.scatter(centre_boresight[:,0],centre_boresight[:,1],centre_boresight[:,2], marker='*', s=100, linewidth=0, c='k')
#    ax.set_title(title+" sum of detector row %i during raster scan in J2000 coordinates" %chosen_window_top)
#    ax.set_xlabel("X in S.S. reference frame")
#    ax.set_ylabel("Y in S.S. reference frame")
#    ax.set_zlabel("Z in S.S. reference frame")
#    ax.azim=-120
#    ax.elev=1
#    if save_figs: plt.savefig(title+"_sum_of_detector_row_%i_during_raster_scan_in_J2000_coordinates.png" %chosen_window_top)
#
#    boresight_lat_lons=sw.reclatx(boresights_all[meas_indices,:])
#    centre_lat_lons=sw.reclatx(centre_boresight)
#    crossing_lat_lons=sw.reclatx(crossing_boresights)
#    sun_lat_lons=sw.reclatx(sun_pointing_vector)
#    
#    print(crossing_lat_lons)
#
#    plt.figure(num=100,figsize=(9,9))
#    plt.scatter(boresight_lat_lons[:,1],boresight_lat_lons[:,2], c=marker_colour, marker='o', linewidth=0)
#    plt.scatter(sun_lat_lons[:,1],sun_lat_lons[:,2], marker='*', s=100, linewidth=0, c='r')
#    plt.scatter(sun_lat_lons[angles.argmin(),1],sun_lat_lons[angles.argmin(),2], marker='*', s=100, linewidth=0, c='g')
#    plt.scatter(centre_lat_lons[0][1],centre_lat_lons[0][2], marker='*', s=100, linewidth=0, c='k')
#    plt.scatter(crossing_lat_lons[:,1],crossing_lat_lons[:,2], marker='s', linewidth=0, c='b')
#    y1=crossing_lat_lons[0,2]
#    y2=crossing_lat_lons[-1,2]
#    m,dy=np.polyfit(crossing_lat_lons[:,2],crossing_lat_lons[:,1],1)
#    plt.plot([y1*m+dy,y2*m+dy],[y1,y2])
#    plt.plot([boresight_lat_lons[centre_indices[0],1],boresight_lat_lons[centre_indices[1],1]], \
#             [boresight_lat_lons[centre_indices[0],2],boresight_lat_lons[centre_indices[1],2]],'b')
#        
#    plt.title(title+" sum of detector row %i during raster scan in lat lons" %chosen_window_top)
#    plt.xlabel("Longitude (degrees)")
#    plt.ylabel("Latitude (degrees)")
#    if save_figs: plt.savefig(title+"_sum_of_detector_row_%i_during_raster_scan_in_lat_lons.png" %chosen_window_top)
#
#    print("_calculated_lon_lats=[%f,%f]" %(sun_lat_lons[angles.argmin(),1],sun_lat_lons[angles.argmin(),2]))
#    print("_sun_time=%f" %epoch_times_all[angles.argmin()])
#
#
#    """add calculated values here"""
#    if title=="SO Raster 4A" or title=="SO Raster 4B":
#        #left right and up down
#        so_measured_lon_lats = [73.3200,26.3243] #found from measurement data
#        so_calculated_lon_lats = [73.376989,26.335803]
#        so_sun_time=519304568.137519
#        
#        lon_lat_misalignment = [so_calculated_lon_lats[0] - so_measured_lon_lats[0],so_calculated_lon_lats[1] - so_measured_lon_lats[1]]
#        print("Misalignment= %0.2f arcmins lon, %0.2f arcmins lat" %(lon_lat_misalignment[0] * 60.0,lon_lat_misalignment[1] * 60.0))
#        
#        so_new_lon_lat = [so_calculated_lon_lats[0] + lon_lat_misalignment[0],so_calculated_lon_lats[1] + lon_lat_misalignment[1]]
#        
#        so_new_vector = sw.latrec(1.00000, so_new_lon_lat[0],so_new_lon_lat[1])
#        print("SO new vector= "+"%0.7f "*3 %(so_new_vector[0],so_new_vector[1],so_new_vector[2]))
#        so_cmatrix = sw.ckgp1(so_sun_time, 1)
#        so_new_boresight=np.dot(so_cmatrix,so_new_vector)
#        print("SO new boresight= "+"%0.7f "*3 %(so_new_boresight[0],so_new_boresight[1],so_new_boresight[2]))
#        
#        #now reverse calcualation to check numbers
#        so_vector_recalc = find_boresight([so_sun_time,so_sun_time+1],1.00000,so_new_boresight) #use two boresights to run functions correctly
#        so_lon_lat_recalc = sw.reclatx(so_vector_recalc)[0,:]
#        ax.scatter(so_vector_recalc[0,0],so_vector_recalc[0,1],so_vector_recalc[0,2],marker="*",s=150,c="c")
#        plt.scatter(so_lon_lat_recalc[1],so_lon_lat_recalc[2],marker="*",s=150,c="c")
#    elif title=="LNO Raster 4A" or title=="LNO Raster 4B":
#        lno_measured_lon_lats = [73.3754,26.3505] #found from measurement data
#        lno_calculated_lon_lats=[73.403552,26.338780]
#        lno_sun_time=519307568.191518
#
#        lon_lat_misalignment = [lno_calculated_lon_lats[0] - lno_measured_lon_lats[0],lno_calculated_lon_lats[1] - lno_measured_lon_lats[1]]
#        print("Misalignment= %0.2f arcmins lon, %0.2f arcmins lat" %(lon_lat_misalignment[0] * 60.0,lon_lat_misalignment[1] * 60.0))
#        
#        lno_new_lon_lat = [lno_calculated_lon_lats[0] + lon_lat_misalignment[0],lno_calculated_lon_lats[1] + lon_lat_misalignment[1]]
#        
#        lno_new_vector = sw.latrec(1.00000, lno_new_lon_lat[0],lno_new_lon_lat[1])
#        print("LNO new vector= "+"%0.7f "*3 %(lno_new_vector[0],lno_new_vector[1],lno_new_vector[2]))
#        lno_cmatrix = sw.ckgp1(lno_sun_time, 1)
#        lno_new_boresight=np.dot(lno_cmatrix,lno_new_vector)
#        print("LNO new boresight= "+"%0.7f "*3 %(lno_new_boresight[0],lno_new_boresight[1],lno_new_boresight[2]))
#        
#        #now reverse calcualation to check numbers
#        lno_vector_recalc = find_boresight([lno_sun_time,lno_sun_time+1],1.00000,lno_new_boresight) #use two boresights to run functions correctly
#        lno_lon_lat_recalc = sw.reclatx(lno_vector_recalc)[0,:]
#        ax.scatter(lno_vector_recalc[0,0],lno_vector_recalc[0,1],lno_vector_recalc[0,2],marker="*",s=150,c="c")
#        plt.scatter(lno_lon_lat_recalc[1],lno_lon_lat_recalc[2],marker="*",s=150,c="c")
#
#    
#
#    
#if option==17:
#    """display frame"""
#    detector_data_all = get_dataset_contents(hdf5_file,"YBins")[0] #get data
#    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
#    hdf5_file.close()
#    
#    chosen_frame=100
#    
#    detector_data_all[chosen_frame,12,:]=interpolate_bad_pixel(detector_data_all[chosen_frame,12,:],84)
#    detector_data_all[chosen_frame,12,:]=interpolate_bad_pixel(detector_data_all[chosen_frame,12,:],269)
#    detector_data_all[chosen_frame,12,:]=interpolate_bad_pixel(detector_data_all[chosen_frame,12,:],199)
#    detector_data_all[chosen_frame,8,:]=interpolate_bad_pixel(detector_data_all[chosen_frame,8,:],256)
#
#    calc_order = find_order(channel,aotf_freq_all[chosen_frame])
#    wavenumbers = spectral_calibration_simple(channel,calc_order)
#
#
#    plt.figure(figsize=(10,8))
#    plt.imshow(detector_data_all[chosen_frame,6:20,:],interpolation='none',cmap=plt.cm.gray, aspect=2.4, extent=[wavenumbers[0],wavenumbers[-1],6,20])
#    plt.colorbar()
#    plt.title("Solar spectrum taken on 15th April 2016")
##    plt.xlabel("Horizontal pixel number (spectral direction)")
#    plt.xlabel("Wavenumbers (cm-1)")
#    plt.ylabel("Vertical pixel number (spatial direction)")
#    if save_figs: plt.savefig("Solar_spectrum_taken_on_14th_April_2016.png", dpi=400)
#
#
#if option==18:
#    """plot straylight intensity vs time"""
#    if channel=="so" or channel=="lno":
#        detector_data_all,_,_ = get_dataset_contents(hdf5_file,"YBins")
##        exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
#        binning = get_dataset_contents(hdf5_file,"BINNING")[0][0]+1
#        window_top = get_dataset_contents(hdf5_file,"WINDOW_TOP")[0][0]
#        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
#        time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0][:,0]
#        date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0][:,0]
#        hdf5_file.close()
#
#        frame_indices=[]
##        exponent_all=[]
#        chosen_aotf_freq=aotf_freq_all[0]
#        dark_indices=[]
#        times=[]
#        dates=[]
#        for frame_loop in range(detector_data_all.shape[0]): #loop through all frames
#            if aotf_freq_all[frame_loop]==0:
#                dark_indices.append(frame_loop)
#            if aotf_freq_all[frame_loop]==chosen_aotf_freq:
#                frame_indices.append(frame_loop)
#                times.append(time_data_all[frame_loop])
#                dates.append(date_data_all[frame_loop])
#                
##        print("Calculating times"
#        epoch_times=convert_hdf5_time_to_spice_utc(times,dates)
#        
#        sum_corrected_frame_centre=[]
#        dark_indices_before=[]
#        dark_indices_after=[]
#        corrected_frames=np.zeros((len(frame_indices),detector_data_all.shape[1],detector_data_all.shape[2]))
#        for frame_loop,light_frame_index in enumerate(frame_indices): #loop through chosen light frames
#            found_dark=bisect.bisect_left(dark_indices, light_frame_index) #find index of next dark frame
#            dark_index_after=dark_indices[found_dark] #find index of next dark frame
#            dark_indices_after.append(dark_index_after)
#            if found_dark==0: #if first dark frame
#                dark_index_before=dark_index_after #set index to same value
#            else:
#                dark_index_before=dark_indices[found_dark-1]
#            dark_indices_before.append(dark_index_before)
#            #subtract mean of dark frames on either side from light frame
#            corrected_frames[frame_loop,:,:]=detector_data_all[light_frame_index,:,:] - np.mean([detector_data_all[dark_index_before,:,:],detector_data_all[dark_index_after,:,:]], axis=0)
#            sum_corrected_frame_centre.append(np.mean(corrected_frames[frame_loop,1:2,200:250]))
#        
##        plt.figure(figsize=(10,8))
##        plt.imshow(detector_data_all[64,:,:], aspect=binning, interpolation='none')
##        plt.colorbar()
##        
##        plt.figure(figsize=(10,8))
##        plt.imshow(corrected_frames[64,:,:], aspect=binning, interpolation='none')
##        plt.colorbar()
#        
#        
#        time=np.arange(len(frame_indices))
#        
#        plt.figure(figsize=(10,8))
#        plt.plot(time,sum_corrected_frame_centre)
##        plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
#        plt.ylabel("Sum signal ADU for pixel bin 1:2,columns 200:250")
#        plt.xlabel("Approx time after pre-cooling ends (seconds)")
#        plt.title(title+": after background subtraction")
#        plt.yscale("log")
#        if save_figs: plt.savefig(checkout+" "+title+"_intensity_versus_time_log.png")
#        
##        np.savetxt(title+".txt", np.transpose(np.asfarray([time,sum_centre_all])), delimiter=",")
#    
#        plt.figure(figsize=(10,8))
#        plt.plot(time,sum_corrected_frame_centre)
##        plt.ylabel("Sum signal ADU for pixels %i:%i,220:236" %((detector_centre-4),(detector_centre+4)))
#        plt.ylabel("Sum signal ADU for pixel bin 1:2,columns 200:250")
#        plt.xlabel("Approx time after pre-cooling ends (seconds)")
#        plt.title(title+": after background subtraction")
#        if save_figs: plt.savefig(checkout+" "+title+"_intensity_versus_time.png")
#
#        marker_colour=np.log(1+np.asfarray(sum_corrected_frame_centre)-min(np.asfarray(sum_corrected_frame_centre)))  
#        
#        old_so_boresight_to_tgo=(-0.92136,-0.38866,0.00325) #define so boresight in tgo reference frame
#        old_lno_boresight_to_tgo=(-0.92126,-0.38890,0.00368)
#        old_uvis_boresight_to_tgo=(-0.921550000000000,-0.388220000000000,0.003710000000000)
#    
##        print("Calculating boresights"
#        time_error=1
#        boresights_all=find_boresight(epoch_times,time_error,old_so_boresight_to_tgo)
#    
#        fig = plt.figure(figsize=(9,9))
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(boresights_all[:,0],boresights_all[:,1],boresights_all[:,2], marker='.', linewidth=0, c=marker_colour)
#        ax.set_title(title+" background subtracted ADU bins 1:2,columns 200:250 in J2000")
#        ax.set_xlabel("X in S.S. reference frame")
#        ax.set_ylabel("Y in S.S. reference frame")
#        ax.set_zlabel("Z in S.S. reference frame")
#        ax.azim=-130
#        ax.elev=-0
#        if save_figs: plt.savefig(checkout+" "+title+"_background_subtracted_ADU_bins_1-2_columns_200-250_in_J2000.png")
#
##        print("Calculating lat lons"
#        [_,boresight_lons,boresight_lats] = find_rad_lon_lat(boresights_all)
#
#        plt.figure(figsize=(9,9))
#        plt.scatter(boresight_lons,boresight_lats, marker='.', linewidth=0, c=marker_colour)
#        plt.text(boresight_lons[0],boresight_lats[0],"Raster Start")
#        plt.text(boresight_lons[-1],boresight_lats[-1],"Raster End")
#
#        plt.xlabel("Longitude (degrees)")
#        plt.ylabel("Latitude (degrees)")
#        plt.title(title+" background subtracted ADU bins 1:2,columns 200:250 in J2000")
#        if save_figs: plt.savefig(checkout+" "+title+"background_subtracted_ADU_bins_1-2_columns_200-250_solar_lat_lons.png")
#
#if option==19:
#    """print angular differences between all FOV centres"""
#    lno_old_boresight=[-0.92163,-0.38800,0.00653] #pre MCC
#    so_old_boresight=[-0.92191,-0.38736,0.00608] #pre MCC
#    lno_new_boresight=[ -0.9214767, -0.3883830, 0.0062766 ] #post MCC
#    so_new_boresight=[ -0.9215576, -0.3881924, 0.0061777 ] #post MCC
#    uvis_new_boresight=[-0.92207,-0.38696,0.00643 ] #UVIS no change
#    
#    print("lno_new_vs_uvis = %0.7f" %(py_ang(lno_new_boresight,uvis_new_boresight) * 180.0 / np.pi * 60.0))
#    print("lno_new_vs_so_new = %0.7f" %(py_ang(lno_new_boresight,so_new_boresight) * 180.0 / np.pi * 60.0))
#    print("so_new_vs_uvis = %0.7f" %(py_ang(so_new_boresight,uvis_new_boresight) * 180.0 / np.pi * 60.0))
#    print("lno_new_vs_lno_old = %0.7f" %(py_ang(lno_new_boresight,lno_old_boresight) * 180.0 / np.pi * 60.0))
#    print("so_new_vs_so_old = %0.7f" %(py_ang(so_new_boresight,so_old_boresight) * 180.0 / np.pi * 60.0))
#    print("lno_old_vs_uvis = %0.7f" %(py_ang(lno_old_boresight,uvis_new_boresight) * 180.0 / np.pi * 60.0))
#    print("so_old_vs_uvis = %0.7f" %(py_ang(so_old_boresight,uvis_new_boresight) * 180.0 / np.pi * 60.0))
#
#    boresight_vector=lno_new_boresight
#    
#    time=519304568.137519
#    cmatrix = sw.ckgp1(time, 1)
#    vector=np.dot(cmatrix,boresight_vector)
#    lon_lat = sw.reclat(vector)
#    
#    
#    
#    
#if option==23:
#    """make animations"""
#    from scipy.signal import argrelextrema
##    from scipy.interpolate import UnivariateSpline as spline
#    from scipy.interpolate import interp1d
#    from scipy.signal import savgol_filter as sg
#    """animate frames or plots"""
#    
##    what_to_animate="frames"
#    what_to_animate="lines"
#    
#    variable_changing="aotf"
#    
#    sum_vertically=True
##    sum_vertically=False
#    sum_frames=True
##    sum_frames=False
##    animate=True
#    animate=False
#    
#    line_number=11
#    nframes_to_average=5
#
#    """get data from file, plot single detector frame"""
#    if what_to_animate=="frames":
#        detector_data = get_dataset_contents(hdf5_file,"Y")[0][0:116,:,:] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#        
#    elif what_to_animate=="lines":
#        if title=="LNO Fullscan":
#            detector_data_sun = get_dataset_contents(hdf5_file,"YBins")[0][[59,174,289,519,404],:,:] #get YBins data (24 lines of spectra) from file. Data has 3 dimensions: time x line x spectrum
#            exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
#            for index,frame in enumerate(detector_data_sun):
#                detector_data_sun[index,:,:] = frame[:,:]*2**exponent[index]
#            detector_data_sun = np.sum(detector_data_sun, axis=(0,1))
#            stop()
#
#        detector_data = get_dataset_contents(hdf5_file,"YBins")[0][0:116,:,:] #get YBins data (24 lines of spectra) from file. Data has 3 dimensions: time x line x spectrum
#    
#    exponent,_,_ = get_dataset_contents(hdf5_file,"EXPONENT")
#    for index,frame in enumerate(detector_data):
#        detector_data[index,:,:] = frame[:,:]*2**exponent[index]
#    
#    if variable_changing=="aotf":
#        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTF_FREQUENCY")[0]
#        orders_all = [find_order(channel,freq,silent=True) for index,freq in enumerate(aotf_freq_all)]
#        wavenumbers = [spectral_calibration_simple(channel,order,silent=True) for order in orders_all]
#    hdf5_file.close()
#
#            
#    if sum_vertically and what_to_animate != "frames":
#        line_number=0
#        detector_data = np.sum(detector_data, axis=1)
#
#    
#    if sum_frames and what_to_animate=="lines":
#        detector_data_averaged = np.zeros((int(np.ceil(len(detector_data)/nframes_to_average)),320))
#        for index in range(int(np.ceil(len(detector_data)/nframes_to_average))):
#            index_start=index*nframes_to_average
#            index_end=(index+1)*nframes_to_average
#            detector_data_averaged[index,:] = np.mean(detector_data[index_start:index_end,:], axis=0)
##            plt.plot(wavenumbers[0],detector_data_1_frame, color="k")
##            plt.xlabel("Wavenumber cm-1")
##            plt.ylabel("Detector signal ADU")
##            plt.title("%s: Mean of all frames" %(title))
#
#        detector_data = detector_data_averaged
#    n_frames = len(detector_data)
#    max_value=np.nanmax(detector_data)
#    fig=plt.figure(1, figsize=(10,8))
#    num=0
#    
#    if what_to_animate=="frames": 
#        plot = plt.imshow(detector_data[num,:,:], vmin=0, vmax=max_value, animated=True, interpolation=None, extent=[min(wavenumbers[num]),max(wavenumbers[num]),255,0], aspect=0.1)
#        plotbar = plt.colorbar()
#        plt.xlabel("Wavenumber cm-1")
#        plt.ylabel("Detector vertical (spatial) direction")
#    if what_to_animate=="lines":
#        plt.ylim((0,max_value))
#        if sum_vertically:
#            if animate:
#                plot, = plt.plot(wavenumbers[num],detector_data[num,:], color="k", animated=True)
#            else:
#                plot, = plt.plot(wavenumbers[num],detector_data[num,:], color="k")
#        else:
#            if animate:
#                plot, = plt.plot(wavenumbers[num],detector_data[num,line_number,:], color="k", animated=True)
#            else:
#                plot, = plt.plot(wavenumbers[num],detector_data[num,line_number,:], color="k")
#        
#    if variable_changing=="aotf":
#        plottitle = plt.title("%s: Frame %i AOTF %0.0f kHz" %(title,num,aotf_freq_all[num]))
#
#    if animate:
#        def updatefig(num): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
#            global plot,plottitle#,detector_data,variable_changing,what_to_animate,line_number,sum_vertically
#            if np.mod(num,50)==0:
#                print(num)
#            if what_to_animate=="frames": 
#                plot.set_array(detector_data[num,:,:])
#            elif what_to_animate=="lines": 
#                if sum_vertically:
#                    plot.set_data(wavenumbers[num], detector_data[num,:])
#                else:
#                    plot.set_data(wavenumbers[num], detector_data[num,line_number,:])
#            if variable_changing=="aotf":
#                plottitle.set_text("%s: Frame %i Exponent %i AOTF %0.0f kHz" %(title,num,exponent[num],aotf_freq_all[num]))
#            return plot,
#                
#        ani = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=50, blit=True)
#        if save_figs: ani.save(title+"_detector_%s.mp4" %what_to_animate, fps=20, extra_args=['-vcodec', 'libx264'])
#        plt.show()
#        
#   
#    sg_lengths = [5,11]    
#    interp_kinds = ["cubic","linear","slinear"]
##    y 
#    
#    plt.figure(2, figsize=(10,8))
#    plt.xlabel("Wavenumber cm-1")
#    plt.ylabel("Normalised radiance")
##    plt.title("%s: Mean of %i frames" %(title,nframes_to_average))
#    plt.title("Typical nadir spectrum from LNO using various fitting types")
#    
#    plt.figure(3, figsize=(10,8))
#    plt.xlabel("Wavenumber cm-1")
#    plt.ylabel("Signal ADUs")
##    plt.title("%s: Mean of %i frames" %(title,nframes_to_average))
#    plt.title("Fitting points to nadir curves")
#    
#
#    for sg_length in sg_lengths:
#        for interp_kind in interp_kinds:
#    
#            #remove very noisy wings from spectrum
#            detector_line = detector_data[0,:]
#            wavenumber_range = wavenumbers[0]
#            adu_cutoff = 3000
#            detector_line_centre = detector_line[np.where(detector_line[:]>adu_cutoff)[0]]
#            wavenumber_centre = wavenumber_range[np.where(detector_line[:]>adu_cutoff)[0]]
#            #smooth data, remove gross variations
#            detector_line_presmooth = sg(detector_line_centre, window_length=sg_length, polyorder=3)
#            #then find local maxima and fit to these
#            loc_max_indices = list(argrelextrema(detector_line_presmooth, np.greater)[0]) #find indices of local maxima
#           
#            plt.figure(3)
#            plt.scatter(wavenumber_centre[loc_max_indices],detector_line_presmooth[loc_max_indices])
#            
#        #    spl = spline(wavenumbers[0][loc_max_indices],detector_data[0,loc_max_indices], w=detector_data[0,loc_max_indices])
#        #    y_new = spl(wavenumbers[0])
#        #    plt.plot(wavenumbers[0],y_new)
#            
#            spl = interp1d(wavenumber_centre[loc_max_indices],detector_line_centre[loc_max_indices], kind=interp_kind)
#            wavenumbers_bounded = wavenumber_centre[min(loc_max_indices):max(loc_max_indices)]    
#            detector_data_bounded = detector_line_centre[min(loc_max_indices):max(loc_max_indices)]
#        
#            y_new = spl(wavenumbers_bounded)
#            plt.plot(wavenumbers_bounded,y_new, label="%s-%s" %(interp_kind,sg_length))
#            
#            y_above = y_new[:]
#            for index in range(len(y_above)):
#                if y_above[index] < detector_data_bounded[index]:
#                    y_above[index] = detector_data_bounded[index]
#            plt.plot(wavenumbers_bounded,y_above)
#            
#            plt.figure(2)
#            plt.plot(wavenumbers_bounded,detector_data_bounded/y_above)
#    
#    plt.legend()
#
#    plt.figure(4, figsize=(10,8))
#    for index in range(len(detector_data[:,0]))[1::]: #first spectrum bad
#        plt.plot(wavenumbers[0],detector_data[index,:]/detector_data_sun)
#    plt.xlabel("Wavenumber cm-1")
#    plt.ylabel("Radiance factor vs Sun")
##    plt.title("%s: Mean of %i frames" %(title,nframes_to_average))
#    plt.title("Typical nadir spectrum from LNO divided by solar spectrum")
#    
#    if save_figs:
#        os.chdir(BASE_DIRECTORY)
#        plt.figure(2)
#        plt.savefig("Typical_nadir_spectrum_from_LNO.png")
#        plt.figure(3)
#        plt.savefig("Fitting_points_to_nadir_curves.png")
#        plt.figure(4)
#        plt.savefig("Typical_nadir_spectrum_from_LNO_divided_by_solar_spectrum.png")
##    poly_factors = np.polyfit(wavenumbers[0][loc_max_indices], detector_data[0,loc_max_indices], 5, w=detector_data[0,loc_max_indices]*detector_data[0,loc_max_indices])
##    poly = np.poly1d(poly_factors)
##    y_new = poly(wavenumbers[0])
##    plt.plot(wavenumbers[0],y_new)
#
##    filtered_data = sg_filter(detector_data[0,loc_max_indices],window_size=25,order=1)
##    plt.plot(wavenumbers[0][loc_max_indices],filtered_data)
#
#
#    
#        
#if option==29:
#    """analyse lno limb scan 1"""
#    
#    """get data from file"""
#    detector_data_bins = get_dataset_contents(hdf5_file,"YBins")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
##    data_name_all = get_dataset_contents(hdf5_file,"Name")[0] #get data
#    time_data_all = get_dataset_contents(hdf5_file,"Observation_Time")[0]
#    date_data_all = get_dataset_contents(hdf5_file,"Observation_Date")[0]
#    hdf5_file.close()
#    
#    chosen_range = [50,270]
#    zero_indices = range(0,20)+range(300,320)
#    
#    mean_offsets = np.zeros_like(detector_data_bins)
#    
#    mean_offset = np.mean(detector_data_bins[:,:,zero_indices], axis=2)
#    for column_index in range(320):
#        mean_offsets[:,:,column_index] = mean_offset
#    
#    offset_data_bins = detector_data_bins - mean_offsets
#    
#    spec_summed_data = np.sum(detector_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
#    offset_spec_summed_data = np.sum(offset_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
#    
#    frame_number = 0
#    plt.figure(figsize = (10,8))
#    plt.imshow(detector_data_bins[frame_number,:,:])
#    plt.title("Frame %i" %frame_number)
#    plt.xlabel("Detector horizontal (spectral) direction")
#    plt.ylabel("Detector vertical (spatial) direction")
#    plt.colorbar()
#
#    frame_number = 0
#    plt.figure(figsize = (10,8))
#    plt.imshow(offset_data_bins[frame_number,:,:])
#    plt.title("Frame %i" %frame_number)
#    plt.xlabel("Detector horizontal (spectral) direction")
#    plt.ylabel("Detector vertical (spatial) direction")
#    plt.colorbar()
#
##    binned_detector_data = np.sum(detector_data_bins, axis=1) #sum all lines vertically
##    detector_data_sum = np.sum(binned_detector_data[:,], axis=1)
#
##    plt.figure(figsize = (10,8))
##    plt.xlabel("Time")
##    plt.ylabel("Signal sum")
##    for detector_row in list(np.transpose(spec_summed_data)): #plot each row separately
##        plt.plot(detector_row)
##    plt.title(title+": Detector rows versus time")
###    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+".png")
#
#    plt.figure(figsize = (10,8))
#    plt.xlabel("Time")
#    plt.ylabel("Signal sum")
#    for row_index in [2,8,14,20]: #range(len(offset_spec_summed_data[0,:])): #plot each row separately
#        plt.plot(offset_spec_summed_data[:,row_index],"o", linewidth=0, label="%i" %row_index)
#    plt.legend()
#    plt.title(title+": Detector rows versus time")
#    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+".png")
#
#
#
#if option==35:
#    """check ACS solar pointing test"""
#    """get data from file"""
#    dark_detector_data_bins = get_dataset_contents(hdf5_files[0],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#    hdf5_files[0].close()
#
#    file_number=1
#    light_detector_data_bins = get_dataset_contents(hdf5_files[file_number],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#    exponent_all = get_dataset_contents(hdf5_files[file_number],"Exponent")[0]
#    hdf5_files[file_number].close()
#
#    pixel_number=100
#    bin_number=1
#    
#    exponent_values = 2.0**np.asfarray(list(exponent_all)[bin_number::][::4])
#
##    light_line1 = detector_data_bins[:,23,:]
##    dark_sum1=np.mean(dark_line1, axis=1)
##    light_sum1=np.mean(light_line1, axis=1)
#
#
#    light1 = np.asfarray(list(light_detector_data_bins[:,pixel_number])[bin_number::][::4])
#    dark1 = np.asfarray(list(dark_detector_data_bins[:,pixel_number])[bin_number::][::4])
#    
#    sub1 = light1-dark1
#    sub_all = np.asfarray(list(light_detector_data_bins)[bin_number::][::4])-np.asfarray(list(dark_detector_data_bins)[bin_number::][::4])
#
#    plt.figure(figsize=(10,8))
#    plt.plot(dark1, label="Dark pixel %i" %pixel_number)
#    plt.plot(light1, label="Light pixel %i" %pixel_number)
#    plt.title(title+" "+obspaths[file_number])
#    plt.legend()
#    plt.xlabel("Frame Number")
#    plt.ylabel("Signal on pixel %i" %pixel_number)
#    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_dark_and_light.png")
#
#    plt.figure(figsize=(10,8))
#    plt.plot(exponent_values)
#    plt.xlabel("Frame Number")
#    plt.ylabel("Exponent")
#    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_exponent_value.png")
#
#    plt.figure(figsize=(10,8))
##    plt.plot(sub1)
#    plt.errorbar(range(len(sub1)),sub1,yerr=exponent_values, ecolor="r")
#    plt.xlabel("Frame Number")
#    plt.ylabel("Dark subtracted signal on pixel %i" %pixel_number)
#    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+"_dark_sub.png")
#
#    
#
#    if title=="SO ACS Solar Pointing Test":
#        frames_to_plot=range(50,200,20)
#    elif title=='SO Raster A':
#        frames_to_plot=range(100,120,2)+range(1040,1060,2)
#    elif title=="SO Light to Dark":
#        frames_to_plot=range(50,500,50)
#
#    plt.figure(figsize=(10,8))
#    for frame_to_plot in frames_to_plot:
#        for subframe in range((frame_to_plot*4),(frame_to_plot*4+4),1):
#            plt.plot(light_detector_data_bins[subframe,:], label="Frame=%i-%i" %(frame_to_plot,subframe))
#    plt.legend()
#    plt.xlabel("Pixel Number")
#    plt.ylabel("Detector Signal")
#    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+ "_spectra_comparison.png")
#
#
#        
#    plt.figure(figsize=(10,8))
#    for frame_to_plot in frames_to_plot:
#        plt.plot(sub_all[frame_to_plot], label="Frame=%s" %frame_to_plot)
#    plt.legend()
#    plt.xlabel("Pixel Number")
#    plt.ylabel("Background Subtracted Detector Signal")
##    plt.yscale("log")
#    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title.replace(" ","_")+ "_bg_sub_spectra.png")
#
#
#
#
#
#if option==36:
#
#    for hdf5_file in hdf5_files:
#
##        binned_detector_data = np.squeeze(get_dataset_contents(hdf5_file,"Y")[0]) #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#        binned_detector_data = np.mean(get_dataset_contents(hdf5_file,"Y")[0],axis=1) #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#        aotf_freq_all = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
#        time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0][:,0]
#        hdf5_file.close()
#        
#        pxstart=160
#        pxend=200
#    
#        for aotf_freq_subd in aotf_freq_all[1:4]:
##        for aotf_freq_subd in [aotf_freq_all[1]]:
#            binned_data_subd = np.asfarray([frame for index,frame in enumerate(list(binned_detector_data)) if aotf_freq_all[index]==aotf_freq_subd])
#            time_data_subd = [time_out for index,time_out in enumerate(list(time_data_all)) if aotf_freq_all[index]==aotf_freq_subd]
#        
#            order = spectral_calibration("aotf2order",channel,aotf_freq_subd,0)
#        #        plt.plot(np.sum(binned_data_subd[:,pxstart:pxend], axis=1), label=order)
#    #        plt.legend()
#        
#            wavenumbers = spectral_calibration("pixel2waven",channel,order,-15.0)
#            
#        #    zero_indices=range(0,20)+range(300,320) #assume mean of first and last 20 values are centred on zero. this will become offset
#        #    offset_data = np.mean(binned_detector_data[:,zero_indices], axis=1) #calculate offset for every frame
#        #    
#        #    if apply_offset:
#        #        for index,offset_value in enumerate(offset_data):
#        #            binned_detector_data[index,:] = binned_detector_data[index,:] - offset_value #subtract offset from every summed detector line
#            
#        #    plt.figure(figsize = (10,8))
#        #    plt.title("Vertically binned spectra")
#        #    plt.xlabel("Pixel number")
#        #    plt.ylabel("Signal value")
#        #    plt.plot(np.transpose(binned_data_subd[:,:]))
#        
#        #    plt.figure(figsize = (10,8))
#        #    plt.title("Vertically binned spectra frame 140")
#        #    plt.xlabel("Pixel number")
#        #    plt.ylabel("Signal value")
#        #    plt.plot(np.transpose(binned_data_subd[140,:]))
#        
#            plt.figure(figsize = (figx/2,figy/2))
#            plt.xlabel("Wavenumber (cm-1)")
#            plt.ylabel("Signal value")
##            frame_ranges=[[90,110],[130,150],[170,190]] #lno inertial dayside
#            frame_ranges=[[10,30],[30,50],[50,70],[70,90],[90,110],[110,130]] #lno inertial dayside
#        
#            range_title = "Summed vertically binned spectra order %i\nframes " %order
#            for frame_range in frame_ranges:
#                range_title=range_title + "%i-%i," %(frame_range[0],frame_range[1])
#                summed_binned_frames_subd = np.mean(binned_data_subd[range(frame_range[0],frame_range[1]),:],axis=0)
#        #        plt.plot(wavenumbers,summed_binned_frames_subd, label="Frames %i-%i" %(frame_range[0],frame_range[1]))
#                plt.plot(wavenumbers,summed_binned_frames_subd, label=time_data_subd[int(np.mean([frame_range[0],frame_range[1]]))])
#            plt.title(title+": "+range_title)
##            plt.title("NOMAD LNO Infrared Spectra of Mars, 22 November 2016")
##            plt.title("NOMAD LNO Infrared Spectra of Mars, 6th March 2017")
#            plt.legend()
#        #    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+"LNO_spectra_mars_xx_November_2016.png", dpi=400)
#            if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+title+"order %i.png" %order)
#    
#    
#    
#    
#    
#if option==37:
#    """analyse lno limb scan 1 and 2"""
#   
#    ref="J2000"
#    abcorr="None"
#    tolerance="1"
#    method="Intercept: ellipsoid"
#    formatstr="C"
#    prec=3
#    
#    use_both_subdomains=1
#    subdomain=1
#
#    if title=="LNO Limb Scan 1":
#        bins_to_use=range(12)
#        scaler=4.0 #conversion factor between the two diffraction orders
#        signal_cutoff=6000.0
#        signal_peak=23000.0
#    elif title=="LNO Limb Scan 2":
#        bins_to_use=range(1,12)
#        scaler=3.8 #conversion factor between the two diffraction orders
#        if subdomain==0:
#            signal_cutoff=2000.0
#            signal_peak=8000.0
#        elif subdomain==1:
#            signal_cutoff=7000.0
#            signal_peak=30000.0
#
#    
#    """get data from file"""
#    detector_data_bins = get_dataset_contents(hdf5_file,"YBins")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#    aotf_freq_all = get_dataset_contents(hdf5_file,"AOTFFrequency")[0]
#    time_data_all = get_dataset_contents(hdf5_file,"ObservationTime")[0]
#    date_data_all = get_dataset_contents(hdf5_file,"ObservationDate")[0]
#    binning_all = get_dataset_contents(hdf5_file,"Binning")[0]
#    bins_all = get_dataset_contents(hdf5_file,"Bins")[0]
#    hdf5_file.close()
#    
#    """assumes all bins identical!"""
#    bins_mid = np.mean(bins_all[0,:,:],axis=1)
#    offset_from_centre = bins_mid - detector_centre
#    bin_size = float(binning_all[0]+1)
#   
#    epoch_times_start=convert_hdf5_time_to_spice_utc(list(time_data_all[:,0]),list(date_data_all[:,0]))
#    epoch_times_end=convert_hdf5_time_to_spice_utc(list(time_data_all[:,1]),list(date_data_all[:,1]))
#    epoch_times_mid = np.mean(np.asarray([epoch_times_start,epoch_times_end]),axis=0)
#    
#    
#    aotf_freq_subd=aotf_freq_all[0]
#    if use_both_subdomains==1:
#        detector_data_bins = np.asfarray([frame*scaler if aotf_freq_all[index]==aotf_freq_subd else frame for index,frame in enumerate(list(detector_data_bins))])
#    else:
#        if title=="LNO Limb Scan 1":
#            detector_data_bins = np.asfarray([frame*scaler for index,frame in enumerate(list(detector_data_bins)) if aotf_freq_all[index]==aotf_freq_subd])
#        elif title=="LNO Limb Scan 2":
#            if subdomain==0:
#                detector_data_bins = np.asfarray([frame for index,frame in enumerate(list(detector_data_bins)) if aotf_freq_all[index]==aotf_freq_subd])
#                epoch_times_mid = np.asfarray([et for index,et in enumerate(list(epoch_times_mid)) if aotf_freq_all[index]==aotf_freq_subd])
#            elif subdomain==1:
#                detector_data_bins = np.asfarray([frame for index,frame in enumerate(list(detector_data_bins)) if aotf_freq_all[index]!=aotf_freq_subd])
#                epoch_times_mid = np.asfarray([et for index,et in enumerate(list(epoch_times_mid)) if aotf_freq_all[index]!=aotf_freq_subd])
#    
#
##    chosen_range = [50,270]
#    chosen_range = [140,200]
#    zero_indices = range(0,20)+range(300,320) #for scaling all spectra to a common zero level on first and last pixels
#    
#    mean_offsets = np.zeros_like(detector_data_bins)
#    
#    mean_offset = np.mean(detector_data_bins[:,:,zero_indices], axis=2)
#    for column_index in range(320):
#        mean_offsets[:,:,column_index] = mean_offset
#    
#    offset_data_bins = detector_data_bins - mean_offsets
#    
#    spec_summed_data = np.sum(detector_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
#    offset_spec_summed_data = np.sum(offset_data_bins[:,:,chosen_range[0]:chosen_range[1]], axis=2)
#    frame_range = np.arange(len(offset_spec_summed_data))
#    
#    colours=["bo","go","ro","co","mo","yo","ko","bs","gs","rs","cs","ms","ys","ks","bd","gd"]
#    line_colours=["b-","g-","r-","c-","m-","y-","k-","b-","g:","r:","c:","m:","y:","k:","b--","g--"]
#
#    plt.figure(figsize = (figx,figy))
#    plt.xlabel("Frame Number")
#    plt.ylabel("Sum of signal from chosen detector region")
#    
#    limb_time=np.zeros((12,3))
#    time_offset=np.zeros((12,3))
#    offset_from_centre_corrected=np.zeros((12,3))
#    offset_from_centre_uncorrected=np.zeros((12,3))
#
#        
#    for row_index in bins_to_use: #range(len(offset_spec_summed_data[0,:])): #plot each row separately
#    #    for row_index in [1,4,7,10]: #range(len(offset_spec_summed_data[0,:])): #plot each row separately
#    #    for row_index in [3,4]:
#        summed_row = offset_spec_summed_data[:,row_index]
#        plt.scatter(epoch_times_mid,summed_row, linewidth=0, label="Detector region %i" %row_index)
#    
#        start_index1=0
#        start_index2=0
#        start_range=np.min(np.where(summed_row[start_index1:]<signal_cutoff)[0])+start_index1
#        end_range=np.min(np.where(summed_row[start_index2:]>signal_cutoff)[0])+start_index2
#        dark_range1 = range(start_range,end_range)
#    
#        start_index1=85*(use_both_subdomains+1)
#        start_index2=130*(use_both_subdomains+1)
#        start_range=np.min(np.where(summed_row[start_index1:]>signal_cutoff)[0])+start_index1
#        end_range=np.min(np.where(summed_row[start_index2:]<signal_cutoff)[0])+start_index2
#        light_range1 = range(start_range,end_range)
#        plt.plot(epoch_times_mid[light_range1],summed_row[light_range1],line_colours[row_index], label="Mars line 1 %i" %row_index)
#
#        start_index1=170*(use_both_subdomains+1)
#        start_index2=200*(use_both_subdomains+1)
#        start_range=np.min(np.where(summed_row[start_index1:]>signal_cutoff)[0])+start_index1
#        end_range=np.min(np.where(summed_row[start_index2:]<signal_cutoff)[0])+start_index2
#        light_range3 = range(start_range,end_range)
#        plt.plot(epoch_times_mid[light_range3],summed_row[light_range3],line_colours[row_index], label="Mars line %i" %row_index)
#
#        start_index1=0
#        limb_index1 = np.min(np.where(summed_row[start_index1:]>signal_cutoff)[0])+start_index1
#        plt.plot(epoch_times_mid[limb_index1],summed_row[limb_index1],colours[row_index], label="Limb crossing line %i" %row_index)
#        limb_ratio = summed_row[limb_index1]/signal_peak
#        limb_time[row_index,0] = epoch_times_mid[limb_index1]
#        print(limb_ratio)
#        offset_from_centre_uncorrected[row_index,0] = offset_from_centre[row_index]
#        offset_from_centre_corrected[row_index,0] = offset_from_centre[row_index] + (limb_ratio*bin_size - bin_size/2.0)
#        
#        start_index1=130*(use_both_subdomains+1)
#        if row_index==0: start_index1=140*(use_both_subdomains+1) #fudge to make first row work        
#        
#        limb_index2 = np.min(np.where(summed_row[start_index1:]<signal_cutoff)[0])+start_index1 -1
#        plt.plot(epoch_times_mid[limb_index2],summed_row[limb_index2],colours[row_index], label="Limb crossing line %i" %row_index)
#        limb_ratio = summed_row[limb_index2]/signal_peak
#        limb_time[row_index,1] = epoch_times_mid[limb_index2]
#        print(limb_ratio)
#        offset_from_centre_uncorrected[row_index,1] = offset_from_centre[row_index]
#        offset_from_centre_corrected[row_index,1] = offset_from_centre[row_index] + (limb_ratio*bin_size - bin_size/2.0)
#        
#        start_index1=170*(use_both_subdomains+1)
#        limb_index3 = np.min(np.where(summed_row[start_index1:]>signal_cutoff)[0])+start_index1
#        plt.plot(epoch_times_mid[limb_index3],summed_row[limb_index3],colours[row_index], label="Limb crossing line %i" %row_index)
#        limb_ratio = summed_row[limb_index3]/signal_peak
#        limb_time[row_index,2] = epoch_times_mid[limb_index3]
#        print(limb_ratio)
#        offset_from_centre_uncorrected[row_index,2] = offset_from_centre[row_index]
#        offset_from_centre_corrected[row_index,2] = offset_from_centre[row_index] + (limb_ratio*bin_size - bin_size/2.0)
#            
#    if 0 not in bins_to_use:
#        offset_from_centre_uncorrected=np.delete(offset_from_centre_uncorrected,0,axis=0)
#        offset_from_centre_corrected=np.delete(offset_from_centre_corrected,0,axis=0)
#        limb_time=np.delete(limb_time,0,axis=0)
#    
#    
##    plt.legend()
#    plt.title("LNO Channel Limb Scan during Mars Capture Orbit Calibration Part 2")
#    if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+"LNO Channel Limb Scan during Mars Capture Orbit Calibration Part 2.png", dpi=300)
#
#    for crossing_index in range(3):
#
#        plt.figure(figsize = (figx-7,figy-3))
#        plt.xlabel("Offset in arcmins of detector bin from centre of FOV (negative is towards detector line 0)")
#        plt.ylabel("Time of limb crossing (seconds)")
##        plt.plot(offset_from_centre_uncorrected[:,crossing_index],limb_time[:,crossing_index],label="Limb crossing time without signal correction")
#        plt.plot(offset_from_centre_corrected[:,crossing_index],limb_time[:,crossing_index],label="Limb crossing time with signal correction")
#        plt.ylim((min(limb_time[:,crossing_index]-1),max(limb_time[:,crossing_index]+2)))
#    
#        fit=np.polyfit(offset_from_centre_corrected[:,crossing_index],limb_time[:,crossing_index],1)
#        fit_residual = np.sum(np.sqrt((limb_time[:,crossing_index] - np.polyval(fit,offset_from_centre_corrected[:,crossing_index]))**2))
#        print(fit_residual)
#        
#        fitx = np.arange(-50,50,1)
#        fity = np.polyval(fit,fitx)
#        plt.plot(fitx,fity,label="Fit to corrected crossing time")
#        plt.legend()
#        
#        crossing_et = np.polyval(fit,0)
#        plt.plot(0,crossing_et,"ko")
#        crossing_time = sp.et2utc(crossing_et,formatstr,prec)
#        print(crossing_time)
#        if save_figs: plt.savefig(BASE_DIRECTORY+os.sep+"LNO Limb Scan Linear fit vs limb crossing time %i for each detector bin.png" %crossing_index)
#       
#        
#     
#
#
#if option ==49:
#    
#    soDetectorDataAll = get_dataset_contents(hdf5_files[0],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#    lnoDetectorDataAll = get_dataset_contents(hdf5_files[1],"Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#    hdf5_files[0].close()
#    hdf5_files[1].close()
#    
#    frameNumber = 20
#    columnNumber = 180
#    vPixels = np.arange(24)
#    soStartPixel = 2
#    lnoStartPixel = 0
#    
#    plt.figure(figsize = (figx-5,figy-3))
#    plt.imshow(soDetectorDataAll[frameNumber,:,:], aspect=4)
#    plt.figure(figsize = (figx-5,figy-3))
#    plt.imshow(lnoDetectorDataAll[frameNumber,:,:], aspect=4)
#    
#    soDetectorLine = soDetectorDataAll[frameNumber,:,columnNumber]
#    
#    soPolyCoeffs = np.polyfit(vPixels[soStartPixel::],soDetectorLine[soStartPixel::], 2)
#    soPoly = np.polyval(soPolyCoeffs, vPixels[soStartPixel::])
#    
#
#
#
#
#    lnoDetectorLine = lnoDetectorDataAll[frameNumber,:,columnNumber]
#    
#    lnoPolyCoeffs = np.polyfit(vPixels[lnoStartPixel::],lnoDetectorLine[lnoStartPixel::], 2)
#    lnoPoly = np.polyval(lnoPolyCoeffs, vPixels[lnoStartPixel::])
#    
#    plt.figure(figsize = (figx-5,figy-3))
#    plt.plot(vPixels[soStartPixel::],soDetectorLine[soStartPixel::], label="SO Vertical Slice")
#    plt.plot(vPixels[soStartPixel::],soPoly, label="SO Polynomial")
#    plt.plot(vPixels[lnoStartPixel::],lnoDetectorLine[lnoStartPixel::], label="LNO Vertical Slice")
#    plt.plot(vPixels[lnoStartPixel::],lnoPoly, label="LNO Polynomial")
#    plt.legend()
#
#    print(soPolyCoeffs)
#    print("SO centre at pixel %0.1f" %(-1.0*soPolyCoeffs[1]/(2.0*soPolyCoeffs[0])))
#    print(lnoPolyCoeffs)
#    print("LNO centre at pixel %0.1f" %(-1.0*lnoPolyCoeffs[1]/(2.0*lnoPolyCoeffs[0])))
#
#
#
#
#
#
#
