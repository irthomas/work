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
    METAKERNEL_NAME = "em16_ops_win.tm"
#    sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
    print(sp.tkvrsn("toolkit"))
    print("KERNEL_DIRECTORY=%s" %KERNEL_DIRECTORY)
    SPICE_ABERRATION_CORRECTION = "None"
    SPICE_OBSERVER = "-143"


rcParams["axes.formatter.useoffset"] = False
PASSWORD = ""


SAVE_FIGS = False
#SAVE_FIGS = True

PLOT_FIGS = True
#PLOT_FIGS = False









#option=1

#checkout="Ground"
#checkout="NEC"
#checkout="MCC"
#checkout="MCO1"
#checkout="MCO2"


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
#obspaths = ["20180511_084630_0p3a_SO_1_C","20180522_221149_0p3a_SO_1_C"]
#obspaths = ["20180428_023343_0p2a_SO_1_C", "20180511_084630_0p3a_SO_1_C"] #SO scans (first 1 cut off by atmosphere!)
#obspaths = ["20180511_084630_0p2a_SO_1_C"] #UVIS

#obspaths = ["*2018*_0p2a*_SO_1_C"]
#fileLevel = "hdf5_level_0p3a"



"""MTP005"""
#obspaths = ["20180821_193241_0p3a_SO_1_C","20180828_223824_0p3a_SO_1_C"]
#fileLevel = "hdf5_level_0p3a"

"""MTP009+010"""
obspaths = ["20181225_025140_0p2a_SO_1_C", "20190118_183336_0p2a_SO_1_C"]
fileLevel = "hdf5_level_0p2a"



"""SO fullscans FOV check"""
#obspaths = [
#    "20180424_201833_0p3a_SO_1_S",
#    "20180429_125925_0p3a_SO_1_S",
#    "20180524_045641_0p3a_SO_1_S",
#    "20180813_223104_0p3a_SO_1_S",
#    "20180907_082257_0p3a_SO_1_S",
#    ]
#obsTitles = [
#    "SO ingress with SO boresight 24 bins x 1 pixel per order",
#    "SO egress with SO boresight 24 bins x 1 pixel per order",
#    "Something",
#    ]
#fileLevel = "hdf5_level_0p3a"
#frameNumbers = [100,200,200,200,200]







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
#    return obs2SunUnitVector 
    return -1 * obs2SunUnitVector #-1 is there to switch the directions to be like in cosmographia




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
        if hdf5Filenames[fileIndex] in ["20180511_084630_0p3a_SO_1_C"]:
            lineScanCentreIndicesAll = [range(60,120,1),range(280,340,1),range(400,490,1),range(630,720,1),range(760,850,1),range(990,1060,1),range(1150,1200,1)]
            completeLineScanIndices = range(7)
            analyse=True
            parallel=True
            firstBinRow = 116
            firstBinScience = 0
        elif hdf5Filenames[fileIndex] in ["20180428_023343_0p3a_SO_1_C"]:
            lineScanCentreIndicesAll = [range(230,370,1),range(380,520,1),range(520,680,1)]
            completeLineScanIndices = range(3)
            analyse=False
            parallel=False
            firstBinRow = 116
            firstBinScience = 0
        elif hdf5Filenames[fileIndex] in ["20180522_221149_0p3a_SO_1_C"]:
            lineScanCentreIndicesAll = [range(280,300,1),range(440,470,1),range(590,650,1),range(760,815,1),range(925,970,1),range(1100,1120,1)]
            completeLineScanIndices = range(1,5,1)
            analyse=True
            parallel=False
            firstBinRow = 116
            firstBinScience = 0
        elif hdf5Filenames[fileIndex] in ["20180821_193241_0p3a_SO_1_C"]:
            lineScanCentreIndicesAll = [range(60,120,1),range(280,340,1),range(400,490,1),range(630,720,1),range(760,850,1),range(990,1060,1),range(1150,1200,1)]
            completeLineScanIndices = range(7)
            analyse=True
            parallel=True
            firstBinRow = 128
            firstBinScience = 1
        elif hdf5Filenames[fileIndex] in ["20180828_223824_0p3a_SO_1_C"]:
            lineScanCentreIndicesAll = [range(60,140,1),range(250,330,1),range(430,510,1),range(610,690,1),range(790,870,1),range(960,1030,1),range(1150,1220,1)]
            completeLineScanIndices = range(1,5,1)
            analyse=True
            parallel=False
            firstBinRow = 128
            firstBinScience = 1
        else:
            print("Error: please add filename and indices to list")
            detectorDataAll = get_dataset_contents(hdf5File, "Y")[0]
            print(detectorDataAll.shape)
            plt.figure()
            plt.plot(detectorDataAll[:, :, 200])
    
        hdf5File = hdf5Files[fileIndex]
        print("Reading in file %i: %s" %(fileIndex + 1, obspaths[fileIndex]))
        
        detectorDataAll = get_dataset_contents(hdf5File, "Y")[0]
        binsAll = get_dataset_contents(hdf5File, "Bins")[0]
        observationTimeStringsAll = get_dataset_contents(hdf5File, "ObservationDateTime")[0]
        #        hdf5File.close()
        
        centreBinIndices = np.where(binsAll[:,0,0] == firstBinRow)[0] #window stepping, so find 1st bin
        centreRowIndex = np.where(binsAll[firstBinScience,:,0] == 128)[0] #then find which row is the centre of the detector in that bin
        centrePixelIndex = 200
        
        detectorDataCentreFrames = detectorDataAll[centreBinIndices,:,:] #get data for frames containing centre bins
        observationTimeStringsCentreFrames = observationTimeStringsAll[centreBinIndices,:]
        observationTimesCentreFrames = np.asfarray([np.mean([sp.utc2et(timeCentreFrame[0]),sp.utc2et(timeCentreFrame[1])]) for timeCentreFrame in list(observationTimeStringsCentreFrames)])
        
        detectorDataCentrePixel = detectorDataCentreFrames[:,centreRowIndex,centrePixelIndex] #get data for detector centre row
        
        
        
        centreLineCentrePixel = detectorDataCentreFrames[:, centreRowIndex, centrePixelIndex].flatten()
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
        #        marker_colour = np.log(detectorDataCentrePixel+1000).flatten()
        marker_colour = detectorDataCentrePixel.flatten()
        ax.set_xlim([-0.004,0.004])
        ax.set_ylim([-0.004,0.004])
        ax.set_xlabel("%s FRAME X" %referenceFrame)
        ax.set_ylabel("%s FRAME Y" %referenceFrame)
        ax.scatter(unitVectors[:,0], unitVectors[:,1], c=marker_colour, vmin=300000, alpha=0.5, cmap="jet", linewidths=0)
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
        """plot and label new/old boresights on plot"""
        ax.scatter(0.0,0.0, marker="x", c="k", s=160)
#        ax.scatter(meanParFit,meanPerpFit, marker="x", c="k", s=160)
        offset = 0.0001
        ax.text(0.0+offset,0.0+offset*1.5,"Origin")
#        ax.text(meanParFit+offset,meanPerpFit-offset*2.0,"New Boresight")

        ax.plot([meanParFit,meanParFit],[maxValuesVectors[0,1],maxValuesVectors[-1,1]], "k")
        ax.plot([meanParFit,meanParFit],[np.min(unitVectors[:,1]),np.max(unitVectors[:,1])], "k")
        circle1 = plt.Circle((0.0,0.0), 10.0/60.0/sp.dpr(), color='k', alpha=0.1)
#        circle1 = plt.Circle((meanParFit,meanPerpFit), 10.0/60.0/sp.dpr(), color='k', alpha=0.1)
        ax.add_artist(circle1)


def analyseLineScan2(hdf5Files,hdf5Filenames):

#    referenceFrame = "TGO_NOMAD_UVIS_OCC"
    referenceFrame = "TGO_NOMAD_SO"
#    referenceFrame = "TGO_SPACECRAFT"
    fig, ax = plt.subplots(figsize=(FIG_X, FIG_Y))

    for fileIndex, hdf5File in enumerate(hdf5Files):
        print("Reading in file %i: %s" %(fileIndex + 1, obspaths[fileIndex]))
        
        detectorDataAll = get_dataset_contents(hdf5File, "Y")[0]
#        binsAll = get_dataset_contents(hdf5File, "Bins")[0]
        observationTimeStringsAll = get_dataset_contents(hdf5File, "ObservationDateTime")[0]
        
        centrePixelIndex = 200
        detectorDataBin1 = detectorDataAll[:, 1, centrePixelIndex].flatten()
        detectorDataBin2 = detectorDataAll[:, 2, centrePixelIndex].flatten()
        detectorDataCentrePixel = np.mean((detectorDataBin1, detectorDataBin2), axis=0)
        
        print("max value = %0.0f, min value = %0.0f" %(np.max(detectorDataCentrePixel), np.min(detectorDataCentrePixel)))
        observationTimes = np.asfarray([np.mean([sp.utc2et(time[0]),sp.utc2et(time[1])]) for time in list(observationTimeStringsAll)])
        
        unitVectors = np.asfarray([get_vector(datetime,referenceFrame) for datetime in observationTimes])
        #        marker_colour = np.log(detectorDataCentrePixel+1000).flatten()
        marker_colour = np.log(detectorDataCentrePixel.flatten())
        ax.set_xlim([-0.004,0.004])
        ax.set_ylim([-0.004,0.004])
        ax.set_xlabel("%s FRAME X" %referenceFrame)
        ax.set_ylabel("%s FRAME Y" %referenceFrame)
        ax.scatter(unitVectors[:,0], unitVectors[:,1], c=marker_colour, vmin=np.log(200000), alpha=0.5, cmap="jet", linewidths=0)
        ax.set_aspect("equal")
        
        circle1 = plt.Circle((0, 0), 0.0016, color='yellow', alpha=0.1)
        ax.add_artist(circle1)
        
#        ax.scatter(maxValuesVectors[:,0], maxValuesVectors[:,1], c="k")
    


def showFullFrame03A(hdf5Files, obspaths, titles, frame_numbers, uvis=False):
    fileIndices = list(range(len(hdf5Files)))

    for fileIndex in fileIndices:
        hdf5File = hdf5Files[fileIndex]
        frame_number = frame_numbers[fileIndex]
        #get data from file
        print("Reading in file %i: %s" %(fileIndex+1, obspaths[fileIndex]))
        detectorData = get_dataset_contents(hdf5File, "Y")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
#        wavenumberData = get_dataset_contents(hdf5File, "X")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        bins = get_dataset_contents(hdf5File, "Bins")[0] #get Y data (full frame) from file. Data has 3 dimensions: time x vertical detector x horizontal
        altPoint0 = get_dataset_contents(hdf5File, "TangentAlt", chosen_group="Geometry/Point0")[0]
        diffractionOrders = get_dataset_contents(hdf5File, "DiffractionOrder")[0]
        hdf5File.close()
        nSpectra = detectorData.shape[0]
        print("File contains %i detector frames" %np.int(nSpectra/24))

        plt.figure(figsize=(FIG_X, FIG_Y))
        detectorFrame = detectorData[range(24*frame_number, 24*(frame_number+1)), :]
        altitude = altPoint0[24*frame_number,0]
        diffractionOrder = diffractionOrders[24*frame_number]
#        wavenumbers = wavenumberData[24*frame_number,:]
        plt.imshow(detectorFrame, aspect = 5)
        plt.title(titles[fileIndex]+": position of illumination on detector frame %i order %i altitude %0.1fkm" %(frame_number, diffractionOrder, altitude))
        print(bins[range(24*frame_number, 24*(frame_number+1)), 0])
#        plt.xticks(range(0,int(len(wavenumbers)),39), ["%0.1f" %wavenumber for wavenumber in wavenumbers[range(0,int(len(wavenumbers)),39)]])
        if SAVE_FIGS:
            plt.savefig(titles[fileIndex].replace(" ", "_")+"_position_of_illumination_on_detector.png")

        plt.figure(figsize=(FIG_X, FIG_Y))
        plt.title(titles[fileIndex]+": illumination intensity per detector row")
        plt.xlabel("Detector row")
        plt.ylabel("Detector counts (bg subtracted)")
        for index in range(0,np.int(nSpectra/24)):
            plt.plot(bins[range(24*index, 24*(index+1)), 0],detectorData[range(24*index, 24*(index+1)), 160])
#            plt.plot(bins[range(24*index,24*(index+1)),0],normalise(detectorData[range(24*index,24*(index+1)),160]))
        if SAVE_FIGS:
            plt.savefig(titles[fileIndex].replace(" ", "_")+"_illumination_intensity_per_detector_row.png")
    
    return detectorFrame


"""plot linescans"""
hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
analyseLineScan2(hdf5Files, hdf5Filenames)
#ani = animateFrames(hdf5Files, hdf5Filenames, titles); plt.show(plt.show())

#don't trust the calculation - angles estimated manually from data
#angleSeparationA = -89.08792599580298
#angleSeparationB = 67.15504926474100 + 3.0/60.0 #old value plus 3 arcminutes
#printBoresights(angleSeparationA, angleSeparationB)




"""plot full detector frame"""
#hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel)
#detectorFrame = showFullFrame03A(hdf5Files, hdf5Filenames, titles,  frameNumbers)


