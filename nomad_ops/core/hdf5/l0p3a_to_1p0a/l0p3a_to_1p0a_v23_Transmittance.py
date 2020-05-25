# -*- coding: utf-8 -*-

import logging
import os
import matplotlib
matplotlib.use('Agg')
import h5py
import numpy as np
import time

from nomad_ops.config import NOMAD_TMP_DIR
import nomad_ops.core.hdf5.generic_functions as generics
#from pipeline.pipeline_config_v04 import NA_VALUE, NA_STRING


import warnings
warnings.simplefilter('ignore', np.RankWarning) #turn off polyfit warnings!


__project__   = "NOMAD"
__author__    = "Loïc Trompet"
__contact__   = "loic.trompet@aeronomie.be"

#============================================================================================
# 7. CONVERT HDF5 LEVEL 0.3A TO LEVEL 1.0A for S
#
# TRANSMITTANCE ESTIMATION
#
#Be aware that filename have the following definition:
# Date_Time_Level_Channel_Science_Type.h5
#   1   2    3      4       5       6  
#
#============================================================================================

logger = logging.getLogger( __name__ )

VERSION = 80
OUTPUT_VERSION = "1.0A"

#################################
# from pipeline_mappings_v04.py #
################################################################################################
#DATASETS_TO_BE_REMOVED = [
#    "Science/Y",
#    "Science/YTypeFlag",
#    "Science/YUnitFlag",
#    "Science/YError",
#    "Science/YErrorFlag"
#]
#
#ATTRIBUTES_TO_BE_REMOVED = ['YCalibRef', 'YErrorRef','NSpec']
#
#DATASETS_NOT_TO_BE_RESHAPED = ['BadPixelMap','PointXY','Housekeeping','Telecommand20','QualityFlag','DateTime','Timestamp','PacketSize']    #dset not to change
#DATASETS_NOT_TO_BE_RESHAPED_COMPRESSED = ['Telecommand20','QualityFlag']


#copied from level 1 functions
ATTRIBUTES_TO_BE_REMOVED = [
        'YCalibRef', 
        'YErrorRef',
        'NSpec',
        'Name',
        'Timestamp',
        ] 

DATASETS_NOT_TO_BE_RESHAPED = [
        'Science/BadPixelMap',
        'PointXY',
        'Housekeeping/',
        ]

DATASETS_NOT_TO_BE_RESHAPED_NOR_COMPRESSED = [
        'Telecommand20/',
        'QualityFlag/',
        'Temperature/',
        ]


DATASETS_TO_BE_REMOVED=[
        "Science/Y",
        "Science/YUnmodified",
        "Science/YTypeFlag",
        "Science/YUnitFlag",
        "Science/YError",
        "Science/YErrorFlag",
        #"Geometry/Point0/LST",    #LST to be removed
        "Science/XNbBin",
        "Science/Xpix_1b",

        'Channel/PacketSize',
        'BkgTimestamp',
        'Science/BackgroundY',  # darks no more reshaped: problems for cases where 1 order scanned several times and cases where dark scanned several times
        ]

################################################################################################



def doTransmittanceCalibrationForS(hdf5file_path, hdf5FilepathOut):
    '''
    For fullscans: we just take the mean of the Sun region
    Loops have to be ordered like this: OCC(1file per occ) then ORDER then BIN
    #1: Spectra as read: _1
    #1: Background subtracted: _1  (same than previous step)
    #2: Spectra separated following the bins,Then if it is a double occultation, it is split _2  (bins will be returned in the same file)
    #2: Egress have to be inverted _2 (same than previous step)
    #3: Spectra are gathered for each bin but are in separated files for the different occultation _3
    '''
    with h5py.File(hdf5file_path, "r") as hdf5FileIn:
        #binTop_1 = hdf5FileIn["Science/Bins"][:,0]
        npixels = np.array(hdf5FileIn["Science/YNb"])[0]
        orders = np.array(hdf5FileIn["Channel/DiffractionOrder"])
        spectra_1 = np.array(hdf5FileIn["Science/Y"])
        NBins=hdf5FileIn.attrs['NBins']
        NSpec=hdf5FileIn.attrs['NSpec']
        alt_1 = np.array(hdf5FileIn['Geometry/Point0/TangentAltAreoid'])[:,1]
    """PARAMETERS DEFINITION"""
    altTmin=-8      #altitude between Transmittance and umbra: difference between aeroide and surface.
    #In case of S fullscans the background is subtrackted on-board
    uOrders=np.unique(orders)
    ''' Check if double occultation '''
    if alt_1[0] > 250.0 and alt_1[-1] > 250.0:
        logger.info("convert: %s is a double occultation.", hdf5file_path)
        nocc=range(2)
        if np.min(alt_1) < -100:
            occultationType = "merged"
        else:
            occultationType = "grazing" 
    else:
        nocc=[0]
        occultationType = "normal"
    """BEGIN LOOP OVER OCC FOR DOUBLE OCCULTATION"""
    hdf5FilepathOutLIST=[]
    for occ in nocc:
        #files have to be created in this loop
        #change filename if double occultation: 1 becomes 'I' or 'E'
        filename = os.path.basename(os.path.splitext(hdf5file_path)[0]).split('_')
        if len(nocc) == 2:
            if occ==0:
                filename[4] = 'I'
            elif occ==1:
                filename[4] = 'E'
        elif len(nocc) == 1:
            if alt_1[0]<alt_1[-1]: #Egress
                filename[4] = 'E'
            else:
                filename[4] = 'I'
        filename = '_'.join(filename)
        """LOOP OVER ORDERS"""
        dT_1=np.zeros_like((spectra_1))
        snr_1=np.zeros_like((spectra_1))
        allindT=[]
        IndBin=np.zeros_like((alt_1))
        for order in uOrders:
            if order in range(146,155) or order in range(167,168):
                altSmin=160 
            elif order in range(154,158):
                altSmin=180
            elif order in range(158,167):
                altSmin=200
            else: #if order in range(148,150) or order in range(164,170):
                altSmin=120 
            """GET INDICES OF BINS AND OCC"""
            indorder=[i for i,o in enumerate(orders) if o==order]
            # The bins are suposed to be ordered and that they always begin with the first one
            indtemp = [np.array(indorder[i::NBins]) for i in range(NBins)] #Gives the ordering of the first bins in the 0.3 file for the right order
            """CUT DOUBLE OCCULTATIONS"""
            #select spectra in case of double occ:
            ind=[[] for i in range(NBins)]
            if occultationType in ['merged','grazing']:
                #we take all the umbra part for each occultation
                for ibin in range(NBins):
                    if np.any(alt_1[indtemp[ibin]]==-999): # to be checked as some orders might be grazinf and other not.
                        inddark=[i for i,a in enumerate(alt_1[indtemp[ibin]]) if a==-999]
                    else:   #grazing 
                        inddark=[alt_1[indtemp[ibin]].shape[0]/2] #this is not a dark altitude
                    inddark=[int(i) for i in inddark] # convert in integers
                    if occ==0:
                        indcut=inddark[-1]
                        ind[ibin]=indtemp[ibin][:indcut]
                    if occ==1:
                        indcut=inddark[0]
                        ind[ibin]=indtemp[ibin][indcut:]
                        """INVERT EGRESS"""
                        ind[ibin]=ind[ibin][::-1]
            else:
                for ibin in range(NBins):
                    ind[ibin]=indtemp[ibin]
                    """INVERT EGRESS"""
                    if alt_1[0]<alt_1[-1]: #Egress
                        ind[ibin]=ind[ibin][::-1]
            """LOOK IF TRANSMITTANCES CAN BE COMPUTED"""
            # Only one spectrum
            if alt_1[ind[0]].shape[0]<2: #    #if one order.bin has less than 2 indexes, no transmittance can be estimated for that order
                comment="{0} order {2} occ {1} : Only one spectrum. No transmittance computed.".format(os.path.basename(hdf5file_path),occ,order)
                logger.info(comment)
                continue
            # No spectrum in T
            if np.min(alt_1[ind[0]])>=altSmin:
                comment="{0} order {2} occ {1} : No atmospheric spectrum recorded. No transmittance computed.".format(os.path.basename(hdf5file_path),occ,order)
                logger.info(comment)
                continue
            #No spectrum in S
            if np.max(alt_1[ind[0]])<altSmin:
                comment="{0} order {2} occ {1} : No solar spectrum recorded. No transmittance computed.".format(os.path.basename(hdf5file_path),occ,order)
                logger.info(comment)
                continue
            """BEGIN LOOP OVER BINS"""
            for ibin in range(NBins):
                ind_ibin=ind[ibin]
                spectra_2 = spectra_1[ind_ibin]
                alt_2 = alt_1[ind_ibin]
                """
                BEGINNING OF TRANSMITTANCE CALCULATIONS
                We work with index number, not altitude.
                indS, indT, indU, INDUNITY, smin, smax, tmin, umin in index of spectra_2
                """
                smin=np.argmin(np.abs(alt_2-altSmin))  # closest spectra index to 200km
                if alt_2[smin]<altSmin:                 # to be above 200km
                    smin=smin-1
                smax=0 
                tmin=np.argmin(np.abs(alt_2-altTmin))  # index of -8km
                if alt_2[tmin]<altTmin:
                    tmin=tmin-1
                # compute dU
                umin=len(alt_2)                        # index of the lower bound of Umbra
                if min(alt_2)<altTmin:
                    indU=range(tmin+1,umin)
                    umean=np.mean(spectra_2[indU])
                    dU=umean
                else:
                    # no spectrum in U so set dU=1
                    dU=np.ones_like(npixels)
                """
                Begin Transmittances computation
                """
                # indices wrt spectra
                indS=range(smax,smin+1)
                indT=range(smin+1,tmin+1)
                smean=np.mean(spectra_2[indS],axis=0)
                Tmean=np.divide(spectra_2[indT],smean)
                # compute Uncertainties
                dSmean=np.sqrt(smean)
                dPmean=(dSmean-dU)*np.sqrt(np.abs(Tmean))+dU
                dTmean=np.divide(np.sqrt(dPmean**2+Tmean**2*dSmean**2),smean)
                snr=np.divide(spectra_2[indT],dPmean)  
                """
                NO PLOTS
                """
                """SAVE T AND ALT_2"""
                indT_1=ind_ibin[indT]
                spectra_1[indT_1]=Tmean
                dT_1[indT_1]=dTmean
                snr_1[indT_1]=snr
                allindT.extend(indT_1)
                IndBin[indT_1]=ibin
            """
            End loop over bins
            """
        """
        End loop over orders
        """      
        """RECOMBINE BINS"""
        # This is the list of the index of spectra_1 that have to be saved.
        indrec=np.argsort(alt_1)
        indrec=[i for i in indrec if i in allindT]
        """SAVE FILE"""
        #Write a h5 file if no error occured
        #if not error:  # Produce an h5 file only if no error
        tableCreationDatetime = np.string_(time.strftime("%Y-%m-%dT%H:%M:%S"));
        YCalibRef_tmpl = "CalibrationTime=%s"
        YCalibRef = YCalibRef_tmpl % (tableCreationDatetime)
        YErrorRef_tmpl = "CalibrationTime=%s"
        YErrorRef = YErrorRef_tmpl % (tableCreationDatetime)
        filename=filename  + '.h5'
        hdf5FilepathOut = os.path.join(NOMAD_TMP_DIR, filename)
        with h5py.File(os.path.splitext(hdf5FilepathOut)[0] + '.h5', "w") as hdf5FileOut:
            hdf5FileOut.attrs["YCalibRef"] = YCalibRef
            hdf5FileOut.attrs["YErrorRef"] = YErrorRef
            NSpec=len(indrec)
            hdf5FileOut.attrs["NSpec"] = NSpec
            dset_path="Channel"
            generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
            dset_path="Science"
            generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
            dset_path="Geometry"
            generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
            dset_path="Geometry/Point0"
            generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
            hdf5FileOut.create_dataset("Channel/IndBin", dtype=np.uint16, data=IndBin[indrec],
                                       compression="gzip", shuffle=True)
    #                hdf5FileOut.create_dataset("Geometry/Point0/TangentAlt", dtype=np.float, data=alt_1[indrec],
    #                                           compression="gzip", shuffle=True)
            hdf5FileOut.create_dataset("Science/SNR", dtype=np.float32, data=snr_1[indrec],
                                       compression="gzip", shuffle=True)
            hdf5FileOut.create_dataset("Science/Y", dtype=np.float32, data=spectra_1[indrec],
                                       compression="gzip", shuffle=True)
            hdf5FileOut.create_dataset("Science/YError", dtype=np.float32, data=dT_1[indrec],
                                       compression="gzip", shuffle=True)
            hdf5FileOut.create_dataset("Science/YTypeFlag", dtype=np.uint16, data=2*np.ones((NSpec)),
                                       compression="gzip", shuffle=True)
            hdf5FileOut.create_dataset("Science/YUnitFlag", dtype=np.uint16, data=1*np.ones((NSpec)),
                                       compression="gzip", shuffle=True)
            hdf5FileOut.create_dataset("Science/YErrorFlag", dtype=np.uint16, data=2*np.ones((NSpec)),
                                       compression="gzip", shuffle=True)
            #on ouvre le 0p3j où on va chercher les autres datasets
            with h5py.File(hdf5file_path, "r") as hdf5FileIn:
                generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)
                for dset_path, dset in generics.iter_datasets(hdf5FileIn):
                    generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
                    if not dset_path in DATASETS_TO_BE_REMOVED:
                        if any([i in dset_path for i in DATASETS_NOT_TO_BE_RESHAPED_NOR_COMPRESSED]):
                            dsetcopy=np.array(dset)
                            hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy)
                        elif any([i in dset_path for i in DATASETS_NOT_TO_BE_RESHAPED]):
                            dsetcopy=np.array(dset)
                            hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy,compression="gzip", shuffle=True)
                        else:
    #                        print(dset_path)
                            dsetcopy=np.array(dset[...][indrec])
                            hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy,compression="gzip", shuffle=True)

#            with h5py.File(hdf5file_path, "r") as hdf5FileIn:
#                generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)
#                for dset_path, dset in generics.iter_datasets(hdf5FileIn):
#                    if dset_path in DATASETS_TO_BE_REMOVED:
#                        continue
#                    generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
#                    if any([i in dset_path for i in DATASETS_NOT_TO_BE_RESHAPED]):
#                        dsetcopy=np.array(dset)
#                        if any([i in dset_path for i in DATASETS_NOT_TO_BE_RESHAPED_COMPRESSED]):
#                            hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy)
#                        else:
#                            hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy,compression="gzip", shuffle=True)
#                    else:
#                        dsetcopy=np.array(dset[...][indrec])
#                        hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy,compression="gzip", shuffle=True)
        hdf5FilepathOutLIST.append(hdf5FilepathOut)    
    """
    End loop over occ
    """
    return hdf5FilepathOutLIST
