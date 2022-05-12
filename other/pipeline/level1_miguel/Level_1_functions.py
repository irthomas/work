# -*- coding: utf-8 -*-
"""
============================================================================================
7. CONVERT HDF5 LEVEL 0.3J TO LEVEL 1.0A

DONE:

TRANSMITTANCE ESTIMATION

STILL TO DO:

COMPUTE COVARIANCE PROPERLY FOR THE UNCERTAINTIES COMPUTATION

Filename definition:
For SO:
Date_Time_Level_Channel_Science_Type_Order.h5
  0   1    2      3       4       5    6

For UVIS:
Date_Time_Level_Channel_Type.h5
  0   1    2      3      4

============================================================================================
"""

import os
import re
import time
import logging

import numpy as np
import h5py
#from scipy.special import stdtrit
#from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
if True: # not to be used for TESTING
    import matplotlib
    matplotlib.use('Agg')

# from nomad_ops.config import ROOT_STORAGE_PATH
# import nomad_ops.core.hdf5.generic_functions as generics

ROOT_STORAGE_PATH = "."
import generic_functions as generics


__project__   = "NOMAD"
__author__    = "Loïc Trompet"
__contact__   = "loic.trompet@aeronomie.be"


logger = logging.getLogger( __name__ )

VERSION = 80
OUTPUT_VERSION = "1.0A"

#################################
# from pipeline_mappings_v04.py #
############################################################################

ATTRIBUTES_TO_BE_REMOVED = [
        'YCalibRef',
        'YErrorRef',
        'NSpec',
        'Name',
        'Date',
        'Time',
        'Timestamp',
        ]

def dsnottoreshape(num,channel):
    """ Those datasets must not be reshaped when writting the new file
    for the next level """
    if channel == 'UVIS':
        dts = [
            'TM11/',
            'TM29/',
            'Science/BadPixelMap',
            'PointXY',  # housekeeping has to be reshaped for UVIS
            'PacketSize',
#            'DateTime',
            'Science/CircuitNoise',
            ]
    elif channel in ['SO','LNO'] :
        if num == 1:
            dts = [
                'Science/BadPixelMap',
                'PointXY',
                'Housekeeping/',
#                'PacketSize',
#                'DateTime',
                ]
        else:
            dts = [
                'Science/BadPixelMap',
                'PointXY',
                'Housekeeping/',
#                'DateTime',
                ]
    return dts

DS_NOT_TO_BE_RESHAPED_NOR_COMPRESSED = [
        'Telecommand20/',
        'QualityFlag/',
        'Temperature/',
        "Channel/VStart_YMaskROI_0b",
        ]


DS_TO_BE_REMOVED = [
        "Science/Y",
        "Science/YUnmodified",
        "Science/YTypeFlag",
        "Science/YUnitFlag",
        "Science/YError",
        "Science/YErrorFlag",
        "Science/YErrorRandom",
        "Science/YErrorSystematic",
        "Science/XNbBin",
        "Science/Xpix_1b",

        'Channel/PacketSize',
        'BkgTimestamp',
        # darks no more reshaped: problems for cases where 1 order scanned
        # several times and cases where dark scanned several times
        'Science/BackgroundY',
        ]


#####################################################################################



def prepare_fig_tree(figName):
    """ Prepare forders for figures """
    channel = figName.split('_')[3]
    # Move to config
    PATH_TRANS_LINREG_FIG = os.path.join(ROOT_STORAGE_PATH,
                                         "thumbnails_1p0a_trans", channel)
    m = re.match(r"(\d{4})(\d{2})(\d{2}).*", figName)
    year = m.group(1)
    month = m.group(2)
    path_fig = os.path.join(PATH_TRANS_LINREG_FIG, year, month)
    if not os.path.isdir(path_fig):
        os.makedirs(path_fig, exist_ok=True)
    return os.path.join(path_fig, figName)


# functions for dark removal
def multiInterp2(ts, bts, dark,NBins):
    ''' Interpolation of the darks from bts to ts '''
    newdark = np.zeros((ts.size,dark.shape[1]))
    # warning: 4 bins: 4 times the same timestamp
    for i in range(NBins):
        indx = np.arange(i,ts.size,NBins)
        x = ts[indx]
        indxp = np.arange(i,bts.size,NBins)
        xp = bts[indxp]
        fp = dark[indxp]
        #i = np.arange(x.size)
        # We have to add an element at the beginning of xp
        xp = np.insert(xp,0,2*xp[0]-xp[1])   #xp[0]-(xp[1]-xp[0])=2*xp[0]-xp[1]]]
        j = np.searchsorted(xp, x) - 1
        d = (x - xp[j]) / (xp[j + 1] - xp[j])
        #d = (x[1:]-xp[:-1]) / (xp[1:] - xp[:-1])
        fp = np.insert(fp,0,2*fp[0]-fp[1],0)
        fp = fp.transpose(1,0)
        newfp = (1 - d) * fp[:,j] + fp[:,j+1] * d
        #newfp = (1 - d) * fp[:,:-1] + fp[:,1:] * d
        newdark[indx] = newfp.transpose(1,0)
    return newdark

def FindNum(h5file_in_path):
    ''' Gives the ratio of number of orders scanned over the number of background
    variable for SO and LNO.
    Used by WriteFile
    '''
    channel = os.path.basename(h5file_in_path).split('_')[3]
    if channel == "UVIS" :
        num = 1
    elif channel in ['SO','LNO']:
        with h5py.File(h5file_in_path, "r") as fh5:
            #if bg subtraction on
            bckg_sub = fh5['Channel/BackgroundSubtraction'][0]
        if bckg_sub:
            num = 1
        else:
            with h5py.File(h5file_in_path, "r") as fh5:
                ts = np.array(fh5['Timestamp'])
                bkgts = np.array(fh5['BkgTimestamp'])
            num = ts.size/bkgts.size
    return num

def FindDark(h5file_in_path):   #can be used for SO and LNO
    ''' Get the darks interpolated '''
    with h5py.File(h5file_in_path, "r") as fh5:
        NBins = fh5.attrs['NBins']
        npixels = np.array(fh5['Science/YNb'])[0]
        dark = np.array(fh5['Science/BackgroundY'])
        #interp on dark has there is maybe a problem of straylight of 1% of the signal
        ts = np.array(fh5['Timestamp'])
        bkgts = np.array(fh5['BkgTimestamp'])
        num = np.array(fh5["Channel/NumberOrderScanned"])[0]
    # multiInterp2 works if same shape ts and bkgts
    # thus problem if an order is scanned several times
    # need to mnage if num!=1:
    if num > 1:
        intdark = np.zeros((ts.size,npixels))
        # If 2 orders (num=2): 4 bins o1 + 4 bins o2 +4 bins o1 + ...
        # we want first  0 1 2 3 8 9 10 11 ...
        #then            4 5 6 7 12 13 14 15
        #thus step for same bin: 2*NBins
        step = NBins*num
        for i in np.arange(num):
            ind =  []
            for j in np.arange(NBins):
                ind.extend(np.arange(NBins*i+j,ts.size,step))
            ind = [int(i) for i in ind]
            ind.sort()
            intdark[ind] = multiInterp2(ts[ind], bkgts, dark, NBins)
    elif num < 1:
        # when several times dark are scanned:
        # make a linear interpolation using oll the values 1 s before and 1 sec after
        invnum = int(bkgts.size/ts.size)
        intdark = np.zeros((ts.size,npixels))
        # on doit changer bkgts et dark pour chaque bin
        for ibin in np.arange(NBins):
            ind = ibin+np.arange(0,bkgts.size,NBins)
            bkgtstemp = bkgts[ind]
            darktemp = dark[ind]
            indts = ibin+np.arange(0,ts.size,NBins)
            tstemp = ts[indts]
            indin = [int(i) for i in np.searchsorted(bkgtstemp, tstemp)]
            intdarkbin = np.zeros((tstemp.size,npixels))
            for i,ii in enumerate(indin):
                if ii == 0:
                    coeffs = np.polyfit(bkgtstemp[ii:ii+invnum],darktemp[ii:ii+invnum],1)
                    intdarkbin[i] = np.polyval(coeffs,np.tile(tstemp[i],(npixels,1)).T)
                elif ii == bkgtstemp.size:
                    coeffs = np.polyfit(bkgtstemp[ii-invnum:ii],darktemp[ii-invnum:ii],1)
                    intdarkbin[i] = np.polyval(coeffs,tstemp[i])
                else:
                    coeffs = np.polyfit(bkgtstemp[ii-invnum:ii+invnum],
                                      darktemp[ii-invnum:ii+invnum],1)
                    intdarkbin[i] = np.polyval(coeffs,tstemp[i])
            intdark[indts] = intdarkbin
    else:
        intdark = multiInterp2(ts, bkgts, dark, NBins)
    return intdark
# end functions for dark removal


def get_parameters(channel,order):
    '''
    Get general parameters for channel and order.
    altRmax is the altitude where the atmosphere begins.
    '''
    if channel == 'UVIS':
        altSmin = 120    #altitude under which we expect to see lines
        altRmax = 150    #definition of the altitude where the atmosphere begins
        threshold = 0.7  #threshold for the criteria
        nfit = 2         #fit polynomial degree
        snrmin = 100.     # to be changed but keep float
        psmin = 20      #minimum number of points mandatory in S
        pumin = 4
        factordt = 2
    elif channel in ['SO','LNO']:
        # S contains R, altSmin is the lower bound of S and R
        if order in range(146,155) or order in range(167,168):
            altSmin = 160
            altRmax = 200
        elif order in range(155,158):
            altSmin = 180
            altRmax = 220
        elif order in range(158,167):
            altSmin = 200
            altRmax = 230
        else: #if order in range(148,150) or order in range(164,170):
            altSmin = 120
            altRmax = 150
        threshold = 0.8  #threshold for the criteria
        nfit = 2         #fit polynomial degree
        snrmin = 200.     # to be changed but keep float
        psmin = 20      #minimum number of spectra in S
        pumin = 4
        factordt = 2
    return altSmin,altRmax,threshold,nfit,snrmin,psmin,pumin,factordt


def Compute_dP(dS,dU,T):
    """ Compute uncertainties on signal """
    return (dS-dU)*np.abs(T)+dU

def Compute_dT(dP,T,dS,sint,r):
    """ Compute uncertainties on transmittances """
    #return np.sqrt(dP**2+(T*dS)**2)/sint
    return np.sqrt(dP**2 + (T*dS)**2 - 2*r*T*dP*dS)/sint

def compute_uncertainties(pixnorm, channel,
                          indS, indT, indR, lenS, lenT, lenR, spectra_2, params,
                          dPuvis, dPuvisMR, dPuvisMS, initdU):
    """ Compute all uncertainties for the different datasets. """
    # Taking into account the uncertainty on the slope
    # for T and Tfit, NOT FOR TMEAN
    #xfit = np.linspace(0,1,lenS+lenT)
    # warning: not indS as we have less spectra than
    # the total number of spectra in whole S region
    #xfitS = np.tile(xfit[:lenS],(npixels,1)).T
    #xfitT = np.tile(xfit[lenS:],(npixels,1)).T
    #xmean = xfitS.mean()
    # Get proba from student-t distribution
    # This is to get an interval of confidence not std
    #alpha = 0.2 # should be 0.317 ?
    #stut = stdtrit(lenS, 1 - alpha) #for lenS=inf at 80% confidence
    # gives back t, first argument is the degree of freedom,
    # second argument is the integral from -inf to t
    # but we should multiply dS_slope by sqrt(lenS) to get std
    # otherwhise it is a standard error !
    #sqrt_extrap = np.sqrt(1+(1/lenS)+((xfitT-xmean)**2
    #                                    /(np.sum((xfitS-xmean)**2))))
    #sqrt_extrap *= np.sqrt(lenS) # to get std and not ste

    # For T (linear regressions)
    coeffs = np.polyfit(indS, spectra_2[indS],params[3]-1) # fit wrt xfit and not tgalt
    sint = np.polyval(coeffs, np.asarray(indT)[:,None]) # projection of S in T
    T = np.divide(spectra_2[indT], sint)
    # For Tmean (no linear regression):
    smean = np.mean(spectra_2[indS], axis=0)
    Tmean = spectra_2[indT]/smean
    # For R (R is in S, not in T !)
    sinr = np.polyval(coeffs,np.asarray(indR)[:,None])
    R = spectra_2[indR]/sinr
    # compute dS and dSmean
    if channel in ['SO', 'LNO']:
        # need to normalize S before to compute dS
        Sn = spectra_2[indS] - np.mean(spectra_2[indS][:,pixnorm], axis=1)[:,None]
        dSmean = np.std(Sn, axis=0, ddof=1)
        # dS (called dSnlr before)
        coeffsSn = np.polyfit(indS, Sn, params[3]-1)
        Snfit = np.polyval(coeffsSn, np.asarray(indS)[:,None])
        dS = np.tile(np.sqrt(np.sum((Sn-Snfit)**2, axis=0)
                              /(lenS-2)), (lenT,1))
    elif channel == 'UVIS':
        # Uncertainties on the reg. lin. so need to use std and not dPuvis !
        dSmean = np.std(spectra_2[indS], axis=0, ddof=1)
        Sfit = np.polyval(coeffs, np.asarray(indS)[:,None])
        dS = np.tile(np.sqrt(np.sum((spectra_2[indS]-Sfit)**2, axis=0)
                         /(lenS-2)),(lenT,1))
    #dS_slope = dS*sqrt_extrap

    dSr = np.tile(np.sqrt(np.sum((spectra_2[indR]-sinr)**2, axis=0)
                        /(lenS-2)),(lenR,1))
    #must be a vector of length 320

    # Compute all dP
    if channel == 'UVIS':
        # for UVIS the YError given in 0.3c are
        # already to the square !
        dPr = np.sqrt(dPuvis[indR]) # dPr is for the criteria in R
        dPUVIS = np.sqrt(dPuvis[indT])
        dP = dPUVIS
        dPmean = dPUVIS
        dPmeanMR = np.sqrt(dPuvisMR[indT])
        dPmeanMS = np.sqrt(dPuvisMS[indT])
    elif channel in ['SO', 'LNO']:
        dU = np.tile(initdU,(lenT,1))
        dUr = np.tile(initdU,(lenR,1))
        dPr = Compute_dP(dSr,dUr,R)
        dPmean = Compute_dP(dSmean,dU,Tmean)
        #dP = np.sqrt(np.abs(spectra_2[indT]))
        #dP = np.sqrt(((dS**2-dU**2)*np.abs(T))+dU**2)
        dP = Compute_dP(dS,dU,T)
    # correlation coeff between P and S, not dP and dS !
    #pearsonr_coeff = np.array([pearsonr(spectra_2[indS,i]-spectra_2[indS[0],i],
    #                        Sfit[:,i]-Sfit[0,i])[0] for i in range(npixels)])
    pearsonr_coeff = 0.
    # Compute dT
    #Check for R region which is used to compute the coeffs and
    #to check the criteria 1 to 3
    dTr = Compute_dT(dPr,R,dSr,sinr,pearsonr_coeff)
    dTmean = Compute_dT(dPmean,Tmean,dSmean,smean,pearsonr_coeff)
    dT = Compute_dT(dP,T,dS,sint,pearsonr_coeff)
    #  ! : Now dT is norm and does not consider dS_slope !
    if channel == 'UVIS':
        dTmeanMR = Compute_dT(dPmeanMR, Tmean, dSmean, smean, pearsonr_coeff)
        dTmeanMS = Compute_dT(dPmeanMS, Tmean, dSmean, smean, pearsonr_coeff)
    else:
        dTmeanMR = None
        dTmeanMS = None
    return coeffs, dP, R, dTr, dSmean, smean, T, dT, Tmean, dTmean, dTmeanMR, dTmeanMS


def TransmittancesAlgo(h5file_in_path,h5file_out_path,make_plots=False):
    ''' MAIN function '''
    logger.info("Calculating transmittance for %s",h5file_in_path)
    filename = os.path.basename(h5file_in_path)
    channel = filename.split('_')[3]
    if channel == 'UVIS':
        with h5py.File(h5file_in_path, "r") as fh5:
            dPuvis_1 = np.array(fh5['Science/YError'])
            dPuvisMR_1 = np.array(fh5['Science/YErrorRandom'])
            dPuvisMS_1 = np.array(fh5['Science/YErrorSystematic'])
            # spectra_1 has to be taken now to remove the dark if necessary
            spectra_1 = np.array(fh5['Science/Y'])
            npixels = int(fh5['Science/YNb'][0])
            X = np.array(fh5['Science/X']) # for plots
            alt_1 = np.mean(np.array(fh5['Geometry/Point0/TangentAltAreoid']),axis=1)
            NSpec = fh5.attrs['NSpec']
        pixnorm = np.arange(0,npixels)
        bckgSub = True
        order = -999
        NBins = 1
        critpix = np.arange(int(npixels/4),int(3*npixels/4))
    elif channel in ['SO','LNO']:
        with h5py.File(h5file_in_path, "r") as fh5:
            order = fh5['Channel/DiffractionOrder'][0]
            bckgSub = fh5['Channel/BackgroundSubtraction'][0]==1
            bckgInFile = 'Science/BackgroundY' in fh5
            spectra_1 = np.array(fh5['Science/Y'])
            npixels = fh5['Science/YNb'][0]
            X = np.array(fh5['Science/X']) # for plots
            NBins = fh5.attrs['NBins']
            NSpec = fh5.attrs['NSpec']
            alt_1 = np.mean(np.array(fh5['Geometry/Point0/TangentAltAreoid']),
                            axis=1)
        pixnorm = np.arange(0,npixels)
        critpix = range(70,300)
        # BACKGROUND SUBTRACTION
        #Dark removal should be done only if 5 orders not 6
        if not bckgSub:
            if bckgInFile:
                dark = FindDark(h5file_in_path)
                spectra_1 -= dark
            else:
                comment = ('{0} : Background should be subtracted but no background '
                         'file found. No transmittance estimated.').format(h5file_in_path)
                logger.error(comment)
    params = get_parameters(channel,order)
    #params contains altSmin,altRmax,threshold,nfit,snrmin,psmin,pumin,factordt
    spectra_raw_1 = np.copy(spectra_1)

    acceptfirstcalc = False
    if not np.any(np.logical_and(alt_1 > 60, alt_1 < 140)): # science 2 case
        acceptfirstcalc = True

    error = 0

    IndBin = np.zeros_like(alt_1)
    dT_1 = np.zeros_like((spectra_1))
    snr_1 = np.zeros_like((spectra_1))
    allindT = []
    Coeffs = np.zeros((alt_1.size,params[3],npixels))
    Tmean_1 = np.zeros_like((spectra_1))
    dTmean_1 = np.zeros_like((spectra_1))
    if channel == 'UVIS':
        dTmeanMR_1 = np.zeros_like((spectra_1))
        dTmeanMS_1 = np.zeros_like((spectra_1))
    else:
        dTmeanMR_1 = None
        dTmeanMS_1 =  None
    # BEGIN LOOP OVER BINS
    ncrit = 2
    tCalculated = False
    binaccepted = np.zeros((NBins))
    SRegIndex = np.zeros((NBins,2))  # in index
    SRegAlt = np.zeros((NBins,2))    # in alt
    Criteria = np.zeros((NBins,ncrit))
    altTmin = -10    # the slit height projected on tangent altitude is 5 km
    for ibin in range(NBins):
        ind_ibin = np.arange(ibin,NSpec,NBins)
        # INVERT EGRESS
        if alt_1[0] < alt_1[-1]: #Egress
            ind_ibin = ind_ibin[::-1]
        spectra_2 = spectra_1[ind_ibin]
        alt_2 = alt_1[ind_ibin]
        if channel == 'UVIS':
            dPuvis = dPuvis_1[ind_ibin]
            dPuvisMR = dPuvisMR_1[ind_ibin]
            dPuvisMS = dPuvisMS_1[ind_ibin]
        else:
            dPuvis = None
            dPuvisMR = None
            dPuvisMS = None
        # LOOK FOR SPECIAL CASES
        #sequence technologique
        if np.min(alt_2) >= params[0]:
            comment = ('{0} bin {1} : No atmosphere spectra recorded. '
                'No transmittance estimation.').format(filename,ibin)
            logger.info(comment)
            error = 1
            lenT = 1
            T = -999*np.ones((lenT,npixels))
            Tmean = -999*np.ones((lenT,npixels))
            continue
        #No spectrum outside the atmosphere
        if np.max(alt_2) < params[0]:
            comment=('{0} bin {1} : does not contain any valid spectra outside '
                     'the atmosphere. No transmittance calculated.').format(filename,ibin)
            logger.info(comment)
            error = 1
            lenT = 1
            T = -999*np.ones((lenT,npixels))
            Tmean = -999*np.ones((lenT,npixels))
            continue
        # BEGINNING OF THE ALGORITHM
        # We work with index number, not altitude
        # S contains R, altSmin is the lower bound of S and R
        smin = np.argmin(np.abs(alt_2-params[0]))  # closest spectra index to 200km
        if alt_2[smin] < params[0]:                 # to be above 200km
            smin = smin-1
        smax = 0
        #test signal enough strong
        indS = range(smax,smin+1)
        moyS = np.mean(spectra_2[indS][:,int(npixels/2.)])
        stdS = np.sqrt(moyS)
        if moyS < 2 * stdS:
            comment = ('The signal in the Sun part of {0} bin {1} is too weak ( < 2*noise).'
                     ' No transmittance calculated').format(h5file_in_path,ibin)
            logger.info(comment)
            error = 1
            lenT = 1
            T = -999*np.ones((lenT,npixels))
            Tmean = -999*np.ones((lenT,npixels))
            continue
        tmin = np.argmin(np.abs(alt_2-altTmin))  # index of 0km
        if alt_2[tmin]<altTmin:
            tmin = tmin-1
        umin = len(alt_2)                        # index of the lower bound of Umbra
        lenallS = smin+1-smax
        Rmax = np.argmin(np.abs(alt_2-params[1]))
        indR = range(Rmax,smin)
        lenR = len(indR)
        #indT has to be defined here to avoid problem when recombining all bins.
        indT = range(smin+1,tmin+1)
        lenT = len(indT)
        if np.where(alt_2 < altTmin)[0].shape[0] > params[6]:
            indU = range(tmin+1, umin)
            initdU = np.std(spectra_2[indU] -
                             np.mean(spectra_2[indU][:,pixnorm],axis=1)[:,None],
                             axis=0, ddof=1)
        elif not bckgSub:
            #if no spectra under 0 km of tangent altitude, use the darks
            dark = FindDark(h5file_in_path)
            initdU = np.std( dark[ind_ibin] -
                          np.mean(dark[ind_ibin][:,pixnorm],axis=1)[:,None] ,axis=0, ddof=1)
        else:                  ### avoid this for UVIS !!!!!!
            comment = ('{0}: Grazing occultation with on-board background subtraction. '
                   'Noise computed with dU=0.').format(filename)
            logger.info(comment)
            initdU = np.zeros((npixels))
        # Begin loop over smax
        bestSumc = 0
        accepted = False
        tCalculatedForBin = False
        crit = np.zeros((ncrit))
        while not accepted:
            if smin-smax<100:
                stepmax = 2
            elif smin-smax<1000:
                stepmax = 4
            else:
                stepmax = 8
            # indices wrt spectra
            indS = range(smax,smin+1)
            lenS = len(indS)
            # reduce R to S if S smaller than R
            if lenR > lenS:
                indR = indS
                lenR = lenS
            if lenS >= params[5] or acceptfirstcalc:
                #compute T, tr + criteria
                tCalculatedForBin = True
                tCalculated = True
                SRegIndex[ibin,0] = smin
                SRegIndex[ibin,1] = smax
                SRegAlt[ibin,0] = alt_2[smin]
                SRegAlt[ibin,1] = alt_2[smax]

                (coeffs, dP, R, dTr, dSmean, smean, T, dT, Tmean, dTmean, dTmeanMR,
                 dTmeanMS) = compute_uncertainties(pixnorm, channel,
                          indS, indT, indR, lenS, lenT, lenR, spectra_2, params,
                          dPuvis, dPuvisMR, dPuvisMS, initdU)

                # Compute SNR
                snr = np.divide(spectra_2[indT],dP)

                crit[0] = np.mean(np.abs(1-R[:,critpix]) < params[7]*dTr[:,critpix])
                #crit[1] = np.mean(np.mean(dTr<(1./params[4]),0)[critpix]) # no more useful
                #not exactly like for SOIR # no more useful
                #crit[2] = np.mean(np.mean(dTr<params[7]*np.std(R,ddof=1),axis=0) [critpix]
                # Need a test to be sure that the noise is not too big
                crit[1] = np.mean(dSmean[critpix] < params[7]*np.sqrt(smean[critpix]))
                #crit[3] = np.mean(T[:,critpix]-1<params[7]*dT[:,critpix]) # no more useful
                #crit[4] = np.mean(np.abs(1-T[0,critpix])<params[7]*dT[0,critpix])
                if acceptfirstcalc:
                    accepted = True
                else:
                    accepted = crit[0] > params[2] and crit[1] > params[2]
                if not accepted:
                    sumc = np.sum(crit+lenS/lenallS) # add size of S to criteria
                    if bestSumc < sumc:
                        bestSumc = sumc
                        bestT = T
                        bestTmean = Tmean
                        bestdTmean = dTmean
                        bestdT = dT
                        bestsnr = snr
                        bestsmin = smin
                        bestsmax = smax
                        bestcoeffs = coeffs
                        bestindT = indT
                        bestindR = indR
                        bestcrit = crit
                        if channel == 'UVIS':
                            bestdTmeanMR = dTmeanMR
                            bestdTmeanMS = dTmeanMS
                    smax += stepmax
                else:
                    binaccepted[ibin] = 1
                    break
            else:
                #rejected
                comment = ('{0} bin {1}: NOT ACCEPTED. Best transmittance estimated '
                         'for this set of spectra saved.').format(filename,ibin)
                logger.info(comment)
                if 'bestT' in vars():
                    T = bestT
                    Tmean = bestTmean
                    dTmean = bestdTmean
                    dT = bestdT
                    snr = bestsnr
                    coeffs = bestcoeffs
                    indT = bestindT  #necessary to save the data
                    indR = bestindR
                    SRegIndex[ibin,0] = bestsmin
                    SRegIndex[ibin,1] = bestsmax
                    SRegAlt[ibin,0] = alt_2[smin]
                    SRegAlt[ibin,1] = alt_2[smax]
                    crit = bestcrit
                    if channel == 'UVIS':
                        dTmeanMR = bestdTmeanMR
                        dTmeanMS = bestdTmeanMS
                break
        # PLOTS
        if make_plots:
            if tCalculatedForBin:
                if accepted:
                    plotTitle = filename + " bin {0}".format(ibin) + " ACCEPTED"
                else:
                    plotTitle = filename + " bin {0}".format(ibin) + "NOT ACCEPTED"
                #graphe REG
                try:
                    PlotReg(plotTitle,alt_2,npixels,spectra_2,indS,coeffs,indR,
                        indT,ibin,acceptfirstcalc,filename)
                except Exception:
                    pass # Not a problem if the plot could not be made.
                #graphe transmittance
                try:
                    PlotTr(X, ind_ibin, indT, T, plotTitle, ibin, critpix, filename)
                except Exception:
                    pass
            else:
                plotTitle=filename + " bin {0}".format(ibin) + " NOT ACCEPTED"
                try:
                    PlotRegTNotCalculated(plotTitle,alt_2,spectra_2,npixels,ibin,filename)
                except Exception:
                    pass
        # SAVE T AND ALT_2
        if tCalculatedForBin:   # to be replaced by "if accepted:"
            indT_1 = ind_ibin[indT] # need to find back the index in spectra_1
            spectra_1[indT_1] = T
            Tmean_1[indT_1] = Tmean
            dTmean_1[indT_1] = dTmean
            dT_1[indT_1] = dT
            snr_1[indT_1] = snr
            allindT.extend(indT_1)    # LIGNE LA PLUS IMPORTANTE
            IndBin[indT_1] = ibin
            Coeffs[indT_1] = coeffs
            Criteria[ibin] = crit
            if channel == 'UVIS':
                dTmeanMR_1[indT_1] = dTmeanMR
                dTmeanMS_1[indT_1] = dTmeanMS
    # End loop over bins

    #SAVE FILE
    if tCalculated and not error:
        # Produce an h5 file only if no error
        # faire un dictionnaire au lieu de toutes ces variables
        h5file_out_path = WriteFile(alt_1, allindT, params,
                        IndBin, Coeffs, binaccepted, SRegIndex, SRegAlt,
                        Criteria, snr_1, spectra_1, spectra_raw_1,
                        Tmean_1, dTmean_1, dTmeanMR_1, dTmeanMS_1, dT_1,
                        h5file_in_path, h5file_out_path)
        h5file_out_path = [h5file_out_path]
    else:
        h5file_out_path = []
    return h5file_out_path

def PlotReg(plotTitle, alt_2, npixels, spectra_2, indS, coeffs, indR, indT, ibin,
            acceptfirstcalc, filename):
    """ Plot regression computed """
    plt.ioff()
    plt.figure()
    altplot = np.matrix.copy(alt_2)
    altplot[altplot==-999.0] = -20.0
    #plot first pixel middle with labels
    pix = int(npixels/2.)
    plt.plot(altplot, spectra_2[:,pix], 'k', label='Signal')
    plt.plot(altplot[indS], np.polyval(coeffs[:,pix],indS), 'g', label='S (reg.)')
    if acceptfirstcalc:
        indR = 0
    plt.plot(altplot[indR], np.polyval(coeffs[:,pix], indR), 'r', label='R (Test)')
    plt.plot(altplot[indT], np.polyval(coeffs[:,pix], indT), 'm', label='E (extrap.)')
    plt.text(altplot[0], spectra_2[0,pix], 'pixel ' + str(pix))
    #plot other pixels
    pixtoplot = [int(i) for i in np.linspace(20, npixels-20, 5)]
    for pix in pixtoplot:
        # be careful: indE in index of T !!!
        plt.plot(altplot, spectra_2[:,pix], 'k')
        plt.plot(altplot[indS], np.polyval(coeffs[:,pix], indS), 'g')
        plt.plot(altplot[indR], np.polyval(coeffs[:,pix], indR), 'r')
        plt.plot(altplot[indT], np.polyval(coeffs[:,pix], indT), 'm')
        plt.text(altplot[0], spectra_2[0,pix], 'pixel ' + str(pix))
    plt.title(plotTitle)
    plt.xlabel("Altitude [km]")
    plt.ylabel("Signal [ADU]")
    plt.grid()

    #fontP = FontProperties()
    #fontP.set_size('small')
    plt.legend()#bbox_to_anchor=(1.01,0.5), loc='upper right', prop=fontP)

    figName = os.path.splitext(filename)[0] + "_bin{0}_REG.png".format(ibin)
    fig_dest_path = prepare_fig_tree(figName)
    plt.savefig(fig_dest_path, bbox_inches='tight') # important
    plt.close()

def PlotRegTNotCalculated(plotTitle,alt_2,spectra_2,npixels,ibin,filename):
    """ Plot regression if T could not be computed """
    plt.ioff()
    plt.figure()
    altplot = np.matrix.copy(alt_2)
    altplot[altplot==-999.0] = -20.0
    pix = int(npixels/2.)
    plt.plot(altplot,spectra_2[:,pix],'k',label='Signal')
    plt.title(plotTitle)
    plt.xlabel("Altitude [km]")
    plt.ylabel("ADU")
    plt.grid()
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(bbox_to_anchor=(1.01,0.5),loc='upper right',prop=fontP)
    figName = os.path.splitext(filename)[0] + "_bin{0}_REG.png".format(ibin)
    fig_dest_path = prepare_fig_tree(figName)
    plt.savefig(fig_dest_path, bbox_inches='tight') # important
    plt.close()

def PlotTr(X,ind_ibin,indT,T,plotTitle,ibin,critpix,filename):
    """ Plot transmittances """
    plt.ioff()
    plt.figure()
    plt.plot(X[ind_ibin[indT]][:,critpix].T,T[:,critpix].T)
    plt.title(plotTitle)
    channel = filename.split('_')[3]
    if channel in ['SO','LNO']:
        plt.xlabel("Wavenumber [cm-1]")
    elif channel == 'UVIS':
        plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmittance")
    figName = os.path.splitext(filename)[0] + "_bin{0}_TR.png".format(ibin)
    fig_dest_path = prepare_fig_tree(figName)
    #plt.show()
    plt.savefig(fig_dest_path,bbox_inches='tight') # important
    plt.close()

def WriteFile(alt_1, allindT, params, IndBin, Coeffs, binaccepted, SRegIndex, SRegAlt,
              Criteria, snr_1, spectra_1, spectra_raw_1, Tmean_1, dTmean_1, dTmeanMR_1,
              dTmeanMS_1, dT_1, h5file_in_path, h5file_out_path):
    """ Write new file for next level """
    #RECOMBINE BINS
    # select the index of alt_1 corresponding to the current occ
    indrec = np.argsort(alt_1)
    indrec = [i for i in indrec if i in allindT]
    # This is the list of the index of spectra_1 that have to be saved.

    tableCreationDatetime = np.string_(time.strftime("%Y-%m-%dT%H:%M:%S"))

    YCalibRef_tmpl = "CalibrationTime=%s; Threshold=%s; snrmin=%s; altSmin=%s; altRmax=%s"
    YCalibRef = YCalibRef_tmpl % (tableCreationDatetime,params[2],params[4],
                                  params[0],params[1])
    YErrorRef_tmpl = "CalibrationTime=%s"
    YErrorRef = YErrorRef_tmpl % (tableCreationDatetime)

    with h5py.File(h5file_out_path, "w") as hdf5FileOut:

        hdf5FileOut.attrs["YCalibRef"] = YCalibRef
        hdf5FileOut.attrs["YErrorRef"] = YErrorRef
        NSpec=len(indrec)
        hdf5FileOut.attrs["NSpec"] = NSpec
        channel = os.path.basename(h5file_in_path).split('_')[3]
        if channel in ['SO','LNO']:
            indtype = 5
        elif channel == 'UVIS':
            indtype = 4
        obstype = os.path.basename(h5file_out_path).split('_')[indtype]
        hdf5FileOut.attrs["ObservationType"] = obstype

        dset_path = "Channel"
        generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
        dset_path = "Science"
        generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
        dset_path = "Criteria"
        generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])

        hdf5FileOut.create_dataset("Channel/IndBin", dtype=np.uint16, data=IndBin[indrec],
                                   compression="gzip", shuffle=True)

        hdf5FileOut.create_dataset("Criteria/Transmittance/RegLin", dtype=np.float32,
                                   data=Coeffs[indrec],compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Criteria/Transmittance/BinAccepted",
                                   dtype=np.uint16, data=binaccepted,
                                   compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Criteria/Transmittance/SRegIndex",
                                   dtype=np.uint16, data=SRegIndex,
                                   compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Criteria/Transmittance/SRegAlt",
                                   dtype=np.float32, data=SRegAlt,
                                   compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Criteria/Transmittance/Criteria",
                                   dtype=np.float32, data=Criteria,
                                   compression="gzip", shuffle=True)

        hdf5FileOut.create_dataset("Science/SNR", dtype=np.float32, data=snr_1[indrec],
                                   compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/Y", dtype=np.float32, data=spectra_1[indrec],
                                   compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/YMean", dtype=np.float32, data=Tmean_1[indrec],
                                   compression="gzip", shuffle=True)

        hdf5FileOut.create_dataset("Science/YError", dtype=np.float32, data=dT_1[indrec],
                                   compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/YErrorMean", dtype=np.float32,
                                   data=dTmean_1[indrec],
                                   compression="gzip", shuffle=True)
        if channel=='UVIS':
            hdf5FileOut.create_dataset("Science/YErrorMeanRandom", dtype=np.float32,
                                       data=dTmeanMR_1[indrec],compression="gzip",
                                       shuffle=True)
            hdf5FileOut.create_dataset("Science/YErrorMeanSystematic", dtype=np.float32,
                                       data=dTmeanMS_1[indrec],compression="gzip",
                                       shuffle=True)

        hdf5FileOut.create_dataset("Science/YTypeFlag", dtype=np.uint16,
                                   data=2*np.ones((NSpec)),
                                   compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/YUnitFlag", dtype=np.uint16,
                                   data=1*np.ones((NSpec)),
                                   compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/YErrorFlag", dtype=np.uint16,
                                   data=2*np.ones((NSpec)),
                                   compression="gzip", shuffle=True)
        hdf5FileOut.create_dataset("Science/SortIndices", dtype=np.uint16,
                                   data=indrec,
                                   compression="gzip", shuffle=True)

        hdf5FileOut.create_dataset("Science/YUnmodified", dtype=np.float32,
                                   data=spectra_raw_1[indrec],
                                   compression="gzip", shuffle=True)

        # take other datasets from previous level and slice them correclty if necessary.
        with h5py.File(h5file_in_path, "r") as fh5:
            generics.copyAttributesExcept(
                fh5, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)
            for dset_path, dset in generics.iter_datasets(fh5):
                generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
                if not dset_path in DS_TO_BE_REMOVED:
                    if any([i in dset_path
                            for i in DS_NOT_TO_BE_RESHAPED_NOR_COMPRESSED]):
                        hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype,
                                                   data=np.array(dset))
                    elif any([i in dset_path
                              for i in dsnottoreshape(
                                      FindNum(h5file_in_path),channel)]):
                        hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype,
                                                   data=np.array(dset),compression="gzip",
                                                   shuffle=True)
                    else:
                        hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype,
                                                   data=np.array(dset[...][indrec]),
                                                   compression="gzip", shuffle=True)
    return h5file_out_path