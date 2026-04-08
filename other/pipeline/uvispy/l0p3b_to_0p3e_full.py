# -*- coding: utf-8 -*-
"""
Convert 0p3b to 0p3c

Also combines the post_matlab_cleanup script, removing non-science frames

"""


import logging
import h5py
import numpy as np
import os
from scipy.interpolate import interp1d
import shutil

import nomad_ops.core.hdf5.generic_functions as generics

from nomad_ops.config import NOMAD_TMP_DIR, PFM_AUXILIARY_FILES
from nomad_ops.core.hdf5.l0p2b_to_0p3b.l0p2b_to_0p3b import Define_ROI, Define_Top_Bottom_NonIlluminated_Lines, \
    define_mask_values
from nomad_ops.core.tools.progress_bar import progress


__project__ = "NOMAD"
__author__ = "Ian Thomas"
__contact__ = "ian.thomas@aeronomie.be"

logger = logging.getLogger(__name__)

VERSION = 90
OUTPUT_VERSION = "0.3E"


ATTRIBUTES_TO_BE_REMOVED = ["NSpec"]
DATASETS_TO_BE_REMOVED = [
    "Science/Y",
    "Science/YMask",
    "Science/YValidFlag",
    "Science/YError",
    "Science/YErrorRandom",
    "Science/YErrorSystematic",
    "Science/StraylightInflight",
    "Science/X",
    "Science/CircuitNoise",
    "Science/YNb",
    "Science/XNbBin",
    "Science/Xpix_1b",
    "Science/YMaskROI",
    "Channel/VStart_YMaskROI_0b",

    "Science/YErrorSysNL",
]

DATASETS_NOT_TO_BE_RESHAPED = [
    'BadPixelMap',
    'PointXY',
    #        'Housekeeping',
    'Telecommand20/',
    'QualityFlag/',
    'CircuitNoise',  # for UVIS
    'TM11/',
    'TM29/',
    'XNbBin',
    'Xpix_1b',
    'FirstDC_Corr',
    'FirstDC_LevInit',
    'VStart_YMaskROI_0b',
    'Temperature/',
    'InstrF_allROIPixels',
    'nbrBinV_allROIPixels',
    'YMaskROI_allROIPixels',
    'nbrBinV_ExtendROI',
    'Percent_AnomalyDark',
    'Percent_HotPixels'
]

DATASETS_NOT_TO_BE_RESHAPED_COMPRESSED = [
    'Telecommand20',
    'QualityFlag'
]


"""
function [IntegrationInterval,Interval_int] = Prepare_IntegrationIntervalDoor_BPs(Lambda)

	Interval_int.fwhm_550_600 = [549.5 597.0];
	Interval_int.fwhm_550_600_addon = [597.0 602.5];
	Interval_int.fwhm_600_650 = [602.5 647.5];
	Interval_int.fwhm_650_700 = [647.5 701.5];
	Interval_int.fwhm_700_750 = [701.5 748.5];
	Interval_int.fwhm_750_800 = [748.5 798.5];
	Interval_int.fwhm_800_850 = [798.5 850.5];
	Interval_int.fwhm_850_900 = [850.5 902.1];
	Interval_int.fwhm_900_950 = [902.0 956.0];
	Interval_int.fwhm_950_1000 = [956.0 1004.0];
	Interval_int.fwhm_1000_1100 = [1004.0 1099.999];
	
	Interval_int.fromBase_550_600 = [547 599];
	Interval_int.fromBase_600_650 = [600 650];
	Interval_int.fromBase_650_700 = [645 708];
	Interval_int.fromBase_700_750 = [698 752];
	Interval_int.fromBase_750_800 = [746 806];
	Interval_int.fromBase_800_850 = [795 862];
	Interval_int.fromBase_850_900 = [845 915];
	Interval_int.fromBase_900_950 = [896 968];
	Interval_int.fromBase_950_1000 = [944 1011];
	Interval_int.fromBase_1000_1100 = [996 1099.999];
    
	
	doTranspose = 0;
	if (size(Lambda,1) == 1)
		doTranspose = 1;
		Lambda = Lambda';
	end
	
	doReverse = 0;
	if Lambda(1) > Lambda(end)
		doReverse = 1;
		Lambda = flipdim(Lambda,1);
	end
	
	
	nbrLambda = size(Lambda,1);
	Delta_Lambda(2:nbrLambda-1,1) = abs(Lambda(3:nbrLambda,1)-Lambda(1:nbrLambda-2,1))/2;
	Delta_Lambda(1,1) = abs(Lambda(2,1)-Lambda(1,1));
	Delta_Lambda(end+1,1) = abs(Lambda(end,1)-Lambda(end-1,1));
		
	
	if size(Lambda,1) > 0.5
	
		MidLambda = (Lambda(2:end)+Lambda(1:end-1))/2;
		DeltaLambda = (Lambda(2:end)-Lambda(1:end-1));
		
		% interval_550_600
		temp_interval = Interval_int.fwhm_550_600;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_550_600 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_550_600 = IntegrationInterval.Door_int_550_600 .* Delta_Lambda;
		
		% interval_550_600 addon (597-602.5)
		temp_interval = Interval_int.fwhm_550_600_addon;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_550_600_addon = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_550_600_addon = IntegrationInterval.Door_int_550_600_addon .* Delta_Lambda;
		
		% interval_600_650
		temp_interval = Interval_int.fwhm_600_650;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_600_650 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_600_650 = IntegrationInterval.Door_int_600_650 .* Delta_Lambda;
		
		% interval_650_700
		temp_interval = Interval_int.fwhm_650_700;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_650_700 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_650_700 = IntegrationInterval.Door_int_650_700 .* Delta_Lambda;
		
		% interval_700_750
		temp_interval = Interval_int.fwhm_700_750;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_700_750 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_700_750 = IntegrationInterval.Door_int_700_750 .* Delta_Lambda;
		
		% interval_750_800
		temp_interval = Interval_int.fwhm_750_800;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_750_800 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_750_800 = IntegrationInterval.Door_int_750_800 .* Delta_Lambda;
		
		% interval_800_850
		temp_interval = Interval_int.fwhm_800_850;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_800_850 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_800_850 = IntegrationInterval.Door_int_800_850 .* Delta_Lambda;
		
		% interval_850_900
		temp_interval = Interval_int.fwhm_850_900;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_850_900 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_850_900 = IntegrationInterval.Door_int_850_900 .* Delta_Lambda;
		
		% interval_900_950
		temp_interval = Interval_int.fwhm_900_950;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_900_950 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_900_950 = IntegrationInterval.Door_int_900_950 .* Delta_Lambda;
		
		% interval_950_1000
		temp_interval = Interval_int.fwhm_950_1000;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_950_1000 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_950_1000 = IntegrationInterval.Door_int_950_1000 .* Delta_Lambda;
		
		% interval_1000_1100
		temp_interval = Interval_int.fwhm_1000_1100;
        temp_Vect_Porte = zeros(size(Lambda));
		temp_Vect_Porte(1:end-1,1) = (MidLambda > temp_interval(1)) .* (MidLambda < temp_interval(2));
		indMin = min(find(temp_Vect_Porte));
		indMax = max(find(temp_Vect_Porte))+1;
		temp_Vect_Porte(indMin,1) = abs(temp_interval(1)-MidLambda(indMin)) / DeltaLambda(indMin);
		temp_Vect_Porte(indMax,1) = abs(temp_interval(2)-MidLambda(indMax-1)) / DeltaLambda(indMax-1);
		IntegrationInterval.Door_int_1000_1100 = temp_Vect_Porte;
		IntegrationInterval.DeltaLambda_int_1000_1100 = IntegrationInterval.Door_int_1000_1100 .* Delta_Lambda;
		
	end
	
	if doReverse
		IntegrationInterval.Door_int_550_600 = flipdim(IntegrationInterval.Door_int_550_600,1);
		IntegrationInterval.Door_int_550_600_addon = flipdim(IntegrationInterval.Door_int_550_600_addon,1);
		IntegrationInterval.Door_int_600_650 = flipdim(IntegrationInterval.Door_int_600_650,1);
		IntegrationInterval.Door_int_650_700 = flipdim(IntegrationInterval.Door_int_650_700,1);
		IntegrationInterval.Door_int_700_750 = flipdim(IntegrationInterval.Door_int_700_750,1);
		IntegrationInterval.Door_int_750_800 = flipdim(IntegrationInterval.Door_int_750_800,1);
		IntegrationInterval.Door_int_800_850 = flipdim(IntegrationInterval.Door_int_800_850,1);
		IntegrationInterval.Door_int_850_900 = flipdim(IntegrationInterval.Door_int_850_900,1);
		IntegrationInterval.Door_int_900_950 = flipdim(IntegrationInterval.Door_int_900_950,1);
		IntegrationInterval.Door_int_950_1000 = flipdim(IntegrationInterval.Door_int_950_1000,1);
		IntegrationInterval.Door_int_1000_1100 = flipdim(IntegrationInterval.Door_int_1000_1100,1);
		
		IntegrationInterval.DeltaLambda_int_550_600 = flipdim(IntegrationInterval.DeltaLambda_int_550_600,1);
		IntegrationInterval.DeltaLambda_int_550_600_addon = flipdim(IntegrationInterval.DeltaLambda_int_550_600_addon,1);
		IntegrationInterval.DeltaLambda_int_600_650 = flipdim(IntegrationInterval.DeltaLambda_int_600_650,1);
		IntegrationInterval.DeltaLambda_int_650_700 = flipdim(IntegrationInterval.DeltaLambda_int_650_700,1);
		IntegrationInterval.DeltaLambda_int_700_750 = flipdim(IntegrationInterval.DeltaLambda_int_700_750,1);
		IntegrationInterval.DeltaLambda_int_750_800 = flipdim(IntegrationInterval.DeltaLambda_int_750_800,1);
		IntegrationInterval.DeltaLambda_int_800_850 = flipdim(IntegrationInterval.DeltaLambda_int_800_850,1);
		IntegrationInterval.DeltaLambda_int_850_900 = flipdim(IntegrationInterval.DeltaLambda_int_850_900,1);
		IntegrationInterval.DeltaLambda_int_900_950 = flipdim(IntegrationInterval.DeltaLambda_int_900_950,1);
		IntegrationInterval.DeltaLambda_int_950_1000 = flipdim(IntegrationInterval.DeltaLambda_int_950_1000,1);
		IntegrationInterval.DeltaLambda_int_1000_1100 = flipdim(IntegrationInterval.DeltaLambda_int_1000_1100,1);
	end
	
	if doTranspose
		IntegrationInterval.Door_int_550_600 = IntegrationInterval.Door_int_550_600';
		IntegrationInterval.Door_int_550_600_addon = IntegrationInterval.Door_int_550_600_addon';
		IntegrationInterval.Door_int_600_650 = IntegrationInterval.Door_int_600_650';
		IntegrationInterval.Door_int_650_700 = IntegrationInterval.Door_int_650_700';
		IntegrationInterval.Door_int_700_750 = IntegrationInterval.Door_int_700_750';
		IntegrationInterval.Door_int_750_800 = IntegrationInterval.Door_int_750_800';
		IntegrationInterval.Door_int_800_850 = IntegrationInterval.Door_int_800_850';
		IntegrationInterval.Door_int_850_900 = IntegrationInterval.Door_int_850_900';
		IntegrationInterval.Door_int_900_950 = IntegrationInterval.Door_int_900_950';
		IntegrationInterval.Door_int_950_1000 = IntegrationInterval.Door_int_950_1000';
		IntegrationInterval.Door_int_1000_1100 = IntegrationInterval.Door_int_1000_1100';
		
		IntegrationInterval.DeltaLambda_int_550_600 = IntegrationInterval.DeltaLambda_int_550_600';
		IntegrationInterval.DeltaLambda_int_550_600_addon = IntegrationInterval.DeltaLambda_int_550_600_addon';
		IntegrationInterval.DeltaLambda_int_600_650 = IntegrationInterval.DeltaLambda_int_600_650';
		IntegrationInterval.DeltaLambda_int_650_700 = IntegrationInterval.DeltaLambda_int_650_700';
		IntegrationInterval.DeltaLambda_int_700_750 = IntegrationInterval.DeltaLambda_int_700_750';
		IntegrationInterval.DeltaLambda_int_750_800 = IntegrationInterval.DeltaLambda_int_750_800';
		IntegrationInterval.DeltaLambda_int_800_850 = IntegrationInterval.DeltaLambda_int_800_850';
		IntegrationInterval.DeltaLambda_int_850_900 = IntegrationInterval.DeltaLambda_int_850_900';
		IntegrationInterval.DeltaLambda_int_900_950 = IntegrationInterval.DeltaLambda_int_900_950';
		IntegrationInterval.DeltaLambda_int_950_1000 = IntegrationInterval.DeltaLambda_int_950_1000';
		IntegrationInterval.DeltaLambda_int_1000_1100 = IntegrationInterval.DeltaLambda_int_1000_1100';
	end


end
"""


def prepare_integration_interval_door_bps(lambda_values):
    interval_int = {
        'fwhm_550_600': [549.5, 597.0],
        'fwhm_550_600_addon': [597.0, 602.5],
        'fwhm_600_650': [602.5, 647.5],
        'fwhm_650_700': [647.5, 701.5],
        'fwhm_700_750': [701.5, 748.5],
        'fwhm_750_800': [748.5, 798.5],
        'fwhm_800_850': [798.5, 850.5],
        'fwhm_850_900': [850.5, 902.1],
        'fwhm_900_950': [902.0, 956.0],
        'fwhm_950_1000': [956.0, 1004.0],
        'fwhm_1000_1100': [1004.0, 1099.999]
    }

    lambda_values = np.atleast_2d(lambda_values).T
    do_reverse = lambda_values[0] > lambda_values[-1]
    if do_reverse:
        lambda_values = np.flip(lambda_values, axis=0)

    delta_lambda = np.zeros_like(lambda_values)
    delta_lambda[1:-1] = np.abs(lambda_values[2:] - lambda_values[:-2]) / 2
    delta_lambda[0] = np.abs(lambda_values[1] - lambda_values[0])
    delta_lambda[-1] = np.abs(lambda_values[-1] - lambda_values[-2])

    mid_lambda = (lambda_values[1:] + lambda_values[:-1]) / 2
    delta_lambda_mid = np.diff(lambda_values, axis=0)

    integration_interval = {}

    for key, temp_interval in interval_int.items():
        temp_vect_porte = np.zeros_like(lambda_values)
        temp_vect_porte[:-1] = (mid_lambda > temp_interval[0]) & (mid_lambda < temp_interval[1])

        indices = np.where(temp_vect_porte[:-1])[0]
        if len(indices) > 0:
            ind_min, ind_max = indices[0], indices[-1] + 1
            temp_vect_porte[ind_min] = np.abs(temp_interval[0] - mid_lambda[ind_min]) / delta_lambda_mid[ind_min]
            temp_vect_porte[ind_max] = np.abs(temp_interval[1] - mid_lambda[ind_max - 1]) / delta_lambda_mid[ind_max - 1]

        integration_interval[f'Door_int_{key}'] = temp_vect_porte
        integration_interval[f'DeltaLambda_int_{key}'] = temp_vect_porte * delta_lambda

    if do_reverse:
        for key in integration_interval:
            integration_interval[key] = np.flip(integration_interval[key], axis=0)

    return integration_interval, interval_int


"""
function [data,u2_data,ymask,ymaskROI,nbrBinMat,VectSaturated] = DoBinning_V(imode,Y,U2Y,YMask,nbrBinLines,MaskCriterion,iObs)
	% Vertically bin data using YMask
	% "Y" can have 2 or 3 dimensions: nbrLambda  x  nbrLines  (x  nbrObs)
	% then output "data" is 1 or 2 dimensions: nbrLambda  (x  nbrObs)

	Y(isnan(Y)) = -999; % unused because removed by Ymask
	U2Y(isnan(U2Y)) = -999; % unused because removed by Ymask

	Y_bin = sum(Y.*(YMask < MaskCriterion),2); 
	U2Y_bin = sum(U2Y.*(YMask < MaskCriterion),2); 
	nbrBinMat = sum((YMask < MaskCriterion),2); % nbrBinMat = number of lines where YmaskROI is equal to zero * number of lambda binned together
	nbrSaturMat = sum((YMask == 2),2); % nbrBinMat = number of lines where YmaskROI is equal to zero * number of lambda binned together

	if (size(Y_bin,3) > 1.5)
		Y_bin = permute(Y_bin,[1 3 2]); % nbrLambda  x  1 binned line  x  nbrObs -> nbrLambda  x  nbrObs
		U2Y_bin = permute(U2Y_bin,[1 3 2]); % nbrLambda  x  1 binned line  x  nbrObs -> nbrLambda  x  nbrObs
		nbrBinMat = permute(nbrBinMat,[1 3 2]); % nbrLambda  x  1 binned line  x  nbrObs -> nbrLambda  x  nbrObs
		nbrSaturMat = permute(nbrSaturMat,[1 3 2]); % nbrLambda  x  1 binned line  x  nbrObs -> nbrLambda  x  nbrObs
	end

	data = Y_bin;
	u2_data = U2Y_bin;

	ymaskROI = uint8(YMask);
	if (imode == 2) % Nadir
		ymask = uint8(not(nbrBinMat>(nbrBinLines*10/100))); % if at least 10% of the nbrBinMat are good --> YMask is ok = set to 0
		VectSaturated = (nbrSaturMat./(nbrBinLines-nbrBinMat)) >= 0.5;
	elseif (imode == 1) % SO
		ymask = uint8(not(nbrBinMat>(nbrBinLines*50/100))); % if at least 50% of the nbrBinMat are good --> YMask is ok = set to 0
		VectSaturated = (nbrSaturMat./(nbrBinLines-nbrBinMat)) >= 0.5;
	end

	data = data ./ not(ymask); % set NaN if values are not ok
	u2_data = u2_data ./ not(ymask); % set NaN if values are not ok

	data(isinf(data)) = NaN;
	u2_data(isinf(u2_data)) = NaN;
end
"""


def do_binning_v(imode, Y, U2Y, YMask, nbr_bin_lines, mask_criterion, iObs):
    # Replace NaN values
    # Y[np.isnan(Y)] = -999
    # U2Y[np.isnan(U2Y)] = -999

    # Vertical binning using the mask
    Y_bin = np.sum(Y * (YMask < mask_criterion), axis=1)
    U2Y_bin = np.sum(U2Y * (YMask < mask_criterion), axis=1)
    nbr_bin_mat = np.sum(YMask < mask_criterion, axis=1)
    nbr_satur_mat = np.sum(YMask == 2, axis=1)

    # if iObs == 2:
    #     print("Y[0,0]=%f, Y_bin[0]=%f", Y[0,0], Y_bin[0])

    # # Adjust dimensions if Y has more than 2 dimensions
    # if Y_bin.ndim > 2:
    #     Y_bin = np.transpose(Y_bin, (0, 2, 1))
    #     U2Y_bin = np.transpose(U2Y_bin, (0, 2, 1))
    #     nbr_bin_mat = np.transpose(nbr_bin_mat, (0, 2, 1))
    #     nbr_satur_mat = np.transpose(nbr_satur_mat, (0, 2, 1))

    data = Y_bin
    u2_data = U2Y_bin

    # Create the mask
    ymask_roi = YMask.astype(np.uint8)

    # if imode == 2:  # Nadir mode
    #     ymask = (~(nbr_bin_mat > (nbr_bin_lines * 10 / 100))).astype(np.uint8)
    #     vect_saturated = (nbr_satur_mat / (nbr_bin_lines - nbr_bin_mat)) >= 0.5
    # elif imode == 1:  # SO mode
    ymask = (~(nbr_bin_mat > (nbr_bin_lines * 50 / 100))).astype(np.uint8)
    vect_saturated = (nbr_satur_mat / (nbr_bin_lines - nbr_bin_mat)) >= 0.5

    # Mask the data
    data = np.where(~ymask, data, np.nan)
    u2_data = np.where(~ymask, u2_data, np.nan)

    # Replace infinities with NaN
    data[np.isinf(data)] = np.nan
    u2_data[np.isinf(u2_data)] = np.nan

    return data, u2_data, ymask, ymask_roi, nbr_bin_mat, vect_saturated


"""
function [YesEnoughSignal] = Test_Enough_Signal_Present(YFrame,YMaskFrame,iStartLineROI,nbrROILines,Xpix_1b)

    Test_Signal = sum(YFrame(:,iStartLineROI:iStartLineROI+nbrROILines-1) .* not(YMaskFrame(:,iStartLineROI:iStartLineROI+nbrROILines-1)),2) ./ sum(not(YMaskFrame(:,iStartLineROI:iStartLineROI+nbrROILines-1)),2);
        
    nbrPixBinned = (Xpix_1b(2)-Xpix_1b(1));
    Test_Signal = Test_Signal / nbrPixBinned;
    
    vectNoNaN = not(isnan(Test_Signal));
    
    nbrSaturation = sum(sum(YMaskFrame(:,iStartLineROI:iStartLineROI+nbrROILines-1)==2));
        
    if ( (mean(Test_Signal(vectNoNaN,1)) > 1000) || nbrSaturation >= 10 )
        YesEnoughSignal = 1;
    else
        YesEnoughSignal = 0;
    end

end % function
"""


def test_enough_signal_present(YFrame, YMaskFrame, iStartLineROI, nbrROILines, Xpix_1b):
    # Calculate the signal, masking out unwanted areas
    signal_sum = np.sum(YFrame[:, iStartLineROI:iStartLineROI + nbrROILines] *
                        ~YMaskFrame[:, iStartLineROI:iStartLineROI + nbrROILines], axis=1)
    mask_sum = np.sum(~YMaskFrame[:, iStartLineROI:iStartLineROI + nbrROILines], axis=1)

    test_signal = signal_sum / mask_sum

    # Normalize by number of binned pixels
    nbr_pix_binned = Xpix_1b[1] - Xpix_1b[0]
    test_signal /= nbr_pix_binned

    # Filter out NaN values
    vect_no_nan = ~np.isnan(test_signal)

    # Count number of saturated pixels (where mask equals 2)
    nbr_saturation = np.sum(YMaskFrame[:, iStartLineROI:iStartLineROI + nbrROILines] == 2)

    # Determine if there's enough signal
    if (np.mean(test_signal[vect_no_nan]) > 1000) or nbr_saturation >= 10:
        return 1
    else:
        return 0


"""
function [StraylightSpectrum,ObsMethodPerformedOnLambda] = StraylightCalculation_TopAndBottomLines_FitPoly2(src,imode,iObs,Instrument,iStartLineCCD,nbrCCDLines,iStartLineBinROI,nbr_bin_roi_lines,DoUseFrame,frame_ff,ymask_ff,Xpix_1b,XNbBin,MaskCriterion,NaNissueTopCDD_MissingLastPackets)
% Interpolate the straylight from top and bottom lines of the CCD (where only straylight is present).
	
    doPlot = 0;

    if DoUseFrame
        YFrame = frame_ff;
		YMaskFrame = ymask_ff;
    else
        YFrame = h5read(src, '/Science/Y', [1 1 iObs], [Inf Inf 1]);
		YMaskFrame = h5read(src, '/Science/YMask', [1 1 iObs], [Inf Inf 1]);
    end	
	nbrLambdaPixels = size(YFrame,1);
    
    if (NaNissueTopCDD_MissingLastPackets == -1) % for calib and occultations: all pixels
        FirstPixToTreat = 1;
        LastPixToTreat = nbrLambdaPixels;
    else                                         % not calib: only needed pixels
        FirstPixToTreat = min(find(Xpix_1b>=720+8));
        LastPixToTreat = nbrLambdaPixels;
    end
	
    
    if size(FirstPixToTreat,1) > 0 % check that there is something to calculate
        
        Xpix_1b = Xpix_1b';
        XNbBin = (single(XNbBin))';
        x = floor( Xpix_1b + (XNbBin/2));

        [iLineTopStart,iLineBottomStart] = Define_Top_Bottom_NonIlluminated_Lines(imode,x,Instrument);

        iLineBottomStart = (iLineBottomStart + 1 - iStartLineCCD)';
        iLineTopStart = (iLineTopStart + 1 - iStartLineCCD)';

        numLine = 1:size(YFrame,2);
        countFig = 0;

        [YesEnoughSignal] = Test_Enough_Signal_Present(YFrame,YMaskFrame,max(iStartLineBinROI),min(nbr_bin_roi_lines),Xpix_1b);

        for iLambda=FirstPixToTreat:LastPixToTreat
            YLine = YFrame(iLambda,:);

            if not(YesEnoughSignal)

                temp_X = [1:iLineBottomStart(iLambda)];
                temp_X_Ok = temp_X(1,not(YMaskFrame(iLambda,temp_X)));
                if size(temp_X_Ok,2) > 1.5
                    pBottom = polyfit(temp_X_Ok,YLine(1,temp_X_Ok),1); % 1st order when not enough signal
                    YLine(1,temp_X) = pBottom(1)*(temp_X) + pBottom(2);
                end

                temp_X = [iLineTopStart(iLambda):size(YLine,2)];
                temp_X_Ok = temp_X(1,not(YMaskFrame(iLambda,temp_X)));
                if size(temp_X_Ok,2) > 1.5
                    pTop = polyfit(temp_X_Ok,YLine(1,temp_X_Ok),1); % 1st order when not enough signal
                    YLine(1,temp_X) = pTop(1)*(temp_X) + pTop(2);
                end

            end


            ObsMethodPerformedOnLambda(iLambda,1) = 1;

			% weight: only a certains number of points next to the illuminated area
            if imode == 2
					nbrPtsAdj = 15; % 15 CCD lines for nadir (more noisy)
            elseif imode == 1
					nbrPtsAdj = 8; % 8 CCD lines for occult  (less noisy)
			end
				
			weight = zeros(1,size(YLine,2));
			if ( (iLineBottomStart(iLambda,1) >= 3) && ( iLineTopStart(iLambda,1) <= (nbrCCDLines-2)) ) % require at least 3 pts (arbitrarily) at top and bottom to perform the fit
				weight(1,max(1,iLineBottomStart(iLambda,1)-(nbrPtsAdj+1)):iLineBottomStart(iLambda,1)-1) = 1;
				weight(1,iLineTopStart(iLambda,1)+1:min(size(YLine,2),iLineTopStart(iLambda,1)+(nbrPtsAdj+1))) = 1;
			else
				ObsMethodPerformedOnLambda(iLambda,1) = 0;
			end
			tempweight= weight;
			weight = logical(weight.*not(YMaskFrame(iLambda,:)).*not(isnan(YLine)));


            if (sum(weight) > 0.5) 
                % order 2 poly fit
                p = polyfit(numLine(weight),YLine(weight),2);
                StraylightFrame(iLambda,:) =  p(1)*(numLine.^2) + p(2)*(numLine) + p(3);
 
            else
                StraylightFrame(iLambda,1:nbrCCDLines) =  NaN;
            end

            % define line number of the illuminated zone where straylight must be calculated
            numLine_Illum = iStartLineBinROI(iLambda):iStartLineBinROI(iLambda)+nbr_bin_roi_lines(iLambda)-1;
            StraylightSpectrum(iLambda,:) = sum(StraylightFrame(iLambda,numLine_Illum).*(YMaskFrame(iLambda,numLine_Illum) < MaskCriterion),2) ./ sum((YMaskFrame(iLambda,numLine_Illum) < MaskCriterion),2);

        end	

        if FirstPixToTreat > 1.5
            StraylightSpectrum(1:FirstPixToTreat-1,1) = 0;
        end

        % interpolate the pixels with NaNs
        vectNoNaN = not(isnan(StraylightSpectrum));
        if (sum(vectNoNaN) > 1.5)

            if (sum(vectNoNaN) < size(vectNoNaN,1))
                StraylightSpectrum = interp1(x(vectNoNaN),StraylightSpectrum(vectNoNaN),x,'linear');
            else
                StraylightSpectrum = StraylightSpectrum';
            end

        else
            clear StraylightSpectrum % clear before 
            StraylightSpectrum(1,1:nbrLambdaPixels) = NaN;
        end	
    
    else
        StraylightSpectrum(1,1:nbrLambdaPixels) = NaN;
        ObsMethodPerformedOnLambda(1:nbrLambdaPixels,1) = 0;
    end

end % function
"""


def straylight_calculation_top_and_bottom_lines_fitpoly2(imode, iObs, instrument, i_start_line_ccd, nbr_ccd_lines, i_start_line_bin_roi, nbr_bin_roi_lines, frame_ff, ymask_ff,
                                                         xpix_1b, xnb_bin, mask_criterion):
    # do_plot = False
    # i_start_line_bin_roi is offset by 1

    y_frame = frame_ff
    ymask_frame = ymask_ff

    nbr_lambda_pixels = y_frame.shape[0]

    # if nan_issue_top_cdd_missing_last_packets == -1: #for calib and occultations: all pixels
    first_pix_to_treat = 0
    last_pix_to_treat = nbr_lambda_pixels
    # else:
    #     first_pix_to_treat = np.min(np.where(xpix_1b >= 728))
    #     last_pix_to_treat = nbr_lambda_pixels

    xpix_1b = xpix_1b.flatten()
    xnb_bin = np.array(xnb_bin, dtype=np.float32).flatten()
    x = np.floor(xpix_1b + (xnb_bin / 2)).astype(int)

    # imode, X, model = imode, x, instrument
    i_line_top_start, i_line_bottom_start = Define_Top_Bottom_NonIlluminated_Lines(imode, x, instrument)
    # if iObs == 2:
    #     print(i_line_top_start)
    #     print(i_line_bottom_start)

    # i_line_bottom_start and top start are the same as matlab except the last value!
    i_line_bottom_start = (i_line_bottom_start + 1 - i_start_line_ccd)
    i_line_top_start = (i_line_top_start + 1 - i_start_line_ccd)
    i_line_bottom_start[-1] -= 1
    i_line_top_start[-1] += 1

    num_line = np.arange(y_frame.shape[1]) + 1  # same as matlab

    yes_enough_signal = test_enough_signal_present(y_frame, ymask_frame, np.max(i_start_line_bin_roi), np.min(nbr_bin_roi_lines), xpix_1b)
    # print(yes_enough_signal)

    straylight_frame = np.full_like(y_frame, np.nan)
    obs_method_performed_on_lambda = np.zeros(nbr_lambda_pixels)

    straylight_spectrum = np.zeros(len(range(first_pix_to_treat, last_pix_to_treat)))

    # i_lambda is one less than matlab
    for i_lambda in range(first_pix_to_treat, last_pix_to_treat):
        y_line = y_frame[i_lambda, :]

        if not yes_enough_signal:
            temp_x = np.arange(1, i_line_bottom_start[i_lambda] + 1).astype(int)
            temp_x_ok = temp_x[~ymask_frame[i_lambda, temp_x - 1].astype(bool)]

            if len(temp_x_ok) > 1:
                p_bottom = np.polyfit(temp_x_ok, y_line[temp_x_ok - 1], 1)
                y_line[temp_x - 1] = np.polyval(p_bottom, temp_x)

            temp_x = np.arange(i_line_top_start[i_lambda], len(y_line) + 1).astype(int)
            temp_x_ok = temp_x[~ymask_frame[i_lambda, temp_x - 1].astype(bool)]

            if len(temp_x_ok) > 1:
                p_top = np.polyfit(temp_x_ok, y_line[temp_x_ok - 1], 1)
                y_line[temp_x - 1] = np.polyval(p_top, temp_x)

        obs_method_performed_on_lambda[i_lambda] = 1

        nbr_pts_adj = 15 if imode == 2 else 8
        weight = np.zeros_like(y_line)

        # weight true indices are the same as matlab
        if (i_line_bottom_start[i_lambda] >= 3) and (i_line_top_start[i_lambda] <= (nbr_ccd_lines - 2)):
            weight[int(max(0, i_line_bottom_start[i_lambda] - (nbr_pts_adj + 2))):int(i_line_bottom_start[i_lambda])-1] = 1
            weight[int(i_line_top_start[i_lambda]):int(min(len(y_line), i_line_top_start[i_lambda] + (nbr_pts_adj + 1)))] = 1
        else:
            obs_method_performed_on_lambda[i_lambda] = 0

        weight2 = weight.astype(bool) & ~ymask_frame[i_lambda, :].astype(bool) & ~np.isnan(y_line)

        if np.sum(weight2) > 0:
            p = np.polyfit(num_line[weight2], y_line[weight2], 2)
            # this gives the same values as matlab
            straylight_frame[i_lambda, :] = np.polyval(p, num_line)

            # if i_lambda == 0:
            # print([(i,v) for i,v in enumerate(weight2)])
            # print(num_line[weight2], y_line[weight2], straylight_frame[i_lambda, :])
            # print(straylight_frame[i_lambda, :])

        else:
            straylight_frame[i_lambda, :] = np.nan

        # index values offset by 1
        num_line_illum = np.arange(i_start_line_bin_roi[i_lambda], i_start_line_bin_roi[i_lambda] + nbr_bin_roi_lines[i_lambda])
        # if i_lambda == 0:
        # print(num_line_illum)

        straylight_spectrum[i_lambda] = np.nansum(straylight_frame[i_lambda, num_line_illum] * (ymask_frame[i_lambda,
                                                  num_line_illum] < mask_criterion)) / np.nansum(ymask_frame[i_lambda, num_line_illum] < mask_criterion)
        # if i_lambda == 0:
        # print(i_lambda, straylight_frame[i_lambda, num_line_illum], ymask_frame[i_lambda, num_line_illum])
        # print(straylight_spectrum[i_lambda])

    straylight_spectrum[:int(first_pix_to_treat)] = 0

    vect_no_nan = ~np.isnan(straylight_spectrum)
    if np.sum(vect_no_nan) > 1:
        if np.sum(vect_no_nan) < len(vect_no_nan):
            interp_func = interp1d(x[vect_no_nan], straylight_spectrum[vect_no_nan], kind='linear', bounds_error=False, fill_value='extrapolate')
            straylight_spectrum = interp_func(x)
        else:
            straylight_spectrum = straylight_spectrum
    else:
        straylight_spectrum = np.full(nbr_lambda_pixels, np.nan)

    # else:
    #     straylight_spectrum = np.full(nbr_lambda_pixels, np.nan)
    #     obs_method_performed_on_lambda = np.zeros(nbr_lambda_pixels)

    return straylight_spectrum, obs_method_performed_on_lambda


"""
function RemoveStraylight(src, dst, aux)
% Remove the Straylight
% pathfile = absolute path to hdf5 file that need to be threated
% version 0.1 (25/05/2018)
	aux_path = aux;
	ChoiceCurveNIR = 1; % 0 for Wolff | 1 for simulated NIR        
        

    Obs_Letter = src(1,end-3);
	imode = h5read(src, '/Channel/Mode', [1], [1]); % 1=SO, 2=Nadir
	ibinning = h5read(src, '/Channel/AcquisitionMode', [1], [1]); % 0=not binned 1=vertical binned
	itypedata = h5read(src,'/Channel/ReverseFlagAndDataTypeFlagRegister'); % 0=dark 1=dark with reverse clock 2=bias 4=science
	timeobs = h5read(src, '/DateTime', [1], [1]); % Time of observations
	iStartLineCCD = double(h5read(src, '/Channel/VStart', [1], [1])) + 1;
	iEndLineCCD = double(h5read(src, '/Channel/VEnd', [1], [1])) + 1;
	nbrLambdaBinned = double(h5read(src, '/Channel/HorizontalAndCombinedBinningSize', [1], [1]) + 1);
	iStartLambda = double(h5read(src, '/Channel/HStart', [1], [1])) + 1;
	iEndLambda = double(h5read(src, '/Channel/HEnd', [1], [1])) + 1;
	nbrCCDLines = iEndLineCCD - iStartLineCCD + 1;
	nbrObs = double(h5readatt(src, '/', 'NSpec'));
	IT = double(h5read(src, '/Channel/IntegrationTime',[1],[1]));
	
	X = h5read(src, '/Science/X', [1 1], [Inf 1]);
	Xpix_1b = double(h5read(src, '/Science/Xpix_1b'));
	XNbBin = double(h5read(src, '/Science/XNbBin'));
	nbrLambdaPixels = size(XNbBin,1);
	XCalibRef = h5readatt(src, '/', 'XCalibRef');
	if strcmp(XCalibRef,'N/A')
		disp('lambda_IASB_v1: manual')
		XCalibRef = ['lambda_IASB_v1'];
	end
	[IntegrationInterval,Interval_int] = Prepare_IntegrationIntervalDoor_BPs(X');
	
	
	[MaskValues,Crit_Mask] = Define_MaskValues;
	MaskCriterion = Crit_Mask.HotPixelsKept; % % include hot pixels in the binning
	SZA(1:size(itypedata,1),1) = 999;
	
	Instrument = h5readatt(src,'/','InstName');
	if strcmp(Instrument,'FlightSpare NOMAD - Nadir and Occultation for MArs Discovery')
		Instrument=[];Instrument='Spare';
	elseif strcmp(Instrument,'NOMAD - Nadir and Occultation for MArs Discovery')
		Instrument=[];Instrument='Flight';
	end

	[ ROI_ok,iStartLineBinROI,iEndLineBinROI,nbr_bin_roi_lines,nbrSmearingLines,MatBinROI ] = Define_ROI(Instrument,imode,ibinning,iStartLineCCD,iEndLineCCD,Xpix_1b);
	
	iStartLineROI = min(iStartLineBinROI);
	iEndLineROI = max(iEndLineBinROI);
	nbrROILines = iEndLineROI - iStartLineROI + 1;
	VStart_YMaskROI_0b = (iStartLineROI+iStartLineCCD-1)-1;
	
	
	%% Setting smearing correction
	aux_Smearing = fullfile(aux_path,'UVIS_Smearing_Matrix.h5');

	iLineRefSmearing = iStartLineCCD; % chosen as the first available line because the other lines are polluted by smearing.
	SmearingFractionNonIllum = zeros(1024,nbrCCDLines);
	StdSmearingFractionNonIllum = zeros(1024,nbrCCDLines);
		
	% resize to the necessary size
	if (size(SmearingFractionNonIllum,2) > iEndLineROI)
		SmearingFractionNonIllum(:,iEndLineROI+1:end) = [];
		StdSmearingFractionNonIllum(:,iEndLineROI+1:end) = [];
	end
    
        
	% checking if a CCD line is saturating in SO, then this line will not be binned (for all observations)
	YMask_check = h5read(src, '/Science/YMask', [1 iStartLineROI 1], [nbrLambdaPixels nbrROILines Inf]);
	ymaskframe_NotBinForOccult = logical(sum(YMask_check > MaskCriterion,3));
		
	%% Binning of the the data
	for iObs=1:nbrObs
		frame = h5read(src, '/Science/Y', [1 iStartLineROI iObs], [nbrLambdaPixels nbrROILines 1]);
		u2_frame = h5read(src, '/Science/YError', [1 iStartLineROI iObs], [nbrLambdaPixels nbrROILines 1]);
		ymaskframe = h5read(src, '/Science/YMask', [1 iStartLineROI iObs], [nbrLambdaPixels nbrROILines 1]);
		u2_frameSysNL = h5read(src, '/Science/YErrorSysNL', [1 iStartLineROI iObs], [nbrLambdaPixels nbrROILines 1]);
		ymaskframe((ymaskframe < MaskCriterion) & ymaskframe_NotBinForOccult) = MaskValues.NotUsedPixSO;
		ymask_ff(:,:,iObs) = h5read(src, '/Science/YMask', [1 1 iObs], [nbrLambdaPixels Inf 1]);
					
		frame_SmeaRem = frame;
		u2_frame_SmeaRem = u2_frame;
			
		ymaskframeBinROI = ymaskframe;
		ymaskframeBinROI(not(MatBinROI)) = ymaskframeBinROI(not(MatBinROI)) + MaskValues.NotInBinROI;
		
		[data(:,iObs),u2_data(:,iObs),ymask(:,iObs),ymaskROI(:,:,iObs),nbrBinMat1(:,iObs),VectSaturated(:,iObs)] = DoBinning_V(imode,frame,u2_frame,ymaskframeBinROI,nbr_bin_roi_lines,MaskCriterion,iObs);
		
		[data_SmeaRem(:,iObs),u2_data_SmeaRem(:,iObs),ymask(:,iObs),ymaskROI(:,:,iObs),nbrBinMat2(:,iObs),VectSaturated(:,iObs)] = DoBinning_V(imode,frame_SmeaRem,u2_frame_SmeaRem,ymaskframeBinROI,nbr_bin_roi_lines,MaskCriterion,iObs);
		
		
		[unused,u2_data_SysNL(:,iObs),unused,unused,unused,unused] = DoBinning_V(imode,u2_frameSysNL,u2_frameSysNL,ymaskframe,nbr_bin_roi_lines,MaskCriterion,iObs);
		
		ymask_ff(:,iStartLineROI:iStartLineROI+nbrROILines-1,iObs) = ymaskframe;
		ymask_ff_BinROI(:,:,iObs) = ymask_ff(:,:,iObs);
		ymask_ff_BinROI(:,:,iObs) = ymask_ff(:,:,iObs);
		
		nbrBinMat(:,iObs) = nbrBinMat2(:,iObs);
					
	end
	
	
	% test if last packets are missing (if so, adapt inflight method without top CCD lines)
	NaNissueTopCDD_MissingLastPackets = -1; % to calculate all pixels
	
	% averaging data per CCD lines
	u2_data_Rdm = (u2_data_SmeaRem ./ (nbrBinMat.^2))';   % error obtained up to here were random error -> to be saved in h5 file
	u2_data_SysNL = (u2_data_SysNL ./ (nbrBinMat.^2))';
	
	data_EM = (data_SmeaRem ./ nbrBinMat)';
	u2_data_EM = u2_data_Rdm + u2_data_SysNL; % add the systematic error due to Non-Linearity correction (if applied)
	Straylight_EM = ones(size(data_EM))* NaN;
	
	data_OM = data_EM;
	u2_data_OM = u2_data_EM;
	Straylight_OM = Straylight_EM;
	
	data_NoCorr = data_EM;
	u2_data_NoCorr = u2_data_EM;

    iObsSci = 0;
    
	for iObs = 1:nbrObs
        
        if (itypedata(iObs)==4) % if science measurement
            iObsSci = iObsSci + 1;
		
			% SO - - - - - - - - - - - - - - - - - - - - -
                
			if (ibinning == 0 || ibinning == 2)
				
				% Straylight calculation                            
				[StraylightSpectrum,ObsMethodPerformedOnLambda] = StraylightCalculation_TopAndBottomLines_FitPoly2(src,imode,iObs,Instrument,iStartLineCCD,nbrCCDLines,iStartLineBinROI,nbr_bin_roi_lines,0,0,ymask_ff(:,:,iObs),Xpix_1b,XNbBin,MaskCriterion,NaNissueTopCDD_MissingLastPackets);

				% Remove the straylight
				data_OM(iObs,:) = data_OM(iObs,:) - StraylightSpectrum; % remove IR straylight
				u2_data_OM(iObs,:) = u2_data_OM(iObs,:) + (StraylightSpectrum*0.05).^2; % then error OM = 5% of the SL_OM
				Straylight_OM(iObs,:) = StraylightSpectrum;
				
				if (sum(not(isnan(data_OM(iObs,:)))) > 0.5)
					yvalidflag(iObs,1) = uint8(1);
				else
					yvalidflag(iObs,1) = uint8(0);
				end
				
			else
				data_OM(iObs,:) = NaN;
				yvalidflag(iObs,1) = uint8(0); % set to non valid measurement
			end
			
			
        else
            
			data_EM(iObs,:) = NaN;
			data_OM(iObs,:) = NaN;
			yvalidflag(iObs,1) = uint8(0); % set to non valid measurement
			
		end
		
    end

	u2_data_Rdm(isinf(u2_data_Rdm)) = NaN; % set a NaN if it is an Inf
	u2_data_Rdm = u2_data_Rdm';
	
	u2_data_SysNL(isinf(u2_data_SysNL)) = NaN;
	u2_data_SysNL = u2_data_SysNL';
	
	data_EM(isinf(data_EM)) = NaN; % set a NaN if it is an Inf
	data_EM = data_EM';
	u2_data_EM(isinf(u2_data_EM)) = NaN; % set a NaN if it is an Inf
	u2_data_EM = u2_data_EM';
	Straylight_EM(isinf(Straylight_EM)) = NaN; % set a NaN if it is an Inf
	Straylight_EM = Straylight_EM';
	u2_data_Sys_EM = u2_data_EM - u2_data_Rdm;
	
        
	data_OM(isinf(data_OM)) = NaN; % set a NaN if it is an Inf
	data_OM = data_OM';
	u2_data_OM(isinf(u2_data_OM)) = NaN; % set a NaN if it is an Inf
	u2_data_OM = u2_data_OM';
	Straylight_OM(isinf(Straylight_OM)) = NaN; % set a NaN if it is an Inf
	Straylight_OM = Straylight_OM';
	u2_data_Sys_OM = u2_data_OM - u2_data_Rdm;
        
	data_NoCorr(isinf(data_NoCorr)) = NaN; % set a NaN if it is an Inf
	data_NoCorr = data_NoCorr';
	u2_data_NoCorr(isinf(u2_data_NoCorr)) = NaN; % set a NaN if it is an Inf
	u2_data_NoCorr = u2_data_NoCorr';
        
	X = h5read(src, '/Science/X');
	CircuitNoise = h5read(src, '/Science/CircuitNoise');
	YNb = h5read(src, '/Science/YNb');
	XNbBin = uint8(XNbBin);
		
	% if resizing needed due to NaNs in the SO measurements
	if (nbrLambdaPixels > sum(ObsMethodPerformedOnLambda))
		
		nbrLambdaPixels = sum(ObsMethodPerformedOnLambda);
		data_OM = data_OM(logical(ObsMethodPerformedOnLambda),:);
		u2_data_OM = u2_data_OM(logical(ObsMethodPerformedOnLambda),:);
		u2_data_Sys_OM = u2_data_Sys_OM(logical(ObsMethodPerformedOnLambda),:);
		u2_data_Rdm = u2_data_Rdm(logical(ObsMethodPerformedOnLambda),:);
		Straylight_OM = Straylight_OM(logical(ObsMethodPerformedOnLambda),:);
		ymask = ymask(logical(ObsMethodPerformedOnLambda),:);
		ymaskROI = ymaskROI(logical(ObsMethodPerformedOnLambda),:,:);
		CircuitNoise = CircuitNoise(logical(ObsMethodPerformedOnLambda),:);
		X = X(logical(ObsMethodPerformedOnLambda),:);
		YNb(:,1) = uint16(nbrLambdaPixels);
		XNbBin = uint8(XNbBin(logical(ObsMethodPerformedOnLambda),1));
		Xpix_1b = Xpix_1b(logical(ObsMethodPerformedOnLambda),1);
		
	end
		
	h5create(dst, '/Science/Y', [nbrLambdaPixels nbrObs], 'DataType', 'single', 'ChunkSize', [nbrLambdaPixels nbrObs], 'Deflate', 4);
	h5create(dst, '/Science/YError', [nbrLambdaPixels nbrObs], 'DataType', 'single', 'ChunkSize', [nbrLambdaPixels nbrObs], 'Deflate', 4);
	h5create(dst, '/Science/YErrorRandom', [nbrLambdaPixels nbrObs], 'DataType', 'single', 'ChunkSize', [nbrLambdaPixels nbrObs], 'Deflate', 4);
	h5create(dst, '/Science/YErrorSystematic', [nbrLambdaPixels nbrObs], 'DataType', 'single', 'ChunkSize', [nbrLambdaPixels nbrObs], 'Deflate', 4);
	h5create(dst, '/Science/YMask', [nbrLambdaPixels nbrObs], 'DataType', 'uint8', 'ChunkSize', [nbrLambdaPixels nbrObs], 'Deflate', 4);
	h5create(dst, '/Science/YValidFlag', [nbrObs], 'DataType', 'uint8', 'ChunkSize', [nbrObs], 'Deflate', 4);
	h5create(dst, '/Science/X', [nbrLambdaPixels nbrObs], 'DataType', 'single', 'ChunkSize', [nbrLambdaPixels nbrObs], 'Deflate', 4);
	h5create(dst, '/Science/CircuitNoise', [nbrLambdaPixels 2]);
	h5create(dst, '/Science/YNb', [nbrObs], 'DataType', 'uint16');
	h5create(dst, '/Science/XNbBin', [nbrLambdaPixels], 'DataType', 'uint8');
	h5create(dst, '/Science/Xpix_1b', [nbrLambdaPixels], 'DataType', 'single');
	
	if (ibinning == 0 || ibinning == 2)
		h5write(dst, '/Science/Y', data_OM);
		h5write(dst, '/Science/YError', u2_data_OM);
		h5write(dst, '/Science/YErrorRandom', u2_data_Rdm);
		h5write(dst, '/Science/YErrorSystematic', u2_data_Sys_OM);
		h5create(dst, '/Science/StraylightInflight', [nbrLambdaPixels nbrObs], 'DataType', 'single', 'ChunkSize', [nbrLambdaPixels nbrObs], 'Deflate', 4);
		h5write(dst, '/Science/StraylightInflight', single(Straylight_OM));
	end
		
	h5write(dst, '/Science/YMask', ymask);
	h5write(dst, '/Science/YValidFlag', yvalidflag);
	h5write(dst, '/Science/X', X);
	h5write(dst, '/Science/CircuitNoise', CircuitNoise);
	h5write(dst, '/Science/YNb', YNb);
	h5write(dst, '/Science/XNbBin', XNbBin);
	h5write(dst, '/Science/Xpix_1b', single(Xpix_1b));
	
	if (ibinning == 0 || ibinning == 2)
		h5create(dst, '/Science/YMaskROI', [nbrLambdaPixels nbrROILines nbrObs], 'DataType', 'uint8', 'ChunkSize', [nbrLambdaPixels 1 min(24,nbrObs)], 'Deflate', 4);
		h5write(dst, '/Science/YMaskROI', ymaskROI);
		h5create(dst, '/Channel/VStart_YMaskROI_0b', [1], 'DataType', 'uint8');
		h5write(dst, '/Channel/VStart_YMaskROI_0b', uint8(VStart_YMaskROI_0b));
	end
		
	
end % function
"""


# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/20240101_090648_0p3b_UVIS_I.h5"
# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3b/2024/01/01/20240101_090648_0p3b_UVIS_I.h5"
# dst= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3e/2024/01/01/20240101_090648_0p3e_UVIS_I.h5"

# if os.path.exists(dst):
#     os.remove(dst)


def remove_straylight(src, dst):
    # if True:
    # aux_path = os.path.join(PFM_AUXILIARY_FILES, "matlab", "v_07")

    hdf5_basename = os.path.basename(src).split(".")[0]

    d_out = {}

    with h5py.File(src, 'r') as f:
        # obs_letter = src[-4]
        imode = f['Channel/Mode'][0]
        ibinning = f['Channel/AcquisitionMode'][0]
        itypedata = f['Channel/ReverseFlagAndDataTypeFlagRegister'][:]
        # timeobs = f['/DateTime'][0]
        iStartLineCCD = int(f['Channel/VStart'][0]) + 1
        iEndLineCCD = int(f['Channel/VEnd'][0]) + 1
        # nbr_lambda_binned = int(f['/Channel/HorizontalAndCombinedBinningSize'][0]) + 1
        # i_start_lambda = int(f['/Channel/HStart'][0]) + 1
        # i_end_lambda = int(f['/Channel/HEnd'][0]) + 1
        nbr_ccd_lines = iEndLineCCD - iStartLineCCD + 1
        nbr_obs = f.attrs['NSpec']
        # integration_time = float(f['/Channel/IntegrationTime'][0])
        Instrument = f.attrs['InstName']
        Y = np.swapaxes(f["Science/Y"][...], 0, 2)
        YError = np.swapaxes(f["Science/YError"][...], 0, 2)
        YMask = np.swapaxes(f["Science/YMask"][...], 0, 2)
        YErrorSysNL = np.swapaxes(f["Science/YErrorSysNL"][...], 0, 2)

        x = np.swapaxes(f['Science/X'][...], 0, 1)
        Xpix_1b = f['Science/Xpix_1b'][...]
        xnb_bin = f['Science/XNbBin'][...]
        nbr_lambda_pixels = xnb_bin.shape[0]
        x_calib_ref = f.attrs.get('XCalibRef', 'N/A')

        CircuitNoise = np.swapaxes(f['Science/CircuitNoise'], 0, 1)
        YNb = f['Science/YNb'][:]

    logger.info("%s: ibinning=%i, nbr_lambda_pixels=%i, iStartLineCCD=%i, iEndLineCCD=%i, nbr_ccd_lines=%i, Yshape=%s",
                hdf5_basename, ibinning, nbr_lambda_pixels, iStartLineCCD, iEndLineCCD, nbr_ccd_lines, Y.shape)

    if x_calib_ref == 'N/A':
        print('lambda_IASB_v1: manual')
        x_calib_ref = 'lambda_IASB_v1'

    IntegrationInterval, Interval_int = prepare_integration_interval_door_bps(x[:, 0])
    MaskValues, Crit_Mask = define_mask_values()
    MaskCriterion = Crit_Mask["HotPixelsKept"]

    if 'FlightSpare' in Instrument:
        Instrument = 'Spare'
    elif 'NOMAD' in Instrument:
        Instrument = 'Flight'

    # mat_bin_roi is the boolean matrix defining the illuminated pixel indices
    roi_ok, start_line_bin_roi, end_line_bin_roi, nbr_bin_roi_lines, smearing_lines, mat_bin_roi = Define_ROI(
        Instrument, imode, float(ibinning), iStartLineCCD, iEndLineCCD, Xpix_1b)
    # stop()
    mat_bin_roi = np.asarray(mat_bin_roi, dtype=bool)

    # python indexing starts at 0 - subtract one
    # start_line_bin_roi -= 1
    # end_line_bin_roi -= 1

    iStartLineROI = min(start_line_bin_roi)
    iEndLineROI = max(end_line_bin_roi)
    nbrROILines = iEndLineROI - iStartLineROI + 1
    VStart_YMaskROI_0b = (iStartLineROI+iStartLineCCD-1)-1

    logger.info("%s: iStartLineROI=%i, iEndLineROI=%i, nbrROILines=%i, VStart_YMaskROI_0b=%i, Yshape=%s",
                hdf5_basename, iStartLineROI, iEndLineROI, nbrROILines, iEndLineCCD, Y.shape)

    # Setting smearing correction
    # aux_Smearing = os.path.join(aux_path,'UVIS_Smearing_Matrix.h5')

    # iLineRefSmearing = iStartLineCCD # chosen as the first available line because the other lines are polluted by smearing.

    # smearing_fraction_non_illum = np.zeros((1024, nbr_ccd_lines))
    # std_smearing_fraction_non_illum = np.zeros((1024, nbr_ccd_lines))

    # # Resizing if necessary
    # if smearing_fraction_non_illum.shape[1] > iEndLineROI:
    #     smearing_fraction_non_illum = smearing_fraction_non_illum[:, :iEndLineCCD]
    #     std_smearing_fraction_non_illum = std_smearing_fraction_non_illum[:, :iEndLineCCD]

    ymask_ff = np.zeros((nbr_lambda_pixels, nbr_ccd_lines, nbr_obs), dtype=np.uint8)

    # checking if a CCD line is saturating in SO, then this line will not be binned (for all observations)
    YMask_check = YMask[0:nbr_lambda_pixels, iStartLineROI:iStartLineROI + nbrROILines, :]
    ymaskframe_NotBinForOccult = np.sum(YMask_check > MaskCriterion, axis=2).astype(bool)

    logger.info("%s: %i bins not consdered" % (hdf5_basename, np.sum(ymaskframe_NotBinForOccult)))

    # Binning of the the data

    data_SmeaRem = np.zeros((nbr_lambda_pixels, nbr_obs), dtype=float)
    u2_data_SmeaRem = np.zeros((nbr_lambda_pixels, nbr_obs), dtype=float)
    ymask = np.zeros((nbr_lambda_pixels, nbr_obs), dtype=float)
    ymaskROI = np.zeros((nbr_lambda_pixels, nbrROILines, nbr_obs), dtype=float)
    nbrBinMat = np.zeros((nbr_lambda_pixels, nbr_obs), dtype=float)
    VectSaturated = np.zeros((nbr_lambda_pixels, nbr_obs), dtype=float)

    u2_data_SysNL = np.zeros((nbr_lambda_pixels, nbr_obs), dtype=float)

    # loop through observations
    for i_obs in range(nbr_obs):

        # no smearing
        frame_SmeaRem = Y[0:nbr_lambda_pixels, iStartLineROI:iStartLineROI + nbrROILines, i_obs]
        u2_frame_SmeaRem = YError[0:nbr_lambda_pixels, iStartLineROI:iStartLineROI + nbrROILines, i_obs]
        ymaskframe = YMask[0:nbr_lambda_pixels, iStartLineROI:iStartLineROI + nbrROILines, i_obs]
        u2_frameSysNL = YErrorSysNL[0:nbr_lambda_pixels, iStartLineROI:iStartLineROI + nbrROILines, i_obs]
        ymaskframe[(ymaskframe < MaskCriterion) & ymaskframe_NotBinForOccult] = MaskValues["NotUsedPixSO"]
        ymask_ff[:, :, i_obs] = YMask[0:nbr_lambda_pixels, :, i_obs]

        ymaskframeBinROI = ymaskframe.copy()  # get bad pixel map 0 = good, 50 = bad
        ymaskframeBinROI[~mat_bin_roi] += MaskValues["NotInBinROI"]  # apply illumination indices to mask
        # 100 = not illuminated
        # 0 = illuminated
        # 150 = bad pixel + not illuminated

        # imode, Y, U2Y, YMask, nbr_bin_lines, mask_criterion, iObs = imode, frame_SmeaRem, u2_frame_SmeaRem, ymaskframeBinROI, nbr_bin_roi_lines, MaskCriterion, i_obs
        # data_SmeaRem and nbrBinMat are the same as matlab
        data_SmeaRem[:, i_obs], u2_data_SmeaRem[:, i_obs], ymask[:, i_obs], ymaskROI[:, :, i_obs], nbrBinMat[:, i_obs], VectSaturated[:,
                                                                                                                                      i_obs] = do_binning_v(imode, frame_SmeaRem, u2_frame_SmeaRem, ymaskframeBinROI, nbr_bin_roi_lines, MaskCriterion, i_obs)
        # if i_obs == 40:
        #     stop()

        _, u2_data_SysNL[:, i_obs], _, _, _, _ = do_binning_v(imode, u2_frameSysNL, u2_frameSysNL, ymaskframe, nbr_bin_roi_lines, MaskCriterion, i_obs)

        ymask_ff[:, iStartLineROI:iStartLineROI + nbrROILines, i_obs] = ymaskframe

    u2_data_Rdm = (u2_data_SmeaRem / (nbrBinMat**2))  # error obtained up to here were random error -> to be saved in h5 file
    u2_data_SysNL = (u2_data_SysNL / (nbrBinMat**2))

    # data_OM is the same as matlab
    data_OM = (data_SmeaRem / nbrBinMat)
    u2_data_OM = u2_data_Rdm + u2_data_SysNL  # add the systematic error due to Non-Linearity correction (if applied)
    Straylight_OM = np.full_like(data_SmeaRem, np.nan)

    # data_NoCorr = (data_SmeaRem / nbrBinMat)
    # u2_data_NoCorr = u2_data_Rdm + u2_data_SysNL # add the systematic error due to Non-Linearity correction (if applied)

    yvalidflag = np.zeros(nbr_obs, dtype=np.uint8)

    for i_obs in range(nbr_obs):
        if itypedata[i_obs] == 4:  # science measurement
            if ibinning == 0 or ibinning == 2:

                # Straylight calculation
                # imode, iObs, instrument, i_start_line_ccd, nbr_ccd_lines, i_start_line_bin_roi, nbr_bin_roi_lines, frame_ff, ymask_ff,     xpix_1b, xnb_bin, mask_criterion = imode, i_obs, Instrument, iStartLineCCD, nbr_ccd_lines, start_line_bin_roi, nbr_bin_roi_lines, Y[:,:,i_obs], YMask[:,:,i_obs], Xpix_1b, xnb_bin, MaskCriterion

                # iStartLineCCD, nbr_ccd_lines, nbr_bin_roi_lines, Xpix_1b, xnb_bin are the same as matlab
                # start_line_bin_roi is offset by 1
                straylight_spectrum, ObsMethodPerformedOnLambda = straylight_calculation_top_and_bottom_lines_fitpoly2(
                    imode, i_obs, Instrument, iStartLineCCD, nbr_ccd_lines, start_line_bin_roi, nbr_bin_roi_lines, Y[:, :, i_obs], YMask[:, :, i_obs], Xpix_1b, xnb_bin, MaskCriterion)
                # if i_obs == 40:
                #     stop()
                data_OM[:, i_obs] -= straylight_spectrum
                Straylight_OM[:, i_obs] = straylight_spectrum
                yvalidflag[i_obs] = 1 if np.sum(~np.isnan(data_OM[:, i_obs])) > 0.5 else 0
        else:
            data_OM[:, i_obs] = np.nan
            yvalidflag[i_obs] = 0  # set to non valid measurement

    # Handling NaNs and transposing
    u2_data_Rdm[np.isinf(u2_data_Rdm)] = np.nan
    # u2_data_SysNL[np.isinf(u2_data_SysNL)] = np.nan
    data_OM[np.isinf(data_OM)] = np.nan
    u2_data_OM[np.isinf(u2_data_OM)] = np.nan
    Straylight_OM[np.isinf(Straylight_OM)] = np.nan
    u2_data_Sys_OM = u2_data_OM - u2_data_Rdm
    # data_NoCorr[np.isinf(data_NoCorr)] = np.nan
    # u2_data_NoCorr[np.isinf(u2_data_NoCorr)] = np.nan

    # Convert XNbBin to uint8
    xnb_bin = xnb_bin.astype(np.uint8)

    # Resizing due to NaNs in SO measurements
    if nbr_lambda_pixels > np.sum(ObsMethodPerformedOnLambda):
        nbr_lambda_pixels = np.sum(ObsMethodPerformedOnLambda)

        mask = ObsMethodPerformedOnLambda.astype(bool)

        data_OM = data_OM[mask, :]
        u2_data_OM = u2_data_OM[mask, :]
        u2_data_Sys_OM = u2_data_Sys_OM[mask, :]
        u2_data_Rdm = u2_data_Rdm[mask, :]
        Straylight_OM = Straylight_OM[mask, :]
        ymask = ymask[mask, :]
        ymaskROI = ymaskROI[mask, :, :]
        CircuitNoise = CircuitNoise[mask, :]
        x = x[mask, :]

        YNb[:] = nbr_lambda_pixels
        xnb_bin = xnb_bin[mask].astype(np.uint8)
        Xpix_1b = Xpix_1b[mask]

    d_out["Science/Y"] = data_OM.astype(np.float32)
    d_out["Science/YMask"] = ymask
    d_out["Science/YValidFlag"] = yvalidflag

    d_out["Science/YError"] = u2_data_OM
    d_out["Science/YErrorRandom"] = u2_data_Rdm
    d_out["Science/YErrorSystematic"] = u2_data_Sys_OM
    d_out["Science/StraylightInflight"] = Straylight_OM

    d_out["Science/X"] = x
    d_out["Science/CircuitNoise"] = CircuitNoise
    d_out["Science/YNb"] = YNb
    d_out["Science/XNbBin"] = xnb_bin
    d_out["Science/Xpix_1b"] = Xpix_1b

    if ibinning == 0 or ibinning == 2:
        d_out["Science/YMaskROI"] = ymaskROI
        d_out["Channel/VStart_YMaskROI_0b"] = np.uint8(VStart_YMaskROI_0b)

    indrec = np.argwhere(itypedata == 4)[:, 0]
    logger.info("NSpec old=%i, NSpec new=%i", len(itypedata), len(indrec))

    with h5py.File(src, 'r') as hdf5FileIn:
        with h5py.File(dst, 'w') as hdf5FileOut:

            generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)

            hdf5FileOut.attrs["NSpec"] = len(indrec)

            # don't copy all datasets to new file
            for dset_path, dset in generics.iter_datasets(hdf5FileIn):
                # print(dset_path)
                if dset_path in DATASETS_TO_BE_REMOVED:  # don't copy
                    # print("Not copying %s" %dset_path)
                    continue

                dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])

                if any([i in dset_path for i in DATASETS_NOT_TO_BE_RESHAPED]):
                    hdf5FileIn.copy(dset_path, dest)
                    # print("Copying but not reshaping %s" %dset_path)
                    continue

                dsetcopy = np.array(dset)
                # print("%s dataset is being reshaped (shape=%s)" %(dset_path, dsetcopy.shape))
                if dsetcopy.ndim == 0:
                    hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy)
                elif dsetcopy.ndim == 1:  # if 1D, use compression
                    # print("Length = %i" %dsetcopy.shape[0])
                    hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy[indrec], compression="gzip", shuffle=True)
                elif len(dsetcopy.shape) == 2:  # if 2D, use compression
                    # print("Length = (%i, %i)" %(dsetcopy.shape[0], dsetcopy.shape[1]))
                    hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy[indrec, :], compression="gzip", shuffle=True)
                elif len(dsetcopy.shape) == 3:  # if 3D, use compression
                    # print("Length = (%i, %i)" %(dsetcopy.shape[0], dsetcopy.shape[1]))
                    hdf5FileOut.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy[indrec, :, :], compression="gzip", shuffle=True)

                # else:
                #     if dset.ndim == 1:
                #         hdf5FileIn.copy(dset_path, dset[indrec])
                #     elif dset.ndim == 2:
                #         hdf5FileIn.copy(dset_path, dset[indrec, :])

            for key, value in d_out.items():
                # print("%s dataset (shape=%i)" %(key, len(value.shape)))
                if key == "Science/YMask":
                    dtype = "uint8"
                else:
                    dtype = "float32"

                if value.ndim == 1:
                    if any([i in key for i in DATASETS_NOT_TO_BE_RESHAPED]):
                        hdf5FileOut.create_dataset(key, data=value, dtype=dtype, compression="gzip", shuffle=True)
                    else:
                        dset = value[indrec]
                        hdf5FileOut.create_dataset(key, data=dset, dtype=dtype, compression="gzip", shuffle=True)
                        # logger.info("Output dataset %s has shape %s", key, dset.shape)
                elif value.ndim == 2:
                    if any([i in key for i in DATASETS_NOT_TO_BE_RESHAPED]):
                        hdf5FileOut.create_dataset(key, data=value.T, dtype=dtype, chunks=(
                            value.shape[1], min(24, value.shape[0])), compression="gzip", compression_opts=4)
                    else:
                        dset = value.T[indrec, :]
                        hdf5FileOut.create_dataset(key, data=dset, dtype=dtype, chunks=(
                            len(indrec), min(24, value.shape[0])), compression="gzip", compression_opts=4)
                        # logger.info("Output dataset %s has shape %s", key, dset.shape)
                elif value.ndim == 3:
                    if any([i in key for i in DATASETS_NOT_TO_BE_RESHAPED]):
                        hdf5FileOut.create_dataset(key, data=np.swapaxes(value, 0, 2), dtype=dtype, chunks=(
                            1, 1, value.shape[0]), compression="gzip", compression_opts=4)
                    else:
                        dset = np.swapaxes(value, 0, 2)[indrec, :, :]
                        hdf5FileOut.create_dataset(key, data=dset, dtype=dtype, chunks=(1, 1, value.shape[0]), compression="gzip", compression_opts=4)
                        # logger.info("Output dataset %s has shape %s", key, dset.shape)
                else:
                    hdf5FileOut.create_dataset(key, data=value, dtype=dtype)


def convert(hdf5file_path):
    """this function should perform the same function as nomad_ops/matlab/src/RemoveStraylight.m"""
    logger.info("Convert: %s", hdf5file_path)
    tmp_file = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5file_path))
    # shutil.copyfile(hdf5file_path, tmp_file)
    remove_straylight(hdf5file_path, tmp_file)
    return [tmp_file]


# def post_matlab_cleanup(hdf5_tmp_file):
#     with h5py.File(hdf5_tmp_file, "r+") as hdf5File:
#         indrec = np.argwhere(np.array(hdf5File['Channel/ReverseFlagAndDataTypeFlagRegister'])==4)
#         indrec = indrec[:,0]
#         hdf5File.attrs["NSpec"] = len(indrec)
#         for dset_path, dset in list(generics.iter_datasets(hdf5File)):
#             if not any([i in dset_path for i in DATASETS_NOT_TO_BE_RESHAPED]):
#                 dset = hdf5File.pop(dset_path)
#                 dsetcopy=dset[()][indrec]
#                 #print(">>> %-64s : %s -> %s" % (dset_path, dset.shape, dsetcopy.shape))
#                 hdf5File.create_dataset(dset_path, dtype=dset.dtype, data=dsetcopy,
#                                         compression="gzip", shuffle=True)


# import matplotlib.pyplot as plt
# with h5py.File(dst, "r") as f:
#     y_new = f["Science/Y"][...]

# dst_old = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3e/2018/05/22/20180522_051504_0p3c_UVIS_I.h5"
# with h5py.File(dst_old, "r") as f:
#     y_old = f["Science/Y"][...]

# plt.figure()
# plt.imshow(y_old)
# plt.title("Y Old")
# plt.show()

# plt.figure()
# plt.imshow(y_new)
# plt.title("Y New")
# plt.show()


# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3b/2024/01/01/20240101_090648_0p3b_UVIS_I.h5"
# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3b/2024/05/02/20240502_193435_0p3b_UVIS_E.h5"
# convert(src)
