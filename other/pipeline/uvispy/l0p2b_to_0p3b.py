# -*- coding: utf-8 -*-
"""
Convert 0p2b to 0p3b

Note that not yet fully working for Spare model

Axes on arrays are switched to work with Yannick's existing code, then reswitched to save to file correctly
Should be nPx x nRows x nFrames in the working code
Should be nFrames x nRows x nPx in the files
2d arrays are also transposed

"""


import logging
import os
# import shutil
import numpy as np
import h5py
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from datetime import datetime
from scipy import interpolate
# from scipy.signal import savgol_filter

import nomad_ops.core.hdf5.generic_functions as generics
from statsmodels.nonparametric.smoothers_lowess import lowess

# from nomad_ops.core.tools.progress_bar import progress
from nomad_ops.config import NOMAD_TMP_DIR, PFM_AUXILIARY_FILES

__project__ = "NOMAD"
__author__ = "Ian Thomas"
__contact__ = "ian.thomas@aeronomie.be"

logger = logging.getLogger(__name__)

VERSION = 90
OUTPUT_VERSION = "0.3B"


ATTRIBUTES_TO_BE_REMOVED = []
DATASETS_TO_BE_REMOVED = [
    'Science/Y',
    'Science/YError',
    'Science/YErrorSysNL',
    'Science/YMask',
    'Science/X',
    'Science/Xpix_1b',
    'Science/CircuitNoise',
    'Science/YNb',
    'Science/XNbBin',
    'Channel/Percent_AnomalyDark',
    'Channel/Percent_HotPixels'
]


"""
function [params,DC1_offset,DC1_corr_RowsUsed] = Load_Params_DC_correction(aux,imode,Obs_Letter,IT,Year)

    path_Table = [aux,'/UVIS_DarkCurrent_vs_Temperature_Table.h5'];

    if imode == 1

        nameChannel = ['/Occultation/'];
        temp_path = [nameChannel,'Years'];
        Years_avail = h5read(path_Table,temp_path, [1], [Inf]);
        MaxYear= max(Years_avail);
        MinYear= min(Years_avail);

        % select IT
        IT_cases = [45 75]; % cases always present
        if sum(IT_cases==IT) == 1
            IT_Selected_DC = IT;
        else
            iClosestCase = find(min(abs(IT_cases-IT))==abs(IT_cases-IT));
            IT_Selected_DC = IT_cases(iClosestCase);
        end

        % select Year
        if (Year <= MaxYear) & (Year >= MinYear)
            Year_Selected_DC = Year;
        elseif (Year >= MaxYear)
            Year_Selected_DC = MaxYear;
        elseif (Year <= MinYear)
            Year_Selected_DC = MinYear;
        end


    elseif imode == 2

        if strcmp(Obs_Letter,'D') | strcmp(Obs_Letter,'C')
            nameChannel = ['/Nadir_D/'];
            IT_cases = [5000 7000 10000 20000]; % cases always present
        elseif strcmp(Obs_Letter,'N') | strcmp(Obs_Letter,'L') | strcmp(Obs_Letter,'O') | strcmp(Obs_Letter,'P') | strcmp(Obs_Letter,'Q')
            nameChannel = ['/Nadir_NLO/'];
            IT_cases = [20000]; % cases always present
        end
        temp_path = [nameChannel,'Years'];
        Years_avail = h5read(path_Table,temp_path, [1], [Inf]);
        MaxYear= max(Years_avail);
        MinYear= min(Years_avail);

        % select IT
        if sum(IT_cases==IT) == 1
            IT_Selected_DC = IT;
        elseif (IT == 13000) & (Year <= 2019)
            IT_Selected_DC = IT;
        else
            iClosestCase = find(min(abs(IT_cases-IT))==abs(IT_cases-IT));
            IT_Selected_DC = IT_cases(iClosestCase(1));
        end

        % select Year
        if (Year <= MaxYear) & (Year >= MinYear)
            Year_Selected_DC = Year;
        elseif (Year >= MaxYear)
            Year_Selected_DC = MaxYear;
        elseif (Year <= MinYear)
            Year_Selected_DC = MinYear;
        end

    end

    CaseFound = 0;
    while not(CaseFound) & (Year_Selected_DC >= MinYear)
        try
            temp_path_DCvsT = ['IT_',num2str(IT_Selected_DC,'%05d'),'ms/',num2str(Year_Selected_DC,'%4d')];
            path_Var_DCvsT = [nameChannel,temp_path_DCvsT];

            params = h5read(path_Table,path_Var_DCvsT, [1], [Inf]);
            CaseFound = 1;
        end
        if not(CaseFound)
            Year_Selected_DC = Year_Selected_DC - 1;
        end
    end

    % load DC1 vs DC2 offset
    try
        temp_path_DC1offset = [nameChannel,'DC1vsDC2_increase/Mean'];
        DC1_offset.mean = h5read(path_Table,temp_path_DC1offset);
        temp_path_DC1offset = [nameChannel,'DC1vsDC2_increase/Median'];
        DC1_offset.median = h5read(path_Table,temp_path_DC1offset);

        temp_path_DC1offset = [nameChannel,'DC1vsDC2_increase/Std'];
        DC1_offset.std = h5read(path_Table,temp_path_DC1offset);
    catch
        DC1_offset.mean = [];
        DC1_offset.std = [];
    end

    temp_path_DC1offset = [nameChannel,'RowsUsed_1b'];
    DC1_corr_RowsUsed = h5read(path_Table,temp_path_DC1offset);

end
"""


def load_params_dc_correction(aux, imode, obs_letter, IT, year):

    path_table = f"{aux}/UVIS_DarkCurrent_vs_Temperature_Table.h5"
    if imode == 1:
        name_channel = "/Occultation/"
        temp_path = f"{name_channel}Years"
        with h5py.File(path_table, 'r') as f:
            years_avail = f[temp_path][:]
        max_year, min_year = np.max(years_avail), np.min(years_avail)

        IT_cases = [45, 75]
        IT_selected_DC = IT if IT in IT_cases else IT_cases[np.argmin(
            np.abs(np.array(IT_cases) - IT))]
        year_selected_DC = min(max(year, min_year), max_year)

    elif imode == 2:
        if obs_letter in ['D', 'C']:
            name_channel = "/Nadir_D/"
            IT_cases = [5000, 7000, 10000, 20000]
        elif obs_letter in ['N', 'L', 'O', 'P', 'Q']:
            name_channel = "/Nadir_NLO/"
            IT_cases = [20000]

        temp_path = f"{name_channel}Years"
        with h5py.File(path_table, 'r') as f:
            years_avail = f[temp_path][:]
        max_year, min_year = np.max(years_avail), np.min(years_avail)

        if IT in IT_cases or (IT == 13000 and year <= 2019):
            IT_selected_DC = IT
        else:
            IT_selected_DC = IT_cases[np.argmin(
                np.abs(np.array(IT_cases) - IT))]

        year_selected_DC = min(max(year, min_year), max_year)

    case_found = False
    while not case_found and year_selected_DC >= min_year:
        try:
            temp_path_DCvsT = f"IT_{IT_selected_DC:05d}ms/{year_selected_DC}"
            path_var_DCvsT = f"{name_channel}{temp_path_DCvsT}"
            with h5py.File(path_table, 'r') as f:
                params = f[path_var_DCvsT][:]
            case_found = True
        except KeyError:
            year_selected_DC -= 1

    try:
        with h5py.File(path_table, 'r') as f:
            DC1_offset = {
                'mean': f[f"{name_channel}DC1vsDC2_increase/Mean"][:],
                'median': f[f"{name_channel}DC1vsDC2_increase/Median"][:],
                'std': f[f"{name_channel}DC1vsDC2_increase/Std"][:]
            }
    except KeyError:
        DC1_offset = {'mean': [], 'std': []}

    with h5py.File(path_table, 'r') as f:
        DC1_corr_RowsUsed = f[f"{name_channel}RowsUsed_1b"][:]

    return params, DC1_offset, DC1_corr_RowsUsed


"""

function [YMask] = DetectSaturation(data,SaturationValue,XNbBin,MaskValueUsed)

    dataSaturated = (data > (SaturationValue*XNbBin));
    YMask=dataSaturated*MaskValueUsed;

end

function [YMask] = DetectLevel02x_NaN(data,YMask,MaskValueUsed)

    YMask = uint8(YMask) + ( uint8(not(YMask)) .* uint8(isnan(data)) * uint8(MaskValueUsed) );

end
"""


def detect_saturation(data, saturation_value, x_nb_bin, mask_value_used):
    data_saturated = data > np.zeros_like(data) + saturation_value * x_nb_bin[0]
    y_mask = data_saturated.astype(np.uint8) * mask_value_used
    return y_mask


def detect_level_02x_nan(data, y_mask, mask_value_used):
    y_mask = np.uint8(y_mask) + (np.uint8(~y_mask) * np.uint8(np.isnan(data)) * np.uint8(mask_value_used))
    return y_mask


"""

function [data_CorrNL,error_CorrNL,ymask] = Non_Linearity_CCD_Correction(data,Corr_NL,LinearityCorrValue,SaturationValue,XNbBin,ymask,MaskValueUsed)
% Correction for Non Linearity of UVIS CCD value. Correction is made for
% CCD values between LinearityCorrValue and SaturationValue.

    % Initialise -------------------------------------------------------------------------------
    Treshold_Val = Corr_NL.count(1);
    step = Corr_NL.count(2) - Corr_NL.count(1);
    data_CorrNL = data;
    error_CorrNL = single(zeros(size(data_CorrNL)));


    % Pixel identification on the binScaled data -----------------------------------------------
    data_BinScaled = data ./ XNbBin;

    % find the pixels concerned by the correction
    iPixNL = find((data_BinScaled > LinearityCorrValue) & (data_BinScaled <= SaturationValue));

    % find the index in the correction curve for each pixel to correct
    iCorrCurve = round((data_BinScaled(iPixNL)-Treshold_Val)/step) + 1;


    % Correction of the NL pixels --------------------------------------------------------------
    data_CorrNL(iPixNL) = data(iPixNL) .* (1+Corr_NL.corr(iCorrCurve));
    error_CorrNL(iPixNL) = data(iPixNL) .* Corr_NL.error(iCorrCurve);

    % assign the mask value
    ymask(iPixNL) = MaskValueUsed; % up to here ymask is: or saturated or NaN in lvl 02a -> both cases imply no non lin correction

end

"""


def non_linearity_ccd_correction(data, corr_nl, linearity_corr_value, saturation_value, x_nb_bin, ymask, mask_value_used):
    """
    Correction for Non-Linearity of UVIS CCD value. Correction is made for
    CCD values between LinearityCorrValue and SaturationValue.
    """
    # Initialize
    threshold_val = corr_nl['count'][0]
    step = corr_nl['count'][1] - corr_nl['count'][0]
    data_corr_nl = np.copy(data)
    error_corr_nl = np.zeros_like(data_corr_nl, dtype=np.float32)

    # Pixel identification on the binScaled data
    data_bin_scaled = data / x_nb_bin[0]

    # Find the pixels concerned by the correction
    i_pix_nl = np.where((data_bin_scaled > linearity_corr_value)
                        & (data_bin_scaled <= saturation_value))

    # Find the index in the correction curve for each pixel to correct
    i_corr_curve = np.round(
        (data_bin_scaled[i_pix_nl] - threshold_val) / step).astype(int) + 1

    # Correction of the NL pixels
    data_corr_nl[i_pix_nl] = data[i_pix_nl] * \
        (1 + corr_nl['corr'][i_corr_curve])
    error_corr_nl[i_pix_nl] = data[i_pix_nl] * corr_nl['error'][i_corr_curve]

    # Assign the mask value
    ymask[i_pix_nl] = mask_value_used

    return data_corr_nl, error_corr_nl, ymask


"""

function [MatrixCosmicRayDark,MatrixHotPixels,dataDark_CosmicRayRemoved] = AnomalyDetectionDark(data,iStartUsefulPix,iEndUsefulPix,ibinning,StrengthCriteria)

    doPlot = 0;

    dataDark = data(iStartUsefulPix:iEndUsefulPix,:,:);
    dataDarkNoExtreme = dataDark;
    nbrDark = size(dataDark,3);

    MatrixCosmicRayDark = zeros(size(data,1), size(data,2), nbrDark);
    MatrixHotPixels = zeros(size(data,1), size(data,2));

    if strcmp(StrengthCriteria,'Strong') % for nightsides at least
        MultFact_Std = [2 2 3];
        MultFact_Std_Fin = 3;
        MultFact_Std_CosmicRay = 3;
    elseif strcmp(StrengthCriteria,'Medium')
        MultFact_Std = [2 3];
        MultFact_Std_Fin = 4;
        MultFact_Std_CosmicRay = 3;
    elseif strcmp(StrengthCriteria,'Occult')
        MultFact_Std = [4];
        MultFact_Std_Fin = 8;
        MultFact_Std_CosmicRay = 16;
    elseif strcmp(StrengthCriteria,'Occult_Bin')
        MultFact_Std = [3];
        MultFact_Std_Fin = 4;
        MultFact_Std_CosmicRay = 6;
    elseif strcmp(StrengthCriteria,'LabMeas') % dark not constant (due to high temperature)
        MultFact_Std = [3];
        MultFact_Std_Fin = 4;
        MultFact_Std_CosmicRay = 4;
    end

    nbrLoop = size(MultFact_Std,2);

    for iDark = 1:nbrDark

        if (ibinning == 0 || ibinning == 2) % useful for unbinned data

            if doPlot
                close all
            end

            for iLoop = 1:nbrLoop

                Ymean(:,iDark) = median(dataDarkNoExtreme(:,:,iDark))';
                Ystd(:,iDark) = std(dataDarkNoExtreme(:,:,iDark))';
                if iLoop == 1
                    Ystd_Init(:,iDark) = Ystd(:,iDark); % used for Cosmic Ray detection in Dark meas.
                end

                for iLine = 1:size(dataDark,2)
                    vectExtreme = logical((abs(dataDark(:,iLine,iDark) - Ymean(iLine,iDark))) > MultFact_Std(iLoop)*Ystd(iLine,iDark));
                    dataDarkNoExtreme(vectExtreme,iLine,iDark) = Ymean(iLine,iDark);


                    % to show results
                    if doPlot
                        ResModuloTest = 6; % to test a specific line of the ccd
                        if mod(iLine,50) == ResModuloTest
                            figure(iLine+iLoop); hold on
                            x = [1:size(dataDark,1)]';
                            meantoplot(x,1) = Ymean(iLine,iDark);
                            stdtoplot(x,1) = MultFact_Std(iLoop)*Ystd(iLine,iDark);

                            errorbar(x,meantoplot,stdtoplot,'g');
                            plot(dataDark(:,iLine,iDark),'k','LineWidth',1.5);
                            plot(dataDarkNoExtreme(:,iLine,iDark),'b','LineWidth',1.5);
                            titre = ['Hot Pix detection loop ',num2str(iLoop,'%d'),': fraction identified ',num2str(sum(vectExtreme)/size(vectExtreme,1),'%5.3f')];
                            title(titre)
                            grid on
                            xlim([1 size(dataDark,1)])
                        end
                    end

                end % for
            end % Loop

            % Last Loop
            Ymean(:,iDark) = median(dataDarkNoExtreme(:,:,iDark))';
            Ystd(:,iDark) = std(dataDarkNoExtreme(:,:,iDark))';
            for iLine = 1:size(dataDark,2)

                if strcmp(StrengthCriteria,'SpareAdapt')
                    RefMean = smooth(dataDarkNoExtreme(:,iLine,iDark),50);
                else
                    RefMean = Ymean(iLine,iDark);
                end
%                 vectAnomalyDark(:,iLine,iDark) = (abs(dataDark(:,iLine,iDark) - RefMean) > MultFact_Std_Fin*Ystd(iLine,iDark)); % stronger criteria for cosmic ray detection (uses the initial std)
                vectAnomalyDark(:,iLine,iDark) = (abs(dataDark(:,iLine,iDark) - RefMean) > MultFact_Std_CosmicRay*Ystd_Init(iLine,iDark)); % stronger criteria for cosmic ray detection (uses the initial std)
                vectAnomalyDark2(:,iLine,iDark) = (abs(dataDark(:,iLine,iDark) - RefMean) > MultFact_Std_Fin*Ystd(iLine,iDark)); % criteria for verifying not a hot pixel

                % to show results
                if doPlot
                    if mod(iLine,50) == ResModuloTest
                        figure(iLine+iLoop+1); hold on
                        x = [1:size(dataDark,1)]';
                        meantoplot(x,1) = RefMean;
                        stdtoplot(x,1) = MultFact_Std_Fin*Ystd(iLine,iDark);

                        errorbar(x,meantoplot,stdtoplot,'y');
                        plot(dataDark(:,iLine,iDark),'k','LineWidth',1.5);
                        vectExtreme = logical((abs(dataDark(:,iLine,iDark) - RefMean)) > MultFact_Std_Fin*Ystd(iLine,iDark));
                        plot(x(not(vectExtreme)),dataDark(not(vectExtreme),iLine,iDark),'r','LineWidth',1.5);
                        titre = ['Hot Pix detection loop ',num2str(iLoop+1,'%d'),': fraction identified ',num2str(sum(vectExtreme)/size(vectExtreme,1),'%5.3f')];
                        title(titre)
                        grid on
                        xlim([1 size(dataDark,1)])
                    end
                end

            end



        end

    end


    nbrDark = size(vectAnomalyDark2,3);
    if nbrDark > 1.5
        vectHotPixels = (sum(vectAnomalyDark2,3) / nbrDark) >= 0.51; % when >= 51% detections (lower criteria) -> set as hot pixel
        vectCosmicRay = (logical(sum(vectAnomalyDark,3)) .* abs(sum(vectAnomalyDark2,3)-1<0.01)); % when only 1 detection -> set as cosmic ray
    else
        vectCosmicRay = (logical(sum(vectAnomalyDark,3))); % all detections (stronger criteria) are considered as cosmic ray
        vectHotPixels = zeros(size(vectCosmicRay)); % no Hot Pixels
    end

    MatrixHotPixels(iStartUsefulPix:iEndUsefulPix,:) = vectHotPixels;
    for iDark = 1:nbrDark
        MatrixCosmicRayDark(iStartUsefulPix:iEndUsefulPix,:,iDark) = vectCosmicRay .* vectAnomalyDark(:,:,iDark);
        vectAllAnomalies = logical(vectHotPixels + MatrixCosmicRayDark(iStartUsefulPix:iEndUsefulPix,:,iDark));
        temp_dataDark = dataDark(:,:,iDark);
        RowsWithCR = find(sum(MatrixCosmicRayDark(iStartUsefulPix:iEndUsefulPix,:,iDark),1));
        if not(isempty(RowsWithCR))
            for iRowsWithCR = RowsWithCR
                iPixWithCR = find(MatrixCosmicRayDark(iStartUsefulPix:iEndUsefulPix,iRowsWithCR,iDark));
                dataDark(iPixWithCR,iRowsWithCR,iDark) = median(temp_dataDark(not(vectAllAnomalies(:,iRowsWithCR)),iRowsWithCR),'omitnan');
            end
        end
        dataDark(:,:,iDark) = temp_dataDark;
    end

    data(iStartUsefulPix:iEndUsefulPix,:,:) = dataDark;

    if (ibinning == 0 || ibinning == 2)
        dataDark_CosmicRayRemoved = data;
    elseif (ibinning == 1) % if binned -> re-set as nPix x nObs
        MatrixCosmicRayDark = permute(MatrixCosmicRayDark,[1 3 2]);
        dataDark_CosmicRayRemoved = permute(data,[1 3 2]);
    end

end % function

"""


def anomaly_detection_dark(data, i_start_useful_pix, i_end_useful_pix, ibinning, strength_criteria):
    # do_plot = False

    data_dark = data[i_start_useful_pix:i_end_useful_pix+1, :, :]
    # print("data_dark.shape", data_dark.shape, data_dark[0,0,0])
    data_dark_no_extreme = np.copy(data_dark)
    nbr_dark = data_dark.shape[2]

    matrix_cosmic_ray_dark = np.zeros_like(data, dtype=bool)
    matrix_hot_pixels = np.zeros(data.shape[:2], dtype=bool)

    criteria_mapping = {
        "Strong": ([2, 2, 3], 3, 3),
        "Medium": ([2, 3], 4, 3),
        "Occult": ([4], 8, 16),
        "Occult_Bin": ([3], 4, 6),
        "LabMeas": ([3], 4, 4)
    }

    mult_fact_std, mult_fact_std_fin, mult_fact_std_cosmic_ray = criteria_mapping.get(
        strength_criteria, ([3], 4, 4))

    anomaly_matrix = np.zeros_like(data_dark, dtype=bool)
    
    for i_dark in range(nbr_dark):
        if ibinning in [0, 2]: # full frame

            for i_loop, mult_fact in enumerate(mult_fact_std):
                y_mean = np.median(data_dark_no_extreme[:, :, i_dark], axis=0)
                y_std = np.std(data_dark_no_extreme[:, :, i_dark], axis=0, ddof=1)
                if i_loop == 0:
                    y_std_init = np.copy(y_std)

                for i_line in range(data_dark.shape[1]):
                    vect_extreme = np.abs(data_dark[:, i_line, i_dark] - y_mean[i_line]) > mult_fact * y_std[i_line]
                    data_dark_no_extreme[vect_extreme, i_line, i_dark] = y_mean[i_line]
                    

            y_mean = np.median(data_dark_no_extreme[:, :, i_dark], axis=0)
            y_std = np.std(data_dark_no_extreme[:, :, i_dark], axis=0, ddof=1)
            # print(y_mean, y_std)

            for i_line in range(data_dark.shape[1]):
                if strength_criteria == "SpareAdapt":
                    ref_mean = median_filter(data_dark_no_extreme[:, i_line, i_dark], size=50)
                else:
                    ref_mean = y_mean[i_line]

                vect_anomaly_dark = np.abs(data_dark[:, i_line, i_dark] - ref_mean) > mult_fact_std_cosmic_ray * y_std_init[i_line]

                anomaly_matrix[:, i_line, i_dark] = vect_anomaly_dark
                # print(i_dark, i_line, np.sum(vect_anomaly_dark), mult_fact_std_cosmic_ray * y_std_init[i_line])
                # if i_line in [0,1]:
                    # print(i_dark, i_line, np.abs(data_dark[:10, i_line, i_dark] - ref_mean), ref_mean, mult_fact_std_cosmic_ray * y_std_init[i_line])
                    
                    # if i_dark==1:
                    #     stop()

    nbr_dark = anomaly_matrix.shape[2]
    if nbr_dark > 1:
        vect_hot_pixels = (np.sum(anomaly_matrix, axis=2) / nbr_dark) >= 0.51
        vect_cosmic_ray = (np.any(anomaly_matrix, axis=2)) & (np.abs(np.sum(anomaly_matrix, axis=2) - 1) < 0.01)
    else:
        vect_cosmic_ray = np.any(anomaly_matrix, axis=2)
        vect_hot_pixels = np.zeros_like(vect_cosmic_ray)
        
    # print("vect_cosmic_ray sum:", np.sum(vect_cosmic_ray))
    # print("vect_hot_pixels sum:", np.sum(vect_hot_pixels))

    matrix_hot_pixels[i_start_useful_pix:i_end_useful_pix+1, :] = vect_hot_pixels

    for i_dark in range(nbr_dark):
        matrix_cosmic_ray_dark[i_start_useful_pix:i_end_useful_pix+1, :, i_dark] = vect_cosmic_ray & anomaly_matrix[:, :, i_dark]
        temp_data_dark = np.copy(data_dark[:, :, i_dark])
        rows_with_cr = np.where(np.sum(matrix_cosmic_ray_dark[i_start_useful_pix:i_end_useful_pix+1, :, i_dark], axis=0) > 0)[0]

        # print("i_dark=%i, matrix_cosmic_ray_dark[i_start_useful_pix:i_end_useful_pix+1, :, i_dark] sum:" % i_dark, np.sum(matrix_cosmic_ray_dark[i_start_useful_pix:i_end_useful_pix+1, :, i_dark]))
        # print("i_dark=%i, rows_with_cr:" % i_dark, rows_with_cr)

        # find rows with cosmic rays
        for i_row in rows_with_cr:
            # find pixels with cosmic rays in each row
            i_pixels_with_cr = np.where(matrix_cosmic_ray_dark[i_start_useful_pix:i_end_useful_pix+1, i_row, i_dark])[0]

            # print("i_dark=%i, i_row_with_cr=%i, pixels_with_cr:" % (i_dark, i_row), i_pixels_with_cr)
            
            #replace cosmic ray values by the median of that row
            temp_data_dark[i_pixels_with_cr, i_row] = np.median(temp_data_dark[~vect_hot_pixels[:, i_row], i_row], axis=0)

        data_dark[:, :, i_dark] = temp_data_dark

    data[i_start_useful_pix:i_end_useful_pix+1, :, :] = data_dark

    if ibinning in [0, 2]:
        data_dark_cosmic_ray_removed = data
    elif ibinning == 1:
        #TODO: fix? shouldn't be here?
        matrix_cosmic_ray_dark = np.transpose(matrix_cosmic_ray_dark, (0, 2, 1))
        data_dark_cosmic_ray_removed = np.transpose(data, (0, 2, 1))

    return matrix_cosmic_ray_dark, matrix_hot_pixels, data_dark_cosmic_ray_removed


"""

function [temperaturefit] = Fitting_Temperature_CCD(itime,temperature,IT)

  %% Treatment of the temperatures :
  % Try to fit the temperature to get rid of the effetc of dircrete temperature coming from the probe. The discrete step between two temperature is around 0.4°K
    for i=1:size(itime,1)
        temps(i)=datenum(itime{i},'yyyy mmm dd HH:MM:SS');
    end
    temps=temps-min(temps);
    temps=temps*86400; % --> convert into seconds
    delta=max(temperature)-min(temperature);
    if ((max(size(temperature))<12)|(delta<1))
        myfittype = fittype('(c + d*log(x))','dependent',{'y'},'independent',{'x'},'coefficients',{'c','d'});
        y=temperature;
        x1=temps+IT/2;[myfit1,gof1] = fit(x1',y',myfittype);
        x2=temps+IT;[myfit2,gof2] = fit(x2',y',myfittype);
        x3=temps+3*IT/2;[myfit3,gof3] = fit(x3',y',myfittype);
        ind=find(min([gof1.rmse gof2.rmse gof3.rmse]));
        switch ind
            case 1
                x=x1;fonction=myfit1;
            case 2
                x=x2;fonction=myfit2;
            case 3
                x=x3;fonction=myfit3;
        end
        temperaturefit=feval(fonction,x');
        txt='Fit log';
    elseif (delta>4)
        fonction=fit(temps',temperature','poly6'); txt='Ordre6';
        temperaturefit=feval(fonction,temps');
    elseif (delta>1.5)
        fonction=fit(temps',temperature','poly3'); txt='Ordre3';
        temperaturefit=feval(fonction,temps');
    else
        fonction=fit(temps',temperature','poly2'); txt='Ordre2';
        temperaturefit=feval(fonction,temps');
    end
%     txt

    % check that no bad fitting occured
    vectBadFit = (abs(temperaturefit'-temperature)>0.39);
    if (sum(vectBadFit) > 0.5)
        temperaturefit(vectBadFit) = temperature(vectBadFit);
    end

end

"""


def log_fit(x, c, d):
    return c + d * np.log(x)


def fitting_temperature_ccd(itime, temperature, IT):
    """
    Fit the temperature data to remove discrete temperature effects from the probe.
    """
    # Convert datetime strings to seconds
    temps = np.array([(datetime.strptime(t.decode(), '%Y %b %d %H:%M:%S.%f') -
                       datetime.strptime(itime[0].decode(), '%Y %b %d %H:%M:%S.%f')).total_seconds() for t in itime])

    delta= np.max(temperature) - np.min(temperature)

    if len(temperature) < 12 or delta < 1:
        x1, x2, x3= temps + IT/2, temps + IT, temps + 3*IT/2

        popt1, _= curve_fit(log_fit, x1, temperature)
        popt2, _= curve_fit(log_fit, x2, temperature)
        popt3, _= curve_fit(log_fit, x3, temperature)

        rmse1= np.sqrt(np.mean((log_fit(x1, *popt1) - temperature) ** 2))
        rmse2= np.sqrt(np.mean((log_fit(x2, *popt2) - temperature) ** 2))
        rmse3= np.sqrt(np.mean((log_fit(x3, *popt3) - temperature) ** 2))

        best_fit= np.argmin([rmse1, rmse2, rmse3])
        x_best= [x1, x2, x3][best_fit]
        best_params= [popt1, popt2, popt3][best_fit]
        temperaturefit= log_fit(x_best, *best_params)
    elif delta > 4:
        poly_coeffs= np.polyfit(temps, temperature, 6)
        temperaturefit= np.polyval(poly_coeffs, temps)
    elif delta > 1.5:
        poly_coeffs= np.polyfit(temps, temperature, 3)
        temperaturefit= np.polyval(poly_coeffs, temps)
    else:
        poly_coeffs= np.polyfit(temps, temperature, 2)
        temperaturefit= np.polyval(poly_coeffs, temps)

    # Check for bad fitting
    vect_bad_fit= np.abs(temperaturefit - temperature) > 0.39
    temperaturefit[vect_bad_fit]= temperature[vect_bad_fit]

    return temperaturefit


"""

function [OverScan] = Calculation_OverScan(data,iStartOverscanPix)
    % data dimension:  nPixels x nRows x nObs
    % OverScan dimension:    1 x nRows x nObs

    nbrRows = size(data,2);
    nbrObs = size(data,3);

    % calculation of the raw overscan: mean on last 8 pixels
    OverScan_Raw = mean(data(iStartOverscanPix:end,:,:),1);


    % smoothing of the raw overscan on several CCD Rows
    nbrSmoothPts = 11;
    nbrExtendPix = floor(nbrSmoothPts/2);
    temp_vectExtend_NaN(1,1:nbrExtendPix) = NaN;
    for iObs = 1:nbrObs
        temp_OvSc_Extend = [temp_vectExtend_NaN OverScan_Raw(1,:,iObs) temp_vectExtend_NaN];
        temp_OvSc_Smoo = smooth(temp_OvSc_Extend',nbrSmoothPts)';

        OverScan(1,1:nbrRows,iObs) = temp_OvSc_Smoo(1,nbrExtendPix+1:nbrExtendPix+nbrRows);
    end

end

"""


def calculation_overscan(data, i_start_overscan_pix):

    # Parameters
    nbrSmoothPts = 11
    nbrExtendPix = nbrSmoothPts // 2
    
    # Extend with NaNs
    temp_vectExtend_NaN = np.full(nbrExtendPix, np.nan)

    if data.ndim == 2:
        overscan_raw= np.mean(data[i_start_overscan_pix:, :], axis=0)
        temp_OvSc_Extend = np.concatenate((temp_vectExtend_NaN, overscan_raw, temp_vectExtend_NaN))

        # Lowess smoothing
        py_low = lowess(temp_OvSc_Extend, np.arange(len(temp_OvSc_Extend)), frac=nbrSmoothPts/len(overscan_raw), it=0)[:, 1]

        return py_low

        
    elif data.ndim == 3:
        nbrRows = data.shape[1]
        nbrObs = data.shape[2]
        overscan = np.zeros((nbrRows, nbrObs))

        for iObs in np.arange(nbrObs):
            overscan_raw= np.mean(data[i_start_overscan_pix:, :, iObs], axis=0)
        
        
            temp_OvSc_Extend = np.concatenate((temp_vectExtend_NaN, overscan_raw, temp_vectExtend_NaN))
        
            # Lowess smoothing
            py_low = lowess(temp_OvSc_Extend, np.arange(len(temp_OvSc_Extend)), frac=nbrSmoothPts/len(overscan_raw), it=0)[:, 1]
            
            overscan[:, iObs] = py_low
    
        return overscan


def calculation_overscan_old(data, i_start_overscan_pix):
    """
    Calculate the overscan values for CCD data.

    Parameters:
        data (numpy.ndarray): 3D array (nPixels x nRows x nObs).
        i_start_overscan_pix (int): Index where overscan pixels start.
        
        data = bias1
        i_start_overscan_pix = iStartOverscanPix

    Returns:
        numpy.ndarray: 1D array (1 x nRows x nObs) containing overscan values.
    """
    overscan_raw= np.mean(data[i_start_overscan_pix:, :], axis=0, keepdims=True)

    # Smoothing the raw overscan over several CCD rows
    nbr_smooth_pts= 11
    overscan_smoothed= uniform_filter1d(overscan_raw, size=nbr_smooth_pts, axis=1, mode='nearest')

    return overscan_smoothed


"""

function [U2RN] = ReadOutNoise_Calculation(indbias,bias1,bias2,iStartUsefulPix,iEndUsefulPix)

    frame = bias1 - bias2;
    frame = frame (iStartUsefulPix:iEndUsefulPix,:);

    readoutnoise = std(frame(1:end),'omitnan');
    U2RN = (readoutnoise/sqrt(2))^2;

end
"""


def readout_noise_calculation(bias1, bias2, i_start_useful_pix, i_end_useful_pix):
    """
    Calculate the readout noise (U2RN) from bias frames.

    Parameters:
        bias1 (numpy.ndarray): First bias frame.
        bias2 (numpy.ndarray): Second bias frame.
        i_start_useful_pix (int): Start index of useful pixels.
        i_end_useful_pix (int): End index of useful pixels.

    Returns:
        float: Readout noise value (U2RN).
    """
    frame= bias1[i_start_useful_pix:i_end_useful_pix, :] - \
        bias2[i_start_useful_pix:i_end_useful_pix, :]
    readout_noise= np.nanstd(frame, ddof=1)
    u2rn= (readout_noise / np.sqrt(2)) ** 2

    return u2rn


"""

function [iLineTopStart,iLineBottomStart] = Define_Top_Bottom_WellIlluminated_Lines(imode,x,Instrument)

    if imode == 1 % occultation

        if strcmp(Instrument,'Flight')

            iLineBottomStart = interp1([1:93:1024]+8,[171 171 170 169 167 165 163 161 159 157 155 152],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineBottomStart = round(iLineBottomStart);
            iLineTopStart = interp1([1:93:1024]+8,[177 177 177 177 177 177 177 177 178 179 181 183],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineTopStart = round(iLineTopStart);

        elseif strcmp(Instrument,'Spare')

        end


    elseif imode == 2 % nadir

        if strcmp(Instrument,'Flight')

            iLineBottomStart = interp1([1:93:1024]+8,[131 130 129 130 129 129 129 129 129 129 128 128],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineBottomStart = round(iLineBottomStart);
            iLineTopStart = interp1([1:93:1024]+8,[217 217 217 217 215 214 212 212 210 210 210 209],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineTopStart = round(iLineTopStart);

        elseif strcmp(Instrument,'Spare')

            iLineBottomStart = interp1([1:93:1024]+8,[55 54 54 54 54 54 54 54 55 55 55 55],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineBottomStart = round(iLineBottomStart);
            iLineTopStart = interp1([1:93:1024]+8,[139 139 139 139 139 138 138 138 137 137 137 137],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineTopStart = round(iLineTopStart);

        end

    end

end
"""


def define_top_bottom_well_illuminated_lines(imode, x, instrument):
    """outputs same as matlab with indices subtracted by 1"""
    if imode == 1:  # Occultation
        if instrument == 'Flight':
            x_ref= np.array([1, 94, 187, 280, 373, 466, 559,
                              652, 745, 838, 931, 1024]) + 8
            bottom_vals= np.array(
                [171, 171, 170, 169, 167, 165, 163, 161, 159, 157, 155, 152])
            top_vals= np.array(
                [177, 177, 177, 177, 177, 177, 177, 177, 178, 179, 181, 183])
        elif instrument == 'Spare':
            return None, None

    elif imode == 2:  # Nadir
        if instrument == 'Flight':
            x_ref= np.array([1, 94, 187, 280, 373, 466, 559,
                              652, 745, 838, 931, 1024]) + 8
            bottom_vals= np.array(
                [131, 130, 129, 130, 129, 129, 129, 129, 129, 129, 128, 128])
            top_vals= np.array(
                [217, 217, 217, 217, 215, 214, 212, 212, 210, 210, 210, 209])
        elif instrument == 'Spare':
            x_ref= np.array([1, 94, 187, 280, 373, 466, 559,
                              652, 745, 838, 931, 1024]) + 8
            bottom_vals= np.array(
                [55, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55])
            top_vals= np.array(
                [139, 139, 139, 139, 139, 138, 138, 138, 137, 137, 137, 137])
        else:
            return None, None
    else:
        return None, None
    
    # python correction
    # x_ref -= 1
    bottom_vals -= 1
    top_vals -= 1
    
    i_line_bottom_start= np.round(
        np.interp(x, x_ref, bottom_vals)).astype(int)
    i_line_top_start= np.round(np.interp(x, x_ref, top_vals)).astype(int)
    # print(i_line_bottom_start[17:19])
    # print(i_line_bottom_start[28:30])
    # print(i_line_top_start[86:88])

    return i_line_top_start, i_line_bottom_start


"""

function [IsDarkMeasurement,Diff] = Check_isDarkMeasurement(Frame,XNbBin,Diff_Ref)

    LimInterval1 = [round((1+8)/XNbBin(1)):round((128+8)/XNbBin(1))]; % 200- 260 nm % 200- 260 nm
    LimInterval2 = [round((432+8)/XNbBin(1)):round((624+8)/XNbBin(1))];% 400-480 nm

    if size(Frame,1) < max(LimInterval2)
        LimInterval1 = [1:round(size(Frame,1)/2)]; % 200- 260 nm
        LimInterval2 = [round(size(Frame,1)/2):size(Frame,1)];% 400-480 nm
    end

    ValueInt1 = mean(median(Frame(LimInterval1,:),1),2);
    ValueInt2 = mean(median(Frame(LimInterval2,:),1),2);
%     RelDiff = abs((ValueInt2/ValueInt1)-1);
    Diff = abs(ValueInt2-ValueInt1);

%     IsDarkMeasurement = RelDiff < RelDiff_Ref; % difference < 5%
    if not(isempty(Diff_Ref))
        IsDarkMeasurement = Diff < Diff_Ref;
    else
        IsDarkMeasurement = 1;
    end


end

"""


def check_is_dark_measurement(frame, x_nb_bin, diff_ref=None):
    """
    Determines if a given frame is a dark measurement by comparing intensity differences
    in two wavelength intervals.

    Parameters:
        frame (numpy.ndarray): The input frame (2D array).
        x_nb_bin (list or numpy.ndarray): Binning factor.
        diff_ref (float, optional): Reference difference threshold.

    Returns:
        is_dark_measurement (bool): True if frame is considered a dark measurement.
        diff (float): Difference between two selected intervals.
    """

    lim_interval1= np.arange(
        round((1+8)/x_nb_bin[0]), round((128+8)/x_nb_bin[0]))  # 200-260 nm
    lim_interval2= np.arange(
        round((432+8)/x_nb_bin[0]), round((624+8)/x_nb_bin[0]))  # 400-480 nm

    if frame.shape[0] < max(lim_interval2, default=0):  # Ensure valid indexing
        lim_interval1= np.arange(0, round(frame.shape[0]/2))
        lim_interval2= np.arange(round(frame.shape[0]/2), frame.shape[0])

    value_int1= np.mean(np.median(frame[lim_interval1, :], axis=1))
    value_int2= np.mean(np.median(frame[lim_interval2, :], axis=1))

    diff= abs(value_int2 - value_int1)

    if diff_ref is not None:
        is_dark_measurement= diff < diff_ref
    else:
        is_dark_measurement= True

    return is_dark_measurement, diff


"""

function [Value,dValue_dT] = DC_vs_T_relation(Instrument,params,T)

    if strcmp(Instrument,'Flight')

        % % Single exponential with 2 params: a * exp(b*T)
        % a = params(1); b = params(2);
        % Value = a * exp(b*T);

        % dValue_dT = (a*b) * exp(b*T);


        % Single exponential with 3 params: a * exp(b*T) + c
        a = params(1); b = params(2); c = params(3);
        Value = a * exp(b*T) + c;

        dValue_dT = (a*b) * exp(b*T);


        % % Double exponential: a * exp(b*T) + c * exp(d*T)
        % a = params(1); b = params(2); c = params(3); d = params(4);
        % Value = a * exp(b*T) + c * exp(d*T);

        % dValue_dT = (a*b) * exp(b*T) + (c*d) * exp(d*T);

    elseif  strcmp(Instrument,'Spare')

        % DC calculation follows law: DC(T) = a * exp(b*T)   (* TI(s))
        a = 1.947e-17; b = 0.1473;

        Value = a * exp(b*T);

        dValue_dT = (a*b) * exp(b*T);

    end

end
"""


def DC_vs_T_relation(Instrument, params, T):
    # if Instrument == 'Flight':
        # Single exponential with 3 params: a * exp(b*T) + c
    a= params[0]
    b= params[1]
    c= params[2]
    Value= a * np.exp(b * T) + c
    dValue_dT= (a * b) * np.exp(b * T)

    # elif Instrument == 'Spare':
    #     # DC calculation follows law: DC(T) = a * exp(b*T)
    #     a= 1.947e-17
    #     b= 0.1473
    #     Value= a * np.exp(b * T)
    #     dValue_dT= (a * b) * np.exp(b * T)

    return Value, dValue_dT


"""

function [DC,U2DC,frac_DC1] = Calculation_DC_FromTempFitFraction(Instrument,ibinning,DC1,DC2,U2DC1,U2DC2,T1,T2,T,params)

    [Value_From_Fit,dValue_From_Fit_dT] = DC_vs_T_relation(Instrument,params,T);
    [Value_T1,dValue_T1_dT1] = DC_vs_T_relation(Instrument,params,T1);
    [Value_T2,dValue_T2_dT2] = DC_vs_T_relation(Instrument,params,T2);

    Delta_Value_T2_T = (Value_T2-Value_From_Fit);
    Delta_Value_T2_T1 = (Value_T2-Value_T1);

    frac_DC1 = abs( Delta_Value_T2_T ./ Delta_Value_T2_T1 );
    frac_DC2 = 1 - frac_DC1;
    Delta_DC = DC1-DC2;

    dDC_dDC1 = frac_DC1;
    dDC_dDC2 = frac_DC2;
    dDC_dT1_part1 = (-(-dValue_T1_dT1) .* ( Delta_Value_T2_T ./ (Delta_Value_T2_T1.^2)));
    dDC_dT1 = permute(dDC_dT1_part1,[2 3 1]) .* Delta_DC; % npix x nlines x nobs
    dDC_dT2_part1 = ((dValue_T2_dT2.*(Delta_Value_T2_T1-Delta_Value_T2_T)) ./ (Delta_Value_T2_T1.^2));
    dDC_dT2 = permute(dDC_dT2_part1,[2 3 1]) .* Delta_DC; % npix x nlines x nobs
    dDC_dT_part1 = (-dValue_From_Fit_dT ./ Delta_Value_T2_T1);
    dDC_dT = permute(dDC_dT_part1,[2 3 1]) .* Delta_DC; % npix x nlines x nobs

    U2T1 = ((0.39/2)^2)/3;
    U2T2 = U2T1;
    U2T = U2T1;

    for iDC = 1:size(T)

        DC(:,:,iDC) = (DC1.*frac_DC1(iDC)) + (DC2.*frac_DC2(iDC));

        U2DC(:,:,iDC) = ((dDC_dDC1(iDC).^2) .* U2DC1) + ((dDC_dDC2(iDC).^2) .* U2DC2) + ((dDC_dT1(:,:,iDC).^2) .* U2T1) + ((dDC_dT2(:,:,iDC).^2) .* U2T2) + ((dDC_dT(:,:,iDC).^2) .* U2T);
    end

    if (ibinning == 1)
        DC = permute(DC,[1 3 2]);
    end

end
"""


def Calculation_DC_FromTempFitFraction(Instrument, ibinning, DC1, DC2, U2DC1, U2DC2, T1, T2, T, params):
    # Get values and derivatives from the fit and for temperatures T1 and T2
    Value_From_Fit, dValue_From_Fit_dT= DC_vs_T_relation(Instrument, params, T)
    Value_T1, dValue_T1_dT1= DC_vs_T_relation(Instrument, params, T1)
    Value_T2, dValue_T2_dT2= DC_vs_T_relation(Instrument, params, T2)

    # Calculate differences for fraction of DC1 and DC2
    Delta_Value_T2_T= Value_T2 - Value_From_Fit
    Delta_Value_T2_T1= Value_T2 - Value_T1
    frac_DC1= np.abs(Delta_Value_T2_T / Delta_Value_T2_T1)
    frac_DC2= 1 - frac_DC1
    Delta_DC= DC1 - DC2

    # Derivatives with respect to DC1, DC2, T1, T2, and T
    dDC_dDC1= frac_DC1
    dDC_dDC2= frac_DC2
    dDC_dT1_part1= -(-dValue_T1_dT1) * (Delta_Value_T2_T / (Delta_Value_T2_T1 ** 2))
    dDC_dT1= dDC_dT1_part1 * np.repeat(Delta_DC, len(dDC_dT1_part1), axis=2)  # npix x nlines x nobs
    dDC_dT2_part1= (dValue_T2_dT2 * (Delta_Value_T2_T1 - Delta_Value_T2_T)) / (Delta_Value_T2_T1 ** 2)
    dDC_dT2= dDC_dT2_part1 * np.repeat(Delta_DC, len(dDC_dT1_part1), axis=2)  # npix x nlines x nobs
    dDC_dT_part1= -dValue_From_Fit_dT / Delta_Value_T2_T1
    dDC_dT= dDC_dT_part1 * np.repeat(Delta_DC, len(dDC_dT1_part1), axis=2)  # npix x nlines x nobs

    # Uncertainty in temperature
    U2T1= (0.39 / 2) ** 2 / 3
    U2T2= U2T1
    U2T= U2T1

    DC= np.zeros((DC1.shape[0], DC1.shape[1], T.shape[0]))
    U2DC= np.zeros((U2DC1.shape[0], U2DC1.shape[1], T.shape[0]))
    
    for iDC in range(T.shape[0]):
        DC[:, :, iDC]= (np.squeeze(DC1) * frac_DC1[iDC]) + (np.squeeze(DC2) * frac_DC2[iDC])

        U2DC[:, :, iDC]= (
            (dDC_dDC1[iDC] ** 2) * np.squeeze(U2DC1) +
            (dDC_dDC2[iDC] ** 2) * np.squeeze(U2DC2) +
            (dDC_dT1[:, :, iDC] ** 2) * U2T1 +
            (dDC_dT2[:, :, iDC] ** 2) * U2T2 +
            (dDC_dT[:, :, iDC] ** 2) * U2T
        )

    if ibinning == 1:
        DC= np.transpose(DC, (0, 2, 1))

    return DC, U2DC, frac_DC1


"""

function [Correction_FirstDC,LevelDC_Dark1] = Correction_DC1_offset_NLO_Obs(
    imode,Xpix_1b,XNbBin,frame_DC1,dataPack,iStartLineCCD,iEndLineCCD,DC1_corr_RowsUsed,Instrument,ibinning,DC1,DC2,U2DC1,U2DC2,T1,T2,T,params)

    iFrame = 1;
    nbrFrames = size(dataPack,3);

    [iLineTopStart,iLineBottomStart] = Define_Top_Bottom_WellIlluminated_Lines(imode,Xpix_1b,Instrument);
    RowsIllum = [min(iLineBottomStart):max(iLineTopStart)] - iStartLineCCD + 1;

    DC_Frame_Obs = permute(mean(median(dataPack(:,:,:),1),2),[3 1 2]);
    DC_Frame_Obs = smooth(DC_Frame_Obs,3);

    [IsDarkMeasurement1,Diff1] = Check_isDarkMeasurement(DC1(:,RowsIllum),XNbBin,[]);
    [IsDarkMeasurementEnd,DiffEnd] = Check_isDarkMeasurement(DC2(:,RowsIllum),XNbBin,[]);
    Diff_Ref = 2*max(Diff1,DiffEnd);

    % find the first measurement that is still dark and not concerned by the unexplained signal rise
    DC_CurrentFrame = DC_Frame_Obs(iFrame,1);
    DC_NextFrame = DC_Frame_Obs(iFrame+1,1);
    [IsDarkMeasurement2,unused] = Check_isDarkMeasurement(dataPack(:,RowsIllum,iFrame+1),XNbBin,Diff_Ref);

    IsDarkMeasurement_Init = (IsDarkMeasurement1 & IsDarkMeasurement2);
    [IsDarkMeasurement,unused] = Check_isDarkMeasurement(dataPack(:,RowsIllum,iFrame+2),XNbBin,Diff_Ref);

    while (DC_NextFrame < DC_CurrentFrame) & IsDarkMeasurement_Init & IsDarkMeasurement & (iFrame < nbrFrames-1)
        iFrame = iFrame + 1;
        DC_CurrentFrame = DC_Frame_Obs(iFrame,1);
        DC_NextFrame = DC_Frame_Obs(iFrame+1,1);

        IsDarkMeasurement = Check_isDarkMeasurement(dataPack(:,RowsIllum,iFrame+2),XNbBin,Diff_Ref);
    end

    % Calculate the correction
    RowsForMean = [max(DC1_corr_RowsUsed(1),iStartLineCCD):min(DC1_corr_RowsUsed(2),iEndLineCCD)]; % in 256 rows reference
    RowsForMean = RowsForMean - iStartLineCCD + 1; % adapted to frame
    if IsDarkMeasurement_Init
        if (DC_NextFrame > DC_CurrentFrame)

            % check if next frame can also be used to produce FakeDC1 (and take the average to reduce the impact of the obs by obs noise)
            nbrOfUsableObs = 1;
            while IsDarkMeasurement & (nbrOfUsableObs < 5)
                nbrOfUsableObs = nbrOfUsableObs + 1;
                IsDarkMeasurement = Check_isDarkMeasurement(dataPack(:,RowsIllum,iFrame+1+nbrOfUsableObs),XNbBin,Diff_Ref);
            end


            [unused,unused,frac_DC1] = Calculation_DC_FromTempFitFraction(Instrument,ibinning,DC1,DC2,U2DC1,U2DC2,T1,T2,T,params);

            nbrPtsSmooth = 5;
            LevelDC_Dark1 = smooth(median(DC1(:,:),1),nbrPtsSmooth)';
            LevelDC_Dark2 = smooth(median(DC2(:,:),1),nbrPtsSmooth)';
            for iUsableObs = 1:nbrOfUsableObs
                LevelDC_Obs = smooth(median(dataPack(:,:,iFrame+iUsableObs),1),nbrPtsSmooth)';
                FakeDC1(iUsableObs,:) = ( LevelDC_Obs - ((1-frac_DC1(iFrame+iUsableObs))*LevelDC_Dark2) ) / frac_DC1(iFrame+iUsableObs);
            end
            MedianFakeDC1_Avg = median(FakeDC1,1);
            Correction_FirstDC = MedianFakeDC1_Avg - LevelDC_Dark1;

        else
            Correction_FirstDC = zeros(1,size(frame_DC1,2)); % no correction
            LevelDC_Dark1 = Correction_FirstDC;
        end
    else
        Correction_FirstDC = zeros(1,size(frame_DC1,2)); % no correction
        LevelDC_Dark1 = Correction_FirstDC;
    end

end
"""


# def Correction_DC1_offset_NLO_Obs(imode, Xpix_1b, XNbBin, frame_DC1, dataPack, iStartLineCCD, iEndLineCCD, DC1_corr_RowsUsed, Instrument, ibinning, DC1, DC2, U2DC1, U2DC2, T1, T2, T, params):
#     iFrame= 1
#     nbrFrames= dataPack.shape[2]

#     # Get top and bottom lines for illumination
#     iLineTopStart, iLineBottomStart= define_top_bottom_well_illuminated_lines(
#         imode, Xpix_1b, Instrument)
#     RowsIllum= np.arange(np.min(iLineBottomStart), np.max(
#         iLineTopStart) + 1) - iStartLineCCD + 1

#     DC_Frame_Obs= np.transpose(
#         np.mean(np.median(dataPack, axis=1), axis=1), (2, 0, 1))
#     DC_Frame_Obs= savgol_filter(DC_Frame_Obs, 3, 1)  # Applying smoothing

#     # Check for dark measurement
#     IsDarkMeasurement1, Diff1= check_is_dark_measurement(
#         DC1[:, RowsIllum], XNbBin)
#     IsDarkMeasurementEnd, DiffEnd= check_is_dark_measurement(
#         DC2[:, RowsIllum], XNbBin)
#     Diff_Ref= 2 * np.maximum(Diff1, DiffEnd)

#     DC_CurrentFrame= DC_Frame_Obs[iFrame, 0]
#     DC_NextFrame= DC_Frame_Obs[iFrame + 1, 0]

#     IsDarkMeasurement2, unused= check_is_dark_measurement(
#         dataPack[:, RowsIllum, iFrame + 1], XNbBin, Diff_Ref)

#     IsDarkMeasurement_Init= IsDarkMeasurement1 & IsDarkMeasurement2
#     IsDarkMeasurement, unused= check_is_dark_measurement(
#         dataPack[:, RowsIllum, iFrame + 2], XNbBin, Diff_Ref)

#     while DC_NextFrame < DC_CurrentFrame and IsDarkMeasurement_Init and IsDarkMeasurement and iFrame < nbrFrames - 1:
#         iFrame += 1
#         DC_CurrentFrame= DC_Frame_Obs[iFrame, 0]
#         DC_NextFrame= DC_Frame_Obs[iFrame + 1, 0]
#         IsDarkMeasurement, unused= check_is_dark_measurement(
#             dataPack[:, RowsIllum, iFrame + 2], XNbBin, Diff_Ref)

#     # Calculate the correction
#     # RowsForMean= np.arange(np.maximum(DC1_corr_RowsUsed[0], iStartLineCCD), np.minimum(
#     #     DC1_corr_RowsUsed[1], iEndLineCCD) + 1) - iStartLineCCD + 1

#     if IsDarkMeasurement_Init:
#         if DC_NextFrame > DC_CurrentFrame:

#             # Check if next frame can be used to produce FakeDC1
#             nbrOfUsableObs= 1
#             while IsDarkMeasurement and nbrOfUsableObs < 5:
#                 nbrOfUsableObs += 1
#                 IsDarkMeasurement= check_is_dark_measurement(
#                     dataPack[:, RowsIllum, iFrame + 1 + nbrOfUsableObs], XNbBin, Diff_Ref)

#             frac_DC1= Calculation_DC_FromTempFitFraction(
#                 Instrument, ibinning, DC1, DC2, U2DC1, U2DC2, T1, T2, T, params)[2]

#             nbrPtsSmooth= 5
#             LevelDC_Dark1= savgol_filter(np.median(DC1, axis=1), nbrPtsSmooth)
#             LevelDC_Dark2= savgol_filter(np.median(DC2, axis=1), nbrPtsSmooth)

#             FakeDC1= np.zeros((nbrOfUsableObs, DC1.shape[1]))
#             for iUsableObs in range(nbrOfUsableObs):
#                 LevelDC_Obs= savgol_filter(
#                     np.median(dataPack[:, :, iFrame + iUsableObs], axis=1), nbrPtsSmooth)
#                 FakeDC1[iUsableObs, :]= (
#                     LevelDC_Obs - ((1 - frac_DC1[iFrame + iUsableObs]) * LevelDC_Dark2)) / frac_DC1[iFrame + iUsableObs]

#             MedianFakeDC1_Avg= np.median(FakeDC1, axis=0)
#             Correction_FirstDC= MedianFakeDC1_Avg - LevelDC_Dark1

#         else:
#             Correction_FirstDC= np.zeros(DC1.shape[1])
#             LevelDC_Dark1= Correction_FirstDC
#     else:
#         Correction_FirstDC= np.zeros(DC1.shape[1])
#         LevelDC_Dark1= Correction_FirstDC

#     return Correction_FirstDC, LevelDC_Dark1


"""

function [U2SN] = ShotNoise_Calculation(data_DCremoved,Gain)

    % Shot noise(e) = sqrt( data(adu) * gain(e/adu) )
    % Shot noise(adu) =  Shot noise(e) / gain(e/adu) = sqrt ( data(adu) / gain(e/adu) )

    U2SN = abs(data_DCremoved) / Gain; % set to zero where negative

end


"""


def ShotNoise_Calculation(data_DCremoved, Gain):
    # Shot noise (e) = sqrt(data(adu) * gain(e/adu))
    # Shot noise (adu) = Shot noise(e) / gain(e/adu) = sqrt(data(adu) / gain(e/adu))

    U2SN= np.abs(data_DCremoved) / Gain  # set to zero where negative
    return U2SN


"""

function [iLineTopStart,iLineBottomStart] = Define_Top_Bottom_NonIlluminated_Lines(imode,X,model)

    delta_x = X(2) - X(1);
    x = floor(X + delta_x/2); % when binned, take the value of the rightest binned pix

    if imode == 1 % occultation

        if strcmp(model,'Flight')

            % % OU wavelength assignment
            % % iLineBottomStart = interp1([9.69 125.75 218.94 454.11 774.66 852.67 1032.39],[166 165 163 158 151 149 145],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            % iLineBottomStart = interp1([9.69 125.75 218.94 454.11 774.66 852.67 1032.39]-8,[166 165 163 158 151 149 145]+1,x,'linear','extrap'); % en 1024 pix
            % iLineBottomStart = floor(iLineBottomStart);
            % % iLineTopStart = interp1([9.69 125.75 218.94 454.11 774.66 852.67 1032.39],[180 182 184 186 188 189 190],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            % iLineTopStart = interp1([9.69 125.75 218.94 454.11 774.66 852.67 1032.39]-8,[180 182 184 186 188 189 190]+1,x,'linear','extrap'); % en 1024 pix
            % iLineTopStart = ceil(iLineTopStart);

            % Conservation for non-illuminated region, but less good for the polyfit2 SL interpolation? (less SL produced?)
            iLineBottomStart = interp1([1:93:1024]+8,[167 164 163 161 158 157 155 153 151 149 146 144],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineBottomStart = floor(iLineBottomStart);
            iLineTopStart = interp1([1:93:1024]+8,[183 183 184 185 185 186 187 188 189 190 192 193],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineTopStart = ceil(iLineTopStart);

%             % Not conserv for non-illuminated region, but better good for the polyfit2 SL interpolation? (more SL produced?)
        elseif strcmp(model,'Spare')

            iLineBottomStart = interp1([1:93:1024]+8,[89 75 77 79 83 81 79 77 75 73 71 69],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineBottomStart = floor(iLineBottomStart);
            iLineTopStart = interp1([1:93:1024]+8,[103 105 106 107 108 109 110 112 114 116 118 120],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineTopStart = ceil(iLineTopStart);

        end


    elseif imode == 2 % nadir

        if strcmp(model,'Flight')

            % Seems ROI has moves between 2015 and 2018: 2019 seems shifted toward the higher ccd lines (2 lines shift?)
            iLineBottomStart = interp1([1:93:1024]+8,[126 125 123 121 119 117 114.5 112 109.5 106.5 103.5 100],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineBottomStart = floor(iLineBottomStart);
            iLineTopStart = interp1([1:93:1024]+8,[225 225 225 225 226 227 228 228 230 231 233 234],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineTopStart = ceil(iLineTopStart);


        elseif strcmp(model,'Spare')
            iLineBottomStart = interp1([1:93:1024]+8,[47 46 45 44 43 41 40 38 36 34 32 30],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineBottomStart = floor(iLineBottomStart);
            iLineTopStart = interp1([1:93:1024]+8,[146 147 147 149 150 151 153 154 156 159 161 163],x,'linear','extrap'); % x counting is in 1048 pix (even if dim(x)=1024))
            iLineTopStart = ceil(iLineTopStart);


        end

    end

end
"""


def Define_Top_Bottom_NonIlluminated_Lines(imode, X, model):
    delta_x= X[1] - X[0]
    # when binned, take the value of the rightest binned pix
    x= np.floor(X + delta_x / 2)

    if imode == 1:  # occultation
        if model == 'Flight':
            # Conservation for non-illuminated region
            iLineBottomStart= np.interp(x, np.arange(1, 1025, 93) + 8.0, np.array(
                [167, 164, 163, 161, 158, 157, 155, 153, 151, 149, 146, 144]), left=None, right=None)
            iLineBottomStart= np.floor(iLineBottomStart).astype(int)
            iLineTopStart= np.interp(x, np.arange(1, 1025, 93) + 8, np.array(
                [183, 183, 184, 185, 185, 186, 187, 188, 189, 190, 192, 193]), left=None, right=None)
            iLineTopStart= np.ceil(iLineTopStart).astype(int)

        elif model == 'Spare':
            iLineBottomStart= np.interp(x, np.arange(1, 1025, 93) + 8, np.array(
                [89, 75, 77, 79, 83, 81, 79, 77, 75, 73, 71, 69]), left=None, right=None)
            iLineBottomStart= np.floor(iLineBottomStart).astype(int)
            iLineTopStart= np.interp(x, np.arange(1, 1025, 93) + 8, np.array(
                [103, 105, 106, 107, 108, 109, 110, 112, 114, 116, 118, 120]), left=None, right=None)
            iLineTopStart= np.ceil(iLineTopStart).astype(int)

    elif imode == 2:  # nadir
        if model == 'Flight':
            # Seems ROI has moved between 2015 and 2018, shifted toward the higher CCD lines (2 lines shift?)
            iLineBottomStart= np.interp(x, np.arange(1, 1025, 93) + 8, np.array(
                [126, 125, 123, 121, 119, 117, 114.5, 112, 109.5, 106.5, 103.5, 100]), left=None, right=None)
            iLineBottomStart= np.floor(iLineBottomStart).astype(int)
            iLineTopStart= np.interp(x, np.arange(1, 1025, 93) + 8, np.array(
                [225, 225, 225, 225, 226, 227, 228, 228, 230, 231, 233, 234]), left=None, right=None)
            iLineTopStart= np.ceil(iLineTopStart).astype(int)

        elif model == 'Spare':
            iLineBottomStart= np.interp(x, np.arange(1, 1025, 93) + 8, np.array(
                [47, 46, 45, 44, 43, 41, 40, 38, 36, 34, 32, 30]), left=None, right=None)
            iLineBottomStart= np.floor(iLineBottomStart).astype(int)
            iLineTopStart= np.interp(x, np.arange(1, 1025, 93) + 8, np.array(
                [146, 147, 147, 149, 150, 151, 153, 154, 156, 159, 161, 163]), left=None, right=None)
            iLineTopStart= np.ceil(iLineTopStart).astype(int)

    return iLineTopStart, iLineBottomStart


"""

function [ ROI_ok,iStartLineROI,iEndLineROI,nbrROILines,nbrSmearingLines_TopROI,MatBinROI ] = Define_ROI( Instrument,imode,ibinning,iStartLineCCD,iEndLineCCD,Xpix_1b )


    [iLineTopStart,iLineBottomStart] = Define_Top_Bottom_WellIlluminated_Lines(imode,Xpix_1b,Instrument);
    iStartLineROI_ff = double(iLineBottomStart);
    iEndLineROI_ff = double(iLineTopStart);

    nbrSmearingLines_TopROI = (max(iEndLineROI_ff)-iStartLineCCD) + 1;

    % Check and define ROI in the recorded current frame (line starting to count at iStartLineCCD) ------------------------------------------------------------
    if (ibinning == 0 || ibinning == 2) %% Fullframe

        iStartLineROI = iStartLineROI_ff - iStartLineCCD + 1;

        if sum(iStartLineROI < 1) == 0
            ROI_ok = 1;
        else
            ROI_ok = 0;
            iStartLineROI(iStartLineROI < 1) = 1;
        end


        iEndLineROI = iEndLineROI_ff - iStartLineCCD + 1;

        if sum(iEndLineROI_ff >= iEndLineCCD) == 0
            ROI_ok = ROI_ok * 1;
        else
            ROI_ok = ROI_ok * 0;
            iEndLineROI(iEndLineROI_ff >= iEndLineCCD) = iEndLineCCD - iStartLineCCD + 1;
        end

        nbrROILines = (iEndLineROI-iStartLineROI) + 1;

    elseif (ibinning == 1)
        nbrROILines = (iEndLineCCD-iStartLineCCD) + 1;
        iStartLineROI = [];
        iEndLineROI = [];
        ROI_ok = [];
    end

    MatBinROI = zeros(size(Xpix_1b,1),max(nbrROILines));
    for ipix =1:size(iLineTopStart,1)
        MatBinROI(ipix,iStartLineROI(ipix)-min(iStartLineROI)+1:iEndLineROI(ipix)-min(iStartLineROI)+1) = 1;
    end

end
"""


def Define_ROI(Instrument, imode, ibinning, iStartLineCCD, iEndLineCCD, Xpix_1b):
    # Define the top and bottom illuminated lines based on the mode and instrument
    # offset by compared to matlab
    iLineTopStart, iLineBottomStart= define_top_bottom_well_illuminated_lines(
        imode, Xpix_1b, Instrument)

    # offset by compared to matlab
    iStartLineROI_ff= np.array(iLineBottomStart, dtype=float)
    iEndLineROI_ff= np.array(iLineTopStart, dtype=float)

    # Calculate the number of smearing lines for the top ROI
    # same values as matlab
    nbrSmearingLines_TopROI= (np.max(iEndLineROI_ff) - iStartLineCCD) + 2

    # Check and define ROI in the current frame
    if ibinning == 0 or ibinning == 2:  # Fullframe
        # offset by compared to matlab
        iStartLineROI= np.asarray(iStartLineROI_ff - iStartLineCCD + 1, dtype=int)

        # Check for out-of-bounds and correct
        if np.sum(iStartLineROI < 1) == 0:
            ROI_ok= 1
        else:
            ROI_ok= 0
            iStartLineROI[iStartLineROI < 1]= 1

        # offset by compared to matlab
        iEndLineROI= np.asarray(iEndLineROI_ff - iStartLineCCD + 1, dtype=int)

        # Ensure iEndLineROI does not exceed the total number of lines in the CCD
        if np.sum(iEndLineROI_ff >= iEndLineCCD) == 0:
            ROI_ok *= 1
        else:
            ROI_ok *= 0
            iEndLineROI[iEndLineROI_ff >= iEndLineCCD]= iEndLineCCD - iStartLineCCD + 1

        # same values as matlab
        nbrROILines= (iEndLineROI - iStartLineROI) + 1

    elif ibinning == 1:  # For binning
        nbrROILines= (iEndLineCCD - iStartLineCCD) + 1
        iStartLineROI= []
        iEndLineROI= []
        ROI_ok= None  # ROI not defined in this case

    # Create the matrix for the binned ROI
    MatBinROI= np.zeros((Xpix_1b.shape[0], int(np.max(nbrROILines))))

    for ipix in range(iLineTopStart.shape[0]):
        MatBinROI[ipix, int(iStartLineROI[ipix] - np.min(iStartLineROI)):int(iEndLineROI[ipix] - np.min(iStartLineROI))+1]= 1

    return ROI_ok, iStartLineROI, iEndLineROI, nbrROILines, nbrSmearingLines_TopROI, MatBinROI


"""



function [MatrixIllum,MatrixROI] = Create_MatrixIllum(Obs_Letter,Instrument,DateYMD,X,iLineBottomStart_NonIllum,iLineTopStart_NonIllum,
                                                      iStartLineCCD,iEndLineCCD,nbrCCDLines,iStartLineBin,iEndLineBin,nbrLambdaBinned)


    if ( strcmp(Obs_Letter,'N') || strcmp(Obs_Letter,'O') )

        MatrixIllum = zeros(size(X,1),nbrCCDLines,2);
        for iLambda = 1:size(X,1)
            MatrixIllum(iLambda,iStartLineBin(iLambda):iEndLineBin(iLambda),1) = 1;
            MatrixIllum(iLambda,1:iLineBottomStart_NonIllum(iLambda),2) = 1;
            MatrixIllum(iLambda,iLineTopStart_NonIllum(iLambda):end,2) = 1;
        end

    else

        MatrixIllum = zeros(size(X,1),nbrCCDLines,3);
        for iLambda = 1:size(X,1)
            MatrixIllum(iLambda,iStartLineBin(iLambda):iEndLineBin(iLambda),1) = 1;
            MatrixIllum(iLambda,1:iLineBottomStart_NonIllum(iLambda),2) = 1;
            MatrixIllum(iLambda,iLineTopStart_NonIllum(iLambda):end,3) = 1;
        end

    end

    MatrixROI = zeros(size(MatrixIllum(:,:,1)));
    for iPix = 1:size(MatrixROI,1)
        MatrixROI(iPix,iStartLineBin(iPix):iEndLineBin(iPix)) = 1;
    end


    MatrixIllum = permute(MatrixIllum,[2 1 3]);
    MatrixROI = MatrixROI';

end % function

"""


def Create_MatrixIllum(Obs_Letter, Instrument, DateYMD, X, iLineBottomStart_NonIllum, iLineTopStart_NonIllum,
                       iStartLineCCD, iEndLineCCD, nbrCCDLines, iStartLineBin, iEndLineBin, nbrLambdaBinned):

    if Obs_Letter == 'N' or Obs_Letter == 'O':
        # Initialize the illumination matrix with zeros for 2nd dimension
        MatrixIllum= np.zeros((X.shape[0], nbrCCDLines, 2))
        for iLambda in range(X.shape[0]):
            MatrixIllum[iLambda, iStartLineBin[iLambda]:iEndLineBin[iLambda], 0]= 1
            MatrixIllum[iLambda, 0:iLineBottomStart_NonIllum[iLambda], 1]= 1
            MatrixIllum[iLambda, iLineTopStart_NonIllum[iLambda]:, 1]= 1
    else:
        # Initialize the illumination matrix with zeros for 3rd dimension
        MatrixIllum= np.zeros((X.shape[0], nbrCCDLines, 3))
        for iLambda in range(X.shape[0]):
            MatrixIllum[iLambda, iStartLineBin[iLambda]:iEndLineBin[iLambda], 0]= 1
            MatrixIllum[iLambda, 0:iLineBottomStart_NonIllum[iLambda], 1]= 1
            MatrixIllum[iLambda, iLineTopStart_NonIllum[iLambda]:, 2]= 1

    # Initialize the ROI matrix with zeros
    MatrixROI= np.zeros(MatrixIllum[:, :, 0].shape)
    for iPix in range(MatrixROI.shape[0]):
        MatrixROI[iPix, iStartLineBin[iPix]:iEndLineBin[iPix]]= 1

    # Permute MatrixIllum (swap the first two axes) and transpose MatrixROI
    # MatrixIllum= np.transpose(MatrixIllum, (1, 0, 2))
    # MatrixROI= np.transpose(MatrixROI)

    return MatrixIllum, MatrixROI


"""

function [MultFact_Std,MultFact_Std_Fin] = choose_MultFact_Std(Instrument,Obs_Letter,iIllum,DateYMD,YesSignal,nbrLambdaBinned)

    if (strcmp(Instrument,'Spare') | (DateYMD < 20160314)) % Lab Measurement

            MultFact_Std = [3];
            MultFact_Std_Fin = [3;4;5];

    else % in flight Measurement
        % ======================================================================================================
        if ( strcmp(Obs_Letter,'D') || (strcmp(Obs_Letter,'L') && YesSignal) )
        % 3eme Dimension MatrixIllum = 5 => 1 Illum, 2&3 non illum (bottom & top) | 4&5 partly illum (bottom & top)

            % ----------------------------------------
            if nbrLambdaBinned <= 2

                if (iIllum == 1)
                    MultFact_Std = [2 2.5];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum == 2)
                    MultFact_Std = [2 2 2.5];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum == 3)
                    MultFact_Std = [2 3];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum >= 4)
                    MultFact_Std = [2];
                    MultFact_Std_Fin = [2.5;3;4];
                end

            % ----------------------------------------
            elseif nbrLambdaBinned <= 4

                if (iIllum == 1)
                    MultFact_Std = [2.5 3];
                    MultFact_Std_Fin = [3;4;6];
                elseif (iIllum == 2)
                    MultFact_Std = [2 2.5];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum == 3)
                    MultFact_Std = [2];
                    MultFact_Std_Fin = [3;4;5];
                elseif (iIllum == 4)
                    MultFact_Std = [2.5];
                    MultFact_Std_Fin = [3;4;5];
                elseif (iIllum == 5)
                    MultFact_Std = [3];
                    MultFact_Std_Fin = [3;4;5];
                end

            % ----------------------------------------
            else
                if (iIllum == 1)
                    MultFact_Std = [3];
                    MultFact_Std_Fin = [3;4;6];
                elseif (iIllum <= 3)
                    MultFact_Std = [3];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum >= 4)
                    MultFact_Std = [3];
                    MultFact_Std_Fin = [3;4;6];
                end
            end


        % ======================================================================================================
        elseif ( strcmp(Obs_Letter,'N') || (strcmp(Obs_Letter,'L') && not(YesSignal)) || strcmp(Obs_Letter,'O') )
        % 3eme Dimension MatrixIllum = 2 ou 4 => 1 Illum | 2 non illum (bottom + top) | 3&4 partly illum (bottom & top)

            % ----------------------------------------
            if nbrLambdaBinned <= 2

                if (iIllum == 1)
                    MultFact_Std = [2 2 3];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum == 2)
                    MultFact_Std = [2 2 2.5];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum >= 3)
                    MultFact_Std = [2];
                    MultFact_Std_Fin = [2.5;3;4];
                end


            % ----------------------------------------
            elseif nbrLambdaBinned <= 4

                if (iIllum == 1)
                    MultFact_Std = [2 2];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum == 2)
                    MultFact_Std = [2 2];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum >= 3)
                    MultFact_Std = [2.5 3];
                    MultFact_Std_Fin = [3;4;5];
                end

            % ----------------------------------------
            else
                if (iIllum == 1)
                    MultFact_Std = [2];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum == 2)
                    MultFact_Std = [2];
                    MultFact_Std_Fin = [2;3;4];
                elseif (iIllum >= 3)
                    MultFact_Std = [2.5];
                    MultFact_Std_Fin = [3;4;5];
                end
            end


        % ======================================================================================================
        elseif ( strcmp(Obs_Letter,'I') || strcmp(Obs_Letter,'E') || strcmp(Obs_Letter,'G') )

            if nbrLambdaBinned <= 2
                MultFact_Std = [3];
                MultFact_Std_Fin = [3;4;5];
            elseif nbrLambdaBinned <= 4
                MultFact_Std = [3.5];
                MultFact_Std_Fin = [3;4;5];
            else
                MultFact_Std = [4];
                MultFact_Std_Fin = [3;4;5];
            end

            % if nbrLambdaBinned <= 2
                % MultFact_Std = [2.5];
                % MultFact_Std_Fin = [2.5;3.5;4.5];
            % else
                % MultFact_Std = [3];
                % MultFact_Std_Fin = [3;4;5];
            % end

        end
    end

end

"""


def choose_MultFact_Std(Instrument, Obs_Letter, iIllum, DateYMD, YesSignal, nbrLambdaBinned):
    if (Instrument == 'Spare' or int(DateYMD) < 20160314):  # Lab Measurement
        MultFact_Std= [3]
        MultFact_Std_Fin= [3, 4, 5]

    else:  # in flight Measurement
        # ======================================================================================================
        if Obs_Letter == 'D' or (Obs_Letter == 'L' and YesSignal):
            # 3rd Dimension MatrixIllum = 5 => 1 Illum, 2&3 non illum (bottom & top) | 4&5 partly illum (bottom & top)

            if nbrLambdaBinned <= 2:
                if iIllum == 1:
                    MultFact_Std= [2, 2.5]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum == 2:
                    MultFact_Std= [2, 2, 2.5]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum == 3:
                    MultFact_Std= [2, 3]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum >= 4:
                    MultFact_Std= [2]
                    MultFact_Std_Fin= [2.5, 3, 4]

            elif nbrLambdaBinned <= 4:
                if iIllum == 1:
                    MultFact_Std= [2.5, 3]
                    MultFact_Std_Fin= [3, 4, 6]
                elif iIllum == 2:
                    MultFact_Std= [2, 2.5]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum == 3:
                    MultFact_Std= [2]
                    MultFact_Std_Fin= [3, 4, 5]
                elif iIllum == 4:
                    MultFact_Std= [2.5]
                    MultFact_Std_Fin= [3, 4, 5]
                elif iIllum == 5:
                    MultFact_Std= [3]
                    MultFact_Std_Fin= [3, 4, 5]

            else:
                if iIllum == 1:
                    MultFact_Std= [3]
                    MultFact_Std_Fin= [3, 4, 6]
                elif iIllum <= 3:
                    MultFact_Std= [3]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum >= 4:
                    MultFact_Std= [3]
                    MultFact_Std_Fin= [3, 4, 6]

        # ======================================================================================================
        elif Obs_Letter == 'N' or (Obs_Letter == 'L' and not YesSignal) or Obs_Letter == 'O':
            # 3rd Dimension MatrixIllum = 2 or 4 => 1 Illum | 2 non illum (bottom + top) | 3&4 partly illum (bottom & top)

            if nbrLambdaBinned <= 2:
                if iIllum == 1:
                    MultFact_Std= [2, 2, 3]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum == 2:
                    MultFact_Std= [2, 2, 2.5]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum >= 3:
                    MultFact_Std= [2]
                    MultFact_Std_Fin= [2.5, 3, 4]

            elif nbrLambdaBinned <= 4:
                if iIllum == 1:
                    MultFact_Std= [2, 2]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum == 2:
                    MultFact_Std= [2, 2]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum >= 3:
                    MultFact_Std= [2.5, 3]
                    MultFact_Std_Fin= [3, 4, 5]

            else:
                if iIllum == 1:
                    MultFact_Std= [2]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum == 2:
                    MultFact_Std= [2]
                    MultFact_Std_Fin= [2, 3, 4]
                elif iIllum >= 3:
                    MultFact_Std= [2.5]
                    MultFact_Std_Fin= [3, 4, 5]

        # ======================================================================================================
        elif Obs_Letter == 'I' or Obs_Letter == 'E' or Obs_Letter == 'G':
            if nbrLambdaBinned <= 2:
                MultFact_Std= [3]
                MultFact_Std_Fin= [3, 4, 5]
            elif nbrLambdaBinned <= 4:
                MultFact_Std= [3.5]
                MultFact_Std_Fin= [3, 4, 5]
            else:
                MultFact_Std= [4]
                MultFact_Std_Fin= [3, 4, 5]

    return np.asarray(MultFact_Std, dtype=float), np.asarray(MultFact_Std_Fin, dtype=float)


"""

function [MatrixAnomaly_full] = CosmicRayDetection2(spectra_full,YMask_full,MaskValueSaturation,nbrLambdaBinned,MatrixIllum_full,MatrixROI_full,iObs,Instrument,Obs_Letter,DateYMD)

    doPlot = 0;

    MatrixAnomaly_full = zeros(size(spectra_full));

    spectra = spectra_full;
    YMask = YMask_full;
    MatrixIllum = MatrixIllum_full;
    MatrixROI = MatrixROI_full;

    nbrPix = size(spectra,2);

    % ------------------------------------
    MeanValPixels = mean(spectra,2);
    MeanValPixels(MeanValPixels<0) = 0;

    MatrixROINaN = MatrixROI./MatrixROI;
    MeanValLinesROI = mean(spectra.*MatrixROINaN,'omitnan');
    StdValLinesROI = std(spectra.*MatrixROINaN,'omitnan');
    MaskNoExtremes = (abs(spectra - MeanValLinesROI) < 2*StdValLinesROI);
    MeanValLinesROI = sum(spectra .* MaskNoExtremes .* MatrixROI) ./ sum(MaskNoExtremes .* MatrixROI);
    MeanValLinesROI(MeanValLinesROI<0) = 0;

    ValMatUsedForRatio = (sqrt(MeanValPixels * MeanValLinesROI) .* MatrixROI);
    temp_MatValDefined = logical(MatrixROI);

    if ( (size(MatrixIllum,2) >= 2) && (sum(sum(MatrixIllum(:,:,2))) > 0.5) )
        MatrixBottomNaN = MatrixIllum(:,:,2) ./ MatrixIllum(:,:,2);
        MeanValLinesBottom = mean(spectra.*MatrixBottomNaN,'omitnan');
        StdValLinesBottom = std(spectra.*MatrixBottomNaN,'omitnan');
        MaskNoExtremes = (abs(spectra - MeanValLinesBottom) < 2*StdValLinesBottom);
        MeanValLinesBottom = sum(spectra .* MaskNoExtremes .* MatrixIllum(:,:,2)) ./ sum(MaskNoExtremes .* MatrixIllum(:,:,2));
        MeanValLinesBottom(MeanValLinesBottom<0) = 0;

        ValMatUsedForRatio = ValMatUsedForRatio + (sqrt(MeanValPixels * MeanValLinesBottom) .* MatrixIllum(:,:,2));
        temp_MatValDefined = logical(MatrixROI + MatrixIllum(:,:,2));
    end

    if ( (size(MatrixIllum,3) >= 3) && (sum(sum(MatrixIllum(:,:,3))) > 0.5) )
        MatrixTopNaN = MatrixIllum(:,:,3) ./ MatrixIllum(:,:,3);
        MeanValLinesTop = mean(spectra./MatrixTopNaN,'omitnan');
        StdValLinesTop = std(spectra./MatrixTopNaN,'omitnan');
        MaskNoExtremes = (abs(spectra - MeanValLinesTop) < 2*StdValLinesTop);
        MeanValLinesTop = sum(spectra .* MaskNoExtremes .* MatrixIllum(:,:,3)) ./ sum(MaskNoExtremes .* MatrixIllum(:,:,3));
        MeanValLinesTop(MeanValLinesTop<0) = 0;

        ValMatUsedForRatio = ValMatUsedForRatio + (sqrt(MeanValPixels * MeanValLinesTop) .* MatrixIllum(:,:,3));
        temp_MatValDefined = logical(MatrixROI + MatrixIllum(:,:,2) + MatrixIllum(:,:,3));
    end

    minValueUsedForRatio = 200;

    if ( sum(sum(temp_MatValDefined==0)) > 0.5 )
        numLine = [1:size(ValMatUsedForRatio,1)]';
        for iLambda = 1:nbrPix
            ValMatUsedForRatio(:,iLambda) = interp1(numLine(temp_MatValDefined(:,iLambda)),ValMatUsedForRatio(temp_MatValDefined(:,iLambda),iLambda),numLine,'linear','extrap');
        end
    end

    ValMatUsedForRatio(ValMatUsedForRatio<minValueUsedForRatio) = minValueUsedForRatio;

    % calculate ratio on adjacent pixels
    Ratio_pre_ini = zeros(size(spectra,1),size(spectra,2));
    Ratio_post_ini = zeros(size(spectra,1),size(spectra,2));
    Ratio_pre_ini(:,2:end) = (spectra(:,2:end) - spectra(:,1:end-1)) ./ ValMatUsedForRatio(:,2:end);
    Ratio_post_ini(:,1:end-1) = (spectra(:,2:end) - spectra(:,1:end-1)) ./ ValMatUsedForRatio(:,1:end-1);
    %
    if nbrLambdaBinned < 1.5
        nextPixel = 3;
    else
        nextPixel = 2;
    end
    Ratio_pre3_ini = zeros(size(spectra,1),size(spectra,2));
    Ratio_post3_ini = zeros(size(spectra,1),size(spectra,2));
    Ratio_pre3_ini(:,1+nextPixel:end) = (spectra(:,1+nextPixel:end) - spectra(:,1:end-nextPixel)) ./ ValMatUsedForRatio(:,1+nextPixel:end);
    Ratio_post3_ini(:,1:end-nextPixel) = (spectra(:,1+nextPixel:end) - spectra(:,1:end-nextPixel)) ./ ValMatUsedForRatio(:,1:end-nextPixel);

    MatDetection = logical(zeros(size(spectra,1),size(spectra,2)));
    if strcmp(Obs_Letter,'L')
        MeanValLines_Illum = sum(spectra .* MaskNoExtremes .* MatrixIllum(:,:,1)) ./ sum(MaskNoExtremes .* MatrixIllum(:,:,1));
        MaxMeanValLines_Illum = max(MeanValLines_Illum / nbrLambdaBinned);

        if MaxMeanValLines_Illum > 200
            YesSignal = 1;
        else
            YesSignal = 0;
            MatrixIllum(:,:,2) = MatrixIllum(:,:,2) + MatrixIllum(:,:,3);
            MatrixIllum(:,:,3) = [];
        end
    else
        YesSignal = [];
    end

    for iIllum = 1:size(MatrixIllum,3) % first illum part then non illum part

        if ( (iIllum < 4 && sum(sum(MatrixIllum(:,:,iIllum))) > 0.5) )

            Mat_Used = single(MatrixIllum(:,:,iIllum));
            Mat_Used_save = Mat_Used;

            [MultFact_Std,MultFact_Std_Fin] = choose_MultFact_Std(Instrument,Obs_Letter,iIllum,DateYMD,YesSignal,nbrLambdaBinned);

            Ratio_pre = Ratio_pre_ini;
            Ratio_post = Ratio_post_ini;
            Ratio_pre3 = Ratio_pre3_ini;
            Ratio_post3 = Ratio_post3_ini;

            if sum(sum(Mat_Used)) > 0.5

                Mat_NaN = Mat_Used ./ Mat_Used;

                Ratio_pre = Ratio_pre .* Mat_NaN;
                Ratio_post = Ratio_post .* Mat_NaN;
                Ratio_pre3 = Ratio_pre3 .* Mat_NaN;
                Ratio_post3 = Ratio_post3 .* Mat_NaN;
                spectra_Used = spectra .* Mat_NaN;

                %% to test
                if doPlot
                    if mod(iObs,30) == 14
                        iLambda = 1007;
                        LineToDisp = [find(sum(Mat_Used_save,2)>0.5)];
                        vectPix = [max(iLambda-1050,1):min(iLambda+1050,nbrPix)];
                        figure('WindowStyle','docked'); hold on;
                        plot(vectPix,spectra_Used(LineToDisp,vectPix));
                        titre=['Spectre initial: obs ',num2str(iObs,'%d'),' (iIllum ',num2str(iIllum,'%d'),', YesSignal ',num2str(YesSignal,'%d'),')'];
                        title(titre)
                    end % fin test
                end

                nbrLoop = size(MultFact_Std,2) + 1; % +1 pour que la derniere boucle soit celle avec le vecteur MultFact_Std_Fin
                for iLoop = 1:nbrLoop

                    if (iLoop > 1.5) % recalcul Ratio avec matExtreme
                        % Ratio_pre
                        clear vectLineToRep
                        vectLineToRep = find(sum(matExtreme_pre,2) > 0.5);
                        for icompt = 1:size(vectLineToRep,1)
                            Ratio_pre(vectLineToRep(icompt,1),matExtreme_pre(vectLineToRep(icompt,1),:)) = MeanRatio_pre(1,matExtreme_pre(vectLineToRep(icompt,1),:));
                        end
                        % Ratio_post
                        clear vectLineToRep
                        vectLineToRep = find(sum(matExtreme_post,2) > 0.5);
                        for icompt = 1:size(vectLineToRep,1)
                            Ratio_post(vectLineToRep(icompt,1),matExtreme_post(vectLineToRep(icompt,1),:)) = MeanRatio_post(1,matExtreme_post(vectLineToRep(icompt,1),:));
                        end

                        % Ratio_pre3
                        clear vectLineToRep
                        vectLineToRep = find(sum(matExtreme_pre3,2) > 0.5);
                        for icompt = 1:size(vectLineToRep,1)
                            Ratio_pre3(vectLineToRep(icompt,1),matExtreme_pre3(vectLineToRep(icompt,1),:)) = MeanRatio_pre3(1,matExtreme_pre3(vectLineToRep(icompt,1),:));
                        end
                        % Ratio_post3
                        clear vectLineToRep
                        vectLineToRep = find(sum(matExtreme_post3,2) > 0.5);
                        for icompt = 1:size(vectLineToRep,1)
                            Ratio_post3(vectLineToRep(icompt,1),matExtreme_post3(vectLineToRep(icompt,1),:)) = MeanRatio_post3(1,matExtreme_post3(vectLineToRep(icompt,1),:));
                        end
                    end

                    MeanRatio_pre = mean(Ratio_pre,'omitnan');
                    MeanRatio_post = mean(Ratio_post,'omitnan');
                    StdRatio_pre = std(Ratio_pre,'omitnan');
                    StdRatio_post = std(Ratio_post,'omitnan');

                    MeanRatio_pre3 = mean(Ratio_pre3,'omitnan');
                    MeanRatio_post3 = mean(Ratio_post3,'omitnan');
                    StdRatio_pre3 = std(Ratio_pre3,'omitnan');
                    StdRatio_post3 = std(Ratio_post3,'omitnan');

                    if (iLoop < nbrLoop)
                        matExtreme_pre = abs(Ratio_pre_ini - MeanRatio_pre) > MultFact_Std(1,iLoop)*StdRatio_pre;
                        matExtreme_post = abs(Ratio_post_ini - MeanRatio_post) > MultFact_Std(1,iLoop)*StdRatio_post;

                        matExtreme_pre(:,1) = 1;
                        matExtreme_post(:,end) = 1;

                        matExtreme_pre3 = abs(Ratio_pre3_ini - MeanRatio_pre3) > MultFact_Std(1,iLoop)*StdRatio_pre3;
                        matExtreme_post3 = abs(Ratio_post3_ini - MeanRatio_post3) > MultFact_Std(1,iLoop)*StdRatio_post3;

                        matExtreme_pre3(:,1:3) = 1;
                        matExtreme_post3(:,end-2:end) = 1;

                        tempSave_MatDetection(:,:,iLoop) = logical((matExtreme_pre + matExtreme_post + matExtreme_pre3 + matExtreme_post3) >= 3);
                    end

                    if (iLoop == nbrLoop)
                        matExtreme_pre = (abs(Ratio_pre_ini - MeanRatio_pre) > MultFact_Std_Fin(1)*StdRatio_pre) * 1; % =1 if > 2std
                        matExtreme_post = (abs(Ratio_post_ini - MeanRatio_post) > MultFact_Std_Fin(1)*StdRatio_post) * 1; % =1 if > 2std
                        matExtreme_pre = matExtreme_pre + (abs(Ratio_pre_ini - MeanRatio_pre) > MultFact_Std_Fin(2)*StdRatio_pre) * 1; % =2 if > 3std
                        matExtreme_post = matExtreme_post +  (abs(Ratio_post_ini - MeanRatio_post) > MultFact_Std_Fin(2)*StdRatio_post) * 1; % =2 if > 3std
                        matExtreme_pre = matExtreme_pre + (abs(Ratio_pre_ini - MeanRatio_pre) > MultFact_Std_Fin(3)*StdRatio_pre) * 1; % =3 if > 4std
                        matExtreme_post = matExtreme_post +  (abs(Ratio_post_ini - MeanRatio_post) > MultFact_Std_Fin(3)*StdRatio_post) * 1; % =3 if > 4std

                        % initiale values
                        matExtreme_pre(:,1) = 1;
                        matExtreme_post(:,end) = 1;

                        matExtreme_pre3 = (abs(Ratio_pre3_ini - MeanRatio_pre3) > MultFact_Std_Fin(1)*StdRatio_pre3) * 1; % =1 if > 2std
                        matExtreme_post3 = (abs(Ratio_post3_ini - MeanRatio_post3) > MultFact_Std_Fin(1)*StdRatio_post3) * 1; % =1 if > 2std
                        matExtreme_pre3 = matExtreme_pre3 + (abs(Ratio_pre3_ini - MeanRatio_pre3) > MultFact_Std_Fin(2)*StdRatio_pre3) * 1; % =2 if > 3std
                        matExtreme_post3 = matExtreme_post3 +  (abs(Ratio_post3_ini - MeanRatio_post3) > MultFact_Std_Fin(2)*StdRatio_post3) * 1; % =2 if > 3std
                        matExtreme_pre3 = matExtreme_pre3 + (abs(Ratio_pre3_ini - MeanRatio_pre3) > MultFact_Std_Fin(3)*StdRatio_pre3) * 1; % =3 if > 4std
                        matExtreme_post3 = matExtreme_post3 +  (abs(Ratio_post3_ini - MeanRatio_post3) > MultFact_Std_Fin(3)*StdRatio_post3) * 1; % =3 if > 4std

                        % initiale values
                        matExtreme_pre3(:,1:3) = 2;
                        matExtreme_post3(:,end-2:end) = 2;

                        tempSave_MatDetection(:,:,iLoop) = logical((matExtreme_pre + matExtreme_post + matExtreme_pre3 + matExtreme_post3) >= 8);
                    end

                end %iLoop

                % check there are enough lines to perform all to perform the number of loops. If not enough, reduce the number of loops
                nbrLinesInTheMean = sum(Mat_Used,1);
                nbrMaxLines = [12 20];
                nbrMaxLoops = [ 2  3];

                temp_MatDetection = tempSave_MatDetection(:,:,iLoop);

                if iIllum >=4
                    spectra_Used(not(Mat_Used)) = NaN;

                    MatDetection(logical(Mat_Used_save)) = logical(temp_MatDetection(logical(Mat_Used_save)));

                else
                    MatDetection(logical(Mat_Used_save)) = logical(temp_MatDetection(logical(Mat_Used_save)));
                end

                %% to test
                if doPlot
                    spectra_Used(not(Mat_Used)) = NaN;
                    spectra_Used(MatDetection) = NaN;
                    spectra_Disp = spectra_Used;
                    spectra_Disp(logical(YMask)) = NaN;
                    if mod(iObs,30) == 14
                        figure('WindowStyle','docked'); hold on;
                        plot(vectPix,spectra_Used(LineToDisp,vectPix));
                        titre=['Spectre final Anomalies removed: obs ',num2str(iObs,'%d'),' (iIllum ',num2str(iIllum,'%d'),', YesSignal ',num2str(YesSignal,'%d'),')'];
                        title(titre)
                        figure('WindowStyle','docked'); hold on;
                        plot(vectPix,spectra_Disp(LineToDisp,vectPix));
                        titre=['Spectre final All YMask removed: obs ',num2str(iObs,'%d'),' (iIllum ',num2str(iIllum,'%d'),', YesSignal ',num2str(YesSignal,'%d'),')'];
                        title(titre)
                    end % fin test
                end

            end % if sum(sum(Mat_NaN==1))

        end % iIllum

end % test MatIllum empty or zero

    MatrixAnomaly_full = MatDetection;

end % function

"""


def CosmicRayDetection2(spectra, YMask, MaskValueSaturation, nbrLambdaBinned, MatrixIllum_full, MatrixROI_full, iObs, Instrument, Obs_Letter, DateYMD):
    # dimensions are 1024 x 60 x N OR 128 x 71 x N
    # MatrixIllum and MatrixROI 1 indices are offset by 1
    
    # doPlot= 0
    printi = -1
    
    # spectra = spectra.T
    # MatrixROI_full = MatrixROI_full.T
    # MatrixIllum_full = np.swapaxes(MatrixIllum_full, 0, 1)

    if iObs == printi:
        print("1 spectra[0,0] =", spectra[0,0])
    

    # Initialize output array
    # MatrixAnomaly_full= np.zeros(spectra.shape)

    MatrixIllum= MatrixIllum_full.copy()
    MatrixROI= MatrixROI_full.copy()

    if iObs == printi:
        print("2 sum(MatrixIllum) =", np.sum(MatrixIllum))
    if iObs == printi:
        print("3 sum(MatrixROI) =", np.sum(MatrixROI))

    nbrPix= spectra.shape[0]
    # nbrCCDLines= spectra.shape[1]

    # ------------------------------------
    MeanValPixels= np.mean(spectra, axis=0)
    MeanValPixels[MeanValPixels < 0]= 0
    if iObs == printi:
        print("4 MeanValPixels[0,0] =", MeanValPixels[0])

    # python change 0 to nan
    MatrixROINaN= MatrixROI.copy()
    MatrixROINaN[MatrixROINaN == 0] = np.nan

    MeanValLinesROI= np.nanmean(spectra * MatrixROINaN, axis=1)
    StdValLinesROI= np.nanstd(spectra * MatrixROINaN, axis=1, ddof=1)
    MaskNoExtremes= np.abs(spectra - MeanValLinesROI[:, None]) < 2 * StdValLinesROI[:, None]

    if iObs == printi:
        print("5 MeanValLinesROI[0] =", MeanValLinesROI[0])
        print("6 StdValLinesROI[0] =", StdValLinesROI[0])
        print("7 sum(MaskNoExtremes) =", np.sum(MaskNoExtremes))

    MeanValLinesROI= np.sum(spectra * MaskNoExtremes * MatrixROI, axis=1) / np.sum(MaskNoExtremes * MatrixROI, axis=1)
    MeanValLinesROI[MeanValLinesROI < 0]= 0

    if iObs == printi:
        print("8 MeanValLinesROI[0] =", MeanValLinesROI[0])

    # size(MatrixROINaN) = [60 1024]
    # size(MeanValLinesROI) = [1 1024]
    # size(MaskNoExtremes) = [60 1024]
    # size(MeanValLinesROI) = [1 1024]
    
    # this is just about correct
    ValMatUsedForRatio= np.sqrt(MeanValPixels[None, :] * MeanValLinesROI[:, None]) * MatrixROI
    if iObs == printi:
        print("9 sum(ValMatUsedForRatio) =", np.sum(ValMatUsedForRatio))

    # size(ValMatUsedForRatio) = [60 1024]
    
    # print("spectra", spectra.shape)
    # print("MatrixROINaN", MatrixROINaN.shape)
    # print("MeanValLinesROI", MeanValLinesROI.shape)
    # print("MaskNoExtremes", MaskNoExtremes.shape)
    # print("MeanValLinesROI", MeanValLinesROI.shape)
    # print("ValMatUsedForRatio", ValMatUsedForRatio.shape)
    


    temp_MatValDefined= MatrixROI.astype(bool)

    if MatrixIllum.shape[2] >= 2 and np.sum(MatrixIllum[:, :, 1]) > 0.5:

        # python change 0 to nan
        MatrixBottomNaN= MatrixIllum[:, :, 1].copy()
        MatrixBottomNaN[MatrixBottomNaN == 0] = np.nan

        MeanValLinesBottom= np.nanmean(spectra * MatrixBottomNaN, axis=1)
        StdValLinesBottom= np.nanstd(spectra * MatrixBottomNaN, axis=1, ddof=1)
        MaskNoExtremes= np.abs(spectra - MeanValLinesBottom[:, None]) < 2 * StdValLinesBottom[:, None]
        if iObs == printi:
            print("10 MeanValLinesBottom[0] =", MeanValLinesBottom[0])
            print("11 StdValLinesBottom[0] =", StdValLinesBottom[0])
            print("12 sum(MaskNoExtremes) =", np.sum(MaskNoExtremes))

        MeanValLinesBottom= np.sum(spectra * MaskNoExtremes * MatrixIllum[:, :, 1], axis=1) / np.sum(MaskNoExtremes * MatrixIllum[:, :, 1], axis=1)
        MeanValLinesBottom[MeanValLinesBottom < 0]= 0

        if iObs == printi:
            print("13 MeanValLinesBottom[0] =", MeanValLinesBottom[0])

        ValMatUsedForRatio += np.sqrt(MeanValPixels[None, :] * MeanValLinesBottom[:, None]) * MatrixIllum[:, :, 1]
        temp_MatValDefined |= MatrixIllum[:, :, 1].astype(bool)

        if iObs == printi:
            print("14 sum(ValMatUsedForRatio) =", np.sum(ValMatUsedForRatio))


    if MatrixIllum.shape[2] >= 3 and np.sum(MatrixIllum[:, :, 2]) > 0.5:

        # python change 0 to nan
        MatrixTopNaN= MatrixIllum[:, :, 2].copy()
        MatrixTopNaN[MatrixTopNaN == 0] = np.nan

        MeanValLinesTop= np.nanmean(spectra * MatrixTopNaN, axis=1)
        StdValLinesTop= np.nanstd(spectra * MatrixTopNaN, axis=1, ddof=1)
        MaskNoExtremes= np.abs(spectra - MeanValLinesTop[:, None]) < 2 * StdValLinesTop[:, None]
        if iObs == printi:
            print("15 MeanValLinesTop[0] =", MeanValLinesTop[0])
            print("16 StdValLinesTop[0] =", StdValLinesTop[0])
            print("17 sum(MaskNoExtremes) =", np.sum(MaskNoExtremes))

        MeanValLinesTop= np.sum(spectra * MaskNoExtremes * MatrixIllum[:, :, 2], axis=1) / np.sum(MaskNoExtremes * MatrixIllum[:, :, 2], axis=1)
        MeanValLinesTop[MeanValLinesTop < 0]= 0

        if iObs == printi:
            print("18 MeanValLinesTop[0] =", MeanValLinesTop[0])

        ValMatUsedForRatio += np.sqrt(MeanValPixels[None, :] * MeanValLinesTop[:, None]) * MatrixIllum[:, :, 2]
        temp_MatValDefined |= MatrixIllum[:, :, 2].astype(bool)

        if iObs == printi:
            print("19 sum(ValMatUsedForRatio) =", np.sum(ValMatUsedForRatio))

    minValueUsedForRatio= 200

    if np.sum(temp_MatValDefined == 0) > 0.5:
        numLine= np.arange(1, spectra.shape[1] + 1)
        # size(ValMatUsedForRatio) = [60 1024]
        for iLambda in range(nbrPix):
            ValMatUsedForRatio[iLambda, :]= interpolate.interp1d(numLine[temp_MatValDefined[iLambda, :]], ValMatUsedForRatio[iLambda, temp_MatValDefined[iLambda, :]],
                                                                  kind='linear', fill_value='extrapolate')(numLine)
        if iObs == printi:
            print("20 sum(ValMatUsedForRatio) =", np.sum(ValMatUsedForRatio))

    ValMatUsedForRatio[ValMatUsedForRatio < minValueUsedForRatio]= minValueUsedForRatio
    if iObs == printi:
        print("21 sum(ValMatUsedForRatio) =", np.sum(ValMatUsedForRatio))


    # Calculate ratio on adjacent pixels
    # spectra is Npx x Nlines in python
    
    Ratio_pre_ini= np.zeros(spectra.shape)
    Ratio_post_ini= np.zeros(spectra.shape)
    Ratio_pre_ini[1:, :]= (spectra[1:, :] - spectra[:-1, :]) / ValMatUsedForRatio[1:, :]
    Ratio_post_ini[:-1, :]= (spectra[1:, :] - spectra[:-1, :]) / ValMatUsedForRatio[:-1, :]

    if nbrLambdaBinned < 1.5:
        nextPixel= 3
    else:
        nextPixel= 2

    Ratio_pre3_ini= np.zeros(spectra.shape)
    Ratio_post3_ini= np.zeros(spectra.shape)
    Ratio_pre3_ini[nextPixel:, :]= (spectra[nextPixel:, :] - spectra[:-nextPixel, :]) / ValMatUsedForRatio[nextPixel:, :]
    Ratio_post3_ini[:-nextPixel, :]= (spectra[nextPixel:, :] -spectra[:-nextPixel, :]) / ValMatUsedForRatio[:-nextPixel, :]

    if iObs == printi:
        print("22 sum(Ratio_pre_ini) =", np.sum(Ratio_pre_ini))
        print("23 sum(Ratio_post_ini) =", np.sum(Ratio_post_ini))
        print("24 sum(Ratio_pre3_ini) =", np.sum(Ratio_pre3_ini))
        print("25 sum(Ratio_post3_ini) =", np.sum(Ratio_post3_ini))

    MatDetection= np.zeros(spectra.shape, dtype=bool)

    # if Obs_Letter == 'L':
    #     MeanValLines_Illum= np.sum(spectra * MaskNoExtremes * MatrixIllum[:, :, 0], axis=1) / np.sum(MaskNoExtremes * MatrixIllum[:, :, 0], axis=1)
    #     MaxMeanValLines_Illum= np.max(MeanValLines_Illum / nbrLambdaBinned)

    #     if MaxMeanValLines_Illum > 200:
    #         YesSignal= 1
    #     else:
    #         YesSignal= 0
    #         MatrixIllum[:, :, 1]= MatrixIllum[:, :, 1] + MatrixIllum[:, :, 2]
    #         MatrixIllum= MatrixIllum[:, :, :2]  # Remove the third channel
    # else:
    #     YesSignal= None
    
    YesSignal= None
    
    tempSave_MatDetection = {}

    # Loop over illumination types (MatrixIllum has 3 layers: illum, top non-illum, bottom non-illum)
    for iIllum in range(MatrixIllum.shape[2]):

        if iObs == printi:
            print("iIllum =", iIllum)

        # always true
        if np.sum(MatrixIllum[:, :, iIllum]) > 0.5: # remove Illum>3 (never occurs)
            Mat_Used= np.float32(MatrixIllum[:, :, iIllum])
            Mat_Used_save= Mat_Used.copy()

            # Function call for MultFact_Std, MultFact_Std_Fin (not provided in your code snippet)
            MultFact_Std, MultFact_Std_Fin= choose_MultFact_Std(
                Instrument, Obs_Letter, iIllum, DateYMD, YesSignal, nbrLambdaBinned)

            # Initialize ratios
            Ratio_pre= Ratio_pre_ini.copy()
            Ratio_post= Ratio_post_ini.copy()
            Ratio_pre3= Ratio_pre3_ini.copy()
            Ratio_post3= Ratio_post3_ini.copy()

            if np.sum(Mat_Used) > 0.5:
                
                if iObs == printi:
                    print("np.sum(Mat_Used) > 0.5")

                
                # python change 0 to nan
                Mat_NaN= Mat_Used.copy()
                Mat_NaN[Mat_NaN == 0] = np.nan

                Ratio_pre *= Mat_NaN
                Ratio_post *= Mat_NaN
                Ratio_pre3 *= Mat_NaN
                Ratio_post3 *= Mat_NaN
                # spectra_Used= spectra * Mat_NaN
                
                if iObs == printi:
                    print("26 sum(Ratio_pre) =", np.nansum(Ratio_pre))
                    print("27 sum(Ratio_post) =", np.nansum(Ratio_post))
                    print("28 sum(Ratio_pre3) =", np.nansum(Ratio_pre3))
                    print("29 sum(Ratio_post3) =", np.nansum(Ratio_post3))

                # Plotting the spectra if necessary (debugging)
                # if doPlot:
                #     if iObs % 30 == 14:
                #         iLambda= 1007
                #         LineToDisp= np.where(
                #             np.sum(Mat_Used_save, axis=1) > 0.5)[0]
                #         vectPix= np.arange(
                #             max(iLambda - 1050, 0), min(iLambda + 1050, nbrPix))
                #         # Assuming you have matplotlib to display the plot
                #         import matplotlib.pyplot as plt
                #         plt.figure(figsize=(10, 5))
                #         plt.plot(vectPix, spectra_Used[LineToDisp, vectPix])
                #         plt.title(
                #             f"Spectre initial: obs {iObs} (iIllum {iIllum}, YesSignal {YesSignal})")
                #         plt.show()

                # Loop for anomaly detection
                nbrLoop= MultFact_Std.shape[0] +1  # Add 1 for MultFact_Std_Fin??

                matExtreme_pre= None  # to define first time round
                matExtreme_post= None  # to define first time round
                MeanRatio_pre= None  # to define first time round
                MeanRatio_post= None  # to define first time round
                MeanRatio_pre3= None  # to define first time round
                MeanRatio_post3= None  # to define first time round
                matExtreme_pre3= None  # to define first time round
                matExtreme_post3= None  # to define first time round

                for iLoop in range(nbrLoop):
                    
                    if iObs == printi:
                        print("iLoop =", iLoop)
                    
                    if iLoop > 0.5:
                        
                        if iObs == printi:
                            print("iLoop > 0.5")

                        # Handle extreme value corrections
                        # if np.sum(matExtreme_pre, axis=1) > 0.5:
                        vectLineToRep= np.where(np.sum(matExtreme_pre, axis=0) > 0.5)[0]
                        if iObs == printi:
                            print("29a sum(vectLineToRep) =", np.sum(vectLineToRep+1))
                        for line in vectLineToRep:
                            Ratio_pre[matExtreme_pre[:, line].astype(bool), line]= MeanRatio_pre[matExtreme_pre[:, line].astype(bool)] # need to change to bool otherwise will just take 0th or 1st element

                        # if np.sum(matExtreme_post, axis=1) > 0.5:
                        vectLineToRep= np.where(np.sum(matExtreme_post, axis=0) > 0.5)[0]
                        if iObs == printi:
                            print("29b sum(vectLineToRep) =", np.sum(vectLineToRep+1))
                        for line in vectLineToRep:
                            Ratio_post[matExtreme_post[:, line].astype(bool), line]= MeanRatio_post[matExtreme_post[:, line].astype(bool)]

                        # if np.sum(matExtreme_pre3, axis=1) > 0.5:
                        vectLineToRep= np.where(np.sum(matExtreme_pre3, axis=0) > 0.5)[0]
                        if iObs == printi:
                            print("29c sum(vectLineToRep) =", np.sum(vectLineToRep+1))
                        for line in vectLineToRep:
                            Ratio_pre3[matExtreme_pre3[:, line].astype(bool), line]= MeanRatio_pre3[matExtreme_pre3[:, line].astype(bool)]

                        # if np.sum(matExtreme_post3, axis=1) > 0.5:
                        vectLineToRep= np.where(np.sum(matExtreme_post3, axis=0) > 0.5)[0]
                        if iObs == printi:
                            print("29d sum(vectLineToRep) =", np.sum(vectLineToRep+1))
                        for line in vectLineToRep:
                            Ratio_post3[matExtreme_post3[:, line].astype(bool), line]= MeanRatio_post3[matExtreme_post3[:, line].astype(bool)]

                    # Compute means and standard deviations
                    MeanRatio_pre= np.nanmean(Ratio_pre, axis=1)
                    MeanRatio_post= np.nanmean(Ratio_post, axis=1)
                    StdRatio_pre= np.nanstd(Ratio_pre, axis=1, ddof=1)
                    StdRatio_post= np.nanstd(Ratio_post, axis=1, ddof=1)

                    MeanRatio_pre3= np.nanmean(Ratio_pre3, axis=1)
                    MeanRatio_post3= np.nanmean(Ratio_post3, axis=1)
                    StdRatio_pre3= np.nanstd(Ratio_pre3, axis=1, ddof=1)
                    StdRatio_post3= np.nanstd(Ratio_post3, axis=1, ddof=1)
                    
                    if iObs == printi:
                        print("30 sum(MeanRatio_pre) =", np.nansum(MeanRatio_pre))
                        print("31 sum(MeanRatio_post) =", np.nansum(MeanRatio_post))
                        print("32 sum(StdRatio_pre) =", np.nansum(StdRatio_pre))
                        print("33 sum(StdRatio_post) =", np.nansum(StdRatio_post))
    
                        print("34 sum(MeanRatio_pre3) =", np.nansum(MeanRatio_pre3))
                        print("35 sum(MeanRatio_post3) =", np.nansum(MeanRatio_post3))
                        print("36 sum(StdRatio_pre3) =", np.nansum(StdRatio_pre3))
                        print("37 sum(StdRatio_post3) =", np.nansum(StdRatio_post3))
                        
                    
                    # python - need to expand dimensions
                    # MeanRatio_pre= np.repeat(MeanRatio_pre[:, np.newaxis], nbrCCDLines, axis=1)
                    # MeanRatio_post= np.repeat(MeanRatio_post[:, np.newaxis], nbrCCDLines, axis=1)
                    # StdRatio_pre= np.repeat(StdRatio_pre[:, np.newaxis], nbrCCDLines, axis=1)
                    # StdRatio_post= np.repeat(StdRatio_post[:, np.newaxis], nbrCCDLines, axis=1)

                    # MeanRatio_pre3= np.repeat(MeanRatio_pre3[:, np.newaxis], nbrCCDLines, axis=1)
                    # MeanRatio_post3= np.repeat(MeanRatio_post3[:, np.newaxis], nbrCCDLines, axis=1)
                    # StdRatio_pre3= np.repeat(StdRatio_pre3[:, np.newaxis], nbrCCDLines, axis=1)
                    # StdRatio_post3= np.repeat(StdRatio_post3[:, np.newaxis], nbrCCDLines, axis=1)
                    

                    if iLoop < nbrLoop - 1:

                        if iObs == printi:
                            print("iLoop < nbrLoop - 1")

                        # for a in [Ratio_pre_ini, MeanRatio_pre, MultFact_Std[iLoop], StdRatio_pre]:
                        #     print(iLoop, a.shape)
                        matExtreme_pre= (np.abs(Ratio_pre_ini - MeanRatio_pre[:, None]) > MultFact_Std[iLoop] * StdRatio_pre[:, None]).astype(int)
                        matExtreme_post= (np.abs(Ratio_post_ini - MeanRatio_post[:, None]) > MultFact_Std[iLoop] * StdRatio_post[:, None]).astype(int)

                        matExtreme_pre[0, :]= 1
                        matExtreme_post[-1, :]= 1

                        matExtreme_pre3= (np.abs(Ratio_pre3_ini - MeanRatio_pre3[:, None]) > MultFact_Std[iLoop] * StdRatio_pre3[:, None]).astype(int)
                        matExtreme_post3= (np.abs( Ratio_post3_ini - MeanRatio_post3[:, None]) > MultFact_Std[iLoop] * StdRatio_post3[:, None]).astype(int)

                        matExtreme_pre3[:3, :]= 1
                        matExtreme_post3[-3:, :]= 1


                        # tempSave_MatDetection is a mess - sometimes it gets overwritten in the same iLoop, sometimes it doesn't, sometimes nothing is written
                        tempSave_MatDetection[iLoop] = matExtreme_pre*1 + matExtreme_post*1 + matExtreme_pre3*1 + matExtreme_post3*1 >= 3
                        if iObs == printi:
                            print("38 sum(tempSave_MatDetection) =", np.sum(np.asarray([v for k,v in tempSave_MatDetection.items()])))
                        

                    if iLoop == nbrLoop-1:
                        
                        if iObs == printi:
                            print("iLoop == nbrLoop-1")

                        # Apply final threshold conditions
                        matExtreme_pre= (np.abs(Ratio_pre_ini - MeanRatio_pre[:, None]) > MultFact_Std_Fin[0] * StdRatio_pre[:, None]).astype(int)
                        matExtreme_post= (np.abs(Ratio_post_ini - MeanRatio_post[:, None]) > MultFact_Std_Fin[0] * StdRatio_post[:, None]).astype(int)

                        matExtreme_pre += (np.abs(Ratio_pre_ini - MeanRatio_pre[:, None]) > MultFact_Std_Fin[1] * StdRatio_pre[:, None]).astype(int)
                        matExtreme_post += (np.abs(Ratio_post_ini - MeanRatio_post[:, None]) > MultFact_Std_Fin[1] * StdRatio_post[:, None]).astype(int)

                        matExtreme_pre += (np.abs(Ratio_pre_ini - MeanRatio_pre[:, None]) > MultFact_Std_Fin[2] * StdRatio_pre[:, None]).astype(int)
                        matExtreme_post += (np.abs(Ratio_post_ini - MeanRatio_post[:, None]) > MultFact_Std_Fin[2] * StdRatio_post[:, None]).astype(int)

                        matExtreme_pre[0, :]= 1
                        matExtreme_post[-1, :]= 1

                        matExtreme_pre3= (np.abs(Ratio_pre3_ini - MeanRatio_pre3[:, None]) > MultFact_Std_Fin[0] * StdRatio_pre3[:, None]).astype(int)
                        matExtreme_post3= (np.abs(Ratio_post3_ini - MeanRatio_post3[:, None]) > MultFact_Std_Fin[0] * StdRatio_post3[:, None]).astype(int)

                        matExtreme_pre3 += (np.abs(Ratio_pre3_ini - MeanRatio_pre3[:, None]) > MultFact_Std_Fin[1] * StdRatio_pre3[:, None]).astype(int)
                        matExtreme_post3 += (np.abs(Ratio_post3_ini - MeanRatio_post3[:, None]) > MultFact_Std_Fin[1] * StdRatio_post3[:, None]).astype(int)

                        matExtreme_pre3 += (np.abs(Ratio_pre3_ini - MeanRatio_pre3[:, None]) > MultFact_Std_Fin[2] * StdRatio_pre3[:, None]).astype(int)
                        matExtreme_post3 += (np.abs(Ratio_post3_ini - MeanRatio_post3[:, None]) > MultFact_Std_Fin[2] * StdRatio_post3[:, None]).astype(int)

                        matExtreme_pre3[:3, :]= 2
                        matExtreme_post3[-3:, :]= 2

                        tempSave_MatDetection[iLoop] = matExtreme_pre*1 + matExtreme_post*1 + matExtreme_pre3*1 + matExtreme_post3*1 >= 8
                        if iObs == printi:
                            print("39 sum(tempSave_MatDetection) =", np.sum(np.asarray([v for k,v in tempSave_MatDetection.items()])))

                # tempSave_MatDetection is a mess - sometimes it gets overwritten in the same iLoop, sometimes it doesn't, sometimes nothing is written
                # just take the last loop?
                temp_MatDetection= tempSave_MatDetection[iLoop].copy()  # changed
                if iObs == printi:
                    print("40 sum(temp_MatDetection) =", np.sum(temp_MatDetection))

                # if iIllum >= 3: # doesn't happen
                #     spectra_Used[~Mat_Used]= np.nan

                MatDetection[Mat_Used_save > 0]= temp_MatDetection[Mat_Used_save > 0]
                if iObs == printi:
                    print("41 sum(Mat_Used_save) =", np.sum(Mat_Used_save))
                    print("42 sum(MatDetection) =", np.sum(MatDetection))

                # Debug plot
                # if doPlot:
                #     spectra_Used[~Mat_Used]= np.nan
                #     spectra_Used[MatDetection]= np.nan
                #     spectra_Disp= spectra_Used.copy()
                #     spectra_Disp[YMask_full]= np.nan
                #     if iObs % 30 == 14:
                #         plt.figure(figsize=(10, 5))
                #         plt.plot(vectPix, spectra_Used[LineToDisp, vectPix])
                #         plt.title(f"Spectre final Anomalies removed: obs {iObs} (iIllum {iIllum}, YesSignal {YesSignal})")
                #         plt.show()
    MatrixAnomaly_full = MatDetection
    if iObs == printi:
        print("43 sum(MatrixAnomaly_full) =", np.sum(MatrixAnomaly_full))

    return MatrixAnomaly_full


"""
function [ MaskValues,Crit_Mask ] = Define_MaskValues()

  MaskValues.NonLinCorr = 1;

  MaskValues.HotPix = 10;
  MaskValues.Saturation = 20;
  MaskValues.Anomaly = 30;

  MaskValues.L02A_NaN = 90;
  MaskValues.L02C_NaN = 80;


  % for SO measurements only (used to keep the same pixels and lines in all the measurements for binning in SO, in order to properly divide all spectra by the reference one usinfg the same YMask)
  MaskValues.NotUsedPixSO = 50;
  MaskValues.NotUsedLineSO = 60;
  MaskValues.NotUsedLambdaSO = 70;

  MaskValues.NotInBinROI = 100;


  % Criteria used to select which pixels are kept/discarded for the vertical binning
  Crit_Mask.GoodPixelsKept = MaskValues.NonLinCorr + 0.5;
  Crit_Mask.HotPixelsKept = MaskValues.HotPix + MaskValues.NonLinCorr + 0.5;

end

"""


def define_mask_values():
    MaskValues= {
        "NonLinCorr": 1,

        "HotPix": 10,
        "Saturation": 20,
        "Anomaly": 30,

        "L02A_NaN": 90,
        "L02C_NaN": 80,

        # For SO measurements only
        "NotUsedPixSO": 50,
        "NotUsedLineSO": 60,
        "NotUsedLambdaSO": 70,

        "NotInBinROI": 100,
    }

    CritMask= {
        "GoodPixelsKept": MaskValues["NonLinCorr"] + 0.5,
        "HotPixelsKept": MaskValues["HotPix"] + MaskValues["NonLinCorr"] + 0.5,
    }
    return MaskValues, CritMask


"""

function AnomalyDetection(src,dst,MatrixCosmicRayDark,MatrixHotPixels,nbrUsefulPix,iStartUsefulPix,iEndUsefulPix,Instrument,Obs_Letter,DateYMD)

  % Mask values
  doPlot = 0;
  MaskValues = Define_MaskValues;

  % load data
    ibinning = h5read(src, '/Channel/AcquisitionMode', [1], [1]); % 0=not binned 1=vertical binned
    itypedata = h5read(src,'/Channel/ReverseFlagAndDataTypeFlagRegister'); % 0=dark 1=dark with reverse clock 2=bias 4=science
    nb_obs = double(h5readatt(src, '/', 'NSpec'));
    iStartLineCCD = double(h5read(src, '/Channel/VStart', [1], [1])) + 1;
    iEndLineCCD = double(h5read(src, '/Channel/VEnd', [1], [1])) + 1;
    nbrCCDLines = iEndLineCCD - iStartLineCCD + 1;
    nbrLambdaBinned = double(h5read(src, '/Channel/HorizontalAndCombinedBinningSize', [1], [1]) + 1);
    imode = h5read(src, '/Channel/Mode', [1], [1]); % 1=SO, 2=Nadir

    Xpix_1b = h5read(dst, '/Science/Xpix_1b');

    Instrument = h5readatt(src,'/','InstName');
    if strcmp(Instrument,'FlightSpare NOMAD - Nadir and Occultation for MArs Discovery')
        Instrument=[];Instrument='Spare';
    elseif strcmp(Instrument,'NOMAD - Nadir and Occultation for MArs Discovery')
        Instrument=[];Instrument='Flight';
    end


    if (ibinning == 0 || ibinning == 2) % FULLFRAME
      %% BB: I modified this part to read 1048 x 1 x nb_obs chunks (line by line)
      %% from the dataset and to write to the output in the same fashion,
      %% so the memory footprint is reduced without touching the underlying
      %% code to look for anomalies. I did not modify the way the matrices
      %% are indexed, but there are a lot of superfluous permutes,
      %% which could be eliminated.

        % identify science measurements
        vectSci = (itypedata == 4); % Identify the science spectra (no dark, ...)
        vectDark = ((itypedata==0)|(itypedata==1)); % Identify the science spectra (no dark, ...)

        % init
        MatrixAnomaly = zeros(nbrUsefulPix,nbrCCDLines,nb_obs);

        % dark
        MatrixAnomaly(:,:,vectDark) = MatrixCosmicRayDark(:,:,:);

        [iLineTopStart256_NonIllum,iLineBottomStart256_NonIllum] = Define_Top_Bottom_NonIlluminated_Lines(imode,Xpix_1b,Instrument);
        iLineBottomStart_NonIllum = iLineBottomStart256_NonIllum - iStartLineCCD + 1;
        iLineBottomStart_NonIllum(iLineBottomStart_NonIllum <= 0) = 0;
        iLineTopStart_NonIllum = iLineTopStart256_NonIllum - iStartLineCCD + 1;
        iLineTopStart_NonIllum(iLineTopStart_NonIllum > nbrCCDLines) = nbrCCDLines + 1;

        [ ROI_ok,iStartLineBin,iEndLineBin,nbrBinLines,nbrSmearingLines,unused ] = Define_ROI( Instrument,imode,ibinning,iStartLineCCD,iEndLineCCD,Xpix_1b );

        [MatrixIllum,MatrixROI] = Create_MatrixIllum(Obs_Letter,Instrument,DateYMD,Xpix_1b,iLineBottomStart_NonIllum,iLineTopStart_NonIllum,
                                                     iStartLineCCD,iEndLineCCD,nbrCCDLines,iStartLineBin,iEndLineBin,nbrLambdaBinned);

        % Loop on Observations (use this one)
        for iObs = 1:nb_obs

            % !!! Read the Y dataset from the output file since AnomalyDetection is run after RemoveDC !!!
            data = h5read(dst, '/Science/Y', [1 1 iObs], [Inf Inf 1]);

            % Read the Y mask from the source
            YMask = h5read(dst, '/Science/YMask', [1 1 iObs], [Inf Inf 1]);

            if vectSci(iObs)

                if not( strcmp(Obs_Letter,'P') | strcmp(Obs_Letter,'Q') )
                    temp_MatrixAnomaly = CosmicRayDetection2(permute(data(:,:), [2 1]), permute(YMask(:,:), [2 1]),MaskValues.Saturation,nbrLambdaBinned,MatrixIllum,MatrixROI,iObs,Instrument,Obs_Letter,DateYMD);

                    MatrixAnomaly(:,:,iObs) = permute(temp_MatrixAnomaly,[2 1]); % ->   nLambda x nLines x nObs
                end

                % set back in the right dimensions (nLambda x nLines x nObs)
            end

            % add anomalies mask to the previous mask (only where not saturated and not NaN in level 02a)
            YMask = uint8(YMask) + (uint8(not(YMask>=MaskValues.Saturation)) .* uint8(MatrixAnomaly(:,:,iObs))) * uint8(MaskValues.Anomaly);

            % add hot pixel mask to the previous mask (only where not saturated, not anomaly and not NaN in level 02a)
            YMask = uint8(YMask) + (uint8(not(YMask>=MaskValues.Saturation)) .* uint8(MatrixHotPixels)) * uint8(MaskValues.HotPix);

            %% Check for NaNs in level 0.2c (if NaN created by removeDC)
            YMask = DetectLevel02x_NaN(data,YMask,MaskValues.L02C_NaN);

            %% Append to YMask dataset
            YMask = uint8(YMask);
            h5write(dst, '/Science/YMask', YMask, [1 1 iObs], [nbrUsefulPix nbrCCDLines 1]);

            clear data YMask YMask_new
        end


    elseif (ibinning == 1) % vertically binned (not treated)

        Error_due_to_V_binned_data

    end


end % function

"""




"""

function RemoveDC(src, dst, aux)
% Remove the Dark Current
% pathfile = absolute path to hdf5 file that need to be threated
% err = String that contain explanation of an error, empty if no error
% version 0.1 (25/05/2018)
    addpath('subpipeline');

        Obs_Letter = src(1,end-3);
        imode = h5read(src, '/Channel/Mode', [1], [1]); % 1=SO, 2=Nadir
        ibinning = h5read(src, '/Channel/AcquisitionMode', [1], [1]); % 0=not binned 1=vertical binned
        itypedata = h5read(src,'/Channel/ReverseFlagAndDataTypeFlagRegister'); % 0=dark 1=dark with reverse clock 2=bias 4=science
        nbrLambdaBinned = h5read(src, '/Channel/HorizontalAndCombinedBinningSize', [1], [1]) + 1;

        iStartLineCCD = double(h5read(src, '/Channel/VStart', [1], [1])) + 1;
        iEndLineCCD = double(h5read(src, '/Channel/VEnd', [1], [1])) + 1;
        nbrCCDLines = iEndLineCCD - iStartLineCCD + 1;

        XNbBin = double(h5read(src, '/Science/XNbBin'));
        Xpix_1b = double(h5read(src, '/Science/Xpix_1b'));
        iStartLambdaPixel = double(h5read(src, '/Channel/HStart', [1], [1])) + 1;
        iEndLambdaPixel = double(h5read(src, '/Channel/HEnd', [1], [1])) + 1;
        if (iStartLambdaPixel == 1)
            iStartUsefulPix = ceil(8 / XNbBin(1)) + 1;
            if (iEndLambdaPixel >= 1032)
                iEndUsefulPix = floor(1032 / XNbBin(1));
                iStartOverscanPix = ceil(1041 / XNbBin(1)); % overscan region used to calculate overscan bias 1041-1048 (prior to that, exponential decrease)
            else
                iEndUsefulPix = size(XNbBin,1);
            end
            nbrUsefulPix = (iEndUsefulPix - iStartUsefulPix) + 1;
        else
            disp('Error! Hstart > 1, need code adaptation to treat this file ')
            Problem_RmDC_01
        end

        temperature = double(h5read(src,'/Housekeeping/TEMP_2_CCD')) + 273.15;
        temperature = temperature';
        itime=(h5read(src,'/DateTime'));
        nb_obs = double(h5readatt(src, '/', 'NSpec'));
        IT = double(h5read(src, '/Channel/IntegrationTime',[1],[1]));

        Gain = (7.0058); % calculated by linear fit on the pix by pix PTC using RS12

        Instrument = h5readatt(src,'/','InstName');
        if strcmp(Instrument,'FlightSpare NOMAD - Nadir and Occultation for MArs Discovery')
            Instrument=[];Instrument='Spare';
        elseif strcmp(Instrument,'NOMAD - Nadir and Occultation for MArs Discovery')
            Instrument=[];Instrument='Flight';
        else
            probleme
        end

        % Non-linearity correction
        MaskValues = Define_MaskValues;
        LinearityCorrValue = 54000;
        SaturationValue = 62000;
        % SaturationValue = 65535; % 1*(2^16)-1
        path_Table = [aux,'/UVIS_NonLinearity_Correction_Table.h5'];
        Corr_NL.count = double(h5read(path_Table, '/NonLinearityCorr/nbrCount'));
        Corr_NL.corr = double(h5read(path_Table, '/NonLinearityCorr/NonLinCorr'));
        Corr_NL.error = double(h5read(path_Table, '/NonLinearityCorr/NonLinCorrError'));

        % date
        DateString = h5read(src, '/DateTime',[1],[1]);
        DateYMD = str2num(datestr(datetime(DateString{1}(1:11),'InputFormat','yyyy MMM dd'),'yyyymmdd'));
        DateY = str2num(datestr(datetime(DateString{1}(1:11),'InputFormat','yyyy MMM dd'),'yyyy'));

        % DC vs T parameters
        [params,DC1_offset,DC1_corr_RowsUsed] = Load_Params_DC_correction(aux,imode,Obs_Letter,IT,DateY);

    %% Integrity checkup
    % Verification of the structure (1 Dark, x Science, 1 Dark/Bias)
    nbrObs=size(itypedata,1);
    error=false;
    for i=1:nbrObs
        if (itypedata(i)==4)
            if ~(itypedata(i-1)==4|itypedata(i-1)==0)&~(itypedata(i+1)==4|itypedata(i+1)==0|itypedata(i+1)==2)
                error=true;
            end
        end
    end

    if error
        exit(7);
    %err=7;return;

    else
        % Repartition of the darks into packets
        % Definition of structure packet contains indices of start DC, end DC,start science, end science
        YesMissingLastDark_ButCanBeTreated=0;
        nbpacket=0;packet(1).ScienceStart=0;packet(1).ScienceEnd=0;packet(1).DCStart=0;packet(1).DCEnd=0;
        i=0;
        while (i<nbrObs)
            i=i+1;
            if (itypedata(i)==4)
                nbpacket=nbpacket+1;
                packet(end+1).ScienceStart=i;packet(end).DCStart=i-1;
                while ( (i < nbrObs) & (itypedata(i+1)==4) )
                    i=i+1;
                end
                if (i == nbrObs & (itypedata(i)==4))
                    disp('Error! orbit does not end by a dark measurement')
                    if nbpacket == 1
                        Problem_RmDC_02_No_Ending_Dark_Meas % orbit does not end by a dark measurement
                    else
                        YesMissingLastDark_ButCanBeTreated = 1;
                        disp('In-between dark meas available: processing in between measurments')
                    end
                end
                packet(end).ScienceEnd=i;
                while (~(itypedata(i)==0) & (i<nbrObs) & not(YesMissingLastDark_ButCanBeTreated)) % because nbrObs science can be followed by dark or bias
                    i=i+1;
                end
                if (itypedata(i)~=0 & nbpacket > 1.5)
                    YesMissingLastDark_ButCanBeTreated = 1;
                end
                packet(end).DCEnd=i;
            end
        end
        packet(1)=[]; % delete of the useless initialised values


        %% prepare the dark dataset for anomaly detection ====================================================================================================================================
        vectNumDark = find(itypedata==0);
        nbrDark = size(vectNumDark,1);
        for iDark = 1:nbrDark
            temp_dataDark(:,:,iDark) = h5read(src, '/Science/Y', [1 1 vectNumDark(iDark)], [Inf Inf 1]); % nPix x nLines x nObs
        end
        %% Check for saturation
        temp_ymask_Dark = DetectSaturation(temp_dataDark,SaturationValue,XNbBin,MaskValues.Saturation);

        %% Check for NaNs in level 0.2a
        temp_ymask_Dark = DetectLevel02x_NaN(temp_dataDark,temp_ymask_Dark,MaskValues.L02A_NaN);

        %% Correction for non linearity
        [temp_dataDark,errSys_CorrNL,temp_ymask_Dark] = Non_Linearity_CCD_Correction(
            temp_dataDark,Corr_NL,LinearityCorrValue,SaturationValue,XNbBin,temp_ymask_Dark,MaskValues.NonLinCorr);
        u2_Sys_CorrNL = errSys_CorrNL.^2; % set it as square (as all the errors)

        % Correcting Dark measurements for Cosmic Rays before using them to remove DC
        if strcmp(Instrument,'Spare')
            [MatrixCosmicRayDark,MatrixHotPixels,dataDark_CosmicRayRemoved] = AnomalyDetectionDark(temp_dataDark,iStartUsefulPix,iEndUsefulPix,ibinning,'LabMeas');
            DateYMD = [];
        else
            if (DateYMD >= 20160314) % launch date
                if (nbrLambdaBinned == 1) % si pas de binning H
                    if ( strcmp(Obs_Letter,'L') | strcmp(Obs_Letter,'N') | strcmp(Obs_Letter,'O') )
                        [MatrixCosmicRayDark,MatrixHotPixels,dataDark_CosmicRayRemoved] = AnomalyDetectionDark(temp_dataDark,iStartUsefulPix,iEndUsefulPix,ibinning,'Strong');
                    elseif ( strcmp(Obs_Letter,'I') | strcmp(Obs_Letter,'E') )
                        [MatrixCosmicRayDark,MatrixHotPixels,dataDark_CosmicRayRemoved] = AnomalyDetectionDark(temp_dataDark,iStartUsefulPix,iEndUsefulPix,ibinning,'Occult');
                    elseif ( strcmp(Obs_Letter,'D') | strcmp(Obs_Letter,'P') | strcmp(Obs_Letter,'Q') )
                        [MatrixCosmicRayDark,MatrixHotPixels,dataDark_CosmicRayRemoved] = AnomalyDetectionDark(temp_dataDark,iStartUsefulPix,iEndUsefulPix,ibinning,'Medium');
                    end
                else % si binning H
                    if ( strcmp(Obs_Letter,'I') | strcmp(Obs_Letter,'E') )
                        [MatrixCosmicRayDark,MatrixHotPixels,dataDark_CosmicRayRemoved] = AnomalyDetectionDark(temp_dataDark,iStartUsefulPix,iEndUsefulPix,ibinning,'Occult_Bin');
                    else
                        [MatrixCosmicRayDark,MatrixHotPixels,dataDark_CosmicRayRemoved] = AnomalyDetectionDark(temp_dataDark,iStartUsefulPix,iEndUsefulPix,ibinning,'Medium');
                    end
                end
            else
                [MatrixCosmicRayDark,MatrixHotPixels,dataDark_CosmicRayRemoved] = AnomalyDetectionDark(temp_dataDark,iStartUsefulPix,iEndUsefulPix,ibinning,'LabMeas');
            end
        end

        Percent_CosmicRayDark = 100 * sum(sum(sum(MatrixCosmicRayDark))) /(size(MatrixCosmicRayDark,1)*size(MatrixCosmicRayDark,2)*size(MatrixCosmicRayDark,3));
        Percent_HotPixels = 100 * sum(sum(MatrixHotPixels)) / (size(MatrixHotPixels,1)*size(MatrixHotPixels,2));

        % save percentage of removed pixels
        h5create(dst, '/Channel/Percent_AnomalyDark', [1], 'DataType', 'double');
        h5create(dst, '/Channel/Percent_HotPixels', [1], 'DataType', 'double');
        h5write(dst, '/Channel/Percent_AnomalyDark',Percent_CosmicRayDark);
        h5write(dst, '/Channel/Percent_HotPixels',Percent_HotPixels);



        clear temp_dataDark temp_ymask_Dark


        %% DC Removal ============================================================================================================================================================

        % Fitting the temperature
        temperaturefit = Fitting_Temperature_CCD(itime,temperature,IT);

%         %% Remove DC in case of full frame
        if (ibinning == 0 || ibinning == 2)

            % Create appendable destination Y dataset
            h5create(dst, '/Science/Y', [nbrUsefulPix nbrCCDLines nb_obs], 'DataType', 'single', 'ChunkSize', [nbrUsefulPix min(24,nbrCCDLines) 1], 'Deflate', 4);
            h5create(dst, '/Science/YError', [nbrUsefulPix nbrCCDLines nb_obs], 'DataType', 'single', 'ChunkSize', [nbrUsefulPix min(24,nbrCCDLines) 1], 'Deflate', 4);
            h5create(dst, '/Science/YErrorSysNL', [nbrUsefulPix nbrCCDLines nb_obs], 'DataType', 'single', 'ChunkSize', [nbrUsefulPix min(24,nbrCCDLines) 1], 'Deflate', 4); % systematic error due to Non Linearity correction
            h5create(dst, '/Science/YMask', [nbrUsefulPix nbrCCDLines nb_obs], 'DataType', 'uint8', 'ChunkSize', [nbrUsefulPix 1 min(24,nb_obs)], 'Deflate', 4);

            % Write first bias frame unchanged to dst Y dataset
            % Saving initial bias (wanna keep them)
            indbias=find(itypedata==2);
            bias1 = h5read(src, '/Science/Y', [1 1 indbias(1)], [Inf Inf 1]);
            h5write(dst, '/Science/Y', bias1(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 indbias(1)], [nbrUsefulPix nbrCCDLines 1]);
            if not(YesMissingLastDark_ButCanBeTreated)
                bias2 = h5read(src, '/Science/Y', [1 1 indbias(2)], [Inf Inf 1]);
                h5write(dst, '/Science/Y', bias2(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 indbias(2)], [nbrUsefulPix nbrCCDLines 1]);
            else
                try
                    bias2 = h5read(src, '/Science/Y', [1 1 indbias(2)], [Inf Inf 1]);
                    h5write(dst, '/Science/Y', bias2(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 indbias(2)], [nbrUsefulPix nbrCCDLines 1]);
                catch
                    bias2 = bias1;
                end

            end

            % remove Overscan from bias
            if iEndUsefulPix < size(bias1,1)
                OverScan_bias1 = Calculation_OverScan(bias1,iStartOverscanPix);
                OverScan_bias2 = Calculation_OverScan(bias2,iStartOverscanPix);
                bias1 = bias1 - OverScan_bias1;
                bias2 = bias2 - OverScan_bias1;
            end

            % remove Overscan from dark
            if iEndUsefulPix < size(dataDark_CosmicRayRemoved,1)
                OverScan_Dark = Calculation_OverScan(dataDark_CosmicRayRemoved,iStartOverscanPix);
                dataDark_CosmicRayRemoved = dataDark_CosmicRayRemoved - OverScan_Dark;
            end

            % Readout Noise Calculation
            U2RN = ReadOutNoise_Calculation(indbias,bias1,bias2,iStartUsefulPix,iEndUsefulPix);

            for iPack=1:nbpacket % loop on the different packets

                if ( not(YesMissingLastDark_ButCanBeTreated) | (iPack~=nbpacket) )

                    iDark_CosRayRem_Start = find(packet(iPack).DCStart==vectNumDark);
                    iDark_CosRayRem_End = find(packet(iPack).DCEnd==vectNumDark);

                    % Shot noise for dark measurements
                    dataDark = h5read(src, '/Science/Y', [1 1 vectNumDark(iDark_CosRayRem_Start)], [Inf Inf 1]);
                    U2DC1 = ShotNoise_Calculation(dataDark,Gain);
                    dataDark = h5read(src, '/Science/Y', [1 1 vectNumDark(iDark_CosRayRem_End)], [Inf Inf 1]);
                    U2DC2 = ShotNoise_Calculation(dataDark,Gain);

                    U2DC1 = U2DC1 + U2RN;
                    U2DC2 = U2DC2 + U2RN;

                    % load data of the packet
                    nbrObsPack = (packet(iPack).DCEnd - packet(iPack).DCStart) + 1;
                    dataPack = h5read(src, '/Science/Y', [1 1 packet(iPack).DCStart], [Inf Inf nbrObsPack]);
                    itypedataPack = h5read(src,'/Channel/ReverseFlagAndDataTypeFlagRegister',[packet(iPack).DCStart],[nbrObsPack]); % 0=dark 1=dark with reverse clock 2=bias 4=science


                    %% Check for saturation
                    ymask = DetectSaturation(dataPack,SaturationValue,XNbBin,MaskValues.Saturation);

                    %% Check for NaNs in level 0.2a
                    ymask = DetectLevel02x_NaN(dataPack,ymask,MaskValues.L02A_NaN);

                    %% Correction for non linearity
                    [dataPack,errSys_CorrNL,ymask] = Non_Linearity_CCD_Correction(dataPack,Corr_NL,LinearityCorrValue,SaturationValue,XNbBin,ymask,MaskValues.NonLinCorr);
                    U2_Sys_CorrNL = errSys_CorrNL.^2; % set it as square (as all the errors)


                    % remove Overscan from data
                    if iEndUsefulPix < size(dataPack,1)
                        OverScan_dataPack = Calculation_OverScan(dataPack,iStartOverscanPix);
                        dataPack = dataPack - OverScan_dataPack;
                    end

                    % Correcting unexplained rise in the first dark measurement
                    if (iPack == 1)

                        if ( strcmp(Obs_Letter,'L') | strcmp(Obs_Letter,'N') | strcmp(Obs_Letter,'O') )

                            [Correction_FirstDC,LevelDC_Dark1] = Correction_DC1_offset_NLO_Obs(imode,Xpix_1b,XNbBin,dataDark_CosmicRayRemoved(:,:,1),dataPack,iStartLineCCD,iEndLineCCD,DC1_corr_RowsUsed,Instrument,ibinning,dataDark_CosmicRayRemoved(:,:,iDark_CosRayRem_Start),dataDark_CosmicRayRemoved(:,:,iDark_CosRayRem_End),U2DC1,U2DC2,temperaturefit(packet(iPack).DCStart),temperaturefit(packet(iPack).DCEnd),temperaturefit(packet(iPack).DCStart:packet(iPack).DCEnd),params);
                            dataDark_CosmicRayRemoved(:,:,1) = dataDark_CosmicRayRemoved(:,:,1) + Correction_FirstDC;

                            h5create(dst, '/Science/FirstDC_Corr', [1 nbrCCDLines], 'DataType', 'double');
                            h5create(dst, '/Science/FirstDC_LevInit', [1 nbrCCDLines], 'DataType', 'double');
                            h5write(dst, '/Science/FirstDC_Corr', Correction_FirstDC);
                            h5write(dst, '/Science/FirstDC_LevInit', LevelDC_Dark1);


                        end

                        dataPack(:,:,1) = dataDark_CosmicRayRemoved(:,:,1); % replace dark measurement of the datapack by the corrected one

                    end
                    % DC calculation + error
                    [SpectreDC,U2DC,unused] = Calculation_DC_FromTempFitFraction(Instrument,ibinning,dataDark_CosmicRayRemoved(:,:,iDark_CosRayRem_Start),dataDark_CosmicRayRemoved(:,:,iDark_CosRayRem_End),U2DC1,U2DC2,temperaturefit(packet(iPack).DCStart),temperaturefit(packet(iPack).DCEnd),temperaturefit(packet(iPack).DCStart:packet(iPack).DCEnd),params);

                    data_DCremoved = dataPack - SpectreDC;

                    % Shot noise calculation
                    U2SN = ShotNoise_Calculation(data_DCremoved,Gain);

                    % Total error
                    U2_Tot = U2DC + U2SN + U2RN;
                else

                    nbrObsPack = (packet(iPack).DCEnd - packet(iPack).DCStart) + 1;
                    data_DCremoved(1:iEndUsefulPix,1:nbrCCDLines,1:nbrObsPack) = NaN;
                    U2_Tot(1:iEndUsefulPix,1:nbrCCDLines,1:nbrObsPack) = NaN;
                    U2_Sys_CorrNL(1:iEndUsefulPix,1:nbrCCDLines,1:nbrObsPack) = NaN;
                    ymask(1:iEndUsefulPix,1:nbrCCDLines,1:nbrObsPack) = MaskValues.L02C_NaN;

                end

                % Write the results in the hdf5
                h5write(dst, '/Science/Y', data_DCremoved(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 packet(iPack).DCStart], [nbrUsefulPix nbrCCDLines nbrObsPack]);
                h5write(dst, '/Science/YError', U2_Tot(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 packet(iPack).DCStart], [nbrUsefulPix nbrCCDLines nbrObsPack]);
                h5write(dst, '/Science/YErrorSysNL', U2_Sys_CorrNL(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 packet(iPack).DCStart], [nbrUsefulPix nbrCCDLines nbrObsPack]);
                h5write(dst, '/Science/YMask', ymask(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 packet(iPack).DCStart], [nbrUsefulPix nbrCCDLines nbrObsPack]);

                clear dataPack ymask data_DCremoved SpectreDC U2DC U2SN U2_Tot U2_Sys_CorrNL
            end

            % Re-initialising bias (wanna keep them)
            for iBias = 1:size(indbias,1)
                bias = h5read(src, '/Science/Y', [1 1 indbias(iBias)], [Inf Inf 1]);
                h5write(dst, '/Science/Y', bias(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 indbias(iBias)], [nbrUsefulPix nbrCCDLines 1]);
            end

        elseif (ibinning == 1) % binned Vert (not performed)

            Error_due_to_V_binned_data

        end % if binning


        % if the end of the file is not [Science, Bias, Dark] but [Science, Dark, Bias, Dark] add the 2 missing frames : Bias and Dark
        if packet(end).DCEnd~=nbrObs
            bias = h5read(src, '/Science/Y', [1 1 nbrObs-1], [Inf Inf 1]); % bias
            h5write(dst, '/Science/Y', bias(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 nbrObs-1], [nbrUsefulPix nbrCCDLines 1]);
            lastdark = h5read(src, '/Science/Y', [1 1 nbrObs], [Inf Inf 1]); % last dark
            lastdark = lastdark - dataDark_CosmicRayRemoved(:,:,end);
            h5write(dst, '/Science/Y', lastdark(iStartUsefulPix:iEndUsefulPix,:,:), [1 1 nbrObs], [nbrUsefulPix nbrCCDLines 1]);
        end


        % save as 1024 pixels the other unmodified variables "CircuitNoise" and "X"
        X = h5read(src, '/Science/X');
        h5create(dst, '/Science/X', [nbrUsefulPix nbrObs], 'DataType', 'single', 'ChunkSize', [nbrUsefulPix nbrObs], 'Deflate', 4);
        h5write(dst, '/Science/X', X(iStartUsefulPix:iEndUsefulPix,:));

        Xpix_1b = h5read(src, '/Science/Xpix_1b');
        h5create(dst, '/Science/Xpix_1b', [nbrUsefulPix], 'DataType', 'single');
        h5write(dst, '/Science/Xpix_1b', Xpix_1b(iStartUsefulPix:iEndUsefulPix,1));

        CircuitNoise = h5read(src, '/Science/CircuitNoise');
        h5create(dst, '/Science/CircuitNoise', [nbrUsefulPix 2]);
        h5write(dst, '/Science/CircuitNoise', CircuitNoise(iStartUsefulPix:iEndUsefulPix,:));

        YNb = h5read(src, '/Science/YNb');
        YNb(:) = nbrUsefulPix;
        YNb = uint16(YNb);
        h5create(dst, '/Science/YNb', [nbrObs], 'DataType', 'uint16');
        h5write(dst, '/Science/YNb', YNb);

        h5create(dst, '/Science/XNbBin', [nbrUsefulPix], 'DataType', 'uint8');
        h5write(dst, '/Science/XNbBin', uint8(XNbBin(iStartUsefulPix:iEndUsefulPix,1)));

        % Detection of anomalies in the DC corrected spectrum

        AnomalyDetection(src,dst,MatrixCosmicRayDark(iStartUsefulPix:iEndUsefulPix,:,:),MatrixHotPixels(iStartUsefulPix:iEndUsefulPix,:,:),nbrUsefulPix,iStartUsefulPix,iEndUsefulPix,Instrument,Obs_Letter,DateYMD);


    end % if error
end



"""

# import matplotlib.pyplot as plt
# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p2b/2018/05/22/20180522_051504_0p2b_UVIS_I.h5"
# dst= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3b/2018/05/22/20180522_051504_0p3b_UVIS_I_py.h5"
# aux= "/bira-iasb/projects/NOMAD/Data/pfm_auxiliary_files/matlab/v_07"

# src= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p2b/2024/01/01/20240101_090648_0p2b_UVIS_I.h5"
# dst= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3b/2024/01/01/20240101_090648_0p3b_UVIS_I_py.h5"

# if os.path.exists(dst):
#     os.remove(dst)

def RemoveDC(src, dst):
# if True:

    hdf5_basename = os.path.basename(src).split(".")[0]

    aux_path = os.path.join(PFM_AUXILIARY_FILES, "matlab", "v_07")

    # Open source HDF5 file
    with h5py.File(src, 'r') as src_file:

        """get inputs from h5 file"""
        Obs_Letter= src[-4]  # Extract the observation letter from filename
        imode= src_file['Channel/Mode'][0]  # 1=SO, 2=Nadir
        # 0=not binned 1=vertical binned
        ibinning= src_file['Channel/AcquisitionMode'][0]
        # Data type flags
        itypedata= src_file['Channel/ReverseFlagAndDataTypeFlagRegister'][...]
        nbrLambdaBinned= int(src_file['Channel/HorizontalAndCombinedBinningSize'][0] + 1)
        iStartLineCCD= int(src_file['Channel/VStart'][0])
        iEndLineCCD= int(src_file['Channel/VEnd'][0])
        nbrCCDLines= iEndLineCCD - iStartLineCCD + 1
        XNbBin= src_file['Science/XNbBin'][...]
        Xpix_1b= src_file['Science/Xpix_1b'][...]
        iStartLambdaPixel= int(src_file['Channel/HStart'][0])
        iEndLambdaPixel= int(src_file['Channel/HEnd'][0])
        

        Y= np.swapaxes(src_file['Science/Y'][...], 0, 2)
        X= np.swapaxes(src_file['Science/X'][...], 0, 1)
        # Xpix_1b= src_file['Science/Xpix_1b'][...]
        CircuitNoise= np.squeeze(np.swapaxes(src_file['Science/CircuitNoise'][...], 0, 2))

        
        YNb= src_file['Science/YNb'][...]
        XNbBin= src_file['Science/XNbBin'][...]
    
        temperature= src_file['Housekeeping/TEMP_2_CCD'][...] + 273.15  # Convert to Kelvin
        itime= src_file['DateTime'][...]
        # nb_obs= src_file.attrs['NSpec']
        IT= src_file['Channel/IntegrationTime'][0]
    
        Instrument= src_file.attrs['InstName']
    
        DateString= src_file['DateTime'][0].decode()

    # iStartLineCCD=%i, iEndLineCCD, iStartLambdaPixel are offset by 1 in python
    # nbrCCDLines, iEndLambdaPixel are the same
    logger.info("%s: ibinning=%i, nbrLambdaBinned=%i, iStartLineCCD=%i, iEndLineCCD=%i, nbrCCDLines=%i, iStartLambdaPixel=%i, iEndLambdaPixel=%i, Yshape=%s", hdf5_basename, ibinning, nbrLambdaBinned, iStartLineCCD, iEndLineCCD, nbrCCDLines, iStartLambdaPixel, iEndLambdaPixel, Y.shape)

    DateYMD= datetime.strptime(
        DateString[0:11], '%Y %b %d').strftime('%Y%m%d')
    DateY= int(datetime.strptime(DateString[0:11], '%Y %b %d').strftime('%Y'))


    # TODO: flight spare
    if Instrument == 'FlightSpare NOMAD - Nadir and Occultation for MArs Discovery':
        Instrument= 'Spare'
    elif Instrument == 'NOMAD - Nadir and Occultation for MArs Discovery':
        Instrument= 'Flight'


    MaskValues, crit_mask= define_mask_values()
    LinearityCorrValue= 54000
    SaturationValue= 62000
    Gain= 7.0058  # Gain constant



    """get inputs from aux file"""
    path_Table= os.path.join(aux_path, "UVIS_NonLinearity_Correction_Table.h5")
    with h5py.File(path_Table, 'r') as table_file:
        Corr_NL= {
            'count': np.array(table_file['/NonLinearityCorr/nbrCount']),
            'corr': np.array(table_file['/NonLinearityCorr/NonLinCorr']),
            'error': np.array(table_file['/NonLinearityCorr/NonLinCorrError'])
        }


    
    if iStartLambdaPixel == 0:
        iStartUsefulPix= int(np.ceil(8 / XNbBin[0]))
        if iEndLambdaPixel >= 1032:
            iEndUsefulPix= int(np.floor(1032 / XNbBin[0])) - 1
            # TODO: check if valid
            # iStartOverscanPix= iEndUsefulPix + 1 
            iStartOverscanPix= int(np.ceil(1041 / XNbBin[0])) - 1 # overscan region last line
        else:
            iEndUsefulPix= len(XNbBin) - 1
        nbrUsefulPix= (iEndUsefulPix - iStartUsefulPix) + 1
    else:
        print("Error: Hstart > 1, need code adaptation to treat this file")
    
    # iStartUsefulPix, iEndUsefulPix, iStartOverscanPix are offset by 1 in python
    # nbrUsefulPix is the same
    logger.info("%s: iStartUsefulPix=%i, iEndUsefulPix=%i, iStartOverscanPix=%i, nbrUsefulPix=%i", hdf5_basename, iStartUsefulPix, iEndUsefulPix, iStartOverscanPix, nbrUsefulPix)

    # already define variables to hold the h5 output, do not write each frame to the file in a loop!
    d_out = {
        "Science/Y":np.zeros_like(Y[iStartUsefulPix:iEndUsefulPix+1, :, :]),
        "Science/YError":np.zeros_like(Y[iStartUsefulPix:iEndUsefulPix+1, :, :]),
        "Science/YErrorSysNL":np.zeros_like(Y[iStartUsefulPix:iEndUsefulPix+1, :, :]),
        "Science/YMask":np.zeros_like(Y[iStartUsefulPix:iEndUsefulPix+1, :, :], dtype=np.uint8),
    }



    params, DC1_offset, DC1_corr_RowsUsed= load_params_dc_correction(aux_path, imode, Obs_Letter, IT, DateY)

    nbrObs= len(itypedata)
    # integrity check loop
    error= False
    for i in range(1, len(itypedata)):
        if itypedata[i] == 4:
            if not ((itypedata[i-1] == 4 or itypedata[i-1] == 0) or (itypedata[i+1] in [4, 0, 2])):
                error= True
                break

    if error:
        print("Error: integrity check failed")
        # return 7

    # Repartition the darks into packets
    YesMissingLastDark_ButCanBeTreated = 0
    nbpacket = 0
    packet = [{'ScienceStart': 0, 'ScienceEnd': 0, 'DCStart': 0, 'DCEnd': 0}]
    i = -1
    
    while i < nbrObs-1:
        i += 1
        if itypedata[i] == 4:
            nbpacket += 1
            packet.append({'ScienceStart': i, 'DCStart': i - 1})
    
            while i < nbrObs and itypedata[i + 1] == 4:
                i += 1
    
            if i == nbrObs and itypedata[i] == 4:
                print('Error! orbit does not end by a dark measurement')
                if nbpacket == 1:
                    print("Orbit does not end by a dark measurement")
                else:
                    YesMissingLastDark_ButCanBeTreated = 1
                    print('In-between dark meas available: processing in between measurements')
    
            packet[-1]['ScienceEnd'] = i
    
            while itypedata[i] != 0 and i < nbrObs and not YesMissingLastDark_ButCanBeTreated:
                i += 1
    
            if itypedata[i] != 0 and nbpacket > 1.5:
                YesMissingLastDark_ButCanBeTreated = 1
    
            packet[-1]['DCEnd'] = i

    # Remove the initially initialized placeholder
    packet.pop(0)
    
    # all packet indices are offset by 1 in python
    # logger.info("%s packets: %s", hdf5_basename, packet)
    
    
    # Process dark data
    vectNumDark= np.where(itypedata == 0)[0]
    nbrDark= len(vectNumDark)

    # Pre-allocate array for dark data
    temp_dataDark= np.zeros((Y.shape[0], nbrCCDLines, nbrDark))
    for iDark in range(nbrDark):
        temp_dataDark[:, :, iDark]= Y[:, :, vectNumDark[iDark]]

    # Detect saturation for Dark Data
    temp_ymask_Dark= detect_saturation(temp_dataDark, SaturationValue, XNbBin, MaskValues['Saturation'])
    
    # print("temp_ymask_Dark sum:", np.sum(temp_ymask_Dark, axis=(0,1)))

    # Check for NaNs in level 0.2a
    temp_ymask_Dark= detect_level_02x_nan(temp_dataDark, temp_ymask_Dark, MaskValues['L02A_NaN'])

    # print("temp_ymask_Dark sum:", np.sum(temp_ymask_Dark, axis=(0,1)))

    # Correction for non-linearity
    # logger.info("Before non linearity correction: temp_dataDark[0,0,:]=%s", temp_dataDark[0,0,:])
    temp_dataDark, errSys_CorrNL, temp_ymask_Dark= non_linearity_ccd_correction(temp_dataDark, Corr_NL, LinearityCorrValue, SaturationValue, XNbBin, temp_ymask_Dark, MaskValues['NonLinCorr'])
    # logger.info("After non linearity correction: temp_dataDark[0,0,:]=%s", temp_dataDark[0,0,:])

    # print("temp_ymask_Dark sum:", np.sum(temp_ymask_Dark, axis=(0,1)))
    
    # plt.figure()
    # plt.imshow(temp_dataDark[:, :, 0])
    # plt.figure()
    # plt.imshow(temp_dataDark[:, :, 1])
    
    # set it as square (as all the errors)
    # u2_Sys_CorrNL= np.square(errSys_CorrNL)

    # Correcting Dark measurements for Cosmic Rays before using them to remove DC
    if Instrument == 'Spare':
        MatrixCosmicRayDark, MatrixHotPixels, dataDark_CosmicRayRemoved= anomaly_detection_dark(temp_dataDark, iStartUsefulPix, iEndUsefulPix, ibinning, 'LabMeas')
        DateYMD= 0
    else:
        if int(DateYMD) >= 20160314:  # launch date
            if nbrLambdaBinned == 1:  # No horizontal binning
                if Obs_Letter in ['L', 'N', 'O']:
                    MatrixCosmicRayDark, MatrixHotPixels, dataDark_CosmicRayRemoved= anomaly_detection_dark(temp_dataDark, iStartUsefulPix, iEndUsefulPix, ibinning, 'Strong')
                elif Obs_Letter in ['I', 'E']:
                    # data, i_start_useful_pix, i_end_useful_pix, ibinning, strength_criteria = temp_dataDark, iStartUsefulPix, iEndUsefulPix, ibinning, 'Occult'
                    MatrixCosmicRayDark, MatrixHotPixels, dataDark_CosmicRayRemoved= anomaly_detection_dark(temp_dataDark, iStartUsefulPix, iEndUsefulPix, ibinning, 'Occult')
                elif Obs_Letter in ['D', 'P', 'Q']:
                    MatrixCosmicRayDark, MatrixHotPixels, dataDark_CosmicRayRemoved= anomaly_detection_dark(temp_dataDark, iStartUsefulPix, iEndUsefulPix, ibinning, 'Medium')
            else:  # With horizontal binning
                if Obs_Letter in ['I', 'E']:
                    # data, i_start_useful_pix, i_end_useful_pix, ibinning, strength_criteria = temp_dataDark, iStartUsefulPix, iEndUsefulPix, ibinning, 'Occult_Bin'
                    MatrixCosmicRayDark, MatrixHotPixels, dataDark_CosmicRayRemoved= anomaly_detection_dark(temp_dataDark, iStartUsefulPix, iEndUsefulPix, ibinning, 'Occult_Bin')
                else:
                    MatrixCosmicRayDark, MatrixHotPixels, dataDark_CosmicRayRemoved= anomaly_detection_dark(temp_dataDark, iStartUsefulPix, iEndUsefulPix, ibinning, 'Medium')
        else:
            MatrixCosmicRayDark, MatrixHotPixels, dataDark_CosmicRayRemoved= anomaly_detection_dark(temp_dataDark, iStartUsefulPix, iEndUsefulPix, ibinning, 'LabMeas')


    # plt.figure()
    # plt.imshow(MatrixCosmicRayDark[:, :, 0])
    # plt.figure()
    # plt.imshow(MatrixCosmicRayDark[:, :, 1])
    

    # logger.info("After anomaly detection: dataDark_CosmicRayRemoved[0,0,:]=%s", dataDark_CosmicRayRemoved[0,0,:])

    # Calculate the percentage of Cosmic Rays detected and Hot Pixels
    Percent_CosmicRayDark= 100 * np.sum(MatrixCosmicRayDark) / (MatrixCosmicRayDark.shape[0] * MatrixCosmicRayDark.shape[1] * MatrixCosmicRayDark.shape[2])
    Percent_HotPixels= 100 * np.sum(MatrixHotPixels) / (MatrixHotPixels.shape[0] * MatrixHotPixels.shape[1])

    logger.info("%s: Percent cosmic ray dark=%0.4f, percent hot pixels=%0.2f", hdf5_basename, Percent_CosmicRayDark, Percent_HotPixels)
    print("Percent cosmic ray dark=%0.4f, percent hot pixels=%0.2f" %(Percent_CosmicRayDark, Percent_HotPixels))

    # Fit temperature using CCD
    temperaturefit= fitting_temperature_ccd(itime, temperature, IT)

    # Check if binning condition is met (Full Frame)
    if ibinning == 0 or ibinning == 2:

        # Write first bias frame unchanged to dst Y dataset
        indbias= np.where(itypedata == 2)[0]
        bias1= Y[:, :, indbias[0]].copy()
        
        
        d_out['Science/Y'][:, :, indbias[0]] = bias1[iStartUsefulPix:iEndUsefulPix+1, :]

        if not YesMissingLastDark_ButCanBeTreated:
            bias2= Y[:, :, indbias[1]].copy()
            d_out['Science/Y'][:, :, indbias[1]]= bias2[iStartUsefulPix:iEndUsefulPix+1, :]
        else:
            # try:
            bias2= Y[:, :, indbias[1]].copy()
            d_out['Science/Y'][:, :, indbias[1]]= bias2[iStartUsefulPix:iEndUsefulPix+1, :]
            # except Exception as e:
            #     print(e)
            #     bias2= bias1
        # logger.info("Before overscan removal: bias1[0,0]=%f, bias2[0,0]=%f", bias1[0,0], bias2[0,0])

        # Remove Overscan from bias
        if iEndUsefulPix < bias1.shape[0]:
            OverScan_bias1= calculation_overscan(bias1, iStartOverscanPix)
            # OverScan_bias2= calculation_overscan(bias2, iStartOverscanPix)
            bias1 -= OverScan_bias1
            bias2 -= OverScan_bias1
            # print(OverScan_bias1)
            # logger.info("After overscan removal: bias1[0,0]=%f, bias2[0,0]=%f", bias1[0,0], bias2[0,0])

        # Remove Overscan from dark
        if iEndUsefulPix < dataDark_CosmicRayRemoved.shape[0]:
            OverScan_Dark= calculation_overscan(dataDark_CosmicRayRemoved, iStartOverscanPix)
            dataDark_CosmicRayRemoved -= OverScan_Dark

        # Readout Noise Calculation
        U2RN= readout_noise_calculation(bias1, bias2, iStartUsefulPix, iEndUsefulPix)

        # Loop on the different packets
        for iPack in range(nbpacket):

            if not (YesMissingLastDark_ButCanBeTreated and iPack == nbpacket):
                iDark_CosRayRem_Start= np.where(packet[iPack]["DCStart"] == vectNumDark)[0]
                iDark_CosRayRem_End= np.where(packet[iPack]["DCEnd"] == vectNumDark)[0]

                # Shot noise for dark measurements
                dataDark1= Y[:, :, vectNumDark[iDark_CosRayRem_Start]]
                U2DC1= ShotNoise_Calculation(dataDark1, Gain)
                dataDark2= Y[:, :, vectNumDark[iDark_CosRayRem_End]]
                U2DC2= ShotNoise_Calculation(dataDark2, Gain)
                
                U2DC1 += U2RN
                U2DC2 += U2RN

                # Load data of the packet
                nbrObsPack= (packet[iPack]["DCEnd"] - packet[iPack]["DCStart"])
                dataPack= Y[:, :, packet[iPack]["DCStart"]:packet[iPack]["DCEnd"]+1].copy()
                
                # logger.info("%s: iPack=%i, nbrObsPack=%i, dataPack.shape=%s", hdf5_basename, iPack, nbrObsPack, dataPack.shape)
                # itypedataPack= itypedata[packet[iPack]["DCStart"]]
                

                # Check for saturation
                ymask= detect_saturation(dataPack, SaturationValue, XNbBin, MaskValues['Saturation'])

                # Check for NaNs in level 0.2a
                ymask= detect_level_02x_nan(dataPack, ymask, MaskValues['L02A_NaN'])

                # Correction for non-linearity
                # logger.info("dataPack[0,0,0] before = %f", dataPack[0,0,0])
                dataPack, errSys_CorrNL, ymask= non_linearity_ccd_correction(dataPack, Corr_NL, LinearityCorrValue, SaturationValue, XNbBin, ymask, MaskValues['NonLinCorr'])
                # logger.info("dataPack[0,0,0] after = %f", dataPack[0,0,0])
                U2_Sys_CorrNL= np.square(errSys_CorrNL)

                # Remove Overscan from data
                if iEndUsefulPix < dataPack.shape[0]:
                    OverScan_dataPack= calculation_overscan(dataPack, iStartOverscanPix)
                    dataPack -= OverScan_dataPack

                # Correcting unexplained rise in the first dark measurement
                if iPack == 0:
                    # if Obs_Letter in ['L', 'N', 'O']:
                        # Correction_FirstDC, LevelDC_Dark1= Correction_DC1_offset_NLO_Obs(
                        #     imode, Xpix_1b, XNbBin, dataDark_CosmicRayRemoved[:, :, 0], dataPack,
                        #     iStartLineCCD, iEndLineCCD, DC1_corr_RowsUsed, Instrument, ibinning,
                        #     dataDark_CosmicRayRemoved[:, :, iDark_CosRayRem_Start],
                        #     dataDark_CosmicRayRemoved[:, :, iDark_CosRayRem_End], U2DC1, U2DC2,
                        #     temperaturefit[packet[iPack]["DCStart"]], temperaturefit[packet[iPack]["DCEnd"]+1],
                        #     temperaturefit[packet[iPack]["DCStart"]:packet[iPack]["DCEnd"]+1], params
                        # )
                        # dataDark_CosmicRayRemoved[:, :, 0] += Correction_FirstDC

                        # d_out['Science/FirstDC_Corr'] = Correction_FirstDC
                        # d_out['Science/FirstDC_LevInit'] = LevelDC_Dark1


                    # replace dark measurement
                    dataPack[:, :, 0]= dataDark_CosmicRayRemoved[:, :, 0]

                # DC calculation + error
                #Instrument, ibinning, DC1, DC2, U2DC1, U2DC2, T1, T2, T, params = Instrument, ibinning, dataDark_CosmicRayRemoved[:, :, iDark_CosRayRem_Start],dataDark_CosmicRayRemoved[:, :, iDark_CosRayRem_End], U2DC1, U2DC2,temperaturefit[packet[iPack]["DCStart"]], temperaturefit[packet[iPack]["DCEnd"]],temperaturefit[packet[iPack]["DCStart"]:packet[iPack]["DCEnd"]], params

                SpectreDC, U2DC, _= Calculation_DC_FromTempFitFraction(
                    Instrument, ibinning, dataDark_CosmicRayRemoved[:, :, iDark_CosRayRem_Start],
                    dataDark_CosmicRayRemoved[:, :, iDark_CosRayRem_End], U2DC1, U2DC2,
                    temperaturefit[packet[iPack]["DCStart"]], temperaturefit[packet[iPack]["DCEnd"]],
                    temperaturefit[packet[iPack]["DCStart"]:packet[iPack]["DCEnd"]+1], params
                )

                data_DCremoved= dataPack - SpectreDC

                # Shot noise calculation
                U2SN= ShotNoise_Calculation(data_DCremoved, Gain)

                # Total error
                U2_Tot= U2DC + U2SN + U2RN

            else:
                # TODO: check this bit
                nbrObsPack= (packet[iPack]["DCEnd"] - packet[iPack]["DCStart"])
                data_DCremoved[:iEndUsefulPix+1,:nbrCCDLines+1, :nbrObsPack+1]= np.nan
                U2_Tot[:iEndUsefulPix+1, :nbrCCDLines+1, :nbrObsPack+1]= np.nan
                U2_Sys_CorrNL[:iEndUsefulPix+1,:nbrCCDLines+1, :nbrObsPack+1]= np.nan
                ymask[:iEndUsefulPix+1, :nbrCCDLines+1, :nbrObsPack+1]= MaskValues['L02C_NaN']

            # Write the results to the output dictionary for saving to file later
            d_out['Science/Y'][:, :, packet[iPack]["DCStart"]:packet[iPack]["DCEnd"]+1]= data_DCremoved[iStartUsefulPix:iEndUsefulPix+1, :, :]
            d_out['Science/YError'][:, :, packet[iPack]["DCStart"]:packet[iPack]["DCEnd"]+1]= U2_Tot[iStartUsefulPix:iEndUsefulPix+1, :, :]
            d_out['Science/YErrorSysNL'][:, :, packet[iPack]["DCStart"]:packet[iPack]["DCEnd"]+1]= U2_Sys_CorrNL[iStartUsefulPix:iEndUsefulPix+1, :, :]
            d_out['Science/YMask'][:, :, packet[iPack]["DCStart"]:packet[iPack]["DCEnd"]+1]= ymask[iStartUsefulPix:iEndUsefulPix+1, :, :]
             

            # Re-initializing bias (keep them)
            for iBias in range(len(indbias)):
                bias= Y[:, :, indbias[iBias]]
                d_out['Science/Y'][:, :, indbias[iBias]]= bias[iStartUsefulPix:iEndUsefulPix+1, :]

                

    # Handle binned vertical case (not performed)
    elif ibinning == 1:
        print("Error_due_to_V_binned_data")

    # If the end of the file is not [Science, Bias, Dark] but [Science, Dark, Bias, Dark], add the 2 missing frames: Bias and Dark
    if packet[-1]["DCEnd"] != nbrObs -1:
        # Read bias frame and write it to the destination
        print("File does not end with [Science, Bias, Dark] frames")
        bias= Y[:, :, nbrObs - 2]  # Bias
        d_out['Science/Y'][:, :, nbrObs - 2]= bias[iStartUsefulPix:iEndUsefulPix+1, :]

        # Read last dark frame, adjust it and write it to the destination
        lastdark= Y[:, :, nbrObs - 1]  # Last dark
        lastdark= lastdark - dataDark_CosmicRayRemoved[:, :, -1]
        d_out['Science/Y'][:, :, nbrObs -1 ]= lastdark[iStartUsefulPix:iEndUsefulPix+1, :]



    # Save as 1024 pixels the other unmodified variables: CircuitNoise and X
    YNb.fill(nbrUsefulPix)  # Fill with the number of useful pixels
    YNb= np.uint16(YNb)
    
    Xpix_1b = Xpix_1b[iStartUsefulPix:iEndUsefulPix+1]
    

    """Detection of anomalies in the DC corrected spectrum"""
    # print("Anomaly detection")

    if ibinning == 0 or ibinning == 2:  # FULLFRAME
        # Identify science measurements
        # Identify the science spectra (no dark, ...)
        vectSci= (itypedata == 4)
        # Identify the dark spectra
        vectDark= ((itypedata == 0) | (itypedata == 1))

        # Initialize MatrixAnomaly
        MatrixAnomaly= np.zeros((nbrUsefulPix, nbrCCDLines, nbrObs))

        # Apply cosmic ray dark matrix for dark spectra
        MatrixAnomaly[:, :, vectDark]= MatrixCosmicRayDark[iStartUsefulPix:iEndUsefulPix+1, :, :]

        # Define top and bottom non-illuminated lines
        iLineTopStart256_NonIllum, iLineBottomStart256_NonIllum= Define_Top_Bottom_NonIlluminated_Lines(imode, Xpix_1b, Instrument)
        iLineBottomStart_NonIllum= np.asarray(iLineBottomStart256_NonIllum - iStartLineCCD, dtype=int)
        iLineBottomStart_NonIllum[iLineBottomStart_NonIllum <= 0]= 0
        iLineTopStart_NonIllum= np.asarray(iLineTopStart256_NonIllum - iStartLineCCD + 1, dtype=int)
        iLineTopStart_NonIllum[iLineTopStart_NonIllum > nbrCCDLines]= nbrCCDLines + 1

        # convert to python
        iLineTopStart_NonIllum -= 2

        # nbrSmearingLines 1 more than matlab
        # iStartLineBin iEndLineBin, nbrBinLines, same as matlab
        ROI_ok, iStartLineBin, iEndLineBin, nbrBinLines, nbrSmearingLines, unused= Define_ROI(Instrument, imode, ibinning, iStartLineCCD, iEndLineCCD, Xpix_1b)
        
        #for python
        iStartLineBin -= 1

        # MatrixIllum[:,:,0] 2d array offset by 1
        # MatrixIllum[:,:,1] 2d array NOT offset by 1
        # MatrixIllum[:,:,2] 2d array offset by 1
        # MatrixROI 2D array offset by 1
        
        MatrixIllum, MatrixROI= Create_MatrixIllum(Obs_Letter, Instrument, DateYMD, Xpix_1b, iLineBottomStart_NonIllum, iLineTopStart_NonIllum, iStartLineCCD, iEndLineCCD, nbrCCDLines, iStartLineBin, iEndLineBin, nbrLambdaBinned)
        
        #plot illumination masks
        # plt.figure()
        # plt.imshow(np.swapaxes(MatrixIllum, 0, 1))
        # plt.xlabel("Spectral axis")
        # plt.title("Illumination Mask")

        # Loop through observations
        # for iObs in progress(range(nbrObs)):
        for iObs in range(nbrObs):
            # with h5py.File(dst, 'r+') as fdst:
            # Read data for the current observation
            data= d_out['Science/Y'][:, :, iObs]
            # Read Y mask
            YMask_in= d_out['Science/YMask'][:, :, iObs]

            if vectSci[iObs]:
                if Obs_Letter not in ['P', 'Q']:
                    
                    # spectra, YMask, MaskValueSaturation, nbrLambdaBinned, MatrixIllum_full, MatrixROI_full, iObs, Instrument, Obs_Letter, DateYMD = data, YMask_in, MaskValues["Saturation"], nbrLambdaBinned, MatrixIllum, MatrixROI, iObs, Instrument, Obs_Letter, DateYMD
                    temp_MatrixAnomaly= CosmicRayDetection2(data, YMask_in, MaskValues["Saturation"], nbrLambdaBinned, MatrixIllum, MatrixROI, iObs, Instrument, Obs_Letter, DateYMD)

                    # Reorder dimensions (nLambda x nLines x nObs)
                    MatrixAnomaly[:, :, iObs]= temp_MatrixAnomaly

            # Update YMask with anomaly mask
            YMask= np.uint8(YMask_in) + (np.uint8(~(YMask_in >= MaskValues["Saturation"])) * np.uint8(MatrixAnomaly[:, :, iObs])) * np.uint8(MaskValues["Anomaly"])

            # Add hot pixel mask to YMask
            YMask= np.uint8(YMask) + (np.uint8(~(YMask >= MaskValues["Saturation"])) * np.uint8(MatrixHotPixels[iStartUsefulPix:iEndUsefulPix+1, :])) * np.uint8(MaskValues["HotPix"])

            # Check for NaNs in level 0.2c (if NaN created by removeDC)
            YMask= detect_level_02x_nan(data, YMask, MaskValues["L02C_NaN"])
            
            # if iObs < 11:
            #     print("iObs=%i, np.sum(YMask)=%i" %(iObs, np.sum(YMask)))
            
            # Write the updated YMask back to the dataset
            d_out['Science/YMask'][:, :, iObs] = np.uint8(YMask)

    elif ibinning == 1:  # Vertically binned (not treated)
        print("Error due to V binned data")

    """save to H5 file only at the end"""
    with h5py.File(src, 'r') as hdf5FileIn:
        with h5py.File(dst, 'w') as hdf5FileOut:
    
            generics.copyAttributesExcept(hdf5FileIn, hdf5FileOut, OUTPUT_VERSION, ATTRIBUTES_TO_BE_REMOVED)
        
            #don't copy all datasets to new file
            for dset_path, dset in generics.iter_datasets(hdf5FileIn):
                if dset_path in DATASETS_TO_BE_REMOVED: #don't copy
                    continue
        
                dest = generics.createIntermediateGroups(hdf5FileOut, dset_path.split("/")[:-1])
                hdf5FileIn.copy(dset_path, dest)
    
    
            
            for key, value in d_out.items():
                if key == "Science/YMask":
                    dtype = "uint8"
                else:
                    dtype = "float32"
                hdf5FileOut.create_dataset(key, data=np.swapaxes(value, 0, 2), dtype=dtype, 
                                 chunks=(1, min(24, nbrCCDLines), nbrUsefulPix), compression="gzip", compression_opts=4)
            
            hdf5FileOut.create_dataset('Science/X', data=np.swapaxes(X[iStartUsefulPix:iEndUsefulPix+1, :], 0, 1), dtype='float32', 
                             chunks=(nbrObs, nbrUsefulPix), compression='gzip', compression_opts=4)
            hdf5FileOut.create_dataset('Science/Xpix_1b', data=Xpix_1b, dtype='float32', compression='gzip', compression_opts=4)
            hdf5FileOut.create_dataset('Science/CircuitNoise', data=np.swapaxes(CircuitNoise[iStartUsefulPix:iEndUsefulPix+1, :], 0, 1), dtype='float32', compression='gzip', compression_opts=4)
            hdf5FileOut.create_dataset('Science/YNb', data=YNb, dtype='uint16')
            hdf5FileOut.create_dataset('Science/XNbBin', data=np.uint8(XNbBin[iStartUsefulPix:iEndUsefulPix+1]), dtype='uint8')
    
            # Save percentage of removed pixels
            hdf5FileOut.create_dataset('Channel/Percent_AnomalyDark',data=Percent_CosmicRayDark)
            hdf5FileOut.create_dataset('Channel/Percent_HotPixels',data=Percent_HotPixels)


        

def convert(hdf5_filepath):
    """this function should perform the same function as nomad_ops/matlab/src/RemoveDC.m"""
    hdf5_basename = os.path.basename(hdf5_filepath).split(".")[0]
    logger.info("convert: %s", hdf5_basename)

    tmp_file = os.path.join(NOMAD_TMP_DIR, os.path.basename(hdf5_filepath))
    # shutil.copyfile(hdf5file_path, tmp_file)
    
    RemoveDC(hdf5_filepath, tmp_file)
    
    return [tmp_file]


#compare files
# old_dst= "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_0p3b/2018/05/22/20180522_051504_0p3b_UVIS_I.h5"
# import matplotlib.pyplot as plt

# d_compare = {}
# with h5py.File(old_dst, "r") as fold:
#     with h5py.File(dst, "r") as fnew:
#         for key in fnew["Science"].keys():
#             dset1 = fnew["Science"][key][...]
#             dset2 = fold["Science"][key][...]
#             print(key, dset1.ndim, np.all(dset1==dset2))
#             d_compare[key] = [fnew["Science"][key][...], fold["Science"][key][...]]

#             if np.any(dset1!=dset2):
#                 if dset1.ndim == 3:
#                     plt.figure()
#                     plt.title("New %s" %key)
#                     plt.imshow(dset1[:, :, 10], aspect="auto")

#                     plt.figure()
#                     plt.title("Old %s" %key)
#                     plt.imshow(dset2[:, :, 10], aspect="auto")
