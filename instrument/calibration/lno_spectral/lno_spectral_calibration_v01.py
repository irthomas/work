# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:55:24 2023

@author: iant

LNO SPECTRAL CALIBRATION - GO THROUGH ALL LNO OCCULTATION FULLSCANS, ORDER BY ORDER, AND WRITE A LOG CONTAINING:
    1. 
"""


import re
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd


# from matplotlib.backends.backend_pdf import PdfPages

# from scipy.optimize import curve_fit
# from scipy.optimize import OptimizeWarning



from tools.spectra.hapi_functions import hapi_transmittance, get_hapi_nu_range, hapi_fetch
from tools.spectra.baseline_als import baseline_als
# from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption
from tools.spectra.solar_spectrum_lno import get_solar_spectrum_hr, smooth_solar_spectrum


from tools.plotting.colours import get_colours

from tools.general.get_minima_maxima import get_local_minima
from tools.general.get_nearest_index import get_nearest_index

from tools.file.hdf5_functions import make_filelist, open_hdf5_file
from tools.file.paths import paths

from tools.file.write_log import write_log



file_level = "hdf5_level_1p0a"

PLOT_SPECTRA = True


GET_X_FROM = "files"
# GET_X_FROM = "formula"

NU_OFFSET = 0.0
# NU_OFFSET = 0.55 #apply offset to X dataset in cm-1 in h5 to correctly locate the lines




# regex = re.compile("20(18|21)04.._......_.*_SO_A_[IE]_%i" %order)
# regex = re.compile("20(18|19|20|21)04.._......_.*_SO_A_[IE]_%i" %order)
# regex = re.compile("20(18|19|20|21)0[4567].._......_.*_SO_A_[IE]_%i" %order)
# regex = re.compile("20(18|19|20|21|22)...._......_.*_LNO_1_S")

# hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level, open_files=False, silent=True)


# colours = get_colours(len(hdf5_filenames))


hdf5_filenames = [
 	  "20181129_223804_0p3a_LNO_1_S",
 	 "20181130_090450_0p3a_LNO_1_S",
 	 "20181130_201600_0p3a_LNO_1_S",
 	 "20181201_064216_0p3a_LNO_1_S",
 	 "20190303_155600_0p3a_LNO_1_S",
 	 "20190308_150728_0p3a_LNO_1_S",
 	 "20191123_101028_0p3a_LNO_1_S",
 	 "20191208_154838_0p3a_LNO_1_S",
 	 "20200306_003856_0p3a_LNO_1_S",
 	 "20200318_202946_0p3a_LNO_1_S",
 	 "20200529_072041_0p3a_LNO_1_S",
 	 "20200706_153502_0p3a_LNO_1_S",
 	 "20200808_204006_0p3a_LNO_1_S",
 	 "20200816_072716_0p3a_LNO_1_S",
 	 "20200824_234630_0p3a_LNO_1_S",
 	 "20200904_151949_0p3a_LNO_1_S",
 	 "20200907_021917_0p3a_LNO_1_S",
 	 "20200909_230913_0p3a_LNO_1_S",
 	 "20200910_070112_0p3a_LNO_1_S",
 	 "20200915_054020_0p3a_LNO_1_S",
 	 "20200920_010731_0p3a_LNO_1_S",
 	 "20200926_230009_0p3a_LNO_1_S",
 	 "20200927_235410_0p3a_LNO_1_S",
 	 "20200930_014450_0p3a_LNO_1_S",
 	 "20200930_125113_0p3a_LNO_1_S",
 	 "20201023_032231_0p3a_LNO_1_S",
 	 "20201030_102016_0p3a_LNO_1_S",
 	 "20201122_093841_0p3a_LNO_1_S",
 	 "20201122_113630_0p3a_LNO_1_S",
 	 "20201127_092855_0p3a_LNO_1_S",
 	 "20201202_171220_0p3a_LNO_1_S",
 	 "20201208_005647_0p3a_LNO_1_S",
 	 "20201228_201906_0p3a_LNO_1_S",
 	 "20210116_014604_0p3a_LNO_1_S",
 	 "20210119_062940_0p3a_LNO_1_S",
 	 "20210121_010500_0p3a_LNO_1_S",
 	 "20210122_182131_0p3a_LNO_1_S",
 	 "20210129_034005_0p3a_LNO_1_S",
 	 "20210206_105549_0p3a_LNO_1_S",
 	 "20210223_074439_0p3a_LNO_1_S",
 	 "20210223_114025_0p3a_LNO_1_S",
 	 "20210223_170643_0p3a_LNO_1_S",
 	 "20210223_173351_0p3a_LNO_1_S",
 	 "20210310_115155_0p3a_LNO_1_S",
 	 "20210310_234051_0p3a_LNO_1_S",
 	 "20210415_121120_0p3a_LNO_1_S",
 	 "20210418_070223_0p3a_LNO_1_S",
 	 "20210511_223723_0p3a_LNO_1_S",
 	 "20210519_005834_0p3a_LNO_1_S",
 	 "20210623_002924_0p3a_LNO_1_S",
 	 "20210626_114224_0p3a_LNO_1_S",
 	 "20210702_104031_0p3a_LNO_1_S",
 	 "20210703_101617_0p3a_LNO_1_S",
 	 "20210704_174345_0p3a_LNO_1_S",
 	 "20210705_021616_0p3a_LNO_1_S",
 	 "20210722_220608_0p3a_LNO_1_S",
 	 "20210801_195522_0p3a_LNO_1_S",
 	 "20210805_165553_0p3a_LNO_1_S",
 	 "20210829_152334_0p3a_LNO_1_S",
 	 "20210831_225714_0p3a_LNO_1_S",
 	 "20210902_003108_0p3a_LNO_1_S",
 	 "20210906_155358_0p3a_LNO_1_S",
 	 "20210907_022429_0p3a_LNO_1_S",
 	 "20210908_090941_0p3a_LNO_1_S",
 	 "20210915_120721_0p3a_LNO_1_S",
 	 "20210917_234914_0p3a_LNO_1_S",
 	 "20211103_084211_0p3a_LNO_1_S",
 	 "20211113_063445_0p3a_LNO_1_S",
 	 "20211125_072852_0p3a_LNO_1_S",
 	 "20211130_210510_0p3a_LNO_1_S",
 	 "20211203_072351_0p3a_LNO_1_S",
 	 "20220101_203234_0p3a_LNO_1_S",
 	 "20220103_051849_0p3a_LNO_1_S",
 	 "20220206_191809_0p3a_LNO_1_S",
 	 "20220227_084232_0p3a_LNO_1_S",
 	 "20220228_042153_0p3a_LNO_1_S",
 	 "20220301_055453_0p3a_LNO_1_S",
 	 "20220305_215626_0p3a_LNO_1_S",
 	 "20220309_190225_0p3a_LNO_1_S",
 	 "20220320_024256_0p3a_LNO_1_S",
 	 "20220326_234840_0p3a_LNO_1_S",
 	 "20220327_034436_0p3a_LNO_1_S",
 	 "20220329_084826_0p3a_LNO_1_S",
 	 "20220403_060452_0p3a_LNO_1_S",
 	 "20220408_043813_0p3a_LNO_1_S",
 	 "20220424_022357_0p3a_LNO_1_S",
 	 "20220501_005456_0p3a_LNO_1_S",
 	 "20220505_193821_0p3a_LNO_1_S",
 	 "20220508_043844_0p3a_LNO_1_S",
 	 "20220513_034755_0p3a_LNO_1_S",
 	 "20220521_185310_0p3a_LNO_1_S",
 	 "20220525_123427_0p3a_LNO_1_S",
 	 "20220528_072416_0p3a_LNO_1_S",
 	 "20220531_155948_0p3a_LNO_1_S",
 	 "20220627_054939_0p3a_LNO_1_S",
 	 "20220720_203340_0p3a_LNO_1_S",
 	 "20220722_071249_0p3a_LNO_1_S",
 	 "20220908_180129_0p3a_LNO_1_S",
 	 "20220914_010248_0p3a_LNO_1_S",
 	 "20221023_132004_0p3a_LNO_1_S",
 	 "20221024_145245_0p3a_LNO_1_S",
 	 "20221108_171339_0p3a_LNO_1_S",
 	 "20221111_235016_0p3a_LNO_1_S",
 	 "20221118_102446_0p3a_LNO_1_S",
 	 "20221119_005005_0p3a_LNO_1_S",
 	 "20221119_080225_0p3a_LNO_1_S",
 	 "20221124_043622_0p3a_LNO_1_S",
 	 "20221223_090345_0p3a_LNO_1_S",
]


order_dict = {
    119:{"molecule":"CO2", "n_lines":6},
    121:{"molecule":"CO2", "n_lines":15},
    134:{"molecule":"H2O", "n_lines":8},
    136:{"molecule":"H2O", "n_lines":6},
    147:{"molecule":"CO2", "n_lines":14},
    148:{"molecule":"CO2", "n_lines":11},
    149:{"molecule":"CO2", "n_lines":13},
    163:{"molecule":"CO2", "n_lines":10},
    164:{"molecule":"CO2", "n_lines":14},
    165:{"molecule":"CO2", "n_lines":15},
    167:{"molecule":"H2O", "n_lines":5},
    168:{"molecule":"H2O", "n_lines":9},
    169:{"molecule":"H2O", "n_lines":9},
    186:{"molecule":"CO", "n_lines":5},
    189:{"molecule":"CO", "n_lines":7},
    190:{"molecule":"CO", "n_lines":8},
    191:{"molecule":"CO", "n_lines":10},
    
}



def it23_waven(diffraction_order, temperature):
    """LNO spectral calibration Ian January 23"""
    A =  -1.371398744964081e-08
    B =  0.0005679172811247807
    C =  22.4713672286215
    D =  -0.858488846171797
    cfpixel = [A, B, C]

    p0_nu = D * temperature #px/°C * instrument temperature
    px_t_offset = np.arange(320.) + p0_nu
    nu  = np.polyval(cfpixel, px_t_offset) * diffraction_order
    
    return nu #wavenumbers of pixels


def fit_polynomial(x_in, y_in, deg=2, error=False, hr_num=100):
    
    x_mean = np.mean(x_in)
    x_centred = x_in - x_mean
    polyfit = np.polyfit(x_centred, y_in, deg)
    
    y_fit = np.polyval(polyfit, x_centred)
    
    if error:
        chi_squared = np.sum(((y_in - y_fit) / y_fit)**2) #divide by yfit to normalise large and small absorption bands
    
    x_hr = np.linspace(x_in[0], x_in[-1], num=hr_num)
    y_hr = np.polyval(polyfit, x_hr - x_mean)
    
    min_index = (np.abs(y_hr - np.min(y_hr))).argmin()
    
    x_min_position = x_hr[min_index]
    
    if error:
        return x_hr, y_hr, x_min_position, chi_squared
    else:
        return x_hr, y_hr, x_min_position




write_log("Filename\tOrder\tTemperature\tAbsorption nu\tAbsorption pixel\tDelta nu\tFitted depth\tFitted max")


# hapi_fetch("H2O", 3745, 3835)
# hapi_fetch("CO2", 3745, 3835)
# hapi_fetch("CO", 4000, 4500)

# for file_index, (h5_path, h5) in enumerate(zip(hdf5_files[-2:-1], hdf5_filenames[-2:-1])):
# fig2, ax2 = plt.subplots(figsize=(12, 5), constrained_layout=True)
for order_ix, order in enumerate(order_dict.keys()):
    
    #loop through chosen orders
    if PLOT_SPECTRA:
        fig1, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)

    
    n_lines = order_dict[order]["n_lines"]
    molecule = order_dict[order]["molecule"]
    
    for file_index, h5 in enumerate(hdf5_filenames):
    
        #loop through files
        
        h5_f = open_hdf5_file(h5, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")
        # h5_f = open_hdf5_file(h5)
        
        orders_all = h5_f["Channel/DiffractionOrder"][...]
        bins = h5_f["Science/Bins"][:, 0]
        
        y_all = h5_f["Science/Y"][:, :]
        x_all = h5_f["Science/X"][:, :]
        
        alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
        temperatures = h5_f["Channel/InterpolatedTemperature"]
        
        #find indices where detector row = 147 or 148
        centre_bin_ixs = np.where((bins == 147) & (orders_all == order))[0]
        if len(centre_bin_ixs) == 0:
            centre_bin_ixs = np.where((bins == 148) & (orders_all == order))[0]
            
    
        spectra_alts = alts[centre_bin_ixs]
        spectrum_px = np.arange(40, 320)
        spectra = y_all[centre_bin_ixs, spectrum_px[0]:(spectrum_px[-1]+1)] #ignore first and last 20 pixels
        spectrum_ts = temperatures[centre_bin_ixs]
    
        spectra_mean = np.mean(spectra, axis=1)
        
        if spectra_mean[0] > spectra_mean[-1]: #if ingress
            lowest_altitude_ix = np.where(spectra_mean > 10000.)[0][-1]
            toa_altitude_ix = np.where(spectra_alts > 50.)[0][-1]
        else: #if egress
            lowest_altitude_ix = np.where(spectra_mean > 10000.)[0][0]
            toa_altitude_ix = np.where(spectra_alts > 50.)[0][0]
    
        #get transmittance i.e. divide best atmos spectrum by toa spectrum
        spectrum = spectra[lowest_altitude_ix, :] / spectra[toa_altitude_ix, :]
        #correct baseline?
        spectrum_cont = baseline_als(spectrum, lam=150.0, p=0.97)
        spectrum /= spectrum_cont
        
        
        spectrum_t = spectrum_ts[lowest_altitude_ix]
        alt = spectra_alts[lowest_altitude_ix]
        print(h5, ":", alt, "km")

        #read in x from file, then apply offset to match data
        if GET_X_FROM == "files":
            spectrum_nu_no_offset = x_all[centre_bin_ixs[0], spectrum_px[0]:(spectrum_px[-1]+1)] #now use new function below
            spectrum_nu = spectrum_nu_no_offset + NU_OFFSET

        elif GET_X_FROM == "formula":
            #when calibration is known, use this instead.
            spectrum_nu_no_offset = it23_waven(order, spectrum_t)[spectrum_px] #use new function
            spectrum_nu = spectrum_nu_no_offset #use new function


        # ax1.plot(np.arange(20, 300)+320*(order-167), spectrum_norm, color=colour_dict[order])
        ax1.plot(spectrum_nu, spectrum, color="C0")
        
        if file_index == 0:
            # plot molecular absorptions?
            hapi_nu_range = get_hapi_nu_range(molecule)
            if spectrum_nu[0]-1. < hapi_nu_range[0] or spectrum_nu[-1]+1. > hapi_nu_range[1]:
                print("Refetching HAPI data")
                clear=True
            else:
                clear=False
                
            occ_sim_nu, occ_sim = hapi_transmittance(molecule, alt, [spectrum_nu[0]-1., spectrum_nu[-1]+1.], 0.001, clear=clear)
            if PLOT_SPECTRA:
                ax1a = ax1.twinx()
                ax1a.plot(occ_sim_nu, occ_sim, "k")
            
            
            # #plot solar spectrum?
            # nu_hr = np.arange(spectra_x[0]-1., spectra_x[-1]+1., 0.001)
            # solar_hr = get_solar_spectrum_hr(nu_hr)
            # solar_lr = smooth_solar_spectrum(solar_hr, 199)
            
            # if PLOT_SPECTRA:
            #     plt.plot(nu_hr, solar_lr, color="gray", alpha=0.7)
        

            abs_ix_hr = get_local_minima(occ_sim) #get solar absorption minima indices
            abs_nu_hrs = occ_sim_nu[abs_ix_hr] #get absorption nu
            abs_y_hrs = occ_sim[abs_ix_hr] #get absorption depth
        
        
            #N strongest lines
            abs_y_cutoff = sorted(abs_y_hrs)[n_lines] #select only the n strongest lines
    
            if PLOT_SPECTRA:
                for abs_index, (abs_nu_hr, abs_y_hr) in enumerate(zip(abs_nu_hrs, abs_y_hrs)):
                    if abs_y_hr < abs_y_cutoff:
                        ax1.text(abs_nu_hr, abs_y_hr, abs_nu_hr)
            
               
            
        #loop through hr absorptions
        for abs_index, (abs_nu_hr, abs_y_hr) in enumerate(zip(abs_nu_hrs, abs_y_hrs)):
            if abs_y_hr < abs_y_cutoff:
                # print("Checking line at %0.3f" %abs_nu_hr)
                
                error = False

                
                """use cal in the file to find approx pixel range of absorption line"""
                # #find local minima close to the simulated lines
                mol_index = get_nearest_index(abs_nu_hr, spectrum_nu)
                
                # #define micro window a few pixels to either side of expected pixel
                local_indices_to_check = np.arange(mol_index - 3, mol_index + 4, 1)
                
                # #avoid lines at edge of detector
                if min(local_indices_to_check) > 8 and max(local_indices_to_check) < len(spectrum_nu) - 8:
                    
                    #find local minima in pixels close to expected location of absorption line
                    local_minimum_index = get_local_minima(spectrum[local_indices_to_check]) + local_indices_to_check[0]
                    
                    if len(local_minimum_index) == 1:
                        #if just one local minima found in spectrum micro window, centre indices on that
                        local_minimum_indices = np.arange(local_minimum_index - 1, local_minimum_index + 2, 1) #just take 1 point +- minimum
                        
                        #find pixel number minimum
                        # x_hr, y_hr, px_min_position, chisq = fit_gaussian_absorption(px[local_minimum_indices], y_cr[local_minimum_indices], error=True)
                        x_hr, y_hr, px_min_position = fit_polynomial(spectrum_px[local_minimum_indices], spectrum[local_minimum_indices], error=False)
                    
                        if len(x_hr) > 0: #if no fitting error
                        
                            abs_depth = np.min(y_hr)
                            abs_max = np.max(y_hr)

                            if abs_depth < 0.99:
                                if abs_index == 0:
                                    label = "Gaussian fit to absorption bands"
                                else:
                                    label = ""
    
    
                                #convert pixel minimum and x_hr fit wavenumbers
                                nu_min_position = np.interp(px_min_position, spectrum_px, spectrum_nu)
                                nu_min_position_no_offset = np.interp(px_min_position, spectrum_px, spectrum_nu_no_offset)
                                delta_nu = nu_min_position - abs_nu_hr
                                delta_nu_no_offset = nu_min_position_no_offset - abs_nu_hr
    
                                if np.abs(delta_nu) > 0.2 or np.abs(delta_nu) > 0.2:
                                    print("Ignoring %0.1f line, too far from expected value" %abs_nu_hr)
                                    error = True
     
                                
                                else: #all test passed
                                    print("Single line found in data: px=%0.3f, delta nu=%0.3f" %(px_min_position, delta_nu_no_offset))
                                    #plot
                                    if PLOT_SPECTRA:
                                        ax1.axvline(x=nu_min_position, color="C2", alpha=0.3)
                                        x_hr_nu = np.interp(x_hr, spectrum_px, spectrum_nu)
                                        ax1.plot(x_hr_nu, y_hr, color="C1", linestyle=":", label=label)
                            else:
                                # print("%s: absorption too small" %(h5), abs_depth)
                                error=True

                        else:
                            # print("%s: fitting failed" %(h5))
                            error = True
                    else:
                        # print("%s: %i lines found" %(h5, len(local_minimum_index)))
                        error = True
        
                else:
                    # print("%s: line too close to detector edge" %(h5), local_indices_to_check)
                    error = True
                        

                if not error:
                    text = "%s\t%i\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" \
                        %(h5, order, spectrum_t, abs_nu_hr, px_min_position, delta_nu_no_offset, abs_depth, abs_max) 
                    write_log(text)



    ax1.set_title("LNO occultation fullscan spectral calibration fitting order %i" %order)
    ax1.grid()
    ax1.set_xlabel("Pixels")
    ax1.set_ylabel("Continuum-removed transmittance")
    fig1.savefig("LNO_occultation_fullscan_spectral_calibration_fitting_order_%i.png" %order)

# ax2.plot(spectra_x[lowest_altitude_ix, :], spectrum_norm, color=order_dict[order])
# ax2.grid()
# ax2.set_title("LNO occultation fullscan after spectral calibration")
# ax2.set_xlabel("Wavenumber cm-1")
# ax2.set_ylabel("Continuum-removed transmittance")
       
