# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:50:48 2022

@author: iant

CHECK LOIC VS GODDARD SPECTRAL CALIBRATION AND FIND WHICH ONE MATCHES DATA THE BEST
"""
import re
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd

from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning



from tools.spectra.molecular_spectrum_so import get_molecular_hr
from tools.spectra.baseline_als import baseline_als
# from tools.spectra.fit_gaussian_absorption import fit_gaussian_absorption

from tools.plotting.colours import get_colours

from tools.general.get_minima_maxima import get_local_minima
from tools.general.get_nearest_index import get_nearest_index

from tools.file.hdf5_functions import make_filelist

from tools.file.write_log import write_log



order_dict = {
    119:{"molecule":"CO2", "smin":1.0e-30, "n_lines":20},
    121:{"molecule":"CO2", "smin":1.0e-30, "n_lines":20},
    134:{"molecule":"H2O", "smin":1.0e-28, "n_lines":9},
    136:{"molecule":"H2O", "smin":1.0e-28, "n_lines":7},
    149:{"molecule":"CO2", "smin":1.0e-28, "n_lines":17},
    165:{"molecule":"CO2", "smin":1.0e-24, "n_lines":20},
    168:{"molecule":"H2O", "smin":1.0e-33, "n_lines":5},
    169:{"molecule":"H2O", "smin":1.0e-33, "n_lines":6},
    186:{"molecule":"CO", "smin":1.0e-33, "n_lines":8},
    189:{"molecule":"CO", "smin":1.0e-33, "n_lines":9},
    190:{"molecule":"CO", "smin":1.0e-33, "n_lines":10},
    191:{"molecule":"CO", "smin":1.0e-33, "n_lines":10},
    
}


chosen_bin = 3
chosen_y_mean = 0.5





def gd21_waven(order, inter_temp, px_in):
    """spectral calibration Goddard Oct 21"""
    cfpixel = [1.75128E-08, 5.55953E-04, 2.24734E+01]             # Blaze free-spectral-range (FSR) [cm-1 from pixel]
    ncoeff  = [-2.44383699e-07, -2.30708836e-05, -1.90001923e-04] # Relative frequency shift coefficients [shift/frequency from Celsius]
    
    
    xdat  = np.polyval(cfpixel, px_in) * order
    xdat += xdat  *np.polyval(ncoeff, inter_temp)
    return xdat


def lt21_waven(order, inter_temp, px_in):
    """spectral calibration Loic Feb 22"""

    cfpixel = [3.32e-8, 5.480e-4, 22.4701]

    p0_nu = -0.8276 * inter_temp #px/°C * T(interpolated)
    px_temp = px_in + p0_nu
    xdat  = np.polyval(cfpixel, px_temp) * order
    
    
    return xdat


# px = np.arange(n_px)
# plt.figure()
# for order in range(119, 195, 10):
#     for inter_temp in range(-10, 10, 2):
#         nu_p_gd = gd21_waven(order, inter_temp, px)
#         nu_p_lt = lt21_waven(order, inter_temp, px)
    
#         delta = nu_p_gd - nu_p_lt
#         plt.plot(delta, label="%i %0.1fC" %(order, inter_temp))
#         plt.legend()
# stop()



# def fit_gaussian_absorption(x_in, y_in, error=False, hr_num=500):
#     """fit inverted gaussian to absorption band.
#     Normalise continuum to 1 first"""

#     def func(x, a, b, c, d):
#         return 1.0 - a * np.exp(-((x - b)/c)**2.0) + d
    
#     x_mean = np.mean(x_in)
#     x_centred = x_in - x_mean
#     try:
#         popt, pcov = curve_fit(func, x_centred, y_in, p0=[0.1, 0.02, 0.25, 0.0])
#     except OptimizeWarning :
#         if error:
#             return [], [], [], []
#         else:
#             return [], [], []
#     except RuntimeError:
#         if error:
#             return [], [], [], []
#         else:
#             return [], [], []
    
#     if error:
#         y_fit = func(x_centred, *popt)
#         chi_squared = np.sum(((y_in - y_fit) / y_fit)**2) #divide by yfit to normalise large and small absorption bands
        
#     x_hr = np.linspace(x_in[0], x_in[-1], num=hr_num)
#     y_hr = func(x_hr - x_mean, *popt)
    
#     min_index = (np.abs(y_hr - np.min(y_hr))).argmin()
    
#     x_min_position = x_hr[min_index]
    
#     if error:
#         return x_hr, y_hr, x_min_position, chi_squared
#     else:
#         return x_hr, y_hr, x_min_position
    

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
    
    
    

from matplotlib.colors import LinearSegmentedColormap

redgreyblue = LinearSegmentedColormap.from_list('redgreyblue', (
    # Edit this gradient at https://eltos.github.io/gradient/#redgreyblue=0:0053FB-48:9EA0DC-50:989696-52:DC9E9F-100:FF000C
    (0.000, (0.000, 0.325, 0.984)),
    (0.480, (0.620, 0.627, 0.863)),
    (0.500, (0.596, 0.588, 0.588)),
    (0.520, (0.863, 0.620, 0.624)),
    (1.000, (1.000, 0.000, 0.047))))



def make_spectral_comparison_log(order_dict, chosen_bin, chosen_y_mean):

    write_log("Filename\tOrder\tTemperature\tAbsorption nu\tAbsorption pixel\tGD nu\tLT nu\tGD delta nu\tLT delta nu\tFitted depth\tFitted max\tFit Chi-squared")

    for order, order_info in order_dict.items():
        
        molecule = order_info["molecule"]
        smin = order_info["smin"]
        n_lines = order_info["n_lines"]
        
        n_px = 320.0 #consider whole detector
        file_level = "hdf5_level_1p0a"
    
    
    
        # regex = re.compile("202104.._......_.*_SO_A_[IE]_%i" %order)
        regex = re.compile("20(18|19|20|21)04.._......_.*_SO_A_[IE]_%i" %order)
        # regex = re.compile("20(18|19|20|21)0[4567].._......_.*_SO_A_[IE]_%i" %order)
    
        hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)
        
        
        colours = get_colours(len(hdf5_filenames))
        
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18,10), sharex=True, constrained_layout=True)
        fig.suptitle("Order %i" %order)
        for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5_files, hdf5_filenames)):
            
            colour = colours[file_index]
            
            
            y_all = hdf5_file["Science/Y"][:, :]
            y_mean = np.mean(y_all, axis=1)
        
            if "InterpolatedTemperature" not in hdf5_file["Channel"].keys():
                print("%s: Error temperatures not in file" %hdf5_filename)
                continue

            temperatures = hdf5_file["Channel/InterpolatedTemperature"][...]
            
            bins = hdf5_file["Science/Bins"][:, 0]
            unique_bins = sorted(list(set(bins)))
            
            bin_indices = np.where(bins == unique_bins[chosen_bin])[0]
            
            y_bin = y_all[bin_indices, :]
            y_mean_bin = y_mean[bin_indices]
        
            index = get_nearest_index(chosen_y_mean, y_mean_bin)
            
            y = y_bin[index, :]
            t = temperatures[index]
            
            y_cont = baseline_als(y)
            y_cr = y / y_cont
        
            px = np.arange(n_px)
        
            #for plotting use only Loic's code
            # x_gd = gd21_waven(order, t, px)
            x_lt = lt21_waven(order, t, px)
            
            # ax1.plot(x_gd, y_cr, color=colour, linestyle="--", label="%s GD" %hdf5_filename)
            ax1.plot(x_lt, y_cr, color=colour, linestyle="-", label="%s LT" %hdf5_filename)
        
            #do simulation only for first file
            if file_index == 0:
                nu_hr = gd21_waven(order, t, np.arange(-10.0, n_px + 10.0, 0.001))
                mol_hr = 1.0 - get_molecular_hr(molecule, nu_hr, Smin=smin)
                ax2.plot(nu_hr, mol_hr)
        
                abs_ix_hr = get_local_minima(mol_hr) #get hitran absorption minima indices
                abs_nu_hrs = nu_hr[abs_ix_hr] #get absorption nu
                abs_y_hrs = mol_hr[abs_ix_hr] #get absorption depth
            
            
                #N strongest lines
                abs_y_cutoff = sorted(abs_y_hrs)[n_lines] #select only the n strongest lines
        
                for abs_index, (abs_nu_hr, abs_y_hr) in enumerate(zip(abs_nu_hrs, abs_y_hrs)):
                    if abs_y_hr < abs_y_cutoff:
                        ax2.text(abs_nu_hr, abs_y_hr, abs_nu_hr)
                
               
            
            #loop through hr absorptions
            for abs_index, (abs_nu_hr, abs_y_hr) in enumerate(zip(abs_nu_hrs, abs_y_hrs)):
                if abs_y_hr < abs_y_cutoff:
                    # print("Checking line at %0.3f" %abs_nu_hr)
                    
                    error = False

                    
                    """use Loic's cal to find approx pixel range of absorption line"""
                    #find local minima close to the simulated lines
                    mol_gd_index = get_nearest_index(abs_nu_hr, x_lt)
                    
                    #define micro window a few pixels to either side of expected pixel
                    local_indices_to_check = np.arange(mol_gd_index - 3, mol_gd_index + 4, 1)
                    
                    #avoid lines at edge of detector
                    if min(local_indices_to_check) > 10 and max(local_indices_to_check) < n_px - 8:
                        
                        #find local minima in pixels close to expected location of absorption line
                        local_minimum_index = get_local_minima(y_cr[local_indices_to_check]) + local_indices_to_check[0]
                        
                        if len(local_minimum_index) == 1:
                            #if just one local minima found in spectrum micro window, centre indices on that
                            local_minimum_indices = np.arange(local_minimum_index - 1, local_minimum_index + 2, 1) #just take 1 point +- minimum
                            
                            #find pixel number minimum
                            # x_hr, y_hr, px_min_position, chisq = fit_gaussian_absorption(px[local_minimum_indices], y_cr[local_minimum_indices], error=True)
                            x_hr, y_hr, px_min_position, chisq = fit_polynomial(px[local_minimum_indices], y_cr[local_minimum_indices], error=True)
                        
                            if len(x_hr) > 0: #if no fitting error
                            
                                # print("Single line found in data at %0.3f, chisq %0.3f" %(px_min_position, chisq*1.0e6))
                                abs_depth = np.min(y_hr)
                                abs_max = np.max(y_hr)

                                if abs_index == 0:
                                    label = "Gaussian fit to absorption bands"
                                else:
                                    label = ""
                                    
                                """Goddard"""
                                #convert pixel minimum and x_hr fit wavenumbers
                                nu_min_position_gd = gd21_waven(order, t, px_min_position)
                                delta_nu_gd = nu_min_position_gd - abs_nu_hr

                                """Loic"""
                                #convert pixel minimum and x_hr fit wavenumbers
                                nu_min_position_lt = lt21_waven(order, t, px_min_position)
                                delta_nu_lt = nu_min_position_lt - abs_nu_hr

                                if np.abs(delta_nu_gd) > 0.3 or np.abs(delta_nu_lt) > 0.3:
                                    print("Ignoring %0.1f line, too far from expected value" %abs_nu_hr)
                                    error = True
 
                                
                                else:
                                    #plot
                                    # x_hr_nu_gd = gd21_waven(order, t, x_hr)
                                    x_hr_nu_lt = lt21_waven(order, t, x_hr)
                                    # ax1.axvline(x=nu_min_position_gd, color=colour)
                                    # ax1.plot(x_hr_nu_gd, y_hr, color=colour, linestyle=":", label=label)
                                    ax1.axvline(x=nu_min_position_lt, color=colour)
                                    ax1.plot(x_hr_nu_lt, y_hr, color=colour, linestyle=":", label=label)


                            else:
                                # print("%s: fitting failed" %(hdf5_filename))
                                error = True
                        else:
                            # print("%s: %i lines found" %(hdf5_filename, len(local_minimum_index)))
                            error = True
            
                    else:
                        # print("%s: line too close to detector edge" %(hdf5_filename))
                        error = True
                            

                    if not error:
                        text = "%s\t%i\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" \
                            %(hdf5_filename, order, t, abs_nu_hr, px_min_position, nu_min_position_gd, nu_min_position_lt, \
                              delta_nu_gd, delta_nu_lt, \
                              abs_depth, abs_max, chisq*1.0e6) 
                        write_log(text)
        
        
            hdf5_file.close()
            
            # abs_mean_gd = np.mean(np.abs(delta_nus_gd))
            # abs_mean_lt = np.mean(np.abs(delta_nus_lt))
            
            # # text = "%s\t%0.3f\t%0.3f" %(hdf5_filename, abs_mean_gd, abs_mean_lt) 
            
            # print(text)
            # write_log(text)
            
        # ax1.legend()
        plt.savefig("order_%i.png" %order)
        

        
make_spectral_comparison_log(order_dict, chosen_bin, chosen_y_mean)
        

df = pd.read_csv("log.txt", sep="\t")


temperature_range = np.arange(np.floor(min(df["Temperature"])), np.ceil(max(df["Temperature"])) + 0.25, 0.25)
px_range = np.arange(320)

df["temp_bins"] = pd.cut(df["Temperature"], temperature_range, labels=False)
df["pixel_bins"] = pd.cut(df["Absorption pixel"], px_range, labels=False)

df["GD-LT"] = np.abs(df["GD delta nu"]) - np.abs(df["LT delta nu"])

# grid = np.zeros((len(temperature_range), len(px_range))) + np.nan
grid_list = [[[] for _ in range(len(px_range))] for _ in range(len(temperature_range))]

for i, v in enumerate(df["GD-LT"]):
    # grid[df["temp_bins"][i], df["pixel_bins"][i]] = v
    if 0.8 < df["Fitted max"][i] < 1.1:
        if not ((df["GD delta nu"][i] < 0.02) & (df["GD delta nu"][i] > -0.02)):
            if not ((df["LT delta nu"][i] < 0.02) & (df["LT delta nu"][i] > -0.02)):
                grid_list[int(df["temp_bins"][i])][int(df["pixel_bins"][i])].append(v)

grid_median = np.zeros((len(temperature_range), len(px_range))) + np.nan

grid_counts = np.zeros((len(temperature_range), len(px_range)))
for i in range(len(px_range)):
    for j in range(len(temperature_range)):
        if len(grid_list[j][i]) > 0:
            grid_median[j, i] = np.median(grid_list[j][i])
        grid_counts[j, i] = len(grid_list[j][i])

# plt.figure()
# im = plt.pcolormesh(px_range, temperature_range, grid, cmap="coolwarm")
# plt.colorbar(im)

plt.figure(constrained_layout=True)
im = plt.pcolormesh(px_range, temperature_range, grid_median, cmap=redgreyblue, vmin=-0.1, vmax=0.1)
plt.xlabel("Pixel Number")
plt.ylabel("Interpolated Temperature")
cbar = plt.colorbar(im)
cbar.set_label("abs(delta GD)-abs(delta LT)\n(median if more than 1 value per grid point)", rotation=270, labelpad=20)
plt.title("Comparing spectral calibration methods: Loic vs Goddard")


# plt.figure(constrained_layout=True)
# im = plt.pcolormesh(px_range, temperature_range, grid_counts, cmap="binary")
# plt.xlabel("Pixel Number")
# plt.ylabel("Interpolated Temperature")
# cbar = plt.colorbar(im)
# cbar.set_label("Number of values per grid point", rotation=270, labelpad=20)
# plt.title("Comparing spectral calibration methods: Loic vs Goddard\nNumber of values per temperature-pixel bin")

plt.figure(constrained_layout=True)
plt.plot(np.sum(grid_counts, axis=1), temperature_range)
plt.xlabel("Number of observations at this temperature")
plt.ylabel("Interpolated Temperature")
plt.title("Comparing spectral calibration methods: Loic vs Goddard\nNumber of values per temperature-pixel bin")


# from scipy.interpolate import griddata
# points = np.asfarray([df["Absorption pixel"], df["Temperature"]]).T
# values = df["GD-LT"]
# grid_x, grid_y = np.mgrid[0:320:1, np.floor(min(df["Temperature"])):np.ceil(max(df["Temperature"])):0.001]
# grid2 = griddata(points, values, (grid_x, grid_y), method='linear')

# plt.figure(constrained_layout=True)
# im2 = plt.imshow(grid2.T, aspect="auto", cmap="winter", \
#                  extent=(min(df["Absorption pixel"]), max(df["Absorption pixel"]), min(df["Temperature"]), max(df["Temperature"])), \
#                  vmin=-0.1, vmax=0.1, interpolation=None)
# cbar2 = plt.colorbar(im2)
# cbar2.set_label("abs(delta GD)-abs(delta LT)", rotation=270, labelpad=20)
