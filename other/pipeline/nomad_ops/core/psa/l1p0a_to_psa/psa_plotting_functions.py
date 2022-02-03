# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:08:31 2020

@author: iant


PSA PLOTTING FUNCTIONS

"""
import numpy as np

# import io
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()



#figure sizes for generating browse products
FIG_X = 10
FIG_Y = 6


def plot_so_lno_occultation(channel_obs, hdf5_file, title, path, x, y):
    """return image of transmittances as bytesIO object for writing to zip file"""
    """colour of line denotes measurement altitude"""
    
    # x = hdf5_file["Science/X"][0, :]
    # y = hdf5_file["Science/Y"][...]
    y[np.isnan(y)] = -999.0 #replace all nans with negative values
    n_pixels = x.shape[-1]
    
    logger_out = ""
    
    #get indices of top of atmosphere and surface. Replace by first/last value if none found
    toa_indices = np.where(y[:, int(n_pixels/2)] < 0.99)[0]
    if len(toa_indices) > 0:
        toa_index = np.max(toa_indices)
    else:
        toa_index = len(y[:, int(n_pixels/2)])-1
        logger_out += "Top of atmosphere not found. "

    surface_indices = np.where(y[:, int(n_pixels/2)] > 0.01)[0]
    if len(surface_indices) > 0:
        surface_index = np.min(surface_indices)
    else:
        surface_index = 0
        logger_out += "Surface not found. "
        
    
    alts = np.mean(hdf5_file["Geometry/Point0/TangentAltAreoid"][...], axis=1)
    mean_lon = np.mean(hdf5_file["Geometry/Point0/Lon"][...])
    mean_lat = np.mean(hdf5_file["Geometry/Point0/Lat"][...])
    mean_lst = np.mean(hdf5_file["Geometry/Point0/LST"][...])
    mean_ls = np.mean(hdf5_file["Geometry/LSubS"][...])
    
    cmap = plt.get_cmap('Spectral_r')
    #normalise colour index between top of atmosphere and surface
    colours = np.zeros_like(alts)
    for i, alt in enumerate(alts):
        if i < surface_index:
            colours[i] = 0.0
        elif i > toa_index:
            colours[i] = 1.0
        else:
            colours[i] = (alt - alts[surface_index]) / alts[toa_index]

    fig, ax = plt.subplots(1, 1, figsize=(FIG_X, FIG_Y), constrained_layout=True)
    for i, (colour, y) in enumerate(zip(colours, y)):
        ax.plot(x, y, alpha=0.5, color=cmap(colour))
    plt.ylim(top=1.1)
    plt.ylabel("Transmittance")
    plt.xlabel("Wavenumbers (cm$^{-1}$)")
    plt.title("%s\nMean longitude: %0.2f$^\circ$E; latitude: %0.2f$^\circ$N; LST: %0.2f hours; L$_s$: %0.3f$^\circ$" %(title, mean_lon, mean_lat, mean_lst, mean_ls))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=alts[surface_index], vmax=alts[toa_index]))
    sm.set_array([]) #no idea why you have to do this for old matplotlib versions
    cbar = plt.colorbar(sm)
    cbar.set_label('Tangent altitude above areoid (km)', rotation=270, labelpad=30)
    
    plt.savefig(path)
    return logger_out



def plot_uvis_occultation(channel_obs, hdf5_file, title, path, x, y):
    """return image of transmittances as bytesIO object for writing to zip file"""
    """colour of line denotes measurement altitude"""
    
    # x = hdf5_file["Science/X"][0, :]
    # y = hdf5_file["Science/Y"][...]
    y[np.isnan(y)] = -999.0 #replace all nans with negative values
    n_pixels = x.shape[-1]

    logger_out = ""
    
    #get indices of top of atmosphere and surface. Replace by first/last value if none found
    toa_indices = np.where(y[:, int(n_pixels/2)] < 0.99)[0]
    if len(toa_indices) > 0:
        toa_index = np.max(toa_indices)
    else:
        toa_index = len(y[:, int(n_pixels/2)])-1
        logger_out += "Top of atmosphere not found. "

    surface_indices = np.where(y[:, int(n_pixels/2)] > 0.01)[0]
    if len(surface_indices) > 0:
        surface_index = np.min(surface_indices)
    else:
        surface_index = 0
        logger_out += "Surface not found. "

    alts = np.mean(hdf5_file["Geometry/Point0/TangentAltAreoid"][...], axis=1)
    mean_lon = np.mean(hdf5_file["Geometry/Point0/Lon"][...])
    mean_lat = np.mean(hdf5_file["Geometry/Point0/Lat"][...])
    mean_lst = np.mean(hdf5_file["Geometry/Point0/LST"][...])
    mean_ls = np.mean(hdf5_file["Geometry/LSubS"][...])
    
    cmap = plt.get_cmap('Spectral_r')
    #normalise colour index between top of atmosphere and surface
    colours = np.zeros_like(alts)
    for i, alt in enumerate(alts):
        if i < surface_index:
            colours[i] = 0.0
        elif i > toa_index:
            colours[i] = 1.0
        else:
            colours[i] = (alt - alts[surface_index]) / alts[toa_index]

    fig, ax = plt.subplots(1, 1, figsize=(FIG_X, FIG_Y), constrained_layout=True)
    for i, (colour, y) in enumerate(zip(colours, y)):
        ax.plot(x, y, alpha=0.5, color=cmap(colour))
    plt.ylabel("Transmittance")
    plt.xlabel("Wavelength (nm)")
    plt.title("%s\nMean longitude: %0.2f$^\circ$E; latitude: %0.2f$^\circ$N; LST: %0.2f hours; L$_s$: %0.3f$^\circ$" %(title, mean_lon, mean_lat, mean_lst, mean_ls))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=alts[surface_index], vmax=alts[toa_index]))
    sm.set_array([]) #no idea why you have to do this for old matplotlib versions
    cbar = plt.colorbar(sm)
    cbar.set_label('Tangent altitude above areoid (km)', rotation=270, labelpad=30)
    
    plt.savefig(path)
    return logger_out





def plot_lno_uvis_nadir(channel_obs, hdf5_file, title, path, x, y):
    """return image of radiances as bytesIO object for writing to zip file
    plots image grid where y axis = latitude and x axis = spectral dimension"""
    
    if channel_obs in ["uvis_nadir"]:
        aspect_scalar = 3.0
        x_label = "Wavelength (nm)"
        y_units = "Radiance (W/m$^{2}$/sr/nm)"
        limits = [np.max((np.min(y), 0.0)), np.max(y)] #min value = 0 or minimum (whichever is larger)
    elif channel_obs in ["lno_nadir"]:
        aspect_scalar = 15.0
        x_label = "Wavenumbers (cm$^{-1}$)"
        y_units = "Reflectance factor 0.0-0.6 range (no units)"
        limits = [0.0, 0.6]

    
    mean_lon = np.mean(hdf5_file["Geometry/Point0/Lon"][...])
    mean_ls = np.mean(hdf5_file["Geometry/LSubS"][...])
    min_incid = np.min(hdf5_file["Geometry/Point0/IncidenceAngle"][...])
    logger_out = ""
    
    #plot latitude on the y axis
    lats = np.mean(hdf5_file["Geometry/Point0/Lat"][...], axis=1)
    indices = np.arange(0, len(lats), int(np.ceil(len(lats)/20.0)))
    y_label_list = ["%0.1f" %lat for lat in lats[indices[::-1]]]
    
    cmap = plt.get_cmap('Spectral_r')
    
    #calculate good aspect ratio
    aspect_ratio = y.shape[1] / y.shape[0] / aspect_scalar
    
    fig, ax = plt.subplots(1, 1, figsize=(FIG_X, FIG_Y), constrained_layout=True)
    plot = ax.imshow(y, cmap=cmap, aspect=aspect_ratio, extent=(x[0], x[-1], 0, len(lats)), vmin=limits[0], vmax=limits[1])

    ax.set_yticks(indices)
    ax.set_yticklabels(y_label_list)
    ax.set_ylim((0, len(lats)))
    ax.annotate("Observation start", (0.35, 0.95), xycoords="axes fraction", fontsize=14)
    ax.annotate("Observation end", (0.35, 0.01), xycoords="axes fraction", fontsize=14)
    ax.set_ylabel("Latitude (degrees)")
    ax.set_xlabel(x_label)
    ax.set_title("%s\nMean longitude: %0.2f$^\circ$E; L$_s$: %0.3f$^\circ$; minimum solar incidence angle: %0.2f$^\circ$" %(title, mean_lon, mean_ls, min_incid))

    cbar = fig.colorbar(plot)
    cbar.set_label(y_units, rotation=270, labelpad=20)

    plt.savefig(path)
    return logger_out




    
#import h5py
#import os
#
#filename = "20180422_001650_1p0a_SO_A_E_167"
#filename = "20180422_001650_1p0a_UVIS_E"
#filename = "20180422_003456_1p0a_LNO_1_D_167"
##filename = "20180422_003456_1p0a_UVIS_D"
##filename = "20180522_021448_1p0a_UVIS_D"
##
#hdf5_file = h5py.File(os.path.join(r"C:\Users\iant\Dropbox\NOMAD\Python\output\psa", filename+".h5"), "r")
#title = "nmd_cal_sc_so_20180422T001650-20180422T003123-a-e-167"
#
#if "SO" in filename:
#    plot_so_lno_occultation(hdf5_file, title, to_buffer=False)
#elif "UVIS_E" in filename or "UVIS_I" in filename:
#    plot_uvis_occultation(hdf5_file, title, to_buffer=False)
#elif "LNO" in filename:
#    plot_lno_uvis_nadir_radiance("lno_nadir", hdf5_file, title, to_buffer=False)
#elif "UVIS_D" in filename:
#    plot_lno_uvis_nadir_radiance("uvis_nadir", hdf5_file, title, to_buffer=False)


