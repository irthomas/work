# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:08:21 2022

@author: iant


READ GEM MARS DATA FROM WEBSERVER

E.G. https://gem-mars.aeronomie.be/vespa-gem?myear=35&lat=-56.5&lon=111.4&ls=123.4&lst=0.6
"""
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

import matplotlib.pyplot as plt




DUST_STORM_YEARS = [34]
DEFAULT_NON_STORM_YEAR = 35




def get_mars_year_ls(dt):
    """get mars year and ls from a UTC datetime without loading SPICE kernels
    Calculation approximation from Piqueux et al. 2015 http://dx.doi.org/10.1016/j.icarus.2014.12.014"""

    j2000 = datetime(2000, 1, 1, 12)
    dpr = 57.29577951308232

    my_offset = 24.0 #Mars year at J2000 epoch
    
    
    delta_days_j2000 = (dt - j2000).days + (dt - j2000).seconds / 3600. / 24.
    
    
    
    M = (19.38095 + 0.524020769 * delta_days_j2000) / dpr #mean anomaly
    
    
    ls_total = 270.38859 + 0.524038542 * delta_days_j2000 + 10.67848 * np.sin(M) + 0.62077 * np.sin(2 * M) + 0.05031 * np.sin(3 * M)
    
    divide = np.divmod(ls_total, 360.0) #get quotient and remainder
    
    #quotient is mars year at j2000 epoch. Add offset to get correct year    
    return [divide[0] + my_offset, divide[1]]





def get_gem_data(myear, ls, lat, lon, lst, plot=False):
    """get data from gem model via the BIRA-Vespa interface
    
    Inputs:
    myear: two Martian years are available, 34 (dust storm) or 35 (no dust storm)
    ls = Mars season
    lat = latitude (-90 to +90)
    lon = longitude. Values can be 0-360 or -180 to 180 East positive
    lst = local solar time
    plot = plot output?
    
    """

    #choose correct Martian year
    if int(myear) in DUST_STORM_YEARS:
        myear = 34
    else:
        myear = DEFAULT_NON_STORM_YEAR


    url = f"https://gem-mars.aeronomie.be/vespa-gem?myear={myear}&lat={lat:#0.2f}&lon={lon:#0.2f}&ls={ls:#0.2f}&lst={lst:#0.3f}"
    
    

    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    header_data = soup.find_all("field")
    table_data = soup.find_all("td")
    
    
    data = np.zeros(len(table_data))
    if len(table_data) == 0:
        print("Error getting GEM data")
    
    for i, element in enumerate(table_data):
        data[i] = np.float32(element.text)
        
    data = data.reshape(-1, len(header_data))
    
    atmos_dict = {element["id"].lower():data[:, i] for i, element in enumerate(header_data)}
    
    if plot:
        fig, axes = plt.subplots(figsize=(18, 5), ncols=8, constrained_layout=True)
        fig.suptitle("GEM-Mars: " + url.split("?")[1].replace("&", " "))
        
        axes_dict = {
            "co":{"ax":3, "log":False},
            "co2":{"ax":4, "log":False},
            "h20":{"ax":3, "log":False},
            "nd":{"ax":2, "log":True},
            "o2":{"ax":5, "log":False},
            "o3":{"ax":6, "log":False},
            "p":{"ax":0, "log":True},
            "t":{"ax":1, "log":False},
            "ext_dust":{"ax":7, "log":False},
            "ext_h2o_ice":{"ax":7, "log":False},
            }
        
        
        
        for header in atmos_dict.keys():
            if header in axes_dict:
                ax = axes_dict[header]["ax"]
                log = axes_dict[header]["log"]
                
                axes[ax].plot(atmos_dict[header], atmos_dict["z"], label=header.upper())
                if log:
                    axes[ax].set_xscale("log")
                axes[ax].grid()
                axes[ax].legend()
            
        axes[0].set_ylabel("Tangent Altitude (km)")
        
    return atmos_dict




def get_gem_data_from_h5(h5_f, tangent_altitude=50.0, index=None, plot=False):
    """Get gem data from the geometry of a hdf5 file.

    Inputs:
    h5_f: an open NOMAD hdf5 file from level 0.2a or above 
    index: the index of the spectrum where 0=first spectrum
        if index is not specified:
            if occultation file, get geometry at the tangent point corrsponding to the given tangent_altitude,
            if nadir, get geometry from minimum incidence angle
    plot: plot data?
    """

    if not index:
        if "TangentAltAreoid" in h5_f["Geometry/Point0"].keys():
            #get index closest to reference altitude
            alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
            index = np.abs(alts - tangent_altitude).argmin()
        
        else:
            incid = h5_f["Geometry/Point0/IncidenceAngle"][:, 0]
            index = np.argmin(incid)
            
            print(incid[index])
            
        
    
    lon = h5_f["Geometry/Point0/Lon"][index, 0]
    lat = h5_f["Geometry/Point0/Lat"][index, 0]
    lst = h5_f["Geometry/Point0/LST"][index, 0]
    
    dt_str = h5_f["Geometry/ObservationDateTime"][index, 0].decode()
    
    dt = datetime.strptime(dt_str, "%Y %b %d %H:%M:%S.%f")
    
    myear, ls = get_mars_year_ls(dt)
    
    return get_gem_data(myear, ls, lat, lon, lst, plot=plot)
    
    
    
    
# example run
# atmos_dict = get_gem_data(36, 180.0, 0.0, 0.0, 12.0, plot=True)

