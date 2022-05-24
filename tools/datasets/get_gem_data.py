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


from tools.spice.get_mars_year_ls import get_mars_year_ls



DUST_STORM_YEARS = [34]
DEFAULT_NON_STORM_YEAR = 35

def get_gem_data(myear, ls, lat, lon, lst, plot=False):
    """get data from gem model vespa interface
    Two Martian years are available, 34 (dust storm) and 35 (no dust storm)
    Lons can be given from 0-360 or -180 to 180 East positive"""

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
        plt.figure(constrained_layout=True)
        plt.title(url.split("?")[1].replace("&", " "))
        plt.grid()
        
        
        for header in atmos_dict.keys():
            if header != "z":
                plt.plot(atmos_dict[header]/np.max(atmos_dict[header]), atmos_dict["z"], label=header)
                # plt.plot(atmos_dict[header], atmos_dict["z"], label=header)
                
        plt.legend()
        # plt.xscale("log")
        
    return atmos_dict






def get_gem_data_from_h5(h5_f, reference_altitude=50.0, index=None, plot=False):
    """Get gem data from the geometry of a hdf5 file.
    If index not specified:
    If occultation file, get geometry at the tangent point corrsponding to the reference altitude,
    else get geometry from minimum incidence angle
    """

    if not index:
        if "TangentAltAreoid" in h5_f["Geometry/Point0"].keys():
            #get index closest to reference altitude
            alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
            index = np.abs(alts - reference_altitude).argmin()
        
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
    
    
    
    
    
