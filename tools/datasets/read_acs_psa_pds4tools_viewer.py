# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:16:18 2022

@author: iant

RUN THE GUI.
IN SPYDER THE GRAPHICS MUST BE TKINTER
TOOLS->PREFERENCES->IPYTHON CONSOLE->GRAPHICS->BACKEND
CHANGE BACK TO QT5 WHEN DONE
"""

import os
import pds4_tools

ACS_DATA_PATH_ROOT = r"C:\Users\iant\Documents\DATA\psa\ACS"

ACS_DATA_PATH_CALIBRATED = os.path.join(ACS_DATA_PATH_ROOT, "data_calibrated", "Science_Phase", "Orbit_Range_5600_5699")
ACS_GEOMETRY_PATH = os.path.join(ACS_DATA_PATH_ROOT, "geometry", "Science_Phase", "Orbit_Range_5600_5699")

orbit_dir = "Orbit_5666"

# xml_filename = "acs_cal_sc_nir_20190301T201042-20190301T201643-5666-1-2.xml"
# xml_filename = "acs_cal_sc_mir_20190301T184430-20190301T185948-5666-1-1.xml"
xml_filename = "acs_cal_sc_tir_20190301T190652-20190301T210431-5666-1-occ-009640.xml"

xml_path = os.path.join(ACS_DATA_PATH_CALIBRATED, orbit_dir, xml_filename)


"""Viewer GUI"""
pds4_tools.view(xml_path)
