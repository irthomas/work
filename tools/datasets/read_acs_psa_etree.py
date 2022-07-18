# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:50:36 2022

@author: iant
"""

import os
import xml.etree.ElementTree as ET


ACS_DATA_PATH_ROOT = r"C:\Users\iant\Documents\DATA\psa\ACS"

ACS_DATA_PATH_CALIBRATED = os.path.join(ACS_DATA_PATH_ROOT, "data_calibrated", "Science_Phase", "Orbit_Range_5600_5699")

orbit_dir = "Orbit_5666"

xml_filename = "acs_cal_sc_nir_20190301T201042-20190301T201643-5666-1-2.xml"

xml_path = os.path.join(ACS_DATA_PATH_CALIBRATED, orbit_dir, xml_filename)

tree = ET.parse(xml_path)
root = tree.getroot()

