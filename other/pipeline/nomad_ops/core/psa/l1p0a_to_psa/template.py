# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:14:25 2021

@author: iant

TEMPLATE FUNCTIONS
"""

import logging
import os.path
import re
#import sys
#import h5py
import numpy as np
from lxml import etree
from datetime import datetime, timedelta

import os
#import re
import subprocess



from nomad_ops.core.psa.l1p0a_to_psa.config import \
    SO_OCCULTATION_TEMPLATE, LNO_NADIR_TEMPLATE, UVIS_OCCULTATION_TEMPLATE, UVIS_NADIR_TEMPLATE



def readPsaTemplate(psaTemplate):
    """read in levels, element names and tags from psa template file"""        
    tree = etree.parse(psaTemplate)
    firstLevel = len(tree.getpath(list(tree.iter())[0]).split("/"))
    elementList = [[len(tree.getpath(element).split(r"/"))-firstLevel,element.tag,str(element.text).replace(r"\n","").strip(),element.attrib] for element in list(tree.iter()) if str(element.xpath(r"local-name()")) != r""]
    return elementList



def read_psa_template_file(channel_obs):
    
    if channel_obs in ["so_occultation"]:
        psa_template_file = SO_OCCULTATION_TEMPLATE
    elif channel_obs in ["lno_nadir"]:
        psa_template_file = LNO_NADIR_TEMPLATE
    elif channel_obs in ["uvis_occultation"]:
        psa_template_file = UVIS_OCCULTATION_TEMPLATE
    elif channel_obs in ["uvis_nadir"]:
        psa_template_file = UVIS_NADIR_TEMPLATE
    else:
        psa_template_file = ""
        
    if psa_template_file == "":
        psa_template_element_list = ""
    else:
        psa_template_element_list = readPsaTemplate(psa_template_file)

    return psa_template_element_list


