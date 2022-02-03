# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:11:12 2021

@author: iant

VALIDATION
"""

import logging
import re
import os
#import sys
#import h5py
# import numpy as np
# from datetime import datetime, timedelta
import subprocess


from nomad_ops.config import PFM_AUXILIARY_FILES
from nomad_ops.core.psa.l1p0a_to_psa.config import windows, \
    VALIDATE_WITH_ONLINE_DICTIONARY


logger = logging.getLogger( __name__ )


"""java validator"""
#JAVA_PDS_VALIDATOR = "validate-1.14.0"
# JAVA_PDS_VALIDATOR = "validate-1.20.0"
JAVA_PDS_VALIDATOR = "validate-1.24.0"
VALIDATOR_DIRECTORY = os.path.join(PFM_AUXILIARY_FILES, "psa", JAVA_PDS_VALIDATOR, "bin")
VALIDATOR_RES_DIRECTORY = os.path.join(PFM_AUXILIARY_FILES, "psa", JAVA_PDS_VALIDATOR, "resources")
CATALOG_NAME = "offline_catalog_2.0"
CONTEXT_PRODUCT_NAME = "local_context_products.json"



if windows:
    VALIDATE_COMMAND=[]
else:
    #if running on raspberrypi, no need to load java and set home
    if os.uname()[4] == "armv7l": 
        VALIDATE_COMMAND=[]
    else:
        VALIDATE_COMMAND=["/usr/bin/modulecmd bash load java/jdk-12.0",
                          "export JAVA_HOME=/bira-iasb/softs/opt/java/jdk-12.0.1"]
    


def check_validate_output(output_string):
    
    regex = re.compile("Summary:\W*(\d*) error\(s\)\W*(\d*) warning\(s\)")
    error_count, warning_count = re.findall(regex, output_string)[0]
    return int(error_count), int(warning_count)



def validate_data(xml_path, lid):
    validator_path = os.path.join(VALIDATOR_DIRECTORY, "validate")
    catalog_path = os.path.join(VALIDATOR_DIRECTORY, CATALOG_NAME)
    con_prod_path = os.path.join(VALIDATOR_RES_DIRECTORY, CONTEXT_PRODUCT_NAME)

    if VALIDATE_WITH_ONLINE_DICTIONARY:
        logger.info("Validating %s with online dictionary", lid)
        v_command = "%s %s.xml --add-context-products %s" %(validator_path, xml_path, con_prod_path)
    else:
        logger.info("Validating %s with offline dictionary", lid)
        v_command = "%s %s.xml --catalog %s.xml --add-context-products %s" %(validator_path, xml_path, catalog_path, con_prod_path)

    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output, err = process.communicate(v_command.encode('utf-8'))
    output_str = output.decode('utf-8')

    error_count, warning_count = check_validate_output(output_str)
    
    return error_count, output_str
