# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:12:08 2021

@author: iant

INSTALL CURL
PLACE CONFIG.TXT IN PYTHON DIR
MAKE EMPTY TEST.TXT FILE
RUN TO GET SOLAR SPECTRUM
"""

import os
import numpy as np

file = "test.txt"
server = 'https://psg.gsfc.nasa.gov' # URL of PSG server
os.system('curl -s --data-urlencode file@config.txt %s/api.php > %s' % (server,file))
solar = np.genfromtxt(file); solar = np.flip(solar, 0)
