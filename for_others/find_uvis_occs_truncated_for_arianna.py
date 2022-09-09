# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:21:37 2022

@author: iant
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:18:50 2022

@author: iant
"""

import re

# import matplotlib.pyplot as plt

from tools.file.hdf5_functions import make_filelist, open_hdf5_file
# from tools.file.write_log import write_log
from tools.general.progress_bar import progress_bar



# regex = re.compile("20180[45678].._.*_UVIS_[IE]")
regex = re.compile("20......_.*_UVIS_[IE]")


_, h5s, _= make_filelist(regex, "hdf5_level_1p0a", open_files=False)

truncated = {}

# print("Filename, Integration time")
for h5 in progress_bar(h5s):
    
    h5_f = open_hdf5_file(h5)

    
    x0 = h5_f["Science/X"][0, -1]
    
    if x0 not in truncated.keys():
        truncated[x0] = []
        
    truncated[x0].append(h5)
    
for x0 in truncated.keys():
    if x0 < 650.:
        print("All files where last detector column is %0.3fnm:" %x0)
        
        for h5 in truncated[x0]:
            print(h5)