#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:18:50 2022

@author: iant
"""


from tools.file.hdf5_functions import make_filelist
import matplotlib.pyplot as plt

from tools.file.write_log import write_log

import re

regex = re.compile("20180[45678].._.*_UVIS_I")
# regex = re.compile("20180[45678].._.*_UVIS_E")


h5_fs, h5s, _= make_filelist(regex, "hdf5_level_0p3k")


int_d = {30:[], 45:[], 75:[], 100:[]}

# print("Filename, Integration time")
for h5, h5_f in zip(h5s, h5_fs):
    
    aq = h5_f["Channel/AcquisitionMode"][0]
    it = h5_f["Channel/IntegrationTime"][0]
    
    if "UVIS_I" in regex.pattern:
        y0 = h5_f["Science/Y"][0, :]
    else:
        y0 = h5_f["Science/Y"][-1, :]
        
    x0 = h5_f["Science/X"][0, :]
    
    colours = {30:"r", 45:"g", 75:"b", 100:"k"}
    
    if aq == 0 and max(x0) > 600:
        plt.plot(x0, y0, color=colours[it], label="%s %ims" %(h5, it))
        
        int_d[it].append(h5)
    # print(h5, ",", , "ms")
    
plt.legend()

for it in int_d.keys():
    write_log("%ims integration time" %it)
    for f in sorted(int_d[it]):
        write_log(f)