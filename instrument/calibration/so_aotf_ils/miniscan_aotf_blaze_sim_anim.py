# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:31:38 2021

@author: iant

PLOT JSONS OF SO MINISCAN SIMULATIONS
"""

import numpy as np
import os
import re


from tools.file.json import read_json
from tools.file.paths import paths

from tools.plotting.anim import make_line_anim

#1 khz 
# 20190223_054340_0p2a_SO_1_C
# 20190223_054340_0p2a_SO_2_C
# 20190223_061847_0p2a_SO_1_C
# 20190223_061847_0p2a_SO_2_C
# 20190416_020948_0p2a_SO_1_C

#order 194:
#1 khz
# 20190416_020948_0p2a_SO_1_C

#2 khz
# 20181129_002850_0p2a_SO_2_C

#4 khz
# 20181010_084333_0p2a_SO_2_C
# 20190416_024455_0p2a_SO_1_C

#8 khz
# 20190107_015635_0p2a_SO_2_C
# 20190307_011600_0p2a_SO_1_C
#



hdf5_filename = "20190307_011600_0p2a_SO_1_C"


filenames = sorted([f for f in os.listdir(os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename)) if re.match("so_solar_simulation_.*kHz.json", f)])

pixels = np.arange(320)

d = {"x":{}, "y":{}, "text":[], "text_position":[5,0.05], "xlabel":"Pixel", "ylabel":"", "xlim":[0, 319], "ylim":[0, 1]}
d["legend"] = {"on":True, "loc":"lower right"}
# d["keys"] = ["raw", "1_orders", "2_orders", "3_orders"]
d["title"] = hdf5_filename
d["filename"] = hdf5_filename

d["text"] = [re.match("so_solar_simulation_.*_[0-9][0-9][0-9][0-9]_(.*kHz).json", f).groups()[0] for f in filenames]

for i, filename in enumerate(filenames):
    # plt.figure()
    data = read_json(os.path.join(paths["ANIMATION_DIRECTORY"], hdf5_filename, filename))
    
    if i == 0:
        for key in data.keys():
            d["x"][key] = []
            d["y"][key] = []

    

    for key in data.keys():
        d["x"][key].append(pixels)
        d["y"][key].append(np.asfarray(data[key])/max(data[key]))
        
        # plt.plot(np.asfarray(data[i])/max(data[i]), label=i)
        
    # plt.legend()
    
make_line_anim(d)
