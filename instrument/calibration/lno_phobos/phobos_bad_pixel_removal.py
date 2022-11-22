# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:35:45 2022

@author: iant

CHECK FOR BAD PIXELS IN LNO PHOBOS DETECTOR LINES

CHECK RECENT SOLAR LINE SCANS

"""

import numpy as np
import matplotlib.pyplot as plt

from tools.file.hdf5_functions import open_hdf5_file
from tools.plotting.colours import get_colours

h5 = "20220616_112436_1p0a_LNO_1_CF"

# order = 183



bad_pixels = {
    146:[188, 228, ],
    147:[121, 236, ],
    148:[294, ],
    149:[292, 310, 317, ],
    150:[301, ],
    151:[258, ],
    152:[],
    153:[5, 245, ],
    154:[129, 135, 178, ],
    155:[125, 232, 308, ],
    156:[78, 254, 294, ],
    157:[291, ],
}



phobos_row_starts = [146, 149, 152, 155] #146 to 157


h5_f = open_hdf5_file(h5)
y = h5_f["Science/Y"][...]
bins = h5_f["Science/Bins"][:, 0]
diffraction_orders = h5_f["Channel/DiffractionOrder"][:]

unique_bins = sorted(list(set(bins)))
unique_orders = sorted(list(set(diffraction_orders)))

# detector_rows = np.arange(146, 158)
# detector_rows = np.arange(146, 152)
detector_rows = np.arange(152, 158)
orders = np.arange(182, 186)




order_d = {order:{} for order in orders}

for order in orders:

    data_d = {row:{} for row in detector_rows}
    
    for detector_row in detector_rows:
    
    
        
        indices = np.where((bins == detector_row) & (diffraction_orders == order))[0]
        
        good_indices = indices[0:20]
        
        y_rows = y[good_indices]
        
        y_mean = np.mean(y_rows, axis=0)
        y_stdev = np.std(y_rows, axis=0)
        
        data_d[detector_row]["y_rows"] = y_rows
        data_d[detector_row]["y_mean"] = y_mean
        data_d[detector_row]["y_stdev"] = y_stdev
        
        data_d[detector_row]["deviation"] = np.zeros(320)
        
        
        x_fit = np.arange(len(good_indices))
        for px_ix in range(320):
            y_fit = y_rows[:, px_ix]
            polyfit = np.polyfit(x_fit, y_fit, 1)
            polyval = np.polyval(polyfit, x_fit)
        
            deviation = (y_fit - polyval) / polyval
            # deviation = y_fit - polyval
        
            deviation_mean = np.mean(deviation)
        
            data_d[detector_row]["deviation"][px_ix] = deviation_mean
    
        # plt.plot(x, y)
        # plt.plot(x, polyval)
        
        # stop()
        
        # y_mean = np.mean(y_bins[:, 160:240], axis=1)
        # plt.figure()
        # plt.plot(y_rows.T)
        
        # plt.figure()
        # plt.plot(y_stdev)
    
    
    # y_mean_stdev = np.mean(np.array([y_stdev_d[row] for row in y_stdev_d.keys()]), axis=0)
    # y_stdev_stdev = np.std(np.array([y_stdev_d[row] for row in y_stdev_d.keys()]), axis=0)
        
    
    
    # plt.plot(y_stdev_stdev)
    # plt.plot(y_mean_stdev)
    
    colours = get_colours(len(detector_rows))
    
    
    plt.figure()
    
    for row_ix, detector_row in enumerate(detector_rows):
        
        plt.plot(data_d[detector_row]["deviation"], label=detector_row, color=colours[row_ix])
        
        
    plt.legend()
    
    #scale ylimits
    ylim = plt.gca().get_ylim()
    ylim2 = [ylim[0], ylim[1] / 10.]
    
    plt.ylim(ylim2)
    
    # plt.yscale("log")
    
    
