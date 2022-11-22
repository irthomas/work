# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:16:21 2022

@author: iant

CONVERT PHOBOS TIF TO LOW RES 
"""

import numpy as np
import matplotlib.pyplot as plt


def convert_tiff():
    """convert 99Mb file into something manageable"""

    from PIL import Image
    from scipy.interpolate import interp2d
    phobos_path = r"C:\Users\iant\Downloads\Phobos_Viking_Mosaic_40ppd_DLRcontrol.tif"
    
    im = np.array(Image.open(phobos_path))
    x = np.linspace(-180., 180., num=14400)
    y = np.linspace(-90., 90., num=7200)
    
    interp_f = interp2d(x, y, im)
    
    x = np.arange(-180, 180, 0.5)
    y = np.arange(-90, 90, 0.5)
    
    im_lr = interp_f(x, y)
    
    plt.figure()
    plt.imshow(im)
    
    
    plt.figure()
    plt.imshow(im_lr)
    
    np.savetxt("Phobos_Viking_Mosaic_LowRes.txt", im_lr, fmt="%0.3f")
    
    im_saved = np.loadtxt("Phobos_Viking_Mosaic_LowRes.txt")
    
    plt.figure()
    plt.imshow(im_saved)
    
    
