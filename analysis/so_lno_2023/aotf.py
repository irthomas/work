# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:02:14 2024

@author: iant

AOTF FUNCTION
"""

# import matplotlib.pyplot as plt
import numpy as np



def aotf_custom(nus, centres, heights, widths, super_gaussian=[], ax=None):
    #new aotf
    #13 coefficients + optional super gaussian for the central peak
    
    aotf = np.zeros_like(nus)
    for centre, height, width in zip(centres, heights, widths):
    
        dx = np.arange(-width, width, 0.01) + 0.00001
        
        if centre == 0.0 and len(super_gaussian) == 3:
            #overwrite central sinc2 with the super gaussian
            dx = np.arange(-width*1.75, width*1.75, 0.01) + 0.00001
            sup = super_gaussian[0] * np.exp(- 2.0 * (np.abs(dx - 0.0) / super_gaussian[1])**super_gaussian[2])
            if ax:
                ax.plot(dx+centre, sup, alpha=0.5, color="grey", ls="--")
        else:
            sinc2 = (width*np.sin(np.pi*dx/width)/(np.pi*dx))**2.0 * height
            if ax:
                ax.plot(dx+centre, sinc2, alpha=0.5, color="grey", ls="--")
        
        ixs = np.searchsorted(nus, dx+centre)
        
        if centre == 0.0 and len(super_gaussian) == 3:
            aotf[ixs] += sup
        else:
            aotf[ixs] += sinc2
        
    aotf /= np.max(aotf) #normalise
        
    if ax:
        ax.plot(nus, aotf, "k--")
        
    return aotf


"""for testing"""
# nus = np.arange(-150, 150, 0.01)

# centres = np.asarray([-51., -28.5, 0., 29., 52., 75.])
# heights = np.asarray([0.18, 0.4, 1.0, 0.45, 0.15, 0.08])
# widths = np.asarray([17.0, 17.0, 25.0, 17.0, 17.0, 17.0])

# super_gaussian = [1.0, 15.0, 2.8]

# fig1, ax1 = plt.subplots()

# aotf = new_aotf(nus, centres, heights, widths, super_gaussian=super_gaussian, ax=ax1)    
