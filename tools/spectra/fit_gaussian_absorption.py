# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:52:46 2020

@author: iant

EXPONENTIAL FIT TO ABSORPTION BAND
"""



def fit_gaussian_absorption(x_in, y_in, error=False):
    """fit inverted gaussian to absorption band.
    Normalise continuum to 1 first"""
    import numpy as np
    from scipy.optimize import curve_fit

    def func(x, a, b, c, d):
        return 1.0 - a * np.exp(-((x - b)/c)**2.0) + d
    
    x_mean = np.mean(x_in)
    x_centred = x_in - x_mean
    popt, pcov = curve_fit(func, x_centred, y_in, p0=[0.1, 0.02, 0.25, 0.0])
    
    if error:
        y_fit = func(x_centred, *popt)
        chi_squared = np.sum(((y_in - y_fit) / y_fit)**2) #divide by yfit to normalise large and small absorption bands
        
    x_hr = np.linspace(x_in[0], x_in[-1], num=500)
    y_hr = func(x_hr - x_mean, *popt)
    
    min_index = (np.abs(y_hr - np.min(y_hr))).argmin()
    
    x_min_position = x_hr[min_index]
    
    if error:
        return x_hr, y_hr, x_min_position, chi_squared
    else:
        return x_hr, y_hr, x_min_position
    

#import matplotlib.pyplot as plt
#x = np.array([4275.39904406, 4275.50680047, 4275.61457115, 4275.7223561 ,
#       4275.83015532, 4275.93796881, 4276.04579656, 4276.15363859,
#       4276.26149488, 4276.36936544, 4276.47725027, 4276.58514937,
#       4276.69306274, 4276.80099038, 4276.90893228])
#y = np.array([0.99533129, 1.00310824, 0.99887751, 1.00014976, 0.9798266 ,
#       0.95434861, 0.92002429, 0.90238368, 0.90794792, 0.93271892,
#       0.9749434 , 0.9971337 , 1.00236682, 1.00519508, 1.0017545 ])
#    
#x_hr, y_hr, x_min_position = fit_gaussian_absorption(x, y)
#
#plt.figure()
#plt.plot(x, y, 'ko', label="Original Noisy Data")
#plt.plot(x_hr, y_hr, 'r-', label="Fitted Curve")
#plt.axvline(x=x_min_position)
#plt.legend()
#plt.show()

