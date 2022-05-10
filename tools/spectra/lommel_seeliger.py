# -*- coding: utf-8 -*-
"""
Created on Tue May  3 20:52:08 2022

@author: iant

LOMMEL SEELIGER
https://articles.adsabs.harvard.edu/full/2005JRASC..99...92F

"""

import numpy as np
from matplotlib import pyplot as plt

i = np.arange(0, 90., 10)
e = np.arange(90.)

plt.figure()
for ii in i:
        mu = np.cos(ii * (np.pi / 180.))

        mu0 = np.cos(e * (np.pi / 180.))
        
        w0 = 1.
        
        f = (w0 / (4. * np.pi)) * (1. / (mu + mu0))
        
        plt.plot(e, f, label="i=%.0f" %ii)
        # plt.scatter(ei)
plt.xlabel("e")
plt.ylabel("Lommel-Seeliger reflectance")
plt.legend()