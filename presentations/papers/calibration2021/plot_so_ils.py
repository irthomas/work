# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:22:26 2020

@author: iant
"""

import numpy as np
import matplotlib.pyplot as plt
# from instrument.nomad_so_instrument import nu_mp

pixels = np.arange(320.)
m = 149.0


a1s = 0.0548001 + pixels*(-0.000121804)

a2s = 0.0627182 + pixels*(0.0000759729)

a3s = 0.920421 + pixels*(-0.000192975)

a4s = -0.0414220 + pixels*(-0.000395962)

a5s = -0.0133858 + pixels*(0.000676179)

a6s = 0.683544 + pixels*(-0.00169758)



#order 149 nominal:
t = 0.0
# p0 = t_p0(t)
nu = np.arange(-1., 1, 0.01)


plt.figure(figsize=(8,4))
for i, colour in zip([240, 200, 160, 120], ["yellow", "greenyellow", "aquamarine", "tab:blue"]): 
    a1 = a1s[i]
    a2 = a2s[i]
    a3 = a3s[i]
    a4 = a4s[i]
    a5 = a5s[i]
    a6 = a6s[i]
    
    ils0 = a3 * np.exp(-0.5 * ((nu - a1) / a2) ** 2)
    
    ils1 = a6 * np.exp(-0.5 * ((nu - a4) / a5) ** 2)
    
    ils = ils0 + ils1 

    plt.plot(nu, ils, label="Pixel %i" %i, linewidth=3, color=colour)
    
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.savefig('so_ils.png', transparent=True)
