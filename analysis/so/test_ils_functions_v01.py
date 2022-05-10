# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:34:17 2019

@author: iant
"""

scalar = 1.0
xoffset = 0.0
width = retDict["sconv"] * 0.5


gaussian = (retDict["%i" %iord]["W_blaze"][pixelIndex]*retDict["dnu"])/(np.sqrt(2.*np.pi)*retDict["sconv"])*np.exp(-(retDict["nu_hr"]-retDict["%i" %iord]["nu_pm"][pixelIndex])**2/(2.*retDict["sconv"]**2))
gaussian2 = (retDict["%i" %iord]["W_blaze"][pixelIndex]*retDict["dnu"]*scalar)/(np.sqrt(2.*np.pi)*width)*np.exp(-(retDict["nu_hr"]-retDict["%i" %iord]["nu_pm"][pixelIndex]+xoffset)**2/(2.*width**2))

plt.figure()
plt.plot(gaussian)
plt.plot(gaussian2)

gaussian[(retDict["nu_hr"]-retDict["%i" %iord]["nu_pm"][pixelIndex])>=0] = gaussian2[(retDict["nu_hr"]-retDict["%i" %iord]["nu_pm"][pixelIndex])>=0]

plt.plot(gaussian)
