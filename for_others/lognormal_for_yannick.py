# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 08:52:44 2020

@author: iant

LOG NORM
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

data = np.array([
[0.5, 0],
[1.5, 0],
[2.5, 150],
[3.5, 1481],
[4.5, 3962],
[5.5, 5403],
[6.5, 5456],
[7.5, 4527],
[8.5, 3670],
[9.5, 2828],
[10.5, 2081],
[11.5, 1584],
[12.5, 1123],
[13.5, 879],
[14.5, 720], 
[15.5, 502],
[16.5, 400],
[17.5, 305],
[18.5, 239],
[19.5, 188],
[20.5, 147],
[21.5, 118],
[22.5, 103],
[23.5, 88],
[24.5, 80],
[25.5, 56],
[26.5, 43],
[27.5, 47],
[28.5, 27],
[29.5, 39],
[30.5, 20],
[31.5, 18],
[32.5, 20],
[33.5, 16],
[34.5, 12],
[35.5, 20],
[36.5, 11],
[36.5, 15],
[38.5, 13],
[39.5, 7],
[40.5, 9],
[41.5, 10],
[42.5, 5],
[43.5, 8],
[44.5, 5], 
[45.5, 4],
[46.5, 9],
[47.5, 5],
[48.5, 4],
[49.5, 4],
])

x = data[:, 0]
y =data[:, 1]

def log_fit( x, a, mu, sigma ): #define function to fit - here it is the PDF of the log normal distribution
    return (a / x) * (1. / (sigma * np.sqrt( 2. * np.pi ) )) * np.exp( -( np.log( x ) - mu )**2 / ( 2. * sigma**2 ) )

#find best parameters based on a reasonable first guess
a, mu, sigma = scipy.optimize.curve_fit( log_fit, x, y, p0=[4.5, 2.5, 1.])[0]

mean = np.exp(mu + sigma**2 / 2.0) #find mean using equation from wikipedia
median = np.exp(mu) #find median

sample_dist_fit = log_fit(x, a, mu, sigma)
print(a, mu, sigma)
print(mean)
print(median)

plt.plot(x, y, label="Data")
plt.plot(x, sample_dist_fit, "r", label="Fit: mean=%0.2f, median=%0.2f" %(mean, median))

plt.legend()




