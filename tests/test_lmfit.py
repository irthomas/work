# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:02:41 2023

@author: iant

TEST LMFIT
"""


import numpy as np

from lmfit import minimize, Parameters



def poly(x, a, b, c):
   return a * x**2 + b * x + c


def func(params, x, y):
    """quadratic function"""
    
    a = params["a"]
    b = params["b"]
    c = params["c"]
    
    y_calc = poly(x, a, b, c)
    
    return y_calc - y



#make a polynomial
x = np.arange(-10., 11.)
y = poly(x, 6., 5., 8.)




params = Parameters()
params.add("a", value=4.)
params.add("b", value=4.)
params.add("c", value=4.)

out = minimize(func, params, args=(x, y), max_nfev=1000)

print("a=", out.params["a"].value)
print("b=", out.params["b"].value)
print("c=", out.params["c"].value)
print("number of iterations=", out.nfev)

# out = func(params, x, y)