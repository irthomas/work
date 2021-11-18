# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:49:27 2021

@author: iant

CONVERT SO BORESIGHT TO TGO FRAME

OLD VECTOR: -0.92189730, -0.38738526, 0.00616719

OLD SPICE ROTATION:
 so
M    = |0.0|  * |-67.20504912816348|  * |89.0879257751404|
 sc         Z                       Y                     X

"""

import spiceypy as sp
import numpy as np

from tools.spice.load_spice_kernels import load_spice_kernels

load_spice_kernels()

time = "2021 OCT 05"

et = sp.utc2et(time)

rot_mat = sp.pxform("TGO_NOMAD_SO", "TGO_SPACECRAFT", et)

v_tgo = np.dot(rot_mat, [1., 0., 0.])
print("Old")
print(v_tgo)

v_old = [-0.9218973, -0.38738526, 0.00616719]

rot_y_old = np.arcsin(v_old[0]) * sp.dpr()
rot_x_old = np.arctan(-1. * v_old[1] / v_old[2]) * sp.dpr()
print(0.0, rot_y_old, rot_x_old)

"""
OUTPUT:
tgo = [ 0.38743435 -0.92178049  0.01467479]


NOW REPLACE BORESIGHT VECTOR IN SPICE WITH NEW ROTATION ANGLES
"""

v = [-0.921772236, -0.387682744, 0.006167188]

rot_y = np.arcsin(v[0]) * sp.dpr()
rot_x = np.arctan(-1. * v[1] / v[2]) * sp.dpr()

print("New")
print(0.0, rot_y, rot_x)


"""
NEW SPICE ROTATION
0.0 -67.18656150381663 89.08862582246874

THEN RUN PXFORM AGAIN:

tgo = [ 0.3877317951, -0.9216556268, 0.01466153346]

"""