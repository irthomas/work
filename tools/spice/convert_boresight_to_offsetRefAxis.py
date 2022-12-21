# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:49:27 2021

@author: iant

CONVERT SO BORESIGHT TO THE TGO FRAME TO CALCULATE THE OFFSET REF AXIS (NECESSARY FOR LINESCAN/ORIENTATION PURPOSES)

"""

import spiceypy as sp
import numpy as np

from tools.spice.load_spice_kernels import load_spice_kernels

load_spice_kernels()

time = "2021 OCT 05"

et = sp.utc2et(time)



"""SO BORESIGHTS"""
print("### SO ###")


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








"""LNO BORESIGHTS - THIS PART ISN'T CORRECT!"""
print("### LNO ###")
print("Old")

rot_mat = sp.pxform("TGO_NOMAD_LNO_OPS_NAD", "TGO_SPACECRAFT", et)


v_tgo = np.dot(rot_mat, [1., 0., 0.])
print(v_tgo)

v = [-0.001047198, -0.9999786, 0.006457718]

rot_y = np.arcsin(v_old[0]) * sp.dpr()
rot_x = np.arctan(-1. * v[1] / v[2]) * sp.dpr()
print(0.0, rot_y, rot_x)

"""
OUTPUT:
tgo = [ 9.99999452e-01 -1.04717616e-03  6.76251309e-06]


NOW REPLACE BORESIGHT VECTOR IN SPICE WITH NEW ROTATION ANGLES
"""

new_vector = 0.3158360947251464
sp.pdpool("TKFRAME_-143311_ANGLES", [new_vector, 0.6000003670468607, 0.00000000000000000] )


print("New")


rot_mat = sp.pxform("TGO_NOMAD_LNO_OPS_NAD", "TGO_SPACECRAFT", et)


v_tgo = np.dot(rot_mat, [1., 0., 0.])
print(v_tgo)

v = [-0.001047203925760840, -0.999984258561257000, 0.005512349193501250]

rot_y = np.arcsin(v_old[0]) * sp.dpr()
rot_x = np.arctan(-1. * v[1] / v[2]) * sp.dpr()
print(0.0, rot_y, rot_x)




"""
NEW SPICE ROTATION
0.0 -67.18656150381663 89.08862582246874

THEN RUN PXFORM AGAIN:

tgo = [ 0.3877317951, -0.9216556268, 0.01466153346]

"""




