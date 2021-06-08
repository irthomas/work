# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:51:47 2021

@author: iant

PLOT IN 3D
"""


import matplotlib.pyplot as plt
import numpy as np

# Create a sphere
r = 3389.5
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)


#Set colours and render
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='orange', alpha=0.4, linewidth=0)


obs2mars_mars_spkpos_s = np.array([  960.66634138,  -339.05210988, -3617.85258172])
tan_pt_s = np.array([ -539.17741064,  -309.26950148, -3311.03400533])

fov_vec_ref_s = np.array([ 0.98742096, -0.01549128, -0.15747691]) * 3000
tgo2tangent_mars = -1 * np.array([1499.84375202,  -29.7826084 , -306.81857638])

angle = sp.vsep(fov_vec_ref_s, tgo2tangent_mars) * 180 / np.pi



for xyz in [obs2mars_mars_spkpos_s, tan_pt_s]:
    ax.scatter(xyz[0], xyz[1], xyz[2], marker="o")


ref = obs2mars_mars_spkpos_s
for xyz in [fov_vec_ref_s, tgo2tangent_mars]:
    line = [ref[0], ref[0]+xyz[0]], [ref[1], ref[1]+xyz[1]], [ref[2], ref[2]+xyz[2]]
    ax.plot(line[0], line[1], line[2])

ax.set_xlim((-4000,4000))
ax.set_ylim((-4000,4000))
ax.set_zlim((-4000,4000))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()