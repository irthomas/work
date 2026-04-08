# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:39:24 2026

@author: iant

SOMETIMES MAY NEED TO RERUN TWICE TO MAKE AND USE NEW KERNEL
"""

import spiceypy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import time

# make kernel
BAT_PATH = r"C:\Users\iant\Documents\DATA\soveniro_kernels\make_type17.bat"
subprocess.Popen(BAT_PATH)
time.sleep(3)

# eccentricity
rp = 2000 + 6052
ra = 10000 + 6052
e = (ra - rp) / (ra + rp)
a = (rp + ra) / 2
mu = GM = 6.67430e-11 * 4.86731e24
T = 2 * np.pi * (a * 1000)**1.5 / np.sqrt(mu) / 3600  # hours
DMPN_DT = 360 / (T * 3600)  # need this element!

sp.furnsh(r"C:\Users\iant\Documents\DATA\soveniro_kernels\kernels\mk\soveniro.tm")

et_start = sp.str2et("2026 JAN 02 00:00:00")
et_end = sp.str2et("2026 JAN 02 04:03:05.495")

xyzs = []
ets = np.concatenate((np.arange(et_start, et_end, 100), [et_end]))
print("Plotting from %s to %s" % (sp.et2utc(ets[0], "C", 3), sp.et2utc(ets[-1], "C", 3)))
for et in ets:
    pos = sp.spkpos("VENUS", et, "J2000", "NONE", "-222")[0]  # Venus at origin, Sov orbiting
    xyzs.append(pos)
print(xyzs[0] - xyzs[-1])

xyzs = np.asarray(xyzs)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

# Make data
r = 6052  # km
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z)

# Set an equal aspect ratio

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], c='blue', marker='o')

ax.view_init(elev=0, azim=90, roll=0)
ax.set_aspect('equal')

et = ets[0]
xyz = xyzs[0]

ven2sun_pos = sp.spkpos("VENUS", et, "J2000", "NONE", "SUN")[0]  # Venus at origin, direction to Sun
ven2sun_norm = ven2sun_pos / sp.vnorm(ven2sun_pos)

scaler = r * 5

# ax.plot([0, ven2sun_norm[0] * scaler], [0, ven2sun_norm[1] * scaler], [0, ven2sun_norm[2] * scaler], c="red")

# sov2sun_pos = sp.spkpos("-222", et, "J2000", "NONE", "SUN")[0]  # Sov at origin
# sov2ven_pos = sp.spkpos("-222", et, "J2000", "NONE", "VENUS")[0]
# npedln = sp.npedln(r, r, r, xyz, sun_pos)


# sp.kclear()
