# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:41:53 2026

@author: iant
"""
import numpy as np
from shapely import Polygon, LineString, Point
import matplotlib.pyplot as plt
import math
from tools.general.progress_bar import progress
import sys

PLOT_TYPES = ["grating", "diffraction"]
# PLOT_TYPES = ["diffraction"]

# PLOT_TYPES = ["grating_only"]

N_RAYS = 1000
PLOT_EVERY = int(N_RAYS / 10)
WAVELS = [3e-3, 2.99e-3]
N_MIRR_POINTS = 20000
INCIDENCE_ANGLE = 75

if "grating" in PLOT_TYPES or "grating_only" in PLOT_TYPES:
    fig, ax = plt.subplots(figsize=(8, 8))
    lim = [-2, 10]

period = 1.0 / 4.033512  # in mm
depth = 0.05
Ngrooves = 59


blaze = 63.43  # degrees
angle_alpha = 2.60098  # degrees
angle_i = -0.019707  # degrees
t_minus_i = 63.449707  # degrees
t_plus_i = 63.410293  # degrees


coords = []
for n in range(Ngrooves):
    x0 = n * period
    coords.append((x0, -depth))
    coords.append((x0 + 0.01 * period, 0.0))

grat = LineString(coords)

if "grating" in PLOT_TYPES or "grating_only" in PLOT_TYPES:
    ax.plot(*grat.xy)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")

if "grating_only" in PLOT_TYPES:
    sys.exit()


theta_i = np.deg2rad(INCIDENCE_ANGLE)

y0 = 0.5
ray_origins = np.linspace(-2, (Ngrooves * period) - 2, N_RAYS)

intersection_points = []
reflection_dirs = []

rays = []
for x0 in ray_origins:
    ray = LineString([(x0, y0), (x0 + 10 * np.sin(theta_i), y0 - 10 * np.cos(theta_i))])
    rays.append(ray)


rays2 = []
for ray_ix, ray in enumerate(progress(rays)):
    inter = ray.intersection(grat)

    # print(inter.geom_type, inter.is_empty)

    if inter.is_empty:
        # print(ray_ix)
        continue

    if inter.geom_type == "Point":
        p = np.array(inter.coords[0])

    elif inter.geom_type == "MultiPoint":

        ps = np.asarray([[i.x, i.y] for i in inter.geoms])
        # find left-most point
        p = ps[ps[:, 0] == min(ps[:, 0])][0]

    rays2.append(LineString([(ray.xy[0][0], ray.xy[1][0]), (p[0], p[1])]))

    if "grating" in PLOT_TYPES:
        if np.mod(ray_ix, PLOT_EVERY) == 0:
            ax.scatter(*p)
    intersection_points.append(p)

    # stop()

    # Find local segment
    dmin = 1e9
    seg = None
    pts = list(grat.coords)

    for i in range(len(pts) - 1):
        s = LineString([pts[i], pts[i + 1]])
        d = s.distance(Point(p))
        if d < dmin:
            dmin = d
            seg = np.array([pts[i], pts[i + 1]])

    tangent = seg[1] - seg[0]
    tangent /= np.linalg.norm(tangent)

    # ax.plot([seg[0, 0], seg[0, 0] + tangent[0]], [seg[0, 1], seg[0, 1] + tangent[1]], "k--")

    normal = np.array([-tangent[1], tangent[0]])
    normal /= np.linalg.norm(normal)

    # ax.plot([p[0], p[0] + normal[0]], [p[1], p[1] + normal[1]], "k--")

    k_in = np.array([np.sin(theta_i), -np.cos(theta_i)])

    # ax.plot([p[0], p[0] + k_in[0]], [p[1], p[1] + k_in[1]], "k--")

    # reflected direction
    k_out = k_in - 2 * np.dot(k_in, normal) * normal
    reflection_dirs.append(k_out)

    if "grating" in PLOT_TYPES:
        if np.mod(ray_ix, PLOT_EVERY) == 0:
            # plot reflected ray
            ax.plot([p[0], p[0] + k_out[0]], [p[1], p[1] + k_out[1]], "k--")

intersection_points = np.array(intersection_points)


if "grating" in PLOT_TYPES:
    for ray in rays2[::PLOT_EVERY]:
        ax.plot(*ray.xy, "k--")


# if "grating" in PLOT_TYPES:
#     for p, d in zip(intersection_points, reflection_dirs):
#         ax.plot(
#             p[0], p[1],
#             d[0], d[1],
#         )

if "diffraction" in PLOT_TYPES:
    fig2, ax2 = plt.subplots(figsize=(8, 8))

for lam in WAVELS:
    k = 2 * np.pi / lam

    screen_x = np.linspace(-8, -7.8, N_MIRR_POINTS)
    screen_y = np.linspace(0, 2, N_MIRR_POINTS)

    field = np.zeros(screen_x.size, dtype=complex)

    for p in intersection_points:

        r = np.sqrt((screen_x - p[0])**2 + (screen_y - p[1])**2)
        field += np.exp(1j * k * r) / np.sqrt(r)

    intensity = np.abs(field)**2
    intensity /= intensity.max()

    if "diffraction" in PLOT_TYPES:
        ax2.plot(screen_x, intensity)

if "grating" in PLOT_TYPES:
    ax.plot(screen_x, np.ones_like(screen_x) * screen_y)
