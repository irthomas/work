# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:04:53 2017

@author: iant
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

from pipeline_config_v03 import BASE_DIRECTORY,KERNEL_DIRECTORY,figx,figy
from pipeline_mappings_v03 import METAKERNEL_NAME
import spiceypy as sp
import os

abcorr="None"
tolerance="1"
method="Intercept: ellipsoid"
formatstr="C"
prec=3
shape="Ellipsoid"


os.chdir(KERNEL_DIRECTORY)
sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
print sp.tkvrsn("toolkit")
os.chdir(BASE_DIRECTORY)


ref="MARSIAU"
observer="MARS"
target="-143"
utcstring_start="2018NOV30-09:30:00 UTC" #normal and then long grazing occultations
utcstring_end="2018DEC01-09:00:00 UTC"
step=10 #time delta (s)

utctime_start=sp.str2et(utcstring_start)
utctime_end=sp.str2et(utcstring_end)

total_seconds=utctime_end-utctime_start
nsteps=int(np.floor(total_seconds/step))

times=np.arange(nsteps) * step + utctime_start

tgo_pos=np.transpose(np.asfarray([sp.spkpos(target,time_in,ref,abcorr,observer)[0] for time_in in list(times)]))






segments = np.zeros((len(times),2,3))
segments[0,0,0] = tgo_pos[0,0]
segments[0,0,1] = tgo_pos[1,0]
segments[0,0,2] = tgo_pos[2,0]
for index,(x,y,z) in enumerate(zip(list(tgo_pos[0,:-1:]),list(tgo_pos[1,:-1:]),list(tgo_pos[2,:-1:]))):
    segments[index,1,0] = x
    segments[index,1,1] = y
    segments[index,1,2] = z

    segments[index+1,0,0] = x
    segments[index+1,0,1] = y
    segments[index+1,0,2] = z
segments[-1,1,0] = tgo_pos[0,-1]
segments[-1,1,1] = tgo_pos[1,-1]
segments[-1,1,2] = tgo_pos[2,-1]


fig = plt.figure(figsize=(9,9))
ax = p3.Axes3D(fig)
limit=5e3
ax.set_xlim((-1*limit,limit))
ax.set_ylim((-1*limit,limit))
ax.set_zlim((-1*limit,limit))

#colour_list=[(255/255,215/255,0),(0,191/255,255/255),(0,0,255/255),(124/255,252/255,0),(34/255,139/255,34/255),(128/255,128/255,128/255),(0,0,0)]
#cmap = LinearSegmentedColormap.from_list("nomad", colour_list, N=6)
vmin=0
vmax=6
#cmap = ListedColormap(["gold","deepskyblue","deepskyblue","blue","blue","lawngreen","lawngreen","forestgreen","gray","black"])
cmap = ListedColormap(["gold","deepskyblue","blue","lawngreen","forestgreen","gray","black"])
#tgo_line = Line3DCollection(segments[0:10,:,:], array=nomad_obs_type, cmap=["gold","deepskyblue","blue","lawngreen","forestgreen","gray","black"])
tgo_line = Line3DCollection(segments[:2,:,:], array=nomad_obs_type, cmap = cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
#tgo_line.set_array(nomad_obs_type)
ax.add_collection(tgo_line)


def update_lines(num, tgo_line):
    global segments
    tgo_line.set_segments(segments[:num, :,:])
    return 0

line_ani = animation.FuncAnimation(fig, update_lines, len(tgo_pos[0]), fargs=(tgo_line,), interval=1, blit=False)
                                   
plt.show()    




