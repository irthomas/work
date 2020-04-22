# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:31:52 2018

@author: iant
"""
import spiceypy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


method = "Ellipsoid"
target = "MARS"
fixref = "IAU_MARS"
abcorr = "None"
obsrvr = "SUN"


spoint = sp.latrec(3390, -24.6 / sp.dpr(), 18.3 / sp.dpr())
et = sp.utc2et("2003 OCT 28, 00:00:00")

step = 60

for loop in range(100):
    time = et + step * loop
    time_string = sp.et2utc(time, "C", 0)
    sza = sp.ilumin(method, target, time, fixref, abcorr, obsrvr, spoint)[3] * sp.dpr()











def writeLog(file_name, lines_to_write):
    """function to append log file"""
#    global LOG_PATHS
    logFile = open(file_name+".csv", 'w')
    for line_to_write in lines_to_write:
        logFile.write(line_to_write+'\n')
    logFile.close()
#    print(line_to_write)




utcstring_start='2016 MAR 17 12:00:00'
utcstring_end='2017 JAN 17 12:00:00'
step=3600 #time delta (s)

utctime_start=sp.utc2et(utcstring_start)
utctime_end=sp.utc2et(utcstring_end)

total_seconds=utctime_end-utctime_start
nsteps=int(np.floor(total_seconds/step))


times=np.arange(nsteps) * step + utctime_start
time_strings=[sp.et2utc(time, "C", 0) for time in times]

observer="SUN"
target="-143"
ref = "J2000"
tgo_pos=np.asfarray([sp.spkpos(target, et, ref, "NONE", observer)[0] for et in times])

stop()
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

ax.plot(tgo_pos[:,0], tgo_pos[:,1], tgo_pos[:,2])

listToWrite = ["Ephemeris Time, UTC Time, J2000 X, J2000 Y, J2000 Z"]
for index, time_string in enumerate(time_strings):
    listToWrite.append("%0.1f, %s, %0.1f, %0.1f, %0.1f" %(times[index], time_string, tgo_pos[index,0], tgo_pos[index,1], tgo_pos[index,2]))

#writeLog("TGO_earth_mars_trajectory", listToWrite)

#limit=2.5e8
#ax.set_xlim((-1*limit,limit))
#ax.set_ylim((-1*limit,limit))
#ax.set_zlim((-1*limit,limit))
#
#ax.set_xlabel("J2000 Reference Frame (km)")
#ax.set_ylabel("J2000 Reference Frame (km)")
#ax.set_zlabel("J2000 Reference Frame (km)")


