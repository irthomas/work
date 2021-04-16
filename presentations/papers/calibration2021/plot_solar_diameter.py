# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:25:59 2021

@author: iant

APPARENT SOLAR DIAMETER
"""

import numpy as np
import numpy.linalg as la
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator


import spiceypy as sp

from tools.file.paths import FIG_X, FIG_Y

from tools.spice.load_spice_kernels import load_spice_kernels
# from tools.spice.datetime_functions import utc2et

load_spice_kernels(planning=True)

SAVE_FIGS = False

abcorr="None"

    
ref="J2000"
observer="-143" #observer
target = "SUN"


datetime_start = datetime(year=2018, month=4, day=21)
datetimes = [datetime_start + timedelta(days=x) for x in range(365*4)]

date_strs = [datetime.strftime(x, "%Y-%m-%d") for x in datetimes]
date_ets = [sp.str2et(x) for x in date_strs]
tgo_pos = np.asfarray([sp.spkpos(target, time, ref, abcorr, observer)[0] for time in list(date_ets)])



tgo_dist = la.norm(tgo_pos,axis=1)
code = sp.bodn2c(target)
pradii = sp.bodvcd(code, 'RADII', 3) # 10 = Sun
sun_radius = pradii[1][0]
sun_diameter_arcmins = np.arctan(sun_radius/tgo_dist) * sp.dpr() * 60.0 * 2.0



fig, ax = plt.subplots(figsize=(FIG_X, FIG_Y-1))

ax.plot_date(datetimes, sun_diameter_arcmins, linestyle="-", ms=0)


ax.set_xlabel("Date")
ax.set_ylabel("Solar diameter as seen from TGO (arcminutes)")
ax.set_title("Apparent diameter of Sun since start of TGO mission")


ax.xaxis.set_major_locator(MonthLocator(bymonth=None, interval=6, tz=None))    
fig.tight_layout()
if SAVE_FIGS:
    plt.savefig("sun_apparent_diameter.png")
   
