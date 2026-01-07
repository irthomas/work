# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:45:48 2024

@author: iant

COMPARE DIFFERENT WAYS OF CALCULATING LS

"""


from tools.spice.get_mars_year_ls import get_mars_year_ls
from tools.spice.load_spice_kernels import load_spice_kernels
# from tools.general.progress_bar import progress

import spiceypy as sp
import numpy as np

from datetime import datetime, timedelta

from numpy import pi, floor, array, shape, cos, sin, ceil, arcsin, arccos, arange, abs
from collections import namedtuple

d2R = pi/180.


def getJD(iTime):
    """Get the Julian date in seconds"""
    offset = 2440587.5  # JD on 1/1/1970 00:00:00
    year = iTime[0]
    month = iTime[1]
    day = iTime[2]
    hour = iTime[3]
    minute = iTime[4]
    sec = iTime[5]
    date = datetime(year, month, day, hour, minute, sec)

    iTime = [1970, 1, 1, 0, 0, 0]
    year = iTime[0]
    month = iTime[1]
    day = iTime[2]
    hour = iTime[3]
    minute = iTime[4]
    sec = iTime[5]
    ref = datetime(year, month, day, hour, minute, sec)
    deltaTime = (date-ref)
    return deltaTime.total_seconds()/86400. + offset


def getJ2000(iTime):
    '''get date in J2000 epoch.'''
    jd = getJD(iTime)
    T = (jd - 2451545.0)/36525 if iTime[0] < 1972 else 0

    conversion = 64.184 + 59 * T - 51.2 * T**2 - 67.1 * T**3 - 16.4 * T**4

    # convert to Terrestrial Time
    jdTT = jd+(conversion/86400)

    return jdTT - 2451545.0


def getMarsParams(j2000):
    """Mars time parameters"""

    Coefs = array(
        [[0.0071, 2.2353, 49.409],
         [0.0057, 2.7543, 168.173],
         [0.0039, 1.1177, 191.837],
         [0.0037, 15.7866, 21.736],
         [0.0021, 2.1354, 15.704],
         [0.0020, 2.4694, 95.528],
         [0.0018, 32.8493, 49.095]])

    dims = shape(Coefs)
    # Mars mean anomaly:
    M = 19.3870 + 0.52402075 * j2000
    # angle of Fiction Mean Sun
    alpha = 270.3863 + 0.52403840*j2000

    # Perturbers
    PBS = 0
    for i in range(dims[0]):
        PBS += Coefs[i, 0]*cos(((0.985626 * j2000 / Coefs[i, 1]) + Coefs[i, 2])*d2R)
        # Equation of Center
        vMinusM = ((10.691 + 3.0e-7 * j2000)*sin(M*d2R) + 0.623*sin(2*M*d2R) + 0.050*sin(3*M*d2R) + 0.005*sin(4*M*d2R) + 0.0005*sin(5*M*d2R) + PBS)
    return M, alpha, PBS, vMinusM


def getMTfromTime(iTime):
    """Get Mars time information.

    param iTime: 6 element time list [y,m,d,h,m,s]
    returns: a named tuple containing the LS value as well as
         several parameters necessary for other calculations

    """

    DPY = 686.9713
    refTime = [1955, 4, 11, 10, 56, 0]  # Mars year 1
    rDate = getJD(refTime)
    thisTime = getJD(iTime)
    year = floor((thisTime - rDate)/DPY)+1

    j2000 = getJ2000(iTime)
    M, alpha, PBS, vMinusM = getMarsParams(j2000)

    LS = (alpha + vMinusM)

    while LS > 360:
        LS -= 360

    if LS < 0:
        LS = 360. + 360.*(LS/360. - ceil(LS/360.0))

    EOT = 2.861*sin(2*LS*d2R)-0.071*sin(4*LS*d2R)+0.002*sin(6*LS*d2R)-vMinusM

    MTC = (24*(((j2000 - 4.5)/1.027491252)+44796.0 - 0.00096)) % 24
    subSolarLon = ((MTC+EOT*24/360.)*(360/24.)+180) % 360
    solarDec = (arcsin(0.42565*sin(LS*d2R))/d2R+0.25*sin(LS*d2R))

    data = namedtuple('data', 'ls year M alpha PBS vMinusM MTC EOT subSolarLon solarDec')
    d1 = data(ls=LS, year=year, M=M, alpha=alpha, PBS=PBS, vMinusM=vMinusM, MTC=MTC, EOT=EOT,
              subSolarLon=subSolarLon, solarDec=solarDec)

    return d1


load_spice_kernels(planning=True)


start_dt = datetime(2018, 1, 1)
end_dt = datetime(2019, 1, 1)

delta = 3600.0 * 24.0
n_times = (end_dt - start_dt).total_seconds() / delta


dts = [start_dt + timedelta(seconds=delta*i) for i in np.arange(n_times)]
utcs = [datetime.strftime(dt, "%Y-%m-%dT%H:%M:%S") for dt in dts]

ets = [sp.utc2et(utc) for utc in utcs]
iTimes = [[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.day] for dt in dts]


def get_acs_lon(et):
    spkpos = sp.spkpos("MARS", et, "IAU_MARS", "NONE", "SUN")

    lon = -sp.reclat(spkpos[0][0:3])[1] * sp.dpr()
    lon -= 85.06

    return lon


print("Calculating Piqueux et al. 2015 MY Ls")
# https://www.lpl.arizona.edu/~shane/publications/piqueux_etal_icarus_2015.pdf
piq_mys = [get_mars_year_ls(dt) for dt in dts]
piq_mys = np.asarray(piq_mys)

print("Calculating SPICE Lspcn")
# https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/lspcn_c.html
spi_mys = [sp.lspcn("MARS", et, "NONE") * sp.dpr() for et in ets]
spi_mys = np.asarray(spi_mys)

print("Calculating Mars24 time")
# https://www.giss.nasa.gov/tools/mars24/help/algorithm.html
m24_mys = [getMTfromTime(iTime) for iTime in iTimes]
m24_mys = np.asarray(m24_mys)

print("Calculating ACS style Ls (not correct)")
# Lucio personal communication
acs_mys = [get_acs_lon(et) for et in ets]
acs_mys = np.asarray(acs_mys)

i = 0
print(piq_mys[i, 1], spi_mys[i], m24_mys[i][0], acs_mys[i])
