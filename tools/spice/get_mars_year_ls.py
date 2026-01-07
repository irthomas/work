# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:51:05 2022

@author: iant

CONVERT UTC TO MARS YEAR AND LS WITHOUT USING SPICE
"""

from datetime import datetime
import numpy as np


def get_mars_year_ls(dt):
    """get mars year and ls from a datetime without loading SPICE kernels
    Calculation approximation from Piqueux et al. 2015 http://dx.doi.org/10.1016/j.icarus.2014.12.014"""

    j2000 = datetime(2000, 1, 1, 12)
    dpr = 57.29577951308232

    my_offset = 24.0  # Mars year at J2000 epoch

    delta_days_j2000 = (dt - j2000).days + (dt - j2000).seconds / 3600. / 24.

    M = (19.38095 + 0.524020769 * delta_days_j2000) / dpr  # mean anomaly

    ls_total = 270.38859 + 0.524038542 * delta_days_j2000 + 10.67848 * np.sin(M) + 0.62077 * np.sin(2 * M) + 0.05031 * np.sin(3 * M)

    divide = np.divmod(ls_total, 360.0)  # get quotient and remainder

    # quotient is mars year at j2000 epoch. Add offset to get correct year
    return [divide[0] + my_offset, divide[1]]
