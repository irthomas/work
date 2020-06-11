# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:46:38 2020

@author: iant
"""

import spiceypy as sp
#import numpy as np


SPICE_FORMATSTR = "C"
SPICE_PRECISION = 0


def et2utc(et):
    """function to convert et to utc if float is not -"""
    if et == "-":
        return "-"
    else:
#        if type(et) == np.bytes_: #for reading directly from 
#            et = float(et.decode())
        return sp.et2utc(et, SPICE_FORMATSTR, SPICE_PRECISION)

def utc2et(utc):
    """function to convert et to utc if float is not -"""
    if utc == "-":
        return "-"
    else:
        return sp.utc2et(utc)

