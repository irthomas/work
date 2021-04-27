# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:54:15 2021

@author: iant

SPICE ERROR IN 2016 DATA
"""

import spiceypy as sp
from tools.spice.load_spice_kernels import load_spice_kernels

load_spice_kernels()

"... load spice kernels..."


# this works
utc = "2017 Mar 01 00:14:29.970279"
et = sp.utc2et(utc) # et = 541599339.1556617
# sp.spkpos(target, et, reference_frame, aberration_correction, observer)
spkpos = sp.spkpos("SUN", et, "TGO_NOMAD_SO", "None", "-143")


# # this doesn't work
utc = "2017 Feb 01 00:14:29.970279"
et = sp.utc2et(utc) # et 539180139.1550728
# sp.spkpos(target, et, reference_frame, aberration_correction, observer)
spkpos = sp.spkpos("SUN", et, "TGO_NOMAD_SO", "None", "-143")



"""
ERROR TEXT:
    
SpiceFILEOPENFAIL: 
================================================================================

Toolkit version: CSPICE66

SPICE(FILEOPENFAIL) --

Attempt to reconnect logical unit to file '../ck/em16_tgo_sc_spm_20161101_20170301_s20191109_v01.bc' failed. IOSTAT was 2.

spkpos_c --> SPKPOS --> SPKEZP --> SPKGPS --> REFCHG --> ROTGET --> CKFROT --> CKSNS --> DAFBBS --> DAFRFR --> ZZDAFGFR --> ZZDDHHLU

================================================================================

"""