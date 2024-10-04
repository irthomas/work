# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:47:40 2024

@author: iant

CHECK ACS SPICE KERNELS FOR LUCIO B
"""

import os
import spiceypy as sp

et = 778419115.6116189


PATH_TO_KERNELS = r"C:\Users\iant\Documents\DATA\local_spice_kernels"


sp.furnsh(os.path.join(PATH_TO_KERNELS, r"em16_tgo_sc_fsp_349_01_20220101_20240914_s20240827_v01.bc"))  # old file, place in root dir
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\fk\rssd0002.tf"))
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\fk\em16_tgo_v27.tf"))
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\fk\em16_tgo_ops_v03.tf"))
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\ik\em16_tgo_acs_v09.ti"))
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\lsk\naif0012.tls"))
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\pck\pck00010.tpc"))
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\pck\de-403-masses.tpc"))
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"em16_tgo_step_20240906.tsc"))  # old file, place in root dir
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\spk\mar097_20160314_20300101.bsp"))
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"em16_tgo_fsp_349_01_20240429_20241012_v01.bsp"))  # old file, place in root dir
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\spk\em16_tgo_struct_v05.bsp"))
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"em16_tgo_cog_349_01_20220101_20240914_v01.bsp"))  # old file, place in root dir
sp.furnsh(os.path.join(PATH_TO_KERNELS, r"kernels\spk\de432s.bsp"))


subpnt = sp.subpnt('NEAR POINT/ELLIPSOID', 'MARS', et, 'IAU_MARS', 'LT+S', '-143')

dist = sp.vnorm(subpnt[2])

print("TGO Mars ellipsoid distance is", dist, "km")

# Prints to console "TGO Mars ellipsoid distance is 429.48433583463213 km"
