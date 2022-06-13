# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:43:25 2020

@author: iant
"""




def nu_hr_grid(diffraction_order, adj_orders, instrument_temperature):
    """make high res wavenumber grid for given diffraction order +- n adjacent orders"""
    import numpy as np
    from instrument.nomad_lno_instrument_v01 import nu_mp
    
    nu_hr_min = nu_mp(diffraction_order - adj_orders, 0, instrument_temperature) - 5.
    nu_hr_max = nu_mp(diffraction_order + adj_orders, 320., instrument_temperature) + 5.
    dnu = 0.001
    Nbnu_hr = int(np.ceil((nu_hr_max-nu_hr_min)/dnu)) + 1
    nu_hr = np.linspace(nu_hr_min, nu_hr_max, Nbnu_hr)
    dnu = nu_hr[1]-nu_hr[0]
    return nu_hr, dnu
