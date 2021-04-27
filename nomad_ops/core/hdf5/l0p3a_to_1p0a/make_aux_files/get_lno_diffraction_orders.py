# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:25:12 2020

@author: iant
"""

def get_lno_diffraction_orders(aotf_frequencies):
    """get diffraction orders from array or list of aotf frequencies"""
    
    import numpy as np
    
    aotf_order_coefficients = np.array([3.9186850E-09, 6.3020400E-03, 1.3321030E+01])

    diffraction_orders_calculated = [np.int(np.round(np.polyval(aotf_order_coefficients, aotf_frequency))) for aotf_frequency in aotf_frequencies]
    #set darks to zero
    diffraction_orders = np.asfarray([diffraction_order if diffraction_order > 50 else 0 for diffraction_order in diffraction_orders_calculated])
    return diffraction_orders

