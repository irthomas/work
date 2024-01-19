# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:25:44 2023

@author: iant

CONVERT MULTIPLE-YEAR LS/MARTIAN YEAR TO LS RANGE TICK LABELS

E.G. ['120', '180', '240', '300', 
      'MY35', '60', '120', '180', '240', '300', 
      'MY36', '60', '120', '180', '240', '300', 
      'MY37', '60', '120']

"""

import numpy as np


def make_ls_labels(lss, mys, delta_ls):

    my_start = np.min(mys)
    
    ls_range = lss + 360.0 * (mys - my_start)
    
    labels = np.arange(120.0, max(ls_range), delta_ls)
    
    
    label_strs = []
    for label in labels:
        mod = np.mod(label, 360.0)
        if mod == 0:
            label_strs.append("MY%0.f" %(label/360.0 + my_start))
        else:
            label_strs.append("%i" %mod)
    
    # print(label_strs)
    
    return ls_range, labels, label_strs