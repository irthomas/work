# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:03:53 2020

@author: iant
"""

def get_consecutive_indices(list_of_indices, n_consecutive_indices=1):
    """from a list of indices, make a list of lists where each list contains only consecutive indices. If there are less than n consecutive values, ignore"""
    indices_list = []
    sub_list = []
    prev_n = -1
    
    for n in list_of_indices:
        if prev_n+1 != n:            # end of previous subList and beginning of next
            if sub_list:              # if subList already has elements
                if len(sub_list)>n_consecutive_indices:
                    indices_list.append(sub_list)
                sub_list = []
        sub_list.append(n)
        prev_n = n
    
    if len(sub_list)>n_consecutive_indices:
        indices_list.append(sub_list)
    return indices_list

