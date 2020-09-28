# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:18:03 2020

@author: iant

CHECK PYTHON VERSION

"""


def python_version():
    
    import sys
    
    maj = sys.version_info[0]
    minor = sys.version_info[1]
    return maj + minor/10.0
