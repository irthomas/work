# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:18:01 2021

@author: iant
"""

import json



def write_json(filename, dictionary):
    
    
    for key in dictionary.keys():
        dictionary[key] = dictionary[key].tolist()
        
    json.dump(dictionary, open(filename, "w"))
    return 0


def read_json(filename):
    
    return json.loads(open(filename, "r").read())