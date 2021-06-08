# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:18:01 2021

@author: iant
"""

import json
import re


def write_json(filename, dictionary):
    
    
    for key in dictionary.keys():
        if type(dictionary[key]) == list:
            dictionary[key] = dictionary[key]
        else:
            dictionary[key] = dictionary[key].tolist()
        
    json.dump(dictionary, open(filename, "w"))
    return 0


def read_json(filename):
    
    return json.loads(open(filename, "r").read())


def write_json_2(filename, dictionary, prec=4):
    
    print("Converting to list")
    for key in dictionary.keys():
        if type(dictionary[key]) == list:
            dictionary[key] = dictionary[key]
        else:
            dictionary[key] = dictionary[key].tolist()
    
    print("Making JSON string")
    d_string = json.dumps(dictionary)

    # find numbers with 8 or more digits after the decimal point
    pat = re.compile(r"\d+\.\d{5,}")
    def mround(match):
        return "{:.4f}".format(float(match.group()))
    
    print("Saving to file")
    # write the modified string to a file
    with open(filename, 'w') as f:
        f.write(re.sub(pat, mround, d_string))

    return 0



