# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:12:46 2018

@author: ithom

SEARCH IN ALL .PY FILES FOR A STRING
"""

import os
import glob
import posixpath

SEARCH_STRING = 'comparison of'


from tools.file.paths import paths

os.chdir(paths["BASE_DIRECTORY"])

#list all py files
file_list = glob.glob(posixpath.normcase(paths["BASE_DIRECTORY"])+ "/**/*.py", recursive=True)

excluded_strings = ["search_py.py", "old_scripts"]

    

found = False
    
#load .py files one by one
for file in file_list:
    
    #check for excluded strings
    if not any([string in file for string in excluded_strings]):

        line_number = 0
        with open(file, "r", errors='ignore') as f:
            file_lines = f.readlines()
        
            for line in file_lines:
                line_number += 1
                
                #check if string in line. Set lower to ignore case sensitivity
                if SEARCH_STRING.lower() in line.lower():
                    #if so, output name
                    print("%i:%s:%s" %(line_number,file,line))
                    found = True

if not found:
    print("String not found in %s" %paths["BASE_DIRECTORY"])




