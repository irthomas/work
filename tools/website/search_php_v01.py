# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:12:46 2018

@author: ithom

SEARCH IN FILES
"""

import os

SEARCH_STRING = "COM_NOMAD_"

PHP_DIRECTORY = os.path.normcase(r"W:\websites\prod\readonly\nomad")

os.chdir(PHP_DIRECTORY)
#list all py files

#fileList = []
#for fileName in os.listdir("."):
#    if fileName.endswith(".php"):
#        fileList.append(fileName)

found = False

for root, dirs, files in os.walk(PHP_DIRECTORY):
     for file in files:
         
        if file.endswith(".php") or file.endswith(".xml"):
            lineNumber = 0
            with open(os.path.join(root, file), "r", errors='ignore') as f:
                fileLines = f.readlines()
            
                for line in fileLines:
                    lineNumber += 1
                    
                    #check if string in line
                    if SEARCH_STRING.lower() in line.lower():
                        #if so, output name
                        print("%i:%s:%s" %(lineNumber,root+os.sep+file,line))
                        found = True

if not found:
    print("String not found in %s" %PHP_DIRECTORY)




