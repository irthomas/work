# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:40:18 2020

@author: iant

FIND ALL PDF FILES

"""



import os
from datetime import datetime


PRESENTATION_DIRECTORY = r"W:\websites\prod\writable\nomad\ProjectDir\meetings\Team_telecons"
PROJ_DIR = r"ProjectDir/meetings/Team_telecons/"

filenames = reversed(sorted(os.listdir(PRESENTATION_DIRECTORY)))

new_datetime_string = ""

h = ""
for filename in filenames:
    if ".pdf" in filename:
        filename_split = filename.split("_")
        datetime_string = filename_split[3].replace(".pdf", "")
        year = datetime_string[0:2]
        month = datetime_string[2:4]
        day = datetime_string[4:6]
        
        filename_datetime = datetime.strptime(datetime_string, "%y%m%d")
        
        old_datetime_string = new_datetime_string[:]
        new_datetime_string = datetime.strftime(filename_datetime, "%d %B %Y")
        
        if old_datetime_string != new_datetime_string:
            h += "<br>\n"
            h += "<p><b>%s</b></p>\n" %new_datetime_string
            
        if ".pdf" in filename_split[3]:
            h += "<p><a href=\"%s\">%s - %s</a></p>\n" %(PROJ_DIR+filename, filename.replace(".pdf", ""), "Notes")
        else:
            h += "<p><a href=\"%s\">%s</a></p>\n" %(PROJ_DIR+filename, filename.replace(".pdf", ""))

        
print(h)