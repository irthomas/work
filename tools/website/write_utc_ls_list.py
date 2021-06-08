# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:32:13 2021

@author: iant

WRITE TABLE OF UTC VS LS FOR WEBSITE


"""
import spiceypy as sp
from datetime import datetime, timedelta


def writeOutput(filename, lines_to_write):
    """function to write output to a file"""
    outFile = open("%s.txt" %filename, 'w')
    for line_to_write in lines_to_write:
        outFile.write(line_to_write+'\n')
    outFile.close()


def writeTimeLsToFile():
    """make list of time vs ls"""
    SPICE_TARGET = "MARS"
    SPICE_ABERRATION_CORRECTION = "None"
    
    DATETIME_FORMAT = "%d/%m/%Y %H:%M"
    
    
    
    linesToWrite = []
    datetimeStart = datetime(2018, 3, 1, 0, 0, 0, 0)
    for hoursToAdd in range(0, 24*31*12*10, 12): #10 years
        newDatetime = (datetimeStart + timedelta(hours=hoursToAdd)).strftime(DATETIME_FORMAT)
        ls = sp.lspcn(SPICE_TARGET, sp.utc2et(str(datetimeStart + timedelta(hours=hoursToAdd))), SPICE_ABERRATION_CORRECTION) * sp.dpr()
        linesToWrite.append("%s\t%0.1f" %(newDatetime, ls))
    
    writeOutput("Time_vs_Ls.txt", linesToWrite)
