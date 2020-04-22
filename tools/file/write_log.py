# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:07:41 2018

@author: iant


WRITE_LOG

"""

"""function to append log file"""
def writeLog(lineToWrite):
    logFile=open(r"C:\Users\iant\Dropbox\NOMAD\Python\log.txt", 'a')
    logFile.write(lineToWrite+'\n')
    logFile.close()
