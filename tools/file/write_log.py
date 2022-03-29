# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:07:41 2018

@author: iant


WRITE_LOG

"""

#for compatibility with older scripts
def writeLog(lineToWrite, path=""):
    """function to append log file"""
    if path == "":
        logFile=open(r"C:\Users\iant\Dropbox\NOMAD\Python\log.txt", 'a')
    else:
        logFile=open(path, 'a')

    logFile.write(lineToWrite+'\n')
    logFile.close()


def write_log(lineToWrite, path=""):
    """function to append log file"""
    if path == "":
        logFile=open(r"C:\Users\iant\Dropbox\NOMAD\Python\log.txt", 'a')
    else:
        logFile=open(path, 'a')

    logFile.write(lineToWrite+'\n')
    logFile.close()
