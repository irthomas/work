# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:02:38 2020

@author: iant
"""



def check_flags(flagsDict):
    """check if combination of flags is correct"""
    calibrationType = ""
    if flagsDict["Y_UNIT_FLAG"] == 0 and flagsDict["Y_TYPE_FLAG"] == 0: #no calibration
        calibrationType = "None"
    elif flagsDict["Y_UNIT_FLAG"] == 1 and flagsDict["Y_TYPE_FLAG"] == 3: #radiance factor
        calibrationType = "Radiance Factor"
    elif flagsDict["Y_UNIT_FLAG"] == 2 and flagsDict["Y_TYPE_FLAG"] == 1: #radiance in W/cm2/sr/cm-1
        calibrationType = "Radiance"
    elif flagsDict["Y_UNIT_FLAG"] == 3 and flagsDict["Y_TYPE_FLAG"] == 4: #brightness temperature in K
        calibrationType = "Brightness Temperature"
    elif flagsDict["Y_UNIT_FLAG"] == 4 and flagsDict["Y_TYPE_FLAG"] == 5: #radiance in W/cm2/sr/cm-1 and radiance factor together in file
        calibrationType = "Radiance & Radiance Factor"

    errorType = ""
    if flagsDict["Y_ERROR_FLAG"] == 0:
        errorType = "None"
    elif flagsDict["Y_ERROR_FLAG"] == 1:
        errorType = "One Value"
    if flagsDict["Y_ERROR_FLAG"] == 2:
        errorType = "Per Pixel"


    return calibrationType, errorType

