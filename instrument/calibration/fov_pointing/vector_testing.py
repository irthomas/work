# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:58:36 2018

@author: ithom


VECTOR TESTING

"""

import os
import numpy as np
#import numpy.linalg as la

import spiceypy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if os.path.exists(os.path.normcase(r"C:\Users\ithom\Dropbox\NOMAD\Python")):
    BASE_DIRECTORY = os.path.normcase(r"C:\Users\ithom\Dropbox\NOMAD\Python")
    KERNEL_DIRECTORY = os.path.normcase(r"D:\kernels\kernels\mk")
    FIG_X = 18
    FIG_Y = 9

"""load spiceypy kernels"""
METAKERNEL_NAME = r"em16_ops_win.tm"
os.chdir(KERNEL_DIRECTORY)
sp.furnsh(KERNEL_DIRECTORY+os.sep+METAKERNEL_NAME)
print(sp.tkvrsn("toolkit"))
os.chdir(BASE_DIRECTORY)


origin = np.asfarray([0.0,0.0,0.0])

#vectorOffsetX = 3.0**(-1/2)
#vectorOffsetY = 3.0**(-1/2)
vectorOffsetX = np.arcsin(1.9992 / sp.dpr()) #2 degrees
vectorOffsetY = np.arcsin(15.0 * 88/90 / sp.dpr()) #15
vectorOffsetZ = np.sqrt(1.0-vectorOffsetX**2-vectorOffsetY**2)

vector = np.asfarray([vectorOffsetX,vectorOffsetY,vectorOffsetZ])
vectorMagnitude = sp.vnorm(vector) #should be 1

vector2 = np.asfarray([0.0,0.0,1.0])

angleSeparation = sp.vsep(vector/vectorMagnitude,vector2) * sp.dpr()

#X to Z
#vectorA = np.asfarray([vector[0], 0.0, vector[2]])/sp.vnorm([vector[0], 0.0, vector[2]])
vectorA = np.asfarray([vector[0], 0.0, np.sqrt(1.0-vector[0]**2)])
angleSeparationA = sp.vsep(vectorA/sp.vnorm(vectorA),vector2) * sp.dpr()

#Y to Z
#vectorB = np.asfarray([0.0, vector[1], vector[2]])/sp.vnorm([0.0, vector[1], vector[2]])
vectorB = np.asfarray([0.0, vector[1], np.sqrt(1.0-vector[1]**2)])
angleSeparationB = sp.vsep(vectorB/sp.vnorm(vectorB),vector2) * sp.dpr()

#X to Y
#vectorC = np.asfarray([vector[0], vector[1], 0.0])/sp.vnorm([vector[0], vector[1], 0.0])
vectorC = np.asfarray([vector[0], vector[1], 0.0])/sp.vnorm([vector[0], vector[1], 0.0])
angleSeparationC = sp.vsep(vectorC/sp.vnorm(vectorC),vector2) * sp.dpr()


vectorBack = np.asfarray([np.cos(angleSeparationB / sp.dpr()) * np.sin(angleSeparationA / sp.dpr()), \
                          np.cos(angleSeparationB / sp.dpr()) * np.cos(angleSeparationA / sp.dpr()), \
                          np.sin(angleSeparationB / sp.dpr())]) 

#vectorBack = np.asfarray([np.sin(angleSeparation / sp.dpr()),0.0,np.cos(angleSeparation / sp.dpr())])

def plotVector(ax_in,origin_in,vector_in, label="", components=False):
    ax_in.plot([origin_in[0],vector_in[0]],[origin_in[1],vector_in[1]],[origin_in[2],vector_in[2]], label=label)
    if components:
        ax_in.plot([origin_in[0],vector_in[0]],[0.0,0.0],[0.0,0.0], label=label+"_X")
        ax_in.plot([0.0,0.0],[origin_in[1],vector_in[1]],[0.0,0.0], label=label+"_Y")
        ax_in.plot([0.0,0.0],[0.0,0.0],[origin_in[2],vector_in[2]], label=label+"_Z")
        ax_in.scatter([origin_in[0],vector_in[0]],[0.0,0.0],[0.0,0.0])
        ax_in.scatter([0.0,0.0],[origin_in[1],vector_in[1]],[0.0,0.0])
        ax_in.scatter([0.0,0.0],[0.0,0.0],[origin_in[2],vector_in[2]])
        

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect("equal")
#plotVector(ax,origin,vector2,label="Vector2")
##plotVector(ax,origin,vector,label="Vector", components=True)
#plotVector(ax,origin,vector,label="Vector")
#plotVector(ax,origin,vectorA,label="VectorA")
#plotVector(ax,origin,vectorB,label="VectorB")
#plotVector(ax,origin,vectorC,label="VectorC")
#ax.legend()
#ax.set_xlim([-0.1,1.1])
#ax.set_ylim([-0.1,1.1])
#ax.set_zlim([-0.1,1.1])




