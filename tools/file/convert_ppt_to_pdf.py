# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:03:54 2020

@author: iant

CONVERT ALL PPT OR PPTX IN A FOLDER TO PDF AND SAVE A HTML TABLE CONTAINING ALL ENTRIES

ONLY WORKS ON WINDOWS!!
"""

import os
import comtypes.client
import time

PPT_DIRECTORY = os.path.normcase(r"X:\projects\NOMAD\Science\Meetings\SWT15_190912_Wroclaw\SWT Day1")
PPT_DIRECTORY = os.path.normcase(r"X:\projects\NOMAD\Science\Meetings\SWT15_190912_Wroclaw\SWT Day2")


def PPTtoPDF(inputFileName, outputFileName, formatType = 32):
    powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
    powerpoint.Visible = 1

    if outputFileName[-3:] != 'pdf':
        outputFileName = outputFileName + ".pdf"
    deck = powerpoint.Presentations.Open(inputFileName)
    deck.SaveAs(outputFileName, formatType) # formatType = 32 for ppt to pdf
    deck.Close()
    powerpoint.Quit()
    time.sleep(5)
    
    

dayDirName = os.path.split(PPT_DIRECTORY)[-1]
swtDirName = os.path.split(os.path.split(PPT_DIRECTORY)[0])[-1]

pptFilenames = sorted(os.listdir(PPT_DIRECTORY))

htmlHeader = ["Title", "Presenter"]

h = r""
h += r"<table border=0>"+"\n"
h += r"<tr>"+"\n"
for headerColumn in htmlHeader:
    h += r"<th>%s</th>" %headerColumn +"\n"
h += r"</tr>"+"\n"

for pptFilename in pptFilenames:

    pptBasename = os.path.splitext(pptFilename)[-1]
    

    pdfFilename = os.path.splitext(pptFilename)[0] + ".pdf"
        
    if pptBasename in [".ppt", ".pptx"]:
        pptFilepath = os.path.join(PPT_DIRECTORY, pptFilename)
        pdfFilepath = os.path.join(PPT_DIRECTORY, pdfFilename)
        PPTtoPDF(pptFilepath, pdfFilepath)
        time.sleep(5)
    
    
    if pptBasename in [".ppt", ".pptx", ".pdf"]:
        h += r"<tr>"+"\n"
        for element in [pdfFilename, ""]:
            h += r"<td>%s</td>" %(element) +"\n"
        h += r"</tr>"+"\n"
    else:
        print("skipping", pptFilename)

h += r"</table>"+"\n"

f = open(os.path.join(PPT_DIRECTORY, "%s_%s.html" %(swtDirName, dayDirName)), 'w')
f.write(h)
f.close()
