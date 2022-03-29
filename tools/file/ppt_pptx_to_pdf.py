# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:00:38 2021

@author: iant

CONVERT PPT OR PPTX TO PDF
"""

import comtypes.client
import time


def ppt_pptx_to_pdf(filepath_in, filepath_out, format_type=32):
    powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
    powerpoint.Visible = 1

    if filepath_out[-3:] != 'pdf':
        filepath_out = filepath_out + ".pdf"
    deck = powerpoint.Presentations.Open(filepath_in)
    deck.SaveAs(filepath_out, format_type) # formatType = 32 for ppt to pdf
    deck.Close()
    powerpoint.Quit()
    time.sleep(5)
    
