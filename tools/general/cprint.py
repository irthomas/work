# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:09:42 2022

@author: iant

PRINT COLOURED TEXT TO THE CONSOLE
"""


def cprint(text, colour):
    
    d = {"k":30, "r":31, "g":32, "y":33, "b":34, "p":35, "c":36, "w":37}
    print("\033[0;%i;1m%s\033[0;0;0m" %(d[colour], text))