# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:41:11 2022

@author: iant

GET EXM LIST FROM ESAC SERVER (VIA HERA)
"""

import os
import paramiko
import posixpath

from tools.file.passwords import passwords



DIRS = {"spacewire":"farc_nmd", "tc1553":"dat", "tm1553_84":"zip", "tm1553_244":"zip", "tm1553_372":"zip"}



def get_esac_tree_filenames(user, level, host="hera.oma.be"):

    print("Connecting to %s" %host)
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())   # This script doesn't work for me unless this line is added!
    p.connect(host, port=22, username=user, password=passwords["hera"])
    
    print("Connecting to ESAC")
    
    if level in DIRS.keys():
        dir_name = DIRS[level]
    
        stdin, stdout, stderr = p.exec_command('ssh ada6 "ssh exonmd@exoops01.esac.esa.int "ls %s/""' %dir_name)
        
    else:
        print("Error: level not found in dictionary")
    

    filenames = [s for s in stdout.read().decode().split("\n") if s != ""]
    return filenames


# print("Running command")
# stdin, stdout, stderr = vm3.exec_command('find farc_nmd/ -name "*_2022-002T*"') #edited#
# #
# print(stdout.read())



