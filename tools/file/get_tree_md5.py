# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:40:45 2022

@author: iant
"""

import os
import paramiko


def get_linux_md5_dict(user, password, channel="", host="hera.oma.be"):
    print("Connecting to %s" %host)
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    p.connect(host, port=22, username=user, password=password)
    
    d = {}
    
    for year in range(2018, 2023):
        for month in range(1, 13):
            print("%02i/%04i" %(month, year))

    
    
            path = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/hdf5_level_1p0a/%04i/%02i" %(year, month)
            
            if channel == "":
                find_fmt = 'find %s/*/*.h5 -exec md5sum {} +'
                find_cmd = find_fmt % (path)
            else:
                find_fmt = 'find %s/*/*%s*.h5 -exec md5sum {} +'
                find_cmd = find_fmt % (path, channel.upper())
                
                
            
            stdin, stdout, stderr = p.exec_command(find_cmd)
            
            output = stdout.read().decode().split("\n")
            
            for s in output:
                if s != "":
                    d[os.path.basename(s.split()[1]).replace(".h5","")] = s.split()[0]
        
    
    p.close()

    return d


# path2 = r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a\2018\09\30"
# from subprocess import check_output
# cmd = 'for /R %s %%f in (*.*) do @certutil -hashfile "%%f" MD5' %path2
# cmd = 'for /R %%f in (where /r %s *.h5) do @certutil -hashfile "%%f" MD5' %r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a\2018\09"
# # cmd = 'where /r %s *.h5' %r"C:\Users\iant\Documents\DATA\hdf5\hdf5_level_1p0a\2018\09"
# a = check_output(cmd, shell=True)
# print(a.decode().split("\r\n"))