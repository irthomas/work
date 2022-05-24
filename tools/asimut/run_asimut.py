# -*- coding: utf-8 -*-
"""
Created on Mon May 23 21:07:49 2022

@author: iant

RUN ASIMUT ON SERVER
"""

import paramiko

from tools.file.passwords import passwords



def run_remote_command(command):
    """run command on hera/ada"""
    
    print("Connecting to hera")
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    p.connect("hera.oma.be", port=22, username="iant", password=passwords["hera"])
    
    print("Running command %s" %command)
    stdin, stdout, stderr = p.exec_command(command)
    stdin.close()
    #print output
    for line in iter(stdout.readline,""):
        print(line, end='')


def run_asimut(sh_filepath_linux):
    """make the shell script executable and run it"""
    
    
    #chmod shell script to be executable
    command = "chmod 755 %s" %sh_filepath_linux
    run_remote_command(command)
    
    #run the shell script to launch asimut
    command = "%s" %(sh_filepath_linux)
    run_remote_command(command)
    
