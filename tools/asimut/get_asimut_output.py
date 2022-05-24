# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:50:11 2022

@author: iant

DOWNLOAD _OUT.H5 FROM HERA
"""

import paramiko

from tools.file.passwords import passwords



def get_asimut_output(local_path, remote_path):
    """copy asimut output *_out.h5 back to local dir"""
    
    print("Connecting to hera")
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    p.connect("hera.oma.be", port=22, username="iant", password=passwords["hera"])
    
    sftp = p.open_sftp()
    print("Downloading %s to %s" %(remote_path, local_path))
    sftp.get(remote_path, local_path)
    sftp.close()
    p.close()
