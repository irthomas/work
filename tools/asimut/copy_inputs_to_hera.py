# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:31:41 2022

@author: iant
"""



import paramiko

from tools.file.passwords import passwords





def make_remote_dir(sftp, remote_path):
    try:
        sftp.chdir(remote_path)  # Test if remote_path exists
    except IOError:
        sftp.mkdir(remote_path)  # Create remote_path



def put_file(sftp, local_path, remote_path):
    """copy file from local computer to remote path"""
    
    print("Uploading %s to %s" %(local_path, remote_path))
    sftp.put(local_path, remote_path)

def connect_to_hera():
    print("Connecting to hera")
    p = paramiko.SSHClient()
    p.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    p.connect("hera.oma.be", port=22, username="iant", password=passwords["hera"])

    sftp = p.open_sftp()
    
    return p, sftp


def close_hera(p, sftp):
    sftp.close()
    p.close()


def copy_inputs_to_hera(remote_dirpaths, local_filepaths, remote_filepaths):
    p, sftp = connect_to_hera()
    
    for remote_dirpath in remote_dirpaths:
        make_remote_dir(sftp, remote_dirpath)
    
    for local_path, remote_path in zip(local_filepaths, remote_filepaths):
        put_file(sftp, local_path, remote_path)
        
    close_hera(p, sftp)
    
    
