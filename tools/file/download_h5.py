# -*- coding: utf-8 -*-
r"""
Created on Fri Oct  3 10:14:11 2025

@author: iant

DOWNLOAD SINGLE MISSING HDF5 FILE FROM WEBDAV (WITH HERA BACKUP) TO LOCAL COMPUTER

#download_h5("20240915_141221_1p0a_UVIS_CL.h5", path=r"C:\Users\iant\Documents\DATA\hdf5")
            
"""


import os
import posixpath
import requests
from requests.auth import HTTPBasicAuth

import paramiko

from tools.file.paths import paths
from tools.file.passwords import passwords


WEBDAV_USER = "nomadsci"
WEBDAV_URL = "https://webdav.aeronomie.be"
WEBDAV_ROOT_PATH = "/guest/nomadsci/"
WEBDAV_PASSWORD = passwords["nomadsci_webdav"]

HERA_USER = "iant"
HERA_URL = "hera.oma.be"
HERA_ROOT_PATH = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/"
HERA_PASSWORD = passwords["hera"]


def download_h5(h5, path=None, try_webdav=True):

    if path:
        DATA_DIRECTORY = path
    else:
        DATA_DIRECTORY = paths["DATA_DIRECTORY"]

    level = "hdf5_level_" + h5.split("_")[2]

    year = h5[0:4]
    month = h5[4:6]
    day = h5[6:8]

    webdav_h5_path = posixpath.join("https://webdav.aeronomie.be/guest/nomadsci/Data", level, year, month, day, h5)
    local_dir_path = os.path.join(DATA_DIRECTORY, level, year, month, day)
    local_h5_path = os.path.join(local_dir_path, h5)

    os.makedirs(local_dir_path, exist_ok=True)

    print("Downloading from webdav %s" % webdav_h5_path)

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=HERA_URL, username=HERA_USER, password=HERA_PASSWORD)
        sftp = ssh.open_sftp()

        if try_webdav:

            response = requests.get(webdav_h5_path, auth=HTTPBasicAuth(WEBDAV_USER, WEBDAV_PASSWORD))
            if response.ok:
                with open(local_h5_path, 'wb') as f:
                    f.write(response.content)
            else:

                if response.status_code != 404:
                    print(response.text)

                else:
                    sftp_h5_path = posixpath.join(HERA_ROOT_PATH, level, year, month, day, h5)
                    print("Retrying from hera %s" % sftp_h5_path)
                    sftp.get(sftp_h5_path, local_h5_path)
        else:
            sftp_h5_path = posixpath.join(HERA_ROOT_PATH, level, year, month, day, h5)
            print("Downloading from hera %s" % sftp_h5_path)
            sftp.get(sftp_h5_path, local_h5_path)

    except Exception as e:
        print("Error:", e)

    finally:
        sftp.close()
        ssh.close()
