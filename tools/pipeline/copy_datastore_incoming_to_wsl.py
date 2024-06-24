# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:26:30 2024

@author: iant

COPY DATA FROM DATASTORE TO DEBIAN WSL INCOMING TO START THE DATA PIPELINE

"""


import shutil

# copy a day of data at a time
day_to_copy = r"\2024\05\08"

wsl_directories = {
    r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\db\edds\tc1553" + day_to_copy:
    r"\\wsl.localhost\Debian\bira-iasb\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\data_transfer\datastore\auxiliary\tch\378",

    r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\db\edds\spacewire" + day_to_copy:
    r"\\wsl.localhost\Debian\bira-iasb\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\data_transfer\datastore\tm\farc\nomad",

    r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\db\edds\tm1553_84" + day_to_copy:
    r"\\wsl.localhost\Debian\bira-iasb\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\data_transfer\datastore\tm\parc\84",

    r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\db\edds\tm1553_244" + day_to_copy:
    r"\\wsl.localhost\Debian\bira-iasb\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\data_transfer\datastore\tm\parc\244",

    r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\db\edds\tm1553_372" + day_to_copy:
    r"\\wsl.localhost\Debian\bira-iasb\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\data_transfer\datastore\tm\parc\372",
}


for src_dir, dst_dir in wsl_directories.items():
    print("copying %s to %s" % (src_dir, dst_dir))

    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

wsl_files = {
    r"X:\projects\NOMAD\data\pfm_auxiliary_files\observation_type\obs_type.db":
    r"\\wsl.localhost\Debian\bira-iasb\projects\NOMAD\Data\pfm_auxiliary_files\observation_type\obs_type.db",
}


for src, dst in wsl_files.items():
    print("copying %s to %s" % (src, dst))

    shutil.copy(src, dst)
