# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:54:19 2024

@author: iant

GET LIST OF PSA PRODUCTS FROM ZIP FILES IN THE DATASTORE ARCHIVE

CODE TO COPY FILELIST TO ESA SERVER
cd /bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/archive/psa/4.0
scp nomad_filelist_2024-05-14.txt exonmd@exoops01.esac.esa.int:~/nmd/input_filelist
"""


import os
import glob
import zipfile
import platform

from tools.general.progress_bar import progress


if platform.system() == "Windows":
    path = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\archive\psa\4.0\data_calibrated")
elif platform.system() == "Linux":
    path = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/archive/psa/4.0/data_calibrated/")

print("Getting file list (this may take up to 10 minutes)")
zip_filelist = sorted(glob.glob(path + os.sep + "**" + os.sep + "*.zip", recursive=True))


print("Getting filenames from zips")
xml_filelist = ["# browse png,browse xml,data tab,data xml\n"]

for zip_filepath in progress(zip_filelist):

    zip_ = zipfile.ZipFile(zip_filepath)

    zip_contents = sorted(zip_.namelist())

    xml_filelist.append("%s,%s,%s,%s\n" % (*zip_contents,))

print("Writing to file")
with open("nomad_filelist_2024-05-31.txt", "w") as f:
    for line in xml_filelist:
        f.write(line)

print("Done")

# now copy file to /bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/archive/psa/4.0 and scp to ESA
