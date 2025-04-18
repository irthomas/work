# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:32:07 2025

@author: iant

CODE TO READ IN NOMAD OBSERVATION DATABASE JSON OUTPUT AND DOWNLOAD THE FILES

SOMETIMES THE DIALOG BOX OPENS IN THE BACKGROUND - CHECK BEHIND THE PYTHON WINDOW IF NO RESPONSE!

ONLY FOR NOMAD SCIENCE TEAM MEMBERS - YOU WILL NEED THE FTP PASSWORD TO DOWNLOAD THE DATA

"""

import json
import posixpath
import os
import ftplib

# Insert the NOMAD FTP password here
FTP_PASSWORD = "Exm2016nomad"

# Download directory (leave blank for current directory)
DOWNLOAD_DIR = "hdf5/"

# Enter the filename of the JSON downloaded from nomad.aeronomie.be (leave blank to open dialog box)
# JSON_FILENAME = "nomad_observation_list-2024-04-24.json"
JSON_FILENAME = ""


if JSON_FILENAME == "":
    # if no json given above, open a dialog box
    from tkinter.filedialog import askopenfilename
    # Select JSON to load
    filename = askopenfilename()
else:
    filename = JSON_FILENAME

if not os.path.exists(filename):
    print("Error: json file %s doesn't exist" % filename)
else:
    with open(filename, "r") as f:
        obs_json = json.load(f)

    ftp = ftplib.FTP("ftp-ae.oma.be")
    ftp.login("nomadsci", FTP_PASSWORD)

    for obs_line in obs_json:
        if obs_line["onftp"]:
            h5 = obs_line["filename"]

            year = h5[0:4]
            month = h5[4:6]
            day = h5[6:8]

            level = "hdf5_level_1p0a"

            ftp_h5_path = posixpath.join("Data", level, year, month, day, h5)
            local_dir_path = os.path.join(DOWNLOAD_DIR, year, month, day)
            local_h5_path = os.path.join(local_dir_path, h5)

            os.makedirs(local_dir_path, exist_ok=True)

            print("Downloading %s" % h5)
            try:
                ftp.retrbinary("RETR " + ftp_h5_path, open(local_h5_path, 'wb').write)
            except Exception as e:
                print(e)
    ftp.quit()
