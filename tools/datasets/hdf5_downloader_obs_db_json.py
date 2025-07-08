# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:32:07 2025

@author: iant

CODE TO READ IN NOMAD OBSERVATION DATABASE JSON OUTPUT AND DOWNLOAD THE FILES

SOMETIMES THE DIALOG BOX OPENS IN THE BACKGROUND - CHECK BEHIND THE PYTHON WINDOW IF NO RESPONSE!

ONLY FOR NOMAD SCIENCE TEAM MEMBERS - YOU WILL NEED THE WEBDAV PASSWORD TO DOWNLOAD THE DATA

"""
import json
import posixpath
import os
import requests
from requests.auth import HTTPBasicAuth

# Insert the NOMAD sci WebDAV password here
WEBDAV_PASSWORD = ""

# Download directory (leave blank for current directory)
DOWNLOAD_DIR = "hdf5"

# Enter the filename of the JSON downloaded from nomad.aeronomie.be (leave blank to open dialog box)
JSON_FILENAME = ""

# Other constants
WEBDAV_USER = "nomadsci"
HDF5_LEVEL = "hdf5_level_1p0a"


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
    # load JSON
    with open(filename, "r") as f:
        obs_json = json.load(f)

    # loop through items in JSON
    for obs_line in obs_json:
        # if file exists on WebDAV or FTP
        if obs_line["onftp"]:
            h5 = obs_line["filename"]

            year = h5[0:4]
            month = h5[4:6]
            day = h5[6:8]

            # make remote and local paths, maintaining YYYY/MM/DD directory structure locally
            webdav_h5_path = posixpath.join("https://webdav.aeronomie.be/guest/nomadsci/Data", HDF5_LEVEL, year, month, day, h5)
            local_dir_path = os.path.join(DOWNLOAD_DIR, year, month, day)
            local_h5_path = os.path.join(local_dir_path, h5)

            # make local directory if it doesn't already exist
            os.makedirs(local_dir_path, exist_ok=True)

            print("Downloading %s" % h5)
            try:
                # get the file from the BIRA WebDAV server
                response = requests.get(webdav_h5_path, auth=HTTPBasicAuth(WEBDAV_USER, WEBDAV_PASSWORD))
                # check if request was accepted
                if not response.ok:
                    print(response.text)

                else:
                    # if response is good, write file locally
                    with open(local_h5_path, 'wb') as f:
                        f.write(response.content)

            # if error, print the exception
            except Exception as e:
                print(e)


""" Old code for FTP """
# import json
# import posixpath
# import os
# import ftplib

# # Insert the NOMAD FTP password here
# FTP_PASSWORD = ""

# # Download directory (leave blank for current directory)
# DOWNLOAD_DIR = "hdf5"

# # Enter the filename of the JSON downloaded from nomad.aeronomie.be (leave blank to open dialog box)
# JSON_FILENAME = ""


# if JSON_FILENAME == "":
#     # if no json given above, open a dialog box
#     from tkinter.filedialog import askopenfilename
#     # Select JSON to load
#     filename = askopenfilename()
# else:
#     filename = JSON_FILENAME

# if not os.path.exists(filename):
#     print("Error: json file %s doesn't exist" % filename)
# else:
#     with open(filename, "r") as f:
#         obs_json = json.load(f)

#     ftp = ftplib.FTP("ftp-ae.oma.be")
#     ftp.login("nomadsci", FTP_PASSWORD)

#     for obs_line in obs_json:
#         if obs_line["onftp"]:
#             h5 = obs_line["filename"]

#             year = h5[0:4]
#             month = h5[4:6]
#             day = h5[6:8]

#             level = "hdf5_level_1p0a"

#             ftp_h5_path = posixpath.join("Data", level, year, month, day, h5)
#             local_dir_path = os.path.join(DOWNLOAD_DIR, year, month, day)
#             local_h5_path = os.path.join(local_dir_path, h5)

#             os.makedirs(local_dir_path, exist_ok=True)

#             print("Downloading %s" % h5)
#             try:
#                 ftp.retrbinary("RETR " + ftp_h5_path, open(local_h5_path, 'wb').write)
#             except Exception as e:
#                 print(e)
#     ftp.quit()
