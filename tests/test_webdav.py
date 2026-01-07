# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 11:01:43 2025

@author: iant

WEBDAV VERSION: CODE TO READ IN NOMAD OBSERVATION DATABASE JSON OUTPUT AND DOWNLOAD THE FILES
ONLY FOR NOMAD SCIENCE TEAM MEMBERS - YOU WILL NEED THE WEBDAV PASSWORD TO DOWNLOAD THE DATA

SOMETIMES THE DIALOG BOX OPENS IN THE BACKGROUND - CHECK BEHIND THE PYTHON WINDOW IF NO RESPONSE!


"""

import json
import posixpath
import os
import requests
from requests.auth import HTTPBasicAuth

WEBDAV_USER = "nomadsci_adm"
# WEBDAV_USER = "nomadsci"

# Insert the NOMAD sci WebDAV password here
WEBDAV_PASSWORD = "biraEX016adm"
# WEBDAV_PASSWORD = "Exm2016nomad"


# Download directory (leave blank for current directory)
DOWNLOAD_DIR = "hdf5/"

# Enter the filename of the JSON downloaded from nomad.aeronomie.be (leave blank to open dialog box)
# JSON_FILENAME = "nomad_observation_list-2024-04-24.json"
# JSON_FILENAME = r"C:/Users/iant/Downloads/nomad_observation_list-2025-06-30.json"


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

#     for obs_line in obs_json:
#         if obs_line["onftp"]:
#             h5 = obs_line["filename"]

#             year = h5[0:4]
#             month = h5[4:6]
#             day = h5[6:8]

#             level = "hdf5_level_1p0a"

#             webdav_h5_path = posixpath.join("https://webdav.aeronomie.be/guest/nomadsci/Data", level, year, month, day, h5)
#             local_dir_path = os.path.join(DOWNLOAD_DIR, year, month, day)
#             local_h5_path = os.path.join(local_dir_path, h5)

#             os.makedirs(local_dir_path, exist_ok=True)

#             print("Downloading %s" % h5)
#             try:
#                 response = requests.get(webdav_h5_path, auth=HTTPBasicAuth(WEBDAV_USER, WEBDAV_PASSWORD))
#                 if not response.ok:
#                     print(response.text)

#                 with open(local_h5_path, 'wb') as f:
#                     f.write(response.content)
#             except Exception as e:
#                 print(e)


WEBDAV_URL = "https://webdav.aeronomie.be"
WEBDAV_ROOT_PATH = "/guest/nomadsci/"


search_dir = "Data/"
# search_dir = "Data/hdf5_level_1p0a/2025/01/01/"

# list files on webdav


def list_dir_subdirs_filepaths(search_dir, full_path=False):
    search_path = WEBDAV_URL + WEBDAV_ROOT_PATH + search_dir

    dirs = []
    files = []
    headers = {"Depth": "1"}
    response = requests.request(method="PROPFIND", url=search_path, auth=HTTPBasicAuth(WEBDAV_USER, WEBDAV_PASSWORD), headers=headers)
    resp_lines = response.content.decode().split()
    # print(resp_lines)
    for resp_line in resp_lines:
        if "<D:href>" in resp_line:
            dir_or_file = resp_line.replace("<D:href>%s" % WEBDAV_ROOT_PATH, "").replace("</D:href>", "")
            if dir_or_file != search_dir:
                if dir_or_file[-1] == "/":
                    if full_path:
                        dirs.append(dir_or_file[:-1])
                    else:
                        dirs.append(os.path.basename(dir_or_file[:-1]))
                else:
                    if full_path:
                        files.append(dir_or_file)
                    else:
                        files.append(os.path.basename(dir_or_file))

    return dirs, files


# dirs, files = list_dir_subdirs_filepaths(search_dir)

# for dir_path in dir_paths:
#     dir_paths, file_paths = list_dir_subdirs_filepaths(search_dir)

# make dir on webdev
# g = WEBDAV_URL + WEBDAV_ROOT_PATH + search_dir + "testing"
# response = requests.request(method="MKCOL", url=g, auth=HTTPBasicAuth(WEBDAV_USER, WEBDAV_PASSWORD))

# response = requests.request(method="put", url=g, auth=HTTPBasicAuth(WEBDAV_USER, WEBDAV_PASSWORD))

# print(response.content)


WEBDAV_USER = "nomadsci_adm"
WEBDAV_PASSWORD = "biraEXO16adm"
WEBDAV_URL = "https://webdav.aeronomie.be"
WEBDAV_ROOT_PATH = "/guest/nomadsci/"

WEBDAV_DATA_PATH = "Data"


# def dir_list(path, full_path=False):
#     # Return a list of all files and subdirs in specified path (non-recursive)
#     search_path = WEBDAV_URL + WEBDAV_ROOT_PATH + path

#     dirs = []
#     files = []
#     headers = {"Depth": "1"}
#     response = requests.request(method="PROPFIND", url=search_path, auth=HTTPBasicAuth(WEBDAV_USER, WEBDAV_PASSWORD), headers=headers)
#     resp_lines = response.content.decode().split()
#     for resp_line in resp_lines:
#         if "<D:href>" in resp_line:
#             dir_or_file = resp_line.replace("<D:href>%s" % WEBDAV_ROOT_PATH, "").replace("</D:href>", "")
#             if dir_or_file != path:
#                 if dir_or_file[-1] == "/":
#                     if full_path:
#                         dirs.append(dir_or_file[:-1])
#                     else:
#                         dirs.append(os.path.basename(dir_or_file[:-1]))
#                 else:
#                     if full_path:
#                         files.append(dir_or_file)
#                     else:
#                         files.append(os.path.basename(dir_or_file))

#     return (dirs, files)


# a = dir_list(WEBDAV_DATA_PATH)


def make_webdav_path(paths):
    return WEBDAV_URL + posixpath.join(WEBDAV_ROOT_PATH, posixpath.join(*[posixpath.normcase(path) for path in paths]))

# test MKCOL and put file


def make_webdav_dir(webdav_path):

    print("MKCOL path=", webdav_path)
    response = requests.request(method="MKCOL", url=webdav_path, auth=HTTPBasicAuth(WEBDAV_USER, WEBDAV_PASSWORD))
    # print(response)
    if response.ok:
        print("Path made on webdav")
    else:
        print("Webdav mkdir error:", response)


def upload_file(local_filepath, webdav_path):

    print("PUT filepath=", local_filepath, "to", webdav_path)
    with open(local_filepath, "rb") as f:
        response = requests.request(method="PUT", url=webdav_path, data=f, auth=HTTPBasicAuth(WEBDAV_USER, WEBDAV_PASSWORD))
    if response.ok:
        print("Path made on webdav")
    else:
        print("Webdav upload error:", response)


# webdav_path = make_webdav_path([WEBDAV_DATA_PATH, "testing"])

# make_webdav_dir(webdav_path)

# local_filepath = r"C:/Users/iant/Documents/DATA/hdf5/hdf5_level_1p0a/2025/05/08/20250508_200019_1p0a_LNO_1_DP_168.h5"

# webdav_path = make_webdav_path([WEBDAV_DATA_PATH, "testing", "test.h5"])

# upload_file(local_filepath, webdav_path)


f = r"2025/05/08/20250508_200019_1p0a_LNO_1_DP_168.h5"

# src = os.path.join(LEVEL_SWITCH[level], f)
# dst = make_webdav_path([WEBDAV_DATA_PATH, WEBDAV_SWITCH[level], f])
# logger.info("%i/%i: copying %s --> %s to webdav" % (i, len(files), src, dst))
# # try:
# dirname, filename = posixpath.split(dst)
# if dirname not in rem_tree:
#     for g in get_missing_dirs(rem_tree, dirname, posixpath.join(WEBDAV_DATA_PATH, SYNC_FOLDERS[level])):
#         logger.info(f"Creating folder {g} on webdav")
#         make_webdav_dir(g)
#         # TODO: finish this
#         # Add folder key to dictionary, and folder to parent key list
#         rem_tree[g] = []
#         parent, dir = posixpath.split(g)
#         rem_tree[parent].append(dir)
#         # Append filename to remote tree dictionary
#         rem_tree[dirname].append(filename)
#         upload_file(src, dst)

dst = make_webdav_path([WEBDAV_DATA_PATH, "hdf5_level_0p3a", f])
dirname, filename = posixpath.split(dst)
