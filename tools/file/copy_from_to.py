# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:39:07 2020

@author: iant

COPY ANY LEVEL FILES MATCHING REGEX FROM ANY LOCATION TO ANY OTHER
"""

import os
import re
from shutil import copyfile
import posixpath
import pysftp
import ftplib

from tools.file.passwords import passwords

_from = "hera"
to = "local"

#_from = "datastore"
#to = "local"

#_from = "datastore"
#to = "ftp"

#_from = "local"
#to = "ftp"

##hard code paths here to avoid problems###
if _from == "datastore": #when at BIRA
    FROM_ROOT_DIRECTORY = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")
elif _from == "datastore_test":
    FROM_ROOT_DIRECTORY = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\hdf5")
elif _from == "hera": #when at home
    FROM_ROOT_DIRECTORY = posixpath.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5")
elif _from == "hera_test":
    FROM_ROOT_DIRECTORY = posixpath.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/test/ian")
elif _from == "local":
    FROM_ROOT_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\hdf5_copy")

if to == "local": 
    TO_ROOT_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\hdf5_copy")
elif to == "ftp":
    TO_ROOT_DIRECTORY = posixpath.normcase(r"/Data")
    

server = ["hera.oma.be", "iant"]
password = ""
server_ftp = ["ftp-ae.oma.be", "nomadadm"]
password_ftp = passwords["nomadadm"]


    
#OVERWRITE = True
OVERWRITE = False

#regex = "20180422_.*_SO_.*_134"
#regex = "20(18|19|20)[0-1][0-9][0-9][0-9]_.*_LNO_.*_D_(178|167|166|162)"
#regex = "20(18|19|20)[0-1][0-9][0-9][0-9]_.*_SO_.*_[IE]_(129|130)"
#regex = "20(18|19|20)[0-1][0-9][0-9][0-9]_.*_SO"
regex = "2019[0-1][0-9][0-9][0-9]_.*_LNO.*"
#regex = "20(18|19|20)[0-1][0-9][0-9][0-9]_.*_(LNO|SO)_[0-9]_C"
r = re.compile(regex)


#level = "hdf5_l02a"
#level = "hdf5_l03a"
level = "hdf5_l10a"
folderName = "hdf5_level_%sp%s" %(level[-3:-2], level[-2:])

print("Finding files matching regex %s" %regex)

if _from in ["datastore", "datastore_test", "local"]:
    folderPath = os.path.join(FROM_ROOT_DIRECTORY, folderName)
    nFiles = 0
    hdf5DatastoreFilepathsUnsorted = []
    for root, dirs, files in os.walk(folderPath):
        matchingFiles = list(filter(r.match, files))
        for filename in matchingFiles:
            if ".h5" in filename:
                hdf5DatastoreFilepathsUnsorted.append(os.path.join(root, filename))
                nFiles += 1

    hdf5DatastoreFilepaths = sorted(hdf5DatastoreFilepathsUnsorted)



if _from in ["hera", "hera_test"]:
#    folderPath = os.path.join(FROM_ROOT_DIRECTORY, folderName)
    folderPath = posixpath.join(FROM_ROOT_DIRECTORY, folderName)



    def listdir_r(sftp, remotedir, file_paths):
        from stat import S_ISDIR, S_ISREG

        for entry in sftp.listdir_attr(remotedir):
            remotepath = remotedir + "/" + entry.filename
            mode = entry.st_mode
            if S_ISDIR(mode):
                file_paths = listdir_r(sftp, remotepath, file_paths)
            elif S_ISREG(mode):
                if r.match(posixpath.split(remotepath)[1]):
                    file_paths.append(remotepath)
        return file_paths

    if password == "":
        import getpass
        password = getpass.getpass('Password:')


    with pysftp.Connection(server[0], username=server[1], password=password) as sftp:
        file_paths = []
        hdf5DatastoreFilepathsUnsorted = listdir_r(sftp, folderPath, file_paths)
    hdf5DatastoreFilepaths = sorted(hdf5DatastoreFilepathsUnsorted)
    nFiles = len(hdf5DatastoreFilepaths)








print("Transfering %i files from %s to %s" %(nFiles, _from, to))
print("#############")

      
if to == "local":
    file_number = 0
    for hdf5DatastoreFilepath in hdf5DatastoreFilepaths:
    
        filepath, filename = os.path.split(hdf5DatastoreFilepath)
        newFolderPath = os.path.join(TO_ROOT_DIRECTORY, folderName)
        year = filename[0:4] #get the date from the filename to find the file
        month = filename[4:6]
        day = filename[6:8]
        newFilepath = os.path.join(newFolderPath, year, month, day)
        newFilenamepath = os.path.join(newFilepath, filename)
        file_number += 1
        print("File %i/%i %s: " % (file_number, nFiles, filename), end = '')
    
        if not os.path.isdir(newFilepath):
            os.makedirs(newFilepath, exist_ok=True)
            print("Making directory %s" %newFilepath, end = '')
    
        if os.path.exists(newFilenamepath):
            print("File already exists. ", end = '')
            
            if OVERWRITE:
                print("--> overwriting on local drive %s" %(filename))

                if _from in ["datastore", "datastore_test"]:
                    copyfile(hdf5DatastoreFilepath, newFilenamepath)

                if _from in ["hera", "hera_test"]:
                    if password == "":
                        import getpass
                        password = getpass.getpass('Password:')

                    with pysftp.Connection(server[0], username=server[1], password=password) as sftp:
                        sftp.get(hdf5DatastoreFilepath, newFilenamepath) # get a remote file
                
            else:
                print("Skipping")
        else:
            print("--> copying to local drive %s" %(filename))
            if _from in ["datastore", "datastore_test"]:
                copyfile(hdf5DatastoreFilepath, newFilenamepath)

            if _from in ["hera", "hera_test"]:
                if password == "":
                    import getpass
                    password = getpass.getpass('Password:')

                with pysftp.Connection(server[0], username=server[1], password=password) as sftp:
                    sftp.get(hdf5DatastoreFilepath, newFilenamepath) # get a remote file
        


if to == "ftp":

    def open_ftp(server_address, username, password):
        # Open an FTP connection with specified credentials
        ftp_conn = ftplib.FTP(server_address)
        try:
            ftp_conn.login(user=username, passwd=password)
        except ftplib.all_errors as e:
            print("FTP error ({0})".format(e.message))
        return ftp_conn
    
    

    ftp_conn = open_ftp(server_ftp[0], server_ftp[1], password_ftp)

    file_number = 0
    for hdf5DatastoreFilepath in hdf5DatastoreFilepaths:
    
        filepath, filename = os.path.split(hdf5DatastoreFilepath)
        newFolderPath = posixpath.join(TO_ROOT_DIRECTORY, folderName)
        year = filename[0:4] #get the date from the filename to find the file
        month = filename[4:6]
        day = filename[6:8]
        newFilepath = posixpath.join(newFolderPath, year, month, day)
        newFilenamepath = posixpath.join(newFilepath, filename)
        file_number += 1
        print("File %i/%i %s: " % (file_number, nFiles, filename), end = '')

        #loop through subdirectories, making them if they don't already exist
        filepathSplit = newFilepath.split(posixpath.sep)

        filepathSection = "/"
        for filepathEach in filepathSplit:
            filepathSection = posixpath.join(filepathSection, filepathEach)
#            print(filepathSection)
            
            filepathSectionSplit = posixpath.split(filepathSection)
            foldersInDirectory = [a for a,_ in list(ftp_conn.mlsd(filepathSectionSplit[0])) if a not in [".",".."]]
            if filepathSectionSplit[1] in foldersInDirectory or filepathSectionSplit[1] == "":
#                print("Directory %s exists. " %(filepathSection), end = '')
                ftp_conn.cwd(filepathSection)
            else:
                print("Making directory %s. " %(filepathSection), end = '')
                ftp_conn.mkd(filepathSection)

        print("--> copying to ftp %s" %(newFilepath))
        with open(hdf5DatastoreFilepath, 'rb') as f:
            ftp_conn.storbinary('STOR {0}'.format(newFilenamepath), f)
   
    ftp_conn.close()
    
    
