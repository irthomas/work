# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:09:56 2020

@author: iant
"""
import os
import posixpath
import re
from shutil import copyfile

from tools.file.passwords import passwords



#OVERWRITE = True
OVERWRITE = False


_from = "hera"
to = "local"

#_from = "datastore"
#to = "local"

#_from = "datastore"
#to = "ftp"

#_from = "local"
#to = "ftp"

# _from = "datastore_linux"
# to = "archive_linux"



#regex = "20180422_.*_SO_.*_134"
#regex = "20(18|19|20)[0-1][0-9][0-9][0-9]_.*_LNO_.*_D_(178|167|166|162)"
#regex = "20(18|19|20)[0-1][0-9][0-9][0-9]_.*_SO_.*_[IE]_(134|136)"
#regex = "20(18|19|20)[0-1][0-9][0-9][0-9]_.*_SO"
regex = re.compile("20(18|19|20)[0-1][0-9][0-9][0-9]_.*_(LNO|SO)_[0-9]_C")
# regex = "2020050[0-9]_.*_LNO_"


level = "hdf5_l02a"
#level = "hdf5_l03a"
# level = "hdf5_l10a"



def get_input_output_paths(to, _from, level):

    dir_name = "hdf5_level_%sp%s" %(level[-3:-2], level[-2:])
    
    from_dict = {
    "datastore":{"sep":"os", "root":r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"}, #windows at BIRA
    "datastore_test":{"sep":"os", "root":r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\hdf5"}, #windows at BIRA
    
    "hera":{"sep":"posix", "root":"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5"}, #via ssh at home
    "hera_test":{"sep":"posix", "root":"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/test/ian/hdf5"}, #via ssh at home
    
    "datastore_linux":{"sep":"posix", "root":"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5"}, #on linux
    
    "local":{"sep":"os", "root":r"C:\Users\iant\Documents\DATA\hdf5_copy"}, #windows at home
    "local_hdd":{"sep":"os", "root":r"D:\DATA\hdf5"}, #windows at home HDD
    }
    
    to_dict = {
    "local":{"sep":"os", "root":r"C:\Users\iant\Documents\DATA\hdf5_copy"}, #windows computer
    "local_hdd":{"sep":"os", "root":r"D:\DATA\hdf5"}, #windows at home HDD
    "ftp":{"sep":"posix", "root":"/Data"},
    "archive_linux":{"sep":"posix", "root":"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/archive/hdf5"}
        
    }
    
    # if _from == "datastore": #when at BIRA
    #     from_root = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")
    # elif _from == "datastore_test":
    #     from_root = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\hdf5")
    # elif _from == "hera": #when at home via ssh
    #     from_root = posixpath.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5")
    # elif _from == "datastore_linux": #on linux
    #     from_root = posixpath.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5")
    # elif _from == "hera_test": #when at home via ssh
    #     from_root = posixpath.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/test/ian/hdf5")
    # elif _from == "local":
    #     from_root = os.path.normcase(r"C:\Users\iant\Documents\DATA\hdf5_copy")
    # else:
    #     from_root = ""
    
    # if to == "local": 
    #     to_root = os.path.normcase(r"C:\Users\iant\Documents\DATA\hdf5_copy")
    # elif to == "ftp":
    #     to_root = posixpath.normcase(r"/Data")
    # elif to == "archive_linux":
    #     to_root = posixpath.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/archive/hdf5")
    # else:
    #     to_root = ""
    
    if _from in from_dict.keys():
        if from_dict[_from]["sep"] == "os":
            from_path = os.path.join(from_dict[_from]["root"], dir_name)
        elif from_dict[_from]["sep"] == "posix":
            from_path = posixpath.join(from_dict[_from]["root"], dir_name)
    else:
        print("Error: input not found")

    if to in to_dict.keys():
        if to_dict[to]["sep"] == "os":
            to_path = os.path.join(to_dict[to]["root"], dir_name)
        elif to_dict[to]["sep"] == "posix":
            to_path = posixpath.join(to_dict[to]["root"], dir_name)
    else:
        print("Error: input not found")

    
    return [from_path, to_path]



def get_filepaths_from(_from, from_path, regex):

    print("Finding files matching regex %s" %regex)

    server = ["hera.oma.be", "iant"]
    password = passwords["hera"]

    
    if _from in ["datastore", "datastore_test", "local", "datastore_linux"]:
        # nFiles = 0
        hdf5DatastoreFilepathsUnsorted = []
        for root, dirs, files in os.walk(from_path):
            matchingFiles = list(filter(regex.match, files))
            for filename in matchingFiles:
                if ".h5" in filename:
                    hdf5DatastoreFilepathsUnsorted.append(os.path.join(root, filename))
                    # nFiles += 1
    
        # hdf5DatastoreFilepaths = sorted(hdf5DatastoreFilepathsUnsorted)
    
    
    
    if _from in ["hera", "hera_test"]:
        import pysftp
   
        def listdir_r(sftp, remotedir, file_paths):
            from stat import S_ISDIR, S_ISREG
    
            for entry in sftp.listdir_attr(remotedir):
                remotepath = remotedir + "/" + entry.filename
                mode = entry.st_mode
                if S_ISDIR(mode):
                    file_paths = listdir_r(sftp, remotepath, file_paths)
                elif S_ISREG(mode):
                    if regex.match(posixpath.split(remotepath)[1]):
                        file_paths.append(remotepath)
            return file_paths

        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
    
        with pysftp.Connection(host=server[0], username=server[1], password=password, cnopts=cnopts) as sftp:
            file_paths = []
            # print(from_path, file_paths)
            hdf5DatastoreFilepathsUnsorted = listdir_r(sftp, from_path, file_paths)

    hdf5DatastoreFilepaths = sorted(hdf5DatastoreFilepathsUnsorted)
    
    return hdf5DatastoreFilepaths
    


def copy_files_to(to, to_path, _from, hdf5DatastoreFilepaths):

    server = ["hera.oma.be", "iant"]
    password = passwords["hera"]


    nFiles = len(hdf5DatastoreFilepaths)
    print("Transfering %i files to %s" %(nFiles, to))
    print("#############")
    
          
    if to == "local":
        import pysftp
        file_number = 0
        for hdf5DatastoreFilepath in hdf5DatastoreFilepaths:
        
            filepath, filename = os.path.split(hdf5DatastoreFilepath)
            year = filename[0:4] #get the date from the filename to find the file
            month = filename[4:6]
            day = filename[6:8]
            newFilepath = os.path.join(to_path, year, month, day)
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

                        cnopts = pysftp.CnOpts()
                        cnopts.hostkeys = None
    
                        with pysftp.Connection(host=server[0], username=server[1], password=password, cnopts=cnopts) as sftp:
                            sftp.get(hdf5DatastoreFilepath, newFilenamepath) # get a remote file
                    
                else:
                    print("Skipping")
            else:
                print("--> copying to local drive %s" %(filename))
                if _from in ["datastore", "datastore_test"]:
                    copyfile(hdf5DatastoreFilepath, newFilenamepath)
    
                if _from in ["hera", "hera_test"]:
                    cnopts = pysftp.CnOpts()
                    cnopts.hostkeys = None

                    with pysftp.Connection(host=server[0], username=server[1], password=password, cnopts=cnopts) as sftp:
                        sftp.get(hdf5DatastoreFilepath, newFilenamepath) # get a remote file
            
    
    
    if to == "ftp":
        import ftplib
    
        def open_ftp(server_address, username, password):
            # Open an FTP connection with specified credentials
            ftp_conn = ftplib.FTP(server_address)
            try:
                ftp_conn.login(user=username, passwd=password)
            except ftplib.all_errors as e:
                print("FTP error ({0})".format(e.message))
            return ftp_conn
        
        
    
        SC_FTP_ADR = "ftp-ae.oma.be"
        SC_FTP_USR = "nomadadm"
        SC_FTP_PWD = passwords["nomadadm"]
        
        ftp_conn = open_ftp(SC_FTP_ADR, SC_FTP_USR, SC_FTP_PWD)
    
        file_number = 0
        for hdf5DatastoreFilepath in hdf5DatastoreFilepaths:
        
            filepath, filename = os.path.split(hdf5DatastoreFilepath)
            year = filename[0:4] #get the date from the filename to find the file
            month = filename[4:6]
            day = filename[6:8]
            newFilepath = posixpath.join(to_path, year, month, day)
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
        
    
    
    if to == "archive_linux":
        file_number = 0
        for hdf5DatastoreFilepath in hdf5DatastoreFilepaths:
        
            filepath, filename = os.path.split(hdf5DatastoreFilepath)
            year = filename[0:4] #get the date from the filename to find the file
            month = filename[4:6]
            day = filename[6:8]
            newFilepath = os.path.join(to_path, year, month, day)
            newFilenamepath = os.path.join(newFilepath, filename)
            file_number += 1
            print("File %i/%i %s: " % (file_number, nFiles, filename), end = '')
        
            if not os.path.isdir(newFilepath):
                os.makedirs(newFilepath, exist_ok=True)
                print("Making directory %s" %newFilepath, end = '')
        
            if os.path.exists(newFilenamepath):
                print("File already exists. ", end = '')
                
                if OVERWRITE:
                    print("--> overwriting archive file %s" %(filename))
    
                    if _from in ["archive_linux"]:
                        copyfile(hdf5DatastoreFilepath, newFilenamepath)
    
                else:
                    print("Skipping")
            else:
                print("--> copying to archive %s" %(filename))
                if _from in ["datastore_linux"]:
                    copyfile(hdf5DatastoreFilepath, newFilenamepath)

        
    
from_path, to_path = get_input_output_paths(to, _from, level)

hdf5file_list = get_filepaths_from(_from, from_path, regex)
copy_files_to(to, to_path, _from, hdf5file_list)
