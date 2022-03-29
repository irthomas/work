# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:48:02 2021

@author: iant
Code taken from sync_bira_esa.py

GET MODIFIED OR NEW PSA LOGS FROM ESA SERVER
"""

import os
import os.path
import re
import subprocess
import sys
import tarfile
from urllib.parse import urlparse


BIRA_URL = "file:/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/"
BIRA_BASE = urlparse(BIRA_URL).path
ESA_URL = "ssh://exonmd@exoops01.esac.esa.int/home/exonmd/"
ESA_HOME = urlparse(ESA_URL).path



def make_find_command(p_url):
    find_fmt = 'find %s -type f -printf "%%p %%T@ %%s\\n"'
    find_cmd = find_fmt % p_url.path
    if p_url.scheme == "ssh":
        return ["ssh", p_url.netloc, find_cmd]
    elif p_url.scheme == "file":
        return find_cmd
    else:
        sys.exit("Invalid url: %s" % str(p_url))



def exec_find_command(p_url):
    result = {}
    duplicates = {}
    find_cmd = make_find_command(p_url)
    ssh = subprocess.Popen(find_cmd,
                       shell=p_url.scheme == "file",
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    lines = ssh.stdout.readlines()
    for l in lines:
        file_path, secs, size = l.strip().split()
        seconds = int(float(secs))
        size = int(size)
        file_name = os.path.basename(file_path)
        if file_name in result:
            duplicates.setdefault(file_name, set([result[file_name][0]])).add(file_path)
            if len(file_path) < len(result[file_name][0]):
                result[file_name] = (file_path, seconds, size)
        else:
            result[file_name] = (file_path, seconds, size)
    return result, duplicates




def diff_bira_esa(bira_p_url, esa_p_url, accept_re):
    bira_files, bira_duplicates = exec_find_command(bira_p_url)
    esa_files, esa_duplicates = exec_find_command(esa_p_url)

    esa_only = []
    uptodate = []
    outdated = []
    for filename, (esa_path, esa_secs, esa_size) in esa_files.items():
        if accept_re.match(filename) is None:
            continue
        bira_path, bira_secs, bira_size = bira_files.pop(filename, (None, None, None))
        if bira_path is None:
            esa_only.append(esa_path)
        else:
            time_ok = bira_secs >= esa_secs
            if time_ok and esa_size == bira_size:
                uptodate.append((esa_path, bira_path))
            else:
                outdated.append((esa_path, bira_path))

    bira_only = []
    for filename, (bira_path, _, _) in bira_files.items():
        if accept_re.match(filename):
            bira_only.append(bira_path)

    return esa_only, bira_only, outdated, uptodate, bira_duplicates, esa_duplicates



def psa_logs_esa_to_bira():

    print("Getting PSA calibration logs from ESA server")
    dest = BIRA_BASE + "data_transfer/datastore/pds/logs/psa_cal/"
    accept_re = re.compile(b"^nmd-pi-delivery.log.*")
    
    esa_p_url = urlparse(ESA_URL + "logs/")
    bira_p_url = urlparse(BIRA_URL + "db/psa/cal_logs/")
    result = diff_bira_esa(bira_p_url, esa_p_url, accept_re)
    esa_only, _, outdated, _, _, _ = result
    file_list = esa_only + [esa_p for esa_p, _ in outdated]
    print("%d files to download" % len(file_list))
    
    
    if len(file_list) > 0:
        # Run a 'tar' on ESA server
        tar_cmd = ["ssh", esa_p_url.netloc, "tar c -b 512 -T -"]
        with subprocess.Popen(tar_cmd,
                        shell=False,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE) as ssh:
            # Write file list on tar stdin (-T -)
            for f in file_list:
                ssh.stdin.write(b"%s\n" % f)
            ssh.stdin.close()
        
            # Read the tar stream from ESA in a Tarfile object
            tar = tarfile.open(fileobj=ssh.stdout, mode="r|")
        
            ## For each file in the tar stream
            for t_info in iter(tar):
                buf_reader = tar.extractfile(t_info)
                path = os.path.join(dest, os.path.basename(t_info.name))
                n = 0
                with open(path, "wb") as fd:
                    while True:
                        buf = buf_reader.read(512 * 512)
                        if not buf:
                            break
                        n += len(buf)
                        fd.write(buf)
                print("Downloaded: %s (%.1f kB)" %(path, n/1000))
