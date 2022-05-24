# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:28:29 2022

@author: iant

WRITE ASIMUT SHELL SCRIPT
"""

import os


def asimut_sh(ada, sh_filepath, asi_filepath_linux):
    """write asimut shell script to run asimut from hera on ada1-7"""
    
    sh = "#!/bin/bash\n"
    sh += "ssh ada%i <<'ENDSSH'\n" %ada
    sh += "module load 19i/numeric\n"
    sh += "module load 19i/rt\n"
    sh += "module load 19i/hdf-netcdf\n"
    sh += "\n"
    sh += "cd /home/iant/linux/ASIMUT/trunk\n"
    sh += "./asimut %s\n" %asi_filepath_linux
    sh += "ENDSSH\n"

    with open(sh_filepath, "w", newline='\n') as f:
        f.write(sh)    
        
    os.chmod(sh_filepath, 0o755)



