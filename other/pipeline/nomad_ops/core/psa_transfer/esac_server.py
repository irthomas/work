# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:14:47 2021

@author: iant
"""

import subprocess
import platform
from datetime import datetime
import time
from urllib.parse import urlparse

from nomad_ops.core.psa_transfer.config import \
    ESA_PSA_CAL_URL, LOG_FORMAT_STR
    
from nomad_ops.core.psa_transfer.get_psa_logs import \
    last_process_datetime

from nomad_ops.core.psa_transfer.transfer_psa_logs_from_esa import psa_logs_esa_to_bira


windows = platform.system() == "Windows"

def n_products_in_queue():
    """first check for any products not yet moved from nmd directory to PSA staging area"""
 
    esa_p_url = urlparse(ESA_PSA_CAL_URL)

    ssh_cmd2 = ["ssh", esa_p_url.netloc, "ls nmd -1 | wc -l"]
    pipe = subprocess.Popen(ssh_cmd2,
                            shell=False,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    output = pipe.communicate()[0]
    
    n_files = int(output.decode().strip()) - 2 #2 subdirs
    
    return n_files




def wait_until_no_activity():
    
    queue_size = 1
    last_process_delta = 1.0

    #before starting transfer, check that processing is not ongoing and that there are no products waiting
    #loop until ESA server is ready
    while queue_size != 0 or last_process_delta < 3600.:

        #first check for new PSA logs on ESA server
        if not windows:
            psa_logs_esa_to_bira()
        
        
        
        #check active PSA log for datetime of last entry. If within last hour, or files still in queue, wait 1 hour
        if windows:
            queue_size = 0
        else:
            queue_size = n_products_in_queue()
        
        last_process = last_process_datetime()
        last_process_delta = (datetime.now() - last_process).total_seconds()

        #wait an hour
        now = datetime.strftime(datetime.now(), LOG_FORMAT_STR)
        if queue_size != 0:
            print("Time is %s; there are %i files remaining in the ESA server queue" %(now, queue_size))
        if last_process_delta < 3600.:
            print("Time is %s; files were being processed on ESA server %i minutes ago" %(now, last_process_delta / 60.))
        
        if queue_size != 0 or last_process_delta < 3600.:
            for i in range(60):
                time.sleep(60)


    print("ESA server has finished previous activity")

