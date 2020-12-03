# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:21:51 2020

@author: iant

PARSE ERROR LOG
"""
import os
from datetime import datetime
import re

# RUN_PIPELINE_LOG_DIR = os.path.join(ROOT_STORAGE_PATH, "logs","run_pipeline")
RUN_PIPELINE_LOG_DIR = r"C:\Users\iant\Dropbox\NOMAD\Python"

LOG_NAME = "run_pipeline.log"

LOG_PATH = os.path.join(RUN_PIPELINE_LOG_DIR, LOG_NAME)

#function to delete existing error logs
def delete_log():
    
    if os.path.exists(LOG_PATH):
        print("Deleting %s" %(LOG_PATH))
        os.remove(LOG_PATH)
    
    return []



#function to save error logs if errors are found
def save_log():
    
    datetime_string = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    new_path =  "%s_%s%s" %(os.path.splitext(LOG_PATH)[0], datetime_string, os.path.splitext(LOG_PATH)[1])
    
    print("Renaming %s as %s" %(LOG_PATH, new_path))
    os.rename(LOG_PATH, new_path)

#function to parse error logs: find errors, tracebacks, etc.
def parse_errors():
    
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()
    
    error_list = []
    error_found = 0
    
    for i, line in enumerate(lines):
        line = line.rstrip("\n").replace("  ", "")
    
        if " ERROR " in line:
            error_list.append(line)
            # print("Error found", line)
            # stop()
        if "_RemoteTraceback" in line:
            error_found = 1
        
        if error_found > 0:
            if line == '"""':
                error_found += 1
                
            if error_found == 3:
                s = ""
                py_string = lines[i-3]
                py_path = py_string.split('"')[1]
                py_dir = os.path.dirname(py_path).split("/")[-1]
                py_name = os.path.basename(py_path)
                line_number = py_string.split('"')[2].rstrip("\n")
                
                function_string = lines[i-2].strip()
                
                
                error_line = lines[i-1]
                file_path = error_line.split("path='")[1].split("'")[0]
                error = error_line.split(" - ")[0]
                
                file_name = os.path.basename(file_path)
    
                s += "%s: %s/%s%s: %s => %s" %(file_name, py_dir, py_name, line_number, function_string, error)
                
                error_list.append(s)
                error_found = 0
            
            
    return error_list



def check_for_errors():
    
    error_list = parse_errors()
    
    if len(error_list) != 0:
        print("Errors found")
        # save_log()
        
        # send_email()
        
    else:
        print("No errors found")
        delete_log()
        
