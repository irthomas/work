# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:21:51 2020

@author: iant

FUNCTIONS FOR HANDLING PIPELINE MAIN ERROR LOG


"""


import os
from datetime import datetime
# import re
import platform

from nomad_ops.core.tools.send_email import send_bira_email

if platform.system() == "Windows":
    RUN_PIPELINE_LOG_DIR = r"C:\Users\iant\Dropbox\NOMAD\Python"
else:
    from nomad_ops.config import ROOT_STORAGE_PATH
    RUN_PIPELINE_LOG_DIR = os.path.join(ROOT_STORAGE_PATH, "logs","run_pipeline")


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
    
    if os.path.exists(LOG_PATH):
        datetime_string = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
        new_path =  "%s_%s%s" %(os.path.splitext(LOG_PATH)[0], datetime_string, os.path.splitext(LOG_PATH)[1])
        
        print("Renaming %s as %s" %(LOG_PATH, new_path))
        os.rename(LOG_PATH, new_path)
    else:
        print("Log file not found", LOG_PATH)

#function to parse error logs: find errors, tracebacks, etc.
def parse_errors():

        
    if not os.path.exists(LOG_PATH):
        print("Error: no log file exists at", LOG_PATH)
        return ["Error: no log file exists at", LOG_PATH]
    else:
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
    
        error_list = []
        error_found = 0
        
        for i, line in enumerate(lines):
            line = line.rstrip("\n").replace("  ", "")
        
            if " ERROR " in line:
                error_list.append(line)

            if "_RemoteTraceback" in line:
                error_found = 1
            
            if error_found > 0:
                if line == '"""':
                    error_found += 1
                    
                if error_found == 3:
                    parsing_error = False
                    s = ""

                    py_string = lines[i-3]
                    
                    if '"' in py_string:
                        py_path = py_string.split('"')[1]
                        
                        py_dir = os.path.dirname(py_path).split("/")[-1]
                        py_name = os.path.basename(py_path)
                        
                        if '"' in py_string:
                            line_number = py_string.split('"')[2].rstrip("\n")
                        else:
                            parsing_error = True
                    else:
                        parsing_error = True
                            
                    function_string = lines[i-2].strip()
                    
                    
                    error_line = lines[i-1]
                    error_line = error_line.replace("\\", "")
                    
                    if "path='" in error_line:
                        file_path = error_line.split("path='")[1].split("'")[0]
                        file_name = os.path.basename(file_path)
                        error = error_line.split(" - ")[0]
            
                    
                    else:
                        parsing_error = True
     
                    
                    if not parsing_error:
                         s += "%s: %s/%s%s: %s => %s" %(file_name, py_dir, py_name, line_number, function_string, error)
                    else:
                        #simply print last 3 lines of pipeline error
                        s = "Cannot parse error from pipeline log (" + lines[i-3].rstrip("\n") +"; "+ lines[i-2].rstrip("\n") +"; "+ lines[i-1].rstrip("\n") +")"
                    print(s)
                    
                    error_list.append(s)
                    error_found = 0
        
            
    return error_list



def check_for_errors(email=True):
    
    error_list = parse_errors()
    
    if len(error_list) != 0:
        print("Errors found - saving log file")
        save_log()
        
        if email:
            print("Errors found - sending email")
            send_bira_email("### Pipeline Errors ###", "\n".join(error_list), print_output=True)
        else:
            print("### Pipeline Errors ###", "\n".join(error_list))
        
    else:
        print("No errors found")
        delete_log()
        
