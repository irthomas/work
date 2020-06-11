# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:33:00 2020

@author: iant

FIND SPECIAL OBSERVATIONS, E.G. MERGED OCCULTATIONS
"""
import os
import glob
from datetime import datetime, timedelta

from tools.file.paths import paths

HDF5_FILENAME_FORMAT = "%Y%m%d_%H%M%S"


def find_merged_occultations(channel):
    """print a list of all merged occultations. Works only for UVIS at present"""
    file_level = "hdf5_level_1p0a"

    folderPath = os.path.join(paths["DATA_DIRECTORY"], file_level)
    
    
    hdf5_filenames = [os.path.split(f)[1] for f in glob.glob(folderPath + os.sep + "**" + os.sep + "*.h5", recursive=True)]
    
    
    #extract datetime from filename for all ingress occultations
    ingress_datetimes = []
    egress_datetimes = []
    for hdf5_filename in hdf5_filenames:
        split = os.path.splitext(hdf5_filename)[0].split("_")
        
        if channel.lower() == "uvis":
            if split[3] == "UVIS" and split[4] == "I":
                ingress_datetimes.append(split[0] + "_" + split[1])
            if split[3] == "UVIS" and split[4] == "E":
                egress_datetimes.append(split[0] + "_" + split[1])
        else:
            print("Only works for UVIS")
            return
            
    #now compare ingress and egress lists. If datetime found in ingress list then must be merged occultation
    
    merged_datetime_strings = []
    for egress_datetime in egress_datetimes:
        if egress_datetime in ingress_datetimes:
            merged_datetime_strings.append(egress_datetime)
            
    for merged_datetime in sorted(merged_datetime_strings):
        print(merged_datetime)
    return sorted(merged_datetime_strings)
        




def make_reprocessing_script(datetime_strings):
    """print a list of reprocessing lines to add to shell script
    Copy to script x.sh in master branch nomad_ops directory, then run using command ./x.sh"""

    pipeline_filter = ".*UVIS.*"
    pipeline_from = "hdf5_l03c"
    pipeline_to = "hdf5_l10a"
    pipeline_nproc = 8
    #make reprocessing script
    for file_datetime_string in datetime_strings:
        file_datetime = datetime.strptime(file_datetime_string, HDF5_FILENAME_FORMAT)
        
        file_minus_5_mins = file_datetime - timedelta(minutes=5)
        file_plus_5_mins = file_datetime + timedelta(minutes=5)
        
        pipeline_beg = datetime.strftime(file_minus_5_mins, "%Y-%m-%dT%H:%M:%S")
        pipeline_end = datetime.strftime(file_plus_5_mins, "%Y-%m-%dT%H:%M:%S")
        
        print("python3 scripts/run_pipeline.py --log INFO make --from %s --to %s --beg %s --end %s --all --n_proc=%i --filter=\"%s\"" \
              %(pipeline_from, pipeline_to, pipeline_beg, pipeline_end, pipeline_nproc, pipeline_filter))   
        
    

merged_datetime_strings = find_merged_occultations("UVIS")
make_reprocessing_script(merged_datetime_strings)
    