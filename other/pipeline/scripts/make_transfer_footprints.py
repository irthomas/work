# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:30:34 2021

@author: iant

TRANSFER NOMAD OBSERVATION FOOTPRINTS TO PSUP

FILES WILL HAVE THE SAME FILENAME WITH THE SPICE METAKERNEL VERSION APPENDED

"""

import os
import sys
import re
import json
import platform
# from datetime import datetime
import argparse
# import time

if os.path.exists("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD"):
    sys.path.append(".")


from nomad_ops.core.footprints.config import FILE_LEVEL, FOOTPRINT_FILE_DIR, H5_ROOT_DIR

from nomad_ops.core.footprints.db_functions import get_h5s_from_cache

# from nomad_ops.core.footprints.footprint import footprint
from nomad_ops.core.footprints.footprint2 import footprint_json

from nomad_ops.core.footprints.functions import \
    get_mk_version, prepare_footprint_tree, transfer_files, \
    make_datetime, dts_from_h5s, filter_dts



windows = platform.system() == "Windows"

#new types need to be added manually
# OUTPUT_TYPES = ["shp", "geojson"]
OUTPUT_TYPES = ["geojson"]




MANUAL_SELECTION = True #select files based on regex below. Overruled if arg given
# MANUAL_SELECTION = False

MANUAL_SELECTION_REGEX = re.compile(".*")




def make(args):
    if args.filter:
        MANUAL_SELECTION = True
        MANUAL_SELECTION_REGEX = re.compile(args.filter)
    else:
        MANUAL_SELECTION = False


    
    h5s_all, h5_paths_all = get_h5s_from_cache(H5_ROOT_DIR, FILE_LEVEL)
    
    if MANUAL_SELECTION:
        print("Manually generating files based on regex filter %s" %(MANUAL_SELECTION_REGEX.pattern))

        
        #get indices of matching files
        regex_ixs = [i for i, h5 in enumerate(h5s_all) if re.search(MANUAL_SELECTION_REGEX.pattern, h5)]
        print("Regex matches %i files out of %i" %(len(regex_ixs), len(h5s_all)))
    else:
        print("No regex filter applied")
        #get indices of all files
        regex_ixs = list(range(len(h5s_all)))


    if args.beg or args.end:
        print("Filtering by time")
        h5_dts = dts_from_h5s(h5s_all)
        time_ixs = filter_dts(h5_dts, beg_dt=args.beg, end_dt=args.end)
        
        matching_ixs = list(set(regex_ixs).intersection(time_ixs))
    else:
        matching_ixs = regex_ixs    

    print("%i matching files found" %len(matching_ixs))        
        

    #loop through filtered files
    for i in matching_ixs:
        
        # print("Making footprint for %s" %h5s_all[i])

        if windows:
            h5_paths_all[i] = os.path.normcase(h5_paths_all[i].replace("/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD", r"E:\DATA"))
        
        #check if file(s) already exist
        #get metakernel version number
        mk_version = get_mk_version(h5_paths_all[i])
        
        #make year/month/day subdirectories
        footprint_dir_path = prepare_footprint_tree(h5s_all[i])
        
        outputs = {"shp":"", "geojson":""}
        
        #save as a shape file
        if "shp" in OUTPUT_TYPES:
            #remove .h5 from filename first
            filename = "%s_%05i.shp" %(h5s_all[i].replace(".h5", ""), mk_version)
            filepath = os.path.join(footprint_dir_path, filename)
            
            if os.path.exists(filepath) and not args.all:
                print("File %s already exists, skipping" %filename)
            else:
                outputs["shp"] = filepath

        #save as geojson
        if "geojson" in OUTPUT_TYPES:
            #remove .h5 from filename first
            filename = "%s_%05i.geojson" %(h5s_all[i].replace(".h5", ""), mk_version)
            filepath = os.path.join(footprint_dir_path, filename)
            
            if os.path.exists(filepath) and not args.all:
                print("File %s already exists, skipping" %filename)
            else:
                outputs["geojson"] = filepath

        
        
        #make footprints
        if outputs["shp"] != "" or outputs["geojson"] != "":
            print("Making footprints for file %s" %h5s_all[i])
        
            
            #make geopandas data frame
            # gdf_all = footprint(h5s_all[i], h5_paths_all[i])
            
            #make json dictionary
            jsond = footprint_json(h5_paths_all[i]) 
        
        
            # if outputs["shp"] != "":
            #     print("Saving to shape file")
            #     gdf_all.to_file(outputs["shp"])
            if outputs["geojson"] != "":
                print("Saving to geojson file")
                # gdf_all.to_file(outputs["geojson"], driver='GeoJSON')
                with open(outputs["geojson"], "w") as f:
                    json.dump(jsond, f, indent=2)


def transfer(args):
    print("Preparing transfer")

    if args.now:
        WAIT_FOR_USER = False
    else:
        WAIT_FOR_USER = True

    
    #use cache to get list of directories for regex and start/end filters
    #faster than selecting all possible
    h5s_all, h5_paths_all = get_h5s_from_cache(H5_ROOT_DIR, FILE_LEVEL)
    
    
    if MANUAL_SELECTION:
        print("Manually selecting files based on regex filter %s" %(MANUAL_SELECTION_REGEX.pattern))
        
        #get indices of matching files
        regex_ixs = [i for i, h5 in enumerate(h5s_all) if re.search(MANUAL_SELECTION_REGEX.pattern, h5)]

    
    else:
        print("No regex filter applied")
        #get indices of all files
        regex_ixs = list(range(len(h5s_all)))


    if args.beg or args.end:
        h5_dts = dts_from_h5s(h5s_all)
        time_ixs = filter_dts(h5_dts, beg_dt=args.beg, end_dt=args.end)
        
        matching_ixs = list(set(regex_ixs).intersection(time_ixs))
    else:
        matching_ixs = regex_ixs            

    
    #now get list of directories containing the matching files
    footprints_list = []
    for i in matching_ixs:
        h5_path = h5_paths_all[i]
        
        path_split = os.path.normpath(h5_path).split(os.path.sep)
        ymd = os.path.join(*path_split[-4:-1])
        
        #get list of footprints in directory
        footprint_dir = os.path.join(FOOTPRINT_FILE_DIR, ymd)
        footprints_in_dir = os.listdir(footprint_dir)              
        
        matching_footprints = [fp for fp in footprints_in_dir if h5s_all[i].replace(".h5", "") in fp]
        footprints_list.extend([os.path.join(footprint_dir, fp) for fp in matching_footprints])

    print("%i footprints are to be transferred" %(len(footprints_list)))
        
        
    
       
    if WAIT_FOR_USER:
        input("Press any key to continue")

    if not windows:
            transfer_files(footprints_list)
            
    else:
        print("Warning: Running on windows; no files transferred")
    




if __name__ == "__main__":
    
    #if --now keyword given, don't wait for user to confirm transfer
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Choose a command')
    
    parser_a = subparsers.add_parser("make", help="Make geometry files")

    parser_a.add_argument('--beg', type=make_datetime, metavar="YYYY-MM-DDThh:mm:ss", help="Begin time")
    parser_a.add_argument('--end', type=make_datetime, metavar="YYYY-MM-DDThh:mm:ss", help="End time")
    parser_a.add_argument('--all', dest="all", action='store_true', help="Remake all geometry files")
    parser_a.add_argument("--filter", help="Make files based on regex filter", metavar="RE")
    parser_a.set_defaults(func=make)

    parser_b = subparsers.add_parser("transfer", help="Transfer geometry files")

    parser_b.add_argument('--beg', type=make_datetime, metavar="YYYY-MM-DDThh:mm:ss", help="Begin time")
    parser_b.add_argument('--end', type=make_datetime, metavar="YYYY-MM-DDThh:mm:ss", help="End time")
    parser_b.add_argument("--filter", help="Transfer files based on regex filter", metavar="RE")
    parser_b.add_argument("--now", help="Transfer without waiting for user", action="store_true")
    parser_b.set_defaults(func=transfer)

    args = parser.parse_args()
    # print(args)
    args.func(args)



