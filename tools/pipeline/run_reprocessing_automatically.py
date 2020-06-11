# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:40:34 2020

@author: iant

RUN REPROCESSING AUTOMATICALLY

1. MAKE A LOG OF ALL FILES PRESENT MATCHING REGEX IN ALL LEVELS
2. RUN CLEAN COMMAND
3. DETERMINE BEG AND END FOR PIPELINE (opt)
4. CHECK NO FILES REMAIN IN ANY LEVEL (MAKE A LOG)
4. RUN PIPELINE TO REGENERATE ALL
5. MAKE A LOG OF ALL NEW FILES PRESENT


"""

#TESTING=True
TESTING=False



import os
#import re
import time
import glob
import platform
import datetime
import subprocess


if platform.system() == "Windows":
    TESTING = True

if TESTING:
#    DATA_DIRECTORY = r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5"
    DATA_DIRECTORY = r"C:\Users\iant\Documents\DATA\hdf5_copy"
    LOG_DIRECTORY = r"C:\Users\iant\Dropbox\NOMAD\Python"
    SIMULATE = True
else:
    DATA_DIRECTORY = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5"
    LOG_DIRECTORY = r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/logs/reprocessing"
    SIMULATE = False
FORMAT_STR_SECONDS = "%Y-%m-%d %H:%M:%S"
FORMAT_STR_DAYS = "%Y-%m-%d"
FORMAT_STR_REGEX = "%Y%m%d"


REPROCESS_BY = "day"
#REPROCESS_BY = "month" #not yet ready - use day for now

PROCESSING_START = datetime.datetime(2020, 3, 21) #includes this date
PROCESSING_END = datetime.datetime(2020, 4, 14) #does not include this date
PROCESSING_START = datetime.datetime(2020, 1, 1) #includes this date
PROCESSING_END = datetime.datetime(2020, 2, 20) #does not include this date

REPROCESSING_LEVEL_START = "hdf5_l02a"
#REPROCESSING_LEVEL_START = "hdf5_l03c"
#REPROCESSING_LEVEL_END = "hdf5_l03k"
REPROCESSING_LEVEL_END = "hdf5_l10a"

LEVEL_NAMES = ["hdf5_l01a","hdf5_l01d","hdf5_l01e","hdf5_l02a","hdf5_l02b","hdf5_l02c","hdf5_l03a", \
               "hdf5_l03b","hdf5_l03c","hdf5_l03i","hdf5_l03j","hdf5_l03k","hdf5_l10a"]


levelsToBeCleaned = LEVEL_NAMES[LEVEL_NAMES.index(REPROCESSING_LEVEL_START):]
cleanCommand = "%s_and_above" %REPROCESSING_LEVEL_START


if REPROCESSING_LEVEL_START == "hdf5_l02a":
    levelCommands = [
            {"channel":"so", "from":"hdf5_l01e", "to":REPROCESSING_LEVEL_END}, \
            {"channel":"lno", "from":"hdf5_l01e", "to":REPROCESSING_LEVEL_END}, \
            {"channel":"uvis", "from":"hdf5_l01a", "to":REPROCESSING_LEVEL_END}, \
            ]

if REPROCESSING_LEVEL_START == "hdf5_l03c":
    levelCommands = [
            {"channel":"so", "from":"hdf5_l03b", "to":REPROCESSING_LEVEL_END}, \
            {"channel":"lno", "from":"hdf5_l03b", "to":REPROCESSING_LEVEL_END}, \
            {"channel":"uvis", "from":"hdf5_l03b", "to":REPROCESSING_LEVEL_END}, \
            ]






if REPROCESS_BY == "day":
    startDates = []
    currentDate = PROCESSING_START
    while currentDate < PROCESSING_END:
        startDates.append(currentDate)
        currentDate += datetime.timedelta(days=1)

elif REPROCESS_BY == "month":
    startDates = []
    currentDate = PROCESSING_START
    while currentDate < PROCESSING_END:
        startDates.append(currentDate)
        currentDate += datetime.timedelta(months=1)



for startDate in startDates:

    if REPROCESS_BY == "day":
        endDate = startDate + datetime.timedelta(days=1)
        dataDirectorySubstring = "%04i" %startDate.year + os.sep + "%02i" %startDate.month + os.sep + "%02i" %startDate.day
        regex = "%s_.*" %(datetime.datetime.strftime(startDate, FORMAT_STR_REGEX))
    
    
    beg_make_string = datetime.datetime.strftime(startDate, FORMAT_STR_DAYS)
    end_make_string = datetime.datetime.strftime(endDate, FORMAT_STR_DAYS)
    
    
        
    
    
    def writeLog(logPath, linesToWrite):
        """function to append log file"""
        with open(logPath+".txt", 'a') as logFile:
            for lineToWrite in linesToWrite:
                logFile.write(lineToWrite+'\n')
    
    def writeFileListLog(logString, dirSubstring, levelsToBeCleaned, header):
        """function to write list of all files to a log file"""
    
        levels = {}
        for levelToBeCleaned in levelsToBeCleaned:
            levels[levelToBeCleaned] = 0
            

        logDir = os.path.join(LOG_DIRECTORY, logString)
        if header != "":
            writeLog(logDir, [header])
        
        print("%s: " %(dirSubstring), end="")
        for level in levels.keys():
            
            folderName = "hdf5_level_%sp%s" %(level[-3:-2], level[-2:])
            print("%s " %(folderName), end="")
        
            folderPath = os.path.join(DATA_DIRECTORY, folderName, dirSubstring)
                
            files = [os.path.split(f)[1] for f in glob.glob(folderPath + os.sep + "**" + os.sep + "*.h5", recursive=True)]
    #        matchingFiles = list(filter(r.match, files))
            matchingFiles = sorted(files)
            levels[level] = len(matchingFiles)
            
            print("(%i); " %len(matchingFiles), end="")
        
            writeLog(logDir, ["### %s ###" %folderName.upper()])
            writeLog(logDir, matchingFiles)
        print("")
    
        return levels
    
    
    
    now = datetime.datetime.strftime(datetime.datetime.now(), FORMAT_STR_SECONDS)
    print("####### Starting reprocessing of %s at %s" %(beg_make_string, now))
    script_start_time = time.time()
    
    
    
    
    
    writeFileListLog(("reprocessing_log_%s_1_pre_clean_%s" %(beg_make_string, now)).replace(":","_").replace(" ","_"), dataDirectorySubstring, levelsToBeCleaned, "# %s # %s # %s # %s #" %(now, regex, beg_make_string, end_make_string))
    
    #scripts/run_pipeline.py clean hdf5_l03c_and_above --regex "20190[2-4][0-9][0-9]_.*_UVIS_"
    command = ["python3", "scripts/run_pipeline.py", "clean", cleanCommand, "--regex", regex]
    command_string = " ".join(command)
    print("#######", command_string)
    if not SIMULATE:
        subprocess.call(command)
    
    
    levels = writeFileListLog(("reprocessing_log_%s_2_post_clean_%s" %(beg_make_string, now)).replace(":","_").replace(" ","_"), dataDirectorySubstring, levelsToBeCleaned, "# %s # %s # %s # %s #" %(now, regex, beg_make_string, end_make_string))
    error = False
    if levels["hdf5_l10a"] != 0:
        print("Error: Not all level 1.0a files have been deleted")
        if not SIMULATE:
            error = True
    
    
    
    if not error:
        #scripts/run_pipeline.py --log WARNING make --from hdf5_l03b --to hdf5_l10a --beg 2019-02-01 --end 2019-05-01 --all --n_proc=8 --filter=".*_UVIS_.*"
        
        for levelCommand in levelCommands:
            command = ["scripts/run_pipeline.py", "--log", "WARNING", \
                       "make", "--from", levelCommand["from"], "--to", levelCommand["to"], \
                       "--beg", beg_make_string, "--end", end_make_string, "--all", "--n_proc=8", "--filter=.*_%s.*" %levelCommand["channel"].upper()]
            command_string = " ".join(command)
            print("#######", command_string)
            if not SIMULATE:
                subprocess.run(command)
        
    #    writeFileListLog(("reprocessing_log_3_after_reprocessing_%s" %now).replace(":","_"), r, header="### %s ### %s ### %s ###" %(regex, beg_make_string, end_make_string))
        writeFileListLog(("reprocessing_log_%s_3_after_reprocessing_%s" %(beg_make_string, now)).replace(":","_").replace(" ","_"), dataDirectorySubstring, levelsToBeCleaned, "# %s # %s # %s # %s #" %(now, regex, beg_make_string, end_make_string))
    
        script_end_time = time.time()
        script_elapsed_time = script_end_time - script_start_time
        print("Processing of %s finished at %s (duration = %s)" %(beg_make_string, datetime.datetime.now(), str(datetime.timedelta(seconds=script_elapsed_time)).split(".")[0]))
                     
                     
                     
                     
                 
                 