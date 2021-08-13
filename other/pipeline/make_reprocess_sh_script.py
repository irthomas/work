# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:39:09 2021

@author: iant


"""




DATES = [
    ["2018-01-01", "2019-01-01"], 
    ["2019-01-01", "2020-01-01"],
    ["2020-01-01", "2021-01-01"],
    ["2021-01-01", "2021-08-01"],
    ]

LEVELS = [
    # {"in":"inserted",  "out":"raw",       "filter":""},
    # {"in":"raw",       "out":"hdf5_l01a", "filter":""},
    # {"in":"hdf5_l01a", "out":"hdf5_l01d", "filter":".*(SO|LNO).*"},
    # {"in":"hdf5_l01d", "out":"hdf5_l01e", "filter":".*(SO|LNO).*"},
    # {"in":"hdf5_l01e", "out":"hdf5_l02a", "filter":".*(SO|LNO).*"},
    # {"in":"hdf5_l01a", "out":"hdf5_l02a", "filter":".*UVIS.*"},
    # {"in":"hdf5_l02a", "out":"hdf5_l02b", "filter":".*UVIS.*"},
    # {"in":"hdf5_l02b", "out":"hdf5_l03b", "filter":".*UVIS.*"},
    {"in":"hdf5_l03b", "out":"hdf5_l03c", "filter":".*UVIS_D.*"},
    {"in":"hdf5_l03c", "out":"hdf5_l10a", "filter":".*UVIS_D.*"},
    # {"in":"hdf5_l02a", "out":"hdf5_l03a", "filter":".*(SO|LNO).*"},
    # {"in":"hdf5_l03a", "out":"hdf5_l10a", "filter":".*(SO|LNO).*"},


    ]


s = "#!/bin/bash\n\n"
s += '#generated automatically by make_reprocess_sh_script.sh\n\n'
s += '#cd /bira-iasb/projects/NOMAD/Instrument/SOFTWARE-FIRMWARE/nomad_ops\n'
s += '#./scripts/run_as_nomadr /bin/bash -p\n'

s += '#cd /bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/db\n'
s += '#rm -Rf raw\n'
s += '#or selective delete:\n'
s += '#cd hdf5_level_1p0a\n'
s += '#find . -type f -name "*UVIS*.h5" -delete\n'


s += '#cd /bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5/\n'
s += '#rm -Rf hdf5_level_{0p1a,0p1d,0p1e,0p2a,etc.}\n\n\n'

for level_d in LEVELS:
    
    level_s = level_d["in"]
    level_e = level_d["out"]
    filt = level_d["filter"]

    for dates in DATES:
        date_s, date_e = dates
    

        #update log
        s += 'python3 scripts/pipeline_log.py "Starting %s-to-%s %s reprocessing"\n' %(level_s, level_e, date_s[0:4])
        #reprocess all
        s += './scripts/run_as_nomadr ./scripts/run_pipeline.py --log WARNING make --beg %s --end %s --from %s --to %s --n_proc=8 --all' %(date_s, date_e, level_s, level_e)
        
        if filt == "":
            s += "\n"
        else:
            s += " --filter='%s'\n" %filt
        
        #check error log
        s += 'python3 scripts/check_pipeline_log.py\n'
        #log message
        s += 'python3 scripts/pipeline_log.py "%s-to-%s %s reprocessing finished"\n' %(level_s, level_e, date_s[0:4])
        #wait 2 seconds
        s += 'read -t 2\n'
        s += '\n\n'

