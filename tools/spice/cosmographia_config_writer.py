# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 13:45:55 2025

@author: iant

WRITE SPICE CONFIG FILES FOR COSMOGRAPHIA
GET OBSERVATION TIME FROM H5 FILENAME
WRITE AND EXECUTE SCRIPT AND BATCH FILE TO LOAD COSMO WITH CONFIG, AND ZOOM TO GIVEN OBJECT AT SPECIFIED TIME

MUST USE PATH_VALUES IN MK, OTHERWISE FULL PATH IS TOO LONG ON EACH LINE
MUST USE ABSOLUTE PATHS TO ALL KERNELS IN THE MK, ANY IN THE SAME DIRECTORY WONT BE FOUND



"""

import subprocess
import os
from datetime import datetime, timedelta
import json


# MISSION = "NOMAD_OPS"
MISSION = "SOVENIRO"

if MISSION == "NOMAD_OPS":
    KERNEL_ROOT_DIR = r"C:\Users\iant\Documents\DATA\local_spice_kernels_ops\kernels"
    COSMO_SPICE_DIR = "EM16"
if MISSION == "SOVENIRO":
    KERNEL_ROOT_DIR = r"C:\Users\iant\Documents\DATA\soveniro_kernels\kernels"
    COSMO_SPICE_DIR = "SOVENIRO"

COSMO_ROOT_DIR = r"C:\Users\iant\cosmographia-4.2"
SCENARIO_DIR = "py_conf"
COSMO_EXE_NAME = "Cosmographia.exe"

# write new IK and add to MK?
NEW_IK = True
# NEW_IK = False

if MISSION == "NOMAD_OPS":
    # occultation example
    h5 = "20240607_212109_1p0a_SO_A_E_132"
if MISSION == "SOVENIRO":
    h5 = "20300101_010000_test"

# phobos
# h5 = "20250411_203101_0p3a_LNO_1_P_166"

# "20250628_083013_0p3a_LNO_1_P_166",  # tracking
# "20250622_070511_0p3a_LNO_1_P_166",
# "20250619_101826_0p3a_LNO_1_P_166",
# "20250619_022647_0p3a_LNO_1_P_166",
# "20250411_203101_0p3a_LNO_1_P_166",
# "20250408_080101_0p3a_LNO_1_P_166",
# "20250405_190547_0p3a_LNO_1_P_166",
# "20250402_221907_0p3a_LNO_1_P_166",
# "20250402_142717_0p3a_LNO_1_P_166",
# "20241229_103444_0p3a_LNO_1_P_166",  # inertial
# "20241213_052910_0p3a_LNO_1_P_166",
# "20241210_084211_0p3a_LNO_1_P_166",
# "20241207_040411_0p3a_LNO_1_P_166",
# "20241011_012228_0p3a_LNO_1_P_174",
# "20241004_235732_0p3a_LNO_1_P_174",
# "20241001_191914_0p3a_LNO_1_P_174",
# "20240922_210734_0p3a_LNO_1_P_174",
# "20240911_224710_0p3a_LNO_1_P_174",
# "20240831_034846_0p3a_LNO_1_P_165",  # 6 rows
# "20240825_022347_0p3a_LNO_1_P_165",
# "20240819_005847_0p3a_LNO_1_P_165",
# "20240805_163944_0p3a_LNO_1_P_165",
# "20240702_042211_0p3a_LNO_1_P_165",
# "20240627_023139_0p3a_LNO_1_P_165",
# "20240620_092319_0p3a_LNO_1_P_165",


dt = datetime.strptime(h5[0:15], "%Y%m%d_%H%M%S") + timedelta(seconds=(60*15))

dt_str = datetime.strftime(dt, "%Y-%m-%dT%H:%M:%S")

if MISSION == "NOMAD_OPS":
    # phobos example
    preset = {
        "description": "Phobos LNO reduced FOV",
        # "channel": "LNO",
        "parent": "TGO",
        "planet": "Mars",
        "pointing": "nadir",
        "target": "Phobos",
        # "FOV": 144/2,
        "FOV": 12/2,
        "distance": 150,
        "time": dt_str,
        "goto": "Phobos",
        "opacity": 0.0,
    }

    # occultation example
    preset = {
        "description": "Solar occultation",
        # "channel": "SO",
        "parent": "TGO",
        "planet": "Mars",
        "pointing": "occultation",
        "target": "Sun",
        "FOV": 24/2,
        "distance": 150,
        "time": dt_str,
        "goto": "TGO",
        "opacity": 0.8,
    }

    if preset["pointing"] == "nadir":
        channel_names = ["TGO_NOMAD_LNO_OPS_NAD", "TGO_NOMAD_UVIS_NAD"]
    if preset["pointing"] == "occultation":
        channel_names = ["TGO_NOMAD_SO", "TGO_NOMAD_UVIS_OCC"]
    mk_in_path = os.path.join(KERNEL_ROOT_DIR, "mk", "em16_ops.tm")

if MISSION == "SOVENIRO":
    preset = {
        "description": "Test",
        # "channel": "SOLARO",
        "parent": "SOVENIRO",
        "planet": "Venus",
        "pointing": "nadir",
        "target": "Venus",
        "FOV": 24/2,
        "distance": 150,
        "time": dt_str,
        "goto": "VENUS",
        "opacity": 0.0,
    }

    if preset["pointing"] == "nadir":
        channel_names = ["SOVENIRO_SOLARO"]
    if preset["pointing"] == "occultation":
        channel_names = ["SOVENIRO_SOLARO"]
    mk_in_path = os.path.join(KERNEL_ROOT_DIR, "mk", "soveniro.tm")


spacecraft_name = "%s_SPACECRAFT" % preset["parent"]

# file dict
fd = {
    "mk": ["mk.tm", os.path.join(COSMO_ROOT_DIR, COSMO_SPICE_DIR, SCENARIO_DIR, "mk.tm")],
    "ik": ["ik.ti", os.path.join(KERNEL_ROOT_DIR, "ik", "ik.ti")],
    "scen": ["load_scenario.json", os.path.join(COSMO_ROOT_DIR, COSMO_SPICE_DIR, SCENARIO_DIR, "load_scenario.json")],
    "conf": ["conf.json", os.path.join(COSMO_ROOT_DIR, COSMO_SPICE_DIR, SCENARIO_DIR, "conf.json")],
    "sensor": ["sensor.json", os.path.join(COSMO_ROOT_DIR, COSMO_SPICE_DIR, SCENARIO_DIR, "sensor.json")],
    "script": ["script.py", os.path.join(COSMO_ROOT_DIR, COSMO_SPICE_DIR, SCENARIO_DIR, "script.py")],
    "bat": ["cosmo.bat", os.path.join(COSMO_ROOT_DIR, COSMO_SPICE_DIR, SCENARIO_DIR, "cosmo.bat")],
}


KERNELS_TO_SKIP = [
    "_acs_", "_cassis_", "_hga_", "_edm", "_sa_", "_relay_locations_", "_frend_", "earth_", "estrack_", "earthstns_", "phobos_",
    "_scp_", "_scm_", "_spm_", "_sam_", "_fpp_"
]


scenario = {
    "version": "1.0",
    "name": preset["parent"],
    "require": [
        fd["conf"][0],
        fd["sensor"][0],
    ]
}


sensor = {
    "version": "1.0",
    "name": preset["parent"],
    "items": [
        {
            "class": "sensor",
            "name": channel_names[0],
            "parent": preset["parent"],
            "center": preset["parent"],
            "trajectoryFrame": {
                "type": "BodyFixed",
                "body": preset["parent"]
            },
            "geometry": {
                "type": "Spice",
                "instrName": channel_names[0],
                "target": preset["target"],
                "range": 1000,
                "rangeTracking": True,
                "frustumColor": [
                    0.0,
                    1.0,
                    1.0
                ],
                "frustumOpacity": preset["opacity"],
                "gridOpacity": preset["opacity"],
                "footprintOpacity": 0.8,
                "sideDivisions": 125,
                "onlyVisibleDuringObs": False
            }
        },
        # {
        #     "class": "sensor",
        #     "name": channel_names[1],
        #     "parent": preset["parent"],
        #     "center": preset["parent"],
        #     "trajectoryFrame": {
        #         "type": "BodyFixed",
        #         "body": preset["parent"]
        #     },
        #     "geometry": {
        #         "type": "Spice",
        #         "instrName": channel_names[1],
        #         "target": preset["target"],
        #         "range": 1000,
        #         "rangeTracking": True,
        #         "frustumColor": [
        #             0.0,
        #             1.0,
        #             0.0
        #         ],
        #         "frustumOpacity": preset["opacity"],
        #         "gridOpacity": preset["opacity"],
        #         "footprintOpacity": 0.8,
        #         "sideDivisions": 125,
        #         "onlyVisibleDuringObs": False
        #     }
        # }

    ]
}


config = {
    "version": "1.0",
    "name": preset["parent"],
    "spiceKernels": [
        fd["mk"][0]
    ],
    "items": [
        {
            "class": "spacecraft",
            "name": preset["parent"],
            "startTime": "2018-04-21 12:00:00 UTC",
            "endTime": "2030-02-01 01:00:00 UTC",
            "center": preset["planet"],
            "trajectory": {
                "type": "Spice",
                "target": preset["parent"],
                "center": preset["planet"]
            },
            "bodyFrame": {
                "type": "Spice",
                "name": spacecraft_name
            },
            "geometry": {
                "type": "Mesh",
                "meshRotation": [
                    0.5,
                    0.5,
                    0.5,
                    0.5
                ],
                "source": "../models/ExomarsTGO_v0.5_Body_NoEDM_with_solar_panels.3DS",
                "size": 100
            },
            "label": {
                "color": [
                    1,
                    1,
                    1
                ]
            },
            "trajectoryPlot": {
                "color": [
                    1,
                    1,
                    1
                ],
                "duration": "3d",
                "lead": 0,
                "fade": 0.25
            }
        }
    ]
}

if MISSION == "NOMAD_OPS":
    ik = f"""
    INS-143310_FOV_FRAME                 = 'TGO_NOMAD_LNO'
    INS-143310_FOV_SHAPE                 = 'RECTANGLE'
    INS-143310_BORESIGHT                 = (0.000000       0.000000     1.000000)
    INS-143310_FOV_CLASS_SPEC            = 'ANGLES'
    INS-143310_FOV_REF_VECTOR            = (1.000000       0.000000     0.000000)
    INS-143310_FOV_REF_ANGLE             = (  75.000000 )
    INS-143310_FOV_CROSS_ANGLE           = (   2.000000 )
    INS-143310_FOV_ANGLE_UNITS           = 'ARCMINUTES'
    INS-143311_FOV_FRAME                 = 'TGO_NOMAD_LNO_OPS_NAD'
    INS-143311_FOV_SHAPE                 = 'RECTANGLE'
    INS-143311_BORESIGHT                 = (0.000000       0.000000     1.000000)
    INS-143311_FOV_CLASS_SPEC            = 'ANGLES'
    INS-143311_FOV_REF_VECTOR            = (1.000000       0.000000     0.000000)
    INS-143311_FOV_REF_ANGLE             = (  {preset["FOV"]} )
    INS-143311_FOV_CROSS_ANGLE           = (   2.000000 )
    INS-143311_FOV_ANGLE_UNITS           = 'ARCMINUTES'
    INS-143312_FOV_FRAME                 = 'TGO_NOMAD_LNO_OPS_OCC'
    INS-143312_FOV_SHAPE                 = 'RECTANGLE'
    INS-143312_BORESIGHT                 = (0.000000       0.000000     1.000000)
    INS-143312_FOV_CLASS_SPEC            = 'ANGLES'
    INS-143312_FOV_REF_VECTOR            = (1.000000       0.000000     0.000000)
    INS-143312_FOV_REF_ANGLE             = (  75.000000 )
    INS-143312_FOV_CROSS_ANGLE           = (   2.000000 )
    INS-143312_FOV_ANGLE_UNITS           = 'ARCMINUTES'
    INS-143320_FOV_FRAME                 = 'TGO_NOMAD_SO'
    INS-143320_FOV_SHAPE                 = 'RECTANGLE'
    INS-143320_BORESIGHT                 = (0.000000       0.000000     1.000000)
    INS-143320_FOV_CLASS_SPEC            = 'ANGLES'
    INS-143320_FOV_REF_VECTOR            = (1.000000       0.000000     0.000000)
    INS-143320_FOV_REF_ANGLE             = (  {preset["FOV"]} )
    INS-143320_FOV_CROSS_ANGLE           = (   1.000000 )
    INS-143320_FOV_ANGLE_UNITS           = 'ARCMINUTES'
    INS-143331_FOV_FRAME                 = 'TGO_NOMAD_UVIS_NAD'
    INS-143331_FOV_SHAPE                 = 'CIRCLE'
    INS-143331_BORESIGHT                 = (0.000000       0.000000     1.000000)
    INS-143331_FOV_CLASS_SPEC            = 'ANGLES'
    INS-143331_FOV_REF_VECTOR            = (1.000000       0.000000     0.000000)
    INS-143331_FOV_REF_ANGLE             = (  21.50000   )
    INS-143331_FOV_ANGLE_UNITS           = 'ARCMINUTES'
    INS-143332_FOV_FRAME                 = 'TGO_NOMAD_UVIS_OCC'
    INS-143332_FOV_SHAPE                 = 'CIRCLE'
    INS-143332_BORESIGHT                 = (0.000000       0.000000     1.000000)
    INS-143332_FOV_CLASS_SPEC            = 'ANGLES'
    INS-143332_FOV_REF_VECTOR            = (1.000000       0.000000     0.000000)
    INS-143332_FOV_REF_ANGLE             = (   1.00000   )
    INS-143332_FOV_ANGLE_UNITS           = 'ARCMINUTES'
    INS-143310_PLATFORM_ID  = ( -143000 )
    INS-143311_PLATFORM_ID  = ( -143000 )
    INS-143312_PLATFORM_ID  = ( -143000 )
    INS-143320_PLATFORM_ID  = ( -143000 )
    INS-143331_PLATFORM_ID  = ( -143000 )
    INS-143332_PLATFORM_ID  = ( -143000 )
    """
if MISSION == "SOVENIRO":
    ik = f"""
    INS-222320_FOV_FRAME                 = 'SOVENIRO_SOLARO'
    INS-222320_FOV_SHAPE                 = 'RECTANGLE'
    INS-222320_BORESIGHT                 = (0.000000       0.000000     1.000000)
    INS-222320_FOV_CLASS_SPEC            = 'ANGLES'
    INS-222320_FOV_REF_VECTOR            = (1.000000       0.000000     0.000000)
    INS-222320_FOV_REF_ANGLE             = (  {preset["FOV"]} )
    INS-222320_FOV_CROSS_ANGLE           = (   1.000000 )
    INS-222320_FOV_ANGLE_UNITS           = 'ARCMINUTES'
    INS-222320_PLATFORM_ID  = ( -222000 )
    """

script = f"""
import os
import cosmoscripting

cosmo = cosmoscripting.Cosmo()

time = "{preset["time"]}"
cosmo.setTime(time)

path = os.path.join(cosmo.scriptDir(), "{fd["scen"][0]}")
cosmo.displayNote("%s" % path, 6).wait(1)

cosmo.loadCatalogFile(path)

cosmo.displayNote("%s loaded" % path, 3).wait(1)

rate = 1
cosmo.setTimeRate(rate)


cosmo.gotoObject({preset["parent"]}, 1)

cosmo.wait(1)

name = "{preset["goto"]}"
cosmo.gotoObject(name, 1)
cosmo.moveToDistanceFromCenter({preset["distance"]}, 1)
"""

mk = []
mk.append(r"PATH_VALUES = ('%s')" % KERNEL_ROOT_DIR)
mk.append("PATH_SYMBOLS = ('KERNELS')")
# mk = [r"\begindata"]
mk.append("KERNELS_TO_LOAD = (")
with open(mk_in_path, "r") as f:
    lines = f.readlines()

for line in lines:
    if "$KERNELS" in line:
        # clean line
        line_strip = line.strip()

        if NEW_IK:
            # replace ik
            if "em16_tgo_nomad_v" in line_strip:
                line_strip = "'$KERNELS/ik/%s'" % fd["ik"][0]
            if "soveniro_solaro_v" in line_strip:
                line_strip = "'$KERNELS/ik/%s'" % fd["ik"][0]

        if not any(substring in line_strip for substring in KERNELS_TO_SKIP):
            mk.append(line_strip)

mk.append(")")
# mk.append(r"\begintext")


mk = [mk_line.replace("\\", "/") for mk_line in mk]

mk = [r"\begindata", *mk, r"\begintext"]

if NEW_IK:
    ik = [r"\begindata", ik, r"\begintext"]


if NEW_IK:
    with open(fd["ik"][1], "w") as f:
        for line in ik:
            f.write(line + "\n")

with open(fd["mk"][1], "w") as f:
    for line in mk:
        f.write(line + "\n")

with open(fd["scen"][1], "w") as f:
    f.write(json.dumps(scenario, indent=4, sort_keys=False))

with open(fd["conf"][1], "w") as f:
    f.write(json.dumps(config, indent=4, sort_keys=False))

with open(fd["sensor"][1], "w") as f:
    f.write(json.dumps(sensor, indent=4, sort_keys=False))

with open(fd["script"][1], "w") as f:
    f.write(script)

# make batch file to load script, which loads kernels and skips to correct time
# chdir C:\Users\iant\cosmographia-4.2
# Cosmographia.exe -p EM16\py_conf\script.py

script_path = os.path.join(COSMO_SPICE_DIR, SCENARIO_DIR, fd["script"][0])

with open(fd["bat"][1], "w") as f:
    f.write("chdir %s\n" % (COSMO_ROOT_DIR))
    f.write("%s -p %s\n" % (COSMO_EXE_NAME, script_path))

subprocess.Popen(fd["bat"][1])
print("Loading cosmographia, please wait")
