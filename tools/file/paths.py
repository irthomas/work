# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:14:10 2020

@author: iant

PATHS

"""
import os
import platform

if platform.system() == "Windows":
    SYSTEM = "Windows"
else:
    SYSTEM = "Linux"

paths = {}


if SYSTEM == "Linux":  # linux system
    paths["BASE_DIRECTORY"] = os.path.normcase(r"/home/iant/linux/Python")
    paths["DATA_DIRECTORY"] = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5")
    # paths["DATA_DIRECTORY"] = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/test/iant/hdf5")
    paths["DATASTORE_ROOT_DIRECTORY"] = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD")
    paths["LOCAL_DIRECTORY"] = os.path.normcase(r"/home/iant/linux/DATA")
    paths["RETRIEVAL_DIRECTORY"] = os.path.normcase(r"/home/iant/linux/input_tools/Tools")
    paths["FOOTPRINT_DIRECTORY"] = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/footprint")
    paths["DB_DIRECTORY"] = os.path.normcase(r"/bira-iasb/projects/work/NOMAD/Science/ian/db")
    paths["ANIMATION_DIRECTORY"] = os.path.normcase(r"/home/iant/linux/DATA/animations")
    paths["SIMULATION_DIRECTORY"] = os.path.normcase(r"/home/iant/linux/DATA/simulations")
    paths["KERNEL_DIRECTORY"] = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/kernels/mk")
    paths["KERNEL_DIRECTORY_OPS"] = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/kernels/mk")
    paths["COP_TABLE_DIRECTORY"] = os.path.normcase(r"/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/cop_tables")
    paths["REFERENCE_DIRECTORY"] = os.path.normcase(r"/home/iant/linux/Python/reference_files")
    paths["PFM_AUXILIARY_FILES"] = os.path.normcase(r"/bira-iasb/projects/NOMAD/Data/pfm_auxiliary_files")

    paths["FS_DATA_DIRECTORY"] = os.path.normcase(r"/bira-iasb/projects/NOMAD/data/flight_spare/hdf5")

    FIG_X = 8
    FIG_Y = 6

    FIG_X_PDF = 9
    FIG_Y_PDF = 9

    paths["DATASTORE"] = {}
    paths["DATASTORE"]["DIRECTORY_STRUCTURE"] = True
    paths["DATASTORE"]["SEARCH_DATASTORE"] = False
    paths["DATASTORE"]["DATASTORE_SERVER"] = []
    paths["DATASTORE"]["DATASTORE_DIRECTORY"] = ""

    paths["RETRIEVALS"] = {}
    paths["RETRIEVALS"]["RADTRAN_DIR"] = r"/bira-iasb/projects/NOMAD/Science/Radiative_Transfer"
    paths["RETRIEVALS"]["AUXILIARY_DIR"] = os.path.join(paths["RETRIEVALS"]["RADTRAN_DIR"], "Auxiliary_files")
    paths["RETRIEVALS"]["SOLAR_DIR"] = os.path.join(paths["RETRIEVALS"]["AUXILIARY_DIR"], "Solar")


elif os.path.exists(os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")):  # outside BIRA

    paths["BASE_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python")

    # paths["DATA_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\hdf5_copy")
    # paths["DATA_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\hdf5")
    paths["DATA_DIRECTORY"] = os.path.normcase(r"D:\DATA\hdf5")
    # paths["DATA_DIRECTORY"] = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")
    # paths["DATA_DIRECTORY"] = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\test\iant\hdf5")

    paths["LOCAL_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA")

    paths["DATASTORE_ROOT_DIRECTORY"] = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD")
    # paths["DATASTORE_ROOT_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA")
    # paths["DATASTORE_ROOT_DIRECTORY"] = os.path.normcase(r"D:\DATA")

    paths["RETRIEVAL_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\retrievals")
#    paths["RETRIEVAL_DIRECTORY"] = os.path.normcase(r"X:\linux\input_tools\Tools")

    paths["FOOTPRINT_DIRECTORY"] = os.path.normcase(r"D:\DATA\footprint")

    paths["DB_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\db")
    paths["ANIMATION_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\animations")
    paths["SIMULATION_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\simulations")
    paths["GCM_DIRECTORY"] = os.path.normcase(r"D:\DATA\gcm")
    paths["KERNEL_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\local_spice_kernels\kernels\mk")
    paths["KERNEL_DIRECTORY_OPS"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\local_spice_kernels_ops\kernels\mk")

    paths["COP_TABLE_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\cop_tables")
    # paths["COP_TABLE_DIRECTORY"] = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\cop_tables")

    paths["REFERENCE_DIRECTORY"] = os.path.normcase(r"C:\Users\iant\Dropbox\NOMAD\Python\reference_files")
    paths["PFM_AUXILIARY_FILES"] = os.path.normcase(r"C:\Users\iant\Documents\DATA\pfm_auxiliary_files")
#    paths["PFM_AUXILIARY_FILES"] = os.path.normcase(r"X:\projects\NOMAD\data\pfm_auxiliary_files")

    paths["FS_DATA_DIRECTORY"] = os.path.normcase(r"D:\DATA\flight_spare\hdf5")
    # paths["FS_DATA_DIRECTORY"] = os.path.normcase(r"X:\projects\NOMAD\data\flight_spare\hdf5")

    FIG_X = 10  # for PPT
    FIG_Y = 5  # for PPT

    FIG_X_PDF = 9
    FIG_Y_PDF = 9

    paths["DATASTORE"] = {}
    paths["DATASTORE"]["DIRECTORY_STRUCTURE"] = True
    paths["DATASTORE"]["SEARCH_DATASTORE"] = True  # only applies if DIRECTORY_STRUCTURE = True
    paths["DATASTORE"]["DATASTORE_SERVER"] = ["hera.oma.be", "iant"]
    paths["DATASTORE"]["DATASTORE_DIRECTORY"] = r"/ae/data1/SATELLITE/TRACE-GAS-ORBITER/NOMAD/hdf5"
#    paths["DATASTORE"]["DATASTORE_DIRECTORY"] = r"/ae/data1/SATELLITE/TRACE-GAS-ORBITER/NOMAD/test/iant/hdf5"

    paths["RETRIEVALS"] = {}
    paths["RETRIEVALS"]["OBSFILE_DIR"] = r'C:\Users\iant\Dropbox\NOMAD\Python\retrievals'
    paths["RETRIEVALS"]["RADTRAN_DIR"] = r"C:\Users\iant\Documents\DATA\Radiative_Transfer"  # data available offline
    paths["RETRIEVALS"]["GEM_OUTPUT_DIR"] = r"C:\Users\iant\Documents\DATA\gem-mars\output"  # data available offline
    paths["RETRIEVALS"]["APRIORI_FILE_DESTINATION"] = r"C:\Users\iant\Documents\DATA\a_priori"

    paths["RETRIEVALS"]["AUXILIARY_DIR"] = os.path.join(paths["RETRIEVALS"]["RADTRAN_DIR"], "Auxiliary_files")
    paths["RETRIEVALS"]["ATMOSPHERE_DIR"] = os.path.join(paths["RETRIEVALS"]["AUXILIARY_DIR"], "Atmosphere")
    paths["RETRIEVALS"]["PLANET_DIR"] = os.path.join(paths["RETRIEVALS"]["AUXILIARY_DIR"], "Planet")
    paths["RETRIEVALS"]["SOLAR_DIR"] = os.path.join(paths["RETRIEVALS"]["AUXILIARY_DIR"], "Solar")
    paths["RETRIEVALS"]["LIDORT_DIR"] = os.path.join(paths["RETRIEVALS"]["AUXILIARY_DIR"], "Lidort")
    paths["RETRIEVALS"]["HITRAN_DIR"] = os.path.join(paths["RETRIEVALS"]["AUXILIARY_DIR"], "Spectroscopy")
    paths["RETRIEVALS"]["MOLA_DIR"] = os.path.join(paths["RETRIEVALS"]["AUXILIARY_DIR"], "MOLA")


print("BASE_DIRECTORY=%s" % paths["BASE_DIRECTORY"])
print("DATA_DIRECTORY=%s" % paths["DATA_DIRECTORY"])
