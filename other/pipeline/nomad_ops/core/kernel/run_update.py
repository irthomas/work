# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:31:23 2022

@author: iant
"""


import platform
import logging
from nomad_ops.core.kernel.metakernelparser import MetakernelParser

if platform.system() == "Windows":
    from tools.file.paths import paths
    ROOT_DATASTORE_PATH = paths["DATASTORE_ROOT_DIRECTORY"]

else:

    from nomad_ops.config import KERNEL_FTP, KERNEL_FTP_PATH
    from nomad_ops.config import PATH_LOCAL_SPICE_KERNELS



__project__   = "NOMAD"
__author__    = "Bram Beeckman"
__contact__   = "bram.beeckman@aeronomie.be"

META_PARSER_REGEX = "(?<=\'\$KERNELS/).*(?=\'$)"
META_PATH_REGEX = "(?<=PATH_VALUES\s)\s*=\s+\( \'\.\.\' \)"

def run_update(ops_or_plan):
    
    if ops_or_plan == "ops":
        META_VERSION_REGEX = "(?<=em16_ops_)*\d{4}(?:\.|-|_)?\d{2}(?:\.|-|_)?\d{2}(?=_\d{3}.tm$)"
    else:
        META_VERSION_REGEX = "(?<=em16_plan_)*\d{4}(?:\.|-|_)?\d{2}(?:\.|-|_)?\d{2}(?=_\d{3}.tm$)"
        
    parser = MetakernelParser()
    logger = logging.getLogger( __name__ )
    logger.info("Kernel update started.")
    parser.check_tree(PATH_LOCAL_SPICE_KERNELS)
    parser.set_kernel_local_path(PATH_LOCAL_SPICE_KERNELS)
    logger.info("Local path set to {0}".format(PATH_LOCAL_SPICE_KERNELS))
    parser.set_kernel_remote(KERNEL_FTP_PATH)
    logger.info("Remote path set to {0} on {1}".format(KERNEL_FTP_PATH,KERNEL_FTP))

    parser.set_version_regex(META_VERSION_REGEX)
    parser.set_metakernel_regex(META_PARSER_REGEX)
    parser.set_path_regex(META_PATH_REGEX)

    if not parser.openFTP(KERNEL_FTP):
        logger.error("FTP connection failed, aborting.")
    else:
        parser.get_latest_metakernel()
        parser.update_kernels(parser.get_local_tree(), parser.parse_metakernel())
        parser.closeFTP()
