# -*- coding: utf-8 -*-

__project__   = "NOMAD"
__author__    = "Roland Clairquin"
__contact__   = "roland.clairquin@oma.be"

from datetime import timedelta, tzinfo, datetime
import logging
# import os.path

import spiceypy as sp

#from nomad_ops.config import KERNEL_DIRECTORY

logger = logging.getLogger( __name__ )

# A UTC class.
class UTC(tzinfo):
    """UTC"""
    def utcoffset(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return timedelta(0)


def load_spice_kernel(metakernel_path):
    logger.info("Loading spice kernel (%s)", metakernel_path)
    #cwd = os.getcwd()
    #os.chdir(os.path.dirname(metakernel_path))
    sp.reset()
    sp.furnsh(metakernel_path)
    #os.chdir(cwd)


J2000 = datetime(2000, 1, 1, 12, tzinfo=UTC())
J2000_TS = J2000.timestamp()
SP_FMT_SP2000 = "SP2000.######"
TGO_ID = -143

#load_spice_kernel(os.path.join(KERNEL_DIRECTORY, METAKERNEL_NAME))

def obt_to_timestamp(obt):
    obt_i = int(obt)
    obt_str = "%d.%d" % (obt_i, int(65536 * (obt-obt_i) + 0.5))
    et = sp.scs2e(TGO_ID, obt_str)
    return J2000_TS + float(sp.timout(et, SP_FMT_SP2000))
