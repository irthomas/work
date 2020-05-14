# -*- coding: utf-8 -*-

#TESTING=True
#TESTING=False

import bisect
from datetime import datetime
import logging
import os.path
import re
import sys
import inspect

import h5py

#if TESTING:
#    sys.path.append(r"X:\linux\nomad_ops")
#from nomad_ops.config import OBSERVATION_TYPE_LOG


__project__   = "NOMAD"
__author__    = "Ian Thomas & Roland Clairquin"
__contact__   = "roland.clairquin@oma.be"

logger = logging.getLogger( __name__ )

__RE_GET_CHANNEL_TYPE = re.compile("^\d{8}_\d{6}.*_(UVIS|SO|LNO)(?:_(\d))?")
__RE_OBS_BASE_NAME = re.compile("(^\d{8}_\d{6}).*_(UVIS|SO|LNO)")
__DT_FORMAT = "%Y%m%d_%H%M%S"
__MAX_DELTA = 60

#def __makeObservationDict():
#    """make observation type dictionary from csv file. From ICD:
#    I = Ingress (during solar occultation)
#    E = Egress (during solar occultation)
#    S = Fullscan (during solar occultation)
#    F = Fullscan (during nadir)
#    M = Miniscan (during solar occultation)
#    D = Day nadir
#    N = Night nadir
#    L = Limb observation
#    C = Calibration
#    X = Unknown"""  # FixMe !!!
#
##    U = UVIS full frame (during solar occultation) #removed
##    V = UVIS full frame (during nadir) #removed
#
#
#    obs_dict = {}
#    with open(OBSERVATION_TYPE_LOG, "r") as obsTypeFile:
#        for line in obsTypeFile:
#            fname, obs_type = line.strip().split(",")
#            mobj = __RE_OBS_BASE_NAME.match(fname)
#            ts = datetime.strptime(mobj.group(1), __DT_FORMAT).timestamp()
#            obs_dict.setdefault(mobj.group(2), []).append((ts, obs_type))
#    # for l in obs_dict.values():
#    #     l.sort()
#    for k in obs_dict:
#        obs_dict[k].sort()
#        obs_dict[k] = tuple(zip(*obs_dict[k]))
#    return obs_dict
#
#__OBSERVATION_DICT = __makeObservationDict()

def getChannelType(hdf5_file):
    if isinstance(hdf5_file, h5py._hl.files.File):
        channel_type = None
        channel = re.match("^\S*", hdf5_file.attrs["ChannelName"]).group().lower()
        if channel != "uvis":
            channel_type = re.match("^\S*", hdf5_file.attrs["ScienceSet"])
        return channel, channel_type

    # For compatibility with levels > 0.2a, should be removed in the future
    # Always returns Science x of 2 since _1 is in the filename for observations with only 1 science set
    else:
        logger.warning("%s is not a hdf5 file", hdf5_file)
        """get channel and type from input filename"""
        mobj = __RE_GET_CHANNEL_TYPE.match(hdf5_file)
        if mobj:
            channel = mobj.group(1).lower()
            sc_set = mobj.group(2)
            if sc_set is None:
                return channel, "Science 1 of 1"
            else:
                return channel, "Science %s of 2" % sc_set

def getObsDatetime(filename):
    mobj = __RE_OBS_BASE_NAME.match(filename)
    return datetime.strptime(mobj.group(1), __DT_FORMAT)

def getObservationType(hdf5_file):
    if isinstance(hdf5_file, h5py._hl.files.File):
        return hdf5_file.attrs["ObservationType"]
#
#    # For compatibility with levels > 0.2a, should be removed in the future
#    else:
#        logger.warning("%s is not a hdf5 file", hdf5_file)
#        """get observation type from dictionary"""
#        mobj = __RE_OBS_BASE_NAME.match(hdf5_file)
#        dt = datetime.strptime(mobj.group(1), __DT_FORMAT)
#        if dt < datetime(2016, 3, 14): # Before launch, return "G" (Ground)
#            return "G"
#        ts = dt.timestamp()
#        tss, obs_types = __OBSERVATION_DICT[mobj.group(2)]
#        ix = bisect.bisect_left(tss, ts)
#        ixs = []
#        if ix > 0:
#            ixs.append(ix-1)
#        if ix < len(tss):
#            ixs.append(ix)
#        best_ix = None
#        best_delta = __MAX_DELTA
#        for ix in ixs:
#            delta = abs(tss[ix] - ts)
#            if delta < best_delta:
#                best_delta = delta
#                best_ix = ix
#        if best_delta >= __MAX_DELTA:
#            logger.error("Observation type cannot be found for %s (best_delta=%d)",
#                         hdf5_file, int(best_delta))
#            obs_type = "X"  # Unknown, FixMe
#        else:
#            obs_type = obs_types[best_ix]
#        # assert obs_type == getObservationType_(filename)
#        return obs_type

def copyAttributesExcept(src_hdf5, dest_hdf5, output_version, attrsNotToCopy=[]):
    """Copy attributes from one hdf5 file to another"""
    for key, value in src_hdf5.attrs.items():
        if key == "IntDataLevel": #when copying the internal data level, update to next level.
            scriptName =  os.path.basename(sys.argv[0]) #get script name
            #moduleName = os.path.basename(__file__)
            frm = inspect.stack()[1]
            mod = inspect.getmodule(frm[0])
            if mod is None:
                moduleName = "error_retrieving_module_name"
            else:
                moduleName = os.path.basename(mod.__file__)
            IntDataLevel="%s (%s & %s)" %(output_version, scriptName, moduleName)
            dest_hdf5.attrs[key] = IntDataLevel
        elif key not in attrsNotToCopy:
            dest_hdf5.attrs[key] = value

def createIntermediateGroups(root, groups):
    for g in groups:
        root = root.require_group(g)
    return root

def copyDatasets(src_hdf5, dest_hdf5, dset_paths):
    """Copy datasets from one hdf5 file to another"""
    for dset_path in dset_paths:
        dest = createIntermediateGroups(dest_hdf5, dset_path.split("/")[:-1])
        src_hdf5.copy(dset_path, dest)

def iter_datasets(group, root=()):
    for name, obj in group.items():
        path = root + (name,)
        if isinstance(obj, h5py.Dataset):
            yield "/".join(path), obj
        elif isinstance(obj, h5py.Group):
            yield from iter_datasets(obj, path)
