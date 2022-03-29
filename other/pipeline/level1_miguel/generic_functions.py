# -*- coding: utf-8 -*-


import os.path
import sys
import inspect

import h5py



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
