# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:21:08 2022

@author: iant

COMPARE REMOTE HDF5 FILE TREE TO CACHE.DB
"""

import os

from tools.sql.read_cache_db import get_filenames_from_cache
from tools.file.remote_tree import get_tree_filenames
from tools.file.passwords import passwords
from tools.sql.transfer_dbs import transfer_cache_db

SHARED_DIR_PATH = r"C:\Users\iant\Documents\PROGRAMS\web_dev\shared"



user = "iant"
# level = "hdf5_level_1p0a"
level = "spacewire"

channel = ""

#update local db
transfer_cache_db(level)

cache = get_filenames_from_cache(os.path.join(SHARED_DIR_PATH, "db", level + ".db"))
cache_filenames = sorted([s.replace(".h5","") for s in cache[1] if channel.upper() in s])


tree_filenames = sorted(get_tree_filenames(user, passwords["hera"], level, channel=channel))


def return_not_matching(a, b):
    """compare list 1 to list 2 and return non matching entries"""
    return [[x for x in a if x not in b], [x for x in b if x not in a]]





print("Comparing filename lists")
matching_tree_only, matching_cache_only = return_not_matching(tree_filenames, cache_filenames)

print("Writing matches and non-matches to files")
with open("matching_%s_tree_only.txt" %level, "w") as f:
    f.writelines("\n".join(matching_tree_only))
with open("matching_%s_cache_only.txt" %level, "w") as f:
    f.writelines("\n".join(matching_cache_only))
