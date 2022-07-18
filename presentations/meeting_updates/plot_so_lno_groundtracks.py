# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:50:51 2022

@author: iant

MONITOR NADIRS AND OCCULTATIONS:
PRODUCE COVERAGE MATCH FOR EACH MONTH OF OPERATIONS

"""


import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta



from tools.orbit.plot_groundtracks_from_db import plot_files_tracks
from tools.orbit.plot_groundtracks_from_db import plot_3d_files_tracks


def plot_groundtracks_per_month():
    dt = datetime(2018, 3, 1)
    data = True
    while data:
        dt_end = dt + relativedelta(months=3)
        search_tuple = ("files", {"utc_start_time":[dt, dt_end]})
        h5s = plot_files_tracks(search_tuple)
        plt.close()
    
        if len(h5s) == 0:
            data = False
        dt = dt_end

search_tuple = ("files", {"utc_start_time":[datetime(2022, 5, 27), datetime(2022, 5, 28)]})
h5s = plot_3d_files_tracks(search_tuple)
