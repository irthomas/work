# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:42:39 2026

@author: iant

READ IN COMMAND HISTORY, PLOT DATA BACKLOG
"""

import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

dt_format = "%Y-%m-%d %H:%M:%S"

regex = r"(20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| run_pipeline.py --log INFO --profile default --command make --from inserted --to hdf5_l10a --beg 20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} --end (20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) --all True --n_proc 8 --filter 'None'"
dtype = [("exec_date", "S19"), ("data_date", "S19")]


exec_search_dates_all = [
    [datetime(2025, 1, 1), datetime(2027, 1, 1)],
]

ignore_dates_all = [
    [datetime(2025, 7, 5), datetime(2025, 7, 6)],
]

ignore_deltas_gt = timedelta(days=100)  # days
ignore_deltas_lt = timedelta(days=1)  # days


with open("command_history.txt", "r") as f:
    lines_in = np.fromregex(f, regex, dtype)

lines_str = [[line[0].decode(), line[1].decode()] for line in lines_in]

dts_exec_data = [[datetime.strptime(line[0], dt_format), datetime.strptime(line[1], dt_format)] for line in lines_str]

dt_deltas = [line[0] - line[1] for line in dts_exec_data]


for exec_search_dates in exec_search_dates_all:
    good_ixs = []
    for i, (dts, dt_delta) in enumerate(zip(dts_exec_data, dt_deltas)):
        if dts[0] > exec_search_dates[0] and dts[0] < exec_search_dates[1] and dt_delta > ignore_deltas_lt and dt_delta < ignore_deltas_gt:
            good = True
            for ignore_dates in ignore_dates_all:
                if dts[0] > ignore_dates[0] and dts[0] < ignore_dates[1]:
                    good = False

            if good:
                good_ixs.append(i)

x = [dts_exec_data[i][0] for i in good_ixs]
y = [dt_deltas[i].days for i in good_ixs]

plt.figure(figsize=(12, 5))
plt.scatter(x, y)
plt.title("NOMAD Data Backlog %s-%s" % (x[0].year, x[-1].year))
plt.xlabel("Date")
plt.ylabel("Data Pipeline Backlog (Days)")
plt.grid()

plt.savefig("nomad_data_backlog_%s-%s.png" % (x[0].year, x[-1].year))
