# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:33:35 2020

@author: iant

PLOT NUMBER OF DATA ANOMALIES
"""

import re
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# raw_anomaly_path = "/bira-iasb/data/SATELLITE/TRACE-GAS-ORBITER/NOMAD/reports/raw_anomalies"
raw_anomaly_path = r"C:\Users\iant\Documents\DATA\reports"

regex = re.compile("20.*")

filepaths_unsorted = []
filenames_unsorted = []
for root, dirs, files in os.walk(raw_anomaly_path):
    matching_files = list(filter(regex.match, files))
    for filename in matching_files:
        if ".xlsx" in filename:
            filepaths_unsorted.append(os.path.join(root, filename))
            filenames_unsorted.append(filename)


filenames_sorted = sorted(list(set(filenames_unsorted)))

month_list = np.zeros(38)


file_datetimes = []
for filename in filenames_sorted:
    year = int(filename[0:4])
    month = int(filename[4:6])
    # day = [6:8]
    
    if year >= 2018:
        # file_datetime = datetime.strptime(filename[0:8], "%Y%m%d")
        # file_datetimes.append(file_datetime)
        
        month_list[((year-2018)*12+month)] += 1



fig, ax = plt.subplots(figsize=(15, 5))
ax.scatter(np.arange(len(month_list)), month_list)
# labels = ["2018", "2019", "2020", "2021"]

labels = []
labelx = []
for i in np.arange(len(month_list)):
    if np.mod(i, 12) == 1:
        ax.axvline(i, color="k", linestyle="--")
        labelx.append(i)
        labels.append("Jan %i" %int(2018.+i/12.))
    else:
        ax.axvline(i, color="k", linestyle="--", alpha=0.2)

ax.set_xticks(np.arange(1, len(month_list), 12))
ax.set_xticklabels(labels)
ax.set_xlabel("Calendar Month")
ax.set_ylabel("Number of anomaly reports generated per month")
ax.set_title("Anomaly Report Generation Statistics: Per Month")

fig.savefig("anomaly_reports_per_month.png")