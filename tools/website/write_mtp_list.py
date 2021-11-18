# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:14:19 2021

@author: iant

MAKE TABLE OF MTPS WITH 

E.G. 
Deadlines	Pointing	COP row	Execution start
MTP052	06-Dec	02-Mar	19-Mar
MTP053	03-Jan	30-Mar	16-Apr
MTP054	31-Jan	27-Apr	14-May
MTP055	28-Feb	25-May	11-Jun
MTP056	28-Mar	22-Jun	09-Jul

"""

from datetime import datetime, timedelta

time_format = "%d/%m/%Y"

mtp_1 = datetime(year=2018, month=4, day=21)

pointing_delta = (datetime(year=2022, month=3, day=19) - datetime(year=2021, month=12, day=6))
cop_delta = (datetime(year=2022, month=3, day=19) - datetime(year=2022, month=3, day=2))

h = ""
h += "<table border='1'>\n"
h += "<tr>\n   <th>MTP Number</th><th>STPs</th><th>Pointing Fixed</th><th>Planning Fixed</th><th>Execution Start</th>\n</tr>\n"


for mtp in range(200):
    
    h += "<tr>\n"
    
    mtp_start = mtp_1 + timedelta(days=(28*(mtp-1)))
    mtp_pointing = mtp_start - pointing_delta
    mtp_cop = mtp_start - cop_delta
    
    stps = "%i, %i, %i, %i" %(mtp*4-3, mtp*4-2, mtp*4-1, mtp*4)
    
    if mtp > 0:
        h += "   <td>%i</td><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td>\n</tr>\n" %(
            mtp,
            stps,
            datetime.strftime(mtp_pointing, time_format),
            datetime.strftime(mtp_cop, time_format),
            datetime.strftime(mtp_start, time_format),
        )
    else:
        h += "   <td>%i</td><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td>\n</tr>\n" %(
            mtp,
            "-",
            "-",
            "-",
            datetime.strftime(mtp_start, time_format),
        )
        

h += "</table>\n"

with open("mtp_dates.html", "w") as f:
    for line in h:
        f.write(line)