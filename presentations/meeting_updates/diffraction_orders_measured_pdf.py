# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:06:20 2023

@author: iant


READ IN THE PLANNING DATABASE THEN FOR EACH MTP, GET ALL OBSERVATION NAMES AND THEIR RELATIVE FREQUENCIES

"""



import decimal
# from datetime import datetime
import sqlite3
import numpy as np
# import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

from tools.general.progress_bar import progress


import sys
sys.path.append(r"C:\Users\iant\Documents\PROGRAMS\nomad_obs")

from nomad_obs.observation_names import occultationObservationDict, nadirObservationDict

obs_name_d = {"so":{}, "lno":{}}
for obs_name, obs_info in occultationObservationDict.items():
    orders = sorted(obs_info[0])
    obs_name_d["so"][tuple(orders)] = obs_name
for obs_name, obs_info in nadirObservationDict.items():
    orders = sorted(obs_info[0])
    obs_name_d["lno"][tuple(orders)] = obs_name


planning_db_path = r"C:/Users/iant/Documents/PROGRAMS/nomad_obs/planning.db"




# LIST_ONLY = True
LIST_ONLY = False

N_MONTHS = 2


def connect_db(db_path):
    print("Connecting to database %s" %db_path)
    con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    return con


def close_db(con):
    con.close()


def query(con, input_query):
    # print(input_query)
    cur = con.cursor()
    
    c = cur.execute(input_query)
    con.commit()
    output = c.fetchall()
    return output




def read_table(con, table_name):
    query_string = "SELECT * FROM %s" %table_name
    table = query(con, query_string)

    new_table_data = []
    for row in table:
        new_table_data.append([float(element) if type(element) == decimal.Decimal else element for element in row])
    
    return new_table_data




def get_planning_db_dict(table_name):
    con = connect_db(planning_db_path)
    
    if table_name == "occultations":
        occultation_data = read_table(con, table_name)
        
        #convert to dictionaries
        d = {"obs_id":[],
            "prime_instrument":[],
            "orbit_number":[],
            "mtp_number":[],
            "occultation_type":[],
            "utc_start_time":[],
            "utc_transition_time":[],
            "utc_end_time":[],
            "duration":[],
            "start_longitude":[],
            "transition_longitude":[],
            "end_longitude":[],
            "start_latitude":[],
            "transition_latitude":[],
            "end_latitude":[],
            "transition_local_time":[],
            "orbit_type":[],
            "ir_observation_name":[],
            "ir_description":[],
            "uvis_description":[],
            "orbit_comment":[],
            }
        
        for row in occultation_data:
            if row[1] == "NOMAD":
                for i, k in enumerate(d.keys()):
                    d[k].append(row[i])

    if table_name == "nadirs":
        nadir_data = read_table(con, table_name)
        
        #convert to dictionaries
        d = {"obs_id":[],
            "orbit_number":[],
            "mtp_number":[],
            "nadir_type":[],
            "utc_start_time":[],
            "utc_centre_time":[],
            "utc_end_time":[],
            "duration":[],
            "start_longitude":[],
            "centre_longitude":[],
            "end_longitude":[],
            "start_latitude":[],
            "centre_latitude":[],
            "end_latitude":[],
            "centre_incidence_angle":[],
            "centre_local_time":[],
            "orbit_type":[],
            "ir_observation_name":[],
            "ir_description":[],
            "uvis_description":[],
            "orbit_comment":[],
            }
        
        for row in nadir_data:
            if row[3] == "Dayside":
                for i, k in enumerate(d.keys()):
                    d[k].append(row[i])

    return d



for channel in ["so", "lno"]:

    if channel == "so":
        from nomad_obs.observation_names import occultationObservationDict as observation_dict
        y_lim = 12
        obs_type = "occultations"
    elif channel == "lno":
        from nomad_obs.observation_names import nadirObservationDict as observation_dict
        y_lim = 25
        obs_type = "nadirs"
    
    
    
    
    occ_dict = get_planning_db_dict(obs_type)
    
    
    unique_ir_observation_names = [s for s in list(set(occ_dict["ir_observation_name"])) if s]
    
    obs_times = [str(s)[0:7] for s in occ_dict["utc_start_time"]]
    
    unique_times = sorted(list(set(obs_times)))
    
    month_d = {}
    for unique_time in unique_times:
        #get number of obs in each MTP
        month_ixs = [i for i, t in enumerate(obs_times) if t == unique_time and occ_dict["ir_observation_name"][i]]
        
        #count number of indices, make blank entry for each obs name
        month_d[unique_time] = {"n_obs":len(month_ixs), "obs":{unique_ir_observation_name:0 for unique_ir_observation_name in unique_ir_observation_names}}
        
        
        for ix in month_ixs:
            obs_name = occ_dict["ir_observation_name"][ix]
            month_d[unique_time]["obs"][obs_name] += 100. / len(month_ixs)
        
    
    #sort obs names based on the last N months
    end_unique_times = unique_times[-N_MONTHS:]
    
    end_obs_freqs = []
    for unique_ir_observation_name in unique_ir_observation_names:
        obs_freqs = [month_d[t]["obs"][unique_ir_observation_name] for t in end_unique_times]
        end_obs_freqs.append(sum(obs_freqs))
        
    sort_ixs = np.argsort(end_obs_freqs)[::-1]
    
    sorted_unique_ir_observation_names = [unique_ir_observation_names[i] for i in sort_ixs]
    
    if LIST_ONLY:
        
        print("#### %s ####" %channel.upper())

        for unique_ir_observation_name in sorted_unique_ir_observation_names:
            
            #convert high-low to all
            if ";" in unique_ir_observation_name:
                split = unique_ir_observation_name.split(";")
            
                orders_name = split[0].strip()
                orders_name2 = split[1].strip()
    
                orders = observation_dict[orders_name][0]
                orders2 = observation_dict[orders_name2][0]
                
            else:
                orders_name = unique_ir_observation_name
                orders_name2 = ""
    
                orders = observation_dict[unique_ir_observation_name][0]
                orders2 = []
            
            obs_freqs = [month_d[t]["obs"][unique_ir_observation_name] for t in unique_times]
            
            if len(orders) == 1: #skip fullscans and weird observations
                continue
            
            mean_n_months = np.mean(obs_freqs[-N_MONTHS:])
            sum_n_months = np.round(np.sum([month_d[t]["obs"][unique_ir_observation_name]/100 * month_d[t]["n_obs"] for t in unique_times[-N_MONTHS:]]))
            
            if mean_n_months == 0: #if not measured in last N months
                continue
            
            #check for orders in observation dict and get the name
            orders_t = tuple(sorted(orders))
            if orders_t in obs_name_d[channel].keys():
                obs_name = obs_name_d[channel][orders_t]
            else:
                obs_name = ""
                
            #put the dark 0 at the end
            orders = sorted(orders)
            if orders[0] == 0:
                orders = orders[1:] + [0]
            
            print(", ".join(["%s" %i for i in orders])+"\t%s" %unique_ir_observation_name)
            
            
            
            
        
    else:
        
        with PdfPages("%s_%s_observation_frequencies.pdf" %(channel.upper(), obs_type)) as pdf: #open pdf
        
            for unique_ir_observation_name in progress(sorted_unique_ir_observation_names):
                
                #convert high-low to all
                if ";" in unique_ir_observation_name:
                    split = unique_ir_observation_name.split(";")
                
                    orders_name = split[0].strip()
                    orders_name2 = split[1].strip()
        
                    orders = observation_dict[orders_name][0]
                    orders2 = observation_dict[orders_name2][0]
                    
                else:
                    orders_name = unique_ir_observation_name
                    orders_name2 = ""
        
                    orders = observation_dict[unique_ir_observation_name][0]
                    orders2 = []
                
                obs_freqs = [month_d[t]["obs"][unique_ir_observation_name] for t in unique_times]
                
                fig = plt.figure(figsize=(15, 8), constrained_layout=True)
                gs = gridspec.GridSpec(1, 4, figure=fig)
                ax1a = plt.subplot(gs[0, 0:3])
                ax1b = plt.subplot(gs[0, 3])
                
                if len(orders) == 1: #skip fullscans and weird observations
                    plt.close()
                    continue
                
                mean_n_months = np.mean(obs_freqs[-N_MONTHS:])
                sum_n_months = np.round(np.sum([month_d[t]["obs"][unique_ir_observation_name]/100 * month_d[t]["n_obs"] for t in unique_times[-N_MONTHS:]]))
                
                if mean_n_months == 0: #if not measured in last N months
                    plt.close()
                    continue
                
                if orders2:
                    s = ", ".join([str(i) if i>0 else "dark" for i in sorted(orders)])
                    s2 = ", ".join([str(i) if i>0 else "dark" for i in sorted(orders2)])
                    title = "%s %s with diffraction order combination %s & %s" %(channel.upper(), obs_type, s, s2)
                else:
                    s = ", ".join([str(i) if i>0 else "dark" for i in sorted(orders)])
                    title = "%s %s with diffraction order combination %s" %(channel.upper(), obs_type, s)
                    
                
                ax1a.set_title(title)
                ax1a.set_xlabel("Year and month")
                ax1a.set_ylabel("% of total observations in each MTP")
                ax1a.bar(unique_times, obs_freqs, align="center", width=0.8)
                ax1a.grid()
                ax1a.set_ylim((0, y_lim))
                ax1a.tick_params(axis='x', rotation=90)
                ax1a.set_yticks(range(y_lim+1))
        
                
                ax1a.axhline(y=mean_n_months, color="k", linestyle="--")
                ax1a.text(0, mean_n_months+0.1, "Mean %% of observations in the last %i months\n(%i observations in %i months)" %(N_MONTHS, sum_n_months, N_MONTHS))
                
                for order2 in orders2:
                    rect = ax1b.add_patch(
                        Rectangle(xy=(0, order2), width=2,
                                  height=1, facecolor="r")
                        )
                for order in orders:
                    rect = ax1b.add_patch(
                        Rectangle(xy=(0, order), width=2,
                                  height=1, facecolor="k")
                        )
        
                
                ax1b.set_ylim((100,200))
                ax1b.grid()
                ax1b.set_xlabel("\nObservation name:\n%s\n%s" %(orders_name, orders_name2))
                ax1b.set_xticks([])
                ax1b.set_yticks(range(100, 201, 10))
        
        
                pdf.savefig()
                plt.close()
            
            print(orders)
