# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:34:31 2020

@author: iant

TEST READING JSON ON WEB AND PLOT RESULTS
"""


from urllib.request import urlopen
import json
from datetime import datetime
import matplotlib.pyplot as plt


url = ("https://epistat.sciensano.be/Data/COVID19BE_CASES_MUNI.json")

response = urlopen(url)
data = response.read().decode("utf-8")

dict_list_all = json.loads(data)


communes = ["Watermael-Boitsfort", "Auderghem", "Uccle"]
populations = [25300., 33740., 82275]

search_dict = {"TX_DESCR_FR":communes}


matching_dicts_list = [[] for i in communes]
for dictionary in dict_list_all:
    for search_key, search_values in search_dict.items():
        if search_key in dictionary:
            for index, search_value in enumerate(search_values):
                if search_value == dictionary[search_key]:
                    matching_dicts_list[index].append(dictionary)
                

save_dict = [{"DATE":[], "CASES":[]} for i in communes]
for index, matching_dict_list in enumerate(matching_dicts_list):
    for matching_dict in matching_dict_list:
        
        #check all keys are in json
        if len(save_dict[index].keys()) == sum([1 if save_dict_key in matching_dict.keys() else 0 for save_dict_key in save_dict[index].keys()]):
        
            for save_dict_key in save_dict[index].keys():
                
                value = matching_dict[save_dict_key]
                
                if save_dict_key == "DATE":
                    value = datetime.strptime(value, "%Y-%m-%d")
                    
                if save_dict_key == "CASES":
                    if value == "<5":
                        value = 0
                    else:
                        value = int(value)
                
                save_dict[index][save_dict_key].append(value)

# plt.scatter(save_dict["DATE"], save_dict["CASES"])

# last_date = save_dict["DATE"][-1]

averages_list = [{"start":[], "end":[], "cases":[]} for i in communes]

for index, averages in enumerate(averages_list):
    delta = 7
    for start_date in range(0, len(save_dict[index]["CASES"]) - delta + 1):
        end_date = start_date + delta - 1
        # print(save_dict[index]["DATE"][start_date], "-", save_dict[index]["DATE"][end_date], ":", sum(save_dict[index]["CASES"][start_date:end_date+1]))
        averages["start"].append(save_dict[index]["DATE"][start_date])
        averages["end"].append(save_dict[index]["DATE"][end_date])
        averages["cases"].append(sum(save_dict[index]["CASES"][start_date:end_date+1])/populations[index] * 100000.)

plt.figure(figsize=(12, 6))
plt.title("7-day average per 100k residents")
plt.xlabel("End date")
plt.ylabel("Positive cases")
for index, commune in enumerate(communes):
    plt.plot(averages_list[index]["end"], averages_list[index]["cases"], label=commune)
# plt.ylim(ymin=0, ymax=max(averages_list[index]["cases"])+10)
# plt.axhline(y=4)
plt.legend()

plt.savefig("covid.png")



