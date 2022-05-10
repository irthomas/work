# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:20:45 2020

@author: iant

OLD FUNCTIONS
HEATERS_TEMP_DB IS NOW CREATED BY PIPELINE
TGO TEMPERATURES ARE NOW IN HDF5 FILES. GET FROM FILE INSTEAD OF MAKING DB

"""
#dbName = "tgo_temperatures"
#db_obj = obsDB(dbName)
##queryOutput = db_obj.query("SELECT * FROM temperatures ORDER BY ABS(JULIANDAY(utc_start_time) - JULIANDAY('2018-10-07 04:23:19')) LIMIT 1")
##queryOutput = db_obj.query("SELECT * FROM temperatures WHERE utc_start_time BETWEEN '2018-04-22 00:00:00' AND '2018-04-22 01:00:00'")
#out = db_obj.temperature_query('2018-10-07 04:23:19')[0][2] #SO
#db_obj.close()




"""get full table"""
#db_obj = obsDB()
#table = db_obj.read_table("lno_nadir")
#db_obj.close()


"""get dictionary from query and plot"""
#channel = "lno"
#db_obj = obsDB(dbName)
#CURIOSITY = -4.5895, 137.4417
#searchQueryOutput = db_obj.query("SELECT * FROM lno_nadir where latitude < 5 AND latitude > -15 AND longitude < 147 AND longitude > 127 AND incidence_angle < 40 and diffraction_order == 134")
#obsDict = makeObsDict(channel, searchQueryOutput)
#db_obj.close()
#
#plt.figure()
#plt.scatter(obsDict["longitude"], obsDict["latitude"])
#
#    
#fig1, ax1 = plt.subplots()
#
#
#for frameIndex, (x, y) in enumerate(zip(obsDict["x"], obsDict["y"])):
#    ax1.plot(x, y, alpha=0.3)
#
#yMean = np.mean(np.asfarray(obsDict["y"])[:, :], axis=0)
#xMean = np.mean(np.asfarray(obsDict["x"])[:, :], axis=0)
#ax1.plot(xMean, yMean, "k")
#
#yMean = np.mean(np.asfarray(obsDict["y"])[145:, :], axis=0)
#xMean = np.mean(np.asfarray(obsDict["x"])[145:, :], axis=0)
#ax1.plot(xMean, yMean, "r")



#"""write so occultations to text file for Sebastien"""
#channel = "so"
#dbName = "so_1p0a"
#db_obj = obsDB(dbName)
#searchQueryOutput = db_obj.query("SELECT * FROM so_occultation WHERE diffraction_order == 134")
##obsDict = makeObsDict(channel, searchQueryOutput)
#db_obj.close()
#
#table_headers = getTableFields(channel)
#headers = [value["name"] for value in table_headers]
#header_types = [value["type"] for value in table_headers]
#
#with open(os.path.join(BASE_DIRECTORY, "order_134_occultations.txt"), "w") as f:
#    lines = ["%s, %s, %s, %s, %s\n" %(headers[9], headers[10], headers[11], headers[12], headers[13])]
#    for queryLine in searchQueryOutput:
#        lines.append("%s, %0.3f, %0.3f, %0.3f, %0.3f\n" 
#             %(queryLine[9], queryLine[10], queryLine[11], queryLine[12], queryLine[13]))
#    for line in lines:
#        f.write(line)



#"""check for errors in timestamps of TGO readouts in heaters_temp"""
#dbName =  "heaters_temp"
#db_obj = obs_database(dbName)
#db_obj.query("SELECT ts from heaters_temp")
#data = db_obj.query("SELECT ts from heaters_temp")
#db_obj.close()
#ts = [i[0] for i in data[540000:]]
#delta = [ts[i + 1] - ts[i] for i in range(len(ts[:-1]))]
#ind = np.where(np.array(delta) > datetime.timedelta(minutes=20))[0] #find all gaps>20 minutes
#for i in ind:
#    print(ts[i], "-", ts[i + 1])
