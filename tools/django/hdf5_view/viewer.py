# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:14:56 2019

@author: iant
"""
import os
import h5py
import numpy as np
from datetime import datetime, timedelta
from django.conf.urls import url
from django.http import HttpResponse
#from django.template.loader import render_to_string

DEBUG = True
SECRET_KEY = '4l0ngs3cr3tstr1ngw3lln0ts0l0ngw41tn0w1tsl0ng3n0ugh'
ROOT_URLCONF = __name__

#DATA_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\hdf5_copy")
DATA_DIRECTORY = os.path.normcase(r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")


CHANNELS = ["so", "lno", "uvis"]
LEVELS = ["hdf5_level_0p1a", "hdf5_level_0p2a", "hdf5_level_0p3a", "hdf5_level_1p0a"]



def makeMonthList():
    monthList = []
    start = datetime(2018, 3, 1)
    end = datetime.now()
    date = start
    while date < end:
        monthList.append("%04i-%02i" %(date.year, date.month))
        date = (date + timedelta(days=31)).replace(day=1)
    return monthList


MONTHS = makeMonthList()



def getFilePath(hdf5_filename):
    """get full file path from name"""
    
    file_level = "hdf5_level_%s" %hdf5_filename[16:20]
    year = hdf5_filename[0:4]
    month = hdf5_filename[4:6]
    day = hdf5_filename[6:8]

    filename = os.path.join(DATA_DIRECTORY, file_level, year, month, day, hdf5_filename+".h5") #choose a file
    
    return filename





def getFileList(channel, level, year, month):
    
    if channel.lower() in CHANNELS:
        if level in LEVELS:
            data_path = os.path.join(DATA_DIRECTORY, level, year, month)
    
            data_filenames_dates=[]
            for file_path, subfolders, files in os.walk(data_path):
                for each_filename in files:
                    if ".h5" in each_filename and channel.lower() in each_filename.lower():
                        file_stats = os.stat(os.path.join(data_path, file_path, each_filename))
                        mod_date = datetime.utcfromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        
                        data_filenames_dates.append([each_filename.replace(".h5",""), mod_date])
            data_filenames_sorted=sorted(data_filenames_dates)
            return data_filenames_sorted
        else:
            return [["No files in level %s" %level, "-"]]
    else:
        return [["No files for channel %s" %channel, "-"]]





def channelList(request):
    h = "<html><head><title>Data viewer</title></head><body>"
    h += "<h1>CHANNELS</h1>"
    for channel in CHANNELS:
        h+= "<a href='level/?channel=%s'>%s</a><br>" %(channel, channel.upper())
    h += "</body></html>"
    return HttpResponse(h)


def levelList(request):

    channel = request.GET.get("channel", "")

    #sanitize inputs
    if channel.lower() in CHANNELS:

        h = "<html><head><title>%s</title></head><body>" %channel.upper()
        h += "<h1>DATA LEVELS</h1>"
        for level in LEVELS:
            h+= "<a href='month/?channel=%s&level=%s'>%s</a><br>" %(channel, level, level.upper())
        h += "</body></html>"
            
    return HttpResponse(h)


def monthList(request):
    
    channel = request.GET.get("channel", "")
    level = request.GET.get("level", "")

    #sanitize inputs
    if level.lower() in LEVELS:

        h = "<html><head><title>%s</title></head><body>" %channel.upper()
        h += "<h1>MONTHS</h1>"
        for monthString in MONTHS:
            h+= "<a href='filelist/?channel=%s&level=%s&month=%s'>%s</a><br>" %(channel, level, monthString, monthString)
        h += "</body></html>"
            
    return HttpResponse(h)
    


def fileList(request):

    channel = request.GET.get("channel", "")
    level = request.GET.get("level", "")
    monthString = request.GET.get("month", "")
    
    year = monthString[0:4]
    month = monthString[5:7]
    
    
    #sanitize inputs
    if channel.lower() in CHANNELS:
        if level in LEVELS:

            title = "%s - %s" %(channel.upper(), level.upper())
            
            h = "<html><head><title>%s</title></head><body>" %title
            h += "<h1>%s</h1>" %title

            h += "<table border=1><tr>"
            table_headers = ["File name", "Last modified"]
            for table_header in table_headers:
                h += "<th>%s</th>" %table_header
            h += "</tr>"
        
            filenames_dates = getFileList(channel, level, year, month)
            for hdf5Filename, date_modified in filenames_dates:
                h += "<tr><td><a href='%s/?filename=%s'>%s</a></td><td>%s</td></tr>" %(level, hdf5Filename, hdf5Filename, date_modified)
            h += "</table>"
            h += "</body></html>"
        else:
            h = "<html><head><title>Error: level not found</title></head><body></body></html>"
    else:
        h = "<html><head><title>Error: channel not found</title></head><body>"
        h += "<h1>Channels</h1>"
        for channel in CHANNELS:
            h+= "%s" %channel.upper()
        h += "</body></html>"
            
    return HttpResponse(h)







def hdf5_level(request):
    
    hdf5Filename = request.GET.get("filename", "")
    
    title = "%s" %hdf5Filename
    
    h = "<html><head><title>%s</title></head><body>" %title
    h += "<h1>%s</h1>" %title
    
    if "so" in hdf5Filename.lower():
        channel = "so"
        uvis = False
    elif "lno" in hdf5Filename.lower():
        channel = "lno"
        uvis = False
    elif "uvis" in hdf5Filename.lower():
        channel = "uvis"
        uvis = True

    hdf5Filepath = getFilePath(hdf5Filename)
    with h5py.File(hdf5Filepath, "r") as hdf5File:
        
        observationType = hdf5File.attrs["ObservationType"]
        if not uvis:
            diffractionOrders = hdf5File["Channel/DiffractionOrder"][...]
            temperature = np.mean(hdf5File["Housekeeping/SENSOR_1_TEMPERATURE_%s" %channel.upper()][2:10])
            nOrders = hdf5File.attrs["NSubdomains"]
#        utcStartTimes = hdf5File["Geometry/ObservationDateTime"][:, 0]
        longitudes = hdf5File["Geometry/Point0/Lon"][:, 0]
        latitudes = hdf5File["Geometry/Point0/Lat"][:, 0]
        ys = hdf5File["Science/Y"][...]
        xs = hdf5File["Science/X"][...]
#        localSolarTime = hdf5File["Geometry/Point0/LST"][:, 0]
        x = xs[0, :]

        if uvis:
            mean_region = np.where((400 < x) & (x < 600))[0] #400-600nm
        else:
            mean_region = np.arange(160,241,1) #centre of blaze
        meanYs = np.nanmean(ys[:, mean_region], axis=1)
        
        if observationType in ["D", "N"]:
            incidenceAngles = hdf5File["Geometry/Point0/IncidenceAngle"][:, 0]
            if not uvis:
                diffractionOrder = diffractionOrders[0]
        elif observationType in ["I", "E", "G"]:
            altitudes = hdf5File["Geometry/Point0/TangentAltAreoid"][:, 0]
            if not uvis:
                diffractionOrder = diffractionOrders[0]
        else:
            #other types not yet supported
            h += "Observation type %s is not yet supported" %observationType
            h += "</body></html>"
            return HttpResponse(h)
        

    table1 = ""
    table1 += "<table border=1><tr>"
    table1 += "<tr><td>%s</td><td>%s</td></tr>\n" %("File name", hdf5Filename)
    table1 += "<tr><td>%s</td><td>%s</td></tr>\n" %("Observation Type", observationType)
    if not uvis:
        table1 += "<tr><td>%s</td><td>%s</td></tr>\n" %("Diffraction Order", diffractionOrder)
        table1 += "<tr><td>%s</td><td>%0.1fC</td></tr>\n" %("Temperature", temperature)
        table1 += "<tr><td>%s</td><td>%i</td></tr>\n" %("No. of orders", nOrders)
    table1 += "</table><br><br>"

    table2 = ""
    table2 += "<h2>Raw data</h2>"
    table2 += "<table border=1><tr>"
    if observationType in ["D", "N"]:
        table_headers = ["Index", "Incidence Angle", "Longitude", "Latitude"]
    elif observationType in ["I", "E", "G"]:
        table_headers = ["Index", "Tangent Altitude Areoid", "Longitude", "Latitude", "Mean Transmittance %i:%i" %(min(mean_region), max(mean_region))]

    for table_header in table_headers:
        table2 += "<th>%s</th>" %table_header
    table2 += "</tr>"

    if observationType in ["D", "N"]:
        data1 = ""
        for frameIndex, (incidenceAngle, longitude, latitude) in enumerate(zip(incidenceAngles, longitudes, latitudes)):
            table2 += "<tr><td>%i</td><td>%0.1f</td><td>%0.1f</td><td>%0.1f</td></tr>\n" %(frameIndex, incidenceAngle, longitude, latitude)
            data1 += "[%0.1f, %0.4f],\n" %(longitude, latitude)
        table2 += "</table><br><br>"

    if observationType in ["I", "E", "G"]:
        data1 = ""
        for frameIndex, (altitude, longitude, latitude, meanY) in enumerate(zip(altitudes, longitudes, latitudes, meanYs)):
            table2 += "<tr><td>%i</td><td>%0.1f</td><td>%0.1f</td><td>%0.1f</td><td>%0.4f</td></tr>\n" %(frameIndex, altitude, longitude, latitude, meanY)
            data1 += "[%0.1f, %0.4f],\n" %(altitude, meanY)
        table2 += "</table><br><br>"
            


    if observationType in ["I", "E", "G"]:

        chart1 = """
            <div id="container"></div>
            <script src="https://code.highcharts.com/highcharts.src.js"></script>
            <script>
                Highcharts.chart('container', {
                    chart: {
                        type: 'line',
                        width: 1000,
                        zoomType: 'x'
                    },
                    title: {
                        text: 'Solar Occultation Mean Transmittance Vs Altitude',
                    },
                    xAxis: {
                        title: {
                            text: 'Tangent Altitude Areoid',
                        },
                		},
                    yAxis: {
                        title: {
                            text: 'Mean Transmittance',
                        }
                    },
                    series: [{
                        name: 'Mean Transmittance',
                        data: [
                        %s
                        ],
                    }]
                });
            </script>
            """ %data1


        chart2 = """
            <div id="container2"></div>
            <script src="https://code.highcharts.com/highcharts.src.js"></script>
            <script>
                Highcharts.chart('container2', {
                    chart: {
                        type: 'line',
                        height: 1000,
                        zoomType: 'xy'
                    },
                    title: {
                        text: 'Solar Occultation Transmittances',
                    },
                    xAxis: {
                        title: {
                            text: 'Wavenumbers cm-1',
                        },
                		},
                    yAxis: {
                        title: {
                            text: 'Transmittance',
                        }
                    },


                    series: ["""
                    
        for frameIndex, (altitude, meanY, y) in enumerate(zip(altitudes, meanYs, ys)):
            if 0.01 < meanY < 0.99:
#            if 0.1 < meanY < 0.5:
#            if frameIndex == np.abs(meanYs - 0.5).argmin():
                
                data2 = ""
                for xPixel, yPixel in zip(x, y):
                    data2 += "[%0.2f, %0.4f],\n" %(xPixel, yPixel)

                chart2 += "{name: '%0.1fkm'," %altitude
                chart2 += """
                    data: [
                    %s
                    ]},""" %data2
        chart2 += """]
            });
            </script>
            """



    if observationType in ["D","N"]:

        chart1 = """
            <div id="container"></div>
            <script src="https://code.highcharts.com/highcharts.src.js"></script>
            <script>
                Highcharts.chart('container', {
                    chart: {
                        type: 'scatter',
                        width: 1000,
                        zoomType: 'x'
                    },
                    title: {
                        text: 'Nadir Surface Coverage',
                    },
                    xAxis: {
                        title: {
                            text: 'Longitude',
                        },
                		},
                    yAxis: {
                        title: {
                            text: 'Latitude',
                        }
                    },
                    series: [{
                        name: 'Nadir',
                        data: [
                        %s
                        ],
                    }]
                });
            </script>
            """ %data1


        chart2 = """
            <div id="container2"></div>
            <script src="https://code.highcharts.com/highcharts.src.js"></script>
            <script>
                Highcharts.chart('container2', {
                    chart: {
                        type: 'line',
                        height: 1000,
                        zoomType: 'xy'
                    },
                    title: {
                        text: 'Nadir Radiances',
                    },
                    xAxis: {
                        title: {
                            text: 'Wavenumbers cm-1',
                        },
                		},
                    yAxis: {
                        title: {
                            text: 'Radiance',
                        }
                    },


                    series: ["""
                    
        for frameIndex, (incidenceAngle, meanY, y) in enumerate(zip(incidenceAngles, meanYs, ys)):
            if 0.01 < incidenceAngle < 60.0:
#            if 0.1 < meanY < 0.5:
#            if frameIndex == np.abs(meanYs - 0.5).argmin():
                
                data2 = ""
                for xPixel, yPixel in zip(x, y):
                    data2 += "[%0.2f, %0.4f],\n" %(xPixel, yPixel)

                chart2 += "{name: '%0.1f degrees'," %incidenceAngle
                chart2 += """
                    data: [
                    %s
                    ]},""" %data2
        chart2 += """]
            });
            </script>
            """


    h += table1
    h += chart1
    h += chart2
    h += table2

    h += "</body></html>"
            
    return HttpResponse(h)





urlpatterns = [
    url(r"^$", channelList),
    url(r"^level/$", levelList),
    url(r"^level/month/$", monthList),
    url(r"^level/month/filelist/$", fileList),
    url(r"^level/month/filelist/hdf5_level_0p1a/$", hdf5_level),
    url(r"^level/month/filelist/hdf5_level_0p2a/$", hdf5_level),
    url(r"^level/month/filelist/hdf5_level_0p3a/$", hdf5_level),
    url(r"^level/month/filelist/hdf5_level_1p0a/$", hdf5_level),
]













