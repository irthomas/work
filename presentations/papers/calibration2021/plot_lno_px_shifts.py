# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:42:08 2020

@author: iant


CALCULATE PIXEL TEMPERATURE SHIFTS FROM GROUND CAL DATA


"""
import sys
#import os
#import h5py
import numpy as np
#from datetime import datetime
import re


import matplotlib.pyplot as plt
from matplotlib.widgets import Button

#from scipy.signal import savgol_filter, butter, lfilter
#from scipy.optimize import curve_fit

#from hdf5_functions_v03 import get_dataset_contents, get_hdf5_filename_list, get_hdf5_attribute
from tools.file.hdf5_functions import make_filelist
from tools.file.filename_lists import getFilenameList

from tools.spectra.baseline_als import baseline_als

from tools.plotting.colours import get_colours
from tools.spectra.fit_polynomial import fit_polynomial
from tools.sql.get_sql_spectrum_temperature import get_sql_spectrum_temperature

from presentations.papers.calibration2021.plot_figures_for_cal_paper_2020_functions import getExternalTemperatureReadings, findOrder


# DATA_TYPE = "ground"
DATA_TYPE = "inflight"

fileLevel = "hdf5_level_0p1a"
obspaths = []
model = "PFM"
title = ""





SUFFIX = ""
USE_CSL_TEMPERATURES = True
CSL_TEMPERATURE_COLUMN = 4
    
    
USE_TGO_TEMPERATURES = True
TGO_TEMPERATURE_COLUMN = -1


# for i in range(160, 200): print("%i:{\"aotf_inflight\":%0.f.}," %(i, order_data_dict["hdf5Filenames"]["20191207_051654_0p1a_LNO_1"]["aotf_frequencies_all"][i-110]))

order_dicts = {
# 130:{"aotf_frequency":18271.7, "molecule":"ch4", "aotf_inflight":18306.}, #20
# 131:{"aotf_frequency":18428.8, "molecule":"ch4", "aotf_inflight":18461.},
# 132:{"aotf_frequency":18585.8, "molecule":"ch4", "aotf_inflight":18616.},
# 133:{"aotf_frequency":18742.6, "molecule":"ch4", "aotf_inflight":18771.},
# 134:{"aotf_frequency":18899.3, "molecule":"ch4", "aotf_inflight":18927.},
# 135:{"aotf_frequency":19055.9, "molecule":"ch4"},
# 136:{"aotf_frequency":19212.4, "molecule":"ch4"},
# 137:{"aotf_frequency":19368.8, "molecule":"ch4", "aotf_inflight":19391.},
# 138:{"aotf_frequency":19525.1, "molecule":"ch4"},
# 139:{"aotf_frequency":19681.3, "molecule":"ch4"},

# 140:{"aotf_inflight":19856.},
# 141:{"aotf_inflight":20011.},
# 142:{"aotf_inflight":20166.},
# 143:{"aotf_frequency":20305.1, "molecule":"c2h2"},
# 144:{"aotf_frequency":20460.7, "molecule":"c2h2"},
# 145:{"aotf_frequency":20616.4, "molecule":"c2h2"},
# 146:{"aotf_frequency":20771.9, "molecule":"c2h2"},
# 147:{"aotf_frequency":20927.3, "molecule":"c2h2"},
# 148:{"aotf_frequency":21082.6, "molecule":"c2h2"},
# 149:{"aotf_frequency":21237.9, "molecule":"c2h2"},

# 151:{"aotf_inflight":21557.},

# 153:{"aotf_inflight":21867.},

# 155:{"aotf_inflight":22175.},
# 156:{"aotf_inflight":22330.},

# 160:{"aotf_frequency":22940.9, "molecule":"co2", "aotf_inflight":22948.},
# 161:{"aotf_frequency":23095.4, "molecule":"co2"},
#start from here
# 162:{"aotf_frequency":23249.8, "molecule":"co2", "aotf_inflight":23256.},
# 163:{"aotf_inflight":23410.},

# 166:{"aotf_frequency":23866.8, "molecule":"co2", "aotf_inflight":23873.},    
# 167:{"aotf_frequency":24020.9, "molecule":"co2", "aotf_inflight":24026.},    
# 168:{"aotf_inflight":24180.},
# 169:{"aotf_frequency":24329.0, "molecule":"co2", "aotf_inflight":24335.},    


# 170:{"aotf_frequency":24483.0, "molecule":"co2"},    
# 171:{"aotf_frequency":24637.0, "molecule":"co2"},    
# 172:{"aotf_frequency":24791.0, "molecule":"co2", "aotf_inflight":24796.},    
# 173:{"aotf_frequency":24944.9, "molecule":"co2", "aotf_inflight":24951.},    

# 178:{"aotf_inflight":25720.},
# 179:{"aotf_inflight":25873.}, #not great


# 180:{"aotf_inflight":26027.},

# 187:{"aotf_frequency":27097.4, "molecule":"co"},
# 188:{"aotf_frequency":27251.1, "molecule":"co"},
# 189:{"aotf_inflight":27409.},

# 190:{"aotf_frequency":27558.5, "molecule":"co"},

# 193:{"aotf_frequency":28019.6, "molecule":"co"},
# 194:{"aotf_frequency":28173.3, "molecule":"co", "aotf_inflight":28176.},
# 195:{"aotf_frequency":28327.1, "molecule":"co", "aotf_inflight":28330.},
# 196:{"aotf_inflight":28483.},
# 197:{"aotf_inflight":28636.},
# 198:{"aotf_inflight":28789.},




}



class DraggableLinePlot:
    def __init__(self, line):
        self.line = line
        self.press = None
        self.background = None
        self.lock = None  # only one can be animated at a time

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.line.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.line.axes: return
        if self.lock is not None: return
        contains, attrd = self.line.contains(event)
        if not contains: return
        x0 = self.line.get_xdata()
        y0 = self.line.get_ydata()
        print('event contains', x0[0], y0[0])
        self.press = x0, y0, event.xdata, event.ydata
        self.lock  = self

        # draw everything but the selected line and store the pixel buffer
        canvas = self.line.figure.canvas
        axes = self.line.axes
        self.line.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.line.axes.bbox)

        # now redraw just the 
        axes.draw_artist(self.line)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        'on motion we will move the line if the mouse is over us'
        if self.lock is not self:
            return
        if event.inaxes != self.line.axes: return
        x0, y0, xpress, ypress, = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.line.set_xdata(x0+dx)
        self.line.set_ydata(y0+dy)

        canvas = self.line.figure.canvas
        axes = self.line.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current line
        axes.draw_artist(self.line)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if self.lock  is not self:
            return

        self.press = None
        self.lock  = None

        # turn off the line animation property and reset the background
        self.line.set_animated(False)
        self.background = None

        # redraw the full figure
        self.line.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)



chosen_bins = [8,9,10,11,12]

pixels = np.arange(320)
    
colours = get_colours(42, cmap="plasma") #from -20C to +20C



order_data_dict = {}

for diffraction_order, order_dict in order_dicts.items():

    order_data_dict[diffraction_order] = {}


    order_data_dict[diffraction_order]["mean_gradient_all_bins"] = []
    order_data_dict[diffraction_order]["std_gradient_all_bins"] = []
    order_data_dict[diffraction_order]["n_gradients_all_bins"] = []

    
    
    if DATA_TYPE == "ground":
        order_data_dict[diffraction_order]["obspaths_all"] = getFilenameList("ground cal %s cell%s" %(order_dict["molecule"], SUFFIX))
        hdf5Files, hdf5Filenames, _ = make_filelist(order_data_dict[diffraction_order]["obspaths_all"], fileLevel, model=model, silent=True)

    elif DATA_TYPE == "inflight":
        regex = re.compile("(20161121_233000|20180702_112352|20181101_213226|20190314_021825|20190609_011514|20191207_051654)_0p1a_LNO_1")
        hdf5Files, hdf5Filenames, _ = make_filelist(regex, fileLevel)

    
    
    order_data_dict["hdf5Filenames"] = {}

    order_data_dict[diffraction_order]["measurement_temperatures"] = []
    order_data_dict[diffraction_order]["colour"] = []
    order_data_dict[diffraction_order]["hdf5_filenames"] = []
    order_data_dict[diffraction_order]["spectra"] = []
    order_data_dict[diffraction_order]["continuum_mean"] = []
    order_data_dict[diffraction_order]["continuum_std"] = []

    for file_index, (hdf5_file, hdf5_filename) in enumerate(zip(hdf5Files, hdf5Filenames)):

        detector_data_all = hdf5_file["Science/Y"][...]
        
#        window_top_all = hdf5_file["Channel/WindowTop"][...]
#        binning = hdf5_file["Channel/Binning"][0] + 1
#        integration_time = hdf5_file["Channel/IntegrationTime"][0]
        sbsf = hdf5_file["Channel/BackgroundSubtraction"][0]
        measurement_temperature = np.mean(hdf5_file["Housekeeping"]["SENSOR_1_TEMPERATURE_LNO"][2:10])
        datetimes = hdf5_file["DateTime"][...]
        
        aotf_frequencies = hdf5_file["Channel/AOTFFrequency"][...]
        
        order_data_dict["hdf5Filenames"][hdf5_filename] = {"aotf_frequencies_all":aotf_frequencies}
        
        
        if not sbsf:
            print("%s does not have background subtraction" %hdf5_filename)
            sys.exit()
            
        spectra_bins = np.zeros((detector_data_all.shape[0], len(chosen_bins), detector_data_all.shape[2]))
        for bin_index, chosen_bin in enumerate(chosen_bins):
            spectra_bins[:, bin_index, :] = detector_data_all[:, chosen_bin, :]
        
        spectra = np.mean(spectra_bins, axis=1)
        
        

        #selected matching orders in fullscan/miniscan observations
        if DATA_TYPE == "ground":
            desired_aotf = order_dict["aotf_frequency"]
            desired_aotf_range = 1.1
        elif DATA_TYPE == "inflight":
            desired_aotf = order_dict["aotf_inflight"]
            desired_aotf_range = 20.0

        chosen_order_indices = [i for i, aotf_frequency in enumerate(aotf_frequencies) if (desired_aotf-desired_aotf_range) < aotf_frequency < (desired_aotf+desired_aotf_range)]
        order_data_dict[diffraction_order]["aotf_frequencies_all"] = aotf_frequencies
        
        
        if DATA_TYPE == "inflight": #remove all but the first, as there are many in a single observation
            chosen_order_indices = [chosen_order_indices[0]]

        #loop through spectra where aotf frequency is close enough to specified value
        for chosen_order_index in chosen_order_indices:
            
            spectrum = spectra[chosen_order_index, :]
            measurement_time = datetimes[chosen_order_index]
            
            normalised_spectrum = spectrum / np.max(spectrum)
            
            if DATA_TYPE == "ground":
                if USE_CSL_TEMPERATURES: #overwrite temperature with one from external file
                    measurement_temperature = getExternalTemperatureReadings(measurement_time.decode(), CSL_TEMPERATURE_COLUMN)

            if DATA_TYPE == "inflight":
                if USE_TGO_TEMPERATURES: #overwrite temperature with one from external file
                    measurement_temperature = get_sql_spectrum_temperature(hdf5_file, chosen_order_index)


            order_data_dict[diffraction_order]["hdf5_filenames"].append(hdf5_filename)
            order_data_dict[diffraction_order]["measurement_temperatures"].append(measurement_temperature)
            order_data_dict[diffraction_order]["colour"].append(colours[int(measurement_temperature)+20])
            
            
            #remove continuum
            continuum = baseline_als(normalised_spectrum)
            order_data_dict[diffraction_order]["continuum_mean"].append(np.mean(continuum))
            order_data_dict[diffraction_order]["continuum_std"].append(np.std(continuum))

            absorption_spectrum = normalised_spectrum / continuum
            #normalise between 0 and 1
            absorption_spectrum = (absorption_spectrum - np.min(absorption_spectrum))/ (np.max(absorption_spectrum) - np.min(absorption_spectrum))
            order_data_dict[diffraction_order]["spectra"].append(absorption_spectrum)


        if len(chosen_order_indices) == 0:
            text = "AOTF frequency %0.0f kHz (order %i) %0.1fC not found in file %s" %(desired_aotf, diffraction_order, measurement_temperature, hdf5_filename)
            aotf_frequency_all = hdf5_file["Channel/AOTFFrequency"][...]
            diffraction_orders = np.asfarray([findOrder("lno", aotf_frequency, silent=True) for aotf_frequency in aotf_frequency_all])
            text += " (%0.0f-%0.0fkHz; orders=%i-%i)" %(min(aotf_frequency_all), max(aotf_frequency_all), min(diffraction_orders), max(diffraction_orders))
            print(text)

        else:
            print("AOTF frequency %0.0f kHz (order %i) %0.1fC found in file %s. Adding to search list" %(desired_aotf, diffraction_order, measurement_temperature, hdf5_filename))
                        
            

    #sort by temperature
    sort_indices = np.argsort(np.asfarray(order_data_dict[diffraction_order]["measurement_temperatures"]))
    
    for list_name in ["colour", "continuum_mean", "continuum_std", "hdf5_filenames", "measurement_temperatures", "spectra"]:
        order_data_dict[diffraction_order][list_name] = [order_data_dict[diffraction_order][list_name][i] for i in sort_indices]
    
    
            


    fig = plt.figure(figsize=(11,3))
    ax = fig.add_subplot(111)
    first_line = ax.plot(np.arange(320.0), order_data_dict[diffraction_order]["spectra"][0], linestyle="--")
    lines = ax.plot(np.arange(320.0), np.asfarray(order_data_dict[diffraction_order]["spectra"])[1:, :].T)
    plt.xlabel("Pixel Number")
    plt.ylabel("Normalised transmittance")
    plt.legend(order_data_dict[diffraction_order]["measurement_temperatures"])
    plt.tight_layout()
    # plt.savefig("solar_line_temperature_spectra.png", dpi=600)
    # stop()
    
        
    drs = []
    for line in lines:
        dr = DraggableLinePlot(line)
        dr.connect()
        drs.append(dr)
    
    
    
    class button_actions(object):
        
        def save(self, event):
            output_text = "%i, %s: " %(diffraction_order, DATA_TYPE)
            output_text += "%0.3f, %0.3f; " %(order_data_dict[diffraction_order]["measurement_temperatures"][0], 0.0)
    
            for line_index, line in enumerate(lines):
                output_text += "%0.3f, %0.3f; " %(order_data_dict[diffraction_order]["measurement_temperatures"][line_index+1], -1.0*line.get_xdata()[0])
    
            print(output_text)
            with open("pixel_shifts.txt", "a") as f:
                f.write(output_text+"\n")
    
        def close(self, event):
            print("Closing")
            plt.close()
    
    
    callback = button_actions()
    ax_button_save = plt.axes([0.88, 0.05, 0.05, 0.075])
    ax_button_close = plt.axes([0.94, 0.05, 0.05, 0.075])
    
    button_save = Button(ax_button_save, "Save")
    button_close = Button(ax_button_close, "Close")
    
    button_save.on_clicked(callback.save)
    button_close.on_clicked(callback.close)
    
    fig.canvas.manager.full_screen_toggle()
    plt.show()


i = -1
if len(order_dicts) == 0:
    with open("reference_files/pixel_shifts.txt", "r") as f:
        
        shift_dict = {}
        
        text_lines = f.readlines()
        for text_line in text_lines:
            i += 1
            text_line_start = text_line.split(":")[0]
            diffraction_order = int(text_line_start.split(",")[0])
            cal_type = text_line_start.split(",")[1].strip()
            
            if cal_type == "inflight":
                shift_dict[i] = {"diffraction_order":diffraction_order, "cal_type":cal_type, "temperature":[], "shift":[]}
                split_texts_line = text_line.split(":")[1].replace("\n","").split(";")
                for split_text_line in split_texts_line:
                    
                    if split_text_line != " ":
                        temperature = float(split_text_line.split(",")[0])
                        shift = float(split_text_line.split(",")[1])
                    
                        shift_dict[i]["temperature"].extend([temperature])
                        shift_dict[i]["shift"].extend([shift])

    plt.figure(figsize=(11,6))
    plt.xlabel("Instrument temperature")
    plt.ylabel("Relative pixel shift from lowest temperature")
    plt.xlim((-20, 10))
    plt.tight_layout()
    
    
    colours = get_colours(len(shift_dict))
    
    i = -1
    for shift_data in shift_dict.values():
        i += 1
        poly_fit = fit_polynomial(shift_data["temperature"], shift_data["shift"], degree=1, coeffs=True)
        shift_data["gradient"] = poly_fit[1][0]
        
        # plt.scatter(shift_dict[i]["temperature"], shift_dict[i]["shift"], c=[colours[i]], alpha=0.5)
        # linestyle = {"ground":"-", "inflight":"--"}[shift_dict[i]["cal_type"]]
        # plt.plot(shift_dict[i]["temperature"], poly_fit[0], color=colours[i], linestyle=linestyle, label="Order %i: gradient = %0.3f" %(shift_dict[i]["diffraction_order"], shift_dict[i]["gradient"]), alpha=0.5)

        if shift_data["cal_type"] == "inflight":
            plt.scatter(shift_data["temperature"], shift_data["shift"], c=[colours[i]], alpha=0.5)
            plt.plot(shift_data["temperature"], poly_fit[0], color=colours[i], linestyle="-", label="Order %i: gradient = %0.3f" %(shift_data["diffraction_order"], shift_data["gradient"]), alpha=0.5)
    plt.legend(loc="lower right", prop={'size': 7})


    
    plt.figure(figsize=(9,5))
    plt.title("Pixel shift versus temperature")
    # plt.scatter([shift_dict[i]["diffraction_order"] for i in shift_dict.keys() if shift_dict[i]["cal_type"] == "ground"], [shift_dict[i]["gradient"] for i in shift_dict.keys() if shift_dict[i]["cal_type"] == "ground"], c="b", label="Ground calibration")
    # plt.scatter([shift_dict[i]["diffraction_order"] for i in shift_dict.keys() if shift_dict[i]["cal_type"] == "inflight"], [shift_dict[i]["gradient"] for i in shift_dict.keys() if shift_dict[i]["cal_type"] == "inflight"], c="r", label="Inflight calibration")

    orders = [shift_dict[i]["diffraction_order"] for i in shift_dict.keys() if shift_dict[i]["cal_type"] == "inflight"]
    gradients = [shift_dict[i]["gradient"] for i in shift_dict.keys() if shift_dict[i]["cal_type"] == "inflight"]
    plt.scatter(orders, gradients, c="b")
    #add line to show mean gradient
    mean_gradient = np.mean(gradients)
    std_gradient = np.std(gradients)
    plt.axhline(mean_gradient, color="b", linestyle="--", label="Mean gradient = %0.3f" %mean_gradient)
    plt.fill_between([120, 200], mean_gradient - std_gradient, mean_gradient + std_gradient, alpha=0.5, label="Mean +- 1 standard deviation")
    plt.xlim((129, 199))
    plt.ylim((0.81, 0.85))
    

    plt.legend()
    plt.xlabel("Diffraction order")
    plt.ylabel("Gradient (pixel shift per degree temperature change)")
    