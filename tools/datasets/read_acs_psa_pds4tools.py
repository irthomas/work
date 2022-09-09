# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:16:18 2022

@author: iant

DIFFERENT ROWS ILLUMINATED IN REFERENCE FRAME
DOES ORDER STRIPE NUMBER MEAN CENTRE OF ROW? WHAT ABOUT THE OTHER ROWS
ABSORPTION LINES APPEAR TO BE TILTED BUT WAVELENGTH FRAME SHOWS SAME VALUES
HOW IS THE DATA ANALYSED? EXPECTED SPECTRA RATHER THAN DETECTOR FRAMES AS A SCIENTIST WOULD USE
DETECTOR ARTEFACTS PRESENT, STREAK ACROSS ROWS 95-105, 
IS THERE A DOUBLE ILS? REFERENCE TO THIS IN THE LITERATURE
"""

import os
import pds4_tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

from tools.plotting.colours import get_colours
from tools.file.write_log import write_log

plot = True
# plot = False

# log = True
log = False

ACS_DATA_PATH_ROOT = r"C:\Users\iant\Documents\DATA\psa\ACS"

ACS_DATA_PATH_CALIBRATED = os.path.join(ACS_DATA_PATH_ROOT, "data_calibrated", "Science_Phase")#, "Orbit_Range_5600_5699")


channel = "mir"
# channel = "nir"
# channel = "tir_occ"

xml_paths = sorted(glob.glob(ACS_DATA_PATH_CALIBRATED + os.sep + "**" + os.sep + "*%s*.xml" %channel.replace("_","*"), recursive=True))



ACS_GEOMETRY_PATH = os.path.join(ACS_DATA_PATH_ROOT, "geometry", "Science_Phase")#, "Orbit_Range_5600_5699")



# xml_dts = []
# for xml_path in xml_paths[0:10]:
    
#     xml_filename = os.path.basename(xml_path)





for xml_path in xml_paths[0:1]:
    
    xml_filename = os.path.basename(xml_path)
    orbit_dir = os.path.basename(os.path.dirname(xml_path))
    orbit_range_dir = os.path.basename(os.path.dirname(os.path.dirname(xml_path)))
    
    
    
    structures = pds4_tools.read(xml_path)
    
    
    
        
    if channel in ["mir", "nir"]:

        if plot:
            header = structures["Header"]
            #NIR: TOA orders x (counts and error) x detector rows x detector spectral pixels
            #MIR: 1 x detector rows x detector spectral pixels (remove 1)
            reference_s = structures["Reference"]
            reference_data = reference_s.data
            if reference_data.ndim == 3:
                reference_data = reference_data[0] #remove 1d dimension
            
            
            #MIR: 1 x detector rows x detector spectral pixels (remove 1). Each frame contains multiple orders in different rows
            wavelength_s = structures["Wavelength"]
            wavelength_data = wavelength_s.data
            
            if wavelength_data.ndim == 3:
                wavelength_data = wavelength_data[0] #remove 1d dimension
            wavelength_data[wavelength_data == 0] = np.nan
            
            
            #MIR: 1 x orders x (order and stripe number). 
            orders_s = structures["Orders"]
            orders_data = orders_s.data[...]
            
            if orders_data.ndim == 3:
                orders_data = orders_data[0, :, :] #2D array

            #MIR: frames x detector rows x (transmittance and error) x columns
            data_s = structures["Data"]
            data = data_s.data


            
    elif channel in ["tir_occ"]:
        
        #TIRVIM is completely different
        structures_dict = {"SpectrumRe":{},
                      "SpectrumIm":{},
                      "ResPhase":{},
                      "PhaseError":{},
                      "Transmittance":{}}
        
        fig1, axes1 = plt.subplots(nrows=4, sharex=True)
        fig2, ax2 = plt.subplots(nrows=1)
        
        for i, structure in enumerate(structures_dict.keys()):
            data_s = structures[structure]
            data = data_s.data
            
            if structure == "Transmittance":
                ax2.plot(data)
            else:
                axes1[i].plot(data, label=structure)
            
        # plt.legend()
        
        tangent_altitude = structures.label.findall('em16_tgo_acs:tir_target_altitude')
        stop()

    
    


    #get geometry from aux file
    lid_refs = structures.label.findall('.//lid_reference')
    
    geometry_lids = [lid_ref.text for lid_ref in lid_refs if "geometry" in lid_ref.text]
    
    if len(geometry_lids) == 1:
        geometry_lid = geometry_lids[0]
        
    if not geometry_lid:
        print("Error: no geometry")

    else:
        geometry_xml_path = os.path.join(ACS_GEOMETRY_PATH, orbit_range_dir, orbit_dir, geometry_lid.split(":")[-1]+".xml")
        geom_structures = pds4_tools.read(geometry_xml_path)
        
        geometry1_s = geom_structures["Geometry-1"]
        geometry1=pd.DataFrame(geometry1_s.data).to_numpy()
        
        tangent_altitudes = geometry1[:, -1]


    
    if channel in ["mir"]: 
        geometry2_s = geom_structures["Geometry-2"]
        geometry3_s = geom_structures["Geometry-3"]
        
        geometry2=pd.DataFrame(geometry2_s.data).to_numpy()
        geometry3=pd.DataFrame(geometry3_s.data).to_numpy()
    
        if log:
            write_log("%s %i %i %i %i %i" %(xml_filename, data.shape[0], geometry1.shape[0], geometry2.shape[0], geometry3.shape[0], data.shape[0]-geometry1.shape[0]))
    
    else:
        if log:
            write_log("%s %i %i %i" %(xml_filename, data.shape[0], geometry1.shape[0], data.shape[0]-geometry1.shape[0]))
    


    
    
    
    if plot:
        if channel in ["mir"]: 
        
        
        
            plt.figure(figsize=(12, 8), constrained_layout=True)
            plt.title("MIR reference frame counts %s" %xml_filename)
            plt.imshow(reference_data)
            plt.colorbar()
        
            
        
            plt.figure(figsize=(12, 8), constrained_layout=True)
            plt.title("MIR wavelength frame %s" %xml_filename)
            plt.imshow(wavelength_data)
            plt.colorbar()
        
            frame_no = 101
            plt.figure(figsize=(12, 8), constrained_layout=True)
            plt.title("MIR transmittance frame %s" %xml_filename)
            plt.imshow(data[frame_no, :, 0, :], vmin=0.9, vmax=1.1)
            plt.colorbar()
        
        
            #to plot data, first get stripe column numbers
            #then plot transmittance vs wavelength for each stripe
            plt.figure(figsize=(12, 8), constrained_layout=True)
            for order, stripe_no in orders_data[:, :]:
                plot_wavenumbers = wavelength_data[stripe_no, :]
                plot_transmittance = data[frame_no, stripe_no, 0, :]
                plt.plot(plot_wavenumbers, plot_transmittance, label=order)
            plt.grid()
            plt.legend()
        
        
        
        
        if channel in ["nir"]: 
            
            
            orders = np.array([i[0] for i in orders_data])
            unique_orders = np.array(sorted(list(set(orders))))
            
            # nu_px = 
            
            for unique_order in unique_orders[0:1]:
                frame_ixs = np.where(unique_order == orders)[0]
                
                colours = get_colours(len(frame_ixs))
                
                nu_px = wavelength_data[frame_ixs[0]]
        
        
                plt.figure(figsize=(18, 8), constrained_layout=True)
                plt.title("NIR occultation spectra %s order %i" %(xml_filename, unique_order))
                for i, frame_ix in enumerate(frame_ixs):
                    
                    for row_ix in range(data.shape[2]):


                        if len(tangent_altitudes) == data.shape[0]:
                            tangent_altitude = tangent_altitudes[frame_ix]
                            if row_ix == 0:
                                label = "%0.2fkm" %tangent_altitude
                            else:
                                label = ""
                        else:
                            tangent_altitude = 0.0
                            label = ""
                    
                        if tangent_altitude > 0.0:
                            y = data[frame_ix, 0, row_ix, :]
                            y_error = data[frame_ix, 1, row_ix, :]
                            
                            print(tangent_altitude, np.mean(y), np.std(y))
                            if np.mean(y) < 0.999 and np.std(y) > 0.004:
                                plt.plot(nu_px, y, color=colours[i], label=label)
                            elif np.mean(y) < 0.9:
                                plt.plot(nu_px, y, color=colours[i], label=label)
                plt.legend(prop={'size': 8})
                plt.ylabel("Transmittance")
                plt.xlabel("Wavenumber cm-1")
                plt.grid()
                
                # stop()
                
            
            
            # plt.figure(figsize=(12, 8), constrained_layout=True)
            # plt.title("NIR reference frame counts %s" %xml_filename)
            # plot_labels = ["Order %i, %0.1fcm-1" %(i[0], i[1]) for i in orders_data[0:10]]
            # plot_wavenumbers = [i[0] for i in orders_data[0:10]]
            # for i in range(reference_data.shape[2]):
            #     if i == 0:
            #         plt.plot(wavelength_data[0:10, :].T, reference_data[:, 0, i, :].T, label=plot_labels)
            #     else:
            #         plt.plot(wavelength_data[0:10, :].T, reference_data[:, 0, i, :].T)
            # plt.legend()
            # plt.xlabel("Wavenumber cm-1")
            # plt.grid()
            # # plt.savefig(xml_filename.replace(".xml", "_counts.png"))
            
            # plt.figure(figsize=(12, 8), constrained_layout=True)
            # plt.title("NIR reference frame transmittance error? %s" %xml_filename)
            # for i in range(reference_data.shape[2]):
            #     if i == 0:
            #         plt.plot(wavelength_data[0:10, :].T, reference_data[:, 1, i, :].T, label=plot_labels)
            #     else:
            #         plt.plot(wavelength_data[0:10, :].T, reference_data[:, 1, i, :].T)
            # plt.legend()
            # plt.xlabel("Wavenumber cm-1")
            # plt.grid()
            # # plt.savefig(xml_filename.replace(".xml", "_error.png"))
            
            
        
        
    
