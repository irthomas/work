# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:16:18 2022

@author: iant
"""

import os
import pds4_tools
import matplotlib.pyplot as plt

ACS_DATA_PATH_ROOT = r"C:\Users\iant\Documents\DATA\psa\ACS"

ACS_DATA_PATH_CALIBRATED = os.path.join(ACS_DATA_PATH_ROOT, "data_calibrated", "Science_Phase", "Orbit_Range_5600_5699")
ACS_GEOMETRY_PATH = os.path.join(ACS_DATA_PATH_ROOT, "geometry", "Science_Phase", "Orbit_Range_5600_5699")

orbit_dir = "Orbit_5666"

xml_filename = "acs_cal_sc_nir_20190301T201042-20190301T201643-5666-1-2.xml"

xml_path = os.path.join(ACS_DATA_PATH_CALIBRATED, orbit_dir, xml_filename)


structures = pds4_tools.read(xml_path)

header = structures["Header"]

# header_field_md = header.meta_data["Record_Binary"]["Field_Binary"]
# header_fields = [md["description"] for md in header_field_md]


# header_group_field_md = header.meta_data["Record_Binary"]["Group_Field_Binary"]


#TOA orders x (counts and error) x detector rows x detector spectral pixels
reference = structures["Reference"]
reference_data = reference.data

wavelength = structures["Wavelength"]
wavelength_data = wavelength.data

orders = structures["Orders"]
orders_data = orders.data


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
# plt.savefig(xml_filename.replace(".xml", "_counts.png"))

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
# plt.savefig(xml_filename.replace(".xml", "_error.png"))

lid_refs = structures.label.findall('.//lid_reference')

geometry_lids = [lid_ref.text for lid_ref in lid_refs if "geometry" in lid_ref.text]

if len(geometry_lids) == 1:
    geometry_lid = geometry_lids[0]
    
geometry_xml_path = os.path.join(ACS_GEOMETRY_PATH, orbit_dir, geometry_lid.split(":")[-1]+".xml")
geom_structures = pds4_tools.read(geometry_xml_path)

geometry = geom_structures["Geometry-1"]
geometry_data = geometry.data