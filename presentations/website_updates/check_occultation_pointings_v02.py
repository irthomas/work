# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:18:43 2026

@author: iant

CHECK OCCULTATION MISPOINTING FOR ADDING TO WEBSITE USING HDF5 AND OPENDAP V02
USE one_filename_per_occ_dict FUNCTION TO CHECK ALL SOLAR OCCULTATIONS, BEST ORDERS FIRST

"""
from psa_utils import tap
import re
import matplotlib.pyplot as plt
import numpy as np
import os
# from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from tools.file.hdf5_functions import open_hdf5_file
from tools.general.progress_bar import progress
from tools.file.paths import paths, FIG_X_PDF, FIG_Y_PDF
from tools.file.one_filename_per_occ_dict import one_filename_per_occ_dict

# for writing code for website
IMG_WIDTH = 400
IMG_HEIGHT = 400

PLOT = True
# PLOT = False

PLOT_PX_IXS = [200]


ARCMIN_CUTOFF = 0.7

LINK_PSA_PRODUCT_NAMES = True
# LINK_PSA_PRODUCT_NAMES = False


# new code
# get dictionary of files
path = paths["DATA_DIRECTORY"]
file_level = "hdf5_level_1p0a"

# all
# lid date range must match regex
regex = re.compile("........_......_1p0a_SO_*")
lid = "urn:esa:psa:em16_tgo_nmd:data_calibrated:nmd_cal_sc_so_%%"

# test: august 2018 only
# regex = re.compile("201808.._......_1p0a_SO_*")
# lid = "urn:esa:psa:em16_tgo_nmd:data_calibrated:nmd_cal_sc_so_201808%%"


if LINK_PSA_PRODUCT_NAMES:
    print("Getting PSA lid list")
    channel = "SO"
    psa = tap.PsaTap(tap_url="https://psa.esa.int/psa-tap/tap/")
    query = "SELECT * FROM psa.epn_core WHERE processing_level = 3 AND obs_id LIKE '%s'" % lid
    results_dict = psa.query(query).to_dict("list")
    # get lids
    lids = sorted(results_dict["obs_id"])


d = one_filename_per_occ_dict(regex, file_level, path)

# check if preferred orders in the file
order_preference = [134, 136, 133, 129, 128, 127, 132, 121, 119]  # etc.


bad_filelist = []
n_files = 0

for key in progress(d.keys()):
    n_orders = d[key]["n_orders"]

    if n_orders > 29:
        continue

    orders = d[key]["orders"]
    ing_eg = key[1]

    # find best order to plot
    order_to_plot = 0
    for order in order_preference:
        if order in orders:
            order_to_plot = order
            break

    # if none found, take first
    if order_to_plot == 0:
        order_to_plot = orders[0]

    ix = [i for i, order in enumerate(orders) if order == order_to_plot]

    if len(ix) != 1:
        print("Error")
    else:
        ix = ix[0]

    # get best h5 filename
    h5 = d[key]["filenames"][ix]

    # print(h5)

    # open file, check pointing
    h5f = open_hdf5_file(h5, path=path, silent=True)

    tangentAlt = h5f["Geometry/Point0/TangentAltAreoid"][:, 0]
    tangentAltIndices = np.where(tangentAlt > 0.0)[0]
    tangentAlts = tangentAlt[tangentAltIndices]
    sunWobble = h5f["Geometry/FOVSunCentreAngle"][tangentAltIndices, 0]

    bins = h5f["Science/Bins"][tangentAltIndices, 0]
    uniqueBins = sorted(list(set(bins)))
    bin_ixs_all = [np.where(bins == binStart)[0] for binStart in uniqueBins]  # get bin indices for all bins

    # if PLOT:
    #     plt.plot(tangentAlts[binIndices], sunWobble[binIndices], color=colours[fileIndex], label=hdf5_filename[:15])

    if max(sunWobble[bin_ixs_all[0]]) > ARCMIN_CUTOFF:

        # link to PSA products
        if LINK_PSA_PRODUCT_NAMES:
            h5_mod = h5[:15].replace("_", "t")
            h5_prefix = h5.rsplit("_", maxsplit=1)[0]
            lids_split = [lid.rsplit("-", maxsplit=2) for lid in lids]

            lid_ix = [i for i, lid_split in enumerate(lids_split) if h5_mod in lid_split[0] and int(lid_split[2]) == order_to_plot
                      and lid_split[1].lower() == ing_eg.lower()]

            if len(lid_ix) != 1:
                print("Error:", h5_prefix, lid_ix)
                continue
            else:
                lid_ix = lid_ix[0]
                lid = lids_split[lid_ix][0] + "-" + lids_split[lid_ix][1]

            bad_filelist.append([h5_prefix, lid])
            title = "%s\n%s" % (h5_prefix, lid)
        else:
            bad_filelist.append([h5_prefix, ""])
            title = h5_prefix

        if PLOT:

            y = h5f["Science/Y"][tangentAltIndices, :]

            fig, ax1 = plt.subplots(figsize=(FIG_X_PDF, FIG_Y_PDF))
            ax1.set_title(title)
            ax1.set_xlabel("Tangent altitude above areoid (km)")
            ax1.set_ylabel("Transmittance")
            ax2 = ax1.twinx()
            ax2.set_ylabel("Solar Pointing Deviation arcmins")
            for bin_ixs in bin_ixs_all:
                for pixel_ix in PLOT_PX_IXS:
                    ax1.plot(tangentAlts[bin_ixs], y[bin_ixs, pixel_ix])

            ax2.plot(tangentAlts[bin_ixs], sunWobble[bin_ixs], "k--")
            plt.tight_layout()
            plt.savefig("Solar_Pointing_Deviation_%s.png" % h5_prefix)
            plt.close()

    h5f.close()
    n_files += 1


print("Bad occultations:")
for bad_h5 in bad_filelist:
    print(bad_h5[:15])
print("%i of %i files analysed are bad" % (len(bad_filelist), n_files))

h = ""
h += "<h4>Some solar occultation observations are known to be affected by spacecraft pointing issues. Below is a list of file prefixes where a problem was detected. UVIS occultations and all diffraction orders with these filenames will be affected.</h4>"
h += "<br>"
h += "<h4>List of bad occultations as of %s:</h4>" % str(datetime.now())[:10]
h += "<table border='1'>\n"
h += "<tr>\n   <th>HDF5 (internal) filename</th><th>ESA PSA product name (external)</th>\n"

for bad_h5, bad_lid in bad_filelist:
    h += "<tr>\n"
    h += "   <td>%s</td><td>%s</td>\n" % (bad_h5, bad_lid)
    h += "<tr>\n"

h += "</table>\n"
h += "<br>\n"
h += "<br>\n"

h += "<h4>Images:</h4>\n"
h += "<p>In the figures below, the transmittance for a single pixel is given as a function of altitude. Each detector bin is plotted individually (bin 0 is blue; bin 1 is orange; bin 2 is green; bin 3 is red).</p>\n"
h += "<p>The dotted black line shows the deviation from the nominal pointing - if the deviation is sufficient that the field of view of a detector bin moves off the solar disk, the transmittance will decrease. These features should <b>not</b> be interpreted as clouds or other properties of the Martian atmosphere.</p>\n"
h += "<br>\n"


h += "<p>\n"

for bad_h5, bad_lid in bad_filelist:
    h += "<a href=\"ProjectDir/images/bad_observations/Solar_Pointing_Deviation_%s.png\" target=\"_blank\"> <img src=\"ProjectDir/images/bad_observations/Solar_Pointing_Deviation_%s.png\" alt=\"\" width=\"%s\" height=\"%s\" /></a>\n" % (
        bad_h5, bad_h5, IMG_WIDTH, IMG_HEIGHT)
h += "</p>\n"
h += "<br>\n"
h += "<br>\n"


h += "Made with presentations.website_updates.check_occultation_pointings.py<br>\n"
h += "Page last updated %s<br>\n" % str(datetime.now())[:10]

with open("occultation_pointing_errors.html", "w", encoding="utf-8") as f:
    for i, line in enumerate(h):
        f.write(line)

print("Images saved to current directory, html written to occultation_pointing_errors.html")
