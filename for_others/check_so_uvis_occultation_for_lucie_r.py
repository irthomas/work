# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:58:57 2024

@author: iant

PLOT SO AND UVIS OCCULTATIONS FOR 30TH AUGUST FOR LUCIE RIU
"""


import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from tools.file.hdf5_functions import make_filelist2  # , open_hdf5_file
# from tools.spectra.baseline_als import baseline_als
from tools.plotting.colours import get_colours

regex = re.compile("20240830_020601_.*_SO_.*")
regex2 = re.compile("20240830_020601_.*_UVIS_[IE]")

regex = re.compile("20240830_......_.*_SO_.*")
regex2 = re.compile("20240830_......_.*_UVIS_[IE]")

name = regex.pattern[0:15]


def get_so_data(regex, bin_ix):
    file_level = "hdf5_level_1p0a"

    h5_fs, h5s, _ = make_filelist2(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")

    d = {}
    for file_ix, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):

        print("Loading %i/%i" % (file_ix, len(h5s)))

        lats = h5_f["Geometry/Point0/Lat"][:, 0]
        lons = h5_f["Geometry/Point0/Lon"][:, 0]
        alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
        y = h5_f["Science/Y"][...]
        x = h5_f["Science/X"][...]

        bins = h5_f["Science/Bins"][:, 0]
        orders = h5_f["Channel/DiffractionOrder"][...]

        unique_bins = list(set(bins))
        unique_bin = unique_bins[bin_ix]

        bin_ixs = np.where(bins == unique_bin)[0]

        h5_prefix = make_prefix(h5)
        if h5_prefix not in d.keys():
            d[h5_prefix] = {}

        d[h5_prefix][h5] = {"lats": lats[bin_ixs], "lons": lons[bin_ixs], "alts": alts[bin_ixs],
                            "y": y[bin_ixs, :], "x": x[bin_ixs, :], "orders": orders[bin_ixs]}

    return d


def get_uvis_data(regex):
    file_level = "hdf5_level_1p0a"

    h5_fs, h5s, _ = make_filelist2(regex, file_level, path=r"W:\data\SATELLITE\TRACE-GAS-ORBITER\NOMAD\hdf5")

    d = {}
    for file_ix, (h5, h5_f) in enumerate(zip(h5s, h5_fs)):

        print("Loading %i/%i" % (file_ix, len(h5s)))

        lats = h5_f["Geometry/Point0/Lat"][:, 0]
        lons = h5_f["Geometry/Point0/Lon"][:, 0]
        alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
        y = h5_f["Science/Y"][...]
        x = h5_f["Science/X"][...]

        h5_prefix = make_prefix(h5)
        if h5_prefix not in d.keys():
            d[h5_prefix] = {}

        d[h5_prefix][h5] = {"lats": lats, "lons": lons, "alts": alts, "y": y, "x": x}

    return d


def make_prefix(h5):
    if "UVIS" in h5:
        h5_prefix = h5[0:15] + "_" + {"I": "Ingress", "E": "Egress"}[h5.split("_")[4]]
    elif "SO" in h5:
        h5_prefix = h5[0:15] + "_" + {"I": "Ingress", "E": "Egress", "S": "Fullscan"}[h5.split("_")[5]]
    return h5_prefix


if "d" not in globals():
    d = get_so_data(regex, bin_ix=1)
    d2 = get_uvis_data(regex2)


with PdfPages("solar_occultation_%s.pdf" % name) as pdf:

    print("Plotting lat lon")
    plt.figure(figsize=(20, 8), constrained_layout=True, rasterized=True)
    for h5_prefix in d.keys():
        for h5 in list(d[h5_prefix].keys())[0:1]:  # just take first one

            alts = d[h5_prefix][h5]["alts"]

            ixs_low = np.where(alts < 100.)[0]

            lats_low = d[h5_prefix][h5]["lats"][ixs_low]
            lons_low = d[h5_prefix][h5]["lons"][ixs_low]
            alts_low = d[h5_prefix][h5]["alts"][ixs_low]

            plot = plt.scatter(lons_low, lats_low, c=alts_low)
            plt.text(lons_low[0]+3, lats_low[0]-2, h5_prefix)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.grid()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    cbar = plt.colorbar(plot)
    cbar.set_label("Tangent Altitude", rotation=270, labelpad=20)

    pdf.savefig()
    plt.close()

    # loop through h5 prefix, plotting each transmittance vs altitude
    for h5_prefix in d.keys():

        if "Fullscan" not in h5_prefix:

            error = False
            plt.figure(figsize=(20, 8), constrained_layout=True, rasterized=True)
            for h5 in d[h5_prefix].keys():

                # y = d[h5]["y"]
                # x = d[h5]["x"]

                alts = d[h5_prefix][h5]["alts"]
                ixs_low = np.where(alts < 100.)[0]
                lats_low = d[h5_prefix][h5]["lats"][ixs_low]

                # if np.mean(lats_low) > 45. or np.mean(lats_low) < -45.:
                #     error = True
                #     continue  # ignore high latitudes

                ixs_low = np.where(alts < 100.)[0]
                lats_low = d[h5_prefix][h5]["lats"][ixs_low]

                orders = d[h5_prefix][h5]["orders"]

                y_mean = np.max(d[h5_prefix][h5]["y"][:, 150:250], axis=1)

                plt.plot(y_mean, alts, label="Order %s" % orders[0])

            # add UVIS
            if h5_prefix in d2.keys():
                for h5 in d2[h5_prefix].keys():
                    alts = d2[h5_prefix][h5]["alts"]

                    y_mean = np.nanmean(d2[h5_prefix][h5]["y"], axis=1)

                    plt.plot(y_mean, alts, label="UVIS")

            if error:
                print("Error found - maybe the occultation is a fullscan?")
                plt.close()
            else:
                plt.grid()
                plt.xlabel("Transmittance")
                plt.ylabel("Altitude (km)")
                plt.legend()
                plt.title("%s: latitudes %0.1f - %0.1f" % (h5_prefix, np.min(lats_low), np.max(lats_low)))
                plt.ylim([0, 100])

                pdf.savefig()
                plt.close()

    # loop through h5 prefix, plotting transmittance spectra
    for h5_prefix in d.keys():

        if "Fullscan" not in h5_prefix:

            error = False
            if h5_prefix in d2.keys():
                n_cols = len(d[h5_prefix].keys()) + 1
            else:
                n_cols = len(d[h5_prefix].keys())
            fig, axes = plt.subplots(ncols=n_cols, figsize=(20, 8), sharey=True, constrained_layout=True, rasterized=True, squeeze=False)
            for i, h5 in enumerate(d[h5_prefix].keys()):

                orders = d[h5_prefix][h5]["orders"]

                alts = d[h5_prefix][h5]["alts"]
                ixs_low = np.where(alts < 100.)[0]
                lats_low = d[h5_prefix][h5]["lats"][ixs_low]

                # if np.mean(lats_low) > 45. or np.mean(lats_low) < -45.:
                #     error = True
                #     continue  # ignore high latitudes

                y = d[h5_prefix][h5]["y"]
                x = d[h5_prefix][h5]["x"]

                colours = get_colours(x.shape[0])

                for xi, yi, colour in zip(x, y, colours):
                    colour = [np.max((0.0, i-0.2)) for i in colour]
                    axes[0, i].plot(xi, yi, color=colour)

                axes[0, i].set_title("Order %i" % orders[0])
                axes[0, i].grid()

            # add UVIS
            if h5_prefix in d2.keys():
                for h5 in d2[h5_prefix].keys():
                    alts = d2[h5_prefix][h5]["alts"]

                    y = d2[h5_prefix][h5]["y"]
                    x = d2[h5_prefix][h5]["x"]

                    colours = get_colours(x.shape[0])
                    for xi, yi, colour in zip(x, y, colours):
                        colour = [np.max((0.0, i-0.2)) for i in colour]
                        axes[0, -1].plot(xi, yi, color=colour)

                axes[0, -1].set_title("UVIS")
                axes[0, -1].grid()

            if error:
                print("Error found - maybe the occultation is a fullscan?")
                plt.close()
            if not error:
                # fig.add_subplot(111, frameon=False)
                # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
                fig.suptitle("%s: latitudes %0.1f - %0.1f" % (h5_prefix, np.min(lats_low), np.max(lats_low)))
                fig.supxlabel("Wavenumber (cm-1)")
                fig.supylabel("Transmittance")

                pdf.savefig()
                plt.close()
