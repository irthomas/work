# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:44:02 2022

@author: iant

PLOT OUTPUT FROM DICTIONARY
"""

import numpy as np
# import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_asimut_dict_pdf(h5, d, pdf_filepath):
    """make a pdf from the asimut output data dictionary"""

    with PdfPages("%s" %pdf_filepath) as pdf: #open pdf
    
        n = len(d["indices"])
    
        #loop through spectra
        for i in range(n):
            
            if np.mod(i, 10) == 0:
                print("%i/%i" %(i, n))
    
            fig = plt.figure(figsize=(8, 8), constrained_layout=True)
            gs = fig.add_gridspec(2, 1)
            ax1a = fig.add_subplot(gs[0, 0])
            ax1b = fig.add_subplot(gs[1, 0], sharex=ax1a)
            
            i_l1 = d["indices"][i] #get index in l1.0a file
            
            
            #plot figure            
            fig.suptitle("%s: i=%i, %0.1fkm" %(h5, i_l1, d["alts"][i]))
        
            ax_grp = [ax1a, ax1b]
            ax_grp[0].plot(d["x_in"], d["y"][i, :], label="Y")
            ax_grp[0].plot(d["x_in"], d["yr"][i, :], label="Y retrieved")
            
            residual = d["y"][i, :] - d["yr"][i, :] #get fitting residual
            if len(residual) == 320: #if all pixels
                chi_squared = np.sum(((d["yr"][i, 50:300] - d["y"][i, 50:300])**2) / d["y"][i, 50:300])
            
            ax_grp[1].plot(d["x_in"], residual, color="k", label="Residual")
            ax_grp[1].fill_between(d["x_in"], d["y_error"][i, :], -d["y_error"][i, :], color="g", label=d["YError"], alpha=0.3)
            
            ax_grp[0].set_title("AOTF = %0.3f" %d["A_nu0"])
            ax_grp[0].set_ylabel("Transmittance")
            ax_grp[1].set_ylabel("Residual")
            
            ax_grp[0].set_ylim((min(d["y"][i, 20:])-0.01, max(d["y"][i, 20:])+0.01))
            ax_grp[1].set_ylim((-0.015, 0.015))
            if len(residual) == 320: #if all pixels
                ax_grp[1].text(0.5, 0.05, "Chi-squared (pixels 50-300): %0.2e" %chi_squared, transform=ax_grp[1].transAxes, bbox=dict(ec='None', fc='white', alpha=0.8))
            
            ax_grp[0].legend(loc="upper right")
            ax_grp[1].legend(loc="upper right")
            
            ax_grp[0].minorticks_on()
            ax_grp[0].grid(b=True, which='major', linestyle='-')
            ax_grp[0].grid(b=True, which='minor', linestyle='--', alpha=0.5)

            ax_grp[1].minorticks_on()
            ax_grp[1].grid(b=True, which='major', linestyle='-')
            ax_grp[1].grid(b=True, which='minor', linestyle='--', alpha=0.5)
            
            pdf.savefig()
            plt.close()

