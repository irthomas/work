# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:20:14 2020

@author: iant

make txt files containing solar and molecular absorption for each order

"""

import numpy as np
import os
import matplotlib.pyplot as plt


from tools.file.paths import paths
from tools.spectra.smooth_hr import smooth_hr
from tools.spectra.solar_spectrum_lno import nu_hr_grid
from tools.spectra.solar_spectrum import get_solar_hr
from tools.spectra.molecular_spectrum_lno import get_molecular_hr
from instrument.calibration.lno_radiance_factor.lno_rad_fac_orders import rad_fact_orders_dict

FIG_X = 10
FIG_Y = 5


SMOOTHING_LEVEL = 600 #for LNO. must be even number
#SMOOTHING_LEVEL_PFS = 300 #for LNO. must be even number


def buildOrderFiles():
    """builds text files containing solar or most prominent atmospheric molecule lines"""
#    bestOrderMolecules = {118:"CO2", 120:"CO2", 126:"CO2", 130:"CO2", 133:"H2O", 142:"CO2", 151:"CO2", 156:"CO2", 160:"CO2", 162:"CO2", \
#                          163:"CO2", 164:"CO2", 166:"CO2", 167:"H2O", 168:"H2O", 169:"H2O", 173:"H2O", 174:"H2O", 178:"H2O", 179:"H2O", \
#                          180:"H2O", 182:"H2O", 184:"CO", 189:"CO", 194:"CO", 195:"CO", 196:"CO"}
#    for order in range(115, 196):
#        if order not in bestOrderMolecules.keys():
#            bestOrderMolecules[order] = ""


    for diffractionOrder, rad_fact_order_dict in rad_fact_orders_dict.items():

        if "solar_molecular" in rad_fact_order_dict.keys(): #"solar", "molecular" or no entry
            solar_molecular = rad_fact_order_dict["solar_molecular"]
        else:
            solar_molecular == ""
            
        if "molecule" in rad_fact_order_dict.keys(): #molecule name or no entry
            molecule = rad_fact_order_dict["molecule"]
        else:
            molecule = ""
            
        #print the chosen detection method in solid line, otherwise dashed line.
        linestyles = {
                "solar":{"solar":"-", "molecular":"--", "":"--"}[solar_molecular],
                "molecular":{"solar":"--", "molecular":"-", "":"--"}[solar_molecular],
                }
         
        print("diffractionOrder:", diffractionOrder, solar_molecular, molecule)


        
        fig1, (ax1a, ax1b) = plt.subplots(nrows=2, figsize=(FIG_X+6, FIG_Y+2), sharex=True)
        fig1.suptitle("Diffraction Order %i" %diffractionOrder)
        
        """get high res solar spectra and convolve to LNO-like resolution"""
        #get high res wavenumber grid for diffraction order
        nuHr, _ = nu_hr_grid(diffractionOrder, adj_orders=0, instrument_temperature=0.0)

        if diffractionOrder < 196:
            #get high res ace solspec solar spectrum
            solspecFilepath = os.path.join(paths["REFERENCE_DIRECTORY"], "nomad_solar_spectrum_solspec.txt")
            solarHr = get_solar_hr(nuHr, solspecFilepath, nu_limit=2.0)
            smoothing_level = SMOOTHING_LEVEL
        else:
            #get high res ace solspec solar spectrum
            solspecFilepath = os.path.join(paths["REFERENCE_DIRECTORY"], "pfsolspec_hr.dat")
            solarHr = get_solar_hr(nuHr, solspecFilepath, nu_limit=2.0)
            smoothing_level = SMOOTHING_LEVEL
        
        #convolve high res solar spectrum to lower resolution. Normalise to 1
        solarSpectrum = smooth_hr(solarHr, window_len=(smoothing_level-1))
#        normalisedSolarSpectrum = (solarSpectrum / (np.max(solarSpectrum)))[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)]
        normalisedSolarSpectrum = solarSpectrum[int(smoothing_level/2-1):-1*int(smoothing_level/2-1)] / np.max(solarSpectrum)

        #fudge to remove the strongest doublet from solar data in an important order
        if diffractionOrder == 193:
            normalisedSolarSpectrum[27720:27920] = normalisedSolarSpectrum[27720]
        



        ax1b.set_title("Solar Spectrum")
        ax1b.plot(nuHr, normalisedSolarSpectrum, "k", linestyle=linestyles["solar"])
        
        if molecule != "":
            
            #get high res molecular spectrum
            molecularSpectrumHr = get_molecular_hr(molecule, nuHr)
 
           #convolve high res molecular spectrum to lower resolution. Normalise to 1
            molecularSpectrum = smooth_hr(molecularSpectrumHr, window_len=(SMOOTHING_LEVEL-1))
#            normalisedMolecularSpectrum = 1.0 - (molecularSpectrum)[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)] * 1e18
            normalisedMolecularSpectrum = 1.0 - molecularSpectrum[int(SMOOTHING_LEVEL/2-1):-1*int(SMOOTHING_LEVEL/2-1)]

        else:
            normalisedMolecularSpectrum = np.ones_like(nuHr)

        ax1a.set_title("Molecular Spectrum %s" %molecule)
        ax1a.plot(nuHr, normalisedMolecularSpectrum, "k", linestyle=linestyles["molecular"])
    

        #format and write output to text file, one per order
        output = []
        for nu, solar, molecular in zip(nuHr, normalisedSolarSpectrum, normalisedMolecularSpectrum):
            output.append("%0.4f, %0.8f, %0.8f\n" %(nu, solar, molecular))
        
        with open(os.path.join(paths["BASE_DIRECTORY"], "order_%i.txt" %diffractionOrder), "w") as f:
            for line in output:
                f.write(line)
    
        plt.savefig(os.path.join(paths["BASE_DIRECTORY"], "order_%i.png" %diffractionOrder))
        plt.close()
    
    
buildOrderFiles()
