# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 08:55:18 2020

@author: iant

READ M WOLFF RMS NOISE

pro nomad_trans_error,wave,trans,err,model=model,modversion=modversion,$
   cvers=cvers,errorfile=errfile

; model = 128, 256, otherwise 1024
; for given (wave,trans) spectrum, returns associated standard deviation of 
; of Gaussian distribution for given wavelength,Transmittance pair
;
;
; version 
cvers = '20200924'

if N_ELEMENTS(model) ne 1 then model=128
dpath = '/Users/mwolff/processing_local/nomad/transerror/'
sz = size(trans)

if sz[1] ne N_ELEMENTS(wave) then begin
   help,wave,trans
   message,'dimensions of wave and trans not compatible'
endif

if model eq 128 then begin
   errfile = 'transerr_err2_nt21_128.idlsav' & modversion='nt21a'
endif else if model eq 256 then begin
   errfile = 'transerr_err2_nt21_256.idlsav' & modversion='nt21a'
endif else begin
   errfile = 'transerr_err2_nt21_1024.idlsav' & modversion='nt21a'
endelse
message,'using error model '+errfile,/info

; structure is x, but rename to a
restore,dpath+errfile,/verb
a = x
x = 0

; create error array in a lazy way
err = trans*0.


; loop over wavelength
for i=0,N_ELEMENTS(wave)-1 do begin

;  for given wave[i], find the index within the boundary a.wave_pt
;  a.wave_bin represents the center of each bin
   isel = VALUE_LOCATE(a.wave_pt,wave[i])

; smooth input transmittance values to remove "spikes", which do not affect
; the distribution of error, i.e., 1-sigma values
; could probably be more smoothing, but this triangle filter works okay
   xin = smooth(smooth(reform(trans[i,*]),3),3)

; replace any values lt the smallest transmittance grid point, this is mainly
; negative transmittance at bottom of atmosphere but can be very small 
; transmittance values which produce a problem with interpolation
   idsmall = where(xin lt a.trans_bin[0])
   if idsmall[0] ne -1 then begin
;      print,'replaced input transmittance values of ',xin[idsmall]
      xin[idsmall] = a.trans_bin[0]
   endif

; okay, reduce to 1-D by using the isel-th wavelength bin of the grid
; independent axis is not transmittance
   y = reform(a.rms[*,isel])
   x = a.trans_bin
   idx = where(xin gt 1.)
   if idx[0] ne -1 then xin[idx] = 1.

; create sigma array through interpolation
   moderr = interpol(y,x,xin)

; smooth output errors...we are looking for sigma values that vary smoothly 
; with height (transmittance)
; again, a bit sketchy
   tst_in = smooth(smooth(moderr,3),3)
   err[i,*] = tst_in
   ide = where(err[i,*] lt 0.)

; there should not be any remaining way of producing a negative error
; (which was a problem for extrapolation)
   if ide[0] ne -1 then begin
      message,'negative error!  likely due to use of small or negative '+$
	'transmittance values.  If you see this message, fix code',/info
      stop
   endif
   ;cgplot,moderr,psym=2
   ;cgplot,tst_in,/over,linestyle=2
   ;stop
endfor


return
end

"""

import numpy as np
import os
import matplotlib.pyplot as plt



from nomad_ops.core.hdf5.l1p0a_to_1p0b.functions.uvis_rms_noise_functions import get_rms_dict, find_index, convolve_smooth
from nomad_ops.core.hdf5.l1p0a_to_1p0b.functions.prepare_uvis_rms_fig_tree import prepare_uvis_rms_fig_tree
from nomad_ops.core.hdf5.l1p0a_to_1p0b.config import WAVELENGTH_TO_PLOT, PLOT_FIGS



# from tools.file.hdf5_functions import get_file
# hdf5_filename, hdf5FileIn = get_file("20180426_022949_1p0a_UVIS_E", "hdf5_level_1p0a", 0)
# hdf5_basename = os.path.splitext(os.path.basename(hdf5_filename))[0]





def uvis_rms_noise(hdf5_filename, hdf5FileIn):

    y = hdf5FileIn["Science/YMean"][...]
    x = hdf5FileIn["Science/X"][0, :]
    
    rms_dict = get_rms_dict(hdf5FileIn)

    plot_px_index = find_index(WAVELENGTH_TO_PLOT, x)
    
    
    rms_noise_smooth_all = np.zeros_like(y)
    """loop through each pixel"""
    for px_index in range(len(x)): #[151]
    
        y_px = y[:, px_index]
        x_px = x[px_index]
        
        
        #remove v. small transmittance values => set to value of smallest bin
        small_value_indices = np.where(y_px < rms_dict["TRANS_BIN"][0])[0]
        # if len(small_value_indices)>0:
        #     print("Replaced input transmittance values of", y_px[small_value_indices])
        y_px[small_value_indices] = rms_dict["TRANS_BIN"][0]
        
        #remove transmittances>1 => set to 1.0
        large_value_indices = np.where(y_px > 1.0)[0]
        # if len(large_value_indices)>0:
            # print("Replaced input transmittance values of", y_px[large_value_indices])
        y_px[large_value_indices] = 1.0
        
        #smooth transmittances
        # y_px_smooth = savgol_filter(y_px, 5, 1)
        y_px_smooth = convolve_smooth(y_px, 1)
        # y_px_smooth2 = smoothTriangle(smoothTriangle(y_px, 1),1)
        
        wave_index = find_index(x_px, rms_dict["WAVE_PT"]) #isel
        
        #reduce to 1-D by using the isel-th wavelength bin of the grid
        rms_values = rms_dict["RMS"][wave_index, :]
        trans_bin = rms_dict["TRANS_BIN"]
        
        #create sigma array through interpolation
        rms_noise = np.interp(y_px_smooth, trans_bin, rms_values) #moderr
        
        #smooth rms_noise
        rms_noise_smooth = convolve_smooth(rms_noise, 2)
    
        #check for negative errors
        if np.any(rms_noise_smooth < 0.0):
            print("Error: negative noise found")
    
        rms_noise_smooth_all[:, px_index] = rms_noise_smooth
        
        if PLOT_FIGS and px_index == plot_px_index:

            alt = np.mean(hdf5FileIn["Geometry/Point0/TangentAltAreoid"][...], axis=1)

            y_error_px = hdf5FileIn["Science/YErrorMean"][:, px_index]

            #manually update level in filename
            hdf5_filename_new = hdf5_filename.replace("1p0a", "1p0b")
            
            
            fig, ax = plt.subplots(figsize=(9,9))
            plt.title(hdf5_filename_new)
            plt.xlabel("Transmittance", color="b")
            plt.ylabel("TangentAltAreoid (km)")
            ax.errorbar(y_px, alt, xerr=y_error_px, capsize=5, color="b", label="Original YMean & YMeanError at %0.1fnm" %x_px)
            # ax.plot(y_px-rms_noise, alt, "b:", label="YMean $\pm$ RMS noise at %0.1fnm" %x_px)
            # ax.plot(y_px+rms_noise, alt, "b:")
            
            ax.plot(y_px_smooth, alt, "k", label="Smoothed YMean")
            # plt.plot(y_px_smooth-rms_noise_smooth, alt, "k:", label="Smoothed RMS noise at %0.1fnm" %x_px)
            # plt.plot(y_px_smooth+rms_noise_smooth, alt, "k:")
            
            plt.fill_betweenx(alt, y_px-rms_noise_smooth, y_px+rms_noise_smooth, color="k", label="Original YMean $\pm$ Smoothed RMS noise at %0.1fnm" %x_px, alpha=0.2)
            
            # plt.plot(y_px_smooth2, alt, "r", label="Y smoothed2")
            ax.tick_params(axis='x', colors="b")
            
            ax2 = ax.twiny()
            ax2.plot(rms_noise, alt, "r:", label="RMS Noise (top x axis)")
            ax2.plot(rms_noise_smooth, alt, "r", label="Smoothed RMS Noise (top x axis)")
            ax2.set_xlabel("RMS Noise", color="r")
            ax2.tick_params(axis='x', colors="r")
            
            # ask matplotlib for the plotted objects and their labels
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2)
            
            plt.tight_layout()
            # plt.savefig(hdf5_basename+"_%inm.png" %x_px)



            thumbnail_path = prepare_uvis_rms_fig_tree("%s_rms_noise.png" %hdf5_filename_new)
            fig.savefig(thumbnail_path)
            
        #    if SYSTEM != "Windows":
            plt.close(fig)

    rms_noise_dict = {"rms_noise":rms_noise_smooth_all}
    return rms_noise_dict


# rms_noise_dict = uvis_rms_noise(hdf5FileIn, hdf5_basename)

# hdf5FileIn.close()
