# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:31:26 2018

@author: iant
"""

import numpy
import scipy.optimize
from plot_occultations_v02 import checkChiSquared02A
from hdf5_functions_v03 import makeFileList
import pylab as plt


fileLevel = "hdf5_level_0p2a"
obspaths = ["20180529_001626_0p2a_UVIS_U",
    "20180529_005659_0p2a_UVIS_U"]


hdf5Files, hdf5Filenames, titles = makeFileList(obspaths, fileLevel, silent=True)
data = checkChiSquared02A(hdf5Files, hdf5Filenames, titles, ["ObservationEphemerisTime"], alt_range=[100, 200], plot=False)


def fit_sin(yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    yy = numpy.array(yy)
    tt = numpy.arange(len(yy))
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
#    f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w*t + p) + c

    fitted = fitfunc(tt)
    plt.plot(tt, yy, "k-", label="y with noise")
    plt.plot(tt, fitted, "r-", label="y fit curve", linewidth=2)
    plt.legend(loc="best")
    plt.show()
    return fitted

a = fit_sin(data[1])
chiSq, pValue = stats.chisquare(data[1], a)
print(chiSq)

fit = polynomialFit(data[1], 2)
chiSq, pValue = stats.chisquare(data[1], fit)
print(chiSq)