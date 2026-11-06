# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:29:22 2023

@author: iant

READ IN HDF5 FILE

get_gem_tpvmr(molecule, altitude, myear, ls, lat, lon, lst)
"""

import os
import h5py

from analysis.so_lno_2023.config import ROOT_DATA_DIR


def make_filepath(h5):

    year = h5[0:4]
    month = h5[4:6]
    day = h5[6:8]

    h5_filepath = os.path.join(ROOT_DATA_DIR, year, month, day, "%s.h5" % h5)
    return h5_filepath


def read_h5(h5, spec_ix=None, data_path=None):

    if not data_path:
        h5_filepath = make_filepath(h5)
    else:
        h5_filepath = os.path.join(data_path, h5)

    with h5py.File(h5_filepath, "r") as h5_f:

        if not spec_ix:

            y = h5_f["Science/Y"][...]
            y_raw = h5_f["Science/YUnmodified"][...]
            y_error = h5_f["Science/YError"][...]
            x = h5_f["Science/X"][0, :]
            order = h5_f["Channel/DiffractionOrder"][0]
            aotf_freq = h5_f["Channel/AOTFFrequency"][0]

            nomad_t = h5_f["Channel/MeasurementTemperature"][0][0]

            alts = h5_f["Geometry/Point0/TangentAltAreoid"][:, 0]
            lats = h5_f["Geometry/Point0/Lat"][:, 0]
            lons = h5_f["Geometry/Point0/Lon"][:, 0]

            myear = 36  # TODO: calculate this correctly
            ls = h5_f["Geometry/LSubS"][0, 0]  # first value
            lst = h5_f["Geometry/Point0/LST"][:, 0]

        else:

            y = h5_f["Science/Y"][spec_ix, :]
            y_raw = h5_f["Science/YUnmodified"][spec_ix, :]
            y_error = h5_f["Science/YError"][spec_ix, :]
            x = h5_f["Science/X"][spec_ix, :]
            order = h5_f["Channel/DiffractionOrder"][0]
            aotf_freq = h5_f["Channel/AOTFFrequency"][0]

            nomad_t = h5_f["Channel/MeasurementTemperature"][0][0]

            alts = h5_f["Geometry/Point0/TangentAltAreoid"][spec_ix, 0]
            lats = h5_f["Geometry/Point0/Lat"][spec_ix, 0]
            lons = h5_f["Geometry/Point0/Lon"][spec_ix, 0]

            myear = 36  # TODO: calculate this correctly
            ls = h5_f["Geometry/LSubS"][0, 0]  # first value
            lst = h5_f["Geometry/Point0/LST"][spec_ix, 0]

    return {"y": y, "y_raw": y_raw, "y_error": y_error, "x": x, "order": order, "aotf_freq": aotf_freq, "nomad_t": nomad_t,
            "alt": alts, "lat": lats, "lon": lons, "myear": myear, "ls": ls, "lst": lst}
