# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:41:08 2022

@author: iant
"""


obs_file_dicts = {
    "files":{
        "id":["INTEGER PRIMARY KEY AUTOINCREMENT"],
        "orbit":["INTEGER NOT NULL", []],
        "filename":["TEXT NOT NULL", []],
        "utc_start_time":["TIMESTAMP NOT NULL", []],
        "duration":["FLOAT NOT NULL", []],
        "n_spectra":["INTEGER NOT NULL", []],
        "n_orders":["INTEGER NOT NULL", []],
        "bg_subtraction":["INTEGER NOT NULL", []],
    },


    "lno_d":{
        "id":["INTEGER PRIMARY KEY AUTOINCREMENT"],
        "frame_id":["INTEGER NOT NULL", []],
        "temperature":["FLOAT NOT NULL", []],
        "diffraction_order":["INTEGER NOT NULL", []], #order is a protected keyword!
        "longitude":["FLOAT NOT NULL", []],
        "latitude":["FLOAT NOT NULL", []],
        "incidence_angle":["FLOAT NOT NULL", []],
        "local_time":["FLOAT NOT NULL", []],
        "file_id":["INTEGER NOT NULL", []],
    },

    "so_ie":{
    "id":["INTEGER PRIMARY KEY AUTOINCREMENT"],
    "frame_id":["INTEGER NOT NULL", []],
    "temperature":["FLOAT NOT NULL", []],
    "diffraction_order":["INTEGER NOT NULL", []],
    "bin_index":["INTEGER NOT NULL", []],
    "altitude":["FLOAT NOT NULL", []],
    "longitude":["FLOAT NOT NULL", []],
    "latitude":["FLOAT NOT NULL", []],
    "local_time":["FLOAT NOT NULL", []],
    "file_id":["INTEGER NOT NULL", []],
    }
}
