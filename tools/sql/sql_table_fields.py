# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:35:53 2020

@author: iant
"""




def sql_table_fields(server_db=False):

    if server_db:
        table_fields = [
                {"name":"row_id", "type":"int NOT NULL AUTO_INCREMENT", "primary":True}, \
                {"name":"orbit", "type":"int NOT NULL"}, \
                {"name":"filename", "type":"varchar(100) NULL DEFAULT NULL"}, \
                {"name":"frame_id", "type":"int NOT NULL"}, \
                {"name":"temperature", "type":"decimal NOT NULL"}, \
                {"name":"diffraction_order", "type":"int NOT NULL"}, \
                {"name":"sbsf", "type":"int NOT NULL"}, \
                {"name":"bin_index", "type":"int NOT NULL"}, \
                {"name":"utc_start_time", "type":"datetime NOT NULL"}, \
                {"name":"duration", "type":"decimal NOT NULL"}, \
                {"name":"n_spectra", "type":"int NOT NULL"}, \
                {"name":"n_orders", "type":"int NOT NULL"}, \
                {"name":"longitude", "type":"decimal NOT NULL"}, \
                {"name":"latitude", "type":"decimal NOT NULL"}, \
                {"name":"altitude", "type":"decimal NOT NULL"}, \
                {"name":"incidence_angle", "type":"decimal NOT NULL"}, \
                {"name":"local_time", "type":"decimal NOT NULL"}, \
                ]
    else:
        table_fields = [
                {"name":"row_id", "type":"integer primary key"}, \
                {"name":"orbit", "type":"integer"}, \
                {"name":"filename", "type":"text"}, \
                {"name":"frame_id", "type":"integer"}, \
                {"name":"temperature", "type":"real"}, \
                {"name":"diffraction_order", "type":"integer"}, \
                {"name":"sbsf", "type":"integer"}, \
                {"name":"bin_index", "type":"integer"}, \
                {"name":"utc_start_time", "type":"timestamp"}, \
                {"name":"duration", "type":"real"}, \
                {"name":"n_spectra", "type":"integer"}, \
                {"name":"n_orders", "type":"integer"}, \
                {"name":"longitude", "type":"real"}, \
                {"name":"latitude", "type":"real"}, \
                {"name":"altitude", "type":"real"}, \
                {"name":"incidence_angle", "type":"real"}, \
                {"name":"local_time", "type":"real"}, \
                ]

    return table_fields
    


