# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:44:49 2022

@author: iant

GET DATA FROM DB
"""
from datetime import datetime

from tools.sql.table_dicts import obs_file_dicts


def make_empty_dict(table_name):
    obs_dict = obs_file_dicts[table_name]
    
    return {key:[] for key, value in obs_dict.items()}
    


# def make_query_from_search(search_tuple):
#     """build obs database sql query from items in search dictionary"""

#     table_name, search_dict = search_tuple

#     search_query = "SELECT * from %s WHERE " %(table_name)



#     for key, values in search_dict.items():
#         if len(values) == 1:
#             #find exact match - integers only
#             search_query += "AND %s = %i " %(key, values[0])
#         else:
                
#             #find range - convert to floats
#             search_query += "AND %s > %0.6f AND %s < %0.6f " %(key, values[0], key, values[1])
            
#     #remove first AND
#     search_query = search_query.replace("WHERE AND", "WHERE")
    
#     return search_query

def make_query_from_search(search_tuple):
    """build obs database sql query from items in search dictionary"""

    table_name, search_dict = search_tuple

    search_query = "SELECT * from %s WHERE " %(table_name)
    search_variables = []


    for key, values in search_dict.items():
        if len(values) == 1:
            #find exact match - integers only
            search_query += "AND %s = ? " %(key)
            search_variables.append(values[0])
        else:
                
            #find range - convert to floats
            search_query += "AND %s > ? AND %s < ? " %(key, key)
            search_variables.append(values[0])
            search_variables.append(values[1])
            
    #remove first AND
    search_query = search_query.replace("WHERE AND", "WHERE")
    
    return search_query, search_variables



def get_match_data(db, search_tuple):
    """search obs database for lno nadirs matching the search parameters
    output data satisfying criteria"""

    #search channel database for parameters
    search_query, search_variables = make_query_from_search(search_tuple)
    query_output = db.query([search_query, search_variables]) #returns list of tuples

    table_name = search_tuple[0]
    #make empty observation dictionary
    output_dict = make_empty_dict(table_name)
    
    #get dictionary keys
    keys = list(output_dict.keys())[:]
    #get dictionary keys for file table
    keys2 = make_empty_dict("files").keys()

    
    #add query data to observation dictionary
    for output_line in query_output:
        for i, key in enumerate(keys):
            output_dict[key].append(output_line[i])
            
        
        #for each spectrum, get info from files table
        id_query = "SELECT * from files WHERE id = %i" %(output_line[-1])
        id_output = db.query(id_query)
        
        #add files table data to observation dictionary
        for j, key2 in enumerate(keys2):
            if key2 != "id": #skip id, already added from observation dictionary
                if key2 not in output_dict.keys():
                    output_dict[key2] = [] #make blank entry
                output_dict[key2].append(id_output[0][j])
    
    #return a dictionary containing all channel + file info for each matching spectrum
    return output_dict



def get_files_match_data(db, search_tuple):
    """search obs database for lno nadirs matching the search parameters
    output data satisfying criteria"""

    #search channel database for parameters
    search_query, search_variables = make_query_from_search(search_tuple)
    query_output = db.query([search_query, search_variables]) #returns list of tuples

    table_name = search_tuple[0]
    #make empty observation dictionary
    output_dict = make_empty_dict(table_name)
    
    #get dictionary keys
    keys = list(output_dict.keys())[:]
    
    #add query data to observation dictionary
    for output_line in query_output:
        for i, key in enumerate(keys):
            output_dict[key].append(output_line[i])
            
            
    #return a dictionary containing all channel + file info for each matching spectrum
    return output_dict




