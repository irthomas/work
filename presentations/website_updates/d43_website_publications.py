# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:56:59 2023

@author: iant


WRITE HTML CODE TO ADD TO BIRA PLANETARY WEBSITE PUBLICATION LIST

ADAPTED FROM PAGE: https://wiki.aeronomie.be/index.php/UV-Vis:Extract_data_from_the_ORFEO_publication_database

COPY CONTENTS OF FILES ONTO EACH YEAR'S PAGE WHEN IN THE CODE EDITOR VIEW


"""

import requests
import json
 

#list of years to get publications for
YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

#search for these authors
AUTHOR_NAMES = ["Vandaele", "Mahieux", "Daerden", "Vanhellemont"] 


def make_publication_dict(metadata):
    '''
    Extract relevant metadata from a publication item and store in a dictionary.
    '''
    
    pub_dict = {
        'authors':'',
        'title':'',
        'journal':'',
        'volume':'',
        'issue':'',
        'pages':'',
        'doi':'',
        'year':''
    }
    
    # Combine all authors in one string.        
    authors = ''
    for dict_item in metadata:
        if dict_item['key'] == 'dc.contributor.author':
            authors += dict_item['value']+', '
        elif dict_item['key'] == 'dc.title':
            pub_dict['title'] = dict_item['value']
        elif dict_item['key'] == 'dc.source.title':
            pub_dict['journal'] = dict_item['value']
        elif dict_item['key'] == 'dc.source.volume':
            pub_dict['volume'] = dict_item['value']
        elif dict_item['key'] == 'dc.source.issue':
            pub_dict['issue'] = dict_item['value']
        elif dict_item['key'] == 'dc.source.page':
            pub_dict['pages'] = dict_item['value']
        elif dict_item['key'] == 'dc.identifier.doi':
            pub_dict['doi'] = dict_item['value']
        elif dict_item['key'] == 'dc.date':
            pub_dict['year'] = dict_item['value']
        else:
            pass
    
    # Remove trailing ', ' from author list.
    authors = authors[:-2]
    pub_dict['authors'] = authors
    
    return pub_dict
    

def search_bira_pubs(req_data):
    # Loop over all found publication items.
    # Throw away those that are not by BIRA-IASB.
    bira_pubs = [] # This list will contain all found BIRA-IASB publications.
    
    for data in req_data:
        
        #Check if BIRA-IASB publication:
        if data['parentCollection']['name'] == 'BIRA-IASB publications':

            metadata = data['metadata']
            pub_dict = make_publication_dict(metadata)
            bira_pubs.append(pub_dict)
            
        else:
            #Skip if not from BIRA-IASB
            continue

    return bira_pubs



def get_bira_pubs_by_year(year):
    # The ORFEO publication platform rest interface overview.
    URLbase = 'https://orfeo.belnet.be/rest/'
    
    
    # We will search the database by the metadata field 'dc.date' to find all publications of the current year.
    # For further filtering, we also need the metadata for each publication item and information on the collection they belong to.
    URL = URLbase+'items/find-by-metadata-field?expand=parentCollection,metadata'
    
    # We will send the search instructions in json format
    header = {"content-type": "application/json"}
    req_dict = {
        "key": "dc.date",
        "value": str(year),
        "language":None
    }
    
    # Do the request
    req = requests.post(url = URL,data=json.dumps(req_dict), headers=header)
    req_data = req.json()
    # ndata = len(req_data)
    # nfound=0
    # print('Found total number of publications: ',ndata)
    
    bira_pubs = search_bira_pubs(req_data)

    return bira_pubs





html_lists = {}
for year in YEARS:
    print("Getting publications for %i" %year)

    #get BIRA publications
    bira_pubs = get_bira_pubs_by_year(year)
    
    #sort by author name
    bira_pubs = sorted(bira_pubs, key = lambda item: item['authors'])
    
    found_items = []
    
    #make html list
    h = "<h2>Publications in %i</h2><br>\n" %year
    h += "<ul>\n"
    for pub_dict in bira_pubs:
        for author_name in AUTHOR_NAMES:
            if author_name in pub_dict["authors"]:
                
                #check if already added
                if pub_dict in found_items:
                    continue
                
                #else add it to the list
                else:
                    h += "<li>\n"
                    h += "<p><b>%s</b></p>\n" % pub_dict['title']
                    h += "<p>%s</p>\n" % pub_dict['authors']
                    h += "<p>%s, Vol. %s, issue %s, %s (%i), DOI: %s</p>\n" %(pub_dict['journal'], pub_dict['volume'], pub_dict['issue'], pub_dict['pages'], year, pub_dict['doi'])
                    h += "</li><br>\n"
                    
                    found_items.append(pub_dict)
                    
    h += "</ul>\n"
            
    html_lists[year] = h

print("Writing html output to files")
for year, html_list in html_lists.items():
    with open("publications_%s.txt" %year, "w", encoding="utf-8") as f:
        f.write(html_list)