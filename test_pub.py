# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:15:26 2020

@author: iant
"""


import requests
#from fake_useragent import UserAgent
from bs4 import BeautifulSoup

#ua = UserAgent()

query = "exomars"
#number_result = 10

google_url = "https://scholar.google.com/scholar?q=" + query + "&scisbd=1&num=30"
response = requests.get(google_url)
soup = BeautifulSoup(response.text, "html.parser")

result_div = soup.find_all('div', attrs = {'class': 'gs_ri'})


links = []
titles = []
authors = []
for r in result_div:
    # Checks if each element is present, else, raise exception
    try:
        link_title = r.find('a', href = True)
        
        link = link_title["href"]
        title = link_title.text
        author_info = r.find('div', attrs={'class':'gs_a'}).get_text()
#        title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
#        description = r.find('div', attrs={'class':'s3v9rd'}).get_text()
        
        # Check to make sure everything is present before appending
        if link != '' and title != '' and author_info != '': 
            links.append(link)
            titles.append(title)
            authors.append(author_info)
    # Next loop if one element is not present
    except:
        continue
    
h = ""
h += "<html><head></head><body>\n"
h += "<table border=1><tr><th>Title</th><th>Authors</th><th>Link</th></tr>\n"
for link, title, author in zip(links, titles, authors):
    h += "<tr><td>%s</td><td>%s</td><td><a href=%s>%s</a></td>\n" %(title, author, link, link)
    
with open("papers.html", "w", encoding="utf-8") as f:
    f.writelines(str(h))
