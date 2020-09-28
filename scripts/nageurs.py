# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:26:38 2020

@author: iant
"""

import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from tools.general.send_email import send_email


go = True

while go: #repeat
    url = "https://docs.google.com/forms/d/e/1FAIpQLSfW7EognGmkSwh2D5nPWLo75ONoeWopWzFtUZjShdL0jBwF_w/viewform?fbclid=IwAR0mJTMll1oYThsFRt3gzBYwrAUO9c9FwtQuYIzhGriPtzjya0XUJvSEdDM"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    # download the google form
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    datetime_now = str(datetime.now())[:-7].replace(" ","_").replace(":","-")
    
    if str(soup).find("N/A") != -1: #if no times are available
        print(datetime_now, ": no times are available")
        
    else:
        print(datetime_now, ": times are available!")
        send_email("Times are available!", "Sign up")
        go = False
        
        # #save page
        # with open("%s_naguers.html" %datetime_now, "w") as f:
        #     f.writelines(str(soup))
    
    #wait 1 hour
    for i in range(600):
        time.sleep(10)

    