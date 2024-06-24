# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:31:38 2022

@author: iant
"""

import requests
import time
import json
from datetime import datetime, timezone
secret = "L4QLVQ1LOKCQX2193VSEICXW61NP6B1O"
status = ""


BASE_URL = 'https://monitoringapi.solaredge.com'

api_endpoint = '/sites/list'
full_api_url = BASE_URL + api_endpoint
parameters = {
    'status': status,
    'api_key': secret,
}


request = requests.get(full_api_url, params=parameters)
# request = requests.post(url, data=data, timeout=self.timeout)
request.raise_for_status()
response = request.json()
