# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:52:06 2022

@author: iant


"""

from functools import partial
from datetime import datetime
from time import sleep
from packaging.version import Version

import requests

import pyvo

# service = pyvo.dal.TAPService("http://vespa-ae.oma.be/tap")

# query = "SELECT granule_uid FROM nomad.epn_core"

# results = service.search(query)


url = 'http://vespa-ae.oma.be/tap/sync'

data = {'REQUEST': 'doQuery', 'LANG': 'ADQL', 'QUERY': 'SELECT granule_uid FROM nomad.epn_core'}

stream=True

files = {}

response = requests.post(url, data=data, stream=True, files=files)
# requests doesn't decode the content by default
a = response.raw.read = partial(response.raw.read, decode_content=True)