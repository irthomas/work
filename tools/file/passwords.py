# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 08:07:58 2020

@author: iant

GET PASSWORDS FROM EXTERNAL FILE, SAVE TO DICT

PASSWORD FILE SHOULD BE A TEXT FILE OF THE FORM:

{
"name1":"password1",
"name2":"password2",
"name3":"password3",
}

WITHOUT ANY HEADERS OR COMMENTS
"""


import os

from tools.file.paths import paths

with open(os.path.join(paths["REFERENCE_DIRECTORY"], "passwords.txt"), "r") as f:
    lines = "".join(f.readlines())
    
    passwords = eval(lines)
    
