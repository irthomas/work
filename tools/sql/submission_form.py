# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:38:50 2020

@author: iant

MAKE SUBMISSION FORM FOR WEBSITE
"""

from tools.sql.generic_database import database

from tools.sql.sql_table_fields import submission_form

db = database(bira_server=True)

db.new_table("Submission Form", submission_form)
