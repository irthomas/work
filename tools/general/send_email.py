# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:51:35 2020

@author: iant

SEND EMAIL VIA AERONOMIE
"""

import smtplib
from tools.file.passwords import passwords


def send_email(subject, body):

    # write email
    _from = "ithomas84@gmail.com"
    _to  = "ithomas84@gmail.com"
    
    
    message = """Subject: %s\n%s""" % (subject, body)
    
    
    print("From:", _from)
    print("To:", _to)
    print(message)
    
    
    
    # set up email server
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(_from, passwords["gmail"])
        
        # send mail
        server.sendmail(_from, _to, message)
        server.quit()
        print("Email sent")
    except:
        print("Error sending email")
    


#send_email("this is the subject", "this is the body")