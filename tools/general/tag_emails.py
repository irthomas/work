# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 08:59:09 2021

@author: iant
"""

import os
import imaplib
# import base64
import email
from tools.file.passwords import passwords

# from tools.general.send_email import send_bira_email

search_strs = ["invites you to", "has invited you to", r"https://teams.microsoft.com"]
log_filepath = os.path.join("reference_files", "email_log.txt")

EMAIL_FOLDER = "Mail/VenSpecH-Meetings"
N_EMAILS_TO_SEARCH = 100


email_user = "iant"
email_pass = passwords["hera"]
email_server = "mail-ae.oma.be"

M = imaplib.IMAP4_SSL("mail-ae.oma.be", 993)
M.login(email_user, email_pass)
# M.select("Inbox")
M.select(EMAIL_FOLDER)

# result, message_numbers = M.search(None, 'ALL')

result, message_numbers_str = M.sort("DATE", "UTF-8", "ALL")
message_numbers = [int(i) for i in message_numbers_str[0].split()]

#read log file
dates = []
senders = []
if os.path.exists(log_filepath):
    print("Log file found. Reading contents")
    with open(log_filepath, "r") as f:
        lines = f.readlines()
        
        for line in lines:
            date, sender = line.strip().split("\t")
            dates.append(date)
            senders.append(sender)
else:
    print("Log file not found. Making new file")
    #make empty file
    with open(log_filepath, "w") as f:
        pass
        

for num in message_numbers[::-1][0:N_EMAILS_TO_SEARCH]:
    result2, data = M.fetch(str(num).encode(), "(RFC822)")

    msg = email.message_from_bytes(data[0][1])
    date_in = msg["Date"].replace("\t", "").replace("\r", "").replace("\n", "")
    sender_in = msg["From"].replace("\t", "").replace("\r", "").replace("\n", "")
    subject_in = msg["Subject"].replace("\t", "").replace("\r", "").replace("\n", "")
    print("Checking email %i: %s" %(num, date_in))
    
    #search log for previous entries
    ixs = [i for i, (date, sender) in enumerate(zip(dates, senders)) if date == date_in and sender == sender_in]
    
    # for i, (date, sender) in enumerate(zip(dates, senders)):
    #     print(date, sender, date_in, sender_in)
    
    if len(ixs) == 0: #if not found
        print("Adding to log file and checking for invites")
    
        #write to log
        with open(log_filepath, "a") as f:
            f.write("%s\t%s\n" %(date_in, sender_in))

        
        messages = msg.get_payload()
        if type(messages) == list:
            messages = messages[0].get_payload()
            if type(messages) == list:
                messages = messages[0].get_payload()
                if type(messages) == list:
                    messages = messages[0].get_payload()
        
        message = messages.replace("\t", "").replace("\r", "").replace("\n", "")
        
        decode  = email.header.decode_header(message)
        found = False
        for search_str in search_strs:
            if message.find(search_str) > -1:
                found = True
            
            if found:
                print("Invitation found: tagging")
                # send_bira_email(subject_in, message, _to="ithomas84@gmail.com")
                M.store(str(num).encode(), '+FLAGS', '$label1')
    # else:
    #     print("Already found in log")

M.close()
M.logout()



# from imapclient import IMAPClient
# server = IMAPClient(email_server, use_uid=True)
# server.login(email_user, email_pass)

# select_info = server.select_folder('INBOX')
# print('%d messages in INBOX' % select_info[b'EXISTS'])

# messages = server.search(['FROM', 'best-friend@domain.com'])
# print("%d messages from our best friend" % len(messages))

# for msgid, data in server.fetch(messages, ['ENVELOPE']).items():
#     envelope = data[b'ENVELOPE']
#     print('ID #%d: "%s" received %s' % (msgid, envelope.subject.decode(), envelope.date))