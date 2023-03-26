# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 08:47:12 2022

@author: iant

https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
"""


def progress_bar(iterable, prefix = '', suffix = '', decimals = 1, length = 50, fill = '*', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        
        use as follows:
            for item in progress_bar(list):
                action
                note that all print statements must be removed
        
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()
    
    
def progress(iterable):
    """generic progress bar e.g. for i in progress(iterable):"""
    
    length=50
    total = len(iterable)

    def printProgressBar (iteration):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = "*" * filledLength + '-' * (length - filledLength)
        print(f'\r |{bar}| {percent}% ', end = "\r")

    printProgressBar(0)

    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    print()