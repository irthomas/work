# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:40:51 2020

@author: iant
"""
from concurrent.futures import ProcessPoolExecutor
from time import sleep
 
def return_after_5_secs(message):
    sleep(1)
    return message
 
if __name__ == '__main__':
    pool = ProcessPoolExecutor(3)
     
    future = pool.submit(return_after_5_secs, ("hello"))
    print(future.done())
    sleep(5)
    print(future.done())
    print("Result: " + future.result())