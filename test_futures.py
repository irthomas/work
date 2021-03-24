# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:40:51 2020

@author: iant
"""
import concurrent.futures
import time
import queue
import functools

import numpy as np


n_proc = 3

sources = ["abc", "def", "ghi", "jkl", "mno"]

# class process(object):
   
    
#     def wait(self, source):
#         rand = np.random.rand()
#         time.sleep(rand * 3)
#         print(rand, source)
        
        

#     def __call__(self, executor, sources):

#         print("Starting process from %d source files" %(len(sources)))
#         ft_queue = queue.Queue()
#         cb = functools.partial(self.__callback, ft_queue)
        
#         #all sources are passed to generic process. Now loop through each source, running the converter
#         for src in sources:
#             ft = executor.submit(self.wait, src)
#             ft.add_done_callback(cb)

#         # self._products = []
#         # self._processed_srcs = []
#         for _ in sources:
#             ft = ft_queue.get()
#             # try: #TypeError usually means nothing is passed back to here. Always return [] if nothing generated
#             #     self.store(ft.result())
#             # except Exception:
#             #     logger.exception("Error in %s process", self.process_name)
#         # return self.return_products()
#         return sources

#     @staticmethod
#     def __callback(ft_queue, ft):
#         if ft.done():
#             ft_queue.put(ft)
    

 
# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor(n_proc) as executor:
        
#         # Call process - run one level process e.g. Hdf5L01dProcess on all sources
#         new_products, prssd_sources = process(executor, sources)



"""map stops on crash"""

# def wait(_in):
#     source, i = _in
#     rand = np.random.rand()
#     time.sleep(i)
#     if i == 3:
#         cheese == 2
#     return i, source
        
        
# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor(n_proc) as executor:
#         for i, result in executor.map(wait, zip(sources, range(len(sources)))):
#             print(i, result)
        

"""submit doesn't stop on crash, but all futures need to be checked afterwards. Also print statements go to anaconda prompt"""
# def wait(i, source):
#     # i, source = _in
#     # print(i)
#     rand = np.random.rand()
#     time.sleep(i)
#     if i == 3:
#         cheese == 2
#     return i, source
        
# futures_list = []
# results = []
        
# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor(n_proc) as executor:
#         for i, source in enumerate(sources):
#             future = executor.submit(wait, i, source)
#             futures_list.append(future)

#         for future in futures_list:
#             try:
#                 result = future.result()
#                 results.append(result)
#             except Exception:
#                 results.append(None)


"""functools.partial test"""
# def summed(x, y, z):
#     return x*100 + y*10 + z

        
# a = summed(1,2,3)
# print(a)

# summed_p = functools.partial(summed, 0.1, 0.1) #remove x and y and replace with 0.1s
# b = summed_p(3) #x and y are already defined, set z=3
# print(b)


"""functools.wraps test"""

# def logged(func):
#     def with_logging(*args, **kwargs):
#         print(func.__name__ + " was called")
#         return func(*args, **kwargs)
#     return with_logging

# @logged
# def f(x):
#    """does some math"""
#    return x + x * x

# def g(x):
#     """does some math"""
#     return x + x * x
# g = logged(g)




"""classmethod test"""

class Student(object):
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    @classmethod
    def from_string(cls, name_str):
        first_name, last_name = map(str, name_str.split(' '))
        student = cls(first_name, last_name)  #run the class from inside the class
        return student

scott = Student.from_string('Scott Robinson')




"""use submit and queue"""
# def wait(i, source):
#     # i, source = _in
#     # print(i)
#     # rand = np.random.rand()
#     time.sleep(i)
#     if i == 3:
#         cheese == 2
#     return i, source

# def __callback(ft_queue, ft):
#     if ft.done():
#         ft_queue.put(ft)



# futures_list = []
# results = []

# ft_queue = queue.Queue()
# #func cb will be called, with the future as its only argument, when the future is cancelled or finishes running.
# cb = functools.partial(__callback, ft_queue)


# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor(n_proc) as executor:
#         for i, source in enumerate(sources):
#             future = executor.submit(wait, i, source)
#             future.add_done_callback(cb)
#             futures_list.append(future)

#         for _ in sources:
#             ft = ft_queue.get()
#             try:
#                 results.append(ft.result())
#             except Exception:
#                 results.append(None)




    