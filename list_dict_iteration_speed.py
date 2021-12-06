# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 21:31:09 2019

@author: lukas
"""
import time


n = 10000

conc_dict = {}
conc_list = []
for i in range(n):
    conc_dict[str(i)] = i
    conc_list.append(i)

start = time.time()
for i in range(n):
    for item in conc_dict:
        pass
end = time.time()
print('dict time: ', end - start)

start = time.time()
for i in range(n):
    for item in conc_list:
        pass
end = time.time()
print('list time: ', end - start)