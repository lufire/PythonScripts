# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 20:47:02 2020

@author: lukas
"""

import json
import os

test_dict = {'name': 'john', 'float': 1.34, 'int': 1, 'list': [1.3, 5.0, 6.4],
             'boollist': [True, True, False], 'dict': {'subnumber': 1.0e-5, 
                                                       'sublist': ['word',
                                                                   1.0,
                                                                   False]}}

out_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(out_dir, 'test.json'), 'w') as file:
    json.dump(test_dict, file, indent=2)