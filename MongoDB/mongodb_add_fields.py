# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 23:42:26 2022

@author: zbt
"""

#!/usr/bin/env python
# coding: utf-8

# %%

# import relevant libraries
import re
from pymongo import MongoClient
import mongodb_credentials as mc

# %%


# initialize mongo connector object with ip adress
client = MongoClient(mc.HOST_NAME)
# get reference to existing database testDB
db = client.HyperSol
# authentication within database
db.authenticate(mc.USER_NAME, mc.PASSWORD, source='xxx')
# reference collection, if not existent it will be created
current_collection = db['Averaged']

# %%

find_str = 'UVC'
update_field = 'Lamp'

find_str_re = re.compile(find_str, re.I)

current_collection.update_many({'Name': {'$regex': find_str_re}}, 
                               {'$set': {update_field: '25W-UVC'}})



# %%





