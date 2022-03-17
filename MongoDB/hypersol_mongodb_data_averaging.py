#!/usr/bin/env python
# coding: utf-8


# import relevant libraries
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from pymongo import MongoClient
import json
from collections import OrderedDict
from datetime import datetime, timedelta


# initialize mongo connector object with ip adress
client = MongoClient('xxx')
# get reference to existing database testDB
db = client.testDB
# authentication within database
db.authenticate('xxx', 'xxx', source='xxx')

# path to mass spectrum calibration data
ms_calibration_data_file = r'W:\Projekte\NRW_HyperSol_61904\Bearbeitung'\
    '\Massenspektroskopie\Spektren\mass_spectrums.json'
# read amu species correspondence data
with open(ms_calibration_data_file, 'r') as fp:
    ms_calibration_data_json = json.load(fp)
amu_species_dict = OrderedDict(zip(ms_calibration_data_json['amu'], 
                                   ms_calibration_data_json['species']))


# # reference collection, if not existent it will be created
# current_collection = db['HyperSol_61904']
# # insert loaded entry into database collection
# start_date = datetime.strptime('07/13/21', '%m/%d/%y')
# end_date = datetime.strptime('09/14/21', '%m/%d/%y')
# db_entries = current_collection.find({'DateTime': {'$gte': start_date, 
#                                                    '$lte': end_date}})
# db_entries = list(db_entries)

# # get database entry general information
# general_info_keys =  \
#     ['Catalyst Sample', 'Experiment Type', 'Gas Molar Composition (-)', 
#      'Gas Flow Rate (sccm)', 'Liquid Molar Composition (-)', 
#      'Liquid Flow Rate (g/h)', 'Radiation Source', 'Temperature (C)']
    
# reference collection, if not existent it will be created
current_collection = db['HyperSol_61904']
# make regular expression for substring of sample name
sample_name = 'SmartGas'
sample_name_re = re.compile(sample_name, re.I)
# find entry in database collection
db_entries = current_collection.find({'Catalyst Sample': 
                                      {'$regex': sample_name_re}})
db_entries = list(db_entries)


# filter species dict by list of names or list of amus
def filter_species(name_list=None, amu_list=None):
    filtered_amu_species_dict = amu_species_dict
    if name_list is not None:
        filtered_amu_species_dict = \
            OrderedDict([(k, list(set(name_list).intersection(v))) 
                         for k, v in amu_species_dict.items() 
                         if len(list(set(name_list).intersection(v))) > 0])
    if amu_list is not None:
        filtered_amu_species_dict = \
            OrderedDict([(k, v) for k, v in filtered_amu_species_dict.items() 
                         if k in amu_list])        
    return filtered_amu_species_dict

# species_list = ['H2', 'CH4', 'H2O', 'O2', 'N2', 'CO2', 'Ar', 'H2O']
# amu_list = [1, 2, 15, 16, 28, 40, 44]


filtered_amu_species_dict = filter_species() #, amu_list=amu_list)
filtered_amu_list = np.array(list(filtered_amu_species_dict.keys()))

# loop through all entries corresponding to filter
for db_entry in db_entries:

    # get data array from database entry
    data_dict = db_entry['Data']
    y_keys = [k for k in data_dict.keys() 
              if k[0] in (str(i) for i in range(10))]
    y_values = np.asarray([data_dict[key] for key in y_keys])
    
    # get x-values and make size the array equal to y-values array
    x_key = 'Time Relative (sec)'
    x_values = np.asarray(data_dict[x_key])
    # x_values = np.asarray([x_values for i in range(len(y_values))])
    bins = [int(y_key.split('_')[0]) for y_key in y_keys]

    y_labels = [str(k) + ' - ' + '/'.join(v) for k, v 
                in amu_species_dict.items() if k in bins]

    filtered_amu_ids = [bins.index(k) for k in bins 
                        if k in list(filtered_amu_list)]
    filtered_amu_list = [bins[k] for k in filtered_amu_ids]
    filtered_amu_species_dict = filter_species(amu_list=filtered_amu_list)

    # filtered_x_values = x_values[filtered_amu_ids]
    filtered_y_values = y_values[filtered_amu_ids]
    filtered_y_labels = [str(k) + ' - ' + '/'.join(v) 
                         for k, v in filtered_amu_species_dict.items()]
    filtered_y_keys = list(np.asarray(y_keys)[filtered_amu_ids])

    # plot differences between means of two different time intervals
    t_max = np.max(x_values)
    time_interval = 200.0
    t_min = t_max - time_interval
    interval = [t_min, t_max]
    df = pd.DataFrame.from_dict(data_dict)
    df = df[df['Time Relative (sec)'].between(interval[0], interval[1])]
    df_mean = df.mean()
    filtered_amu_list_str = [str(i) for i in filtered_amu_list]
    columns_dict = dict(zip(filtered_y_keys, filtered_amu_list_str))
    df_mean = df_mean.rename(index=columns_dict)
    # ax = df_mean[filtered_amu_list_str].transpose().plot(kind='bar', figsize=(10, 8))
    # ax.set_xlim((-1.0e-9, 1.0e-9))
    
    # replace data in database dict
    db_entry['Data'] = df_mean[filtered_amu_list_str].to_dict()
    db_entry['Averaging Interval (s)'] = interval
    # remove time step data
    
    # insert into database collection
    # reference collection, if not existent it will be created
    current_collection = db['HyperSol_61904_Average_SmartGas']
    # insert loaded entry into database collection
    # current_collection.insert_one(db_entry)

