#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import relevant libraries
import os
import re
import pandas as pd
from pymongo import MongoClient
from datetime import datetime


# In[2]:


# specificy directory with files to be loaded into database
base_dir = \
    r'W:\Projekte\NRW_HyperSol_61904\Bearbeitung\Massenspektroskopie\Messdaten'
contain_strings = ['HyperSol']
exclude_strings = ['SmartGas', 'Smartgas']


def get_file_paths(directory, file_extension, contains=[], excludes=[]):
    for dirpath, _, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1][1:] == file_extension:
                if all(item in name for item in contains):
                    if all(item not in name for item in excludes):
                        yield os.path.abspath(os.path.join(dirpath, name))


file_paths = list(get_file_paths(base_dir, 'dat', contains=contain_strings, 
                                 excludes=exclude_strings))
# file_paths = [path for path in file_paths if exclude_string not in path]
print(file_paths)


# In[3]:


# initialize mongo connector object with ip adress
client = MongoClient('xxx')
# get reference to existing database testDB
db = client.HyperSol
# authentication within database
db.authenticate('xxx', 'xxx', source='admin')
# reference collection, if not existent it will be created
current_collection = db['Raw']


# In[8]:

# # update info in existing entries
# current_collection.update_many({'Name': {'$regex': '25W-UVC'}}, 
#                                {'$set': {'Lamp': '25W-UVC'}})


# In[9]:


# check if data already exists in collection
file_paths_new = []
for path in file_paths:
    file_name = os.path.splitext(os.path.basename(path))[0]
    # print(file_name)
    if not current_collection.count_documents({'Name': file_name}, limit = 1):
        file_paths_new.append(path)
for file in file_paths_new:
    print(os.path.basename(file))
    

# In[10]:
db_entry_dicts = []
for data_path in file_paths_new:
    
    # specify data file directory
    # data_dir = r'W:\Projekte\NRW_HyperSol_61904\Bearbeitung\Massenspektroskopie\Messdaten\001-04-2022'
    # data_file_name = 'LFD_Basic_AllBins_Recipe_HyperSol_120C_1sccmN2_0p6gph-H2O_150W-HMI_Sample-21J4.dat'
    # data_path = os.path.join(data_dir, data_file_name)
    data_file_name = os.path.basename(data_path)
    data_name = os.path.splitext(data_file_name)[0]
    # read annotations from run file
    run_path = os.path.splitext(data_path)[0] + '.isi'
    # read data file as pandas dataframe
    data_df = pd.read_csv(data_path, skiprows=372, sep='\t')
    data_dict = data_df.to_dict('list')
    
    
    # In[337]:
    
    
    # read general information
    with open(data_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if 'General Info' in line:
            general_info_line_count = i
            break
    general_info = \
        lines[general_info_line_count + 1: general_info_line_count + 7]      
    general_info_dict = \
        {line.split('\t')[0].strip(): line.split('\t')[1].strip() 
         for line in general_info}
    sample_name = data_name.split('Sample-')[-1]
    org_path = general_info_dict['Run Path and Name']
    org_dir_name = os.path.dirname(run_path)
    org_path = os.path.join(org_dir_name, data_name + '.isi')
    date_object = datetime.strptime(general_info_dict['Start Time'], 
                                    "%m/%d/%y %H:%M:%S")
    general_info_dict['DateTime'] = date_object
    general_info_dict['Run Path and Name'] = org_path
    
    
    # In[338]:
    
    
    # get start time in seconds
    time_string  = general_info_dict['Start Time'].split(' ')[-1]
    h, m, s = time_string.split(':')
    start_seconds = int(h) * 3600 + int(m) * 60 + int(s)
    
    # read annotations from run file
    with open(run_path, 'r') as file:
        lines = file.readlines()
        annotation_lines = [line for line in lines if 'By:' in line]
        
    annotations = []
    for line in annotation_lines:
        # annotation_text = line.split('By:')[0].replace('\x03', ' ').replace('\x04', ' ').replace('\x02', ' ')
        # annotation_date = line.split('on')[-1]
        annotation_list = re.split('\x03|\x04|\x02|\n', line)[:-1]
        annotation_list = [i for i in annotation_list if i != '']
        text = ' '.join(annotation_list[:annotation_list.index('By:')])
        date = annotation_list[-2]
        time = annotation_list[-1]
        h, m, s = time.split(':')
        anno_absolute_seconds = int(h) * 3600 + int(m) * 60 + int(s)
        rel_seconds = anno_absolute_seconds - start_seconds
        annotations.append({'Text': text, 'Date': date, 
                            'Time': time, 'Relative Time (s)': rel_seconds})
    if annotations:
        annotations[0]['Text'] = annotations[0]['Text'].replace('0,6', '0.6')
        annotations
    
    
    # In[339]:
    
    
    # filter keys for even numbers (atom masses)
    data_dict_filtered = {k: v for k, v in data_dict.items() if '.' not in k}
    data_dict_filtered.update({k[:-3]: v for k, v in data_dict.items() if (k[-3:] == '.00')})
    
    
    # In[340]:
    
    
    def get_between(base_string, split_str, sep_str='_', after=False):
            base_string = os.path.splitext(base_string)[0]
            if after:
                first_split = base_string.split(split_str)
                return [item.split(sep_str)[0] for item in first_split[1:]]
            else:
                first_split = base_string.split(split_str)
                return [item.split(sep_str)[-1] for item in first_split[:-1]]
    
    gas_flow_values = [float(item.replace('p', '.')) for item 
                       in get_between(data_name, 'sccm')]
    
    gas_flow_names = get_between(data_name, 'sccm', after=True)
    vol_flows = dict(zip(gas_flow_names, gas_flow_values))
    try:
        liquid_flow = float(get_between(data_name, 'gph')[0].replace('p', '.'))
    except ValueError:
        liquid_flow = get_between(data_name, 'gph')[0].replace('p', '.')
    temperature = float(get_between(data_name, 'C_')[0].replace('p', '.'))
    lamp = get_between(data_name, 'W-')[0] + 'W-' \
        + get_between(data_name, 'W-', after=True)[0]
    
    # In[341]:
    
    
    # create database entry
    vol_flow = vol_flows # {'CO2': 1.0}
    vol_flow_total = sum(vol_flow.values())
    db_entry_dict = {'Software': 'PV MassSpec-20.08.01',
                     'Name': data_name,
                     'Measurement Type': 'bin',
                     'Catalyst Sample': sample_name,
                     'Temperature (C)': temperature,
                     'Experiment Type': 'continuous',
                     'Lamp': lamp,
                     'Gas Molar Composition (-)': 
                         {k: v / vol_flow_total for k, v in vol_flow.items()},
                     # 'Gas Molar Composition (-)': {'CO2': vol_flow['CO2'] / vol_flow_total , 
                     #                               'CH4': vol_flow['CH4'] / vol_flow_total}, 
                     # 'Liquid Molar Composition (-)': {'H2O': 1.0},
                     # 'Liquid Molar Composition (-)': {'H2O': 1.0 - 0.019, 'C3H8O': 0.019},
                     'Liquid Flow Rate (g/h)': liquid_flow,
                     'Gas Flow Rate (sccm)': vol_flow_total}
    
    if 'IPA' in data_file_name:
        mol_fract_ipa = 0.019
        db_entry_dict['Liquid Molar Composition (-)'] = \
            {'H2O': 1.0 - mol_fract_ipa, 'C3H8O': mol_fract_ipa}
    else:
        db_entry_dict['Liquid Molar Composition (-)'] = {'H2O': 1.0}
        
                     # 'Radiation Source': 'HMI 150W',
    db_entry_dict['Annotations'] = annotations
    db_entry_dict['Data'] = data_dict_filtered
    # add general info to database entry
    db_entry_dict.update(general_info_dict)
    db_entry_dicts.append(db_entry_dict)


# In[342]:


# initialize mongo connector object with ip adress
client = MongoClient('zbts07')
# get reference to existing database testDB
db = client.HyperSol
# authentication within database
db.authenticate('lfd', 'qwertz', source='admin')


# In[343]:

# insert loaded entry into database collection
current_collection.insert_many(db_entry_dicts)


# In[ ]:




