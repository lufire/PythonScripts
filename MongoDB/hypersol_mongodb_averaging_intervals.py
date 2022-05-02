#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import relevant libraries
import numpy as np
import pandas as pd
import os
import re
import copy
import matplotlib.pyplot as plt
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
# import mpl_toolkits.axisartist as axisartist
from pymongo import MongoClient
import json
# from itertools import cycle
from collections import OrderedDict
import plotly.graph_objects as go
import mongodb_credentials as mc
# import plotly.io as pio
# from fpdf import FPDF
# from tabulate import tabulate


# In[3]:


# initialize mongo connector object with ip adress
client = MongoClient(mc.HOST_NAME)
# get reference to existing database testDB
db = client.HyperSol
# authentication within database
db.authenticate(mc.USER_NAME, mc.PASSWORD, source='admin')


# In[4]:
# directories

# path to mass spectrum calibration data
ms_calibration_data_file = r'W:\Projekte\NRW_HyperSol_61904\Bearbeitung\Massenspektroskopie\Spektren\mass_spectrums.json'

# path to save plots
plot_path = r'W:\Projekte\NRW_HyperSol_61904\Bearbeitung\Ergebnisse\Averaged_Results'
# In[ ]:


# # reference collection, if not existent it will be created
# raw_collection = db['HyperSol_61904']
# # insert loaded entry into database collection
# start_date = datetime.strptime('07/13/21', '%m/%d/%y')
# end_date = datetime.strptime('09/14/21', '%m/%d/%y')
# db_entries = raw_collection.find({'DateTime': {'$gte': start_date, '$lte': end_date}})
# db_entries = list(db_entries)
# db_entry = db_entries[17]
# len(db_entries)


# In[146]:


# reference collection, if not existent it will be created
db = client.HyperSol
raw_collection = db['Raw']
# make regular expression for substring of sample name
sample_name = '21J7'
sample_name_re = re.compile(sample_name, re.I)
# db_entries = list(raw_collection.find({'Catalyst Sample': {'$regex': sample_name_re}})
# find entry in database collection
raw_db_entries = raw_collection.find()

averaged_collection = db['Averaged_2']

# %%

# start loop through each entry
counter = 0 
for db_entry in raw_db_entries:
    try: 
        count_docs = \
            averaged_collection.count_documents({'Name': db_entry['Name']})
    except KeyError:
        continue
    # if counter > 3:
    #     break
# In[147]:
    if count_docs == 0:
        counter += 1
        # make local directory to save data
        directory = os.path.join(plot_path, db_entry['Name'])
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        
        # In[148]:
        
        
        # get database entry general information
        general_info_keys = ['Catalyst Sample', 'Experiment Type', 
                             'Gas Molar Composition (-)', 
                             'Gas Flow Rate (sccm)', 
                             'Liquid Molar Composition (-)', 
                             'Liquid Flow Rate (g/h)', 'Radiation Source', 
                             'Temperature (C)']
        
        
        # In[149]:
        
        
        inlet_composition_key = 'Gas Molar Composition (-)'
        headers = ['Species', 'Molar Fraction (-)']
        headers = ['<b>' + header + '</b>'  for header in headers]
        inlet_composition = db_entry[inlet_composition_key]
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=headers,
                        line_color='black',
                        fill_color='white',
                        font_color='black',
                        align='center'),
            cells=dict(values=[list(inlet_composition.keys()), # 1st column
                               list(inlet_composition.values())], # 2nd column
                       line_color='black',
                       fill_color='white',
                       font_color='black',
                       align=['left', 'right']), columnwidth=[0.3, 0.7])])
        fig.update_layout(width=300, height=240, 
                          margin=dict(l=20, r=20, t=20, b=20))
        fig.show()
        file_name = 'Inlet_Composition.png'
        fig.write_image(os.path.join(directory, file_name), scale=2)
        # pio.write_image(fig, os.path.join(directory, file_name), width=400, height=240, scale=2)
        
        
        # In[150]:
        
        
        liquid_composition_key = 'Liquid Molar Composition (-)'
        headers = ['Species', 'Molar Fraction (-)']
        headers = ['<b>' + header + '</b>'  for header in headers]
        if liquid_composition_key in db_entry:
            liquid_composition = db_entry[liquid_composition_key]
        
            fig = go.Figure(data=[go.Table(
                header=dict(values=headers,
                            line_color='black',
                            fill_color='white',
                            font_color='black',
                            align='center'),
                cells=dict(values=[list(liquid_composition.keys()), # 1st column
                                   list(liquid_composition.values())], # 2nd column
                           line_color='black',
                           fill_color='white',
                           font_color='black',
                           align=['left', 'right']), columnwidth=[0.3, 0.7])])
            fig.update_layout(width=300, height=240, 
                              margin=dict(l=20, r=20, t=20, b=20))
            fig.show()
            file_name = 'Liquid_Composition.png'
            fig.write_image(os.path.join(directory, file_name), scale=2)
        
        # pio.write_image(fig, os.path.join(directory, file_name), width=400, height=240, scale=2)
        
        
        # In[151]:
        
        
        # manipulate db entry accordingly
        general_info_dict = OrderedDict({k: v for k, v in db_entry.items()
                             if k in general_info_keys})
        try:
            general_info_dict.pop(inlet_composition_key)
            general_info_dict.pop(liquid_composition_key)
        except KeyError:
            continue
        # general_info_dict.pop('Catalyst Sample')
        # general_info_dict['Sample Name'] = sample_name
        names = ['<b>' + key + '</b>' for key in general_info_dict.keys()]
        values = [names, list(general_info_dict.values())]
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=['x', 'x'],
                        line_color='black',
                        fill_color='white',
                        font_color='black',
                        font_size=1,
                        align='center',
                        height=0),    
            cells=dict(values=values, 
                       line_color='black',
                       fill_color='white',
                       font_color='black',
                       align=['left', 'right'],
                      ), columnwidth=[0.5, 0.5])])
        
        fig.update_layout(width=400, height=300, 
                          margin=dict(l=20, r=20, t=20, b=20))
        fig.for_each_trace(lambda t: t.update(header_fill_color = 'rgba(0,0,0,0)'))
        fig.show()
        file_name = 'General_Information.png'
        fig.write_image(os.path.join(directory, file_name), scale=2)
        
        
        # In[152]:
        
        
        # get data array from database entry
        source_data = db_entry['Data']
        data = {}
        data['y_keys'] = [k for k in source_data.keys() if k[0] in (str(i) for i in range(10))]
        data['y'] = np.asarray([source_data[key] for key in data['y_keys']])
        
        # get x-values and make size the array equal to y-values array
        data['x_key'] = 'Time Relative (sec)'
        data['x'] = np.asarray(source_data[data['x_key']])
        # x_values = np.asarray([x_values for i in range(len(y_values))])
        data['bins'] = [int(y_key.split('_')[0]) for y_key in data['y_keys']]
        
        
        # remove data points with 28 signals below threshold
        amu_id = data['bins'].index(28)
        threshold = 1e-11
        filter_ids =  data['y'][amu_id] > threshold
        data['y'] = data['y'][:, filter_ids]
        data['x'] = data['x'][filter_ids]
        
        # In[153]:
        
        
        # read amu species correspondence data
        with open(ms_calibration_data_file, 'r') as fp:
            ms_calibration_data_json = json.load(fp)
        amu_species_dict = OrderedDict(zip(ms_calibration_data_json['amu'], ms_calibration_data_json['species']))
        
        # filter functions
        def filter_species(name_list=None, amu_list=None):
            filtered_amu_species_dict = copy.deepcopy(amu_species_dict)
            if name_list is not None:
                filtered_amu_species_dict =             OrderedDict([(k, list(set(name_list).intersection(v))) 
                                 for k, v in amu_species_dict.items() 
                                 if len(list(set(name_list).intersection(v))) > 0])
            if amu_list is not None:
                filtered_amu_species_dict =             OrderedDict([(k, v) for k, v in filtered_amu_species_dict.items() 
                                 if k in amu_list])        
            return filtered_amu_species_dict
        
        def filter_data(input_data, name_list=None, amu_list=None):
            result_data = copy.deepcopy(input_data)
            filtered_amu_species_dict = filter_species(name_list=name_list, amu_list=amu_list)
            filtered_amu_list = np.array(list(filtered_amu_species_dict.keys()))
            bins = input_data['bins']
            filtered_amu_ids = [bins.index(k) for k in bins 
                                if k in list(filtered_amu_list)]
            result_data['amus'] = [bins[k] for k in filtered_amu_ids]
            filtered_amu_species_dict = filter_species(amu_list=filtered_amu_list)
            result_data['y'] = input_data['y'][filtered_amu_ids]
            result_data['labels'] = [str(k) + ' - ' + '/'.join(v) 
                              for k, v in filtered_amu_species_dict.items()
                              if k in bins]
            result_data['y_keys'] = list(np.asarray(input_data['y_keys'])[filtered_amu_ids])
            return result_data
        
        
        # In[154]:
        
        
        filtered_data = filter_data(data)


        # select only specific signal bins
        # species_list = ['H2', 'CH4', 'CO2']
        # amu_list = [1, 2, 15, 16, 19, 28, 44]
        amu_list = [1, 2, 15, 16, 19, 27, 28, 44, 45]
        custom_data = filter_data(data, amu_list=amu_list)
        

        # In[155]:        
        
        # time series plots with matplotlib
        # evaluation time limits
        t_start = 0.0
        
        t_end = 50000.0
        
        # get x-values and make size of the array equal to y-values array
        x = custom_data['x']
        ids = np.where((x > t_start) & (x < t_end))
        x_lims = x[ids]
        y_lims = custom_data['y'][:, ids]
        
        # plot box
        plot_dx = 0.0
        plot_dy = 0.0
        plot_width_factor = 1.0
        plot_height_factor = 0.95
        
        font_size = 16
        color_map = 'Set1'
        # get global y-minimum
        y_min = np.min(custom_data['y'])
        # make plots
        fig = plt.figure(figsize=(12, 12), dpi=150)
        fig.patch.set_facecolor('xkcd:white')
        #fig.subplots_adjust(right=1.01)
        
        # fig = plt.figure(tight_layout=True)
        
        # title = '\n'.join([key + ': ' + str(db_entry[key]) for key in general_info_keys])
        # ax = fig.add_subplot(311)
        plt.tight_layout()
        plt.rcParams.update({'font.size': font_size})
        
        # n_bins = len(bins)
        n_custom_bins = len(custom_data['amus'])
        cmap = plt.get_cmap(color_map)
        # graph with linearly scaled y-axis
        ax = fig.add_subplot(211)
        # linestyles = cycle(['solid', 'dotted'])
        ax.set_prop_cycle('color', [cmap(i) for i in np.linspace(0, 1, n_custom_bins)])
        # ax.set_title(title, fontsize=10)
        for i in range(n_custom_bins):
            ax.plot(x_lims, y_lims[i].flatten(), linewidth=1.0) # , linestyle=next(linestyles))
        ax.set_yscale('linear')
        ax.set_ylabel('Signal')
        ax.set_xlim([np.min(x_lims) - 10, np.max(x_lims) + 10])
        # ax.set_ylim([-1e-9, 4.0e-8])
        ax.set_xlabel('Time (s)')
        box = ax.get_position()
        # ax.set_position([plot_x, plot_y_1, box.width * plot_width_factor,  box.height * plot_height_factor])
        # ax.set_position([box.x0 + plot_dx, box.y0 + plot_dy, box.width * plot_width_factor,  box.height * plot_height_factor])
        # ax.legend(y_labels, bbox_to_anchor=(1.01, 1.02), loc='upper left', fontsize=8, edgecolor='k')
        # add annotations
        t_end_no_radiation = None
        # get plot annotations
        annotations = db_entry.get('Annotations', None)
        if annotations is not None:
            for annotation in annotations:
                ax.annotate(annotation['Text'].replace(',', '\n'),
                            xy=(annotation['Relative Time (s)'], y_min), xycoords='data',
                            xytext=(0, 60), textcoords='offset points', size=12,
                            arrowprops=dict(facecolor='black', arrowstyle='->', 
                                            connectionstyle="angle,angleA=0,angleB=-90,rad=10"),
                            horizontalalignment='right', verticalalignment='bottom', rotation=90)
                if 'Lamp on' in annotation['Text']:
                    t_end_no_radiation = annotation['Relative Time (s)']

        
        # graph with log scaled y-axis
        ax = fig.add_subplot(212)
        # linestyles = cycle(['solid', 'dotted'])
        ax.set_prop_cycle('color', [cmap(i) for i in np.linspace(0, 1, n_custom_bins)])
        for i in range(n_custom_bins):
            ax.plot(x_lims, y_lims[i].flatten(), linewidth=1.0) #, linestyle=next(linestyles))
        ax.set_yscale('log')
        ax.set_ylabel('Signal')
        ax.set_xlabel('Time (s)')
        ax.set_xlim([np.min(x_lims) - 10, np.max(x_lims) + 10])
        # ax.set_ylim([1e-12, 4.0e-8])
        
        # box = ax.get_position()
        #ax.set_position([box.x0 + plot_dx, box.y0 + plot_dy, box.width * plot_width_factor,  box.height * plot_height_factor])
        fig.legend(custom_data['amus'], fontsize=font_size-2, 
                   edgecolor='k', bbox_to_anchor=(1.01, 0.89))
        # plt.tight_layout()
        plt.show()
        # save figure
        file_name = 'Raw_Spectrum.png'
        fig.savefig(os.path.join(directory, file_name), bbox_inches='tight')
        
        
        # In[156]:
        
        if t_end_no_radiation is None:
            continue
        
        # create means for two intervals
        t_end = np.asarray([t_end_no_radiation, 
                            np.max(filtered_data['x'])])
        # get maximum value for a specific bin (here 2)
        max_amu = 2
        y = custom_data['y']
        amus = custom_data['amus']
        
        max_id = np.argmax(y, axis=1)[amus.index(2)]

        # t_end = np.asarray([t_end_no_radiation, 5100.0])
        # t_end = np.max(x_values)
        dt = 100.0
        
        t_max_value = filtered_data['x'][max_id]
        t_max = np.max(filtered_data['x'])
        if t_max_value > (t_end_no_radiation + dt):
            t_end_radiation = np.min([t_max, t_max_value + dt * 0.5])
        else:
            t_end_radiation = t_max
        t_end = np.asarray([t_end_no_radiation, t_end_radiation])
        t_start = (t_end - dt)
        intervals = np.vstack([t_start, t_end]).transpose()
        intervals
        df = pd.DataFrame.from_dict(source_data)
        dfs = []
        for i in range(len(intervals)):
            df_interval = df[df['Time Relative (sec)'].between(intervals[i, 0], intervals[i, 1])]
            dfs.append(df_interval.mean())
            # print(df_interval.mean())
        df_mean = pd.concat(dfs, axis=1).transpose()
        filtered_amu_list_str = [str(i) for i in filtered_data['amus']]
        columns_dict = dict(zip(filtered_data['y_keys'], filtered_amu_list_str))
        df_mean = df_mean.rename(columns=columns_dict, index={0: 'Radiation off', 1: 'Radiation on'})
        
        
        # In[157]:
        
        # plot mean signals
        # make plots
        fig, ax = plt.subplots()
        # fig.patch.set_facecolor('xkcd:white')
        # plt.tight_layout()
        # plt.rcParams.update({'font.size': font_size})
        custom_amu_list_str = [str(i) for i in custom_data['amus']]
        df_mean[custom_amu_list_str].transpose().plot(kind='bar', ax=ax, figsize=(10, 8), logy=True)
        ax.set_xlabel('m/z', fontsize=18)
        ax.set_ylabel('Signal', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='x', which='major', labelrotation=0)
        plt.show()
        
        
        # In[158]:
        # save figure
        file_name = 'Averaged_Spectrum.png'
        fig.savefig(os.path.join(directory, file_name), bbox_inches='tight')
        
        
        # In[159]:
        
        
        # replace data in database dict
        db_entry['Data'] = df_mean[filtered_amu_list_str].to_dict()
        db_entry['Averaging Intervals (s)'] = intervals.tolist()
        # remove time step data        
        
        # In[160]:

        # insert into database collection
        averaged_collection.insert_one(db_entry)



