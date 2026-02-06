import pandas as pd
import json
from pandas import json_normalize, read_json
from datetime import datetime, timezone


"""
Parse schedule.json file, making dictionary for schedule table
id bus_name route_name day_type schedule(list)
             WEST        
For item in d[weekday]:
    make schedule_list of times
    for item in d[weekday][busname]

        add time to schedule_list 
    schedule_table[schedule_list] = schedule_list
do same thing above for sat and sunday
"""


with open('shared/schedule.json') as f:
    d = json.load(f)
    

schedule_table = {"bus_name": [], "route_name": [], "day_type": [], "schedule": []}
for item in d['weekday']:

    schedule_table['day_type'].append('weekday')
    schedule_table['bus_name'].append(item)
    if 'WEST' in item:
        schedule_table['route_name'].append('WEST')
    else: 
        schedule_table['route_name'].append('NORTH')
    
    times = list()
    for item in d['weekday'][item]:
        times.append(item[0])
    schedule_table['schedule'].append(times)

for item in d['saturday']:
    schedule_table['day_type'].append('saturday')
    schedule_table['bus_name'].append(item)
    if 'WEST' in item:
         schedule_table['route_name'].append('WEST')
    else: 
       schedule_table['route_name'].append('NORTH')
    
    times = list()
    for item in d['saturday'][item]:
        times.append(item[0])
    schedule_table['schedule'].append(times)

for item in d['sunday']:
    schedule_table['day_type'].append('sunday')
    schedule_table['bus_name'].append(item)
    if 'WEST' in item:
         schedule_table['route_name'].append('WEST')
    else: 
        schedule_table['route_name'].append('NORTH')
    times = list()
    for item in d['sunday'][item]:
        times.append(item[0])
    schedule_table['schedule'].append(times)

schedules = pd.DataFrame.from_dict(schedule_table)

print(schedules)
