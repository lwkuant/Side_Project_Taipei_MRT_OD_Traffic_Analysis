# -*- coding: utf-8 -*-
"""
MRT OD traffic analysis
http://data.taipei/opendata/datalist/datasetMeta?oid=63f31c7e-7fc3-418b-bd82-b95158755b4d
"""


"""
### Load required packages
import pandas as pd 
import numpy as np

import os 
os.chdir('D:/Dataset/Side_project_MRT_OD_traffic_analysis/data')

### Process the data into a table
#df = pd.read_csv('臺北捷運每日分時各站OD流量統計資料_201704.csv', sep = ' ')
#df.head()
# Using the pandas to read this dataset is not a good choice

## Instead I will use the open() from Python 

file = open('臺北捷運每日分時各站OD流量統計資料_201704.csv', encoding = 'utf-8')
df = file.readlines()

## Parse the data
import re
df = [re.split(' +', line) for line in df[2:(len(df) - 4)]]

## Make it a table 
df = pd.DataFrame(df)
df.columns = ['日期', '時段', '進站', '出站', '人次']
df.head()

# Transform the 日期 column to date type
from datetime import datetime
df['日期'] = df['日期'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df['日期'].value_counts()

# Transform the hour into integer
df['時段'] = df['時段'].apply(lambda x: int(x))

# Strip the \n from 人次
df['人次'] =df['人次'].apply(lambda x: x[:-1])
"""



### Write a function to process the data 
def MRT_OD_processing(file_name):
    import pandas as pd
    import numpy as np
    import re
    
    file = open(file_name, encoding = 'utf-8')
    df = file.readlines()
    file.close()
    
    df = [re.split(' +', line) for line in df[2:(len(df) - 4)]]
    
    df = pd.DataFrame(df)
    df.columns = ['日期', '時段', '進站', '出站', '人次']
    
    #from datetime import datetime
    #df['日期'] = df['日期'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    
    df['時段'] = df['時段'].apply(lambda x: int(x))
    
    df['人次'] =df['人次'].apply(lambda x: x[:-1])
    df['人次'] = df['人次'].astype(int)
    
    return df

### Load the data
import os 
os.chdir('D:/Dataset/Side_project_MRT_OD_traffic_analysis/data')
import pandas as pd 
import numpy as np
import seaborn as sns
import re
from scipy import stats

os.listdir()

import time
start_time = time.time()
df = MRT_OD_processing('臺北捷運每日分時各站OD流量統計資料_201701.csv')
print(time.time() - start_time) # roughly 50 sec to read the data 




### Save the data into database
import sqlite3
import os
os.chdir('D:/Dataset/Side_project_MRT_OD_traffic_analysis/data')

conn = sqlite3.connect('MRT.db')

cur = conn.cursor()
cur.execute("CREATE TABLE MRT_OD_Traffic(Date TEXT, Time INT, Origin TEXT, Destination TEXT, Traffic INT)")
conn.close()

start_time = time.time()
for f in os.listdir('D:/Dataset/Side_project_MRT_OD_traffic_analysis/data'):
    if f.startswith('臺北捷運每日分時各站OD流量統計資料'):
        
        conn = sqlite3.connect('MRT.db')
        cur = conn.cursor()
        
        df = MRT_OD_processing(f)
        cur.executemany("INSERT INTO MRT_OD_Traffic VALUES(?, ?, ?, ?, ?)", df.values)
        conn.commit()
        
        print(f)
        
done_time = time.time() - start_time
print(done_time) # 1387.1348185539246s for month 1 - 12

#cur.executemany("INSERT INTO MRT_OD_Traffic VALUES(?, ?, ?, ?, ?)", df.values)
# cur.execute("DROP TABLE IF EXISTS MRT_OD_Traffic")
# cur.execute("SELECT * FROM MRT_OD_Traffic limit 5;") 
# cur.fetchall() 

conn.close()


### Overview of the data
conn = sqlite3.connect('MRT.db')
cur = conn.cursor()
cur.execute("SELECT * FROM MRT_OD_Traffic limit 5;").fetchall() 
t = cur.execute('''SELECT Origin, SUM(Traffic) FROM MRT_OD_Traffic
                GROUP BY Origin''').fetchall() 

t = cur.execute('''SELECT Date, SUM(Traffic) FROM MRT_OD_Traffic
                GROUP BY substr(Date, 6, 2)''').fetchall() 

t = cur.execute('''SELECT Origin, Destination, SUM(Traffic) FROM MRT_OD_Traffic
                GROUP BY Origin, Destination
                ORDER BY SUM(Traffic) DESC''').fetchall() 

cur.execute('''SELECT COUNT(Traffic) FROM MRT_OD_Traffic''').fetchall() 

## 1. Total OD traffic by origin stations
df.groupby(['進站'])['人次'].sum().sort_values(ascending = False)

## 2. Total OD traffic by destination stations
df.groupby(['出站'])['人次'].sum().sort_values(ascending = False)

## 3. Total OD traffic within the same stations 
df.ix[df['進站'] == df['出站'], ].groupby(['進站'])['人次'].sum().sort_values(ascending = False)

## 4. The OD combination with the most traffic 
df.groupby(['進站', '出站'])['人次'].sum().sort_values(ascending = False)

## 5. The distribution of OD traffic by time 
df.groupby(['時段'])['人次'].sum()
sns.set_style("darkgrid")
df.groupby(['時段'])['人次'].sum().plot()

## 6. The distribution of OD traffic days
df.groupby(['日期'])['人次'].sum()
sns.set_style("darkgrid")
df.groupby(['日期'])['人次'].sum().plot()



### Load the exit data
df_exit = pd.read_csv('大臺北地區捷運車站出入口座標.csv')
df_exit['車站名'] = df_exit['出入口名稱'].apply(lambda x: re.sub('站.*', '', x))
df_exit['車站名'] = df_exit['車站名'].apply(lambda x: x+'站' if x == '台北車' else x)

unique_station_exit = np.unique(df_exit['車站名'])
unique_station = np.unique(df['進站'].value_counts().index)
np.setdiff1d(unique_station, unique_station_exit)

df_exit['車站名'] = df_exit['車站名'].apply(lambda x: '台北橋' if x == '臺北橋' else x)
df_exit['車站名'] = df_exit['車站名'].apply(lambda x: '大橋頭站' if x == '大橋頭' else x)
unique_station_exit = np.unique(df_exit['車站名'])
np.setdiff1d(unique_station, unique_station_exit)

## Count the number of exit for each station 
df_exit['車站名'].value_counts()
sns.set_style("darkgrid")
df_exit['車站名'].value_counts().hist()

## The relationship between number of exit and OD traffic 
#df_exit['車站名'].value_counts().index
# by 進站
orig_df_station_index = df.groupby(['進站'])['人次'].sum().reset_index()['進站'].values
OD_traffic = df.groupby(['進站'])['人次'].sum().reset_index().ix[[list(orig_df_station_index).index(x) for x in df_exit['車站名'].value_counts().index], :]['人次'].values
stats.pearsonr(OD_traffic, df_exit['車站名'].value_counts().values)
# by 出站
orig_df_station_index = df.groupby(['出站'])['人次'].sum().reset_index()['出站'].values
OD_traffic = df.groupby(['出站'])['人次'].sum().reset_index().ix[[list(orig_df_station_index).index(x) for x in df_exit['車站名'].value_counts().index], :]['人次'].values
stats.pearsonr(OD_traffic, df_exit['車站名'].value_counts().values)

### EDA

## Total number of passengers
# 進出量統計
df.ix[df['日期'] == '2017-04-01', ].head()
np.sum(df.ix[df['日期'] == '2017-04-01', '人次']) # the first day
np.sum(df.ix[df['日期'] == '2017-04-15', '人次']) 
np.sum(df['人次']) # the whole month
np.sum(df['人次'])/len(np.unique(df['日期'].values)) # daily average 

## Day by hour vis
day_to_hour = df.groupby(['日期', '時段'])
day_to_hour['人次'].sum()
day_to_hour['人次'].sum().unstack()

import os 
os.chdir('D:/Dataset/Side_project_MRT_OD_traffic_analysis')
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='D:/Downloads/Microsoft_JH_6.12/msjh.ttc', size=15)
weekdays = {6: 'Sun',
            0: 'Mon',
            1: 'Tue',
            2: 'Wed',
            3: 'Thu',
            4: 'Fri',
            5: 'Sat'
            
        }

plt.figure(figsize = [15, 8])

from datetime import datetime

sns.heatmap(day_to_hour['人次'].sum().unstack(),
            yticklabels = list([str(x)[:10] + ' ' + weekdays[datetime.strptime(str(x)[:10], '%Y-%m-%d').weekday()] for x in np.unique(df['日期'])]),
            cmap = 'Reds')
plt.xlabel('時段', fontproperties=myfont)
plt.ylabel('日期', fontproperties=myfont)
plt.title('台北捷運4月份各時段進出量統計', fontproperties=myfont, y = 1.05)
plt.savefig('MRT_Apr_Ridership_Counts.png', dpi = 300, format='png')


def plot_day_hour_heatmap(df):
    ### 
    import seaborn as sns 
    import matplotlib.pyplot as plt
    %matplotlib inline
    from datetime import datetime
    import matplotlib.font_manager as fm
    
    # week day mapping
    weekdays = {6: 'Sun',
            0: 'Mon',
            1: 'Tue',
            2: 'Wed',
            3: 'Thu',
            4: 'Fri',
            5: 'Sat'
            
        }
    
    # figure setting
    plt.figure(figsize = [15, 8])
    myfont = fm.FontProperties(fname='D:/Downloads/Microsoft_JH_6.12/msjh.ttc', size=15)
    
    # plotting
    sns.heatmap(day_to_hour['人次'].sum().unstack(),
            yticklabels = list([str(x)[:10] + ' ' + weekdays[datetime.strptime(str(x)[:10], '%Y-%m-%d').weekday()] for x in np.unique(df['日期'])]),
            cmap = 'Reds')    
    plt.xlabel('時段', fontproperties=myfont)
    plt.ylabel('日期', fontproperties=myfont)
    plt.title('台北捷運4月份各時段進出量統計', fontproperties=myfont, y = 1.05)

### plotting
plot_day_hour_heatmap(df)



#==============================================================================
###
# Process the number of exit of each station and their locations
###

import pandas as pd 
import numpy as np 
import os 
os.chdir('D:/Dataset/Side_project_MRT_OD_traffic_analysis')

df_station = pd.read_csv('大臺北地區捷運車站出入口座標.csv', encoding = 'utf-8')

import re
df_station['出入口名稱'] = df_station['出入口名稱'].apply(lambda x: x[:re.search('站', x).start()+1])
df_station.groupby(['出入口名稱'])['出入口編號'].count()
df_station.groupby(['出入口名稱'])[['經度', '緯度']].mean()


