import pandas as pd
import matplotlib.pyplot as plt
from numpy import*

df_smartwatch = pd.read_csv("Data\\9_acquisition_Smartwatch.csv")
HR_Zwift = pd.read_csv("Data\\9_acquisition_Zwift.csv", usecols = ["Heart Rate"])
timestamp_Zwift = pd.read_csv("Data\\9_acquisition_Zwift.csv", usecols = ["Timestamp"])
# print(timestamp_Zwift) # Check if the importation went fine

n_rows_sw = len(df_smartwatch)
n_rows_Zwift = len(HR_Zwift)

HR_smartwatch = df_smartwatch.iloc[2:n_rows_sw, 2]
timestamp_sw = df_smartwatch.iloc[2:n_rows_sw, 1]

# Plotting the figures
# Allineare gli assi dei tempi
'''dat1['timeDiff'] = (dat1['time'] - dat1['time'][0]).astype('timedelta64[D]')
dat2['timeDiff'] = (dat2['time'] - dat2['time'][0]).astype('timedelta64[D]')

fig,ax = plt.subplots()
ax.plot(dat1['timeDiff'],dat1['Value'])
ax.plot(dat2['timeDiff'],dat2['Value'])
plt.show()'''

'''plt.plot(t_smartwatch, HR_smartwatch, 'r') # plotting t, a separately 
plt.plot(t_Zwift, HR_Zwift, 'b') # plotting t, b separately 
plt.show()'''