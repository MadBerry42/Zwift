import pandas as pd
import numpy as np
import Extract_HR_Features
import scipy 
import matplotlib.pyplot as plt
import openpyxl

#-------------------------------------------------------------------------------------------------
    # Import data
#-------------------------------------------------------------------------------------------------
participants = [0]

path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models"

for i, ID in enumerate(participants):
    if ID < 10:
        ID = f"00{ID}"
    elif ID >= 10:
        ID = f"0{ID}"

    data = pd.read_excel(f"{path}\\{ID}_input_file.xlsx")

    # Constant values
    age = int(data.iloc[2, 1])
    gender = int(data.iloc[1, 1])
    weight = int(data.iloc[3, 1])
    height = int(data.iloc[4, 1])/100

    P_info = pd.DataFrame({'Age': [age], 'Height': [height], 'Weight': [weight], 'Gender': [gender]})

    # Time-dependent data
    HR = data.iloc[:, 2]
    RPE = data.iloc[:, 3]
    cadence = data.iloc[:, 4]
    Power_hc = data.iloc[:, 5]
    time = np.linspace(300, 1380, len(Power_hc))
    
    # Remove outliers
    HR = HR.replace(0, np.nan)
    HR = HR.interpolate(method = "linear")
    Power_hc = Power_hc.replace(0, np.nan)
    Power_hc = Power_hc.interpolate(method = "linear")

    window_length = 60
    hr_features = Extract_HR_Features.Extract_HR_Features(HR, window_length, np.max(HR))
    Power_hc_features = Extract_HR_Features.Extract_HR_Features(Power_hc, window_length, np.max(Power_hc))
    cadence_features = Extract_HR_Features.Extract_HR_Features(cadence, window_length, np.max(cadence))

    df = pd.concat([P_info, hr_features.iloc[0, :], cadence_features.iloc[0, :], Power_hc_features.iloc[0, :]], axis = 0)
    # This line does not work and also everything in the dataframe is called hr_something. 


