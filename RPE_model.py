import pandas as pd
import numpy as np
import Extract_HR_Features
import scipy 
import matplotlib.pyplot as plt
import openpyxl

#-------------------------------------------------------------------------------------------------
    # Import data
#-------------------------------------------------------------------------------------------------
participants = [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models"


n_windows = 1 # number of windows for each block
window_length = int(180/n_windows)

for j in range(1, 7):
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


        # Time-dependent data
        HR = data.iloc[(j - 1) * window_length : j * window_length, 2]
        RPE_value = int(data.iloc[(j - 1) * window_length, 3])
        RPE = pd.DataFrame({'RPE': [RPE_value]})
        cadence = data.iloc[(j - 1) * window_length : j * window_length, 4]
        Power_hc = data.iloc[(j - 1) * window_length : j * window_length, 5]
        time = np.linspace(300, 1380, len(Power_hc))
        
        # Remove outliers
        HR = HR.replace(0, np.nan)
        HR = HR.interpolate(method = "linear")
        Power_hc = Power_hc.replace(0, np.nan)
        Power_hc = Power_hc.interpolate(method = "linear")

        for k in range(1, n_windows + 1):
            hr_features = Extract_HR_Features.Extract_HR_Features(HR[(k - 1) * window_length : k * window_length + 1], window_length, np.max(HR), 'hr')
            Power_hc_features = Extract_HR_Features.Extract_HR_Features(Power_hc[(k - 1) * window_length : k * window_length], window_length, np.max(Power_hc), 'P_hc')
            cadence_features = Extract_HR_Features.Extract_HR_Features(cadence[(k - 1) * window_length : k * window_length], window_length, np.max(cadence), 'cadence')

            hr_features.transpose()
            Power_hc_features.transpose()
            cadence_features.transpose()

            hr_features= hr_features.reset_index(drop = True)
            cadence_features = cadence_features.reset_index(drop = True)
            Power_hc_features = Power_hc_features.reset_index(drop = True)
            RPE = RPE.reset_index(drop = True)

            if j == 1 and i == 0 and k == 1:
                P_info = pd.DataFrame({'ID': ID, 'Age': [age], 'Height': [height], 'Weight': [weight], 'Gender': [gender]})
                P_info = P_info.reset_index(drop = True)
                df = pd.concat([P_info, RPE, hr_features, cadence_features, Power_hc_features], axis = 1)
            elif k == 1:
                P_info = pd.DataFrame({'ID': ID, 'Age': [age], 'Height': [height], 'Weight': [weight], 'Gender': [gender]})
                values = pd.concat([P_info, RPE, hr_features, cadence_features, Power_hc_features], axis = 1)
                values.columns = df.columns
                df = pd.concat([df, values], axis=0, ignore_index=True)
            else:
                P_info = pd.DataFrame({'ID': ' ', 'Age': [age], 'Height': [height], 'Weight': [weight], 'Gender': [gender]})
                hr_temp = hr_features.iloc[[k], :].reset_index(drop = True)
                cadence_temp = cadence_features.iloc[[k], :].reset_index(drop = True)
                Power_hc_temp = Power_hc_features.iloc[[k]].reset_index(drop = True)
                values = pd.concat([P_info, RPE, hr_features, cadence_features, Power_hc_features], axis = 1)
                values.columns = df.columns
                df = pd.concat([df, values], axis=0, ignore_index=True)

#---------------------------------------------------------------------------------------------------------------------------------------------
    # Save everything in an excel file
#---------------------------------------------------------------------------------------------------------------------------------------------
length = int(180/n_windows)

writer = pd.ExcelWriter(f'{path}\\RPE Models\\{length}_sec_feature_extraction.xlsx', engine = "openpyxl")
wb = writer.book
df.to_excel(writer, index = False)
wb.save(f'{path}\\RPE Models\\{length}_sec_feature_extraction.xlsx')

print("File saved succesfully")


# Fix all of the participants' data, from 4 on they miss a sample. Just rerun filtering_power.


