import pandas as pd
import numpy as np
import Extract_HR_Features
import ExtractIMU_Features
# import scipy 
import matplotlib.pyplot as plt
# import openpyxl

#-------------------------------------------------------------------------------------------------
    # Import data
#-------------------------------------------------------------------------------------------------
participants = [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]

# path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\Processed Data\\Input to model"
path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models"

n_windows = 3

 # number of windows for each block
window_length = int(180/n_windows)

for j in range(1, 6 + 1):
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        elif ID >= 10:
            ID = f"0{ID}"

        data = pd.read_excel(f"{path}\\{ID}_input_file.xlsx")
        IMU_data_crank = pd.read_csv(f"{path}\\RPE Models\\IMU Data\\{ID}_handcycle_protocol_crank_processed.csv")
        IMU_data_wrist = pd.read_csv(f"{path}\\RPE Models\\IMU Data\\{ID}_handcycle_protocol_wrist_processed.csv")

        ID_int = int(ID)
        if ID_int <= 8:
            fs = 200
            # removing warmup and cooldown
            IMU_data_wrist = IMU_data_wrist[60 * fs * 5 : - (60 * fs * 4)]
            IMU_data_crank = IMU_data_crank[60 * fs * 5 : - (60 * fs * 4)]


        # Constant values
        age = int(data.iloc[2, 1])
        gender = int(data.iloc[1, 1])
        weight = int(data.iloc[3, 1])
        height = int(data.iloc[4, 1])/100


        # Time-dependent data
        HR = data.iloc[(j - 1) * window_length * n_windows : j * window_length * n_windows, 2] # Isolate one single block
        # HR = data.iloc[:, [2]]
        # RPE_value = int(data.iloc[(j - 1) * window_length, 3])
        RPE_value = np.mean(data.iloc[(j - 1) * window_length * n_windows : j * window_length * n_windows, 3])
        RPE = pd.DataFrame({'RPE': [RPE_value]})
        cadence = data.iloc[(j - 1) * window_length * n_windows : j * window_length * n_windows, 4]
        Power_hc = data.iloc[(j - 1) * window_length * n_windows : j * window_length * n_windows, 5]
        time = np.linspace(300, 1380, len(Power_hc))
        
        # Remove outliers
        HR = HR.replace(0, np.nan)
        HR = HR.interpolate(method = "linear")
        Power_hc = Power_hc.replace(0, np.nan)
        Power_hc = Power_hc.interpolate(method = "linear")

        count = 0
        for k in range(1, n_windows + 1):
            PeakHR = 220 - age

            hr_features = Extract_HR_Features.Extract_HR_Features(HR[(k - 1) * window_length : k * window_length + 1], window_length, PeakHR, 'hr')
            Power_hc_features = Extract_HR_Features.Extract_HR_Features(Power_hc[(k - 1) * window_length : k * window_length], window_length, np.max(data.iloc[:, 5]), 'P_hc')
            cadence_features = Extract_HR_Features.Extract_HR_Features(cadence[(k - 1) * window_length : k * window_length], window_length, np.max(data.iloc[:, 4]), 'cadence')
            IMU_features_crank = ExtractIMU_Features.ExtractIMU_Features(participants[i], j, count, IMU_data_crank[(k - 1) * window_length * 200 : k * window_length * 200], window_length * 200, Norm_Accel=False)
            IMU_features_wrist = ExtractIMU_Features.ExtractIMU_Features(participants[i], j, count, IMU_data_wrist[(k - 1) * window_length * 200 : k * window_length * 200], window_length * 200, Norm_Accel=False)
            count = count + 1

            hr_features.transpose()
            Power_hc_features.transpose()
            cadence_features.transpose()

            hr_features= hr_features.reset_index(drop = True)
            cadence_features = cadence_features.reset_index(drop = True)
            Power_hc_features = Power_hc_features.reset_index(drop = True)
            RPE = RPE.reset_index(drop = True)

            if j == 1 and i == 0 and k == 1:
                P_info = pd.DataFrame({'ID': [ID], 'Age': [age], 'Height': [height], 'Weight': [weight], 'Gender': [gender]})
                P_info = P_info.reset_index(drop = True)
                df = pd.concat([P_info, RPE, hr_features, cadence_features, Power_hc_features, IMU_features_crank, IMU_features_wrist], axis = 1)
            elif k == 1:
                P_info = pd.DataFrame({'ID': [ID], 'Age': [age], 'Height': [height], 'Weight': [weight], 'Gender': [gender]})
                values = pd.concat([P_info, RPE, hr_features, cadence_features, Power_hc_features, IMU_features_crank, IMU_features_wrist], axis = 1)
                values.columns = df.columns
                df = pd.concat([df, values], axis=0, ignore_index=True)
            else:
                P_info = pd.DataFrame({'ID': ' ', 'Age': [age], 'Height': [height], 'Weight': [weight], 'Gender': [gender]})
                hr_temp = hr_features.reset_index(drop = True)
                cadence_temp = cadence_features.reset_index(drop = True)
                Power_hc_temp = Power_hc_features.reset_index(drop = True)
                IMU_features_crank_temp = IMU_features_crank.reset_index(drop = True)
                IMU_features_wrist_temp = IMU_features_wrist.reset_index(drop = True)
                values = pd.concat([P_info, RPE, hr_temp, cadence_temp, Power_hc_temp, IMU_features_crank_temp, IMU_features_wrist_temp], axis = 1)
                values.columns = df.columns
                df = pd.concat([df, values], axis=0, ignore_index=True)

#---------------------------------------------------------------------------------------------------------------------------------------------
    # Save everything in an excel file
#---------------------------------------------------------------------------------------------------------------------------------------------
length = int(180/n_windows)

path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\\RPE Models"
# path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\RPE model\\Input files"
writer = pd.ExcelWriter(f'{path}\\{length}_sec_feature_extraction_IMU.xlsx', engine = "openpyxl")
wb = writer.book
df.to_excel(writer, index = False)
wb.save(f'{path}\\{length}_sec_feature_extraction_IMU.xlsx')

print("File saved succesfully")




