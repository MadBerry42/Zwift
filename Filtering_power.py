import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from openpyxl import load_workbook
import Extract_HR_Features

# Participant details
members = { "000":{"Age": 25, "Height": 160, "Weight": 58, "Gender": 1, "FTP": 49, "RPE": [[13, 11, 12], [12, 11, 13]], "Activity": [0, 0, 7*60]},
        "002":{"Age": 26, "Height": 177, "Weight": 75, "Gender": 0, "FTP": 78, "RPE": [[9, 12, 14], [9, 11, 13]], "Activity": [0, 4*45, 6*30]},
        "003":{"Age": 22, "Height": 180, "Weight": 70, "Gender": 0, "FTP": 51, "RPE": [[10, 12, 15], [8, 10, 12]],"Activity": [4*60, 7*15, 5*30]},
        "004":{"Age": 22, "Height": 186, "Weight": 80, "Gender": 0, "FTP": 47, "RPE": [[11, 12, 12], [10, 11, 12]],"Activity": [3*90, 1*5*60, 5*30]},
        "006":{"Age": 23, "Height": 174, "Weight": 87, "Gender": 1, "FTP": 51, "RPE": [[10, 11, 12], [10, 11, 10]],"Activity": [2*40, 0, 7*120]},
        "007":{"Age": 23, "Height": 183, "Weight": 70, "Gender": 0, "FTP": 55, "RPE": [[11, 11, 12], [8, 9, 11]],"Activity": [0, 0, 7*60]},
        "008":{"Age": 23, "Height": 190, "Weight": 82, "Gender": 0, "FTP": 82, "RPE": [[12, 12, 14], [10, 12, 14]],"Activity": [4*90, 0, 7*30]},
        "009":{"Age": 32, "Height": 185, "Weight": 96, "Gender": 0, "FTP": 62, "RPE": [[9, 11, 14], [10, 11, 14]],"Activity": [0, 0, 5*60]},
        "010":{"Age": 24, "Height": 160, "Weight": 56, "Gender": 1, "FTP": 48, "RPE": [[9, 11, 13], [9, 10, 11]],"Activity": [5*60, 2*60, 7*30]}, # Time spent for activity is grossly estimated
        "011":{"Age": 28, "Height": 176, "Weight": 67, "Gender": 0, "FTP": 60, "RPE": [[11, 10, 13], [10, 11, 13]],"Activity": [0, 0, 7*40]},
        "012":{"Age": 28, "Height": 184, "Weight": 70, "Gender": 0, "FTP": 87, "RPE": [[10, 11, 12], [10, 11, 12]],"Activity": [10, 4*20, 7*45]},
        "013":{"Age": 25, "Height": 178, "Weight": 66, "Gender": 0, "FTP": 62, "RPE": [[10, 11, 13], [11, 11, 13]],"Activity": [1*60, 1*60, 3*35]},
        "015":{"Age": 21, "Height": 176, "Weight": 73, "Gender": 0, "FTP": 60, "RPE": [[9, 10, 11], [10, 11, 13]],"Activity": [4*80, 6*2*60, 7*8*60]},
        "016":{"Age": 24, "Height": 173, "Weight": 59, "Gender": 1, "FTP": 37, "RPE": [[9, 10, 11], [7, 9, 10]],"Activity": [2*45, 2*30, 7*60]},
        "017":{"Age": 24, "Height": 187, "Weight": 75, "Gender": 0, "FTP": 58, "RPE": [[8, 8, 8], [8, 8, 11]], "Activity": [0, 0, 4*60, 13*60*7]},
        "019":{"Age": 24, "Height": 175, "Weight": 68, "Gender": 0, "FTP": 73, "RPE": [[8, 9, 9], [10, 9, 10]], "Activity": [4*120, 0, 7*60, 5*60*7]},
        "020":{"Age": 25, "Height": 174, "Weight": 73, "Gender": 0, "FTP": 88, "RPE": [[10, 11, 14], [9, 12, 14]], "Activity": [2*90, 0, 7*30, 8*60*7]},
        }

participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
for i, ID in enumerate(participants):
    # ID = int(input("What is participant's ID?"))
    ID = f"{ID:03}"
    Gender = members[f"{ID}"]["Gender"]
    Age = members[f"{ID}"]["Age"]
    Height = members[f"{ID}"]["Height"]
    Weight = members[f"{ID}"]["Weight"]
    max_HR = 220 - Age

    path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data"

    # Filter details
    window_size = 15

    #-----------------------------------------------------------------------------------------------------------------------------------------
    # Handcycle
    #-----------------------------------------------------------------------------------------------------------------------------------------
    setup = "handcycle"
    data = pd.read_csv(f"{path}\\{ID}\\Zwift\\{ID}_{setup}_protocol.csv")
    beginning = 300 # 300 for the first part of the workout, 840 for the second part
    final = 1380 # 840 for the first part, 1380 for the second part
    data = data[beginning : final]
    power_hc = np.array(data["Power"])
    HR = np.array(data["Heart Rate"])
    RPE = np.array(data["RPE"])
    if RPE[0] != 12:
        RPE[0] = 12
    RPE = RPE.astype(int)
    cadence = np.array(data["Cadence"])
    # Moving average filter
    window = np.ones(window_size) / window_size
    window = window.flatten()
    power_hc = power_hc.flatten()
    power_hc = np.convolve(power_hc, window, mode = "same")
    # Plotting the signal
    t = np.linspace(beginning, final, num = len(power_hc))
    plt.figure()
    plt.plot(t, data["Power"], label = "Original siganl")
    plt.plot(t, power_hc, label = "Filtered signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Power output [W]")
    plt.title(f"Subject {ID}, {setup}")

    # Feature extraction
    # features_hr = Extract_HR_Features.get_features_from_hr_signal(HR, 'hr')

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Bicycle
    #-----------------------------------------------------------------------------------------------------------------------------------------
    setup = "bicycle"
    data = pd.read_csv(f"{path}\\{ID}\\Zwift\\{ID}_{setup}_protocol.csv", usecols = ["Power"])
    data = data[beginning : final]
    power_bc = np.array(data)
    # Moving average filter
    window = np.ones(window_size) / window_size
    window = window.flatten()
    power_bc = power_bc.flatten()
    data_filtered_bc = np.convolve(power_bc, window, mode = "same")

    '''data_filtered = power_hc[1: -1]
    data_filtered_bc = data_filtered_bc[1: -1]'''

    # Two subplots
    '''fig, ax = plt.subplots(2)
    ax[0].plot(t, data)
    ax[0].set_title("Original signal")
    ax[1].plot(t, data_filtered)
    ax[1].set_title("Filtered signal")'''

    # One plot, signals overlapped
    plt.figure()
    plt.plot(t, data["Power"], label = "Original data")
    plt.plot(t, data_filtered_bc, label = "Filtered data")
    plt.xlabel("Time [s]")
    plt.ylabel("Power output [W]")
    plt.title(f"Subject {ID}, {setup}")

    plt.legend()
    # plt.show()

    '''# Save data in a .csv file
    directory = f"Acquisitions\\Protocol\\Processed data\\Input to model"
    file_output = f"{ID}_input_file.xlsx"
    csv_file = os.path.join(directory, file_output)

    Age = np.ones(540) * Age
    Weight = np.ones(540) * Weight
    Height = np.ones(540) * Height
    max_HR = np.ones(540) * max_HR
    Gender = np.ones(540) * Gender'''

    '''with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Gender", "Age", "Weight", "Height", "Power hc", "Heart Rate", "Max HR", "RPE", "Power bc"])
        rows = zip(Gender, Age, Weight, Height, data_filtered, HR, max_HR, RPE, data_filtered_bc)
        writer.writerows(rows)'''

    # Create an excel file
    path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data\Input to models\RPE Model single"
    Col_A = [' ', 'Gender', 'Age', 'Weight', 'Height']
    Col_B = [' ', Gender, Age, Weight, Height]
    Col_A.extend([' '] * (len(power_hc) - len(Col_A)))
    Col_B.extend([' '] * (len(power_hc) - len(Col_B)))

    writer = pd.ExcelWriter(f'{path}\\{ID}_input_file.xlsx', engine = "openpyxl")
    wb = writer.book
    df = pd.DataFrame({'P info': Col_A, ' ' : Col_B, 'Heart Rate': HR, 'RPE': RPE, 'Cadence': cadence, 'Power hc': power_hc, 'Power bc': data_filtered_bc})
    '''features_hr = pd.DataFrame([features_hr])
    df = pd.concat([df, features_hr], axis = 1)
    df = df.fillna(' ')'''

    df.to_excel(writer, index = False)
    wb.save(f'{path}\\{ID}_input_file.xlsx')

    # If you ever only want to save power in your .csv file
    '''csv_file = os.path.join(directory, f"{ID}_filtered_power")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Power hc", "Power bc"])
        rows = zip(Age, Weight, Height, data_filtered, HR, max_HR, RPE, data_filtered_bc)
        writer.writerows(rows)'''

    print(f"File for participant {ID} has been succesfully saved!")