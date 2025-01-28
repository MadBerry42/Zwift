import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from openpyxl import load_workbook
import Extract_HR_Features

# Participant details
ID = 16
Gender = 1 #0 for males, 1 for females
Age = 25
Weight = 59
Height = 173
max_HR = 195


if ID < 10:
    ID = f"00{ID}"
else:
    ID = f"0{ID}"

# Filter details
window_size = 15

#-----------------------------------------------------------------------------------------------------------------------------------------
# Handcycle
#-----------------------------------------------------------------------------------------------------------------------------------------
setup = "handcycle"
data = pd.read_csv(f"C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\{ID}\\Zwift\\{ID}_{setup}_protocol.csv")
data = data[300:840]
power_hc = np.array(data["Power"])
HR = np.array(data["Heart Rate"])
RPE = np.array(data["RPE"])
RPE = RPE.astype(int)
# Moving average filter
window = np.ones(window_size) / window_size
window = window.flatten()
power_hc = power_hc.flatten()
power_hc = np.convolve(power_hc, window, mode = "same")
# Plotting the signal
t = np.linspace(300, 840, num = len(power_hc))
plt.figure()
plt.plot(t, data["Power"])
plt.plot(t, power_hc)
plt.xlabel("Time [s]")
plt.ylabel("Power output [W]")
plt.title(f"Subject {ID}, {setup}")

plt.legend("Original signal", "Filtered signal")

# Feature extraction
features_hr = Extract_HR_Features.get_features_from_hr_signal(HR)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Bicycle
#-----------------------------------------------------------------------------------------------------------------------------------------
setup = "bicycle"
data = pd.read_csv(f"C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\{ID}\\Zwift\\{ID}_{setup}_protocol.csv", usecols = ["Power"])
data = data[300:840]
power_bc = np.array(data)
# Moving average filter
window = np.ones(window_size) / window_size
window = window.flatten()
power_bc = power_bc.flatten()
power_bc = np.convolve(power_bc, window, mode = "same")

data_filtered = power_hc[1: -1]
data_filtered_bc = power_bc[1: -1]


# Two subplots
'''fig, ax = plt.subplots(2)
ax[0].plot(t, data)
ax[0].set_title("Original signal")
ax[1].plot(t, data_filtered)
ax[1].set_title("Filtered signal")'''

# One plot, signals overlapped
plt.figure()
plt.plot(t, data["Power"])
plt.plot(t[1:-1], data_filtered_bc)
plt.xlabel("Time [s]")
plt.ylabel("Power output [W]")
plt.title(f"Subject {ID}, {setup}")

plt.legend("Original signal", "Filtered signal")
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
path = f"Acquisitions\\Protocol\\Processed data\\Input to model" # Location of the output file
Col_A = [' ', 'Gender', 'Age', 'Weight', 'Height', 'max_HR']
Col_B = [' ', Gender, Age, Weight, Height, max_HR]
Col_A.extend([' '] * (len(power_hc) - len(Col_A)))
Col_B.extend([' '] * (len(power_hc) - len(Col_B)))

writer = pd.ExcelWriter(f'{path}\\{ID}_input_file.xlsx', engine = "openpyxl")
wb = writer.book
df = pd.DataFrame({'P info': Col_A, ' ' : Col_B, 'Heart Rate': HR, 'RPE': RPE, 'Power hc': power_hc, 'Power bc': power_bc})
features_hr = pd.DataFrame([features_hr])
df = pd.concat([df, features_hr], axis = 1)
df = df.fillna(' ')

df.to_excel(writer, index = False)
wb.save(f'{path}\\{ID}_input_file.xlsx')

# If you ever only want to save power in your .csv file
'''csv_file = os.path.join(directory, f"{ID}_filtered_power")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Power hc", "Power bc"])
    rows = zip(Age, Weight, Height, data_filtered, HR, max_HR, RPE, data_filtered_bc)
    writer.writerows(rows)'''


print("File has been succesfully saved!")