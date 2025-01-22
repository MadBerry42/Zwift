import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# Import data and select the portion with fixed RPE
setup = "handcycle"

# Participant details
ID = 3
Gender = 0 #0 for males, 1 for females
Age = 22
Weight = 70
Height = 180
max_HR = 199


if ID < 10:
    ID = f"00{ID}"
else:
    ID = f"0{ID}"

# Handcycle
data = pd.read_csv(f"C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\{ID}\\Zwift\\{ID}_{setup}_protocol.csv")
data = data[300:840]
power_hc = np.array(data["Power"])
HR = np.array(data["Heart Rate"])
RPE = np.array(data["RPE"])
# Moving average filter
window_size = 50 # Average computed on a 3-second window; the bigger the window, the more the output is smoothed
window = np.ones(window_size) / window_size
window = window.flatten()
power_hc = power_hc.flatten()
data_filtered = np.convolve(power_hc, window, mode = "same")
# Plotting the signal
t = np.linspace(300, 840, num = len(power_hc))
plt.figure()
plt.plot(t, data["Power"])
plt.plot(t, data_filtered)
plt.xlabel("Time [s]")
plt.ylabel("Power output [W]")
plt.title(f"Subject {ID}, {setup}")

plt.legend("Original signal", "Filtered signal")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Bicycle
#-----------------------------------------------------------------------------------------------------------------------------------------
setup = "bicycle"
data = pd.read_csv(f"C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\{ID}\\Zwift\\{ID}_{setup}_protocol.csv", usecols = ["Power"])
data = data[300:840]
power_bc = np.array(data)
# Moving average filter
window_size = 50 # Average computed on a 3-second window; the bigger the window, the more the output is smoothed
window = np.ones(window_size) / window_size
window = window.flatten()
power_bc = power_bc.flatten()
data_filtered_bc = np.convolve(power_bc, window, mode = "same")


# Two subplots
'''fig, ax = plt.subplots(2)
ax[0].plot(t, data)
ax[0].set_title("Original signal")
ax[1].plot(t, data_filtered)
ax[1].set_title("Filtered signal")'''

# One plot, signals overlapped
plt.figure()
plt.plot(t, data["Power"], label = "Original signal")
plt.plot(t, data_filtered_bc, label = "Filtered signal")
plt.xlabel("Time [s]")
plt.ylabel("Power output [W]")
plt.title(f"Subject {ID}, {setup}")

plt.legend()
plt.show()

# Save data in a .csv file
directory = f"Acquisitions\\Protocol\\Processed data\\Input to model"
file_output = f"{ID}_input_file.csv"
csv_file = os.path.join(directory, file_output)

Age = np.ones(540) * Age
Weight = np.ones(540) * Weight
Height = np.ones(540) * Height
max_HR = np.ones(540) * max_HR
Gender = np.ones(540) * Gender

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Age", "Weight", "Height", "Gender", "Power hc", "Heart Rate", "Max HR", "RPE", "Power bc"])
    rows = zip(Age, Weight, Height, Gender, data_filtered, HR, max_HR, RPE, data_filtered_bc)
    writer.writerows(rows)

'''# If you ever only want to save power in your .csv file
csv_file = os.path.join(directory, f"{ID}_filtered_power")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Power hc", "Power bc"])
    rows = zip(Age, Weight, Height, data_filtered, HR, max_HR, RPE, data_filtered_bc)
    writer.writerows(rows)'''

print("File has been succesfully saved!")