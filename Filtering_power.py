import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

# Import data and select the portion with fixed RPE
setup = "handcycle"
ID = 10
if ID < 10:
    ID = f"00{ID}"
else:
    ID = f"0{ID}"

data = pd.read_csv(f"C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\{ID}\\Zwift\\{ID}_{setup}_protocol.csv", usecols = ["Power"])
data = np.array(data[299: 839])

# Moving average filter
window_size = 50 # Average computed on a 3-second window; the bigger the window, the more the output is smoothed
window = np.ones(window_size) / window_size

window = window.flatten()
data = data.flatten()
data_filtered = np.convolve(data, window, mode = "same")


t = np.linspace(300, 840, num = len(data))

# Two subplots
'''fig, ax = plt.subplots(2)
ax[0].plot(t, data)
ax[0].set_title("Original signal")
ax[1].plot(t, data_filtered)
ax[1].set_title("Filtered signal")'''

# One plot, signals overlapped
plt.figure()
plt.plot(t, data, label = "Original signal")
plt.plot(t, data_filtered, label = "Filtered signal")
plt.xlabel("Time [s]")
plt.ylabel("Power output [W]")
plt.title(f"Subject {ID}")

plt.legend()
# plt.show()

# Save data in a .csv file
directory = f"Acquisitions\\Protocol\\Processed data\\Filtered Power"
file_output = f"{ID}_{setup}_filtered_power.csv"
csv_file = os.path.join(directory, file_output)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Power"])
    writer.writerows(data_filtered)
