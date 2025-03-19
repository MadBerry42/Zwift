import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import os
import csv

# Importing IMU data
ID = 2
test = 'protocol' # FTP or protocol
fs = 200 # Data provided by the manifacturer

IMU_or = pd.read_csv(f"C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\000\\IMU Data\\000_bicycle_protocol_ankle.csv")
IMU = IMU_or.to_numpy()

# Converting timestamps into seconds
timestamps = IMU[:, 0]

for i in range (1, len(timestamps)):
    IMU[i, 0] = (timestamps[i] - timestamps[0])/10**6
IMU[0, 0] = 0

if ID == 11:
    cutting_point = int(round(1748.439 * fs)) # already evaluated
else: 
    print("Insert the x value for the end of the handcycle signal")
    # Visualizing data and finding the cutting point
    fig, ax = plt.subplots()
    lines = ax.plot(IMU[:, 0], IMU[:, 1])
    mplcursors.cursor(lines, hover = 'True')
    plt.show()

    cutting_point = int(round(float(input()) * fs))


# Creating the two different signals
handcycle = IMU[0 : cutting_point - (fs*3), :]
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(handcycle[:, 0], handcycle[:, 1])

bicycle = IMU[cutting_point + 1 :, :]
ax2.plot(bicycle[:, 0], bicycle[:, 1])
plt.show()


# Creating a .csv file containing the new signal
# Handcycle
if ID < 10:
    directory = f"00{ID}\\IMU Data"
    file_output = f"00{ID}_handcycle_{test}_wrist.csv"
elif ID >= 10:
    directory = f"0{ID}\\IMU Data"
    file_output = f"0{ID}_handcycle_{test}_wrist.csv"
csv_file = os.path.join(directory, file_output)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "accX[g]", "accY[g]", "AccZ[g]", "gyrX[dps]", "gyrY[dps]", "gyrZ[dps]"])
    writer.writerows(handcycle)

# Bicycle
if ID < 10:
    directory = f"00{ID}\\IMU Data"
    file_output = f"00{ID}_bicycle_{test}_ankle.csv"
elif ID >= 10:
    directory = f"0{ID}\\IMU Data"
    file_output = f"0{ID}_bicycle_{test}_ankle.csv"
csv_file = os.path.join(directory, file_output)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "accX[g]", "accY[g]", "AccZ[g]", "gyrX[dps]", "gyrY[dps]", "gyrZ[dps]"])
    writer.writerows(bicycle)
