import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import os
import csv

# Importing IMU data
ID = 8
test = 'protocol' # FTP or protocol
setup = 'bicycle' # handcycle or bicycle

if setup == 'bicycle':
    limb = 'ankle'
elif setup == 'handcycle':
    limb = 'wrist'

fs = 200 # Sample frequency of the IMU sensors = 200 Hz. Value provided by the manifacturer
if test == 'FTP':
    warmup = 60 * 8 * fs #warmup in the FTP test lasts 8 minutes
    cooldown = 60 * 10 * fs
elif test == 'protocol':
    warmup = 60 * 5 * fs # warmup in the protocol test lasts 5 minutes
    cooldown = 60 * 4 * fs

if ID < 10:
    IMU_1_or = pd.read_csv(f"00{ID}\\IMU Data\\00{ID}_{setup}_crank_{test}.csv")
    IMU_1 = IMU_1_or.to_numpy()
    IMU_2_or = pd.read_csv(f"00{ID}\\IMU Data\\00{ID}_{setup}_{limb}_{test}.csv")
    IMU_2 = IMU_2_or.to_numpy()
elif ID >= 10:
    IMU_1_or = pd.read_csv(f"0{ID}\\IMU Data\\0{ID}_{setup}_crank_{test}.csv")
    IMU_1 = IMU_1_or.to_numpy()
    IMU_2_or = pd.read_csv(f"0{ID}\\IMU Data\\0{ID}_{setup}_{limb}_{test}.csv")
    IMU_2 = IMU_2_or.to_numpy()




# Converting timestamps into seconds
timestamps_1 = IMU_1[:, 0]
timestamps_2 = IMU_2[:, 0]

for i in range (1, len(timestamps_1)):
    IMU_1[i, 0] = (timestamps_1[i] - timestamps_1[0])/10**6
IMU_1[0, 0] = 0

for i in range (1, len(timestamps_2)):
    IMU_2[i, 0] = (timestamps_2[i] - timestamps_2[0])/10**6
IMU_2[0, 0] = 0



# Visualizing data and finding the delay
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(IMU_1[200:1000, 0], IMU_1[200:1000, 1])
ax1.set_title("Crank sensor")
ax2.plot(IMU_2[200:1000, 0], IMU_2[200:1000, 1])
ax2.set_title(f"Sensor on the {limb}")
mplcursors.cursor(ax1, hover = 'True')
mplcursors.cursor(ax2, hover = 'True')

fig.suptitle("Select two peaks to find the delay")
plt.tight_layout()
plt.show()

print("Insert the x value for the peak chosen for the first signal")
first_sensor = float(input())
print("Insert the x value for the peak chosen for the second signal")
second_sensor = float(input())

diff = round((first_sensor - second_sensor) * fs)
print(f"Signals are off by {diff} samples")



# Synchronisation (realigning the signals)
if diff > 0: # Il primo segnale è in anticipo
    IMU_1 = IMU_1[diff:, :]
    for i in range(1, len(IMU_1)):
        IMU_1[i, 0] = (IMU_1[i, 0] - IMU_1[0, 0])


if diff < 0:
    IMU_2 = IMU_2[abs(diff):, :]
    for i in range(1, len(IMU_1)):
        IMU_2[i, 0] = (IMU_2[i, 0] - IMU_2[0, 0])

# Check if signals are actually aligned
fig2, (ax1, ax2) = plt.subplots(2)
ax1.plot(IMU_1[200:1000, 0], IMU_1[200:1000, 1])
ax1.set_title("Crank sensor")
ax2.plot(IMU_2[200:1000, 0], IMU_2[200:1000, 1])
ax2.set_title(f"Sensor on the {limb}")
mplcursors.cursor(ax1, hover = 'True')
mplcursors.cursor(ax2, hover = 'True')

fig2.suptitle("Check if signals are correctly aligned")
plt.tight_layout()
plt.show()



# Cutting off warmup and cooldown
# Select the beginning of cycling phase 
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(IMU_1[0:round(len(IMU_1)/2), 0], IMU_1[0:round(len(IMU_1)/2), 1])
ax1.set_title("Crank sensor, first half")
ax2.plot(IMU_2[round(len(IMU_1)/2):, 0], IMU_2[round(len(IMU_1)/2):, 1])
ax1.set_title("Crank sensor, last half")
mplcursors.cursor(ax1, hover = 'True')
mplcursors.cursor(ax2, hover = 'True')

fig.suptitle("Select beginning and end of cycling phase")
plt.tight_layout()
plt.show()

print("Insert the x value for the beginning of the cycling phase")
samples = round(float(input()) * fs)
print("Insert the x value for the end of the cycling phase")
tail = round(float(input()) * fs)

IMU_1 = IMU_1[samples:tail, :]
IMU_2 = IMU_2[samples:tail, :]
# print("IMU Data are:", IMU_1, IMU_2)
# print("IMU Data sizes are:", IMU_1.shape)


# Creating a .csv file containing the new signal
# Crank
file_output = f"00{ID}_{setup}_crank_{test}_processed.csv"
directory = f"00{ID}\\Processed Data"
csv_file = os.path.join(directory, file_output)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "accX[g]", "accY[g]", "AccZ[g]", "gyrX[dps]", "gyrY[dps]", "gyrZ[dps]"])
    writer.writerows(IMU_1)

# Limb
directory = f"00{ID}\\Processed Data"
file_output = f"00{ID}_{setup}_{limb}_{test}_processed.csv"
csv_file = os.path.join(directory, file_output)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "accX[g]", "accY[g]", "AccZ[g]", "gyrX[dps]", "gyrY[dps]", "gyrZ[dps]"])
    writer.writerows(IMU_2)

print("Data have been succesfully processed!")



# Visualizing results
fig, (ax1, ax2) = plt.subplots(2)
processed_1 = pd.read_csv(f"00{ID}\\Processed Data\\00{ID}_{setup}_crank_{test}_processed.csv")
processed_2 = pd.read_csv(f"00{ID}\\Processed Data\\00{ID}_{setup}_{limb}_{test}_processed.csv")

processed_1 = processed_1.to_numpy()
processed_2 = processed_2.to_numpy()

timestamps_1 = processed_1[:, 0]
timestamps_2 = processed_2[:, 0]

ax1.plot(timestamps_1, processed_1[:, 1])
ax2.plot(timestamps_2, processed_2[:, 1])

plt.tight_layout()
fig.suptitle("Aligned and cleaned signals")
plt.show()