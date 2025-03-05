import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import os
import csv

# Importing IMU data
ID = 16
test = 'protocol' # FTP or protocol
setup = 'handcycle' # handcycle or bicycle

if ID < 10:
    ID = f"00{ID}"
elif ID >= 10:
    ID = f"0{ID}"

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

path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol"
IMU_1_or = pd.read_csv(f"{path}\\{ID}\\IMU Data\\{ID}_{setup}_{test}_crank.csv")
IMU_1 = IMU_1_or.to_numpy()
IMU_2_or = pd.read_csv(f"{path}\\{ID}\\IMU Data\\{ID}_{setup}_{test}_{limb}.csv")
IMU_2 = IMU_2_or.to_numpy()


# Converting timestamps into seconds
timestamps_1 = IMU_1[:, 0]
timestamps_2 = IMU_2[:, 0]

for i in range (1, len(timestamps_1)):
    IMU_1[i, 0] = (timestamps_1[i] - timestamps_1[0])/10**6
IMU_1[0, 0] = 0

if ID != '011': # 011 was previosly split into handcycle and bicycle, so the timestamps do not require any modifications
    for i in range (1, len(timestamps_2)):
        IMU_2[i, 0] = (timestamps_2[i] - timestamps_2[0])/10**6
    IMU_2[0, 0] = 0

#-----------------------------------------------------------------------------------------------------------------------------------------
    # Visualizing Data and (if necessary) cut it
#-----------------------------------------------------------------------------------------------------------------------------------------
fig, (ax0, ax01) = plt.subplots(2)
ax0.plot(IMU_1[:, 0], IMU_1[:, 1])
ax0.set_title("Crank sensor")
ax01.plot(IMU_2[:, 0], IMU_2[:, 1])
ax01.set_title(f"Sensor on the {limb}")
plt.show()

print("Do you need to cut split the signal? (yes/no)")
ans = input()
if ans == 'yes':
    print("Which signal do you want to cut? (1/2/both)")
    ans1 = input()
    if ans1 == '1':
        sig = IMU_1 
    elif ans1 == '2':
        sig = IMU_2
    elif ans1 == 'both':
        sig1 = IMU_1
        sig2 = IMU_2
        sig1 = sig1[round(max(sig1[:, 0])) : -1, :]
        sig2 = sig2[round(max(sig2[:, 0])) : -1, :]
        fig, axs = plt.subplots(2)
        axs[0].plot(sig1[:, 1])
        axs[1].plot(sig2[:, 1]) 
        mplcursors.cursor(axs, hover = 'True')
        mplcursors.cursor(axs, hover = 'True')
        
        plt.show()

        print("Where do you want to cut the first signal?")
        cut = int(input())
        sig1 = sig1[cut : -1, :]    
    

        print("Where do you want to cut the second signal?")
        cut = int(input())
        sig2 = sig2[cut : -1, :]

        IMU_1 = sig1
        IMU_2 = sig2

        fig, axs = plt.subplots(2)
        axs[0].plot(sig1[:, 1])
        axs[1].plot(sig2[:, 1]) 

        for i in range(0, min(len(IMU_1), len(IMU_2))):
            offset = IMU_1[0, 0]
            IMU_1[i, 0] = (IMU_1[i, 0] - offset)
            offset = IMU_2[0, 0]
            IMU_2[i, 0] = (IMU_2[i, 0] - IMU_2[0, 0])
        

    if ans1 == 1 or ans1 == 2:
        sig = sig[round(max(sig[:, 0])) : -1, :]
        plt.plot(sig[:, 1])
        plt.show()

        print("Where do you want to cut?")
        ans = float(input())

        sig = sig[round(ans) : -1]
        
        plt.plot(sig[:, 1])
        plt.show()

    if ans1 == '1':
        IMU_1 = sig
    elif ans1 == '2':
        IMU_2 = sig

    
    

#-------------------------------------------------------------------------------------------------------------------------------------------
    # Rialigning the signals (fixing the dealy)
#--------------------------------------------------------------------------------------------------------------------------------------------
# Visualizing data and finding the delay
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(IMU_1[200:2000, 0], IMU_1[200:2000, 1])
ax1.set_title("Crank sensor")
ax2.plot(IMU_2[200:2000, 0], IMU_2[200:2000, 1])
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



# Synchronisation
if ans == 'no':
    if diff < 0: # Il primo segnale è in anticipo
        IMU_2 = IMU_2[abs(diff):, :]
        for i in range(1, min(len(IMU_2), len(IMU_2))):
            IMU_2[i, 0] = (IMU_2[i, 0] - IMU_2[0, 0])

    if diff > 0:
        IMU_1 = IMU_1[abs(diff):, :]
        for i in range(1, min(len(IMU_1), len(IMU_2))):
            IMU_1[i, 0] = (IMU_1[i, 0] - IMU_1[0, 0])

if ans == 'yes':
    if diff < 0: # Il primo segnale è in anticipo
        IMU_1 = IMU_1[abs(diff):, :]
        for i in range(0, min(len(IMU_2), len(IMU_2))):
            IMU_1[i, 0] = (IMU_1[i, 0] - IMU_1[0, 0])

    if diff > 0:
        IMU_1 = IMU_1[abs(diff):, :]
        for i in range(0, min(len(IMU_1), len(IMU_2))):
            IMU_1[i, 0] = (IMU_1[i, 0] - IMU_1[0, 0])
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


#-----------------------------------------------------------------------------------------------------------------------------------------
    # Cutting off warmup and cooldown
#-----------------------------------------------------------------------------------------------------------------------------------------
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

IMU_1 = IMU_1[samples + warmup : tail - cooldown, :]
IMU_2 = IMU_2[samples + warmup : tail - cooldown, :]
# print("IMU Data are:", IMU_1, IMU_2)
# print("IMU Data sizes are:", IMU_1.shape)

#------------------------------------------------------------------------------------------------------------------------------------------
    # Exporting data
#------------------------------------------------------------------------------------------------------------------------------------------
# Creating a .csv file containing the new signal
# Crank
path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\IMU Data processing\\Processed Data"
file_output = f"{path}\\{ID}_{setup}_{test}_crank_processed.csv"
directory = f"{path}"
csv_file = os.path.join(directory, file_output)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "accX[g]", "accY[g]", "AccZ[g]", "gyrX[dps]", "gyrY[dps]", "gyrZ[dps]"])
    writer.writerows(IMU_1)

# Limb
directory = f"{path}"
file_output = f"{path}\\{ID}_{setup}_{test}_{limb}_processed.csv"
csv_file = os.path.join(directory, file_output)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "accX[g]", "accY[g]", "AccZ[g]", "gyrX[dps]", "gyrY[dps]", "gyrZ[dps]"])
    writer.writerows(IMU_2)

print("Data have been succesfully processed!")



# Visualizing results
fig, (ax1, ax2) = plt.subplots(2)
processed_1 = pd.read_csv(f"{path}\\{ID}_{setup}_{test}_crank_processed.csv")
processed_2 = pd.read_csv(f"{path}\\{ID}_{setup}_{test}_{limb}_processed.csv")

processed_1 = processed_1.to_numpy()
processed_2 = processed_2.to_numpy()

timestamps_1 = processed_1[:, 0]
timestamps_2 = processed_2[:, 0]

ax1.plot(timestamps_1, processed_1[:, 1])
ax2.plot(timestamps_2, processed_2[:, 1])

plt.tight_layout()
fig.suptitle("Aligned and cleaned signals")
plt.show()
final = "boh"