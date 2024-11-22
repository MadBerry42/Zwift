import fitparse
import csv 
import os
import pandas as pd
import numpy as np


'''Conversion of a .fit file into a .csv file'''
# Choose before running the code
ID = 8
Setup = 'bicycle'
test = 'protocol'

if ID < 10:
    file_input= f'00{ID}\\Zwift\\00{ID}_{Setup}_{test}.fit' #Path of the input .fit file
    directory = f'00{ID}\\Zwift' #Path of the folder which will contain the output .csv file
    file_output = f'00{ID}_{Setup}_{test}.csv' #Name of the output file (DON'T FORGET .CSV AT THE END!)
    ID_number = f"00{ID}"
else:
    file_input= f'0{ID}\\Zwift\\0{ID}_{Setup}_{test}.fit' #Path of the input .fit file
    directory = f'0{ID}\\Zwift' #Path of the folder which will contain the output .csv file
    file_output = f'0{ID}_{Setup}_{test}.csv' #Name of the output file (DON'T FORGET .CSV AT THE END!)
    ID_number = f"0{ID}"

# Importing the file
fitfile = fitparse.FitFile(file_input)
# Reported RPE values
RPE_Warmup = "Warmup"
RPE_15 = 10
RPE_20 = 12
RPE_35 = 14
RPE_Cooldown = 'Cooldown'

RPE_Warmups = np.linspace(6, 11, 5*60)
RPE_Cooldowns = np.linspace(10, 7, 4 * 60)

# Importing the file
fitfile = fitparse.FitFile(file_input)
# Importing the IMU Data
if Setup == "handcycle":
    IMU_limb = f"{ID_number}_{Setup}_wrist_{test}_processed"
    IMU_crank = f"{ID_number}_{Setup}_crank_{test}_processed"
elif Setup == "bicycle":
    IMU_limb = pd.read_csv(f"{ID_number}\\Processed Data\\{ID_number}_{Setup}_ankle_{test}_processed.csv")
    IMU_crank = pd.read_csv(f"{ID_number}\\Processed Data\\{ID_number}_{Setup}_crank_{test}_processed.csv")
    print("IMU_limb ha dimensione: ", IMU_limb.shape)
    # print("IMU_crank ha dimensione: ", IMU_crank.shape)
    # They have the same dimension, a certain amount of rows and 7 columns


# Initializing the arrays which will contain data for the whole acquisition
timestamps = []
position_lats = []
position_longs = []
heart_rates = []
cadences = []
distances = []
powers = []
speeds = []
altitudes = []
fields = []
RPE = []

c = 0
# Reading the file and storing the values inside the variables
for record in fitfile.get_messages("record"):    
    '''Initializing the variables which will contain their respective value for a single record
    record = one single sample of an acquisition, with all the variables of interest for that timestamp'''
    timestamp = None
    position_lat = None
    position_long = None
    heart_rate = None
    cadence = None
    distance = None
    power = None
    speed = None
    altitude = None

    for data in record:
        if data.name == "timestamp":
            timestamp = data.value
        elif data.name == "position_lat":
            position_lat = (data.value * (180/2**31))           
        elif data.name == "position_long":
            position_long = (data.value * (180/2**31))    
        elif data.name == "heart_rate":
            heart_rate = data.value
        elif data.name == "cadence":
            cadence = data.value
        elif data.name == "distance" and data.value is not None:
            distance = data.value
        elif data.name == "power":
            power = data.value
        elif data.name == "enhanced_speed":
            speed = data.value
        elif data.name == "enhanced_altitude":
            altitude = data.value
        fields.append(data.name)
        
        
    ''' After this for cycle each variable (timestamp, distance, speed, etc) contains a single value, 
    corresponding to the values from one specific timestamp (also called record). Once all of the single 
    values has been stored into their respective variable, these will be added (appended) into their arrays 
    (timestamps, distances, speeds, etc) which at the end of the "for record in fitfile.get_message" 
    loop will contain all of their respective values for the whole acquisition'''

    if timestamp is not None:
        timestamps.append(timestamp)
    if position_lat is not None:
        position_lats.append(position_lat)
    if position_long is not None:
        position_longs.append(position_long)
    if heart_rate is not None:
        heart_rates.append(heart_rate)
    if cadence is not None:
        cadences.append(cadence)
    if distance is not None:
        distances.append(distance)
    if power is not None:
        powers.append(power)
    if speed is not None:
        speeds.append(speed)
    if altitude is not None:
        if c == 0:
            altitudes.append(altitude)
            c = c + 1
        if c > 0:
            altitudes.append(altitude - altitudes[0])
            c = c + 1

print("altitude is: ", type(altitudes), "and its size is:", len(altitudes))
print("c is: ", c)
print("altitude has values: ", altitudes)
        


# Save the data into a .csv file
csv_file = os.path.join(directory, file_output)
print("Il file .fit ha lunghezza", len(timestamps))

if test == "protocol":
    for i in range(0, len(timestamps)):
        if i < 5 * 60: # Warm up range
            RPE.append(RPE_Warmups[i])
            # RPE.append(RPE_Warmup)
        if i >= 5 * 60 and i < 8 * 60: 
            RPE.append("12")
        if i >= 8 * 60 and i < 11 * 60:
            RPE.append("14")
        if i >= 11 * 60 and i < 14 * 60:
            RPE.append("15")
        if i >= 14 * 60 and i < 17 * 60:
            # RPE.append(RPE_15[i - 14*60])
            RPE.append(RPE_15)
        if i >= 17 * 60 and i < 20 * 60:
            RPE.append(RPE_20)
        if i >= 20 *60 and i < 23 * 60:
            RPE.append(RPE_35)
        if i >= 23 * 60 and i < 27 * 60:
            RPE.append(RPE_Cooldowns[i - 23*60]-1)
            # RPE.append(RPE_Cooldown)
        if i >= 27*60:
            RPE.append(RPE_Cooldowns[-1]) # if they give a range
            # RPE.append(RPE_Cooldown)


    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Position lat", "Position long", "Heart Rate", "Cadence", "Distance", "Power", "Speed", "Altitude", "RPE"])
        rows = zip(timestamps, position_lats, position_longs, heart_rates, cadences, distances, powers, speeds, altitudes, RPE)
        writer.writerows(rows)
    
else:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Position lat", "Position long", "Heart Rate", "Cadence", "Distance", "Power", "Speed", "Altitude"])
        rows = zip(timestamps, position_lats, position_longs, heart_rates, cadences, distances, powers, speeds, altitudes)
        writer.writerows(rows)

print(f"Data hase been succesfully written to {csv_file}")
 
# Troubleshooting: if your file as not been saved a .csv file, make sure that file_output contains .csv at the end
