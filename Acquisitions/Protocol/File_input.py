# Creating the input file for the model
import pandas as pd
from Extract_HR_Features import Extract_HR_Features
from ExtractIMU_Features import ExtractIMU_Features
import math
ID = 3
setup = 'handcycle' # handcycle or bicycle

# Personal info
gender = "M" # M or F
weight = 70 # kg
height = 1.80 # m
age = 22 # years
# max_hr = 220 - age
max_hr = 190 # if tested
p_info = [{"gender": gender,  "weight": weight, "age": age, "height": height, "max hr": max_hr}]
p_data = pd.DataFrame.from_dict(p_info)

if setup == 'handcycle':
    limb = "wrist"
elif setup == "bicycle":
    limb = "ankle"
test = 'protocol' # ftp or protocol

if ID < 10:
    ID = f"00{ID}"
elif ID >= 10:
    ID = f"0{ID}"

# Importing data
IMU_limb = pd.read_csv(f"Processed Data\\{ID}\\{ID}_{setup}_{test}_{limb}_processed.csv")
IMU_crank = pd.read_csv(f"Processed Data\\{ID}\\{ID}_{setup}_{test}_crank_processed.csv")
Zwift = pd.read_csv(f"{ID}\\Zwift\\{ID}_{setup}_{test}.csv")


# Extracting features: 30 seconds windows
length_win = 30 # s
n_windows = int(math.floor(len(Zwift)/length_win))


# Heart Rate
hr = Zwift.iloc[:, 3]
hr_dict = []
hr_data = Extract_HR_Features(hr, length_win, max_hr)
print(type(hr_data))



# IMU Data
fs = 200 # Data provided by the manifacturer
IMU_limb = ExtractIMU_Features(IMU_limb, length_win*fs, False)
IMU_crank = ExtractIMU_Features(IMU_crank, length_win*fs, False)



# Power




# Create the Excel file
writer = pd.ExcelWriter(f"Data\\{ID}_{setup}_data.xlsx", engine = "xlsxwriter")
p_data.to_excel(writer, sheet_name = "P info")
IMU_limb.to_excel(writer, sheet_name = f"IMU {limb}")
IMU_crank.to_excel(writer, sheet_name = f"IMU crank")
hr_data.to_excel(writer, sheet_name = "HR")

writer.close()


