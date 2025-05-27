import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

ID = "002"
Age = 26

path = r"C:\Users\maddy\Desktop\NTNU\Roya Doshmanziari - Maddalena_ Riccardo Master projects 2024-2025\Test Arduino\Acquisitions\%s\Zwift" %ID

data_hc = pd.read_csv(f"{path}\\{ID}_tweaked_handcycle.csv", usecols=["Power", "Heart Rate", "Speed", "RPE"])
data_bc = pd.read_csv(f"{path}\\{ID}_bicycle_protocol.csv", usecols=["Power", "Heart Rate", "Speed", "RPE"])

data_hc = data_hc[300 : 840]
data_bc = data_bc[300 : 840]

# Figure 1: Power vs time
plt.figure()
t = np.linspace(0, len(data_hc) - 1, len(data_hc))
plt.plot(t, data_hc["Power"], label = "Handcycle, tweaked")
plt.plot(t, data_bc["Power"], label = "Bicycle, original")

plt.title("Power vs. Time")
plt.xlabel("Time [s]")
plt.ylabel("Power [W]")
plt.legend()
plt.grid()

# Figure 2: Power vs Heart Rate
plt.figure()

x_green = data_hc[:len(data_hc)//3]
y_green = data_bc[:len(data_hc)//3]

x_orange = data_hc[len(data_hc)//3 : 2*len(data_hc)//3]
y_orange = data_bc[len(data_bc)//3 : 2*len(data_bc)//3]

x_red = data_hc[2*len(data_hc)//3:]
y_red = data_bc[2*len(data_bc)//3:]

max_hr_bc = 220 - Age
max_hr_hc = 200 - Age

plt.scatter(x_green["Heart Rate"]/max_hr_hc * 100, x_green["Power"], label = "HC, RPE = 12", color = "green")
plt.scatter(y_green["Heart Rate"]/max_hr_bc * 100, y_green["Power"], label = "BC, RPE = 12", color = "green", marker="*")
plt.scatter(x_orange["Heart Rate"]/max_hr_hc * 100, x_orange["Power"], label = "HC, RPE = 14", color = "orange")
plt.scatter(y_orange["Heart Rate"]/max_hr_bc * 100, y_orange["Power"], label = "BC, RPE = 14", color = "orange", marker="*")
plt.scatter(x_red["Heart Rate"]/max_hr_hc * 100, x_red["Power"], label = "HC, RPE = 15", color = "red")
plt.scatter(y_red["Heart Rate"]/max_hr_bc * 100, y_red["Power"], label = "HC, RPE = 15", color = "red", marker="*")

plt.title("Power vs Heart Rate")
plt.xlabel("Heart Rate [% bpm]")
plt.ylabel("Power [W]")
plt.legend()
plt.grid()

# Figure 3: Speed vs time
plt.figure()
plt.plot(t, data_hc["Speed"], label = "Handcycle, tweaked")
plt.plot(t, data_bc["Speed"], label = "Bicycle, original")

plt.title("Speed vs. Time")
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.legend()
plt.grid()
plt.show()

final = "yake"