import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

path = f"C:\\Users\\maddalb\\NTNU\\Roya Doshmanziari - Maddalena_ Riccardo Master projects 2024-2025\\Test Arduino\\Workout Data"

data1 = pd.read_csv(f"{path}\\1_tweak.csv", usecols= ["Power"])
data2 = pd.read_csv(f"{path}\\2_tweak.csv", usecols= ["Power"])
data3 = pd.read_csv(f"{path}\\3_tweak.csv", usecols= ["Power"])

data1 = np.array(data1).flatten()
data2 = np.array(data2).flatten()
data3 = np.array(data3).flatten()

hr1 = pd.read_csv(f"{path}\\1_tweak.csv", usecols= ["Heart Rate"])
hr2 = pd.read_csv(f"{path}\\2_tweak.csv", usecols= ["Heart Rate"])
hr3 = pd.read_csv(f"{path}\\3_tweak.csv", usecols= ["Heart Rate"])



'''# Moving average filter
window_size = 30
window = np.ones(window_size) / window_size
window = window.flatten()
data1 = np.convolve(data1["Power"], window, mode = "same")
data2 = np.convolve(data2["Power"], window, mode = "same")
data3 = np.convolve(data3["Power"], window, mode = "same")'''

# Figure 1: Power vs. Time
plt.figure()
t = np.linspace(0, len(data1) - 1, len(data1))
plt.plot(t, data1, label = "Times 1")
t = np.linspace(0, len(data2) - 1, len(data2))
plt.plot(t, data2, label = "Times 2")
t = np.linspace(0, len(data3) - 1, len(data3))
plt.plot(t, data3, label = "Times 3")

plt.legend()
plt.title("Power vs. time")
plt.xlabel("Time [s]")
plt.ylabel("Power [W]")

# Figure 2: Power vs Heart Rate
plt.figure()
max_hr = 220 - 26

x = hr1/max_hr * 100
plt.scatter(x, data1, label = "Times 1")
'''slope, intercept = np.polyfit(x, data1, 1)
plt.plot(x, slope * x + intercept) 
plt.text(30, 28, f"$\\mathbf{{y = {slope[0]:.2f}x + {intercept[0]:.2f}}}$", fontsize = 10)'''
avg_power = np.average(data1)
plt.axhline(avg_power, color = "blue")
plt.text(30, avg_power + 2, f"Average power: {avg_power:.2f}")

x = hr2/max_hr * 100
plt.scatter(x, data2, label = "Times 2")
'''slope, intercept = np.polyfit(x, data2, 1)
plt.plot(x, slope * x + intercept)
plt.text(30, 52, f"$\\mathbf{{y = {slope[0]:.2f}x + {intercept[0]:.2f}}}$", fontsize = 10) '''
avg_power = np.average(data2)
plt.axhline(avg_power, color = "orange")
plt.text(30, avg_power + 2, f"Average power: {avg_power:.2f}")

x = hr3/max_hr * 100
plt.scatter(x, data3, label = "Times 1")
'''slope, intercept = np.polyfit(x, data3, 1)
plt.plot(x, slope * x + intercept) 
plt.text(30, 72, f"$\\mathbf{{y = {slope[0]:.2f}x + {intercept[0]:.2f}}}$", fontsize = 10)'''
avg_power = np.average(data3)
plt.axhline(avg_power, color = "green")
plt.text(30, avg_power + 2, f"Average power: {avg_power:.2f}")

plt.axvline(39.17, color = "black")
plt.grid()

plt.legend()
plt.title("Power vs. Heart Rate")
plt.xlabel("Heart Rate [%]")
plt.ylabel("Power [W]")
plt.xlim([30, 45])

# Figure 3: derivative of power vs time
deriv1 = savgol_filter(data1.flatten(), 100, 4, deriv = 1)
deriv2 = savgol_filter(data2, 100, 4, deriv = 1)
deriv3 = savgol_filter(data3, 100, 4, deriv = 1)

plt.figure()
t = np.linspace(0, len(data1) - 1, len(data1))
plt.plot(t, deriv1, label = "Times 1")
t = np.linspace(0, len(data2) - 1, len(data2))
plt.plot(t, deriv2, label = "Times 2")
t = np.linspace(0, len(data3) - 1, len(data3))
plt.plot(t, deriv3, label = "Times 3")

plt.legend()
plt.title("Derivative of Power vs. Time")
plt.xlabel("Time [s]")
plt.ylabel("Derivative of Power [W/s]")
plt.show()

final = "bro"