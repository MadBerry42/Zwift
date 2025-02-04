import numpy as np
<<<<<<< HEAD
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from scipy.stats import f
import math

# Importing data
ID = 4
bicycle = pd.read_csv(f"Csv files\\00{ID}_bicycle.csv", usecols = ["Timestamp", "Heart Rate", "Cadence", "Power", "Speed"])
handcycle = pd.read_csv(f"Csv files\\00{ID}_handcycle.csv", usecols = ["Timestamp", "Heart Rate", "Cadence", "Power", "Speed"])

# Mean and standard deviation: Heart Rate
bc = bicycle.iloc[:, 1] # x, our gold standard
hc = handcycle.iloc[:, 1] # y, method that requires validation
bc = bc.astype(int)
hc = hc.astype(int)
bc_mean = statistics.mean(bc)
hc_mean = statistics.mean(hc)
bc_stdev = statistics.stdev(bc)
hc_stdev = statistics.stdev(hc)
# OLP line
b = hc_stdev/bc_stdev
a = hc_mean - b*bc_mean

if len(bc) < len(hc):
    hc = hc[len(hc) - len(bc) : len(hc)]
if len(bc) > len(hc):
    bc = bc[len(bc) - len(hc) : len(bc)]

coefficients = np.polyfit(bc, hc, 1)
y_fit = np.polyval(coefficients, bc)

fig1, axs1 = plt.subplots(2, 2)
fig1.suptitle(f"Acquisition {ID}")
plt.tight_layout()
axs1[0, 0].scatter(bc, b * hc + a, s = 10, label = "Data")
axs1[0, 0].plot(bc, bc, color = "r", label = "Bisecant")
axs1[0, 0].plot(bc, y_fit, color = "g", label = "Interpolating line")
axs1[0, 0].legend(loc = "upper left", fontsize = "small")
axs1[0, 0].set_title(f"Heart rate: a is {a:.2f}, b is {b:.2f}")

# Pearson's coefficient: Heart rate
R = np.corrcoef(bc, hc)
r = R[0, 1]
print("Pearson's coefficient for Heart Rate is: ", r)

# Bland Altmann-Plot: Heart Rate
alpha = 0.05
n = len(bc)
B = f.ppf(1 - alpha, 1, n-2) * (1 - r**2)/(n-2)
CI_b = [b * math.sqrt(B + 1) - math.sqrt(B), b * math.sqrt(B + 1) + math.sqrt(B)]
CI_a = [hc_mean - b * (math.sqrt(B + 1) + math.sqrt(B))*bc_mean, hc_mean - b*(math.sqrt(B+1) - math.sqrt(B)) * bc_mean]
print("CI_a for HR is", CI_a, "CI_b for HR is", CI_b)

Sxy = np.std(bc - hc)
dxy = np.mean(bc-hc)
lim_sup = dxy + 2 * Sxy
lim_inf = dxy - 2 * Sxy

fig2, axs2 = plt.subplots(2, 2)
fig2.suptitle(f"Acquisition {ID}")
axs2[0, 0].plot((bc+hc)/2, (bc-hc)/2, '*')
axs2[0, 0].axhline(y = dxy, color = "b")
axs2[0, 0].axhline(y = lim_sup, linestyle = "-.")
axs2[0, 0].axhline(y = lim_inf, linestyle = "-.")
axs2[0, 0].set_title("Heart Rate")

y = b * hc + a
new_data_HR = pd.DataFrame({'Data': 'HR', 'ID': [ID], 'Pearson r': [round(r, 2)], 'a': [round(a, 2)], 'b': [round(b, 2)], 'CI_a': [[round(x, 2) for x in CI_a]], 'CI_b': [[round(x, 2) for x in CI_b]]})
BA_data_HR = pd.DataFrame({'ID': [ID], 'Data': 'HR', 'Standard deviation': [Sxy], 'Mean Difference': [dxy]})


# Mean and standard deviation: Power
bc = bicycle.iloc[:, 3] # x, our gold standard
hc = handcycle.iloc[:, 3] # y, method that requires validation
bc = bc.astype(int)
hc = hc.astype(int)
bc_mean = statistics.mean(bc)
hc_mean = statistics.mean(hc)
bc_stdev = statistics.stdev(bc)
hc_stdev = statistics.stdev(hc)
# OLP line
b = hc_stdev/bc_stdev
a = hc_mean - b*bc_mean

if len(bc) < len(hc):
    hc = hc[len(hc) - len(bc) : len(hc)]
if len(bc) > len(hc):
    bc = bc[len(bc) - len(hc) : len(bc)]

coefficients = np.polyfit(bc, hc, 1)
y_fit = np.polyval(coefficients, bc)

axs1[0, 1].scatter(bc, b * hc + a, s = 10, label = "Data")
axs1[0, 1].plot(bc, bc, color = "r", label = "Bisecant")
axs1[0, 1].plot(bc, y_fit, color = "g", label = "Interpolating line")
axs1[0, 1].legend(loc = "upper left", fontsize = "small")
axs1[0, 1].set_title(f"Power: a is {a:.2f}, b is {b:.2f}")

# Pearson's coefficient: Power
R = np.corrcoef(bc, hc)
r = R[0, 1]
print("Pearson's coefficient for Power is: ", r)

# Bland Altmann-Plot: Power
alpha = 0.05
n = len(bc)
B = f.ppf(1 - alpha, 1, n-2) * (1 - r**2)/(n-2)
CI_b = [b * math.sqrt(B + 1) - math.sqrt(B), b * math.sqrt(B + 1) + math.sqrt(B)]
CI_a = [hc_mean - b * (math.sqrt(B + 1) + math.sqrt(B))*bc_mean, hc_mean - b*(math.sqrt(B+1) - math.sqrt(B)) * bc_mean]
print("CI_a for Power is", CI_a, "CI_b for Power is", CI_b)

Sxy = np.std(bc - hc)
dxy = np.mean(bc-hc)
lim_sup = dxy + 2 * Sxy
lim_inf = dxy - 2 * Sxy

axs2[0, 1].plot((bc+hc)/2, (bc-hc)/2, '*')
axs2[0, 1].axhline(y = dxy, color = "b")
axs2[0, 1].axhline(y = lim_sup, linestyle = "-.")
axs2[0, 1].axhline(y = lim_inf, linestyle = "-.")
axs2[0, 1].set_title("Power")

new_data_power = pd.DataFrame({'Data': 'Power', 'Pearson r': [round(r, 2)], 'a': [round(a, 2)], 'b': [round(b, 2)], 'CI_a': [[round(x, 2) for x in CI_a]], 'CI_b': [[round(x, 2) for x in CI_b]],})
BA_data_power = pd.DataFrame({'Data': 'Power', 'Standard deviation': [Sxy], 'Mean Difference': [dxy]})


# Mean and standard deviation: Speed
bc = bicycle.iloc[:, 4] # x, our gold standard
hc = handcycle.iloc[:, 4] # y, method that requires validation
bc = bc.astype(int)
hc = hc.astype(int)
bc_mean= statistics.mean(bc)
hc_mean = statistics.mean(hc)
bc_stdev = statistics.stdev(bc)
hc_stdev = statistics.stdev(hc)
# OLP line
b = hc_stdev/bc_stdev
a = hc_mean - b*bc_mean

if len(bc) < len(hc):
    hc = hc[len(hc) - len(bc) : len(hc)]
if len(bc) > len(hc):
    bc = bc[len(bc) - len(hc) : len(bc)]

coefficients = np.polyfit(bc, hc, 1)
y_fit = np.polyval(coefficients, bc)

axs1[1, 0].scatter(bc, b * hc + a, s = 10, label = "Data")
axs1[1, 0].plot(bc, bc, color = "r", label = "Bisecant")
axs1[1, 0].plot(bc, y_fit, color = "g", label = "Interpolating line")
axs1[1, 0].legend(loc = "upper left", fontsize = "small")
axs1[1, 0].set_title(f"Speed: a is {a:.2f}, b is {b:.2f}")
# WTF dude, honestly

# Pearson's coefficient: Speed
R = np.corrcoef(bc, hc)
r = R[0, 1]
print("Pearson's coefficient for Speed is: ", r)

# Bland Altmann-Plot: Speed
alpha = 0.05
n = len(bc)
B = f.ppf(1 - alpha, 1, n-2) * (1 - r**2)/(n-2)
CI_b = [b * math.sqrt(B + 1) - math.sqrt(B), b * math.sqrt(B + 1) + math.sqrt(B)]
CI_a = [hc_mean - b * (math.sqrt(B + 1) + math.sqrt(B))*bc_mean, hc_mean - b*(math.sqrt(B+1) - math.sqrt(B)) * bc_mean]
print("CI_a for Speed is", CI_a, "CI_b for Speed is", CI_b)

Sxy = np.std(bc - hc)
dxy = np.mean(bc-hc)
lim_sup = dxy + 2 * Sxy
lim_inf = dxy - 2 * Sxy

axs2[1, 0].plot((bc+hc)/2, (bc-hc)/2, '*')
axs2[1, 0].axhline(y = dxy, color = "b")
axs2[1, 0].axhline(y = lim_sup, linestyle = "-.")
axs2[1, 0].axhline(y = lim_inf, linestyle = "-.")
axs2[1, 0].set_title("Speed")

new_data_speed = pd.DataFrame({'Data': 'Speed', 'Pearson r': [round(r, 2)], 'a': [round(a, 2)], 'b': [round(b, 2)], 'CI_a': [[round(x, 2) for x in CI_a]], 'CI_b': [[round(x, 2) for x in CI_b]],})
BA_data_speed = pd.DataFrame({'Data': 'Speed', 'Standard deviation': [Sxy], 'Mean Difference': [dxy]})


# Mean and standard deviation: Cadence
bc = bicycle.iloc[:, 2] # x, our gold standard
hc = handcycle.iloc[:, 2] # y, method that requires validation
bc = bc.astype(int)
hc = hc.astype(int)
bc_mean= statistics.mean(bc)
hc_mean = statistics.mean(hc)
bc_stdev = statistics.stdev(bc)
hc_stdev = statistics.stdev(hc)
# OLP line
b = hc_stdev/bc_stdev
a = hc_mean - b*bc_mean

if len(bc) < len(hc):
    hc = hc[len(hc) - len(bc) : len(hc)]
if len(bc) > len(hc):
    bc = bc[len(bc) - len(hc) : len(bc)]

coefficients = np.polyfit(bc, hc, 1)
y_fit = np.polyval(coefficients, bc)

axs1[1, 1].scatter(bc, b * hc + a, s = 10, label = "Data")
axs1[1, 1].plot(bc, bc, color = "r", label = "Bisecant")
axs1[1, 1].plot(bc, y_fit, color = "g", label = "Interpolating line")
axs1[1, 1].legend(loc = "upper left", fontsize = "small")
axs1[1, 1].set_title(f"Cadence: a is {a:.2f}, b is {b:.2f}")
# WTF dude, honestly

# Pearson's coefficient: Cadence
R = np.corrcoef(bc, hc)
r = R[0, 1]
print("Pearson's coefficient for Cadence is: ", r)

# Bland Altmann-Plot: Cadence
alpha = 0.05
n = len(bc)
B = f.ppf(1 - alpha, 1, n-2) * (1 - r**2)/(n-2)
CI_b = [b * math.sqrt(B + 1) - math.sqrt(B), b * math.sqrt(B + 1) + math.sqrt(B)]
CI_a = [hc_mean - b * (math.sqrt(B + 1) + math.sqrt(B))*bc_mean, hc_mean - b*(math.sqrt(B+1) - math.sqrt(B)) * bc_mean]
print("CI_a for Cadence is", CI_a, "CI_b for Cadence is", CI_b)

Sxy = np.std(bc - hc)
dxy = np.mean(bc-hc)
lim_sup = dxy + 2 * Sxy
lim_inf = dxy - 2 * Sxy

axs2[1, 1].plot((bc+hc)/2, (bc-hc)/2, '*')
axs2[1, 1].axhline(y = dxy, color = "b")
axs2[1, 1].axhline(y = lim_sup, linestyle = "-.")
axs2[1, 1].axhline(y = lim_inf, linestyle = "-.")
axs2[1, 1].set_title("Cadence")

new_data_cadence = pd.DataFrame({'Data': 'Cadence', 'Pearson r': [round(r, 2)], 'a': [round(a, 2)], 'b': [round(b, 2)], 'CI_a': [[round(x, 2) for x in CI_a]], 'CI_b': [[round(x, 2) for x in CI_b]],})
BA_data_cadence = pd.DataFrame({'Data': 'Cadence', 'Standard deviation': [Sxy], 'Mean Difference': [dxy]})


# Work on the Excel file
file = "Csv files\\Data from Acquisitions.xlsx"
excel = pd.read_excel(file, engine = 'openpyxl')

if ID not in excel["ID"].values:
    # df_HR = df_HR.dropna(axis=1, how='all')          
    # OLP lines
    excel = pd.read_excel(file, sheet_name = "OLP", engine = 'openpyxl')
    OLP = pd.concat([excel, new_data_HR, new_data_power, new_data_speed, new_data_cadence], ignore_index=True)

    # Bland Altmann plot data
    excel_BA = pd.read_excel(file, sheet_name = "Bland-Altmann", engine = 'openpyxl')
    BA = pd.concat([excel_BA, BA_data_HR, BA_data_power, BA_data_speed, BA_data_cadence], ignore_index = True)

    # Add data to the excel file
    writer = pd.ExcelWriter(file, engine = 'openpyxl')
    OLP.to_excel(writer, sheet_name = "OLP", index = False)
    BA.to_excel(writer, sheet_name = "Bland-Altmann", index = False)
    writer.close()

    print("Data has been successfully added to the Excel file")

else:
    print("Data already in the file")

# plt.tight_layout()
# plt.show()
=======

aerobic = [3, 4, 5]
anaerobic = [1, 2, 4]

R = np.corrcoef(aerobic, anaerobic)
r_pearson = R[0, 1]

# print(R)
print("Pearson's coefficient is: ", r_pearson)

>>>>>>> 2ae72e98d1d531da6c3fb86d1b8bd28226ed8c58
