'''This code gets in input the csv file containing data about a specific acquisition and as an outut gives back a 
.xslx file containing the paramenters for the model P_bc = alpha P_hc + beta + ...'''

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
# import openpyxl

# Decay model
# subjects = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
subjects = [16]

for i in range(0, len(subjects)):
    ID = subjects[i]
    if ID < 10:
        ID = f"00{ID}"
    else:
        ID = f"0{ID}"

    path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\Processed Data\\Input to model"
    data = pd.read_excel(f"{path}\\{ID}_input_file.xlsx")
    HR = data.iloc[:, 2]
    RPE = data.iloc[:, 3]
    Power_hc = data.iloc[:, 4]
    Power_bc = data.iloc[:, 5]
    time = np.linspace(301, 840, len(Power_hc))


    
    # Define the model with HR and RPE as multiplicative factors
    def model(t:np.ndarray, alpha:float, gamma:float, delta_hr:float, delta_rpe:float):
        m = alpha * Power_hc * (1 - gamma * t) * (1 + delta_hr * HR) * (1 + delta_rpe * RPE)
        return m

    # Fit the model to the data
    popt, pcov = curve_fit(model, time, Power_bc, p0=[0.1, 0.001, 0.001, 0.001], maxfev = 10000)

    # Extract the fitted parameters
    alpha_fitted, gamma_fitted, delta_hr_fitted, delta_rpe_fitted = popt
    print(f"Estimated alpha: {alpha_fitted}, Estimated gamma: {gamma_fitted}")
    print(f"Estimated delta_hr: {delta_hr_fitted}, Estimated delta_rpe: {delta_rpe_fitted}")

    # Plot the data and the fitted curve
    plt.plot(time, Power_bc, label="Observed bicycle power")
    plt.plot(time, model(time, *popt), label="Non linear model")
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.ylabel('Power (W)')
    plt.title(f"Decay model - linear, participant {ID}")

    popt_fitted = popt

    # Exponential model
    # Define the model with HR and RPE as multiplicative factors
    def model(t, alpha, gamma, delta_hr, delta_rpe):
        m = alpha * Power_hc * np.exp(-(gamma + delta_hr * HR + delta_rpe * RPE)*t) 
        return m

    # Fit the model to the data
    popt, pcov = curve_fit(model, time, Power_bc, p0=[3, 0.2, 0.036, 0.0003])

    # Extract the fitted parameters
    alpha_fitted, gamma_fitted, delta_hr_fitted, delta_rpe_fitted = popt
    print(f"Estimated alpha: {alpha_fitted}, Estimated gamma: {gamma_fitted}")
    print(f"Estimated delta_hr: {delta_hr_fitted}, Estimated delta_rpe: {delta_rpe_fitted}")

    # Plot the data and the fitted curve
    plt.plot(time, model(time, *popt), label="Exponential model")
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.ylabel('Power (W)')
    plt.title(f"Decay model, exponential, participant {ID}")
    plt.show()

    popt_exponential = popt



    # Save everything into an excel file




