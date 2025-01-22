'''This code gets in input the csv file containing data about a specific acquisition and as an outut gives back a 
.xslx file containing the paramenters for the model P_bc = alpha P_hc + beta + ...'''

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import math

# Decay model
# subjects = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
subjects = [10]

for i in range(0, len(subjects)):
    ID = subjects[i]
    if ID < 10:
        ID = f"00{ID}"
    else:
        ID = f"0{ID}"

    path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\Processed Data\\Filtered Power"
    Power_hc = data_hc
    data_hc = pd.read_csv(f"{path}\\{ID}_handcycle_filtered_power.csv")
    HR = data_hc[:, 4]
    RPE = data_hc[:, 10]
    time = np.linspace(0, len(Power_hc), len(Power_hc))

    data_bc = pd.read_csv(f"{path}\\{ID}_bicycle_protocol_filtered")
    Power_bc = data_bc[:, 7]

    
    # Define the model with HR and RPE as multiplicative factors
    def model(t, alpha, gamma, delta_hr, delta_rpe):
        return alpha * Power_hc[0] * (1 - gamma * t) * (1 + delta_hr * HR) * (1 + delta_rpe * RPE)

    # Fit the model to the data
    popt, pcov = curve_fit(model, time, Power_bc, p0=[1, 0.01, 0.01, 0.01])

    # Extract the fitted parameters
    alpha_fitted, gamma_fitted, delta_hr_fitted, delta_rpe_fitted = popt
    print(f"Estimated alpha: {alpha_fitted}, Estimated gamma: {gamma_fitted}")
    print(f"Estimated delta_hr: {delta_hr_fitted}, Estimated delta_rpe: {delta_rpe_fitted}")

    # Plot the data and the fitted curve
    plt.plot(time, Power_bc, 'o', label="Observed bicycle power")
    plt.plot(time, model(time, *popt), label="Fitted model")
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.ylabel('Power (W)')
    plt.show()
    plt.title("Decay model - linear")

    popt_fitted = popt

    # Exponential model
    # Define the model with HR and RPE as multiplicative factors
    def model(t, alpha, gamma, delta_hr, delta_rpe):
        return alpha * Power_hc[0] * math.exp(-(gamma + delta_hr * HR + delta_rpe * RPE)*time)

    # Fit the model to the data
    popt, pcov = curve_fit(model, time, Power_bc, p0=[1, 0.01, 0.01, 0.01])

    # Extract the fitted parameters
    alpha_fitted, gamma_fitted, delta_hr_fitted, delta_rpe_fitted = popt
    print(f"Estimated alpha: {alpha_fitted}, Estimated gamma: {gamma_fitted}")
    print(f"Estimated delta_hr: {delta_hr_fitted}, Estimated delta_rpe: {delta_rpe_fitted}")

    # Plot the data and the fitted curve
    plt.plot(time, Power_bc, 'o', label="Observed bicycle power")
    plt.plot(time, model(time, *popt), label="Fitted model")
    plt.legend()
    plt.xlabel('Time (minutes)')
    plt.ylabel('Power (W)')
    plt.show()
    plt.title("Decay model, exponential")

    popt_exponential = popt

    # Save everything into an excel file




