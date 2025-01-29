'''This code gets in input the csv file containing data about a specific acquisition and as an outut gives back a 
.xslx file containing the paramenters for the model P_bc = alpha P_hc + beta + ...'''

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
# import openpyxl

# Decay model
# subjects = [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
subjects = [11]

for i in range(0, len(subjects)):
    ID = subjects[i]
    if ID < 10:
        ID = f"00{ID}"
    else:
        ID = f"0{ID}"

    path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\Processed Data\\Input to model"
    data = pd.read_excel(f"{path}\\{ID}_input_file.xlsx")
    # data = pd.read_csv("Fake_data3.csv")
    HR = data.iloc[:, 2]
    RPE = data.iloc[:, 3]
    Power_hc = data.iloc[:, 4]
    Power_bc = data.iloc[:, 5]
    time = np.linspace(301, 840, len(Power_hc))
    Age = data.iloc[2, 1]
    Weight = data.iloc[3, 1]
    Height = data.iloc[4, 1]/100

#------------------------------------------------------------------------------------------------------------------
    # Linear model
#------------------------------------------------------------------------------------------------------------------
    # Define the model with alpha = Average power ratio
    alpha_array = np.zeros(3)
    Power_hc_tweaked = np.zeros(len(Power_hc))
    for i in range(2):
        avg_power_hc = np.mean(Power_hc[i * 180 : (i+1) * 180 - 1])
        avg_power_bc = np.mean(Power_bc[i * 180 : (i+1) * 180 - 1])
        alpha_array[i] = avg_power_bc/avg_power_hc
    
    alpha = np.mean(alpha_array)
    Power_hc_tweaked = alpha * Power_hc

    # Print output
    print("Linear model:")
    print(f"Estimated alpha: {alpha}")

    # Plot the signal
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(time, Power_bc, label="Observed bicycle power")
    axs[0, 0].plot(time, Power_hc_tweaked, label="Linear model")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Power [W]")
    axs[0, 0].set_title(f"Linear Model, participant {ID}")
    axs[0, 0].legend()
    # plt.show()

    alpha_linear = alpha_array

#------------------------------------------------------------------------------------------------------------------
    # Simple decay model: Non linear model, least square methods, no HR or RPE
#------------------------------------------------------------------------------------------------------------------
    def model(t, alpha, gamma):
        m = alpha * Power_hc * (1 - gamma) * t
        return m

    popt, pcov = curve_fit(model, time, Power_bc, p0=[0.1, 0.001], maxfev = 10000)

    # Extract the fitted parameters and print their values
    alpha_fitted, gamma_fitted = popt
    print("Simple decay model:")
    print(f"Estimated alpha: {alpha_fitted}, Estimated gamma: {gamma_fitted}")

    # Plotting data
    axs[1, 0].plot(time, Power_bc, label = "Observed bicycle power")
    axs[1, 0].plot(time, model(time, *popt), label="Non linear model")
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Power [W]')
    axs[1, 0].set_title(f"Simple decay model, participant {ID}")
    # plt.show()

    popt_simple_decay = popt


#------------------------------------------------------------------------------------------------------------------
    # HR and RPE as Modifiers for Fatigue (Multiplicative Adjustment)
#------------------------------------------------------------------------------------------------------------------
    # Define the model with HR and RPE as multiplicative factors
    def model(t:np.ndarray, alpha:float, gamma:float, delta_hr:float, delta_rpe:float):
        m = alpha * Power_hc * (1 - gamma * t) * (1 + delta_hr * HR) * (1 + delta_rpe * RPE)
        return m

    # Fit the model to the data
    popt, pcov = curve_fit(model, time, Power_bc, p0=[0.1, 0.001, 0.001, 0.001], maxfev = 10000)

    # Extract the fitted parameters and print them
    alpha_fitted, gamma_fitted, delta_hr_fitted, delta_rpe_fitted = popt
    print("Multiplicatve adjustment model:")
    print(f"Estimated alpha: {alpha_fitted}, Estimated gamma: {gamma_fitted}")
    print(f"Estimated delta_hr: {delta_hr_fitted}, Estimated delta_rpe: {delta_rpe_fitted}")

    # Plot the data and the fitted curve
    axs[0, 1].plot(time, Power_bc, label = "Observed bicycle power")
    axs[0, 1].plot(time, model(time, *popt), label="Non linear model")
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Power [W]')
    axs[0, 1].set_title(f"Decay model - multiplicative, participant {ID}")
    # plt.show()

    decay, axs2 = plt.subplots(3)
    axs2[0].plot(time, Power_bc, label = "Observed bicycle power")
    axs2[0].plot(time, model(time, *popt), label="Non linear model")
    axs2[0].legend()
    axs2[0].set_xlabel('Time [s]')
    axs2[0].set_ylabel('Power [W]')
    axs2[0].set_title(f"HR and RPE, participant {ID}")
    decay.suptitle("Multiplicative decay models")

    popt_decay_multiplicative = popt

#------------------------------------------------------------------------------------------------------------------
    # HR and RPE, Age as Modifiers for Fatigue (Multiplicative Adjustment)
#------------------------------------------------------------------------------------------------------------------
    # Define the model with HR and RPE as multiplicative factors
    def model(t:np.ndarray, alpha:float, gamma:float, delta_hr:float, delta_rpe:float, delta_age:float):
        m = alpha * Power_hc * (1 - gamma * t) * (1 + delta_hr * HR) * (1 + delta_rpe * RPE) * (1 + delta_age * Age)
        return m

    # Fit the model to the data
    popt, pcov = curve_fit(model, time, Power_bc, p0=[0.1, 0.001, 0.001, 0.001, 0.1], maxfev = 10000)

    # Extract the fitted parameters and print them
    alpha_fitted, gamma_fitted, delta_hr_fitted, delta_rpe_fitted, delta_age_fittef = popt
    print("Multiplicatve adjustment model:")
    print(f"Estimated alpha: {alpha_fitted}, Estimated gamma: {gamma_fitted}")
    print(f"Estimated delta_hr: {delta_hr_fitted}, Estimated delta_rpe: {delta_rpe_fitted}")

    # Plot the data and the fitted curve
    axs2[1].plot(time, Power_bc, label = "Observed bicycle power")
    axs2[1].plot(time, model(time, *popt), label="Non linear model")
    axs2[1].legend()
    axs2[1].set_xlabel('Time [s]')
    axs2[1].set_ylabel('Power [W]')
    axs2[1].set_title(f"HR, RPE, Age, participant {ID}")
    # plt.show()

    popt_decay_multiplicative_Age = popt

#------------------------------------------------------------------------------------------------------------------
    # HR and RPE, Age, BMI as Modifiers for Fatigue (Multiplicative Adjustment)
#------------------------------------------------------------------------------------------------------------------
    # Define the model with HR and RPE as multiplicative factors
    BMI = Weight/Height**2 
    
    def model(t:np.ndarray, alpha:float, gamma:float, delta_hr:float, delta_rpe:float, delta_age:float, delta_BMI:float):
        m = alpha * Power_hc * (1 - gamma * t) * (1 + delta_hr * HR) * (1 + delta_rpe * RPE) * (1 + delta_age * Age) * (1 + delta_BMI)
        return m

    # Fit the model to the data
    popt, pcov = curve_fit(model, time, Power_bc, p0=[0.1, 0.001, 0.001, 0.001, 0.01, 0.1], maxfev = 10000)

    # Extract the fitted parameters and print them
    alpha_fitted, gamma_fitted, delta_hr_fitted, delta_rpe_fitted, delta_age_fitted, delta_bmi_fitted = popt
    print("Multiplicatve adjustment model:")
    print(f"Estimated alpha: {alpha_fitted}, Estimated gamma: {gamma_fitted}")
    print(f"Estimated delta_hr: {delta_hr_fitted}, Estimated delta_rpe: {delta_rpe_fitted}")
    print(f"Estimated delta_age: {delta_age_fitted}, Estimated delta_bmi: {delta_bmi_fitted}")

    # Plot the data and the fitted curve
    axs2[2].plot(time, Power_bc, label = "Observed bicycle power")
    axs2[2].plot(time, model(time, *popt), label="Non linear model")
    axs2[2].legend()
    axs2[2].set_xlabel('Time [s]')
    axs2[2].set_ylabel('Power [W]')
    axs2[2].set_title(f"HR, RPE, Age, BMI, participant {ID}")
    # plt.show()

    popt_decay_multiplicative_Age = popt

#------------------------------------------------------------------------------------------------------------------
    # Exponential model: HR and RPE as Separate Fatigue Factors (Adding to the Decay Term)
#------------------------------------------------------------------------------------------------------------------
    # Define the model with HR and RPE as multiplicative factors
    def model(t, alpha, gamma, delta_hr, delta_rpe):
        m = alpha * Power_hc * np.exp(-(gamma + delta_hr * HR+ delta_rpe * RPE) * t) 
        return m

    # Fit the model to the data
    popt, pcov = curve_fit(model, time, Power_bc, p0=[3, 0.01, 0.0001, 0.001])

    # Extract the fitted parameters
    alpha_fitted, gamma_fitted, delta_hr_fitted, delta_rpe_fitted = popt
    print("Exponential decay model: ")
    print(f"Estimated alpha: {alpha_fitted}, Estimated gamma: {gamma_fitted}")
    print(f"Estimated delta_hr: {delta_hr_fitted}, Estimated delta_rpe: {delta_rpe_fitted}")

    # Plot the data and the fitted curve
    axs[1, 1].plot(time, Power_bc, label = "Observed power bicycle")
    axs[1, 1].plot(time, model(time, *popt), label="Exponential model")
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Power [W]')
    axs[1, 1].set_title(f"Decay model - exponential, participant {ID}")
    plt.show()

    popt_decay_exponential = popt


    #-----------------------------------------------------------------------------------------------------------------------------------
        # Evaluating the model performance
    #------------------------------------------------------------------------------------------------------------------------------------
                                             

    # Save everything into an excel file




