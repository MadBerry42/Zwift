'''This code gets in input the csv file containing data about a specific acquisition and as an outut gives back a 
.xslx file containing the paramenters for the model P_bc = alpha P_hc + beta + ...'''

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import Statistical_analysis

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
    # data = pd.read_excel(f"{path}\\{ID}_input_file.xlsx")
    data = pd.read_csv("Fake_data.csv")
    HR = data.iloc[:, 0]
    RPE = data.iloc[:, 1]
    Power_hc = data.iloc[:, 2]
    Power_bc = data.iloc[:, 3]
    time = np.linspace(0, 540, len(Power_hc))
    # Age = data.iloc[2, 1]
    # Weight = data.iloc[3, 1]
    # Height = data.iloc[4, 1]/100

    Age = 25
    Weight = 58
    Height = 1.6

#------------------------------------------------------------------------------------------------------------------
    # Linear model
#------------------------------------------------------------------------------------------------------------------
    # Define the model with alpha = Average power ratio
    alpha_array = np.zeros(3)
    for i in range(2):
        avg_power_hc = np.mean(Power_hc[i * 180 : (i+1) * 180 - 1])
        avg_power_bc = np.mean(Power_bc[i * 180 : (i+1) * 180 - 1])
        alpha_array[i] = avg_power_bc/avg_power_hc
    
    alpha = np.mean(alpha_array)
    Power_hc_tweaked_linear = alpha * Power_hc

    # Print output
    print("Linear model:")
    print(f"Estimated alpha: {alpha}")
    print("\n")

    # Plot the signal
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(time, Power_bc, label="Observed bicycle power")
    axs[0, 0].plot(time, Power_hc_tweaked_linear, label="Linear model")
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
    alpha, gamma = popt
    print("Simple decay model:")
    print(f"Estimated alpha: {alpha}, Estimated gamma: {gamma}")
    print("\n")

    # Plotting data
    axs[1, 0].plot(time, Power_bc, label = "Observed bicycle power")
    axs[1, 0].plot(time, model(time, *popt), label="Non linear model")
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Power [W]')
    axs[1, 0].set_title(f"Simple decay model, participant {ID}")
    # plt.show()

    popt_simple_decay = popt
    Power_hc_tweaked_simple_decay = model(time, *popt)


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
    alpha, gamma, delta_hr, delta_rpe = popt
    print("Multiplicatve adjustment model: HR and RPE")
    print(f"Estimated alpha: {alpha}, Estimated gamma: {gamma}")
    print(f"Estimated delta_hr: {delta_hr}, Estimated delta_rpe: {delta_rpe}")
    print("\n")

    # Plot the data and the fitted curve
    axs[0, 1].plot(time, Power_bc, label = "Observed bicycle power")
    axs[0, 1].plot(time, model(time, *popt), label="Non linear model")
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Power [W]')
    axs[0, 1].set_title(f"Decay model - multiplicative, participant {ID}")
    # plt.show()

    popt_decay_multiplicative = popt
    Power_hc_tweaked_decay_multiplicative = model(time, *popt)

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
    alpha, gamma, delta_hr, delta_rpe = popt
    print("Exponential decay model: ")
    print(f"Estimated alpha: {alpha}, Estimated gamma: {gamma}")
    print(f"Estimated delta_hr: {delta_hr}, Estimated delta_rpe: {delta_rpe}")
    print("\n")
    print("\n")

    # Plot the data and the fitted curve
    axs[1, 1].plot(time, Power_bc, label = "Observed power bicycle")
    axs[1, 1].plot(time, model(time, *popt), label="Exponential model")
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Power [W]')
    axs[1, 1].set_title(f"Decay model - exponential: HR and RPE, participant {ID}")
    
    plt.tight_layout()
    # plt.show()

    popt_decay_exponential = popt
    Power_hc_tweaked_exponential_decay = model(time, *popt)


#-----------------------------------------------------------------------------------------------------------------------------------
    # Evaluating the model performance: residuals, MSE, RMSE, OLP Line
#------------------------------------------------------------------------------------------------------------------------------------
    # Linear model
    Power_hc = Power_hc_tweaked_linear
    residuals = Power_bc - Power_hc
    sum_residuals = sum(residuals)
    print(f"The sum of residuals for the linear model is: {sum_residuals:.3f}") 
    print(f"The maximum residual is: {max(residuals):.3f}")    

    MSE, RMSE = Statistical_analysis.get_MSE_and_RMSE(Power_bc, Power_hc)
    print(f"MSE for this model is: {MSE:.3f}")
    print(f"RMSE for this model is: {RMSE:.3f}")

    a, b, y_fit, CI_a, CI_b = Statistical_analysis.olp_line(Power_bc, Power_hc)
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("OLP line")
    axs[0, 0].scatter(Power_bc, Power_hc, c = 'orange', label = "Data")
    axs[0, 0].plot(Power_bc, Power_bc, c = 'green', label = "Bisecant")
    axs[0, 0].plot(Power_bc, y_fit, c = "red", label = "Interpolating line")
    axs[0, 0].legend(loc = "upper left", fontsize = "small")
    axs[0, 0].set_title(f"Linear model, a = {a:.2f}, b = {b:.2f}")
    print(f"The parameter of the OLP line are a = {a:.3f}, b = {b:.3f}, with a Confidence Interval at 95% of {CI_a} for a and {CI_b} for b")
    print("\n")

    dxy, Sxy, lim_sup, lim_inf = Statistical_analysis.get_Bland_Altman_plot(Power_bc, Power_hc)
    fig2, axs2 = plt.subplots(2, 2)
    fig2.suptitle("Bland-Altman Plot")
    axs2[0, 0].plot((Power_bc + Power_hc)/2, (Power_bc - Power_hc)/2, '*')
    axs2[0, 0].axhline(y = dxy, color = "b")
    axs2[0, 0].axhline(y = lim_sup, linestyle = "-.")
    axs2[0, 0].axhline(y = lim_inf, linestyle = "-.")
    axs2[0, 0].set_title("Linear model")


    # Simple decay model
    Power_hc = Power_hc_tweaked_simple_decay
    residuals = Power_bc - Power_hc
    sum_residuals = sum(residuals)
    print(f"The sum of residuals for the simple decay model is: {sum_residuals:.3f}") 
    print(f"The maximum residual is: {max(residuals):.3f}")    

    MSE, RMSE = Statistical_analysis.get_MSE_and_RMSE(Power_bc, Power_hc)
    print(f"MSE for this model is: {MSE:.3f}")
    print(f"RMSE for this model is: {RMSE:.3f}")

    a, b, y_fit, CI_a, CI_b = Statistical_analysis.olp_line(Power_bc, Power_hc)
    axs[1, 0].scatter(Power_bc, Power_hc, c = 'orange', label = "Data")
    axs[1, 0].plot(Power_bc, Power_bc, c = 'green', label = "Bisecant")
    axs[1, 0].plot(Power_bc, y_fit, c = "red", label = "Interpolating line")
    axs[1, 0].legend(loc = "upper left", fontsize = "small")
    axs[1, 0].set_title(f"Linear model, a = {a:.2f}, b = {b:.2f}")
    print(f"The parameter of the OLP line are a = {a:.3f}, b = {b:.3f}, with a Confidence Interval at 95% of {CI_a} for a and {CI_b} for b")
    print("\n")

    dxy, Sxy, lim_sup, lim_inf = Statistical_analysis.get_Bland_Altman_plot(Power_bc, Power_hc)
    axs2[1, 0].plot((Power_bc + Power_hc)/2, (Power_bc - Power_hc)/2, '*')
    axs2[1, 0].axhline(y = dxy, color = "b")
    axs2[1, 0].axhline(y = lim_sup, linestyle = "-.")
    axs2[1, 0].axhline(y = lim_inf, linestyle = "-.")
    axs2[1, 0].set_title("Simple decay model")

    # Multiplicative decay model
    Power_hc = Power_hc_tweaked_decay_multiplicative
    residuals = Power_bc - Power_hc
    sum_residuals = sum(residuals)
    print(f"The sum of residuals for the multiplicative decay model is: {sum_residuals:.3f}") 
    print(f"The maximum residual is: {max(residuals):.3f}")    

    MSE, RMSE = Statistical_analysis.get_MSE_and_RMSE(Power_bc, Power_hc)
    print(f"MSE for this model is: {MSE:.3f}")
    print(f"RMSE for this model is: {RMSE:.3f}")

    a, b, y_fit, CI_a, CI_b = Statistical_analysis.olp_line(Power_bc, Power_hc)
    axs[0, 1].scatter(Power_bc, Power_hc, c = 'orange', label = "Data")
    axs[0, 1].plot(Power_bc, Power_bc, c = 'green', label = "Bisecant")
    axs[0, 1].plot(Power_bc, y_fit, c = "red", label = "Interpolating line")
    axs[0, 1].legend(loc = "upper left", fontsize = "small")
    axs[0, 1].set_title(f"Linear model, a = {a:.2f}, b = {b:.2f}")
    print(f"The parameter of the OLP line are a = {a:.3f}, b = {b:.3f},, with a Confidence Interval at 95% of {CI_a} for a and {CI_b} for b")
    print("\n")

    dxy, Sxy, lim_sup, lim_inf = Statistical_analysis.get_Bland_Altman_plot(Power_bc, Power_hc)
    axs2[0, 1].plot((Power_bc + Power_hc)/2, (Power_bc - Power_hc)/2, '*')
    axs2[0, 1].axhline(y = dxy, color = "b")
    axs2[0, 1].axhline(y = lim_sup, linestyle = "-.")
    axs2[0, 1].axhline(y = lim_inf, linestyle = "-.")
    axs2[0, 1].set_title("Multiplicative decay model")

    # Exponential model
    Power_hc = Power_hc_tweaked_exponential_decay
    residuals = Power_bc - Power_hc
    sum_residuals = sum(residuals)
    print(f"The sum of residuals for the exponential decay model is: {sum_residuals:.3f}") 
    print(f"The maximum residual is: {max(residuals):.3f}")    

    MSE, RMSE = Statistical_analysis.get_MSE_and_RMSE(Power_bc, Power_hc)
    print(f"MSE for this model is: {MSE:.3f}")
    print(f"RMSE for this model is: {RMSE:.3f}")

    a, b, y_fit, CI_a, CI_b = Statistical_analysis.olp_line(Power_bc, Power_hc)
    axs[1, 1].scatter(Power_bc, Power_hc, c = 'orange', label = "Data")
    axs[1, 1].plot(Power_bc, Power_bc, c = 'green', label = "Bisecant")
    axs[1, 1].plot(Power_bc, y_fit, c = "red", label = "Interpolating line")
    axs[1, 1].legend(loc = "upper left", fontsize = "small")
    axs[1, 1].set_title(f"Linear model, a = {a:.2f}, b = {b:.2f}")
    print(f"The parameter of the OLP line are a = {a:.3f}, b = {b:.3f}, with a Confidence Interval at 95% of {CI_a} for a and {CI_b} for b")
    print("\n")

    dxy, Sxy, lim_sup, lim_inf = Statistical_analysis.get_Bland_Altman_plot(Power_bc, Power_hc)
    axs2[1, 1].plot((Power_bc + Power_hc)/2, (Power_bc - Power_hc)/2, '*')
    axs2[1, 1].axhline(y = dxy, color = "b")
    axs2[1, 1].axhline(y = lim_sup, linestyle = "-.")
    axs2[1, 1].axhline(y = lim_inf, linestyle = "-.")
    axs2[1, 1].set_title("Exponential decay model")

    plt.tight_layout()
    plt.show()


        # Notes 
    #----------------
    # Residuals: least residuals?
    # Best in terms of MSE and RMSE?
    # Bland-Altman Plot: Where is the horizontal line closest to zero (least fixed bias)? Where are the 95% confidence level the smallest?
    # OLP line: which model has the value of a closest to 0 and b closest to 1? (consider also the CI for a and b)
    # --> Create a table in the powerpoint presentation highlighting which model worked best


    #----------------------------------------------------------------------------------------------------------------------------------------
        # Choose the best parameters for the best performing model (exponential model)
    #----------------------------------------------------------------------------------------------------------------------------------------
        # Useful point in thesis discussion: which parameters seem to be most relevant into describing the power output? See coefficients
        # in the model. IF I UNDERSTOOD IT CORRECTLY (and idk) the higher the coefficient, the more weight a variable has.



    #----------------------------------------------------------------------------------------------------------------------------------------
        # Save everything into an excel file
    #----------------------------------------------------------------------------------------------------------------------------------------






