'''This code gets in input the csv file containing data about a specific acquisition and as an outut gives back a 
.xslx file containing the paramenters for the model P_bc = alpha P_hc + beta + ...'''

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import Functions

# Decay model
subjects = [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# subjects = [8, 9]

for j in range(0, len(subjects)):
    ID = subjects[j]
    if ID < 10:
        ID = f"00{ID}"
    else:
        ID = f"0{ID}"

    path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\Processed Data\\Input to model"
    data = pd.read_excel(f"{path}\\{ID}_input_file.xlsx")
    # data = pd.read_csv("Fake_data.csv")
    HR = data.iloc[:, 2]
    RPE = data.iloc[:, 3]
    Power_hc = data.iloc[:, 4]
    Power_bc = data.iloc[:, 5].to_numpy()
    time = np.linspace(0, 540, len(Power_hc))

    Gender = data.iloc[1, 1]
    Age = data.iloc[2, 1]
    Weight = data.iloc[3, 1]
    Height = data.iloc[4, 1]/100


#------------------------------------------------------------------------------------------------------------------
    # Linear model
#------------------------------------------------------------------------------------------------------------------
    # Define the model with alpha = Average power ratio
    alpha_array = np.zeros(3)
    for i in range(3):
        avg_power_hc = np.mean(Power_hc[i * 180 : (i+1) * 180])
        avg_power_bc = np.mean(Power_bc[i * 180 : (i+1) * 180])
        alpha_array[i] = avg_power_bc/avg_power_hc
    
    alpha = np.mean(alpha_array)
    Model_linear = alpha * Power_hc.to_numpy()

    # Print output
    print("Linear model:")
    print(f"Estimated alpha: {alpha}")
    print("\n")

    # Plot the signal
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Models vs bicycle, participant {ID}")
    axs[0, 0].plot(time, Power_bc, label="Observed bicycle power")
    axs[0, 0].plot(time, Model_linear, label="Linear model")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Power [W]")
    axs[0, 0].set_title(f"Linear Model")
    axs[0, 0].legend()

    # plt.show()

    Linear_parameters = alpha_array
    Linear_averages = alpha

#------------------------------------------------------------------------------------------------------------------
    # Simple decay model: Non linear model, least square methods, no HR or RPE
#------------------------------------------------------------------------------------------------------------------
    Power_hc_or = Power_hc.copy()
    HR_or = HR.copy()
    RPE_or = RPE.copy()

    def model(t, alpha, gamma):
        m = alpha * Power_hc * (1 - gamma)* t
        return m

    alpha_array = np.zeros(3)
    gamma_array = np.zeros(3)
    for i in range(3):
        t = time[i * 180 : (i+1) * 180]
        Power_hc = Power_hc_or[i * 180 : (i+1) * 180]
        popt, pcov = curve_fit(model, t,  Power_bc[i * 180 : (i+1) * 180], p0=[0.1, 0.001], maxfev = 10000)
        alpha_array[i], gamma_array[i] = popt
        Hc = model(t, *popt)
        if i == 0:
            Model_hc = Hc
        else:
            Model_hc = np.concatenate((Model_hc, Hc))

    alpha = np.mean(alpha_array)
    gamma = np.mean(gamma_array)

    # Extract the fitted parameters and print their values
    print("Simple decay model:")
    print(f"Estimated alpha values: {alpha_array[0]:.3f}, {alpha_array[1]:.3f}, {alpha_array[2]:.3f}, average value: {alpha:.3f}")
    print(f"Estimated gamma values: {gamma_array[0]:.3f}, {gamma_array[1]:.3f}, {gamma_array[2]:.3f}, average value: {gamma:.3f}")
    print("\n")

    # Plotting data
    axs[1, 0].plot(time, Power_bc, label = "Observed bicycle power")
    axs[1, 0].plot(time, Model_hc, label="Non linear model")
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Power [W]')
    axs[1, 0].set_title(f"Simple decay model")
    # plt.show()

    # Save the values
    Model_simple_decay = Model_hc.copy()
    Simple_decay_parameters = alpha_array, gamma_array
    Simple_decay_parameters = np.array(Simple_decay_parameters)
    Simple_decay_averages = alpha, gamma



#------------------------------------------------------------------------------------------------------------------
    # HR and RPE as Modifiers for Fatigue (Multiplicative Adjustment)
#------------------------------------------------------------------------------------------------------------------
    # Define the model with HR and RPE as multiplicative factors
    def model(t:np.ndarray, alpha:float, gamma:float, delta_hr:float, delta_rpe:float):
        m = alpha * Power_hc * (1 - gamma * t) * (1 + delta_hr * HR) * (1 + delta_rpe * RPE)
        return m

    # Fit the model to the data
    alpha_array = np.zeros(3)
    gamma_array = np.zeros(3)
    delta_hr_array = np.zeros(3)
    delta_rpe_array = np.zeros(3)
    for i in range(3):
        Power_hc = Power_hc_or[i * 180 : (i+1) * 180]
        HR = HR_or[i * 180 : (i+1) * 180]
        RPE = RPE_or[i * 180 : (i+1) * 180]

        popt, pcov = curve_fit(model, time[i * 180 : (i+1) * 180], Power_bc[i * 180 : (i+1) * 180], p0=[0.1, 0.001, 0.001, 0.001], maxfev = 10000)
        alpha_array[i], gamma_array[i], delta_hr_array[i], delta_rpe_array[i]  = popt
        Model_hc[i * 180 : (i+1) * 180] = model(time[i * 180 : (i+1) * 180], *popt)

    alpha = np.mean(alpha_array)
    gamma = np.mean(gamma_array)
    delta_hr = np.mean(delta_hr_array)
    delta_rpe = np.mean(delta_rpe_array)
    
    print("Multiplicatve adjustment model: HR and RPE")
    print(f"Estimated alpha values: {alpha_array[0]:.3f}, {alpha_array[1]:.3f}, {alpha_array[2]:.3f}, average value: {alpha:.3f}")
    print(f"Estimated gamma values: {gamma_array[0]:.3f}, {gamma_array[1]:.3f}, {gamma_array[2]:.3f}, average value: {gamma:.3f}")
    print(f"Estimated delta_hr values: {delta_hr_array[0]:.3f}, {delta_hr_array[1]:.3f}, {delta_hr_array[2]:.3f}, average value: {delta_hr:.3f}")
    print(f"Estimated delta_rpe values: {delta_rpe_array[0]:.3f}, {delta_rpe_array[1]:.3f}, {delta_rpe_array[2]:.3f}, average value: {delta_rpe:.3f}")
    print("\n")

    # Plot the data and the fitted curve
    axs[0, 1].plot(time, Power_bc, label = "Observed bicycle power")
    axs[0, 1].plot(time, Model_hc, label="Non linear model")
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Power [W]')
    axs[0, 1].set_title(f"Decay model - multiplicative adjustment")
    # plt.show()

    Model_multiplicative = Model_hc
    Multiplicative_parameters = alpha_array, gamma_array, delta_hr_array, delta_rpe_array
    Multiplicative_parameters = np.array(Multiplicative_parameters)
    Multiplicative_averages = alpha, gamma, delta_hr, delta_rpe


#------------------------------------------------------------------------------------------------------------------
    # Exponential model: HR and RPE as Separate Fatigue Factors (Adding to the Decay Term)
#------------------------------------------------------------------------------------------------------------------
    # Define the model with HR and RPE as multiplicative factors
    def model(t, alpha, gamma, delta_hr, delta_rpe):
        m = alpha * Power_hc * np.exp(-(gamma + delta_hr * HR+ delta_rpe * RPE) * t)
        return m

    # Fit the model to the data
    alpha_array = np.zeros(3)
    gamma_array = np.zeros(3)
    delta_hr_array = np.zeros(3)
    delta_rpe_array = np.zeros(3)
    Model_hc = np.zeros(len(Power_hc_or))
    for i in range(3):
        Power_hc = Power_hc_or[i * 180 : (i+1) * 180]
        HR = HR_or[i * 180 : (i+1) * 180]
        RPE = RPE_or[i * 180 : (i+1) * 180]

        popt, pcov = curve_fit(model, time[i * 180 : (i+1) * 180], Power_bc[i * 180 : (i+1) * 180], p0=[0.1, 0.01, 0.0001, 0.0001], maxfev = 10000)#
        alpha_array[i], gamma_array[i], delta_hr_array[i], delta_rpe_array[i]   = popt
        Model_hc[i * 180 : (i+1) * 180] = model(time[i * 180 : (i+1) * 180], *popt)

    alpha = np.mean(alpha_array)
    gamma = np.mean(gamma_array)
    delta_hr = np.mean(delta_hr_array)
    delta_rpe = np.mean(delta_rpe_array)


    print("Exponential decay model: HR and RPE")
    print(f"Estimated alpha values: {alpha_array[0]:.3f}, {alpha_array[1]:.3f}, {alpha_array[2]:.3f}, average value: {alpha:.3f}")
    print(f"Estimated gamma values: {gamma_array[0]:.3f}, {gamma_array[1]:.3f}, {gamma_array[2]:.3f}, average value: {gamma:.3f}")
    print(f"Estimated delta_hr values: {delta_hr_array[0]:.3f}, {delta_hr_array[1]:.3f}, {delta_hr_array[2]:.3f}, average value: {delta_hr:.3f}")
    print(f"Estimated delta_rpe values: {delta_rpe_array[0]:.3f}, {delta_rpe_array[1]:.3f}, {delta_rpe_array[2]:.3f}, average value: {delta_rpe:.3f}")
    print("\n")
    print("\n")

    # Plot the data and the fitted curve
    axs[1, 1].plot(time, Power_bc, label = "Observed power bicycle")
    axs[1, 1].plot(time, Model_hc, label="Exponential model")
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Power [W]')
    axs[1, 1].set_title(f"Decay model - exponential")

    for ax in axs.flatten():
        ax.axvspan(0, 179, color = "green", alpha = 0.2)
        ax.axvspan(180, 359, color = "yellow", alpha = 0.2)
        ax.axvspan(360, 540, color = "red", alpha = 0.2)

        green_patch = mpatches.Patch(color='green', alpha=0.2, label='Low effort')
        yellow_patch = mpatches.Patch(color='yellow', alpha=0.2, label='Medium effort')
        red_patch = mpatches.Patch(color='red', alpha=0.2, label='High effort')



    plt.tight_layout()
    # plt.show()

    Model_exponential = Model_hc
    Exponential_parameters = alpha_array, gamma_array, delta_hr_array, delta_rpe_array
    Exponential_parameters = np.array(Exponential_parameters)
    Exponential_averages = alpha, gamma, delta_hr, delta_rpe



#-----------------------------------------------------------------------------------------------------------------------------------
    # Evaluating the model performance: residuals, MSE, RMSE, OLP Line
#------------------------------------------------------------------------------------------------------------------------------------
    # Linear model
    Power_hc = Model_linear
    residuals = Power_bc - Power_hc
    sum_residuals = sum(residuals)
    print(f"The sum of residuals for the linear model is: {sum_residuals:.3f}") 
    print(f"The maximum residual is: {max(residuals):.3f}")    

    MSE, RMSE = Functions.get_MSE_and_RMSE(Power_bc, Power_hc)
    print(f"MSE for this model is: {MSE:.3f}")
    print(f"RMSE for this model is: {RMSE:.3f}")

    a, b, y_fit, CI_a, CI_b = Functions.olp_line(Power_bc, Power_hc)
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"OLP line, participant {ID}")
    plt.tight_layout()
    axs[0, 0].scatter(Power_bc, Power_hc, c = 'orange', label = "Data")
    axs[0, 0].plot(Power_bc, Power_bc, c = 'green', label = "Bisecant")
    axs[0, 0].plot(Power_bc, y_fit, c = "red", label = "Interpolating line")
    axs[0, 0].legend(loc = "upper left", fontsize = "small")
    axs[0, 0].set_title(f"Linear model, a = {a:.2f}, b = {b:.2f}")
    print(f"The parameter of the OLP line are a = {a:.3f}, b = {b:.3f}, with a Confidence Interval at 95% of [{CI_a[0]:.3f}, {CI_a[1]:.3f}] for a and [{CI_b[0]:.3f}, {CI_b[1]:.3f}] for b")
    print("\n")

    dxy, Sxy, lim_sup, lim_inf = Functions.get_Bland_Altman_plot(Power_bc, Power_hc)
    fig2, axs2 = plt.subplots(2, 2)
    fig2.suptitle(f"Bland-Altman Plot, participant {ID}")
    axs2[0, 0].plot((Power_bc + Power_hc)/2, (Power_bc - Power_hc)/2, '*')
    axs2[0, 0].axhline(y = dxy, color = "b")
    axs2[0, 0].axhline(y = lim_sup, linestyle = "-.")
    axs2[0, 0].axhline(y = lim_inf, linestyle = "-.")
    axs2[0, 0].set_title("Linear model")

    # For the Excel file
    Col_residuals = [f"Sum: {sum_residuals:.3f}", f"Max: {max(residuals):.3f}", " ", " "]
    Col_MSE = [f"MSE = {MSE:.3f}", f"RMSE = {RMSE:.3f}",  " ", " "]
    Col_OLP = [f"b = {b:.3f}", f"a = {a:.3f}", f"CI_a = [{CI_a[0]:.3f}, {CI_a[1]:.3f}]", f"CI_b = [{CI_b[0]:.3f}, {CI_b[1]:.3f}]"]
    Col_BA = [f"Mean difference is: {dxy:.3f}", f"Deviation is: {2*Sxy:.3f}", " ", " "]



    # Simple decay model
    Power_hc = Model_simple_decay
    residuals = Power_bc - Power_hc
    sum_residuals = sum(residuals)
    print(f"The sum of residuals for the simple decay model is: {sum_residuals:.3f}") 
    print(f"The maximum residual is: {max(residuals):.3f}")    

    MSE, RMSE = Functions.get_MSE_and_RMSE(Power_bc, Power_hc)
    print(f"MSE for this model is: {MSE:.3f}")
    print(f"RMSE for this model is: {RMSE:.3f}")

    a, b, y_fit, CI_a, CI_b = Functions.olp_line(Power_bc, Power_hc)
    axs[1, 0].scatter(Power_bc, Power_hc, c = 'orange', label = "Data")
    axs[1, 0].plot(Power_bc, Power_bc, c = 'green', label = "Bisecant")
    axs[1, 0].plot(Power_bc, y_fit, c = "red", label = "Interpolating line")
    axs[1, 0].legend(loc = "upper left", fontsize = "small")
    axs[1, 0].set_title(f"Simple decay, a = {a:.2f}, b = {b:.2f}")
    print(f"The parameter of the OLP line are a = {a:.3f}, b = {b:.3f}, with a Confidence Interval at 95% of [{CI_a[0]:.3f}, {CI_a[1]:.3f}] for a and [{CI_b[0]:.3f}, {CI_b[1]:.3f}] for b")
    print("\n")

    dxy, Sxy, lim_sup, lim_inf = Functions.get_Bland_Altman_plot(Power_bc, Power_hc)
    axs2[1, 0].plot((Power_bc + Power_hc)/2, (Power_bc - Power_hc)/2, '*')
    axs2[1, 0].axhline(y = dxy, color = "b")
    axs2[1, 0].axhline(y = lim_sup, linestyle = "-.")
    axs2[1, 0].axhline(y = lim_inf, linestyle = "-.")
    axs2[1, 0].set_title("Simple decay model")

    # For the Excel file
    Col_residuals = Col_residuals + [f"Sum: {sum_residuals:.3f}", f"Max: {max(residuals):.3f}", " ", " "]
    Col_MSE = Col_MSE + [f"MSE = {MSE:.3f}", f"RMSE = {RMSE:.3f}", " ", " "]
    Col_OLP = Col_OLP + [f"b = {b:.3f}", f"a = {a:.3f}", f"CI_a = [{CI_a[0]:.3f}, {CI_a[1]:.3f}]", f"CI_b = [{CI_b[0]:.3f}, {CI_b[1]:.3f}]"]
    Col_BA = Col_BA + [f"Mean difference is: {dxy:.3f}", f"Deviation is: {2*Sxy:.3f}", " ", " "]


    # Multiplicative decay model
    Power_hc = Model_multiplicative
    residuals = Power_bc - Power_hc
    sum_residuals = sum(residuals)
    print(f"The sum of residuals for the multiplicative decay model is: {sum_residuals:.3f}") 
    print(f"The maximum residual is: {max(residuals):.3f}")    

    MSE, RMSE = Functions.get_MSE_and_RMSE(Power_bc, Power_hc)
    print(f"MSE for this model is: {MSE:.3f}")
    print(f"RMSE for this model is: {RMSE:.3f}")

    a, b, y_fit, CI_a, CI_b = Functions.olp_line(Power_bc, Power_hc)
    axs[0, 1].scatter(Power_bc, Power_hc, c = 'orange', label = "Data")
    axs[0, 1].plot(Power_bc, Power_bc, c = 'green', label = "Bisecant")
    axs[0, 1].plot(Power_bc, y_fit, c = "red", label = "Interpolating line")
    axs[0, 1].legend(loc = "upper left", fontsize = "small")
    axs[0, 1].set_title(f"Multiplicative adjustment model, a = {a:.2f}, b = {b:.2f}")
    print(f"The parameter of the OLP line are a = {a:.3f}, b = {b:.3f}, with a Confidence Interval at 95% of [{CI_a[0]:.3f}, {CI_a[1]:.3f}] for a and [{CI_b[0]:.3f}, {CI_b[1]:.3f}] for b")
    print("\n")

    dxy, Sxy, lim_sup, lim_inf = Functions.get_Bland_Altman_plot(Power_bc, Power_hc)
    axs2[0, 1].plot((Power_bc + Power_hc)/2, (Power_bc - Power_hc)/2, '*')
    axs2[0, 1].axhline(y = dxy, color = "b")
    axs2[0, 1].axhline(y = lim_sup, linestyle = "-.")
    axs2[0, 1].axhline(y = lim_inf, linestyle = "-.")
    axs2[0, 1].set_title("Multiplicative decay model")

    # For the Excel file
    Col_residuals = Col_residuals + [f"Sum: {sum_residuals:.3f}", f"Max: {max(residuals):.3f}", " ", " "]
    Col_MSE = Col_MSE + [f"MSE = {MSE:.3f}", f"RMSE = {RMSE:.3f}", " ", " "]
    Col_OLP = Col_OLP + [f"b = {b:3f}", f"a = {a:3f}", f"CI_a = [{CI_a[0]:.3f}, {CI_a[1]:.3f}]", f"CI_b = [{CI_b[0]:.3f}, {CI_b[1]:.3f}]"]
    Col_BA = Col_BA + [f"Mean difference is: {dxy:.3f}", f"Deviation is: {2*Sxy:.3f}", " ", " "]


    # Exponential model
    Power_hc = Model_exponential
    residuals = Power_bc - Power_hc
    sum_residuals = sum(residuals)
    print(f"The sum of residuals for the exponential decay model is: {sum_residuals:.3f}") 
    print(f"The maximum residual is: {max(residuals):.3f}")    

    MSE, RMSE = Functions.get_MSE_and_RMSE(Power_bc, Power_hc)
    print(f"MSE for this model is: {MSE:.3f}")
    print(f"RMSE for this model is: {RMSE:.3f}")

    a, b, y_fit, CI_a, CI_b = Functions.olp_line(Power_bc, Power_hc)
    axs[1, 1].scatter(Power_bc, Power_hc, c = 'orange', label = "Data")
    axs[1, 1].plot(Power_bc, Power_bc, c = 'green', label = "Bisecant")
    axs[1, 1].plot(Power_bc, y_fit, c = "red", label = "Interpolating line")
    axs[1, 1].legend(loc = "upper left", fontsize = "small")
    axs[1, 1].set_title(f"Exponential model, a = {a:.2f}, b = {b:.2f}")
    print(f"The parameter of the OLP line are a = {a:.3f}, b = {b:.3f}, with a Confidence Interval at 95% of [{CI_a[0]:.3f}, {CI_a[1]:.3f}] for a and [{CI_b[0]:.3f}, {CI_b[1]:.3f}] for b")
    print("\n")

    dxy, Sxy, lim_sup, lim_inf = Functions.get_Bland_Altman_plot(Power_bc, Power_hc)
    axs2[1, 1].plot((Power_bc + Power_hc)/2, (Power_bc - Power_hc)/2, '*')
    axs2[1, 1].axhline(y = dxy, color = "b")
    axs2[1, 1].axhline(y = lim_sup, linestyle = "-.")
    axs2[1, 1].axhline(y = lim_inf, linestyle = "-.")
    axs2[1, 1].set_title(f"Exponential decay model")

    # For the Excel file
    Col_residuals = Col_residuals + [f"Sum: {sum_residuals:.3f}", f"Max: {max(residuals):.3f}", " ", " "]
    Col_MSE = Col_MSE + [f"MSE = {MSE:.3f}", f"RMSE = {RMSE:.3f}", " ", " "]
    Col_OLP = Col_OLP + [f"b = {b:3f}", f"a = {a:3f}", f"CI_a = [{CI_a[0]:.3f}, {CI_a[1]:.3f}]", f"CI_b = [{CI_b[0]:.3f}, {CI_b[1]:.3f}]"]
    Col_BA = Col_BA + [f"Mean difference is: {dxy:.3f}", f"Deviation is: {2*Sxy:.3f}", " ", " "]


    plt.tight_layout()
    # plt.show()


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
    # Create an excel file
    if Gender == 0:
        Gender = "M"
    else:
        Gender = "F"

    # Linear model
    path = f"Acquisitions\\Protocol\\Processed data\\Input to model" # Location of the output file
    Col_ID = [f"{ID}", " ", " ", " "]
    Col_info = [f"Gender: {Gender}", f"Age: {Age}", f"Weight: {Weight}", f"Height: {Height}"]
    Col_model = ["Linear", " ", " ", " "]
    Col_legend = ["Low", "Medium", "High", "Average"]
    Col_alpha = np.append(Linear_parameters, Linear_averages)
    Col_gamma = np.zeros(4)
    Col_delta_hr = np.zeros(4)
    Col_delta_rpe = np.zeros(4)
    
    # Simple decay
    Col_ID = Col_ID + [" ", " ", " ", " "]
    Col_info = Col_info + [" ", " ", " ", " "]
    Col_model = Col_model + ["Simple decay", " ", " ", " "]
    Col_legend = Col_legend + ["Low", "Medium", "High", "Average"]
    Col_alpha = np.append(Col_alpha, np.concatenate([Simple_decay_parameters[0, :], np.array([Simple_decay_averages[0]])]))
    Col_gamma = np.append(Col_gamma, np.concatenate([Simple_decay_parameters[1, :], np.array([Simple_decay_averages[1]])]))
    Col_delta_hr = np.append(Col_delta_hr, np.zeros(4))
    Col_delta_rpe = np.append(Col_delta_rpe, np.zeros(4))

    # Multiplicative adjustment
    Col_ID = Col_ID + [" ", " ", " ", " "]
    Col_info = Col_info + [" ", " ", " ", " "]
    Col_model = Col_model + ["Multiplicative adjustment", " ", " ", " "]
    Col_legend = Col_legend + ["Low", "Medium", "High", "Average"]
    Col_alpha = np.append(Col_alpha, np.concatenate([Multiplicative_parameters[0, :], np.array([Multiplicative_averages[0]])]))
    Col_gamma = np.append(Col_gamma, np.concatenate([Multiplicative_parameters[1, :], np.array([Multiplicative_averages[1]])]))
    Col_delta_hr = np.append(Col_delta_hr, np.concatenate([Multiplicative_parameters[2, :], np.array([Multiplicative_averages[2]])]))
    Col_delta_rpe = np.append(Col_delta_rpe, np.concatenate([Multiplicative_parameters[1, :], np.array([Multiplicative_averages[3]])]))

    # Exponential adjustment
    Col_ID = Col_ID + [" ", " ", " ", " "]
    Col_info = Col_info + [" ", " ", " ", " "]
    Col_model = Col_model + ["Exponential adjustment", " ", " ", " "]
    Col_legend = Col_legend + ["Low", "Medium", "High", "Average"]
    Col_alpha = np.append(Col_alpha, np.concatenate([Exponential_parameters[0, :], np.array([Exponential_averages[0]])]))
    Col_gamma = np.append(Col_gamma, np.concatenate([Exponential_parameters[1, :], np.array([Exponential_averages[1]])]))
    Col_delta_hr = np.append(Col_delta_hr, np.concatenate([Exponential_parameters[2, :], np.array([Exponential_averages[2]])]))
    Col_delta_rpe = np.append(Col_delta_rpe, np.concatenate([Exponential_parameters[1, :], np.array([Exponential_averages[3]])]))

    df = pd.DataFrame({'ID': Col_ID, 'P info' : Col_info, 'Model': Col_model, 'Intensity': Col_legend, 'alpha': Col_alpha, 'gamma': Col_gamma, 'delta HR': Col_delta_hr, 'delta RPE': Col_delta_rpe, 'Residuals': Col_residuals, "MSE and RMSE": Col_MSE, 'OLP line': Col_OLP, 'Bland-Altman': Col_BA})

    if j == 0:
        Excel_df = df.copy()
    else:
        Excel_df = pd.concat([Excel_df, df], axis = 0)

# Writing a file
path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\Processed Data"
Excel_df.to_excel(f'{path}\\Parameters.xlsx')

print("File saved succesfully!")




