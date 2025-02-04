import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import pandas as pd
import Functions
import mplcursors

matplotlib.use('TkAgg')

# subjects = [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
subjects = [3]
path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models"

for j, ID in enumerate(subjects):
    if ID < 10:
        ID = f"00{ID}"
    elif ID >= 10:
        ID = f"0{ID}"

    # Import data
    dataset = Functions.DataSet(path, ID)

#-------------------------------------------------------------------------------------------------------------------------------
    # MODELS
#-------------------------------------------------------------------------------------------------------------------------------

    
    # Linear Model
#--------------------------------------------------------------------------------------------------------------------------------
    linear_model = Functions.LinearModel()
    model = linear_model.fit(dataset)

    # Save the parameters
    model_linear = linear_model.predict(dataset.Power_hc)
    Parameters_linear = linear_model.alpha_array
    Averages_linear = linear_model.alpha

    # Plot the signal
    linear_model.plot(ID, dataset.time, dataset.Power_bc, model_linear)


    
    # Simple decay model
#---------------------------------------------------------------------------------------------------------------------------------
    simple_model = Functions.SimpleDecay(linear_model.fig, linear_model.axs)
    x = np.zeros((2, len(dataset.Power_hc)))
    x[0, :] = dataset.time
    x[1, :] = dataset.Power_hc

    model_simple = simple_model.fit(dataset)

    Parameters_simple = np.zeros((2, len(simple_model.alpha_array)))
    Parameters_simple[0, :] = simple_model.alpha_array
    Parameters_simple[1, :] = simple_model.gamma_array
    Averages_simple = [simple_model.alpha, simple_model.gamma]

    # Plot the signal
    simple_model.plot(ID, dataset.time, dataset.Power_bc, model_simple)


    
    # Multiplicative adjustment model
#---------------------------------------------------------------------------------------------------------------------------------
    multiplicative_model = Functions.MultiplicativeDecay(linear_model.fig, linear_model.axs)
    x = np.zeros((4, len(dataset.Power_hc)))
    x[0, :] = dataset.time
    x[1, :] = dataset.Power_hc
    x[2, :] = dataset.HR
    x[3, :] = dataset.RPE

    model_multiplicative = multiplicative_model.fit(dataset)

    # Save values
    Parameters_multiplicative = np.zeros((4, len(multiplicative_model.alpha_array)))
    Parameters_multiplicative[0, :] = multiplicative_model.alpha_array
    Parameters_multiplicative[1, :] = multiplicative_model.gamma_array
    Parameters_multiplicative[2, :] = multiplicative_model.delta_hr_array
    Parameters_multiplicative[3, :] = multiplicative_model.delta_rpe_array

    Averages_multiplicative = [multiplicative_model.alpha, multiplicative_model.gamma, multiplicative_model.delta_hr, multiplicative_model.delta_rpe]

    # Plot the figure
    multiplicative_model.plot(ID, dataset.time, dataset.Power_bc, model_multiplicative)
   



    # Exponential adjustment model
#----------------------------------------------------------------------------------------------------------------------------------------------------
    exponential_model = Functions.ExponentialDecay(linear_model.fig, linear_model.axs)
    x = np.zeros((4, len(dataset.Power_hc)))
    x[0, :] = dataset.time
    x[1, :] = dataset.Power_hc
    x[2, :] = dataset.HR
    x[3, :] = dataset.RPE

    model_exponential = exponential_model.fit(dataset)

    # Save values
    Parameters_exponential = np.zeros((4, len(exponential_model.alpha_array)))
    Parameters_exponential[0, :] = exponential_model.alpha_array
    Parameters_exponential[1, :] = exponential_model.gamma_array
    Parameters_exponential[2, :] = exponential_model.delta_hr_array
    Parameters_exponential[3, :] = exponential_model.delta_rpe_array

    Averages_multiplicative = [exponential_model.alpha, exponential_model.gamma, exponential_model.delta_hr, exponential_model.delta_rpe]

    # Plot the figure
    exponential_model.plot(ID, dataset.time, dataset.Power_bc, model_exponential)
   
    plt.tight_layout()
    # plt.show()


#--------------------------------------------------------------------------------------------------------------------------------
    # Statistical analysis
#--------------------------------------------------------------------------------------------------------------------------------
    statistical_analysis = Functions.StatisticalAnalysis()

    # Linear Model
#--------------------------------------------------------------------------------------------------------------------------------
    model = model_linear
    # OLP line
    olp = statistical_analysis.olp_line(dataset.Power_bc, model)
    fig2, axs2 = plt.subplots(2, 2)
    statistical_analysis.plot_olp(olp.b, olp.a, dataset.Power_bc, model, olp.y_fit, "Linear model", [0, 0], fig2, axs2)
    plt.suptitle(f"OLP line, subject {ID}")
    olp_linear = olp

    # MSE and RMSE
    MSE_linear = statistical_analysis.get_MSE_and_RMSE(dataset.Power_bc, model)


    # Bland-Altman
    ba = statistical_analysis.bland_altman(dataset.Power_bc, model)

    fig3, axs3 = plt.subplots(2, 2)
    plt.suptitle(f"Bland-Altman plot, participant {ID}")
    statistical_analysis.plot_ba(dataset.Power_bc, model, "Linear", [0, 0], fig3, axs3)

    ba_linear = ba




    # Simple Decay model
#----------------------------------------------------------------------------------------------------------------------------------
    model = model_simple
    # OLP Line
    olp = statistical_analysis.olp_line(dataset.Power_bc, model)
    statistical_analysis.plot_olp(olp.b, olp.a, dataset.Power_bc, model, olp.y_fit, "Simple decay", [0, 1], fig2, axs2)

    olp_simple = olp


    # MSE and RMSE
    MSE_simple = statistical_analysis.get_MSE_and_RMSE(dataset.Power_bc, model)


    # Bland-Altman
    ba = statistical_analysis.bland_altman(dataset.Power_bc, model)

    plt.suptitle(f"Bland-Altman plot, participant {ID}")
    statistical_analysis.plot_ba(dataset.Power_bc, model, "Simple decay", [0, 1], fig3, axs3)

    ba_simple = ba



    # Mutliplicative adjustment decay model
#----------------------------------------------------------------------------------------------------------------------------------
    model = model_multiplicative
    # OLP Line
    olp = statistical_analysis.olp_line(dataset.Power_bc, model)
    statistical_analysis.plot_olp(olp.b, olp.a, dataset.Power_bc, model, olp.y_fit, "Multiplicative adjustment decay", [1, 0], fig2, axs2)

    olp_multiplicative = olp

    # MSE and RMSE
    MSE_multiplicative = statistical_analysis.get_MSE_and_RMSE(dataset.Power_bc, model)


    # Bland-Altman
    ba = statistical_analysis.bland_altman(dataset.Power_bc, model)
    statistical_analysis.plot_ba(dataset.Power_bc, model, "Multiplicative adjustment decay", [1, 0], fig3, axs3)

    ba_multiplicative = ba


    # Exponential adjustment decay model
#----------------------------------------------------------------------------------------------------------------------------------
    model = model_exponential
    # OLP Line
    olp = statistical_analysis.olp_line(dataset.Power_bc, model)
    statistical_analysis.plot_olp(olp.b, olp.a, dataset.Power_bc, model, olp.y_fit, "Multiplicative adjustment decay", [1, 1], fig2, axs2)

    olp_exponential = olp


    # MSE and RMSE
    MSE_exponential = statistical_analysis.get_MSE_and_RMSE(dataset.Power_bc, model)


    # Bland-Altman
    ba = statistical_analysis.bland_altman(dataset.Power_bc, model)
    ax = statistical_analysis.plot_ba(dataset.Power_bc, model, "Exponential adjustment decay", [1, 1], fig3, axs3)

    ba_multiplicative = ba

    mplcursors.cursor(ax, hover = 'True')
    plt.tight_layout()
    plt.show()

    