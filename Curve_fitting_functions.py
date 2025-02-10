import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import pandas as pd
import Functions
import mplcursors

matplotlib.use('TkAgg')

subjects = [0, 10, 16]

# Save the final parameters for the model
Alpha_linear = np.zeros((len(subjects)))

Alpha_simple = np.zeros((len(subjects)))
Gamma_simple = np.zeros((len(subjects)))


for j, ID in enumerate(subjects):
    if ID < 10:
        ID = f"00{ID}"
    elif ID >= 10:
        ID = f"0{ID}"
    
    file = f"C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\\{ID}_input_file.xlsx"

    # Import data
    dataset = Functions.DataSet(file)

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
    Alpha_linear[j] = linear_model.alpha

    # Plot the signal
    # linear_model.plot(ID, dataset.time, dataset.Power_bc, model_linear)


    
    # Simple decay model
#---------------------------------------------------------------------------------------------------------------------------------
    simple_model = Functions.SimpleDecay() # linear_model.fig, linear_model.axs is passed as an argument of plot is active
    x = np.zeros((2, len(dataset.Power_hc)))
    '''x[0, :] = dataset.time
    x[1, :] = dataset.Power_hc'''

    model_simple = simple_model.fit(dataset)

    Parameters_simple = np.zeros((2, len(simple_model.alpha_array)))
    Parameters_simple[0, :] = simple_model.alpha_array
    Parameters_simple[1, :] = simple_model.gamma_array
    Averages_simple = [simple_model.alpha, simple_model.gamma]
    Alpha_simple[j], Gamma_simple[j] = Averages_simple

    # Plot the signal
    # simple_model.plot(ID, dataset.time, dataset.Power_bc, model_simple)


#--------------------------------------------------------------------------------------------------------------------------------
    # Linear regression model
#--------------------------------------------------------------------------------------------------------------------------------
    linear_regression = Functions.LinearRegression()

    A, b, lam = linear_regression.create_matrices(dataset, "false") # true if RPE is considered as a variable, false if it's not
    X = linear_regression.regression()
    model_regression = A@X[1, :]

    '''linear_regression.plot_regression(A, , "true")
    plt.tight_layout()
    plt.legend()
    plt.show()'''




#--------------------------------------------------------------------------------------------------------------------------------
    # Statistical analysis
#--------------------------------------------------------------------------------------------------------------------------------
    statistical_analysis = Functions.StatisticalAnalysis()

    # Linear Model
#--------------------------------------------------------------------------------------------------------------------------------
    model = model_linear
    # OLP line
    olp = statistical_analysis.olp_line(dataset.Power_bc, model)
    '''fig2, axs2 = plt.subplots(2, 2)
    statistical_analysis.plot_olp(olp.b, olp.a, dataset.Power_bc, model, olp.y_fit, "Linear model", [0, 0], fig2, axs2)
    plt.suptitle(f"OLP line, subject {ID}")'''
    olp_linear = olp

    # MSE and RMSE
    MSE_linear = statistical_analysis.get_MSE_and_RMSE(dataset.Power_bc, model)

    # Bland-Altman
    ba = statistical_analysis.bland_altman(dataset.Power_bc, model)

    # fig3, axs3 = plt.subplots(2, 2)
    '''plt.suptitle(f"Bland-Altman plot, participant {ID}")
    statistical_analysis.plot_ba(dataset.Power_bc, model, "Linear", [0, 0], fig3, axs3)'''

    ba_linear = ba




    # Simple Decay model
#----------------------------------------------------------------------------------------------------------------------------------
    model = model_simple
    # OLP Line
    olp = statistical_analysis.olp_line(dataset.Power_bc[0 : 540], model)
    # statistical_analysis.plot_olp(olp.b, olp.a, dataset.Power_bc[0 : 540], model, olp.y_fit, "Simple decay", [0, 1], fig2, axs2)

    olp_simple = olp


    # MSE and RMSE
    MSE_simple = statistical_analysis.get_MSE_and_RMSE(dataset.Power_bc[0 : 540], model)


    # Bland-Altman
    ba = statistical_analysis.bland_altman(dataset.Power_bc[0 : 540], model)

    '''plt.suptitle(f"Bland-Altman plot, participant {ID}")
    statistical_analysis.plot_ba(dataset.Power_bc, model, "Simple decay", [0, 1], fig3, axs3)'''

    ba_simple = ba


    
    # Linear regression
#-------------------------------------------------------------------------------------------------------------------------------
    model = model_regression
    olp = statistical_analysis.olp_line(dataset.Power_bc, model)
    # statistical_analysis.plot_olp(olp.b, olp.a, dataset.Power_bc, model, olp.y_fit, "Multiplicative adjustment decay", [1, 1], fig2, axs2)

    olp_regression = olp


    # MSE and RMSE
    MSE_regression = statistical_analysis.get_MSE_and_RMSE(dataset.Power_bc, model)


    # Bland-Altman
    ba = statistical_analysis.bland_altman(dataset.Power_bc, model)
    # ax = statistical_analysis.plot_ba(dataset.Power_bc, model, "complete adjustment decay", [1, 1], fig3, axs3)

    ba_regression = ba

#--------------------------------------------------------------------------------------------------------------------------------
    # Computing the final parameters for the model
#--------------------------------------------------------------------------------------------------------------------------------
    if j == 0:
        X_total = X
    else:
        X_total = X_total + X

# Linear model
alpha = np.mean(Alpha_linear)
print("\n")
print(f"alpha for the simple linear model is: {alpha}")

# Linear regression model
X_mean = X_total/len(subjects)
X_mean = repr(X_mean)
print("\n")
print("Matrix of coefficient for linear regression model is: ")
print(X_mean)

