import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import Functions

matplotlib.use('TkAgg')


#---------------------------------------------------------------------------------------------------------------------------------------
    # Importing data
#---------------------------------------------------------------------------------------------------------------------------------------

subjects = [6]
# subjects = [3]

for i in range(len(subjects)):
    ID = subjects[i]
    if ID < 10:
        ID = f"00{ID}"
    elif ID >= 10:
        ID = f"0{ID}"

    file = f"C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\\{ID}_input_file.xlsx"

    dataset = Functions.DataSet(file)

#---------------------------------------------------------------------------------------------------------------------------------------
    # Linear Model
#---------------------------------------------------------------------------------------------------------------------------------------
    mode = "false"
    gender = "female" # false for a mixed gender model, female for a female based model, male for a male based model

    if gender == "false":
        alpha = 2.675603149893744
    if gender == "female":
        alpha =   2.6599925611289836
    
    fig, axs = plt.subplots(2)
    plt.suptitle(f"Participant {ID}")
    validate_model = Functions.ValidateModel(dataset)
    model_linear = validate_model.implement_model("linear", alpha, 0, fig, axs, "true")

#---------------------------------------------------------------------------------------------------------------------------------------
    # Linear regression model
#---------------------------------------------------------------------------------------------------------------------------------------
    linear_regression = Functions.LinearRegression()

    if gender == "false":
        if mode == "true":
            X = [[ 8.62569535e-01,  4.94533423e-02,  8.97969232e+00,
         4.29083596e-01, -2.19196040e-02, -8.81161850e-01,
        -2.95636520e-01],
       [ 8.69029459e-01,  4.94319110e-02,  8.90594540e+00,
         4.30705418e-01, -9.96173364e-06, -9.76010770e-01,
        -6.19086872e-07],
       [ 1.36217426e+00,  8.18448235e-02,  2.44095691e+00,
         5.61195679e-01,  8.58172073e-04, -3.28475632e-01,
        -8.14054020e-03]]
        elif mode == "false":
            X = [[ 1.61671835e+00,  1.80666352e-02,  6.36123523e-01,
        -6.96306918e-04, -4.14043907e-02, -2.74057532e-03],
       [ 1.61579492e+00,  1.91785701e-02,  6.35943311e-01,
         1.20824653e-04, -3.98681484e-02, -5.03257686e-09],
       [ 1.56392278e+00,  7.98279954e-02,  6.15292233e-01,
        -1.03094531e-04, -9.07583377e-02, -2.36457455e-03]]
    
    if gender == "female":
        X = [[ 1.17912928e+00, -3.33736375e-01,  1.01302120e+00,
         9.30621491e-03,  3.17268863e-01,  1.26630758e-01],
       [ 1.17569534e+00, -3.31578659e-01,  1.01362843e+00,
         1.30520626e-04,  3.65024020e-01, -5.28567384e-09],
       [ 1.12280979e+00, -3.17860447e-01,  1.02404290e+00,
        -6.43958247e-03,  3.55786940e-01, -4.47530952e-09]]


    model_regression = validate_model.implement_model("linear regression", X, 1, fig, axs, mode)

    plt.legend()

plt.show()

# Check if the difference between the power output in the second part of the workout is consistent with
# the difference in the reported RPE value
