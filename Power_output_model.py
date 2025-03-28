import pandas as pd
import numpy as np
import Functions
# For creating the linear regression model
from sklearn.linear_model import LinearRegression
# Visualising data
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------------------------------------------------
    # Importing data and creating dataset
#----------------------------------------------------------------------------------------------------------------------------------
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
Skip = "Yes"
# Set it to No if the Input file needs to be recreated

if  Skip == "No":
    path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol"
    # path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\\RPE Models"


    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        elif ID >= 10:
            ID = f"0{ID}"

        # Correct gender participant 2

        # Import data and remove warmup and cooldown
        data_tmp = pd.read_csv(f"{path}\\{ID}\\Zwift\\{ID}_handcycle_protocol.csv", usecols = ["Power", "Heart Rate", "Cadence", "RPE"])
        data_bc = pd.read_csv(f"{path}\\{ID}\\Zwift\\{ID}_bicycle_protocol.csv", usecols = ["Power", "RPE"])
        
        data_tmp = data_tmp.iloc[300 : 1380, :]
        data_bc = data_bc.iloc[300 : 1380, :]

        # Filtering Power
        window_size = 15
        window = np.ones(window_size) / window_size
        window = window.flatten()
        Power_hc = data_tmp.iloc[:, 2]
        Power_hc = np.convolve(Power_hc, window, mode = "same")
        Power_bc = data_bc.iloc[:, 0]
        Power_bc = np.convolve(Power_bc, window, mode = "same")

        # Create P info columns
        Gender = int(input(f"What is participant {ID}'s gender? (0 = male, 1 = female) "))
        Age = int(input(f"What is participant {ID}'s age? "))
        Height = int(input(f"What is participant {ID}'s height? "))
        Weight = int(input(f"What is participant {ID}'s weight? "))

        Gender = Gender * np.ones((len(data_tmp)))
        Age = Age * np.ones((len(data_tmp)))
        Height = Height * np.ones((len(data_tmp)))
        Weight = Weight * np.ones((len(data_tmp)))
    

        # Create DataFrame
        df = pd.DataFrame({"ID": ID, "Gender": Gender, "Age": Age, "Height": Height, "Weight": Weight, "Heart Rate": data_tmp.iloc[:, 0], "Cadence": data_tmp.iloc[:, 1], "Power hc": Power_hc, "Power bc": Power_bc, "RPE hc": data_tmp.iloc[:, 3], "RPE bc": data_bc.iloc[:, 1]})

        if i == 0:
            data = df
        else:
            data = pd.concat([data, df], axis = 0)

    path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Power output model\\Input files"
    writer = pd.ExcelWriter(f'{path}\\Input_file_power_model.xlsx', engine = "openpyxl")
    wb = writer.book

    data.to_excel(writer, index = False)
    wb.save(f'{path}\\Input_file_power_model.xlsx')


#-----------------------------------------------------------------------------------------------------------------------------------
    # Modeling: Leave-p-out method for validation
#-----------------------------------------------------------------------------------------------------------------------------------
path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Power output model\\Input files"
data = pd.read_excel(f"{path}\\Input_file_power_model.xlsx")

    # Inizializing useful methods
#-----------------------------------------------------------------------------------------------------------------------------------
Power_model = Functions.PowerOutputModel()

for i, ID in enumerate(participants):

        # Extracting test set before scaling and shuffling
#-----------------------------------------------------------------------------------------------------------------------------------
    test_or = data[data["ID"] == ID]
    training_or = data[data["ID"] != ID]
    
    # Processing the data: Removing ID and saving Power bc
#-----------------------------------------------------------------------------------------------------------------------------------
    training_or = training_or.drop(columns = ["ID", "RPE bc"])
    test_or = test_or.drop(columns = ["ID", "RPE bc"])

    training, test, scaler = Power_model.preprocessing(training_or, test_or)

    Power_bc = scaler.inverse_transform(training)
    Power_bc = Power_bc[:, 7]
    training = training.drop(columns = "Power bc")

    bc_test = scaler.inverse_transform(test)
    Power_bc_test = bc_test[:, 7]
    Power_hc_test = bc_test[:, 6]
    test = test.drop(columns = ["Power bc"])

    # Linear regression model
#-----------------------------------------------------------------------------------------------------------------------------------
    linear_regression = LinearRegression().fit(training, Power_bc)
    predicted = linear_regression.predict(test)

    plt.figure()
    plt.plot(predicted)
    plt.plot(Power_bc_test)
    plt.plot(Power_hc_test)

    plt.legend(["Linear regression", "Original bc signal", "Original hc signal"])
    plt.title(f"Linear regression, participant {ID}")
    plt.xlabel("Time [s]")
    plt.ylabel("Power [W]")

    plt.show()

final = "Yake"

# Brainstorming: PCA sul dataset per vedere se spiega la variabilità
# SVR per vedere se migliora il modello



    


