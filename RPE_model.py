import Functions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Feature extraction
import Extract_HR_Features
# Linear Model
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import os

mode = "raw" # filtered, raw or fe (model with feature extraction)
create_file = "No"
feature_extraction = "No"
model = "No"
plots = "Yes"
setup = "handcycle"
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data"
#------------------------------------------------------------------------------------------------------------------------------------
    # Create input File
#------------------------------------------------------------------------------------------------------------------------------------
if create_file == "Yes":
    members = { "000":{"Age": 25, "Height": 160, "Weight": 58, "Gender": 1, "FTP": 49},
            "002":{"Age": 26, "Height": 177, "Weight": 75, "Gender": 0, "FTP": 78},
            "003":{"Age": 22, "Height": 180, "Weight": 70, "Gender": 0, "FTP": 51},
            "004":{"Age": 22, "Height": 186, "Weight": 80, "Gender": 0, "FTP": 47},
            "006":{"Age": 23, "Height": 174, "Weight": 87, "Gender": 1, "FTP": 51},
            "007":{"Age": 23, "Height": 183, "Weight": 70, "Gender": 0, "FTP": 55},
            "008":{"Age": 23, "Height": 190, "Weight": 82, "Gender": 0, "FTP": 82},
            "009":{"Age": 32, "Height": 185, "Weight": 96, "Gender": 0, "FTP": 62},
            "010":{"Age": 24, "Height": 160, "Weight": 56, "Gender": 1, "FTP": 48},
            "011":{"Age": 28, "Height": 176, "Weight": 67, "Gender": 0, "FTP": 60},
            "012":{"Age": 28, "Height": 184, "Weight": 70, "Gender": 0, "FTP": 87},
            "013":{"Age": 25, "Height": 178, "Weight": 66, "Gender": 0, "FTP": 62},
            "015":{"Age": 21, "Height": 176, "Weight": 73, "Gender": 0, "FTP": 60},
            "016":{"Age": 24, "Height": 173, "Weight": 59, "Gender": 1, "FTP": 37},
            }
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f'00{ID}'
        elif ID >= 10:
            ID = f'0{ID}'
        
        data_tmp = pd.read_csv(f"{path}\\{ID}\\Zwift\\{ID}_{setup}_protocol.csv", usecols = ["Power", "Heart Rate", "Cadence", "Distance", "RPE"])

        data_tmp = data_tmp.iloc[300 : 1380, :]

        if mode == "filtered":
            window_size = 15
            window = np.ones(window_size) / window_size
            window = window.flatten()
            Power = data_tmp["Power"]
            Power = pd.DataFrame(np.convolve(Power, window, mode = "same"))
            Cadence = data_tmp["Cadence"]
            cadence = pd.DataFrame(np.convolve(Cadence, window, mode = "same"))
            Heart_rate = data_tmp["Heart Rate"]
            Heart_rate = pd.DataFrame(np.convolve(Heart_rate, window, mode = "same"))
        elif mode == "raw":
            Power = data_tmp["Power"]
            Heart_rate = data_tmp["Heart Rate"]
            cadence = data_tmp["Cadence"]

        # Remove outliers
        Heart_rate = Heart_rate.replace(0, np.nan)    
        Heart_rate = Heart_rate.interpolate(method = "linear").to_numpy().flatten()
        Heart_rate = Heart_rate
        Power = Power.replace(0, np.nan)
        Power = Power.interpolate(method = "linear").to_numpy().flatten()
        cadence = cadence.replace(0, np.nan)
        cadence = cadence.interpolate(method = "linear").to_numpy().flatten()

        # Personal Info
        personal_data = members[f"{ID}"]
        Gender = personal_data["Gender"] * np.ones((len(Heart_rate)))
        Age = personal_data["Age"] * np.ones((len(Heart_rate)))
        PeakHR = (220 - Age) * np.ones((len(Heart_rate)))
        Height = personal_data["Height"] * np.ones((len(Heart_rate)))
        Weight = personal_data["Weight"] * np.ones((len(Heart_rate)))
        FTP = personal_data["FTP"] * np.ones((len(Heart_rate)))
            
        data_tmp = pd.DataFrame({"ID": ID, "Gender": Gender, "Age": Age, "Height": Height, "Weight": Weight, "FTP": FTP, "PeakHR": PeakHR, "Heart Rate": Heart_rate, "Cadence": cadence, "Power": Power, "RPE": data_tmp["RPE"]})

        if i == 0:
            data = data_tmp
        else:
            data = pd.concat([data, data_tmp], axis = 0)
        
        # Save separate files for participants
        writer = pd.ExcelWriter(f'{path}\\Input to models\\Power output models\\{ID}_Input_{setup}_{mode}.xlsx', engine = "openpyxl")
        wb = writer.book
        data_tmp.to_excel(writer, index = False)
        wb.save(f'{path}\\Input to models\\Power output models\\{ID}_Input_{setup}_{mode}.xlsx')

    # Save one file containing all participants
    '''writer = pd.ExcelWriter(f'{path}\\Input to models\\Power output models\\Input_RPE_{setup}_{mode}.xlsx', engine = "openpyxl")
    wb = writer.book
    data.to_excel(writer, index = False)
    wb.save(f'{path}\\Input to models\\Power output models\\Input_RPE_{setup}_{mode}.xlsx')'''

    print("File succesfully saved!")

if feature_extraction == "Yes":
    # Feature Extraction
#----------------------------------------------------------------------------------------------------------------------------------------------
    data = pd.read_excel(f"{path}\\Input to Models\\RPE Models\\Input_RPE_{setup}_{mode}.xlsx")
    Age = data["Age"]
    Heart_rate = data["Heart Rate"]
    Power = data["Power"]
    Cadence = data["Cadence"]

    # 180 second windows on the whole dataset
    window_length = 180
    hr_features = Extract_HR_Features.Extract_HR_Features(Heart_rate, window_length, 220 - Age, 'hr')
    Power_features = Extract_HR_Features.Extract_HR_Features(Power, window_length, 0, 'Power')
    cadence_features = Extract_HR_Features.Extract_HR_Features(Cadence, window_length, 0, 'cadence')

    # Create DataFrame and save Excel File
    n_windows = 1080 / window_length
    P_info = pd.concat([data.iloc[int(1080 * i): int(1080 * i + n_windows)].loc[:, ["ID", "Gender", "Age", "Height", "Weight", "FTP", "RPE"]] 
     for i in range(len(participants))], 
    ignore_index=True)

    df = pd.concat([P_info, hr_features, Power_features, cadence_features], axis = 1)

    writer = pd.ExcelWriter(f'{path}\\Input to models\\RPE Models\\Input_RPE_{setup}_{mode}_feature_extraction_{window_length}.xlsx', engine = "openpyxl")
    wb = writer.book
    df.to_excel(writer, index = False)
    wb.save(f'{path}\\Input to models\\RPE Models\\Input_RPE_{setup}_{mode}_feature_extraction_{window_length}.xlsx')

    # 60 second windows on the whole dataset
    window_length = 60
    hr_features = Extract_HR_Features.Extract_HR_Features(Heart_rate, window_length, 220 - Age, 'hr')
    Power_features = Extract_HR_Features.Extract_HR_Features(Power, window_length, 0, 'Power')
    cadence_features = Extract_HR_Features.Extract_HR_Features(Cadence, window_length, 0, 'cadence')

    # Create DataFrame and save Excel File
    n_windows = 1080 / window_length
    P_info = pd.concat([data.iloc[int(1080 * i): int(1080 * i + n_windows)].loc[:, ["ID", "Gender", "Age", "Height", "Weight", "FTP", "RPE"]] 
     for i in range(len(participants))], 
    ignore_index=True)

    df = pd.concat([P_info, hr_features, Power_features, cadence_features], axis = 1)

    writer = pd.ExcelWriter(f'{path}\\Input to models\\RPE Models\\Input_RPE_{setup}_{mode}_feature_extraction_{window_length}.xlsx', engine = "openpyxl")
    wb = writer.book
    df.to_excel(writer, index = False)
    wb.save(f'{path}\\Input to models\\RPE Models\\Input_RPE_{setup}_{mode}_feature_extraction_{window_length}.xlsx')




    print("Feature extraction files succesfully saved!")

if model == "Yes":
    path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data"

    n_windows = 1
    window_length = int(180/n_windows)

    model_RPE = Functions.modelRPE()
    fig, axs = plt.subplots(4, 4)
    for i, ID in enumerate(participants):
        if mode != "fe":
            data = pd.read_excel(f"{path}\\Input to Models\\RPE Models\\Input_RPE_{setup}_{mode}.xlsx")
        elif mode == "fe":
            data = pd.read_excel(f"{path}\\\\Input to Models\\RPE Models\\Input_RPE_{setup}_{mode}_feature_extraction_{window_length}.xlsx")
        
            # Extract training and test set
        #-----------------------------------------------------------------------------------------------------------------------------
        test_or = data[data["ID"] == ID]
        training_or = data[data["ID"] != ID]

        training = training_or.drop(columns = ["ID"])
        test = test_or.drop(columns = ["ID"])

            # Linear regression Model
        #------------------------------------------------------------------------------------------------------------------------------
        linear_model = LinearRegression.fit(training.iloc[:, training.columns != ["RPE"]], training["RPE"])
        RPE_predicted = linear_model.predict(test.iloc[:, test.columns != "RPE"])
        
            # Visualize results
        #-------------------------------------------------------------------------------------------------------------------------------
        axs[i] = plt.scatter(RPE_predicted)
        axs[i] = plt.scatter(test["RPE"])

        # SVR and PCA?
        training, test, scaler_RPE = model_RPE.preprocessing(training, test) # Removes ID, shuffle and scales datase

#-----------------------------------------------------------------------------------------------------------------------
    # Plots
#-----------------------------------------------------------------------------------------------------------------------
if plots == "Yes":
    mode = "raw"

    # Define figure details
    n_rows = 4
    n_columns = 4
    window_length = 180
    n_windows = int(6 * 180/window_length)

    slopes = np.zeros((len(participants), 7))
    intercepts = np.zeros((len(participants), 7))
    
    fig1, axs1 = plt.subplots(n_rows, n_columns)
    axs1 = axs1.flatten()
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        data_hc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_handcycle_{mode}.xlsx")
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx")

        fig1.suptitle("Handcycle, Power vs HR")
        y = data_hc["Heart Rate"]
        x = data_hc["Power"]
        # Plot and color the dots: red for fixed RPE, blue for fixed Power
        for j, x_value in enumerate(x):
            if j < len(x)/2:
                axs1[i].scatter(x_value, y[j], color = 'r')
            if j >= len(x)/2:
                axs1[i].scatter(x_value, y[j], color = 'b')

        axs1[i].set_ylabel("Heart Rate [bpm]")
        axs1[i].set_xlabel("Power [W]")

        # Find and plot the intercept and compute r^2
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(min(x), max(x), 100) 
        y_line = slope * x + intercept
        axs1[i].plot(x, y_line, color='green') 
        slopes[i, 0] = slope
        intercepts[i, 0] = intercept
        r_squared = r2_score(y, y_line)
        axs1[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")

    plt.tight_layout()
    plt.show()

        # HR vs Power: Bicycle
    #-------------------------------------------------------------------------------------------------------------------
    fig2, axs2 = plt.subplots(n_rows, n_columns)
    axs2 = axs2.flatten()
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        data_hc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_handcycle_{mode}.xlsx")
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx")

        fig2.suptitle("Bicycle, Power vs HR")
        y = data_bc["Heart Rate"]
        x = data_bc["Power"]

        # Plot and color the dots: red for fixed RPE, blue for fixed Power
        for j, x_value in enumerate(x):
            if j < len(x)/2:
                axs2[i].scatter(x_value, y[j], color = 'r')
            if j >= len(x)/2:
                axs2[i].scatter(x_value, y[j], color = 'b')
        
        axs2[i].set_ylabel("Heart Rate [bpm]")
        axs2[i].set_xlabel("Power [W]")

        # Find the intercept
        slope, intercept = np.polyfit(x, y, 1)
        y_line = slope * x + intercept 
        axs2[i].plot(x, y_line, color='green') 
        slopes[i, 1] = slope
        intercepts[i, 1] = intercept
        r_squared = r2_score(y, y_line)
        axs2[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")

    plt.tight_layout()
    plt.show()

        # Power and % of PeakHR: Handcycle
    #--------------------------------------------------------------------------------------------------------------------
    fig3, axs3 = plt.subplots(n_rows, n_columns)
    axs3 = axs3.flatten()
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        data_hc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_handcycle_{mode}.xlsx")
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx")

        fig3.suptitle("Handcycle, Power vs % PeakHR")
        y = data_hc["Heart Rate"]/data_hc["PeakHR"]
        x = data_bc["Power"]

        # Plot and color the dots: red for fixed RPE, blue for fixed Power
        for j, x_value in enumerate(x):
            if j < len(x)/2:
                axs3[i].scatter(x_value, y[j], color = 'r')
            if j >= len(x)/2:
                axs3[i].scatter(x_value, y[j], color = 'b')

        axs3[i].set_ylabel("Heart Rate [bpm]")
        axs3[i].set_xlabel("Power [W]")

        # Find the intercept
        slope, intercept = np.polyfit(x, y, 1)
        y_line = slope * x + intercept 
        axs3[i].plot(x, y_line, color='green') 
        slopes[i, 2] = slope
        intercepts[i, 2] = intercept
        r_squared = r2_score(y, y_line)
        axs3[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")
    
    plt.tight_layout()
    plt.show()

        # Power and % of PeakHR: Bicycle
    #----------------------------------------------------------------------------------------------------------------------
    fig4, axs4 = plt.subplots(n_rows, n_columns)
    axs4 = axs4.flatten()
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        data_hc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_handcycle_{mode}.xlsx")
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx")

        fig4.suptitle("Bicycle, Power vs % PeakHR")
        y = data_bc["Heart Rate"]/data_bc["PeakHR"]
        x = data_bc["Power"]
                # Plot and color the dots: red for fixed RPE, blue for fixed Power
        for j, x_value in enumerate(x):
            if j < len(x)/2:
                axs4[i].scatter(x_value, y[j], color = 'r')
            if j >= len(x)/2:
                axs4[i].scatter(x_value, y[j], color = 'b')
        axs4[i].set_xlabel("Heart Rate [bpm]")
        axs4[i].set_ylabel("Power [W]")

        # Find the intercept
        slope, intercept = np.polyfit(x, y, 1)
        y_line = slope * x + intercept 
        axs4[i].plot(x, y_line, color='green') 
        slopes[i, 3] = slope
        intercepts[i, 3] = intercept
        r_squared = r2_score(y, y_line)
        axs4[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")
    
    plt.tight_layout()
    plt.show()

        # Power vs avgHR: Handcycle
    #-------------------------------------------------------------------------------------------------------------------
    fig5, axs5 = plt.subplots(n_rows, n_columns)
    axs5 = axs5.flatten()
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        data_hc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_handcycle_{mode}.xlsx")
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx")

        fig5.suptitle("Handcycle, Power vs avgHR")
        power = data_hc["Power"]
        # RPE = data_hc["RPE"]
        Heart_rate = data_hc["Heart Rate"]
        y = np.array([np.mean(power[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        x = np.array([np.mean(Heart_rate[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        
        # Plot and color the dots: red for fixed RPE, blue for fixed Power
        for j, x_value in enumerate(x):
            if j < len(x)/2:
                axs5[i].scatter(x_value, y[j], color = 'r')
            if j >= len(x)/2:
                axs5[i].scatter(x_value, y[j], color = 'b')

        axs5[i].set_ylabel("RPE")
        axs5[i].set_xlabel("Power [W]")

        # Find the intercept
        slope, intercept = np.polyfit(x, y, 1)
        y_line = slope * x + intercept 
        axs5[i].plot(x, y_line, color='green') 
        slopes[i, 4] = slope
        intercepts[i, 4] = intercept
        r_squared = r2_score(y, y_line)
        axs5[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")

    plt.tight_layout()
    plt.show()

         # Power vs avgHR: Handcycle
    #------------------------------------------------------------------------------------------------------------------------
    fig6, axs6 = plt.subplots(n_rows, n_columns)
    axs6 =axs6.flatten()
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        data_hc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_handcycle_{mode}.xlsx")
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx")

        fig6.suptitle("Bicycle, Power vs avgHR")
        power = data_bc["Power"]
        # RPE = data_bc["RPE"]
        Heart_rate = data_bc["Heart Rate"]
        y = np.array([np.mean(power[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        x = np.array([np.mean(Heart_rate[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])

        for j, x_value in enumerate(x):
            if j < len(x)/2:
                axs6[i].scatter(x_value, y[j], color = 'r')
            if j >= len(x)/2:
                axs6[i].scatter(x_value, y[j], color = 'b')

        axs6[i].set_ylabel("RPE")
        axs6[i].set_xlabel("Power [W]")

        # Find the intercept and the r^2 value
        slope, intercept = np.polyfit(x, y, 1) 
        y_line = slope * x + intercept 
        axs6[i].plot(x, y_line, color='green') 
        slopes[i, 5] = slope
        intercepts[i, 5] = intercept
        r_squared = r2_score(y, y_line)
        axs6[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")
    
    plt.tight_layout()
    plt.show()

        # Comparing Powers
    #--------------------------------------------------------------------------------------------------------------------------
    fig7, axs7 = plt.subplots(n_rows, n_columns)
    axs7 =axs7.flatten()
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        data_hc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_handcycle_{mode}.xlsx")
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx")

        fig7.suptitle("Handcycle vs bicycle power")
        y = data_hc["Power"]
        x = data_bc["Power"]
        axs7[i].scatter(x, y)
        axs7[i].set_ylabel("Power hc [W]")
        axs7[i].set_xlabel("Power bc [W]")
        slope, intercept = np.polyfit(x, y, 1) 
        y_line = slope * x + intercept 
        axs7[i].plot(x, y_line, color='red') 
        slopes[i, 6] = slope
        intercepts[i, 6] = intercept
        r_squared = r2_score(y, y_line)
        axs7[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")

    plt.tight_layout()
    plt.show()

        # Comparing RPE lines: bicycle and handcycle
    #--------------------------------------------------------------------------------------------------------------------
    fig8, axs8 = plt.subplots(n_rows, n_columns)
    axs8 = axs8.flatten()
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        data_hc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_handcycle_{mode}.xlsx")
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx")

        fig8.suptitle("Handcycle vs bicycle power")
        power = data_hc["Power"]
        RPE = data_hc["RPE"]
        y = np.array([np.mean(power[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        x = np.array([np.mean(RPE[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])

        for j, x_value in enumerate(x):
            if j < len(x)/2:
                axs6[i].scatter(x_value, y[j], color = 'r')
            if j >= len(x)/2:
                axs6[i].scatter(x_value, y[j], color = 'b')

        axs8[i].set_ylabel("Power [W]")
        axs8[i].set_xlabel("RPE")

        # Find the intercept and the r^2 value
        slope, intercept = np.polyfit(x, y, 1) 
        y_line = slope * x + intercept 
        axs8[i].plot(x, y_line, color='red') 

        # Bicycle data
        power = data_bc["Power"]
        RPE = data_bc["RPE"]
        y = np.array([np.mean(power[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        x = np.array([np.mean(RPE[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])

        for j, x_value in enumerate(x):
            if j < len(x)/2:
                axs6[i].scatter(x_value, y[j], color = 'g')
            if j >= len(x)/2:
                axs6[i].scatter(x_value, y[j], color = 'k')

        axs8[i].set_ylabel("Power [W]")
        axs8[i].set_xlabel("RPE")

        # Find the intercept and the r^2 value
        slope, intercept = np.polyfit(x, y, 1) 
        y_line = slope * x + intercept 
        axs8[i].plot(x, y_line, color='green') 
        axs8[i].set_title(f"Participant {ID}")

    plt.tight_layout()
    plt.show()
