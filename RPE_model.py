import Functions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import math
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Feature extraction
import Extract_HR_Features
# Linear Model
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import os
from matplotlib.lines import Line2D

mode = "raw" # filtered, raw or fe (model with feature extraction)
create_file = "Yes"
feature_extraction = "No"
model = "No"
plots = "No"
model_perc_hr = "No"
setup = "handcycle"
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data"
#------------------------------------------------------------------------------------------------------------------------------------
    # Create input File
#------------------------------------------------------------------------------------------------------------------------------------
if create_file == "Yes":
    members = { "000":{"Age": 25, "Height": 160, "Weight": 58, "Gender": 1, "FTP": 49},#, "Activity": [0, 0, 60*7, 16*60*7]},
            "002":{"Age": 26, "Height": 177, "Weight": 75, "Gender": 0, "FTP": 78},# "Activity": [0, 4*45, 6*30, 4*60*7]},
            "003":{"Age": 22, "Height": 180, "Weight": 70, "Gender": 0, "FTP": 51},# "Activity": [4*60, 7*15, 3*30, 8*60*7]},
            "004":{"Age": 22, "Height": 186, "Weight": 80, "Gender": 0, "FTP": 47},# "Activity": [3*90, 1*5, 5*30, 6*60*7]},
            "006":{"Age": 23, "Height": 174, "Weight": 87, "Gender": 1, "FTP": 51},# "Activity": [2*40, 0, 7*120, 9*60*7 ]},
            "007":{"Age": 23, "Height": 183, "Weight": 70, "Gender": 0, "FTP": 55},# "Activity": [0, 0, 60*7, 16*60*7]}, # Unsure about the sitting time
            "008":{"Age": 23, "Height": 190, "Weight": 82, "Gender": 0, "FTP": 82},# "Activity": [4*90, 0, 7*30, 8*60*7]},
            "009":{"Age": 32, "Height": 185, "Weight": 96, "Gender": 0, "FTP": 62},# "Activity": [0, 0, 5*60, 12*60*7]},
            "010":{"Age": 24, "Height": 160, "Weight": 56, "Gender": 1, "FTP": 48},# "Activity": [5*60, 2*60, 7*30, 10*60*7]}, # Time spent for activity is grossly estimated
            "011":{"Age": 28, "Height": 176, "Weight": 67, "Gender": 0, "FTP": 60},# "Activity": [0, 0, 7*40, 12*60*7]},
            "012":{"Age": 28, "Height": 184, "Weight": 70, "Gender": 0, "FTP": 87},# "Activity": [0, 0, 4*20, 7*45, 10*60*7]},
            "013":{"Age": 25, "Height": 178, "Weight": 66, "Gender": 0, "FTP": 62},# "Activity": [1*60, 1*60, 3*35, 4*60*7]},
            "015":{"Age": 21, "Height": 176, "Weight": 73, "Gender": 0, "FTP": 60},# "Activity": [4*80, 6*120, 7*8*60, 3*60*7]},
            "016":{"Age": 24, "Height": 173, "Weight": 59, "Gender": 1, "FTP": 37},# "Activity": [2*45, 2*30, 7*60, 9*60*7]},
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
class Plots():
    def __init__(self):
        pass
    
    def plot_graphs(self, x, y, title:str, x_label:str, y_label:str, ID, i, fig, axs, color_coded:str):
        if color_coded == "No": 
            # Plot the dots
            axs[i].scatter(x, y, color='b', s = 10)

            # Find the intercept and plot it
            slope, intercept = np.polyfit(x, y, 1) 
            y_line = slope * x + intercept 
            axs[i].plot(x, y_line, color='k') 

            # Set figure properties
            axs[i].set_xlabel(x_label)
            axs[i].set_ylabel(y_label)
            if intercept >= 0:
                axs[i].set_title(f"{ID}: y = {slope:.3f} * x + {intercept:.3f}")
            else:
                axs[i].set_title(f"{ID}: y = {slope:.3f} * x - {abs(intercept):.3f}")
            fig.suptitle(title)

        return slope, intercept

draw_plots = Plots()

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
        x_red = x[:len(x)//2]
        y_red = y[:len(y)//2]

        x_blue = x[len(x)//2:]
        y_blue = y[len(y)//2:]

        axs1[i].scatter(x_red, y_red, color='r', label = "Fixed RPE")
        axs1[i].scatter(x_blue, y_blue, color='b', label = "Fixed power")

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

    handles, labels = axs1[i].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower right')
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
        x_red = x[:len(x)//2]
        y_red = y[:len(y)//2]

        x_blue = x[len(x)//2:]
        y_blue = y[len(y)//2:]

        axs2[i].scatter(x_red, y_red, color='r', label = "Fixed RPE")
        axs2[i].scatter(x_blue, y_blue, color='b', label = "Fixed power")
        
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

    handles, labels = axs2[i].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower right')
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
        y = data_hc["Heart Rate"]/data_hc["PeakHR"] * 100
        x = data_hc["Power"]

        # Plot and color the dots: red for fixed RPE, blue for fixed Power
        x_red = x[:len(x)//2]
        y_red = y[:len(y)//2]

        x_blue = x[len(x)//2:]
        y_blue = y[len(y)//2:]

        axs3[i].scatter(x_red, y_red, color='r', label = "Fixed RPE")
        axs3[i].scatter(x_blue, y_blue, color='b', label = "Fixed power")

        axs3[i].set_ylabel("% Peak HR")
        axs3[i].set_xlabel("Power [W]")

        # Find the intercept
        slope, intercept = np.polyfit(x, y, 1)
        y_line = slope * x + intercept 
        axs3[i].plot(x, y_line, color='green') 
        slopes[i, 2] = slope
        intercepts[i, 2] = intercept
        r_squared = r2_score(y, y_line)
        axs3[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")
    
    handles, labels = axs3[i].get_legend_handles_labels()
    fig3.legend(handles, labels, loc='lower right')
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

        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx")

        fig4.suptitle("Bicycle, Power vs % PeakHR")
        x = data_bc["Heart Rate"]/data_bc["PeakHR"] * 100
        y = data_bc["Power"]
        # Plot and color the dots: red for fixed RPE, blue for fixed Power
        x_red = x[:len(x)//2]
        y_red = y[:len(y)//2]

        x_blue = x[len(x)//2:]
        y_blue = y[len(y)//2:]

        axs4[i].scatter(x_red, y_red, color='r', label = "Fixed RPE")
        axs4[i].scatter(x_blue, y_blue, color='b', label = "Fixed power")

        axs4[i].set_ylabel("% Peak HR")
        axs4[i].set_xlabel("Power [W]")

        # Find the intercept
        slope, intercept = np.polyfit(x, y, 1)
        y_line = slope * x + intercept 
        axs4[i].plot(x, y_line, color='green') 
        slopes[i, 3] = slope
        intercepts[i, 3] = intercept
        r_squared = r2_score(y, y_line)
        axs4[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")
    
    handles, labels = axs4[i].get_legend_handles_labels()
    fig4.legend(handles, labels, loc='lower right')
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
        x = np.array([np.mean(power[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        y = np.array([np.mean(Heart_rate[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        
        # Plot and color the dots: red for fixed RPE, blue for fixed Power
        x_red = x[:len(x)//2]
        y_red = y[:len(y)//2]

        x_blue = x[len(x)//2:]
        y_blue = y[len(y)//2:]

        axs5[i].scatter(x_red, y_red, color='r', label = "Fixed RPE")
        axs5[i].scatter(x_blue, y_blue, color='b', label = "Fixed power")

        axs5[i].set_ylabel("Average HR [bpm]")
        axs5[i].set_xlabel("Power [W]")

        # Find the intercept
        slope, intercept = np.polyfit(x, y, 1)
        y_line = slope * x + intercept 
        axs5[i].plot(x, y_line, color='green') 
        slopes[i, 4] = slope
        intercepts[i, 4] = intercept
        r_squared = r2_score(y, y_line)
        axs5[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")

    handles, labels = axs5[i].get_legend_handles_labels()
    fig5.legend(handles, labels, loc='lower right')
    plt.tight_layout()
    plt.show()

         # Power vs avgHR: Bicycle
    #------------------------------------------------------------------------------------------------------------------------
    fig6, axs6 = plt.subplots(n_rows, n_columns)
    axs6 = axs6.flatten()
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
        x = np.array([np.mean(power[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        y = np.array([np.mean(Heart_rate[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])

        x_red = x[:len(x)//2]
        y_red = y[:len(y)//2]

        x_blue = x[len(x)//2:]
        y_blue = y[len(y)//2:]

        axs6[i].scatter(x_red, y_red, color='r', label = "Fixed RPE")
        axs6[i].scatter(x_blue, y_blue, color='b', label = "Fixed power")

        axs6[i].set_ylabel("Avg HR [bpm]")
        axs6[i].set_xlabel("Power [W]")

        # Find the intercept and the r^2 value
        slope, intercept = np.polyfit(x, y, 1) 
        y_line = slope * x + intercept 
        axs6[i].plot(x, y_line, color='green') 
        slopes[i, 5] = slope
        intercepts[i, 5] = intercept
        r_squared = r2_score(y, y_line)
        axs6[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")
    
    handles, labels = axs6[i].get_legend_handles_labels()
    fig6.legend(handles, labels, loc='lower right')
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

        x_red = x[:len(x)//2]
        y_red = y[:len(y)//2]

        x_blue = x[len(x)//2:]
        y_blue = y[len(y)//2:]

        axs7[i].scatter(x_red, y_red, color='r', label = "Fixed RPE")
        axs7[i].scatter(x_blue, y_blue, color='b', label = "Fixed power")

        axs7[i].set_ylabel("Power hc [W]")
        axs7[i].set_xlabel("Power bc [W]")

        # Find the intercept
        slope, intercept = np.polyfit(x, y, 1) 
        y_line = slope * x + intercept 
        axs7[i].plot(x, y_line, color='green') 
        slopes[i, 6] = slope
        intercepts[i, 6] = intercept
        r_squared = r2_score(y, y_line)
        axs7[i].set_title(f"{ID}: r^2 = {r_squared:.2f}")

    handles, labels = axs7[i].get_legend_handles_labels()
    fig7.legend(handles, labels, loc='lower right')
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

        fig8.suptitle("Comparing RPE vs Power intercepts")
        power = data_hc["Power"]
        RPE = data_hc["RPE"]
        x = np.array([np.mean(power[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        y = np.array([np.mean(RPE[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])

        x_red = x[:len(x)//2]
        y_red = y[:len(y)//2]

        x_blue = x[len(x)//2:]
        y_blue = y[len(y)//2:]

        axs8[i].scatter(x_red, y_red, color='r', label = "Fixed RPE hc")
        axs8[i].scatter(x_blue, y_blue, color='y', label = "Fixed power hc")

        axs8[i].set_xlabel("Power [W]")
        axs8[i].set_ylabel("RPE")

        # Find the intercept and the r^2 value
        slope, intercept = np.polyfit(x, y, 1) 
        y_line = slope * x + intercept 
        axs8[i].plot(x, y_line, color='orange', label = "handcycle") 

        # Bicycle data
        power = data_bc["Power"]
        RPE = data_bc["RPE"]
        x = np.array([np.mean(power[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])
        y = np.array([np.mean(RPE[i * window_length : (i + 1) * window_length]) for i in range(n_windows)])

        x_red = x[:len(x)//2]
        y_red = y[:len(y)//2]

        x_blue = x[len(x)//2:]
        y_blue = y[len(y)//2:]

        axs8[i].scatter(x_red, y_red, color='k', label = "Fixed RPE bc")
        axs8[i].scatter(x_blue, y_blue, color='b', label = "Fixed power hc")

        axs8[i].set_xlabel("Power [W]")
        axs8[i].set_ylabel("RPE")

        # Find the intercept and the r^2 value
        slope, intercept = np.polyfit(x, y, 1) 
        y_line = slope * x + intercept 
        axs8[i].plot(x, y_line, color='green', label = "bicycle") 
        axs8[i].set_title(f"Participant {ID}")

    handles, labels = axs8[i].get_legend_handles_labels()
    fig8.legend(handles, labels, loc='lower right')
    plt.tight_layout()
    plt.show()

if model_perc_hr == "Yes":
    # Create P info columns
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

    fig1, axs1 = plt.subplots(4, 4)
    axs1 = axs1.flatten()
    fig2, axs2 = plt.subplots(4, 4)
    axs2 = axs2.flatten()
    fig3, axs3 = plt.subplots(4, 4)
    axs3 = axs3.flatten()

    slopes = np.zeros((len(participants), 2))
    intercepts = np.zeros((len(participants), 2))
    slopes_model = np.zeros((len(participants)))
    intercepts_model = np.zeros((len(participants)))

    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        data_hc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_handcycle_{mode}.xlsx", engine="openpyxl")
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power output models\\{ID}_Input_bicycle_{mode}.xlsx", engine="openpyxl")

            # Find coefficients describing real output
        #-------------------------------------------------------------------------------------------------------------------------------------
        # %HR vs Power: handcycle
        data = data_hc
        data = data.iloc[:int(len(data)/2), :]
        x = data["Heart Rate"] / (220 - members[f"{ID}"]["Age"] - 20) * 100
        y = data["Power"]
        slopes[i, 0], intercepts[i, 0] = draw_plots.plot_graphs(x, y, "Handcycle: %HR vs Power",
                                                                "% PeakHR", "Power[W]", participants[i], 
                                                                i, fig1, axs1, color_coded = "No")

        # %HR vs Power: bicycle
        data = data_bc
        x = data["Heart Rate"] / data["PeakHR"] * 100
        y = data["Power"]
        slopes[i, 1], intercepts[i, 1] = draw_plots.plot_graphs(x, y, "Bicycle: %HR vs Power", 
                                                                 "% PeakHR", "Power[W]", participants[i],
                                                                i, fig2, axs2, color_coded = "No")

            # Predicting P_hc and P_bc
        #--------------------------------------------------------------------------------------------------------------------------------------------
        # Find coefficients which predict Pbc and Phc based on % Peak HR
        perc_hr = np.linspace(50, 80, 100)
        P_hc_pred = slopes[i, 0] * perc_hr + intercepts[i, 0]
        P_bc_pred = slopes[i, 1] * perc_hr + intercepts[i, 1]

        # Plot results and find regression line
        slopes_model[i], intercepts_model[i] = draw_plots.plot_graphs(P_hc_pred, P_bc_pred, "Handcycle: predicted Power Outputs", 
                               "Predicted hc [W]", "Predicted bc [W]", participants[i], # Labels for x and y axes
                               i, fig3, axs3, color_coded = "No") # Plot properties
        
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------
        # Plot coefficients and color code them 
    #--------------------------------------------------------------------------------------------------------------------------------------------------
    fig5, axs5 = plt.subplots(2, 3)
    axs5 = axs5.flatten()

    # Split into age groups
    ages = [member["Age"] for member in members.values()]
    heights = [member["Height"] for member in members.values()]
    weights = [member["Weight"] for member in members.values()]
    ages = [member["Age"] for member in members.values()]

    bins_age = pd.cut(ages, 5)
    bins_height = pd.cut(heights, 5)
    bins_weight = pd.cut(weights, 5)


    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"
        
        # Color code by gender
    #--------------------------------------------------------------------------------------------------------------------------------
        if members[f"{ID}"]["Gender"] == 0:
            c = 'b'
        else:
            c = 'r'
        
        axs5[0].scatter(intercepts_model[i], slopes_model[i], color = c)
        axs5[0].legend(["Females", "Males"], loc = 'upper right')
        axs5[0].set_title("Gender")

            # Color code by age
        #----------------------------------------------------------------------------------------------------------------------------------
        bins = bins_age
        if members[f"{ID}"]["Age"] in bins.categories[0]:
            c = "yellow"
        elif members[f"{ID}"]["Age"] in bins.categories[1]:
            c = "orange"
        elif members[f"{ID}"]["Age"] in bins.categories[2]:
            c = "red"
        elif members[f"{ID}"]["Age"] in bins.categories[3]:
            c = "green"
        elif members[f"{ID}"]["Age"] in bins.categories[4]:
            c = "blue"
        
        axs5[1].scatter(intercepts_model[i], slopes_model[i], color=c)

        # Add legend and figure details
        legend_labels = [f"{math.ceil(bins.categories[0].left)}-{math.floor(bins.categories[0].right)}", 
                         f"{math.ceil(bins.categories[1].left)}-{math.floor(bins.categories[1].right)}", 
                         f"{math.ceil(bins.categories[2].left)}-{math.floor(bins.categories[2].right)}",
                         f"{math.ceil(bins.categories[3].left)}-{math.floor(bins.categories[3].right)}",
                         f"{math.ceil(bins.categories[4].left)}-{math.floor(bins.categories[4].right)}"]
        legend_colors = ["yellow", "orange", "red", "green", "blue"]
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
        axs5[1].legend(legend_handles, legend_labels, loc='upper right')
        axs5[1].set_title("Age")


        # Color code by height
    #-----------------------------------------------------------------------------------------------------------------------------------------
        bins = bins_height
        feature = "Height"

        color_map = {
            bins.categories[0]: "yellow",
            bins.categories[1]: "orange",
            bins.categories[2]: "red",
            bins.categories[3]: "green",
            bins.categories[4]: "blue"
        }

        category = next(bin for bin in bins.categories if members[f"{ID}"][f"{feature}"] in bin)
        c = color_map[category]
        
        axs5[2].scatter(intercepts_model[i], slopes_model[i], color=c)

        # Add legend and figure details
        legend_labels = [f"{math.ceil(bins.categories[i].left)}-{math.floor(bins.categories[i].right)}" for i in range(5)]
        legend_colors = ["yellow", "orange", "red", "green", "blue"]
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
        axs5[2].legend(legend_handles, legend_labels, loc='upper right')
        axs5[2].set_title(f"{feature}")

        # Color code by height
    #-----------------------------------------------------------------------------------------------------------------------------------------
        bins = bins_weight
        feature = "Weight"
        idx = 3
        
        color_map = {
                bins.categories[0]: "yellow",
            bins.categories[1]: "orange",
            bins.categories[2]: "red",
            bins.categories[3]: "green",
            bins.categories[4]: "blue"
        }
    
        category = next(bin for bin in bins.categories if members[f"{ID}"][f"{feature}"] in bin)
        c = color_map[category]
        
        axs5[idx].scatter(intercepts_model[i], slopes_model[i], color=c)

        # Add legend and figure details
        legend_labels = [f"{math.ceil(bins.categories[i].left)}-{math.floor(bins.categories[i].right)}" for i in range(5)]
        legend_colors = ["yellow", "orange", "red", "green", "blue"]
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
        axs5[idx].legend(legend_handles, legend_labels, loc='upper right')
        axs5[idx].set_title(f"{feature}")

    plt.tight_layout()
    plt.show()
            
final = "Yake"