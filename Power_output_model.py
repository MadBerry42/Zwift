import pandas as pd
import numpy as np
import Functions
# For creating the linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
# Visualising data
import matplotlib.pyplot as plt
# from tqdm import tqdm # Progress bar


#----------------------------------------------------------------------------------------------------------------------------------
    # Importing data and creating dataset
#----------------------------------------------------------------------------------------------------------------------------------
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
Skip = "Yes" # Set it to No if the Input file needs to be recreated
Model = "No"


if  Skip == "No":
    # path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\{ID}\\Zwift"
    path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\\Originals"

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

    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        elif ID >= 10:
            ID = f"0{ID}"

        # Import data and remove warmup and cooldown
        data_tmp = pd.read_csv(f"{path}\\{ID}_handcycle_protocol.csv", usecols = ["Power", "Heart Rate", "Cadence", "RPE"])
        data_bc = pd.read_csv(f"{path}\\{ID}_bicycle_protocol.csv", usecols = ["Power", "RPE"])
        
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

        personal_data = members[f"{ID}"]

        Gender = personal_data["Gender"]
        Age = personal_data["Age"]
        Height = personal_data["Height"]
        Weight = personal_data["Weight"]
        FTP = personal_data["FTP"]

        Gender = Gender * np.ones((len(data_tmp)))
        Age = Age * np.ones((len(data_tmp)))
        Height = Height * np.ones((len(data_tmp)))
        Weight = Weight * np.ones((len(data_tmp)))
        FTP = FTP * np.ones((len(data_tmp)))

        # Create DataFrame
        df = pd.DataFrame({"ID": ID, "Gender": Gender, "Age": Age, "Height": Height, "Weight": Weight, "FTP": FTP, "Heart Rate": data_tmp.iloc[:, 0], "Cadence": data_tmp.iloc[:, 1], "Power hc": Power_hc, "Power bc": Power_bc, "RPE hc": data_tmp.iloc[:, 3], "RPE bc": data_bc.iloc[:, 1]})

        if i == 0:
            data = df
        else:
            data = pd.concat([data, df], axis = 0)

    # path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Power output model\\Input files"
    path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\\Power output models"
    writer = pd.ExcelWriter(f'{path}\\Input_file_power_model.xlsx', engine = "openpyxl")
    wb = writer.book

    data.to_excel(writer, index = False)
    wb.save(f'{path}\\Input_file_power_model.xlsx')

    print(f"Document has been saved succesfully in {path}")


# path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Power output model\\Input files"
path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\\Power output models"
data = pd.read_excel(f"{path}\\Input_file_power_model.xlsx")
#-----------------------------------------------------------------------------------------------------------------------------------
    # Modeling: Leave-p-out method for validation
#-----------------------------------------------------------------------------------------------------------------------------------
if Model == "Yes":

        # Inizializing useful methods
    #-----------------------------------------------------------------------------------------------------------------------------------
    Power_model = Functions.PowerOutputModel()

    for i, ID in enumerate(participants):

        # Extracting test set before scaling and shuffling
    #-----------------------------------------------------------------------------------------------------------------------------------
        test_or = data[data["ID"] == ID]
        training_or = data[data["ID"] != ID]

        training_or = shuffle(training_or, random_state = None)
        
        # Processing the data: Removing ID and saving Power bc
    #-----------------------------------------------------------------------------------------------------------------------------------
        training = training_or.drop(columns = ["ID", "RPE bc"])
        test = test_or.drop(columns = ["ID", "RPE bc"])
        Power_bc = training_or.iloc[:, 8]

        # training, test, scaler = Power_model.preprocessing(training_or, test_or)
        training, test, scaler = Power_model.preprocessing(training, test)

        '''# Power_bc = scaler.inverse_transform(training)
        # Power_bc = Power_bc[:, 7]
        training = training.drop(columns = "Power bc")

        bc_test = scaler.inverse_transform(test)
        Power_bc_test = bc_test[:, 7]
        Power_hc_test = bc_test[:, 6]
        test = test.drop(columns = ["Power bc"])'''

        # Linear regression model
    #-----------------------------------------------------------------------------------------------------------------------------------
        linear_regression = LinearRegression().fit(training.iloc[:, training.columns != "Power bc"], training.iloc[:, training.columns == "Power bc"])
        predicted = linear_regression.predict(test.iloc[:, test.columns != "Power bc"])

        predicted = scaler.inverse_transform(predicted)

        plt.figure()
        plt.plot(predicted)
        plt.plot(test_or.iloc[:, test_or.columns == "Power bc"].reset_index().drop(columns = ["index"]))
        plt.plot(test_or.iloc[:, test_or.columns == "Power hc"].reset_index().drop(columns = ["index"]))

        plt.legend(["Linear regression", "Original bc signal", "Original hc signal"])
        plt.title(f"Linear regression, participant {ID}")
        plt.xlabel("Time [s]")
        plt.ylabel("Power [W]")

        plt.show()


        # SVR Model
    #-----------------------------------------------------------------------------------------------------------------------------------
        # training, test = Power_model.preprocessing(training_or.drop(columns = ["ID", "RPE bc"]), test_or.drop(columns = ["ID", "RPE bc"])) 

        # svr_model = Power_model.run_SVR(training_or.iloc[:, training_or.columns != ["Power bc", "ID", "RPE bc"]], training.iloc[:, training.columns == "Power bc"], test.iloc[:, test.columns != "Power bc"], test.iloc[:, test.columns == "Power bc"])
        '''print(f"Participant: {ID}")
        svr_model = Power_model.run_SVR(training.iloc[:, training.columns != "Power bc"], training.iloc[:, training.columns == "Power bc"], test.iloc[:, test.columns != "Power bc"], test.iloc[:, test.columns == "Power bc"])

        svr_results = scaler.inverse_transform(svr_model[0])

        
        plt.figure()
        plt.plot(svr_results)
        plt.plot(test_or.iloc[:, test_or.columns == "Power bc"])
        plt.plot(test_or.iloc[:,  test_or.columns == "Power hc"])

        plt.title(f"SVR, participant {ID}")
        plt.legend(["SVR Model", "Original bc signal", "Original hc signal"])
        plt.xlabel("Time [s]")
        plt.ylabel("Power [W]")

        plt.show()'''


#---------------------------------------------------------------------------------------------------------------------------------
    # Analizing the entire dataset
#---------------------------------------------------------------------------------------------------------------------------------
    
    # Initializing functions and preprocessing data
#---------------------------------------------------------------------------------------------------------------------------------
perform_pca = Functions.PerformPCA()
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
data = data.reset_index()
data = data.drop(columns = ["index"])

feature_labels = data.drop(columns = ["ID", "RPE bc", "Power bc"]).columns
dataset, pca = perform_pca.perform_pca(data.drop(columns = ["ID", "RPE bc", "Power bc"]))
perform_pca.plot_variance_table(pca, feature_labels)
perform_pca.get_sorted_table(pca, feature_labels, n_pcs = 5, top_n = len(feature_labels))

plt.show()
final = "Yake"


    


