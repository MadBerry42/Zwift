import pandas as pd
import Functions
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#-------------------------------------------------------------------------------------------------------------------------------------------
    # Import and pre-process data 
#-------------------------------------------------------------------------------------------------------------------------------------------
n_windows = 1
length_windows = int(180/n_windows)

participants = [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\RPE Models"
data_or = pd.read_excel(f"{path}\\{length_windows}_sec_feature_extraction.xlsx")

# MinMax scaling
data = data_or.drop(columns = ["ID", "RPE"])
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
scaler_rpe = MinMaxScaler()
RPE_or = pd.DataFrame(scaler_rpe.fit_transform(data_or.iloc[:, [5]]))

#-------------------------------------------------------------------------------------------------------------------------------------------
    # Linear regression
#-------------------------------------------------------------------------------------------------------------------------------------------
linear_regression = Functions.LinearRegression(lam=1e5)
coeff = np.zeros((len(participants), data.shape[1]))
prediction = np.zeros((len(participants), n_windows * 6))
for i in range(len(participants)):
    # Remove the test participant for cross validation
    test = np.zeros((n_windows * 6, data.shape[1]))
    for j in range (6 * n_windows):
        n = j * len(participants) + i
        test[j, :] = data.iloc[[n], :]
        if j == 0:
            dataset = data.drop([n])
            RPE = RPE_or.drop([n])
        else:
            dataset = dataset.drop([n])
            RPE = RPE.drop([n])
    
    A, b = linear_regression.create_matrices(dataset, RPE, "false")
    X = linear_regression.regression()

    # Save results in a matrix
    coeff[j, :] = X
    
    # Apply the model to the control participant
    predicted = test @ X
    # De-scale the data
    predicted = predicted.reshape(1, -1)
    RPE_predicted = scaler_rpe.inverse_transform(predicted)

    prediction[i, :] = RPE_predicted
    

print(prediction)
final = 'boh'