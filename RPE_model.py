import pandas as pd
import Functions
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

#-------------------------------------------------------------------------------------------------------------------------------------------
    # Import and pre-process data 
#-------------------------------------------------------------------------------------------------------------------------------------------
n_windows = 3
length_windows = int(180/n_windows)

participants = [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]

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
linear_regression = Functions.LinearRegression(lam=1e10)
coeff = np.zeros((len(participants) * n_windows, data.shape[1]))
RPE_predicted = np.zeros((len(participants), n_windows * 6))
RPE_measured = np.zeros((len(participants), 6 * n_windows))
for i in range(len(participants)):
    # Create the test participant for cross validation
    test = np.zeros((n_windows * 6, data.shape[1]))
    RPE_test = np.zeros((n_windows * 6, 1))
    for j in range (6):
        start = len(participants) * n_windows * j + i 
        test[n_windows * j : n_windows * (j + 1)] = data.iloc[start :  start + n_windows]
        RPE_test[n_windows * j : n_windows * (j + 1)] = RPE_or.iloc[start :  start + n_windows]

    # Remove the test participant
    for j in range(n_windows):  
        index = [(len(participants) * n_windows) * k for k in [0, 1, 2, 3, 4, 5]]
        index = [x + (j + i * n_windows) for x in index]
        dataset = data.drop(index)
        RPE = RPE_or.drop(index)

    
    A, b = linear_regression.create_matrices(dataset, RPE, "false")
    X = linear_regression.regression()

    # Save results in a matrix
    coeff[j, :] = X
    
    # Apply the model to the control participant
    predicted = test @ X
    # De-scale the data
    predicted = predicted.reshape(1, -1)
    prediction = scaler_rpe.inverse_transform(predicted)
    # prediction = np.round(prediction)

    RPE_predicted[i, :] = prediction
    RPE_test = RPE_test.reshape(1, -1)
    RPE_measured[i, :] = scaler_rpe.inverse_transform(RPE_test)


#-----------------------------------------------------------------------------------------------------------------------------------------------    
    # Visualize the results
#-----------------------------------------------------------------------------------------------------------------------------------------------
# 3D plot
ax = plt.figure().add_subplot(projection='3d')
# for i in range(len(participants)):
i = 0
x = participants
y = np.linspace(1, n_windows * 6, n_windows * 6)

colors = iter(cm.rainbow(np.linspace(0, 1, len(y))))
for j in range(len(y)):
    c = next(colors)
    ax.scatter(x[i], y, RPE_predicted[i, j], color = c)
    ax.scatter(x[i], y, RPE_measured[i, j], color = c)

ax.set_xlabel('Participant number')
ax.set_ylabel('Block')
ax.set_zlabel('RPE value')

ax.set_title(f'All of participants')

# plt.show()

# 2D plot
plt.figure()
for i in range(len(participants)):
    x = participants[i]
    y_measured = RPE_measured[i, :]
    y_predicted = RPE_predicted[i, :]


    '''colors = iter(cm.rainbow(np.linspace(0, 1, len(y_measured))))
    for y1, y2 in zip(y_measured, y_predicted):
        c = next(colors)
        plt.scatter(x + 0.5, y1, marker = 'x', color = c)
        plt.scatter(x, y2, marker = 'o', color = c)'''

    for y1, y2 in zip(y_measured, y_predicted):
        if y1 < 11:
            c = (173/255, 216/255, 230/255) # Light blue color
        elif y1 == 11:
            c = (144/255, 238/255, 144/255) # Light green color
        elif y1 == 12:
            c = (0, 1, 0) # Green color
        elif y1 == 13:
            c = (0, 1, 1) # Cyan color
        elif y1 == 14:
            c = (1, 0.647, 0) # Orange
        elif y1 == 15:
            c = (1, 0, 0)
        elif y1 > 15:
            c = (0, 0, 0)

        plt.scatter(x + 0.5, y1, marker = 'x', color = c)
        plt.scatter(x, y2, marker = 'o', color = c)

# Legend for color coding the RPE signal
legend_labels = ['<11 (Very light)', '11 (light)', '12 (light)', '13 (slightly hard)', '14 (hard)', '15 (really hard)', '>15 (exhaustion)']
legend_colors = [(173/255, 216/255, 230/255), (144/255, 238/255, 144/255), (0, 1, 0) , (0, 1, 1), (1, 0.647, 0), (1, 0, 0), (0, 0, 0)]
legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
plt.legend(legend_handles, legend_labels, loc='best', fontsize=11)

plt.xlabel("Participant ID")
plt.ylabel("RPE values")
plt.xticks(participants)
plt.show()

#------------------------------------------------------------------------------------------------------------------------------
    # Compute residuals
#------------------------------------------------------------------------------------------------------------------------------
residuals = np.zeros((RPE_predicted.shape[0], RPE_predicted.shape[1]))
for i in range(len(participants)):
    residuals[i, :] = abs(RPE_predicted[i] - RPE_measured[i])

# Parameters:
average = np.mean(residuals)
maximum = np.max(residuals)
minimum = np.min(residuals)

print("\n")
print(f"The average residual is: {average:.2f}, with a maximum value of {maximum:.2f} and a minimum value of {minimum:.2f}")

# Percentages
avg_percentage = abs(np.mean(RPE_predicted) - np.mean(RPE_measured))/np.mean(RPE_measured) * 100
print(f"The average error is: {avg_percentage:.2f} %")



final = 'boh'