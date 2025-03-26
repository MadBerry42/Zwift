import matplotlib
matplotlib.use("TkAgg")
import Functions
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text



#------------------------------------------------------------------------------------------------------------------------------------
    # First subplot: 180-second long windows
#------------------------------------------------------------------------------------------------------------------------------------
n_windows = 1
length_windows = int(180/n_windows)
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]

IMU = "True"
# path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\RPE model"
path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\\RPE Models"

if IMU == "True":
    data_or = pd.read_excel(f"{path}\\Windowed files IMU\\{length_windows}_sec_feature_extraction_IMU.xlsx")
else:
    data_or = pd.read_excel(f"{path}\\Windowed files\\{length_windows}_sec_feature_extraction.xlsx")   

RPE_model = Functions.RPEModel(n_windows, participants)
data, RPE_or = RPE_model.preprocessing(data_or)

plt.show()

  # PCA on the whole datasest
#--------------------------------------------------------------------------------------------------------------------------------
scaler = MinMaxScaler()
dataset = pd.DataFrame(scaler.fit_transform(data.values), columns = data.columns)
pca = PCA() 
dataset = pca.fit_transform(dataset.values)

variance_plot = Functions.VisualizeResults()
variance_plot.extra_functions_for_PCA(pca, data.columns, length_windows)
percentage = variance_plot.plot_feature_importance_long(pca, data.columns, 180, n_pcs = 14)
variance_plot.get_num_pca_to_run(dataset, show_plot='True')

if IMU == "False":
    fig_heat = variance_plot.get_heat_map(pca, data.columns, percentage[:, 0:5], 10, 10, 'vertical')
    fig_heat.show()

table_180 = variance_plot.get_sorted_table(pca, data.columns, percentage, n_pcs = 5, top_n = 10)


    # Modeling on train and test set
#--------------------------------------------------------------------------------------------------------------------------------
RPE_measured_180, RPE_predicted_180, test_180_svr, train_180_svr, pca_180 = RPE_model.leave_p_out(data, RPE_or)

    # Plot the loadings plot for the first two components
#---------------------------------------------------------------------------------------------------------------------------------
# Save the loadings in a dataframe
n_features = 20 # Number 
n_pcs = 2
percentages, features, indices = Functions.VisualizeResults.sort_variables(feature_labels = data.columns, percentage = percentage, n_pcs = n_pcs, top_n = n_features)

components = pd.DataFrame(data = [[pca.components_[i, f] for f in indices] for i in range(n_pcs)], 
                          columns = [f"PC{i + 1}" for i in range(len(indices))], 
                         )

loadings = pd.DataFrame([components.iloc[i].values * np.sqrt(pca.explained_variance_[i]) for i in range(n_pcs)],
                        columns = components.columns)
loadings = loadings.T

# Original, backup code
fig, ax = plt.subplots()
distances = np.sqrt(loadings.iloc[:, 0].values**2 + loadings.iloc[:, 1].values**2)
normalized_sizes = 200 * (distances / distances.max())
# These two lines compute the distance from the origin, so that the size of the dot is proportionate to how much each feature weighs

for i in range(n_features):
    feature_name = features[i]
    # Problema: Features ha due componenti tra cui muoversi, perciò i>2 darà index out of bound
    if 'cadence' in feature_name:
        c = (128/255, 0, 32/155)
    elif 'P_hc' in feature_name:
        c = (1, 1, 0)
    elif 'hr' in feature_name:
        c = (1, 165/255, 0)
    elif 'gyro' in feature_name:
        c = (0, 1, 0)
    elif 'imu' in feature_name:
        c = (0, 1, 1)
    elif 'accel' in feature_name:
        c = (0, 0, 0)
    else:
        c = (0, 0, 1)

    plt.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], color=c, s=normalized_sizes[i] + 10)
    ax.annotate(feature_name, (loadings.iloc[i, 0], loadings.iloc[i, 1]), 
                fontsize = 10,
                )

ax.set_title("Loading plot for component 1 and 2, 180-second windows")
ax.set_xlabel(f"First Component ({pca_180.explained_variance_ratio_[0] * 100:.2f} %)")
ax.set_ylabel(f"Second Component ({pca_180.explained_variance_ratio_[1] * 100:.2f} %)")

# Plot the score plot for the first two components
# Scores are atored in the 
plt.show()

######
plt.figure()
distances = np.sqrt(loadings.iloc[:, 0].values**2 + loadings.iloc[:, 1].values**2)
normalized_sizes = 100 * (distances / distances.max())
# These two lines compute the distance from the origin, so that the size of the dot is proportionate to how much each feature weighs

for i in range(pca_180.components_.shape[1]):
    feature_name = data.columns[i]
    if 'cadence' in feature_name:
        c = (128/255, 0, 32/155)
    elif 'P_hc' in feature_name:
        c = (1, 1, 0)
    elif 'hr' in feature_name:
        c = (1, 165/255, 0)
    elif 'gyro' in feature_name:
        c = (0, 1, 0)
    elif 'imu' in feature_name:
        c = (0, 1, 1)
    elif 'accel' in feature_name:
        c = (0, 0, 0)
    else:
        c = (0, 0, 1)

    plt.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], color=c, s=normalized_sizes[i] + 10)


x = loadings.iloc[:, 0]
y = loadings.iloc[:, 1]
texts = [plt.text(x[i], y[i], data.columns[i], ha='center', va='center') for i in range(len(x))]
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

plt.title("Loading plot for component 1 and 2, 180-second windows")
plt.xlabel(f"First Component ({pca_180.explained_variance_ratio_[0] * 100:.2f} %)")
plt.ylabel(f"Second Component ({pca_180.explained_variance_ratio_[1] * 100:.2f} %)")
plt.grid()

# Plot the score plot for the first two components
plt.show()




#------------------------------------------------------------------------------------------------------------------------------------
    # Second subplot: 60-second long windows
#------------------------------------------------------------------------------------------------------------------------------------
n_windows = 3
length_windows = int(180/n_windows)

if IMU == "True":
    data_or = pd.read_excel(f"{path}\\Windowed files IMU\\{length_windows}_sec_feature_extraction_IMU.xlsx")
else:
    data_or = pd.read_excel(f"{path}\\Windowed files\\{length_windows}_sec_feature_extraction.xlsx")   

RPE_model = Functions.RPEModel(n_windows, participants)
data, RPE_or = RPE_model.preprocessing(data_or)

plt.show()

data, RPE_or = RPE_model.preprocessing(data_or)

  # PCA on the whole datasest
#--------------------------------------------------------------------------------------------------------------------------------
scaler = MinMaxScaler()
dataset = pd.DataFrame(scaler.fit_transform(data.values), columns = data.columns)
pca = PCA() 
dataset = pca.fit_transform(dataset.values)

variance_plot = Functions.VisualizeResults()
variance_plot.extra_functions_for_PCA(pca, data.columns, length_windows)
percentage = variance_plot.plot_feature_importance_long(pca, data.columns, 60, n_pcs = 24)
variance_plot.get_num_pca_to_run(dataset, show_plot='True')

if IMU == "False":
    fig_heat = variance_plot.get_heat_map(pca, data.columns, percentage[:, 0:5], 10, 10, 'vertical')
    fig_heat.show()

table_60 = variance_plot.get_sorted_table(pca, data.columns, percentage, n_pcs = 5, top_n = 10)
plt.show()

    # Modeling on train and test set
#--------------------------------------------------------------------------------------------------------------------------------
RPE_measured_60, RPE_predicted_60, test_60_svr, train_60_svr, pca_60 = RPE_model.leave_p_out(data, RPE_or)

plt.show()

# Save the loadings in a dataframe
components_60 = pd.DataFrame(pca_60.components_.T, columns = [f"PC{i + 1}" for i in range(pca_60.components_.shape[0])], index = data.columns)
loadings_60 = components_60 * np.sqrt(pca_60.explained_variance_)

# Original, backup code
fig, ax = plt.subplots()
distances = np.sqrt(loadings_60.iloc[:, 0].values**2 + loadings_60.iloc[:, 1].values**2)
normalized_sizes = 200 * (distances / distances.max())
# These two lines compute the distance from the origin, so that the size of the dot is proportionate to how much each feature weighs

for i in range(pca_60.components_.shape[1]):
    feature_name = data.columns[i]
    if 'cadence' in feature_name:
        c = (128/255, 0, 32/155)
    elif 'P_hc' in feature_name:
        c = (1, 1, 0)
    elif 'hr' in feature_name:
        c = (1, 165/255, 0)
    elif 'gyro' in feature_name:
        c = (0, 1, 0)
    elif 'imu' in feature_name:
        c = (0, 1, 1)
    elif 'accel' in feature_name:
        c = (0, 0, 0)
    else:
        c = (0, 0, 1)

    plt.scatter(loadings_60.iloc[i, 0], loadings_60.iloc[i, 1], color=c, s=normalized_sizes[i] + 10)
    ax.annotate(feature_name, (loadings_60.iloc[i, 0], loadings_60.iloc[i, 1]), 
                fontsize = 10,
                )

ax.set_title("Loading plot for component 1 and 2, 60-second windows")
ax.set_xlabel(f"First Component ({pca_60.explained_variance_ratio_[0] * 100:.2f} %)")
ax.set_ylabel(f"Second Component ({pca_60.explained_variance_ratio_[1] * 100:.2f} %)")

# Plot the score plot for the first two components
# Scores are atored in the 
plt.show()

######
plt.figure()
distances = np.sqrt(loadings_60.iloc[:, 0].values**2 + loadings_60.iloc[:, 1].values**2)
normalized_sizes = 100 * (distances / distances.max())
# These two lines compute the distance from the origin, so that the size of the dot is proportionate to how much each feature weighs

for i in range(pca_60.components_.shape[1]):
    feature_name = data.columns[i]
    if 'cadence' in feature_name:
        c = (128/255, 0, 32/155)
    elif 'P_hc' in feature_name:
        c = (1, 1, 0)
    elif 'hr' in feature_name:
        c = (1, 165/255, 0)
    elif 'gyro' in feature_name:
        c = (0, 1, 0)
    elif 'imu' in feature_name:
        c = (0, 1, 1)
    elif 'accel' in feature_name:
        c = (0, 0, 0)
    else:
        c = (0, 0, 1)

    plt.scatter(loadings_60.iloc[i, 0], loadings_60.iloc[i, 1], color=c, s=normalized_sizes[i] + 10)


x = loadings_60.iloc[:, 0]
y = loadings_60.iloc[:, 1]
texts = [plt.text(x[i], y[i], data.columns[i], ha='center', va='center') for i in range(len(x))]
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

plt.title("Loading plot for component 1 and 2, 180-second windows")
plt.xlabel(f"First Component ({pca_60.explained_variance_ratio_[0] * 100:.2f} %)")
plt.ylabel(f"Second Component ({pca_60.explained_variance_ratio_[1] * 100:.2f} %)")
plt.grid()

# Plot the score plot for the first two components
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------
    # Visualize results
#-------------------------------------------------------------------------------------------------------------------------------------
'''print("For 60 second windows, linear regression,")
scatter_60 = RPE_model.visualize_results_scatter(RPE_measured_60, RPE_predicted_60, 60)
print("For 180 second windows, linear regression,")
scatter_180 = RPE_model.visualize_results_scatter(RPE_measured_180, RPE_predicted_180, 180)'''

print("For 60 second windows, support vector regression,")
r_squared = np.mean(test_60_svr[:, -3])
mse = np.mean(test_60_svr[:, -2])
rmse = np.mean(test_60_svr[:, -1])
print(f"R^2 is {r_squared}")
print(f"MSE is {mse}")
print(f"RMSE is {rmse}")

print("For 180 second windows, support vector regression,")
r_squared = np.mean(test_180_svr[:, -3])
mse = np.mean(test_180_svr[:, -2])
rmse = np.mean(test_180_svr[:, -1])
print(f"R^2 is {r_squared}")
print(f"MSE is {mse}")
print(f"RMSE is {rmse}")


n_rows = 4
n_columns = 4
fig1, axs1 = plt.subplots(n_rows, n_columns)
fig2, axs2 = plt.subplots(n_rows, n_columns)
fig3, axs3 = plt.subplots(n_rows, n_columns)
fig4, axs4 = plt.subplots(n_rows, n_columns)

k = 0
j = 0
for i in range(len(participants)):
    if i % n_columns == 0 and i > 0:
        j = 0
        k = k + 1
    if i % n_columns != 0:
        j = j + 1

    plot_60 = RPE_model.visualize_results_plot(RPE_measured_60[i, :], RPE_predicted_60[i, :], 3, fig1, axs1, j, k)
    axs1[j, k].set_title(f"Participant {participants[i]}, r^2: {r2_score(RPE_measured_60[i, :], RPE_predicted_60[i, :]):.3f}")
    plot_180 = RPE_model.visualize_results_plot(RPE_measured_180[i, :], RPE_predicted_180[i, :], 1, fig2, axs2, j, k)
    axs2[j, k].set_title(f"Participant {participants[i]}, r^2: {r2_score(RPE_measured_180[i, :], RPE_predicted_180[i, :]):.3f}")
    # SVR
    RPE_model.visualize_results_plot(RPE_measured_60[i, :], test_60_svr[i, 0:18], 3, fig3, axs3, j, k)
    axs3[j, k].set_title(f"Participant {participants[i]}, r^2: {test_60_svr[i, -3]:.3f}")
    RPE_model.visualize_results_plot(RPE_measured_180[i, :], test_180_svr[i, 0:6], 1, fig4, axs4, j, k)
    axs4[j, k].set_title(f"Participant {participants[i]}, r^2: {test_180_svr[i, -3]:.3f}")
    

# Visualize data from one single participant
'''participant = 10
m = participants.index(participant)

fig5, axs5 = plt.subplots(2)
x_axis = np.linspace(0, 17, 18)
axs5[0].scatter(x_axis, RPE_measured_60[m, :], color = (1, 0, 0), marker = 'x', s = 20, label = "Reported values")
axs5[0].scatter(x_axis, RPE_measured_60[m, :], color = (1, 0, 0), marker = 'x', s = 20, label = "Reported values")
axs5[0].scatter(x_axis, RPE_predicted_60[m, :], color = (0, 0, 1), marker = 'o', s = 20, label = "Predicted values")
axs5[0].set_title("60 second window")
x_axis = np.linspace(0, 5, 6)
axs5[1].scatter(x_axis, RPE_measured_180[m, :], color = (1, 0, 0), marker = 'x', s = 20, label = "Reported values")
axs5[1].scatter(x_axis, RPE_predicted_180[m, :], color = (0, 0, 1), marker = 'o', s = 20, label = "Predicted values")
axs5[1].set_title("180 second window")

handles, labels = axs3[1].get_legend_handles_labels()
fig3.legend(handles, labels, loc = 'upper right', fontsize = 15)
fig3.suptitle(f"Participant {participant}")'''

if IMU:
    fig1.suptitle(f"60 second long windows, linear regression, IMU")
    fig2.suptitle(f"180 second long windows, linear regression, IMU")
    fig3.suptitle(f"60 second long windows, support vector regression, IMU")
    fig4.suptitle(f"180 second long windows, support vector regression, IMU")
else:
    fig1.suptitle(f"60 second long windows, linear regression, no IMU")
    fig2.suptitle(f"180 second long windows, linear regression, no IMU")
    fig3.suptitle(f"60 second long windows, support vector regression, no IMU")
    fig4.suptitle(f"180 second long windows, support vector regression, no IMU")

plt.show()

final = 'boh'