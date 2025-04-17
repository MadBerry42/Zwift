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
from matplotlib.lines import Line2D

#------------------------------------------------------------------------------------------------------------------------------------
    # First subplot: 180-second long windows
#------------------------------------------------------------------------------------------------------------------------------------
n_windows = 1
length_windows = int(180/n_windows)
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20]
IMU = "False"
mode = "raw"
want_pca = "No"

path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data\Input to models"
if IMU == "True":
    data_or = pd.read_excel(f"{path}\\Windowed files IMU\\{length_windows}_sec_feature_extraction_IMU.xlsx")
else:
    data_or = pd.read_excel(f"{path}\\Windowed files - no IMU\\{length_windows}_sec_feature_extraction.xlsx")  

RPE_model = Functions.RPEModel(n_windows, participants)
data = data_or.drop(columns=["ID", "RPE"])
RPE_or = data_or[["ID", "RPE"]]

  # PCA on the whole datasest
#--------------------------------------------------------------------------------------------------------------------------------
if want_pca == "Yes":
    scaler = MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(data.values), columns = data.columns)
    pca = PCA() 
    dataset = pca.fit_transform(dataset.values)

    variance_plot = Functions.VisualizeResults()
    variance_plot.extra_functions_for_PCA(pca, data.columns, length_windows)
    percentage = variance_plot.plot_feature_importance_long(pca, data.columns, length_windows, n_pcs = 14)
    variance_plot.get_num_pca_to_run(dataset, show_plot='True')

    if IMU == "False":
        fig_heat = variance_plot.get_heat_map(pca, data.columns, percentage[:, 0:5], 10, 10, 'vertical')
        fig_heat.show()

    table_180 = variance_plot.get_sorted_table(pca, data.columns, percentage, n_pcs = 5, top_n = 10)


        # Plot the loadings plot for the first two components
    #---------------------------------------------------------------------------------------------------------------------------------
    # Save the loadings in a dataframe
    n_features = 20 # Number 
    n_pcs = 2
    variance_plot = Functions.VisualizeResults()
    percentage = variance_plot.plot_feature_importance_long(pca, data.columns, length_windows, n_pcs = 14)
    percentages, features, indices = Functions.VisualizeResults.sort_variables(feature_labels = data.columns, percentage = percentage, n_pcs = n_pcs, top_n = n_features)
    features = features[0] + features[1]

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

    ax.set_title(f"Loading plot for component 1 and 2, {length_windows}-second windows")
    ax.set_xlabel(f"First Component ({pca.explained_variance_ratio_[0] * 100:.2f} %)")
    ax.set_ylabel(f"Second Component ({pca.explained_variance_ratio_[1] * 100:.2f} %)")

    # Plot the score plot for the first two components
    plt.show()

    ######
    plt.figure()
    distances = np.sqrt(loadings.iloc[:, 0].values**2 + loadings.iloc[:, 1].values**2)
    normalized_sizes = 100 * (distances / distances.max())
    # These two lines compute the distance from the origin, so that the size of the dot is proportionate to how much each feature weighs

    for i in range(n_features):
        feature_name = features[i]
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
    texts = [plt.text(x[i], y[i], features[i], ha='center', va='center') for i in range(len(x))]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    legend_labels = ["Cadence", "Power", "Heart rate", "Personal info"]
    legend_colors = [(128/255, 0, 32/155),  (1, 1, 0), (1, 165/255, 0), (0, 0, 1)]
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]

    plt.title(f"Loading plot for component 1 and 2, {length_windows}-second windows")
    plt.xlabel(f"First Component ({pca.explained_variance_ratio_[0] * 100:.2f} %)")
    plt.ylabel(f"Second Component ({pca.explained_variance_ratio_[1] * 100:.2f} %)")
    plt.grid()
    plt.legend(legend_handles, legend_labels, loc='lower right')

    # Plot the score plot for the first two components
    plt.show()

    # Modeling on train and test set
#--------------------------------------------------------------------------------------------------------------------------------
data = data_or.drop(columns = ["RPE"])
RPE_measured_180, RPE_predicted_180, test_180_svr, train_180_svr, pca_180 = RPE_model.leave_p_out(data, RPE_or)

#------------------------------------------------------------------------------------------------------------------------------------
    # Second subplot: 60-second long windows
#------------------------------------------------------------------------------------------------------------------------------------
n_windows = 3
length_windows = int(180/n_windows)
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20]
IMU = "False"
mode = "raw"
want_pca = "No"

path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data\Input to models"
if IMU == "True":
    data_or = pd.read_excel(f"{path}\\Windowed files IMU\\{length_windows}_sec_feature_extraction_IMU.xlsx")
else:
    data_or = pd.read_excel(f"{path}\\Windowed files - no IMU\\{length_windows}_sec_feature_extraction.xlsx")  

data = data_or.drop(columns = ["RPE"])


  # PCA on the whole datasest
#--------------------------------------------------------------------------------------------------------------------------------
if want_pca == "Yes":
    scaler = MinMaxScaler()
    dataset = pd.DataFrame(scaler.fit_transform(data.values), columns = data.columns)
    pca = PCA() 
    dataset = pca.fit_transform(dataset.values)

    variance_plot = Functions.VisualizeResults()
    variance_plot.extra_functions_for_PCA(pca, data.columns, length_windows)
    percentage = variance_plot.plot_feature_importance_long(pca, data.columns, length_windows, n_pcs = 14)
    variance_plot.get_num_pca_to_run(dataset, show_plot='True')

    if IMU == "False":
        fig_heat = variance_plot.get_heat_map(pca, data.columns, percentage[:, 0:5], 10, 10, 'vertical')
        fig_heat.show()

    table_180 = variance_plot.get_sorted_table(pca, data.columns, percentage, n_pcs = 5, top_n = 10)


        # Plot the loadings plot for the first two components
    #---------------------------------------------------------------------------------------------------------------------------------
    # Save the loadings in a dataframe
    n_features = 20 # Number 
    n_pcs = 2
    variance_plot = Functions.VisualizeResults()
    percentage = variance_plot.plot_feature_importance_long(pca, data.columns, length_windows, n_pcs = 14)
    percentages, features, indices = Functions.VisualizeResults.sort_variables(feature_labels = data.columns, percentage = percentage, n_pcs = n_pcs, top_n = n_features)
    features = features[0] + features[1]

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

    ax.set_title(f"Loading plot for component 1 and 2, {length_windows}-second windows")
    ax.set_xlabel(f"First Component ({pca.explained_variance_ratio_[0] * 100:.2f} %)")
    ax.set_ylabel(f"Second Component ({pca.explained_variance_ratio_[1] * 100:.2f} %)")

    # Plot the score plot for the first two components
    plt.show()

    ######
    plt.figure()
    distances = np.sqrt(loadings.iloc[:, 0].values**2 + loadings.iloc[:, 1].values**2)
    normalized_sizes = 100 * (distances / distances.max())
    # These two lines compute the distance from the origin, so that the size of the dot is proportionate to how much each feature weighs

    for i in range(n_features):
        feature_name = features[i]
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
    texts = [plt.text(x[i], y[i], features[i], ha='center', va='center') for i in range(len(x))]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    legend_labels = ["Cadence", "Power", "Heart rate", "Personal info"]
    legend_colors = [(128/255, 0, 32/155),  (1, 1, 0), (1, 165/255, 0), (0, 0, 1)]
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]

    plt.title(f"Loading plot for component 1 and 2, {length_windows}-second windows")
    plt.xlabel(f"First Component ({pca.explained_variance_ratio_[0] * 100:.2f} %)")
    plt.ylabel(f"Second Component ({pca.explained_variance_ratio_[1] * 100:.2f} %)")
    plt.grid()
    plt.legend(legend_handles, legend_labels, loc='lower right')

    # Plot the score plot for the first two components
    plt.show()

    # Modeling on train and test set
#--------------------------------------------------------------------------------------------------------------------------------
data = data_or.drop(columns=["RPE"])
RPE_or = data_or[["ID", "RPE"]]
RPE_model = Functions.RPEModel(n_windows, participants)
RPE_measured_60, RPE_predicted_60, test_60_svr, train_60_svr, pca_60 = RPE_model.leave_p_out(data, RPE_or)

#-----------------------------------------------------------------------------------------------------------------------------------
    # 30-second windows
#-----------------------------------------------------------------------------------------------------------------------------------    
n_windows = 6
length_windows = int(180/n_windows)

path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data\Input to models"
if IMU == "True":
    data_or = pd.read_excel(f"{path}\\Windowed files IMU\\{length_windows}_sec_feature_extraction_IMU.xlsx")
else:
    data_or = pd.read_excel(f"{path}\\Windowed files - no IMU\\{length_windows}_sec_feature_extraction.xlsx")  

data = data_or.drop(columns=["RPE"])
RPE_or = data_or[["ID", "RPE"]]

RPE_model = Functions.RPEModel(n_windows, participants)
RPE_measured_30, RPE_predicted_30, test_30_svr, train_30_svr, pca_30 = RPE_model.leave_p_out(data, RPE_or)


#-------------------------------------------------------------------------------------------------------------------------------------
    # Visualize results
#-------------------------------------------------------------------------------------------------------------------------------------
print("For 30 second windows, support vector regression,")
r_squared = np.mean(test_30_svr[:, -3])
mse = np.mean(test_30_svr[:, -2])
rmse = np.mean(test_30_svr[:, -1])
print(f"R^2 is {r_squared}")
print(f"MSE is {mse}")
print(f"RMSE is {rmse}")

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

n_rows = 5
n_columns = 4
fig1, axs1 = plt.subplots(n_rows, n_columns)
fig2, axs2 = plt.subplots(n_rows, n_columns)
fig3, axs3 = plt.subplots(n_rows, n_columns)
fig4, axs4 = plt.subplots(n_rows, n_columns)
fig5, axs5 = plt.subplots(n_rows, n_columns)
fig6, axs6 = plt.subplots(n_rows, n_columns)

axs1 = axs1.flatten()
axs2 = axs2.flatten()
axs3 = axs3.flatten()
axs4 = axs4.flatten()
axs5 = axs5.flatten()
axs6 = axs6.flatten()


for i in range(len(participants)):
    plot_60 = RPE_model.visualize_results_plot(RPE_measured_60[i, :], RPE_predicted_60[i, :], 3, fig1, axs1, i)
    axs1[i].set_title(f"Participant {participants[i]}, r^2: {r2_score(RPE_measured_60[i, :], RPE_predicted_60[i, :]):.3f}")
    axs1[i].set_xlabel("Block")
    axs1[i].set_ylabel("RPE value")
    plot_180 = RPE_model.visualize_results_plot(RPE_measured_180[i, :], RPE_predicted_180[i, :], 1, fig2, axs2, i)
    axs2[i].set_title(f"Participant {participants[i]}, r^2: {r2_score(RPE_measured_180[i, :], RPE_predicted_180[i, :]):.3f}")
    axs2[i].set_xlabel("Block")
    axs2[i].set_ylabel("RPE value")
    plot_30 = RPE_model.visualize_results_plot(RPE_measured_30[i, :], RPE_predicted_30[i, :], 1, fig3, axs3, i)
    axs3[i].set_title(f"Participant {participants[i]}, r^2: {r2_score(RPE_measured_30[i, :], RPE_predicted_30[i, :]):.3f}")
    axs3[i].set_xlabel("Block")
    axs3[i].set_ylabel("RPE value")
    # SVR
    RPE_model.visualize_results_plot(RPE_measured_60[i, :], test_60_svr[i, 0:18], 3, fig4, axs4, i)
    axs4[i].set_title(f"Participant {participants[i]}, r^2: {test_60_svr[i, -3]:.3f}")
    axs4[i].set_xlabel("Block")
    axs4[i].set_ylabel("RPE value")
    RPE_model.visualize_results_plot(RPE_measured_180[i, :], test_180_svr[i, 0:6], 1, fig5, axs5, i)
    axs5[i].set_title(f"Participant {participants[i]}, r^2: {test_180_svr[i, -3]:.3f}")
    axs5[i].set_xlabel("Block")
    axs5[i].set_ylabel("RPE value")
    RPE_model.visualize_results_plot(RPE_measured_30[i, :], test_30_svr[i, 0:36], 1, fig6, axs6, i)
    axs6[i].set_title(f"Participant {participants[i]}, r^2: {test_30_svr[i, -3]:.3f}")
    axs6[i].set_xlabel("Block")
    axs6[i].set_ylabel("RPE value")
    

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

if IMU == "True":
    fig1.suptitle(f"60 second long windows, linear regression, IMU")
    fig2.suptitle(f"180 second long windows, linear regression, IMU")
    fig3.suptitle(f"60 second long windows, support vector regression, IMU")
    fig4.suptitle(f"180 second long windows, support vector regression, IMU")
else:
    fig1.suptitle(f"60 second long windows, linear regression, no IMU")
    fig2.suptitle(f"180 second long windows, linear regression, no IMU")
    fig3.suptitle(f"30 second long windows, linear regression, no IMU")
    fig4.suptitle(f"60 second long windows, support vector regression, no IMU")
    fig5.suptitle(f"180 second long windows, support vector regression, no IMU")
    fig6.suptitle(f"30 second long windows, support vector regression, no IMU")

plt.show()

final = 'boh'