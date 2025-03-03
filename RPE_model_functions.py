import Functions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#------------------------------------------------------------------------------------------------------------------------------------
    # First subplot: 180-second long windows
#------------------------------------------------------------------------------------------------------------------------------------
n_windows = 1
length_windows = int(180/n_windows)
participants = [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]

path = "C:\\Users\\maddy\\Desktop\\Roba seria\\II ciclo\\Tesi\\Acquisitions\\Input to models\RPE Models"
# path = "C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\RPE model\\Input files"
data_or = pd.read_excel(f"{path}\\{length_windows}_sec_feature_extraction.xlsx")
RPE_model = Functions.RPEModel(n_windows, participants)

data, RPE_or = RPE_model.preprocessing(data_or)
RPE_measured_180, RPE_predicted_180 = RPE_model.leave_p_out(data, RPE_or)

#------------------------------------------------------------------------------------------------------------------------------------
    # Second subplot: 60-second long windows
#------------------------------------------------------------------------------------------------------------------------------------
n_windows = 3
length_windows = int(180/n_windows)

data_or = pd.read_excel(f"{path}\\{length_windows}_sec_feature_extraction.xlsx")
RPE_model = Functions.RPEModel(n_windows, participants)

data, RPE_or = RPE_model.preprocessing(data_or)
RPE_measured_60, RPE_predicted_60 = RPE_model.leave_p_out(data, RPE_or)



#-------------------------------------------------------------------------------------------------------------------------------------
    # Visualize results
#-------------------------------------------------------------------------------------------------------------------------------------
print("For 60 second windows:")
scatter_60 = RPE_model.visualize_results_scatter(RPE_measured_60, RPE_predicted_60, 60)
print("\n")
print("For 180 second windows:")
scatter_180 = RPE_model.visualize_results_scatter(RPE_measured_180, RPE_predicted_180, 180)

n_rows = 4
n_columns = 4
fig1, axs1 = plt.subplots(n_rows, n_columns)
fig2, axs2 = plt.subplots(n_rows, n_columns)

k = 0
j = 0
for i in range(len(participants)):
    if i % n_columns == 0 and i > 0:
        j = 0
        k = k + 1
    if i % n_columns != 0:
        j = j + 1

    plot_60 = RPE_model.visualize_results_plot(RPE_measured_60[i, :], RPE_predicted_60[i, :], 3, fig1, axs1, j, k)
    axs1[j, k].set_title(f"Participant {participants[i]}")
    plot_180 = RPE_model.visualize_results_plot(RPE_measured_180[i, :], RPE_predicted_180[i, :], 1, fig2, axs2, j, k)
    axs2[j, k].set_title(f"Participant {participants[i]}")
    

# Visualize data from one single participant
'''participant = 10
m = participants.index(participant)

fig3, axs3 = plt.subplots(2)
x_axis = np.linspace(0, 17, 18)
axs3[0].scatter(x_axis, RPE_measured_60[m, :], color = (1, 0, 0), marker = 'x', s = 20, label = "Reported values")
axs3[0].scatter(x_axis, RPE_predicted_60[m, :], color = (0, 0, 1), marker = 'o', s = 20, label = "Predicted values")
axs3[0].set_title("60 second window")
x_axis = np.linspace(0, 5, 6)
axs3[1].scatter(x_axis, RPE_measured_180[m, :], color = (1, 0, 0), marker = 'x', s = 20, label = "Reported values")
axs3[1].scatter(x_axis, RPE_predicted_180[m, :], color = (0, 0, 1), marker = 'o', s = 20, label = "Predicted values")
axs3[1].set_title("180 second window")

handles, labels = axs3[1].get_legend_handles_labels()
fig3.legend(handles, labels, loc = 'upper right', fontsize = 15)
fig3.suptitle(f"Participant {participant}")'''


fig1.suptitle(f"60 second long windows")
fig2.suptitle(f"180 second long windows")
plt.show()

final = 'boh'