import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import math
# Linear regression for predicting coefficients
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr

participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20]
path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data"
mode = "raw"
plot_hr = "No"
simple_model = "No"

class Plots():
    def __init__(self):
        pass
    
    def plot_graphs(self, x, y, title:str, x_label:str, y_label:str, ID, i, fig, axs, color_coded:str):
        if color_coded == "No": 
            # Plot the dots
            axs[i].scatter(x, y, color='b', s = 10)
            
        if color_coded == "Yes":
            x_yellow = x[:len(x)//3]
            y_yellow = y[:len(y)//3]

            x_orange = x[len(x)//3 : 2*len(x)//3]
            y_orange = y[len(y)//3 : 2*len(x)//3]

            x_red = x[2*len(x)//3:]
            y_red = y[2*len(y)//3:]

            axs[i].scatter(x_red, y_red, color='red', label = "RPE = 15", s = 10)
            axs[i].scatter(x_orange, y_orange, color='orange', label = "RPE = 14", s = 10)
            axs[i].scatter(x_yellow, y_yellow, color='yellow', label = "RPE = 12", s = 10)

            handles, labels = axs[i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right')

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

# Create P info columns

members = { "000":{"Age": 25, "Height": 160, "Weight": 58, "Gender": 1, "FTP": 49, "RPE": [[13, 11, 12], [12, 11, 13]], "Activity": [0, 0, 7*60]},
        "002":{"Age": 26, "Height": 177, "Weight": 75, "Gender": 0, "FTP": 78, "RPE": [[9, 12, 14], [9, 11, 13]], "Activity": [0, 4*45, 6*30]},
        "003":{"Age": 22, "Height": 180, "Weight": 70, "Gender": 0, "FTP": 51, "RPE": [[10, 12, 15], [8, 10, 12]],"Activity": [4*60, 7*15, 5*30]},
        "004":{"Age": 22, "Height": 186, "Weight": 80, "Gender": 0, "FTP": 47, "RPE": [[11, 12, 12], [10, 11, 12]],"Activity": [3*90, 1*5*60, 5*30]},
        "006":{"Age": 23, "Height": 174, "Weight": 87, "Gender": 1, "FTP": 51, "RPE": [[10, 11, 12], [10, 11, 10]],"Activity": [2*40, 0, 7*120]},
        "007":{"Age": 23, "Height": 183, "Weight": 70, "Gender": 0, "FTP": 55, "RPE": [[11, 11, 12], [8, 9, 11]],"Activity": [0, 0, 7*60]},
        "008":{"Age": 23, "Height": 190, "Weight": 82, "Gender": 0, "FTP": 82, "RPE": [[12, 12, 14], [10, 12, 14]],"Activity": [4*90, 0, 7*30]},
        "009":{"Age": 32, "Height": 185, "Weight": 96, "Gender": 0, "FTP": 62, "RPE": [[9, 11, 14], [10, 11, 14]],"Activity": [0, 0, 5*60]},
        "010":{"Age": 24, "Height": 160, "Weight": 56, "Gender": 1, "FTP": 48, "RPE": [[9, 11, 13], [9, 10, 11]],"Activity": [5*60, 2*60, 7*30]}, # Time spent for activity is grossly estimated
        "011":{"Age": 28, "Height": 176, "Weight": 67, "Gender": 0, "FTP": 60, "RPE": [[11, 10, 13], [10, 11, 13]],"Activity": [0, 0, 7*40]},
        "012":{"Age": 28, "Height": 184, "Weight": 70, "Gender": 0, "FTP": 87, "RPE": [[10, 11, 12], [10, 11, 12]],"Activity": [10, 4*20, 7*45]},
        "013":{"Age": 25, "Height": 178, "Weight": 66, "Gender": 0, "FTP": 62, "RPE": [[10, 11, 13], [11, 11, 13]],"Activity": [1*60, 1*60, 3*35]},
        "015":{"Age": 21, "Height": 176, "Weight": 73, "Gender": 0, "FTP": 60, "RPE": [[9, 10, 11], [10, 11, 13]],"Activity": [4*80, 6*2*60, 7*8*60]},
        "016":{"Age": 24, "Height": 173, "Weight": 59, "Gender": 1, "FTP": 37, "RPE": [[9, 10, 11], [7, 9, 10]],"Activity": [2*45, 2*30, 7*60]},
        "017":{"Age": 24, "Height": 187, "Weight": 75, "Gender": 0, "FTP": 58, "RPE": [[8, 8, 8], [8, 8, 11]], "Activity": [0, 0, 4*60, 13*60*7]},
        "019":{"Age": 24, "Height": 175, "Weight": 68, "Gender": 0, "FTP": 73, "RPE": [[8, 9, 9], [10, 9, 10]], "Activity": [4*120, 0, 7*60, 5*60*7]},
        "020":{"Age": 25, "Height": 174, "Weight": 73, "Gender": 0, "FTP": 88, "RPE": [[10, 11, 14], [9, 12, 14]], "Activity": [2*90, 0, 7*30, 8*60*7]},
        }

n_col = 5
n_rows = 4
hpad = 1

fig1, axs1 = plt.subplots(n_col, n_rows, constrained_layout = "True")
axs1 = axs1.flatten()
fig2, axs2 = plt.subplots(n_col, n_rows, constrained_layout = "True")
axs2 = axs2.flatten()
fig3, axs3 = plt.subplots(n_col, n_rows, constrained_layout = "True")
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

    data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_handcycle_{mode}.xlsx")
    data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_bicycle_{mode}.xlsx")

        # Find coefficients describing real output
    #-------------------------------------------------------------------------------------------------------------------------------------
    # %HR vs Power: handcycle
    data = data_hc
    data = data.iloc[:int(len(data)/2), :]
    x = data["Heart Rate"] / (220 - members[f"{ID}"]["Age"] - 20) * 100
    y = data["Power"]
    slopes[i, 0], intercepts[i, 0] = draw_plots.plot_graphs(x, y, "Handcycle: %HR vs Power",
                                                            "% PeakHR", "Power[W]", participants[i], 
                                                            i, fig1, axs1, color_coded = "Yes")
    # axs1[i].set_xlim([50, 80])
    axs1[i].set_ylim([20, 100])

    # %HR vs Power: bicycle
    data = data_bc
    data = data.iloc[:int(len(data)/2), :]
    x = data["Heart Rate"] / (220 - members[f"{ID}"]["Age"]) * 100
    y = data["Power"]
    slopes[i, 1], intercepts[i, 1] = draw_plots.plot_graphs(x, y, "Bicycle: %HR vs Power", 
                                                                "% PeakHR", "Power[W]", participants[i],
                                                            i, fig2, axs2, color_coded = "Yes")
    # axs2[i].set_xlim([50, 80])
    axs2[i].set_ylim([50, 250])

        # Predicting P_hc and P_bc
    #--------------------------------------------------------------------------------------------------------------------------------------------
    # Find coefficients which predict Pbc and Phc based on % Peak HR
    perc_hr = np.linspace(1, 100, 100)
    P_hc_pred = slopes[i, 0] * perc_hr + intercepts[i, 0]
    P_bc_pred = slopes[i, 1] * perc_hr + intercepts[i, 1]

    # Plot results and find regression line
    slopes_model[i], intercepts_model[i] = draw_plots.plot_graphs(P_hc_pred, P_bc_pred, "Handcycle: predicted Power Outputs", 
                            "Predicted hc [W]", "Predicted bc [W]", participants[i], # Labels for x and y axes
                            i, fig3, axs3, color_coded = "No") # Plot properties
    axs3[i].set_xlim([0, 100])
    axs3[i].set_ylim([50, 250])
    
#--------------------------------------------------------------------------------------------------------------------------------------------------
    # Plot coefficients and color code them 
#--------------------------------------------------------------------------------------------------------------------------------------------------
fig5, axs5 = plt.subplots(2, 3, constrained_layout = "True")
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
    axs5[0].set_xlabel("Beta")
    axs5[0].set_ylabel("Alpha")

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
    axs5[1].set_xlabel("Beta")
    axs5[1].set_ylabel("Alpha")


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
    axs5[2].set_xlabel("Beta")
    axs5[2].set_ylabel("Alpha")


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
    axs5[idx].set_xlabel("Beta")
    axs5[idx].set_ylabel("Alpha")

        # Color code by IPAQ result
    #------------------------------------------------------------------------------------------------------------------------------------
    feature = "Level of physical activity"
    idx = 4
    # Save the number of minutes per week dedicated to each level of activity
    low_activity = members[f"{ID}"]["Activity"][2]
    moderate_activity = members[f"{ID}"]["Activity"][1]
    vigorous_activity = members[f"{ID}"]["Activity"][0]

    ipaq = 3.3 * low_activity + 4 * moderate_activity + 8 * vigorous_activity 

    if ipaq <= 600:
        c = "green"
    elif ipaq > 600 and ipaq <= 1500:
        c = "yellow"
    elif ipaq > 1500:
        c = "red"

    axs5[idx].scatter(intercepts_model[i], slopes_model[i], color = c)

    legend_labels = ["Low", "Moderate", "Vigorous"]
    legend_colors = ["green", "yellow", "red"]
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
    axs5[idx].legend(legend_handles, legend_labels, loc='upper right')
    axs5[idx].set_title(f"{feature}")
    axs5[idx].set_xlabel("Beta")
    axs5[idx].set_ylabel("Alpha")
 

if plot_hr == "Yes":
    fig6, axs6 = plt.subplots(n_col, n_rows, constrained_layout = "True")
    axs6 = axs6.flatten()
    for i, ID in enumerate(participants):
        if ID < 10:
            ID = f"00{ID}"
        else:
            ID = f"0{ID}"

        # Handcycle
        data = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_handcycle_{mode}.xlsx")
        data = data.iloc[:int(len(data)/2), :]
        y = data["Heart Rate"] / (220 - members[f"{ID}"]["Age"] - 20) 
        axs6[i].plot(y, label = "Handcycle")

        # Bicycle
        data = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_bicycle_{mode}.xlsx")
        data = data.iloc[:int(len(data)/2), :]
        y = data["Heart Rate"] / (220 - members[f"{ID}"]["Age"]) 
        axs6[i].plot(y, label = "Bicycle")
        axs6[i].set_title(f"Participant {ID}")


    handles, labels = axs6[i].get_legend_handles_labels()
    fig6.legend(handles, labels, loc='lower right')
    fig6.suptitle("Heart rate trend")

#-------------------------------------------------------------------------------------------------------------------------------
    # Test coefficients
#-------------------------------------------------------------------------------------------------------------------------------

   # Define the plot creation method
#---------------------------------------------------------------------------------------------------------------------------------------------
class TestModel():
    def __init__(self):
        pass

    def plot_graph(self, coeff, fig, axs):
        for i, ID in enumerate(participants):
            ID = f"{ID:03}"

            # Original data
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_handcycle_{mode}.xlsx", usecols = ["Power"])
            data_hc = data_hc.iloc[int(len(data_hc)/2):]
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_bicycle_{mode}.xlsx", usecols = ["Power"])
            data_bc = data_bc.iloc[int(len(data_bc)/2):]

            # Corrected output
            predicted_bc = data_hc * coeff[i, 0] + coeff[i, 1]
            axs[i].plot(predicted_bc, color = "orange")
            axs[i].plot(data_bc, color = "blue")
            axs[i].set_title(f"Participant {ID}")
            axs[i].set_xlabel("Time [s]")
            axs[i].set_ylabel("Power [W]")

            RPE_hc = members[f"{ID}"]["RPE"][0]
            RPE_bc = members[f"{ID}"]["RPE"][1]

            legend_labels = [str(RPE_hc), str(RPE_bc)]   
            legend_colors = ["orange", "blue"] 
            legend_handles = [Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
            axs[i].legend(legend_handles, legend_labels, loc='upper left', prop={'size': 6})
            
            legend_labels = ["Tweaked handcycle", "Original bicycle"]
            fig.legend(legend_handles, legend_labels, loc='lower right')
test_model = TestModel()

#-------------------------------------------------------------------------------------------------------------------------------------------
    # Model with the original coefficients
#-------------------------------------------------------------------------------------------------------------------------------------------
slopes_model = slopes_model.reshape(-1, 1)
intercepts_model = intercepts_model.reshape(-1, 1)

fig7, axs7 = plt.subplots(n_col, n_rows, constrained_layout = "True")
axs7 = axs7.flatten()
coef_orig = np.concatenate([slopes_model, intercepts_model], axis = 1)
test_model.plot_graph(coef_orig, fig7, axs7)
fig7.suptitle("Tweaked Power output")

#--------------------------------------------------------------------------------------------------------------------------------------------
    # Multioutput regression (Same results as computing slopes and intercept separately)
#--------------------------------------------------------------------------------------------------------------------------------------------
fig8, axs8 = plt.subplots(2)
coef_predicted = np.zeros((len(participants), 2))

for i, ID in enumerate(participants):
    ID = f"{ID:03}"

    training = [
    [sub_dict[key] for key in ["Age", "Height", "Weight", "Gender", "FTP"] if key in sub_dict]
    for key, sub_dict in members.items() if key != ID]
    test = [
    [sub_dict[key] for key in ["Age", "Height", "Weight", "Gender", "FTP"] if key in sub_dict]
    for key, sub_dict in members.items() if key == ID]

    Y_train = np.delete(np.concatenate([slopes_model, intercepts_model], axis = 1), i, axis = 0)

    # Linear regression
    linear_regression = LinearRegression().fit(training, Y_train)
    linear_model = linear_regression.predict(test)
    coef_predicted[i, :] = linear_model[0, :]

    axs8[0].scatter(i, linear_model[0, 0], color = "red")
    axs8[1].scatter(i, linear_model[0, 1], color = "red")
    axs8[0].scatter(i, slopes_model[i], color = "blue")
    axs8[1].scatter(i, intercepts_model[i], color = "blue")

    legend_labels = ["Predicted", "Original"]   
    legend_colors = ["red", "blue"] 
    legend_handles = [Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
    axs8[0].legend(legend_handles, legend_labels, loc='upper left', prop={'size': 6})
    axs8[1].legend(legend_handles, legend_labels, loc='upper left', prop={'size': 6})
    
   # Test the predicted coefficients
#----------------------------------------------------------------------------------------------------------------------------------------------------
fig9, axs9 = plt.subplots(n_col, n_rows, constrained_layout = "True")
axs9 = axs9.flatten()
test_model.plot_graph(coef_predicted, fig9, axs9)
fig9.suptitle("Linear model: tweaked Power output")

    # Compare Original model and Linear Model
#-----------------------------------------------------------------------------------------------------------------------------------------------------
fig10, axs10 = plt.subplots(n_col, n_rows, constrained_layout = "True")
axs10 = axs10.flatten()


for i, ID in enumerate(participants):
    ID = f"{ID:03}"

    # Original data
    data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_handcycle_{mode}.xlsx", usecols = ["Power"])
    data_hc = data_hc.iloc[int(len(data_hc)/2):]

    # Corrected output
    predicted = data_hc * coef_predicted[i, 0] + coef_predicted[i, 1]
    orig = data_hc * coef_orig[i, 0] + coef_orig[i, 1]
    axs10[i].plot(predicted, color = "orange")
    axs10[i].plot(orig, color = "blue")
    axs10[i].set_title(f"Participant {ID}")
    axs10[i].set_xlabel("Time [s]")
    axs10[i].set_ylabel("Power [W]")

    RPE_hc = members[f"{ID}"]["RPE"][0]
    RPE_bc = members[f"{ID}"]["RPE"][1]

    legend_labels = [str(RPE_hc), str(RPE_bc)]   
    legend_colors = ["orange", "blue"] 
    legend_handles = [Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
    axs10[i].legend(legend_handles, legend_labels, loc='upper left', prop={'size': 6})
    
    legend_labels = ["Linear regression", "Original coefficients"]
    fig10.legend(legend_handles, legend_labels, loc='lower right')
    plt.title("Original coefficients vs predicted coefficients")

# plt.show()
    

#-------------------------------------------------------------------------------------------------------------------------------------
    # Look for a correlation
#-------------------------------------------------------------------------------------------------------------------------------------
age = np.zeros((len(participants)))
gender = np.zeros((len(participants)))
height = np.zeros((len(participants)))
weight = np.zeros((len(participants)))
IPAQ = np.zeros((len(participants)))
for i, ID in enumerate(participants):
    ID = f"{ID:03}"
    age[i] = members[f"{ID}"]["Age"]
    gender[i] = members[f"{ID}"]["Gender"]
    height[i] = members[f"{ID}"]["Height"]
    weight[i] = members[f"{ID}"]["Weight"]
    IPAQ[i] = 3.3 * members[f"{ID}"]["Activity"][2] + 4 * members[f"{ID}"]["Activity"][1] + 8 * members[f"{ID}"]["Activity"][0]

p = np.zeros((5, 2))
r = np.zeros((5, 2))
feature_labels = ["Age", "Gender", "Height", "Weight", "IPAQ"]

r[0, 0], p[0, 0] = pearsonr(age, coef_orig[:, 0])
r[1, 0], p[1, 0] = pearsonr(gender, coef_orig[:, 0])
r[2, 0], p[2, 0] = pearsonr(height, coef_orig[:, 0])
r[3, 0], p[3, 0] = pearsonr(weight, coef_orig[:, 0])
r[4, 0], p[4, 0] = pearsonr(IPAQ, coef_orig[:, 0])
r[0, 1], p[0, 1] = pearsonr(age, coef_orig[:, 1])
r[1, 1], p[1, 1] = pearsonr(gender, coef_orig[:, 1])
r[2, 1], p[2, 1] = pearsonr(height, coef_orig[:, 1])
r[3, 1], p[3, 1] = pearsonr(weight, coef_orig[:, 1])
r[4, 1], p[4, 1] = pearsonr(IPAQ, coef_orig[:, 1])

fig12, axs12 = plt.subplots(1, 2)
im1 = axs12[1].imshow(p, cmap='Blues', origin='upper', aspect='auto')
for i in range(p.shape[0]):
    for j in range(p.shape[1]):
        text = axs12[1].text(j, i, f"{p[i, j]:.2f}", ha="center", va="center", color="k", size='smaller')
            
axs12[1].set_xticks(np.arange(coef_orig.shape[1]), labels=["Alpha", "Beta"])
axs12[1].set_yticks(np.arange(5), labels = feature_labels)
axs12[1].xaxis.tick_top()
axs12[1].set_title("p value")

im2 = axs12[0].imshow(r, cmap='Blues', origin='upper', aspect='auto')
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        text = axs12[0].text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", color="k", size='smaller')
            
axs12[0].set_xticks(np.arange(coef_orig.shape[1]), labels=["Alpha", "Beta"])
axs12[0].set_yticks(np.arange(5), labels = feature_labels)
axs12[0].xaxis.tick_top()
axs12[0].set_title("r coefficient")

fig12.suptitle(f'Correlation between features and coefficients - Pearson')


if simple_model == "Yes":
    fig11, axs11 = plt.subplots(n_col, n_rows)
    axs11 = axs11.flatten()

    for i, ID in enumerate(participants):
        alpha = 0
        ID = ID = f"{ID:03}"

        # Import data
        data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_handcycle_{mode}.xlsx", usecols = ["Power", "RPE"])
        data_hc_RPE = data_hc.iloc[:int(len(data_hc)/2)] # Blocks at constant RPE target
        data_hc_power = data_hc.iloc[int(len(data_hc)/2):] # Blocks at constant power target
        data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_bicycle_{mode}.xlsx", usecols = ["Power", "RPE"])
        data_bc_RPE = data_bc.iloc[:int(len(data_hc)/2)] # Blocks at constant RPE target
        data_bc_power = data_bc.iloc[int(len(data_hc)/2):] # Blocks at constant power target

        alpha = alpha + np.mean(data_bc_RPE.loc[data_bc_RPE["RPE"] == 12, :])/np.mean(data_hc_RPE.loc[data_hc_RPE["RPE"] == 12, :])
        alpha = alpha + np.mean(data_bc_RPE.loc[data_bc_RPE["RPE"] == 14, :])/np.mean(data_hc_RPE.loc[data_hc_RPE["RPE"] == 14, :])
        alpha = alpha + np.mean(data_bc_RPE.loc[data_bc_RPE["RPE"] == 15, :])/np.mean(data_hc_RPE.loc[data_hc_RPE["RPE"] == 15, :])

        alpha = alpha / 3

        Predicted_output = alpha * data_hc_power["Power"]

        axs11[i].plot(data_bc_power["Power"])
        axs11[i].plot(Predicted_output)

        axs11[i].set_xlabel("Time [s]")
        axs11[i].set_ylabel("Power [W]")


        axs11[i].set_title(f"{ID}: alpha = {alpha:.2f}")
        legend_labels = ["Original bicycle", "Predicted value"]
        legend_colors = ["Blue", "Orange"]
        legend_handles = [Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
        fig11.legend(legend_handles, legend_labels, loc='lower right')

    fig11.suptitle("Simple model")

final = "Ne yake"
    
    
'''plt.figure()
feature = height
for i in enumerate(participants):
    values, y = np.unique(feature, return_counts=True)
plt.plot(values, y)
plt.axvline(np.mean(feature), color='red', linestyle='--', linewidth=2, label=f'Media = {np.mean(feature):.2f}')
plt.xlabel("Height")
plt.ylabel("recurrence")
'''

