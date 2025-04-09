import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import math

participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
path = r"C:\Users\maddy\Desktop\NTNU\Julia Kathrin Baumgart - Protocol Data"
mode = "raw"
plot_hr = "No"

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
'''members = { "000":{"Age": 25, "Height": 160, "Weight": 58, "Gender": 1, "FTP": 49},
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
            }'''

members = { "000":{"Age": 25, "Height": 160, "Weight": 58, "Gender": 1, "FTP": 49, "Activity": [0, 0, 60*7, 16*60*7]},
        "002":{"Age": 26, "Height": 177, "Weight": 75, "Gender": 0, "FTP": 78, "Activity": [0, 4*45, 6*30, 4*60*7]},
        "003":{"Age": 22, "Height": 180, "Weight": 70, "Gender": 0, "FTP": 51, "Activity": [4*60, 7*15, 3*30, 8*60*7]},
        "004":{"Age": 22, "Height": 186, "Weight": 80, "Gender": 0, "FTP": 47, "Activity": [3*90, 1*5, 5*30, 6*60*7]},
        "006":{"Age": 23, "Height": 174, "Weight": 87, "Gender": 1, "FTP": 51, "Activity": [2*40, 0, 7*120, 9*60*7 ]},
        "007":{"Age": 23, "Height": 183, "Weight": 70, "Gender": 0, "FTP": 55, "Activity": [0, 0, 60*7, 16*60*7]}, # Unsure about the sitting time
        "008":{"Age": 23, "Height": 190, "Weight": 82, "Gender": 0, "FTP": 82, "Activity": [4*90, 0, 7*30, 8*60*7]},
        "009":{"Age": 32, "Height": 185, "Weight": 96, "Gender": 0, "FTP": 62, "Activity": [0, 0, 5*60, 12*60*7]},
        "010":{"Age": 24, "Height": 160, "Weight": 56, "Gender": 1, "FTP": 48, "Activity": [5*60, 2*60, 7*30, 10*60*7]}, # Time spent for activity is grossly estimated
        "011":{"Age": 28, "Height": 176, "Weight": 67, "Gender": 0, "FTP": 60, "Activity": [0, 0, 7*40, 12*60*7]},
        "012":{"Age": 28, "Height": 184, "Weight": 70, "Gender": 0, "FTP": 87, "Activity": [0, 0, 4*20, 7*45, 10*60*7]},
        "013":{"Age": 25, "Height": 178, "Weight": 66, "Gender": 0, "FTP": 62, "Activity": [1*60, 1*60, 3*35, 4*60*7]},
        "015":{"Age": 21, "Height": 176, "Weight": 73, "Gender": 0, "FTP": 60, "Activity": [4*80, 6*120, 7*8*60, 3*60*7]},
        "016":{"Age": 24, "Height": 173, "Weight": 59, "Gender": 1, "FTP": 37, "Activity": [2*45, 2*30, 7*60, 9*60*7]},
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
    perc_hr = np.linspace(50, 80, 100)
    P_hc_pred = slopes[i, 0] * perc_hr + intercepts[i, 0]
    P_bc_pred = slopes[i, 1] * perc_hr + intercepts[i, 1]

    # Plot results and find regression line
    slopes_model[i], intercepts_model[i] = draw_plots.plot_graphs(P_hc_pred, P_bc_pred, "Handcycle: predicted Power Outputs", 
                            "Predicted hc [W]", "Predicted bc [W]", participants[i], # Labels for x and y axes
                            i, fig3, axs3, color_coded = "No") # Plot properties
    # axs3[i].set_xlim([40, 100])
    axs3[i].set_ylim([50, 250])
    
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

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig5.tight_layout()
plt.show()

if plot_hr == "Yes":
    fig6, axs6 = plt.subplots(4, 4)
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

        