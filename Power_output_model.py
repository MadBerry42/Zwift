import Analyze_power_ouput
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import savgol_filter

# Import data
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20]
# path = f"C:\\Users\\maddy\\Desktop\\NTNU\\Julia Kathrin Baumgart - Protocol Data"
path = f"C:\\Users\\maddalb\\NTNU\\Julia Kathrin Baumgart - Protocol Data"

# Assign constant values
n_rows = 4
n_cols = 5

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


#------------------------------------------------------------------------------------------------------------------------------------------
    # Linear regression model
#------------------------------------------------------------------------------------------------------------------------------------------
alpha = np.zeros((len(participants)))
beta = np.zeros((len(participants)))
alpha_pred = np.zeros((len(participants)))
beta_pred = np.zeros((len(participants)))
model_lr_raw = np.zeros((len(participants), 180 * 6))
model_lr_mav = np.zeros((len(participants), 180 * 6))
model_lr_sg = np.zeros((len(participants), 180 * 6))
model_lr_pred_raw = np.zeros((len(participants), 180 * 6))
model_lr_pred_mav = np.zeros((len(participants), 180 * 6))
model_lr_pred_sg = np.zeros((len(participants), 180 * 6))

fig, axs = plt.subplots(n_rows, n_cols)
axs = axs.flatten()
fig1, axs1 = plt.subplots(n_rows, n_cols)
axs1 = axs1.flatten()

alpha_beta = Analyze_power_ouput.AlphaBeta()

for j in range(3):
    for i, ID in enumerate(participants):
        ID = f"{ID:03}"

        if j == 0: # Import raw data
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_handcycle_raw.xlsx")
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_bicycle_raw.xlsx") 
        if j == 1: # Data filtered with moving average filter
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate hc", "RPE", "Cadence", "Power hc mav"])
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate bc", "RPE", "Cadence", "Power bc mav"])
        if j == 2: # Data filtered with salgov filter
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate hc", "RPE", "Cadence","Power hc SG"])
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate bc", "RPE", "Cadence","Power bc SG"])
                            
        if j == 0:
            alpha[i], beta[i] = alpha_beta.compute_coeff(data_hc["Power"], data_hc["Heart Rate"], data_bc["Power"], data_bc["Heart Rate"], members, ID)
        if j == 1:
            alpha[i], beta[i] = alpha_beta.compute_coeff(data_hc["Power hc mav"], data_hc["Heart Rate hc"], data_bc["Power bc mav"], data_bc["Heart Rate bc"], members, ID)
        if j == 2:
            alpha[i], beta[i] = alpha_beta.compute_coeff(data_hc["Power hc SG"], data_hc["Heart Rate hc"], data_bc["Power bc SG"], data_bc["Heart Rate bc"], members, ID)

        # Model the tweaked signal
        if j == 0:
            model = data_hc["Power"] * alpha[i] + beta[i]
            power_bc = data_bc["Power"]
        if j == 1:
            model= data_hc["Power hc mav"] * alpha[i] + beta[i]
            power_bc = data_bc["Power bc mav"]
        if j == 2:
            model = data_hc["Power hc SG"] * alpha[i] + beta[i]
            power_bc = data_bc["Power bc SG"]

        # Plot Model vs original signal
        t = np.linspace(0, 1080, 1080)
        axs[i].plot(t, model, color = "red")
        axs[i].plot(t, power_bc, color = "blue")
        MSE, RMSE, r_squared, max_error = alpha_beta.quantify_error(model, power_bc)
        axs[i].set_title(f"r^2 = {r_squared:.2f}, max error: {max_error:.2f}, \n MSE: {MSE:.2f}, RMSE: {RMSE:.2f}", fontsize = 10)

        if j == 0:
            fig.suptitle("Original coefficients, Raw data")
            df = pd.DataFrame.from_dict(members, orient="index").reset_index().rename(columns={"index": "ID"})
            ids = df["ID"].values.reshape(-1, 1)
            ages = np.array([member["Age"] for member in members.values()]).reshape(-1, 1)
            heights =  np.array([member["Height"] for member in members.values()]).reshape(-1, 1)
            weights =  np.array([member["Weight"] for member in members.values()]).reshape(-1, 1)
            gender =  np.array([member["Gender"] for member in members.values()]).reshape(-1, 1)
            df = pd.DataFrame(np.concatenate((ids, np.array(ages), np.array(gender), np.array(heights), np.array(weights), alpha.reshape(-1, 1), beta.reshape(-1, 1)), axis = 1))
            df.columns = ["ID", "Age", "Gender", "Height", "Weight", "Alpha raw", "Beta raw"]
        if j == 1 and i == len(participants) - 1:
            fig.suptitle("Original coefficients, MAV filter")
            df_tmp = pd.DataFrame({"Alpha mav": alpha, "Beta mav": beta})
            df = pd.concat((df, df_tmp), axis = 1)
        if j == 2 and i == len(participants) - 1: 
            fig.suptitle("Original coefficients, SG filter")
            df_tmp = pd.DataFrame({"Alpha SG": alpha, "Beta SG": beta})
            df = pd.concat((df, df_tmp), axis = 1)
        
        axs[i].set_xlabel("Time [s]")
        axs[i].set_ylabel("Power [W]")

        if j == 0:
            model_lr_raw[i] = model
        if j == 1:
            model_lr_mav[i] = model
        if j == 2:
            model_lr_sg[i] = model
        
    legend_labels = ["Bicycle data", "Model"]
    legend_colors = ["blue", "red"] 
    legend_handles = [Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
    fig.legend(legend_handles, legend_labels, loc='lower right')
    fig.tight_layout()

    # Predict coefficients from personal info via multioutput linear regression
    for i, ID in enumerate(participants):
        ID = f"{ID:03}"

        if j == 0: # Import raw data
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_handcycle_raw.xlsx")
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_bicycle_raw.xlsx") 
        if j == 1: # Data filtered with moving average filter
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate hc", "RPE", "Cadence", "Power hc mav"])
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate bc", "Power bc mav"])
        if j == 2: # Data filtered with salgov filter
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate hc", "RPE", "Cadence","Power hc SG"])
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate bc", "Power bc SG"])
           
        alpha_pred[i], beta_pred[i] = alpha_beta.predict_coefficients(members, alpha, beta, i, ID)
        if j == 0:
            model_pred = data_hc["Power"] * alpha_pred[i] + beta_pred[i]
            power_bc = data_bc["Power"]
            fig1.suptitle("Predicted coefficients, Raw Data")
            model_lr_pred_raw[i] = model_pred
        if j == 1:
            model_pred = data_hc["Power hc mav"] * alpha_pred[i] + beta_pred[i]
            power_bc = data_bc["Power bc mav"]
            fig1.suptitle("Predicted coefficients, MAV filter")
            model_lr_pred_mav[i] = model_pred
        if j == 2:
            model_pred = data_hc["Power hc SG"] * alpha_pred[i] + beta_pred[i]
            power_bc = data_bc["Power bc SG"]
            fig1.suptitle("Predicted coefficients, SG filter")
            model_lr_pred_sg[i] = model_pred

        # Plot Model vs original signal
        t = np.linspace(0, 1080, 1080)
        axs1[i].plot(t, model_pred, color = "red")
        axs1[i].plot(t, power_bc, color = "blue")

    legend_labels = ["Bicycle data", "Model"]
    legend_colors = ["blue", "red"] 
    legend_handles = [Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
    fig1.legend(legend_handles, legend_labels, loc='lower right')
    fig1.tight_layout()

    plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Gamma model
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
gamma = Analyze_power_ouput.Gamma()
model_gamma_raw = np.zeros((len(participants), 180 * 6))
model_gamma_mav = np.zeros((len(participants), 180 * 6))
model_gamma_sg = np.zeros((len(participants), 180 * 6))
model = np.zeros((180 * 6))
gamma_matrix = np.zeros((len(participants), 3))


fig, axs = plt.subplots(n_rows, n_cols)
axs = axs.flatten()

for j in range(3):
    for i, ID in enumerate(participants):
        ID = f"{ID:03}"
        if j == 0: # Import raw data
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_handcycle_raw.xlsx")
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_bicycle_raw.xlsx")
            power_hc = data_hc["Power"]
            power_bc = data_bc["Power"]
            gamma1, gamma2, gamma3 = gamma.compute_coefficients(data_hc[["Power", "RPE"]], data_bc[["Power", "RPE"]])

        if j == 1: # Data filtered with moving average filter
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate hc", "RPE", "Cadence", "Power hc mav"])
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate bc", "RPE", "Power bc mav"])
            power_hc = data_hc["Power hc mav"]
            power_bc = data_bc["Power bc mav"]
            gamma1, gamma2, gamma3 = gamma.compute_coefficients(data_hc[["Power hc mav", "RPE"]], data_bc[["Power bc mav", "RPE"]])

        if j == 2: # Data filtered with salgov filter
            data_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate hc", "RPE", "Cadence","Power hc SG"])
            data_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Heart Rate bc", "RPE", "Power bc SG"])
            power_hc = data_hc["Power hc SG"]
            power_bc = data_bc["Power bc SG"]
            gamma1, gamma2, gamma3 = gamma.compute_coefficients(data_hc[["Power hc SG", "RPE"]], data_bc[["Power bc SG", "RPE"]])

        gamma_matrix[i, 0] = gamma1
        gamma_matrix[i, 1] = gamma2
        gamma_matrix[i, 2] = gamma3
        
        # Create and plot the model
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # First block
        t = np.linspace(0, 179, 180)
        model[0 : 180] = power_hc[0 : 180] * gamma1
        axs[i].plot(t, model[0 : 180], color = "red")
        axs[i].plot(t, power_bc[0 : 180], color = "blue")
        # Second block
        t = np.linspace(180, 359, 180)
        model[180 : 360] = power_hc[180 : 360] * gamma2
        axs[i].plot(t, model[180 : 360], color = "red")
        axs[i].plot(t, power_bc[180 : 360], color = "blue")
        # Third block
        t = np.linspace(360, 540, 180)
        model[360 : 540] = power_hc[360 : 540] * gamma3
        axs[i].plot(t, model[360 : 540], color = "red")
        axs[i].plot(t, power_bc[360 : 540], color = "blue")
        # Fourth block
        t = np.linspace(540, 720, 180)
        gamma_star = gamma.predict_coefficients(gamma1, gamma2, gamma3, members[f"{ID}"], 0)
        model[540 : 720] = power_hc[540 : 720] * gamma_star
        axs[i].plot(t, model[540 : 720], color = "red")
        axs[i].plot(t, power_bc[540 : 720], color = "blue")
        # Fifth block
        t = np.linspace(720, 900, 180)
        gamma_star = gamma.predict_coefficients(gamma1, gamma2, gamma3, members[f"{ID}"], 1)
        model[720 : 900] = power_hc[720 : 900] * gamma_star
        axs[i].plot(t, model[720 : 900], color = "red")
        axs[i].plot(t, power_bc[720 : 900], color = "blue")
        # Sixth block
        t = np.linspace(900, 1080, 180)
        gamma_star = gamma.predict_coefficients(gamma1, gamma2, gamma3, members[f"{ID}"], 2)
        model[900:] = power_hc[900:] * gamma_star
        axs[i].plot(t, model[900:], color = "red")
        axs[i].plot(t, power_bc[900:], color = "blue")

        if j == 0:
            model_gamma_raw[i] = model
            fig.suptitle("Gamma correction factor, raw data")
            fig.show()
        if j == 1:
            model_gamma_mav[i] = model
            fig.suptitle("Gamma correction factor, mav filter")
            fig.show()
        if j == 2:
            model_gamma_sg[i] = model
            fig.suptitle("Gamma correction factor, sg filter")
            fig.show()

        MSE, RMSE, r_squared, max_error = gamma.quantify_error(model, power_bc)    

        axs[i].set_title(f"r^2 = {r_squared:.2f}, max error: {max_error:.2f}, \n MSE: {MSE:.2f}, RMSE: {RMSE:.2f}", fontsize = 10)
        axs[i].set_xlabel("Time [s]")
        axs[i].set_ylabel("Power [W]")            

        # Add to the the dataframe to save the Excel file
        if j == 0 and i == len(participants) - 1:
            df_tmp = pd.DataFrame({"Gamma1 raw": gamma_matrix[:, 0], "Gamma2 raw": gamma_matrix[:, 1], "Gamma3 raw": gamma_matrix[:, 2]})
            df = pd.concat((df, df_tmp), axis = 1)
        if j == 1 and i == len(participants) - 1:
            df_tmp = pd.DataFrame({"Gamma1 mav": gamma_matrix[:, 0], "Gamma2 mav": gamma_matrix[:, 1], "Gamma3 mav": gamma_matrix[:, 2]})
            df = pd.concat((df, df_tmp), axis = 1)
        if j == 2 and i == len(participants) - 1:
            df_tmp = pd.DataFrame({"Gamma1 SG": gamma_matrix[:, 0], "Gamma2 SG": gamma_matrix[:, 1], "Gamma3 SG": gamma_matrix[:, 2]})
            df = pd.concat((df, df_tmp), axis = 1)
        
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Plot both models at the same time, with colored background
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(n_rows, n_cols)
axs = axs.flatten()

color_map = {
    0: (1, 1, 0),
    1: (1, 165/255, 0),
    2: "r",
    3: "g"
}

for i, ID in enumerate(participants):
    ID = f"{ID:03}"
    power_bc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Power bc mav"])
    power_hc = pd.read_excel(f"{path}\\Input to models\\Power Output models\\{ID}_Input_file_filtered.xlsx", usecols = ["Power hc mav"])
    t = np.linspace(0, 1079, 1080)

    # Plot signals
    axs[i].plot(t, model_lr_mav[i], color = "red")
    axs[i].plot(t, model_gamma_mav[i], color = "green")
    axs[i].plot(t, power_bc, color = "blue")
    axs[i].plot(t, power_hc, color = "c")

    # Set axis labels and titles
    axs[i].set_xlabel("Time [s]")
    axs[i].set_ylabel("Power [W]")
    age = members[f"{ID}"]["Age"]
    gender = members[f"{ID}"]["Gender"]
    height = members[f"{ID}"]["Height"]
    weight = members[f"{ID}"]["Weight"]
    axs[i].set_title(f"{ID}: A = {age}, G: {gender}, \n H: {height}, W: {weight}", fontsize = 10)

    # Color background according to RPE difference
    RPE_diff = abs(np.array(members[f"{ID}"]["RPE"][0]) - np.array(members[f"{ID}"]["RPE"][1]))
    background_patches = [Patch(color=color, label=f"RPE diff = {key}") for key, color in color_map.items()]

    for j in range(3):
        start = 180 * (j + 3)
        final = 180 * (j + 4)
        c = color_map.get(RPE_diff[j])
        axs[i].axvspan(start, final, facecolor = c, alpha = 0.5)

    # Set labels
    legend_labels = ["Linear regression model", "Gamma correction factor", "Original bc Data", "Original hc data"]
    legend_colors = ["red", "green", "blue", "cyan"] 
    legend_handles = [Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
    
    handles = legend_handles + background_patches
    labels = legend_labels + [patch.get_label() for patch in background_patches]
    
    fig.legend(handles, labels, loc='lower right')
    fig.suptitle("Comparing different models")

plt.show()

# Save coefficient in an excel file
'''writer = pd.ExcelWriter(f'{path}\\Coefficients_raw_filtered.xlsx', engine = "openpyxl")
wb = writer.book
df.to_excel(writer, index = False)
wb.save(f'{path}\\Coefficients_raw_filtered.xlsx')
print("File has been succesfully saved!")'''

#-------------------------------------------------------------------------------------------------------------------------------------------------------
    # Observe the derivatives of the signal
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Subplots
fig, axs = plt.subplots(1, 2)
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20]
requested = [20]

# First derivative
for i, ID in enumerate(requested):
    ID = f"{ID:03}"

    '''path = f"C:\\Users\\maddalb\\NTNU\\Julia Kathrin Baumgart - FTP tests data\\{ID}\\Zwift"
    power_hc = pd.read_csv(f"{path}\\{ID}_handcycle_FTP.csv", usecols= ["Power"])
    power_bc = pd.read_csv(f"{path}\\{ID}_bicycle_FTP.csv", usecols= ["Power"])
    power_bc = power_bc[480 : 1000]
    power_hc = power_hc[480 : 1000]'''

    path = f"C:\\Users\\maddalb\\NTNU\\Julia Kathrin Baumgart - Protocol Data\\Input to models\\Power Output models\\"
    power_bc = pd.read_excel(f"{path}\\{ID}_Input_file_filtered.xlsx", usecols = ["Power bc mav"])
    power_hc = pd.read_excel(f"{path}\\{ID}_Input_file_filtered.xlsx", usecols = ["Power hc mav"])

    power_bc = power_bc[100:500]
    power_hc = power_hc[100:500]
     
    power_bc_sg = savgol_filter(np.array(power_bc).flatten(), 100, 4, deriv = 1)
    power_hc_sg = savgol_filter(np.array(power_hc).flatten(), 100, 4, deriv = 1)

    t = np.linspace(100, 499, 400)
    axs[0].plot(t, power_bc, color = "blue", linewidth = 3)
    axs[0].plot(t, power_hc, color = "red", linewidth = 1)
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel(r"$\Delta$Power [W/s]")
    axs[0].grid()

    axs[1].plot(t, power_bc_sg, color = (0.5, 0.5, 0.5), linewidth = 3)
    axs[1].plot(t, power_hc_sg, color = (0, 0, 0), linewidth = 1)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel(r"$\Delta$Power [W/s]")
    axs[1].grid()

    legend_labels = ["Bicycle", "Handcycle"]
    legend_colors = [(0.5, 0.5, 0.5), (0, 0, 0)] 
    legend_handles = [Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=20) for color in legend_colors]
    plt.legend(legend_handles, legend_labels, loc='upper left', prop={"size": 15})

plt.suptitle("First Derivative of the signal")

# Plots, FTP test
fig, axs = plt.subplots(1, 2)
participants = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20]
requested = [12]


# First derivative
for i, ID in enumerate(requested):
    ID = f"{ID:03}"

    path = f"C:\\Users\\maddalb\\NTNU\\Julia Kathrin Baumgart - FTP tests data\\{ID}\\Zwift"
    power_hc = pd.read_csv(f"{path}\\{ID}_handcycle_FTP.csv", usecols= ["Power"])
    power_bc = pd.read_csv(f"{path}\\{ID}_bicycle_FTP.csv", usecols= ["Power"])

    # Filter power
    window_size = 5
    window = np.ones(window_size) / window_size
    window = window.flatten()
    power_bc = np.convolve(np.array(power_bc).flatten(), window, mode = "same")
    power_hc = np.convolve(np.array(power_hc).flatten(), window, mode = "same")

    power_hc = power_hc[480 : 780]
    power_bc = power_bc[300 : 600]

    power_bc_sg = savgol_filter(np.array(power_bc).flatten(), 100, 4, deriv = 1)
    power_hc_sg = savgol_filter(np.array(power_hc).flatten(), 100, 4, deriv = 1)

    t = np.linspace(0, len(power_bc) - 1, len(power_bc))
    axs[1].plot(t, power_bc_sg, color = (0.5, 0.5, 0.5), linewidth = 3, label = "Bicycle")
    axs[1].plot(t, power_hc_sg, color = (0, 0, 0), linewidth = 1, label = "Handcycle")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel(r"$\Delta$Power [W/s]")
    axs[1].set_title("First derivative of the signal")
    axs[1].legend()
    axs[1].grid()

    axs[0].plot(t, power_bc, color = "blue", label = "Bicycle")
    axs[0].plot(t, power_hc, color = "red", label = "Handcycle")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel(r"$\Delta$Power [W/s]")
    axs[0].set_title("Raw Data")
    axs[0].legend()
    axs[0].grid()

    

