import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#----------------------------------------------------------------------------------------------------------------------------------------------
    # Linear regression model
#----------------------------------------------------------------------------------------------------------------------------------------------
class AlphaBeta():
    def __init__(self):
        pass

    def compute_coeff(self, power_hc, hr_hc, power_bc, hr_bc, members, ID):
        # %HR vs Power: handcycle
        # Handcycle
        data = power_hc
        data = data.iloc[:int(len(data)/2)]
        x = hr_hc / (220 - members[f"{ID}"]["Age"] - 20) * 100
        y = power_hc
        slope_hc, intercept_hc = np.polyfit(x, y, 1) 
        # Bicycle
        data = power_bc
        data = data.iloc[:int(len(data)/2)]
        x = hr_bc / (220 - members[f"{ID}"]["Age"]) * 100
        y = power_bc
        slope_bc, intercept_bc = np.polyfit(x, y, 1)

        # Find coefficients which predict Pbc and Phc based on % Peak HR
        perc_hr = np.linspace(1, 100, 100)
        P_hc_pred = slope_hc * perc_hr + intercept_hc
        P_bc_pred = slope_bc * perc_hr + intercept_bc

        alpha, beta = np.polyfit(P_hc_pred, P_bc_pred, 1)

        return alpha, beta
    
    def predict_coefficients(self, members, alpha_array, beta_array, i, ID):
        # Define training and test set
        training = [
        [sub_dict[key] for key in ["Age", "Height", "Weight", "Gender", "FTP"] if key in sub_dict]
        for key, sub_dict in members.items() if key != ID]
        test = [
        [sub_dict[key] for key in ["Age", "Height", "Weight", "Gender", "FTP"] if key in sub_dict]
        for key, sub_dict in members.items() if key == ID]
        Y_train = np.delete(np.concatenate([alpha_array.reshape(1, -1), beta_array.reshape(1, -1)], axis = 0), i, axis = 1)
        linear_regression = LinearRegression().fit(training, Y_train.T)
        linear_model = linear_regression.predict(test)
        alpha, beta = linear_model[0, :]

        return alpha, beta
    
    def quantify_error(self, model, orig):
        x = orig - model
        MSE = sum(x**2)/(len(x)/2) 
        RMSE = np.sqrt(MSE)
        r_squared = r2_score(model, orig)
        max_error = max(abs(x))

        return MSE, RMSE, r_squared, max_error


class Gamma():
    def __init__(self):
        pass

    def compute_coefficients(self, data_hc, data_bc):
        data_hc = data_hc.iloc[:int(len(data_hc)/2)]
        data_bc = data_bc.iloc[:int(len(data_bc)/2)]

        # Find coefficients gamma
        gamma1 = np.mean(data_bc.loc[data_bc["RPE"] == 12, :].drop(columns = "RPE"))/np.mean(data_hc.loc[data_hc["RPE"] == 12, :].drop (columns = "RPE"))
        gamma2 = np.mean(data_bc.loc[data_bc["RPE"] == 14, :].drop(columns = "RPE"))/np.mean(data_hc.loc[data_hc["RPE"] == 14, :].drop(columns = "RPE"))
        gamma3 = np.mean(data_bc.loc[data_bc["RPE"] == 15, :].drop(columns = "RPE"))/np.mean(data_hc.loc[data_hc["RPE"] == 15, :].drop(columns = "RPE"))

        return gamma1, gamma2, gamma3
    
    def predict_coefficients(self, gamma1, gamma2, gamma3, members, k):
        # Find coefficient gamma_star
        IPAQ = 3.3 * members["Activity"][2] + 4 * members["Activity"][1] + 8 * members["Activity"][0]
        training = [[members["Age"], members["Weight"], members["Height"], members["Gender"], IPAQ, 12],
                    [members["Age"], members["Weight"], members["Height"], members["Gender"], IPAQ, 14],
                    [members["Age"], members["Weight"], members["Height"], members["Gender"], IPAQ, 15],
                    ]
        Y_train = [gamma1, gamma2, gamma3]
        linear_regression = LinearRegression().fit(training, Y_train)
        test = np.array([members["Age"], members["Weight"], members["Height"], members["Gender"], IPAQ, members["RPE"][0][k]])

        gamma_star = linear_regression.predict(test.reshape(1, -1))

        return gamma_star
    
    def quantify_error(self, model, orig):
        x = orig - model
        MSE = sum(x**2)/(len(x)/2) 
        RMSE = np.sqrt(MSE)
        r_squared = r2_score(model, orig)
        max_error = max(abs(x))

        return MSE, RMSE, r_squared, max_error


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


