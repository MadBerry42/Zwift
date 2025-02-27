#---------------------------------------------------------------------------------------------------------------------------------------
    # Correlation between data
#---------------------------------------------------------------------------------------------------------------------------------------

import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f
import math
import pandas as pd 
from scipy.optimize import curve_fit, minimize
# RPE model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D

#----------------------------------------------------------------------------------------------------------------------------------------
    # Linear model
#----------------------------------------------------------------------------------------------------------------------------------------
class DataSet():

    def __init__(self, file:str):
        
        data = pd.read_excel(f"{file}")
        # data = pd.read_csv("Fake_data.csv")
        self.HR = data.iloc[:, 2]
        self.RPE = data.iloc[:, 3]
        self.cadence = data.iloc[:, 4]
        self.Power_hc = data.iloc[:, 5].to_numpy()
        self.Power_bc = data.iloc[:, 6].to_numpy()
        self.time = np.linspace(300, 1379, len(self.Power_hc))

        self.Gender = int(data.iloc[1, 1])
        self.Age = int(data.iloc[2, 1])
        self.Weight = int(data.iloc[3, 1])
        self.Height = int(data.iloc[4, 1])/100
        

class LinearModel():
    
    def __init__(self):
        pass

    def fit(self, D:DataSet):
        alpha_array = np.zeros(3)
        Power_hc = D.Power_hc
        Power_bc = D.Power_bc

        for i in range(3):
            avg_power_hc = np.mean(Power_hc[i * 180 : (i+1) * 180])
            avg_power_bc = np.mean(Power_bc[i * 180 : (i+1) * 180])
            alpha_array[i] = avg_power_bc/avg_power_hc

        self.alpha_array = alpha_array   
        self.alpha = np.mean(alpha_array)

    
    def predict(self, Power_hc):

        m = Power_hc * self.alpha

        return m
    
    def plot(self, ID, time, Power_bc, Model):

        self.fig, self.axs = plt.subplots(2, 2)

        self.fig.suptitle(f"Models vs bicycle, participant {ID}")
        self.axs[0, 0].plot(time, Power_bc, label="Observed bicycle power")
        self.axs[0, 0].plot(time, Model, label="Linear model")
        self.axs[0, 0].set_xlabel("Time [s]")
        self.axs[0, 0].set_ylabel("Power [W]")
        self.axs[0, 0].set_title(f"Linear Model")
        self.axs[0, 0].legend()


#-------------------------------------------------------------------------------------------------------------------------------
    # Simple decay Model
#-------------------------------------------------------------------------------------------------------------------------------

class SimpleDecay():

    def __init__(self): # , fig, axs): 
        '''self.fig = fig
        self.axs = axs'''
        pass

    '''If you wish to see the plot, 
        def __init__(self): # , fig, axs): 
        self.fig = fig
        self.axs = axs'''

    def fit(self, D:DataSet):
        self.predict

        alpha_array = np.zeros(3)
        gamma_array = np.zeros(3)

        for i in range(3):
            t = D.time[i * 180 : (i+1) * 180]
            x = np.zeros((2, 180))
            x[0, :] = D.time[i * 180 : (i+1) * 180]
            x[1, :] = D.Power_hc[i * 180 : (i+1) * 180]

            self.popt, pcov = curve_fit(self.model, x,  D.Power_bc[i * 180 : (i+1) * 180], p0=[0.1, 0.001], maxfev = 10000) 
            alpha_array[i], gamma_array[i] = self.popt[0], self.popt[1]

            Bc = self.predict(t, x[1, :], *self.popt)

            if i == 0:
                Model_hc = Bc
            else:
                Model_hc = np.concatenate((Model_hc, Bc))

        self.alpha_array = alpha_array
        self.gamma_array = gamma_array
        self.alpha = np.mean(alpha_array)
        self.gamma = np.mean(gamma_array)
    
        return Model_hc


    def model(self, x, alpha, gamma):

        t = x[0, :]
        Power_hc = x[1, :]

        m = alpha * Power_hc * (1 - gamma * t)

        return m
    


    def predict(self, t, Power_hc, alpha, gamma):

        x = np.zeros((2, len(t)))
        x[0, :] = t
        x[1, :] = Power_hc

        m = self.model(x, alpha, gamma)

        return m
    
    def plot(self, ID, time, Power_bc, Model):

        self.fig.suptitle(f"Models vs bicycle, participant {ID}")
        self.axs[0, 1].plot(time, Power_bc, label="Observed bicycle power")
        self.axs[0, 1].plot(time, Model, label="Model")
        self.axs[0, 1].set_xlabel("Time [s]")
        self.axs[0, 1].set_ylabel("Power [W]")
        self.axs[0, 1].set_title(f"Simple decay Model")
        self.axs[0, 1].legend()


class LinearRegression():
    def __init__(self, lam):
        self.lam = lam  # Different possible values of lambda 1 (l1 penalty)
        # If l1 = 0, the solution is equivalent to xdag 

    def create_matrices(self, dataset, obj_function, mode:str):
        if mode == "true": # RPE is included
            height = np.ones(len(obj_function)) * dataset.Height
            weight = np.ones(len(obj_function)) * dataset.Weight
            age = np.ones(len(obj_function)) * dataset.Age
            self.A = np.array([dataset.Power_hc, dataset.HR, dataset.RPE, dataset.cadence, height, weight, age]).T
        elif mode == "false": # RPE is not included
            self.A = np.array(dataset)
        
        self.b = obj_function

        self.xdag = np.linalg.pinv(self.A)@self.b
        return self.A, self.b


    def regression(self):
        A = self.A
        b = self.b
        b = np.array(b).flatten()
        lam = self.lam
        xdag = self.xdag

        def reg_norm(x, A, b, lam):
            return np.linalg.norm(A@x - b, ord = 2) + lam * (np.linalg.norm(x, ord = 1) + np.linalg.norm(x, ord=2))
        
        X = np.zeros((1, A.shape[1]))
        xdag = np.array(xdag)
        xdag = xdag.ravel()
        X = np.linalg.lstsq(A, b)
        X = X[0]
        # res = minimize(reg_norm, args = (A, b, lam), x0 = xdag, method='L-BFGS-B')
        # X = np.array(res.x)

        return X
    
    def plot_regression(self, A, X):
        lam = self.lam
        t = np.linspace(300, 840, len(self.b))
        fig, axs = plt.subplots(len(lam))
        fig.suptitle("Linear regression")
        for j in range(len(lam)):
            x = X[j, :]
            axs[j].plot(t, A@x, label = "Model")
            axs[j].plot(t, self.b, label = "Original signal")
            axs[j].set_title(f"Lambda = {lam[j]}")
            
    





class StatisticalAnalysis():
    def __init__(self):
        pass

    # OLP line
    def olp_line(self, gold_standard, validated):
        mean_gs = statistics.mean(gold_standard)
        mean_val = statistics.mean(validated)

        stdev_gs = statistics.stdev(gold_standard)
        stdev_val = statistics.stdev(validated)

        # Parameters for the OLP line
        self.b = stdev_val / stdev_gs
        self.a = mean_val - self.b * mean_gs

        coefficients = np.polyfit(gold_standard, validated, 1)
        self.y_fit = np.polyval(coefficients, gold_standard)

        # Confidence intervals
        R = np.corrcoef(gold_standard, validated)
        r = R[0, 1]
        alpha = 0.05
        n = len(gold_standard)
        B = f.ppf(1 - alpha, 1, n-2) * (1 - r**2)/(n-2)
        self.CI_b = [self.b * math.sqrt(B + 1) - math.sqrt(B), self.b * math.sqrt(B + 1) + math.sqrt(B)]
        self.CI_a = [mean_val - self.b * (math.sqrt(B + 1) + math.sqrt(B))*mean_gs, mean_val - self.b*(math.sqrt(B+1) - math.sqrt(B)) * mean_gs]

        return self

    def plot_olp(self, b, a, gold_standard, validated, y_fit, model, coord, fig, axs):
        axs[coord[0], coord[1]].scatter(gold_standard, validated, c = 'orange', s = 10, label = "Data")
        axs[coord[0], coord[1]].plot(gold_standard, gold_standard, c = 'green', label = "Bisecant")
        axs[coord[0], coord[1]].plot(gold_standard, y_fit, c = 'red', label = "Interpolating line")
        axs[coord[0], coord[1]].legend(loc = "upper left", fontsize = "small")
        axs[coord[0], coord[1]].set_title(f"{model}, a = {a:.3f}, b = {b:.3f}")



    def get_MSE_and_RMSE(self, gold_standard, validated):
        MSE = 0
        for i in range(len(gold_standard)):
            squared_error = (gold_standard[i] - validated[i])**2
            MSE = MSE + squared_error
        self.MSE = MSE / len(gold_standard)
        self.RMSE = np.sqrt(MSE)

        return self


    # Bland Altman plot
    def bland_altman(self, gold_standard, validated):
        self.Sxy = np.std(gold_standard - validated)
        self.dxy = np.mean(gold_standard - validated)
        self.lim_sup = self.dxy + 2 * self.Sxy
        self.lim_inf = self.dxy - 2 * self.Sxy

    def plot_ba(self, gold_standard, validated, model, coord, fig, axs):
        axs[coord[0], coord[1]].plot((gold_standard + validated)/2, (gold_standard - validated)/2, '*')
        axs[coord[0], coord[1]].axhline(y = self.dxy, color = 'b')
        axs[coord[0], coord[1]].axhline(y = self.lim_sup, linestyle = '-.')
        axs[coord[0], coord[1]].axhline(y = self.lim_inf, linestyle = '-.')
        axs[coord[0], coord[1]].set_title(f"{model} model")





#-------------------------------------------------------------------------------------------------------------------------------
    # Validation
#-------------------------------------------------------------------------------------------------------------------------------
class ValidateModel():
    def __init__(self, dataset):
        self.Power_hc = dataset.Power_hc
        self.Power_bc = dataset.Power_bc
        self.HR = dataset.HR
        self.RPE = dataset.RPE
        self.t = dataset.time
        self.Height = dataset.Height
        self.Weight = dataset.Weight
        self.Age = dataset.Age
        self.cadence = dataset.cadence

        pass

    def implement_model(self, model:str, X, index, fig, axs, mode:str):
        if model == "linear":
            Tweaked_signal = self.Power_hc * 2.8442
            axs[0].plot(self.t, Tweaked_signal, label = "Linear model")
            axs[0].plot(self.t, self.Power_bc, label = " Bicycle signal")
            axs[0].plot(self.t, self.Power_hc, label = " Original signal")
            fig.supylabel("Power [W]")
            axs[0].legend()
            axs

        if model == "linear regression":

            coeff = X[index]
            if mode == "true": # with RPE
                Tweaked_signal = coeff[0] * self.Power_hc + coeff[1] * self.HR + coeff[2] * self.RPE + coeff[3] * self.cadence + coeff[4] * self.Height + coeff[5] * self.Weight + coeff[6] * self.Age
            elif mode == "false": # without RPE
                Tweaked_signal = coeff[0] * self.Power_hc + coeff[1] * self.HR + coeff[2] * self.cadence + coeff[3] * self.Height + coeff[4] * self.Weight + coeff[5] * self.Age
            
            axs[1].plot(self.t, Tweaked_signal, label = "Linear regression")
            axs[1].plot(self.t, self.Power_bc, label = " Bicycle signal")
            axs[1].plot(self.t, self.Power_hc, label = " Original signal")
            axs[1].set_xlabel("Time [s]")
            axs[1].legend()
            plt.legend()            

        return Tweaked_signal


class RPEModel():
    def __init__(self, n_windows:int, participants):
        self.n_windows = n_windows
        self.len_window = 180/n_windows
        self.participants = participants

    def preprocessing(self, data_or):
        data = data_or.drop(columns = ["ID", "RPE"])
        self.scaler = MinMaxScaler()
        data = pd.DataFrame(self.scaler.fit_transform(data), columns = data.columns)
        self.scaler_rpe = MinMaxScaler()
        RPE_or = pd.DataFrame(self.scaler_rpe.fit_transform(data_or.iloc[:, [5]]))
        
        return data, RPE_or

    def leave_p_out(self, data, RPE_or):
        linear_regression = LinearRegression(lam=1e10)
        n_windows = self.n_windows
        participants = self.participants

        coeff = np.zeros((len(participants) * n_windows, data.shape[1]))
        RPE_predicted = np.zeros((len(participants), n_windows * 6))
        RPE_measured = np.zeros((len(participants), 6 * n_windows))

        for i in range(len(participants)):
        # Create the test participant for cross validation
            test = np.zeros((n_windows * 6, data.shape[1]))
            RPE_test = np.zeros((n_windows * 6, 1))
            for j in range (6):
                start = len(participants) * n_windows * j + i * n_windows 
                test[n_windows * j : n_windows * (j + 1)] = data.iloc[start :  start + n_windows]
                RPE_test[n_windows * j : n_windows * (j + 1)] = RPE_or.iloc[start :  start + n_windows]

            # Remove the test participant
            for j in range(n_windows):  
                index = [(len(participants) * n_windows) * k for k in [0, 1, 2, 3, 4, 5]]
                index = [x + (j + i * n_windows) for x in index]

                if j == 0:
                    dataset = data.drop(index)
                    RPE = RPE_or.drop(index)
                else:
                    dataset = dataset.drop(index)
                    RPE = RPE.drop(index)

            
            A, b = linear_regression.create_matrices(dataset, RPE, "false")
            X = linear_regression.regression()

            # Save results in a matrix
            coeff[j, :] = X
            
            # Apply the model to the control participant
            predicted = test @ X
            # De-scale the data
            predicted = predicted.reshape(1, -1)
            prediction = self.scaler_rpe.inverse_transform(predicted)
            # prediction = np.round(prediction)

            RPE_predicted[i, :] = prediction
            RPE_test = RPE_test.reshape(1, -1)
            RPE_measured[i, :] = self.scaler_rpe.inverse_transform(RPE_test)

        return RPE_measured, RPE_predicted
        
    def visualize_results_scatter(self, RPE_measured, RPE_predicted, length_window:int):
        participants = self.participants

        # Scattered data points
        plt.figure()
        for i in range(len(participants)):
            x = participants[i]
            y_measured = RPE_measured[i, :]
            y_predicted = RPE_predicted[i, :]

            # Color the background
            a = 0.01
            light_blue = (173/255, 216/255, 230/255)
            plt.axhspan(9, 11, color = light_blue, alpha = a)
            yellow = (1, 1, 0)
            plt.axhspan(11, 12, color = yellow, alpha = a)
            green = (0, 1, 0)
            plt.axhspan(12, 13, color = green, alpha = a)
            cyan = (0, 1, 1)
            plt.axhspan(13, 14, color = cyan, alpha = a)
            orange = (1, 0.647, 0)
            plt.axhspan(14, 15, color = orange, alpha = a)
            red = (1, 0, 0)
            plt.axhspan(15, 16, color = red, alpha = a)
            black = (0, 0, 0)
            plt.axhspan(16, 17, color = black, alpha = a)


            # Plot the data points, color coded accoding to their value
            for y1, y2 in zip(y_measured, y_predicted):
                if y1 <= 10:
                    c = light_blue
                elif y1 == 11:
                    c = yellow
                elif y1 == 12:
                    c = green
                elif y1 == 13:
                    c = cyan
                elif y1 == 14:
                    c = orange
                elif y1 == 15:
                    c = red
                elif y1 > 15:
                    c = black


                # plt.scatter(x + 0.5, y1, marker = 'x', color = c)
                plt.scatter(x, y2, marker = 'o', color = c)

            # Measure the difference between model prediction and actual distribution: R^2 value. Expected values between 0 and 1
            r_squared = r2_score(y_measured, y_predicted)
            print(f"The R^2 value for subject {participants[i]} is: {r_squared:.4f}")

        # Legend for color coding the RPE signal
        legend_labels = ['<11 (Very light)', '11 (light)', '12 (light)', '13 (slightly hard)', '14 (hard)', '15 (really hard)', '>15 (exhaustion)']
        legend_colors = [light_blue, yellow, green, cyan, orange, red, black]
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
        plt.legend(legend_handles, legend_labels, loc='upper center', ncol = 4, fontsize=8)
        plt.title(f"{length_window} second long window")


        plt.xlabel("Participant ID")
        plt.ylabel("RPE values")
        plt.xticks(participants)

        return plt

    # Visualize results as a curve
    def visualize_results_plot(self, RPE_measured, RPE_predicted, n_windows, fig, axs, handle1, handle2):
        x_axis = np.linspace(0, n_windows * 6 - 1, n_windows * 6)

        axs[handle1, handle2].scatter(x_axis, RPE_measured, color = (1, 0, 0), marker = 'x', s = 20, label = "Reported values")
        axs[handle1, handle2].scatter(x_axis, RPE_predicted, color = (0, 0, 1), marker = 'o', s = 20, label = "Predicted values")
        

        fig.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=10)
        fig.tight_layout()

