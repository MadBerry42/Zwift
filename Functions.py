#---------------------------------------------------------------------------------------------------------------------------------------
    # Correlation between data
#---------------------------------------------------------------------------------------------------------------------------------------

import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f
import math
import pandas as pd 
from scipy.optimize import curve_fit

#----------------------------------------------------------------------------------------------------------------------------------------
    # Linear model
#----------------------------------------------------------------------------------------------------------------------------------------
class DataSet():

    def __init__(self, path:str, ID:str):
        
        data = pd.read_excel(f"{path}\\{ID}_input_file.xlsx")
        # data = pd.read_csv("Fake_data.csv")
        self.HR = data.iloc[:, 2]
        self.RPE = data.iloc[:, 3]
        self.Power_hc = data.iloc[:, 4].to_numpy()
        self.Power_bc = data.iloc[:, 5].to_numpy()
        self.time = np.linspace(0, 540, len(self.Power_hc))

        self.Gender = data.iloc[1, 1]
        self.Age = data.iloc[2, 1]
        self.Weight = data.iloc[3, 1]
        self.Height = data.iloc[4, 1]/100
        

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

    def __init__(self, fig, axs):
        self.fig = fig
        self.axs = axs

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



#-------------------------------------------------------------------------------------------------------------------------------
    # Multiplicative adjustment to decay Model
#-------------------------------------------------------------------------------------------------------------------------------

class MultiplicativeDecay():

    def __init__(self, fig, axs):
        self.fig = fig
        self.axs = axs

    def fit(self, D:DataSet):
        self.predict

        alpha_array = np.zeros(3)
        gamma_array = np.zeros(3)
        delta_hr_array = np.zeros(3)
        delta_rpe_array = np.zeros(3)

        for i in range(3):
            t = D.time[i * 180 : (i+1) * 180]
            x = np.zeros((4, 180))
            x[0, :] = D.time[i * 180 : (i+1) * 180]
            x[1, :] = D.Power_hc[i * 180 : (i+1) * 180]
            x[2, :] = D.HR[i * 180 : (i+1) * 180]
            x[3, :] = D.RPE[i * 180 : (i+1) * 180]

            self.popt, pcov = curve_fit(self.model, x,  D.Power_bc[i * 180 : (i+1) * 180], p0=[0.1, 0.001, 0.001, 0.001], maxfev = 10000) 
            alpha_array[i], gamma_array[i], delta_hr_array[i], delta_rpe_array[i] = self.popt

            Bc = self.predict(x, *self.popt)

            if i == 0:
                Model_hc = Bc
            else:
                Model_hc = np.concatenate((Model_hc, Bc))

        self.alpha_array = alpha_array
        self.gamma_array = gamma_array
        self.delta_hr_array = delta_hr_array
        self.delta_rpe_array = delta_rpe_array
        self.alpha = np.mean(alpha_array)
        self.gamma = np.mean(gamma_array)
        self.delta_rpe = np.mean(delta_rpe_array)
        self.delta_hr = np.mean(delta_hr_array)
    
        return Model_hc


    def model(self, x, alpha, gamma, delta_hr, delta_rpe):

        t = x[0, :]
        Power_hc = x[1, :]
        HR = x[2, :]
        RPE = x[3, :]

        m = alpha * Power_hc * (1 - gamma * t) * (1 + delta_hr * HR) * (1 + delta_rpe * RPE)

        return m
    


    def predict(self, x, alpha, gamma, delta_hr, delta_rpe):

        m = self.model(x, alpha, gamma, delta_hr, delta_rpe)

        return m
    

    def plot(self, ID, time, Power_bc, Model):

        self.fig.suptitle(f"Models vs bicycle, participant {ID}")
        self.axs[1, 0].plot(time, Power_bc, label="Observed bicycle power")
        self.axs[1, 0].plot(time, Model, label="Model")
        self.axs[1, 0].set_xlabel("Time [s]")
        self.axs[1, 0].set_ylabel("Power [W]")
        self.axs[1, 0].set_title(f"Multiplicative adjustment Model")
        self.axs[1, 0].legend()


#-------------------------------------------------------------------------------------------------------------------------------
    # Exponential decay Model
#-------------------------------------------------------------------------------------------------------------------------------

class ExponentialDecay():

    def __init__(self, fig, axs):
        self.fig = fig
        self.axs = axs

    def fit(self, D:DataSet):
        self.predict

        alpha_array = np.zeros(3)
        gamma_array = np.zeros(3)
        delta_hr_array = np.zeros(3)
        delta_rpe_array = np.zeros(3)

        for i in range(3):
            t = D.time[i * 180 : (i+1) * 180]
            x = np.zeros((4, 180))
            x[0, :] = D.time[i * 180 : (i+1) * 180]
            x[1, :] = D.Power_hc[i * 180 : (i+1) * 180]
            x[2, :] = D.HR[i * 180 : (i+1) * 180]
            x[3, :] = D.RPE[i * 180 : (i+1) * 180]

            self.popt, pcov = curve_fit(self.model, x,  D.Power_bc[i * 180 : (i+1) * 180], p0=[00.1, 0.001, 0.0001, 0.0001], maxfev = 10000) 
            alpha_array[i], gamma_array[i], delta_hr_array[i], delta_rpe_array[i] = self.popt

            Bc = self.predict(x, *self.popt)

            if i == 0:
                Model_hc = Bc
            else:
                Model_hc = np.concatenate((Model_hc, Bc))

        self.alpha_array = alpha_array
        self.gamma_array = gamma_array
        self.delta_hr_array = delta_hr_array
        self.delta_rpe_array = delta_rpe_array
        self.alpha = np.mean(alpha_array)
        self.gamma = np.mean(gamma_array)
        self.delta_rpe = np.mean(delta_rpe_array)
        self.delta_hr = np.mean(delta_hr_array)
    
        return Model_hc


    def model(self, x, alpha, gamma, delta_hr, delta_rpe):

        t = x[0, :]
        Power_hc = x[1, :]
        HR = x[2, :]
        RPE = x[3, :]

        m = alpha * Power_hc * np.exp(-(gamma + delta_hr * HR+ delta_rpe * RPE) * t)

        return m
    


    def predict(self, x, alpha, gamma, delta_hr, delta_rpe):

        m = self.model(x, alpha, gamma, delta_hr, delta_rpe)

        return m
    

    def plot(self, ID, time, Power_bc, Model):

        self.fig.suptitle(f"Models vs bicycle, participant {ID}")
        self.axs[1, 1].plot(time, Power_bc, label="Observed bicycle power")
        self.axs[1, 1].plot(time, Model, label="Model")
        self.axs[1, 1].set_xlabel("Time [s]")
        self.axs[1, 1].set_ylabel("Power [W]")
        self.axs[1, 1].set_title(f"Exponential decay Model")
        self.axs[1, 1].legend()


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



        





#-----------------------------------------------------------------------------------------------------------------------------------
    # Statistical analysis
#-----------------------------------------------------------------------------------------------------------------------------------


def get_Bland_Altman_plot(gold_standard, validated):
    Sxy = np.std(gold_standard - validated)
    dxy = np.mean(gold_standard - validated)
    lim_sup = dxy + 2 * Sxy
    lim_inf = dxy - 2 * Sxy

    # Uncomment if you wish to see the plot
    '''plt.figure()
    plt.plot((gold_standard + validated)/2, (gold_standard - validated)/2, '*')
    plt.axhline(y = dxy, color = "b")
    plt.axhline(y = lim_sup, linestyle = "-.")
    plt.axhline(y = lim_inf, linestyle = "-.")
    plt.set_title("Bland-Altman Plot")'''

    return Sxy, dxy, lim_sup, lim_inf

