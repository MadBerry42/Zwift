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



#-------------------------------------------------------------------------------------------------------------------------------
    # Multiplicative adjustment to decay Model
#-------------------------------------------------------------------------------------------------------------------------------

class MultiplicativeDecay():

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
    # Complete decay Model
#-------------------------------------------------------------------------------------------------------------------------------

class CompleteDecay():

    def __init__(self, D):
        self.Height = D.Height
        self.Weight = D.Weight
        self.Age = D.Age
    
        '''If you wish to see the plot, after
            def __init__(self): # , fig, axs, D): 
            add
            self.fig = fig
            self.axs = axs'''

    def fit(self, D:DataSet):
        self.predict

        alpha_array = np.zeros(3)
        gamma_array = np.zeros(3)
        delta_hr_array = np.zeros(3)
        delta_rpe_array = np.zeros(3)
        delta_height_array = np.zeros(3)
        delta_weight_array = np.zeros(3)
        delta_age_array = np.zeros(3)

        for i in range(3):
            t = D.time[i * 180 : (i+1) * 180]
            x = np.zeros((4, 180))
            x[0, :] = D.time[i * 180 : (i+1) * 180]
            x[1, :] = D.Power_hc[i * 180 : (i+1) * 180]
            x[2, :] = D.HR[i * 180 : (i+1) * 180]
            x[3, :] = D.RPE[i * 180 : (i+1) * 180]

            self.popt, pcov = curve_fit(self.model, x,  D.Power_bc[i * 180 : (i+1) * 180], p0=[00.1, 0.001, 0.0001, 0.0001, 0.1, 0.1, 0.1], maxfev = 10000) 
            alpha_array[i], gamma_array[i], delta_hr_array[i], delta_rpe_array[i], delta_weight_array[i], delta_height_array[i], delta_age_array[i] = self.popt

            Bc = self.predict(x, *self.popt)

            if i == 0:
                Model_hc = Bc
            else:
                Model_hc = np.concatenate((Model_hc, Bc))

        self.alpha_array = alpha_array
        self.gamma_array = gamma_array
        self.delta_hr_array = delta_hr_array
        self.delta_rpe_array = delta_rpe_array
        self.delta_height_array = delta_height_array
        self.delta_weight_array = delta_weight_array
        self.delta_age_array = delta_age_array

        self.alpha = np.mean(alpha_array)
        self.gamma = np.mean(gamma_array)
        self.delta_rpe = np.mean(delta_rpe_array)
        self.delta_hr = np.mean(delta_hr_array)
        self.delta_height = np.mean(delta_height_array)
        self.delta_weight = np.mean(delta_weight_array)
        self.delta_age = np.mean(delta_age_array)
    
        return Model_hc


    def model(self, x, alpha, gamma, delta_hr, delta_rpe, delta_height, delta_weight, delta_age):

        t = x[0, :]
        Power_hc = x[1, :]
        HR = x[2, :]
        RPE = x[3, :]

        m = alpha * Power_hc * (1 + gamma * t) + (1 + delta_hr * HR) + (1 + delta_rpe * RPE) + (1 + delta_weight *self.Weight) + (1 + delta_height * self.Height) + (1 + delta_age * self.Age) 

        return m
    


    def predict(self, x, alpha, gamma, delta_hr, delta_rpe, delta_height, delta_weight, delta_age):

        m = self.model(x, alpha, gamma, delta_hr, delta_rpe, delta_height, delta_weight, delta_age)

        return m
    

    def plot(self, ID, time, Power_bc, Model):

        self.fig.suptitle(f"Models vs bicycle, participant {ID}")
        self.axs[1, 1].plot(time, Power_bc, label="Observed bicycle power")
        self.axs[1, 1].plot(time, Model, label="Model")
        self.axs[1, 1].set_xlabel("Time [s]")
        self.axs[1, 1].set_ylabel("Power [W]")
        self.axs[1, 1].set_title(f"Complete multiplicative adjustment Model")
        self.axs[1, 1].legend()



class LinearRegression():
    def __init__(self):
        pass

    def create_matrices(self, dataset):
        height = np.ones(len(dataset.Power_bc)) * dataset.Height
        weight = np.ones(len(dataset.Power_bc)) * dataset.Weight
        age = np.ones(len(dataset.Power_bc)) * dataset.Age
        # With RPE
        # self.A = np.array([dataset.Power_hc, dataset.HR, dataset.RPE, dataset.cadence, height, weight, age]).T
        # Without RPE
        self.A = np.array([dataset.Power_hc, dataset.HR, dataset.cadence, height, weight, age]).T
        self.b = dataset.Power_bc

        self.xdag = np.linalg.pinv(self.A)@self.b
        self.lam = np.array([0, 0.1, 0.5]) # Different possible values of lambda 1 (l1 penalty)
        # If l1 = 0, the solution is equivalent to xdag 
        return self.A, self.b, self.lam


    def regression(self):
        A = self.A
        b = self.b
        lam = self.lam
        xdag = self.xdag

        def reg_norm(x, A, b, lam):
            return np.linalg.norm(A@x - b, ord = 2) + lam * np.linalg.norm(x, ord = 1)
        
        X = np.zeros((len(lam), A.shape[1]))
        for j in range(len(lam)):
            res = minimize(reg_norm, args = (A, b, lam[j]), x0 = xdag)
            X[j, :] = np.array(res.x)

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

    def implement_model(self, model, X, lam, fig, axs):
        if model == "linear":
            Tweaked_signal = self.Power_hc * 2.8442
            axs[0, 0].plot(self.t, Tweaked_signal, label = "Linear model")
            axs[0, 0].plot(self.t, self.Power_bc, label = " Bicycle signal")
            axs[0, 0].plot(self.t, self.Power_hc, label = " Original signal")
            axs[0, 0].legend()
        if model == "simple decay": 
            Tweaked_signal = self.Power_hc * 3.0577 * (1 + 0.00097 * self.t)
            axs[0, 1].plot(self.t, Tweaked_signal, label = "Simple decay model")
            axs[0, 1].plot(self.t, self.Power_bc, label = " Bicycle signal")
            axs[0, 1].plot(self.t, self.Power_hc, label = " Original signal")
            axs[0, 1].legend()
        if model == "multiplicative decay":
            Tweaked_signal = 0.689 * self.Power_hc * (1 + 0.00011 * self.t) * (1 + 0.0077* self.HR) * (1 + 0.3978 * self.RPE)
            axs[1, 0].plot(self.t, Tweaked_signal, label = "Multiplicative decay model")
            axs[1, 0].plot(self.t, self.Power_bc, label = " Bicycle signal")
            axs[1, 0].plot(self.t, self.Power_hc, label = " Original signal")
            axs[1, 0].legend()
        if model == "complete decay":
            Tweaked_signal = 1.761 * self.Power_hc + (1 + 1.7194 * self.t) + (1 - 0.225 * self.HR) + (1+ 22.90326 * self.RPE) + (1 - 0.42684 * self.Height) + (1 - 0.42864 * self.Weight) + (1 - 7.42725 * self.Age)
            axs[1, 1].plot(self.t, Tweaked_signal, label = "Complete variables decay model")
            axs[1, 1].plot(self.t, self.Power_bc, label = " Bicycle signal")
            axs[1, 1].plot(self.t, self.Power_hc, label = " Original signal")
            axs[1, 1].legend()
        if model == "linear regression":
            plt.figure()
            coeff = X[lam, :]
            # with RPE
            # Tweaked_signal = 8.39667443e-01 * self.Power_hc + 1.19691252e-01 * self.HR + 9.78136690 * self.RPE + 3.54213310e-01 * self.cadence - 2.55727294e-02 * self.Height - 1.04634556 * self.Weight - 3.54403299e-01 * self.Age
            # without RPE
            Tweaked_signal = coeff[0] * self.Power_hc + coeff[1] * self.HR + coeff[2] * self.cadence + coeff[3] * self.Height + coeff[4] * self.Weight + coeff[5] * self.Age
            plt.plot(self.t, Tweaked_signal, label = "Linear regression model")
            plt.plot(self.t, self.Power_bc, label = " Bicycle signal")
            plt.plot(self.t, self.Power_hc, label = " Original signal")
            plt.legend()            

        return Tweaked_signal

