#---------------------------------------------------------------------------------------------------------------------------------------
    # Correlation between data
#---------------------------------------------------------------------------------------------------------------------------------------

import statistics
import numpy as np
import matplotlib as plt
from scipy.stats import f
import math

# OLP line
def olp_line(gold_standard, validated):
    mean_gs = statistics.mean(gold_standard)
    mean_val = statistics.mean(validated)

    stdev_gs = statistics.stdev(gold_standard)
    stdev_val = statistics.stdev(validated)

    # Parameters for the OLP line
    b = stdev_val / stdev_gs
    a = mean_val - b * mean_gs

    coefficients = np.polyfit(gold_standard, validated, 1)
    y_fit = np.polyval(coefficients, gold_standard)

    # Confidence intervals
    R = np.corrcoef(gold_standard, validated)
    r = R[0, 1]
    alpha = 0.05
    n = len(gold_standard)
    B = f.ppf(1 - alpha, 1, n-2) * (1 - r**2)/(n-2)
    CI_b = [b * math.sqrt(B + 1) - math.sqrt(B), b * math.sqrt(B + 1) + math.sqrt(B)]
    CI_a = [mean_val- b * (math.sqrt(B + 1) + math.sqrt(B))*mean_gs, mean_val - b*(math.sqrt(B+1) - math.sqrt(B)) * mean_gs]

    # Uncomment if you want the plot
    '''plt.figure()
    plt.scatter(gold_standard, b * validated + a, s = 10, label = "Data")
    plt.plot(gold_standard, gold_standard, color = "r", label = "Bisecant")
    plt.plot(gold_standard, y_fit, color = "g", label = "Interpolating line")
    plt.legend(loc = "upper left", fontsize = "small")
    plt.set_title(f"Heart rate: a is {a:.2f}, b is {b:.2f}")
    plt.show'''

    return a, b, y_fit, CI_a, CI_b

def get_MSE_and_RMSE(gold_standard, validated):
    MSE = 0
    for i in range(len(gold_standard)):
        squared_error = (gold_standard[i] - validated[i])**2
        MSE = MSE + squared_error
    MSE = MSE / len(gold_standard)
    RMSE = np.sqrt(MSE)

    return MSE, RMSE

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