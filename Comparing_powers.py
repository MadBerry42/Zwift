import pandas as pd
import array as arr
import numpy as np
import  matplotlib.pyplot as plt

# Importing data
path = "Acquisitions\\Protocol"
n_subjects = 11 # Count the number of folders 
subjects = arr.array("i", [0, 3, 4, 6, 7, 8, 9, 10, 11, 12])
alpha_mean = np.zeros(len(subjects))
case = "Full signal" # Full signal or blocks

if case == "Blocks":
    for j in range(0, len(subjects)):
        P1 = np.zeros((3, 4))
        i = subjects[j]
        if i < 10:
            ID = f"00{i}"
        if i >= 10:
            ID = f"0{i}"
        

        power_hc = pd.read_csv(f"{path}\\{ID}\\Zwift\\{ID}_handcycle_protocol.csv", usecols=['Power'])
        power_bc = pd.read_csv(f"{path}\\{ID}\\Zwift\\{ID}_bicycle_protocol.csv", usecols=['Power'])

        # Convert dataframe into a numpy array for better handling
        power_hc = np.array(power_hc)
        power_bc = np.array(power_bc)


        # Isolating the blocks with a certain RPE
        hc_RPE_12 = power_hc[300 : 479]
        hc_RPE_14 = power_hc[480 : 659]
        hc_RPE_15 = power_hc[660 : 839]

        bc_RPE_12 = power_bc[300 : 479]
        bc_RPE_14 = power_bc[480 : 659]
        bc_RPE_15 = power_bc[660 : 839]



        P1[0, 0] = np.mean(hc_RPE_12)
        P1[1, 0] = np.mean(hc_RPE_14)
        P1[2, 0] = np.mean(hc_RPE_15)

        P1[0, 1] = np.mean(bc_RPE_12)
        P1[1, 1] = np.mean(bc_RPE_14)
        P1[2, 1] = np.mean(bc_RPE_15) 

        P1[0, 3] = P1[0, 1]/P1[0, 0]
        P1[1, 3] =  P1[1, 1]/P1[1, 0]
        P1[2, 3] = P1[2, 1]/P1[2, 0]

        P1[0, 2] = 12
        P1[1, 2] = 14
        P1[2, 2] = 15

        alpha_mean = np.mean([P1[0, 1]/P1[0, 0], P1[1, 1]/P1[1, 0], P1[2, 1]/P1[2, 0]])

        '''print("Subject number", {ID}, ": P_hc, P_bc, RPE, alpha")
        print(P1) '''

        new_hc_12 = alpha_mean * hc_RPE_12
        new_hc_14 = alpha_mean * hc_RPE_14
        new_hc_15 = alpha_mean * hc_RPE_15

        t = np.linspace(0, len(new_hc_12), num = len(new_hc_12)) # Becase sample frequency is 1 Hz

        fig, ax = plt.subplots(3, 1)
        fig.suptitle(f"Subject {ID}")

        ax[0].plot(t, bc_RPE_12)
        ax[0].plot(t, new_hc_12)
        ax[0].set_title("RPE = 12")
        ax[0].legend(["bicycle", "Tweaked handcycle"])


        ax[1].plot(t, bc_RPE_14)
        ax[1].plot(t, new_hc_14)
        ax[1].set_title("RPE = 14")
        ax[1].legend(["bicycle", "Tweaked handcycle"])

        ax[2].plot(t, bc_RPE_15)
        ax[2].plot(t, new_hc_15)
        ax[2].set_title("RPE = 15")
        ax[2].legend(["bicycle", "Tweaked handcycle"])




if case == "Full signal":
    for j in range(0, len(subjects)):
        P1 = np.zeros((3, 4))
        i = subjects[j]
        if i < 10:
            ID = f"00{i}"
        if i >= 10:
            ID = f"0{i}"

        power_hc = pd.read_csv(f"{path}\\{ID}\\Zwift\\{ID}_handcycle_protocol.csv", usecols=['Power'])
        power_bc = pd.read_csv(f"{path}\\{ID}\\Zwift\\{ID}_bicycle_protocol.csv", usecols=['Power'])

        # Convert dataframe into a numpy array for better handling
        power_hc = np.array(power_hc)
        power_bc = np.array(power_bc)

        # Isolating the blocks with a certain RPE
        hc_RPE_12 = power_hc[300 : 479]
        hc_RPE_14 = power_hc[480 : 659]
        hc_RPE_15 = power_hc[660 : 839]

        bc_RPE_12 = power_bc[300 : 479]
        bc_RPE_14 = power_bc[480 : 659]
        bc_RPE_15 = power_bc[660 : 839]

        # Computing alpha_mean
        P1[0, 0] = np.mean(hc_RPE_12)
        P1[1, 0] = np.mean(hc_RPE_14)
        P1[2, 0] = np.mean(hc_RPE_15)

        P1[0, 1] = np.mean(bc_RPE_12)
        P1[1, 1] = np.mean(bc_RPE_14)
        P1[2, 1] = np.mean(bc_RPE_15) 

        P1[0, 3] = P1[0, 1]/P1[0, 0]
        P1[1, 3] =  P1[1, 1]/P1[1, 0]
        P1[2, 3] = P1[2, 1]/P1[2, 0]

        P1[0, 2] = 12
        P1[1, 2] = 14
        P1[2, 2] = 15

        alpha_mean = np.mean([P1[0, 1]/P1[0, 0], P1[1, 1]/P1[1, 0], P1[2, 1]/P1[2, 0]])

        #  Computing the tweaked signal
        new_hc = alpha_mean * power_hc

        # Plotting the signal
        t1 = np.linspace(0, len(new_hc), num = len(new_hc))
        t2 = np.linspace(0, len(power_bc), num = len(power_bc))
        plt.figure()
        plt.plot(t1, power_hc)
        plt.plot(t1, new_hc)
        plt.plot(t2, power_bc)
        plt.legend(["Handcycle", "Tweaked signal", "Bicycle"])
        plt.title(f"Subject {ID}")


plt.show()



